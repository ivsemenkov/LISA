"""Generate audio embeddings and aligned metadata from MASC-MEG stimuli."""

import argparse
import os
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from tqdm import tqdm

from lisa.data.textgrid_utils import parse_textgrid_words, word_majority_overlap
from lisa.utils.constants import (
    AUDIO_SR,
    PREPROCESSED_DATA_DIR,
    PROJECT_ROOT,
    TEST_STORIES,
    TRAIN_STORIES,
)
from lisa.utils.determinism import fix_seed


def load_wav2vec_model(
    wav2vec_ckpt: str, device: str = 'cpu'
):
    """Load Wav2Vec2 feature extractor and model."""
    print(f'Loading {wav2vec_ckpt}...')
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_ckpt)
    wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_ckpt).to(device)
    wav2vec_model.eval()
    return feature_extractor, wav2vec_model


def extract_wav2vec_embedding(
    audio_np: np.ndarray,
    feature_extractor: Wav2Vec2FeatureExtractor,
    wav2vec_model: Wav2Vec2Model,
    device: str,
    sr: int = 16000,
    last_n: int = 4,
    return_all_layers: bool = True,
):
    """Extract pooled Wav2Vec2 embeddings and optional per-layer features."""
    inputs = feature_extractor(audio_np, return_tensors='pt', sampling_rate=sr)
    x = inputs.input_values.to(device)

    wav2vec_model.eval()
    with torch.inference_mode():
        out = wav2vec_model(x, output_hidden_states=True)

    hs = out.hidden_states
    assert (last_n >= 1) and (last_n < len(hs)), (last_n, len(hs))
    selected = hs[-last_n:]

    avg_seq = torch.stack(selected).mean(dim=0).squeeze(0)  # [T, D]

    if return_all_layers:
        per_layer = torch.stack(hs).squeeze(1)  # [L_total, T, D]
        return avg_seq.cpu().numpy(), per_layer.cpu().numpy()
    else:
        return avg_seq.cpu().numpy(), None


def process_all_audios(
    audio_paths: Iterable[Tuple[int, int, str]],
    wav2vec_feature_extractor: Wav2Vec2FeatureExtractor,
    wav2vec_model: Wav2Vec2Model,
    textgrid_root: str,
    chunk_size: float,
    overlap: float,
    amp_threshold: float,
    mode: str,
    chunks_output_dir: str,
    embeddings_output_root: str,
    device: str,
    last_n_w2v: int,
    verbose: bool,
    full_embeddings: bool,
):
    """Process all audio files, saving chunked audio/text and embeddings.

    Uses two-pass approach to cap memory: Pass 1 loads audio, chunks, builds df;
    Pass 2 extracts embeddings into pre-allocated arrays (no list+stack doubling).
    """
    # Load all audio once, resample, keep in memory to avoid double-resampling inconsistency
    audio_dict: dict[Tuple[int, int], np.ndarray] = {}
    for story_id, sound_id, audio_path in tqdm(
        audio_paths, desc='Loading audio'
    ):
        audio, orig_sr = librosa.load(audio_path, sr=None)
        if orig_sr != AUDIO_SR:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=AUDIO_SR)
        audio_dict[(story_id, sound_id)] = audio

    # Convert chunk parameters to sample indices (integer arithmetic)
    chunk_size_samples = round(chunk_size * AUDIO_SR)
    overlap_samples = round(overlap * AUDIO_SR)
    step_samples = chunk_size_samples - overlap_samples

    if step_samples <= 0:
        raise ValueError(
            f"'overlap' ({overlap}) must be < 'chunk_size' ({chunk_size}) to progress."
        )

    overall_info_df = {
        key: []
        for key in [
            'story_id',
            'sound_id',
            'sound_length',
            'sound_fname',
            'segment_start',
            'words',
            'wav_start',
            'wav_stop',
            'wav_index',
        ]
    }

    # Pass 1: chunking, build df, write chunk wav/txt (no Wav2Vec2)
    chunk_idx = 0
    for story_id, sound_id, audio_path in tqdm(
        audio_paths, desc='Pass 1 (chunking)'
    ):
        audio = audio_dict[(story_id, sound_id)]
        audio_name = os.path.basename(os.path.splitext(audio_path)[0])
        textgrid_path = os.path.join(textgrid_root, audio_name + '.TextGrid')
        out_dir = os.path.join(chunks_output_dir, audio_name)
        os.makedirs(out_dir, exist_ok=True)

        # Parse word times as sample indices for deterministic integer comparisons
        word_times = parse_textgrid_words(textgrid_path, sr=AUDIO_SR)

        audio_length_samples = len(audio)
        audio_duration = audio_length_samples / AUDIO_SR

        step = 0
        within_sound_chunk_idx = 0
        while True:
            start_sample = step * step_samples
            end_sample = start_sample + chunk_size_samples

            # Stop when we can't fit a full chunk
            if end_sample > audio_length_samples:
                break

            chunk_audio = audio[start_sample:end_sample]
            assert chunk_audio.size == chunk_size_samples, (
                f'Expected {chunk_size_samples} samples, got {chunk_audio.size}'
            )
            step += 1

            # Check if audio chunk is silent
            peak = np.max(np.abs(chunk_audio))
            if peak < amp_threshold:
                print(f'{audio_name} step {step - 1} is silent, skipping.')
                continue

            # Select words with majority overlap in this chunk (integer sample comparison)
            words_in_chunk = [
                w['word']
                for w in word_times
                if word_majority_overlap(w, start_sample, end_sample)
            ]
            if not words_in_chunk:
                print(
                    f'{audio_name} step {step - 1} has no words (majority overlap), skipping.'
                )
                continue

            overall_info_df['story_id'].append(story_id)
            overall_info_df['sound_id'].append(sound_id)
            overall_info_df['sound_length'].append(audio_duration)
            overall_info_df['sound_fname'].append(os.path.basename(audio_path))
            overall_info_df['segment_start'].append(within_sound_chunk_idx)
            overall_info_df['words'].append(words_in_chunk)
            overall_info_df['wav_start'].append(start_sample)
            overall_info_df['wav_stop'].append(end_sample)
            overall_info_df['wav_index'].append(chunk_idx)

            chunk_name = f'chunk_{chunk_idx}'
            audio_chunk_path = os.path.join(out_dir, f'{chunk_name}.wav')
            words_txt_path = os.path.join(out_dir, f'{chunk_name}.txt')

            sf.write(audio_chunk_path, chunk_audio, AUDIO_SR)
            with open(words_txt_path, 'w', encoding='utf-8') as f:
                f.write(' '.join(words_in_chunk) + '\n')

            if verbose:
                print(
                    f'Saved {audio_chunk_path} and {words_txt_path} with words: {" ".join(words_in_chunk)}'
                )

            chunk_idx += 1
            within_sound_chunk_idx += 1

    overall_info_df = pd.DataFrame(overall_info_df)
    dataframes_output_dir = os.path.join(embeddings_output_root, 'dataframe')
    os.makedirs(dataframes_output_dir, exist_ok=True)
    overall_info_df.to_csv(
        os.path.join(dataframes_output_dir, f'chunks_info_{mode}.csv'), index=False
    )

    N = len(overall_info_df)
    if N == 0:
        print(f'No valid chunks for {mode}, skipping embeddings.')
        return

    # Pass 2: extract embeddings into pre-allocated arrays
    # Get shapes from first chunk
    first_row = overall_info_df.iloc[0]
    story_id, sound_id = int(first_row['story_id']), int(first_row['sound_id'])
    wav_start, wav_stop = int(first_row['wav_start']), int(first_row['wav_stop'])
    chunk_audio = audio_dict[(story_id, sound_id)][wav_start:wav_stop]

    emb_avg, emb_full = extract_wav2vec_embedding(
        audio_np=chunk_audio,
        feature_extractor=wav2vec_feature_extractor,
        wav2vec_model=wav2vec_model,
        device=device,
        sr=AUDIO_SR,
        last_n=last_n_w2v,
        return_all_layers=full_embeddings,
    )

    T, D = emb_avg.shape
    # Embedding array must be indexed by wav_index (DoubleDataset uses hidden[wav_index])
    arr_avg = np.zeros((N, T, D), dtype=emb_avg.dtype)
    wav_idx_first = int(first_row['wav_index'])
    arr_avg[wav_idx_first] = emb_avg

    if full_embeddings:
        L, _, _ = emb_full.shape
        arr_full = np.zeros((N, L, T, D), dtype=emb_full.dtype)
        arr_full[wav_idx_first] = emb_full
    else:
        arr_full = None

    for idx in tqdm(range(1, N), desc='Pass 2 (embeddings)'):
        row = overall_info_df.iloc[idx]
        wav_index = int(row['wav_index'])
        story_id, sound_id = int(row['story_id']), int(row['sound_id'])
        wav_start, wav_stop = int(row['wav_start']), int(row['wav_stop'])
        chunk_audio = audio_dict[(story_id, sound_id)][wav_start:wav_stop]

        emb_avg, emb_full = extract_wav2vec_embedding(
            audio_np=chunk_audio,
            feature_extractor=wav2vec_feature_extractor,
            wav2vec_model=wav2vec_model,
            device=device,
            sr=AUDIO_SR,
            last_n=last_n_w2v,
            return_all_layers=full_embeddings,
        )

        if emb_avg.shape != (T, D):
            raise RuntimeError(
                f'W2V2 pooled shape changed: {emb_avg.shape} vs ({T}, {D})'
            )
        if full_embeddings and emb_full.shape != (L, T, D):
            raise RuntimeError(
                f'W2V2 full-layer shape changed: {emb_full.shape} vs ({L}, {T}, {D})'
            )

        arr_avg[wav_index] = emb_avg
        if full_embeddings:
            arr_full[wav_index] = emb_full

    print('Amount of DataFrame entries:', len(overall_info_df))
    print(
        'Audio embeddings shape:',
        arr_avg.shape,
        arr_avg.dtype,
    )
    if full_embeddings:
        print(
            'Audio embeddings full shape:',
            arr_full.shape,
            arr_full.dtype,
        )
    assert len(overall_info_df) == arr_avg.shape[0]

    audio_embeddings_output_dir = os.path.join(embeddings_output_root, 'audio')
    os.makedirs(audio_embeddings_output_dir, exist_ok=True)
    np.save(
        os.path.join(
            audio_embeddings_output_dir, f'extract_features_{mode}{last_n_w2v}.npy'
        ),
        arr_avg,
    )
    if full_embeddings:
        np.save(
            os.path.join(
                audio_embeddings_output_dir,
                f'extract_features_{mode}_full.npy',
            ),
            arr_full,
        )


def get_audios(mode: str, stimuli_root: str):
    """Return list of (story_id, sound_id, path) tuples for a split."""

    if mode == 'train':
        stories = TRAIN_STORIES
    elif mode == 'test':
        stories = TEST_STORIES
    else:
        raise ValueError("mode should be 'train' or 'test'")

    audios = []
    for story_id in stories.keys():
        story, story_part_start, story_part_end = stories[story_id]
        story_dir = os.path.join(stimuli_root, 'audio')
        for sound_id in range(story_part_start, story_part_end + 1):
            audio_path = os.path.join(story_dir, f'{story}_{sound_id}.wav')
            if not os.path.isfile(audio_path):
                raise FileNotFoundError(f'Audio file {audio_path} not found.')
            audios.append((story_id, sound_id, audio_path))

    print(f'Processing {len(audios)} audio files in {mode} mode:')
    for story_id, sound_id, audio_path in audios:
        print(f'Story: {story_id}, sound: {sound_id}, path: {audio_path}')
    return audios


def main(
    chunk_size: float,
    overlap: float,
    stimuli_root: str,
    wav2vec_ckpt: str,
    amp_threshold: float,
    chunks_output_dir: str,
    embeddings_output_root: str,
    last_n_w2v: int,
    verbose: bool,
    full_embeddings: bool,
) -> None:
    """Run embedding extraction for train/test splits."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    wav2vec_feature_extractor, wav2vec_model = load_wav2vec_model(
        wav2vec_ckpt=wav2vec_ckpt, device=device
    )

    for mode in ['train', 'test']:
        audios = get_audios(mode=mode, stimuli_root=stimuli_root)

        process_all_audios(
            audio_paths=audios,
            wav2vec_feature_extractor=wav2vec_feature_extractor,
            wav2vec_model=wav2vec_model,
            textgrid_root=os.path.join(stimuli_root, 'combined_mfa'),
            chunk_size=chunk_size,
            overlap=overlap,
            amp_threshold=amp_threshold,
            mode=mode,
            chunks_output_dir=chunks_output_dir,
            embeddings_output_root=embeddings_output_root,
            device=device,
            last_n_w2v=last_n_w2v,
            verbose=verbose,
            full_embeddings=full_embeddings,
        )


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for embedding generation."""

    parser = argparse.ArgumentParser(
        description='Generate audio embeddings from stimuli.'
    )

    parser.add_argument(
        '--chunk-size', type=float, default=3.0, help='Chunk size in seconds.'
    )
    parser.add_argument(
        '--overlap', type=float, default=2.0, help='Overlap between chunks in seconds.'
    )
    parser.add_argument(
        '--stimuli-root',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'data', 'MASC-MEG', 'stimuli'),
        help='Root directory of stimuli.',
    )
    parser.add_argument(
        '--wav2vec-ckpt',
        type=str,
        default='facebook/wav2vec2-base-960h',
        help='Wav2Vec2 checkpoint.',
    )
    parser.add_argument(
        '--amp-threshold',
        type=float,
        default=1e-4,
        help='Amplitude threshold to consider chunk as silent.',
    )
    parser.add_argument(
        '--chunks-output-dir',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'data', 'MASC-MEG', 'stimuli', 'chunks'),
        help='Directory to save audio chunks and text files.',
    )
    parser.add_argument(
        '--embeddings-output-root',
        type=str,
        default=PREPROCESSED_DATA_DIR,
        help='Root directory to save embeddings and metadata.',
    )
    parser.add_argument(
        '--last-n-w2v',
        type=int,
        default=4,
        help='Amount of layers for Wav2Vec2 averaging.',
    )
    parser.add_argument(
        '--full-embeddings',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Whether to compute and save full per-layer embeddings (large, train split is already ~15 GB). Default: False.',
    )
    parser.add_argument(
        '--verbose',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Whether to print verbose logs.',
    )
    args = parser.parse_args()
    return args


def main_cli() -> None:
    """CLI entry point for embedding generation."""
    args = parse_arguments()
    fix_seed(42, torch_deterministic=True)
    main(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        stimuli_root=args.stimuli_root,
        wav2vec_ckpt=args.wav2vec_ckpt,
        amp_threshold=args.amp_threshold,
        chunks_output_dir=args.chunks_output_dir,
        embeddings_output_root=args.embeddings_output_root,
        last_n_w2v=args.last_n_w2v,
        verbose=args.verbose,
        full_embeddings=args.full_embeddings,
    )


if __name__ == '__main__':
    main_cli()
