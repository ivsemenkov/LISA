"""Regenerate audio embeddings and metadata at alternative segment lengths.

This implements the segment-length ablation for the MEG->audio retrieval task
*without* changing the frozen 3-second setup that all baseline experiments use.

Design (agreed for the paper):

- The set of anchors (window start positions) is frozen: we reuse the exact
  3-second chunk anchors from ``chunks_info_{split}.csv`` (produced by
  ``lisa-generate-embeddings``). We never re-run the silence/word filter, so the
  identity of retained windows is identical to the baseline.
- For a target window length ``W`` seconds we grow each window to the right of
  its ``wav_start``: audio slice ``[wav_start, wav_start + W*AUDIO_SR)`` and MEG
  window ``[meg_start, meg_start + W*meg_sr)``. Both sides start at the same
  physical time, so the positive pair stays co-temporal.
- Anchors whose window runs past the end of *their own sound's* audio are
  dropped (never zero-padded, never spliced across sounds). This keeps every
  window a genuine, contiguous recording. By default each window is gated by
  itself, so for ``W <= 3`` all anchors survive and only ``W>3`` drops a few
  tail anchors per sound.
- Pass ``--anchor-window S`` to instead gate every window (and both splits) by
  feasibility at the single length ``S`` (e.g. the largest window). Then the
  retained anchor set, its count, and the ``wav_index`` remapping are identical
  across the whole sweep, so the retrieval database size and the number of
  training windows are the same for every segment length -- differences in
  metrics then reflect window length only, not database/training-set size.
- The raw MEG npz is never modified. Slicing happens on the fly in the dataset,
  so only the per-window ``meg{fs}_stop`` and ``wav_stop`` values in the
  dataframe change. Outputs are written to the SAME ``--dirprocess`` under
  window-tagged filenames (``..._{tag}``); the canonical 3-second files and the
  MEG npz are read-only here.

Use ``--verify-baseline`` to confirm the wav2vec pipeline reproduces the frozen
3-second embeddings bit-for-bit before trusting any regenerated window.
"""

import argparse
import os

import librosa
import numpy as np
import pandas as pd

from lisa.data.audio_reduction import (
    get_audio_embeddings_path,
    get_chunks_info_path,
    get_dataframe_path,
)
from lisa.data.generate_all_embeddings import (
    extract_wav2vec_embedding,
    get_audios,
    load_wav2vec_model,
)
from lisa.utils.constants import (
    AUDIO_SR,
    N_SUBJECTS,
    PREPROCESSED_DATA_DIR,
    PROJECT_ROOT,
)
from lisa.utils.determinism import fix_seed


def _subset_key(subject_id: int, session_id: int, story_id: int) -> str:
    """Replicate DoubleDataset's MEG npz key (subject id 1-indexed, zero-padded)."""
    sid = str(int(subject_id))
    sid = '0' + sid if len(sid) == 1 else sid
    return f'subject{sid}_session{int(session_id)}_story{int(story_id)}'


def _expected_conv_time(meg_window_samples: int) -> int:
    """ConvHead output length for the default head (single_conv, stride 2, k=3).

    Mirrors ``ConvHead.output_length`` with the paper-baseline head so we can fail
    early if the regenerated audio length would not match the MEG side under the
    CLIP shape constraint. Assumes head_stride=2, kernel_size=3, padding=0.
    """
    return (meg_window_samples - 3) // 2 + 1


def _default_window_tag(window_size: float) -> str:
    """Derive a filename-safe tag from a window length (e.g. 1.5 -> 'w1p5')."""
    tag = 'w' + f'{window_size:g}'.replace('.', 'p')
    if not tag.isalnum():
        raise ValueError(
            f'Could not derive an alphanumeric tag from window_size={window_size}; '
            'pass --window-tags explicitly.'
        )
    return tag


def _load_audio_dict(audios: list[tuple[int, int, str]]) -> dict[tuple[int, int], np.ndarray]:
    """Load and resample each (story, sound) audio to AUDIO_SR, matching baseline."""
    audio_dict: dict[tuple[int, int], np.ndarray] = {}
    for story_id, sound_id, audio_path in audios:
        audio, orig_sr = librosa.load(audio_path, sr=None)
        if orig_sr != AUDIO_SR:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=AUDIO_SR)
        audio_dict[(int(story_id), int(sound_id))] = audio
    return audio_dict


def _encode_chunk(
    audio: np.ndarray,
    start: int,
    stop: int,
    feature_extractor,
    wav2vec_model,
    device: str,
    last_n_w2v: int,
) -> np.ndarray:
    """Encode a single audio slice with the shared wav2vec routine."""
    chunk_audio = audio[start:stop]
    emb_avg, _ = extract_wav2vec_embedding(
        audio_np=chunk_audio,
        feature_extractor=feature_extractor,
        wav2vec_model=wav2vec_model,
        device=device,
        sr=AUDIO_SR,
        last_n=last_n_w2v,
        return_all_layers=False,
    )
    return emb_avg


def verify_split(
    *,
    split: str,
    dirprocess: str,
    stimuli_root: str,
    feature_extractor,
    wav2vec_model,
    device: str,
    last_n_w2v: int,
    atol: float,
) -> None:
    """Re-encode the frozen 3-second chunks and compare to the baseline .npy."""
    chunks_info = pd.read_csv(get_chunks_info_path(dirprocess, split))
    baseline_path = get_audio_embeddings_path(
        dirprocess=dirprocess, split=split, embedding_layers=last_n_w2v
    )
    baseline = np.load(baseline_path)
    if baseline.shape[0] != len(chunks_info):
        raise RuntimeError(
            f'{split}: baseline embeddings have {baseline.shape[0]} rows but '
            f'chunks_info has {len(chunks_info)} rows.'
        )

    audio_dict = _load_audio_dict(get_audios(mode=split, stimuli_root=stimuli_root))

    max_abs = 0.0
    for row in chunks_info.itertuples(index=False):
        audio = audio_dict[(int(row.story_id), int(row.sound_id))]
        emb = _encode_chunk(
            audio,
            int(row.wav_start),
            int(row.wav_stop),
            feature_extractor,
            wav2vec_model,
            device,
            last_n_w2v,
        )
        ref = baseline[int(row.wav_index)]
        if emb.shape != ref.shape:
            raise RuntimeError(
                f'{split}: shape mismatch at wav_index={int(row.wav_index)}: '
                f'{emb.shape} vs baseline {ref.shape}.'
            )
        max_abs = max(max_abs, float(np.max(np.abs(emb - ref))))

    print(f'[verify] {split}: max |regen - baseline| = {max_abs:.3e} (atol={atol:.0e})')
    if max_abs > atol:
        raise RuntimeError(
            f'{split}: regenerated 3-second embeddings differ from baseline by '
            f'{max_abs:.3e} > atol={atol:.0e}. The wav2vec checkpoint, last-n, '
            'audio loading, or device does not match the baseline; refusing to '
            'trust regenerated windows.'
        )


def regenerate_split(
    *,
    split: str,
    window_size: float,
    window_tag: str,
    dirprocess: str,
    stimuli_root: str,
    meg_sr: int,
    meg_offset: float,
    feature_extractor,
    wav2vec_model,
    device: str,
    last_n_w2v: int,
    overwrite: bool,
    check_conv_length: bool,
    anchor_window: float | None,
) -> None:
    """Regenerate embeddings + dataframe for one split at ``window_size`` seconds.

    If ``anchor_window`` is not None, the set of retained anchors is decided by
    feasibility at ``anchor_window`` seconds (not at ``window_size``), so every
    window generated with the same ``anchor_window`` shares an identical anchor
    set, count, and ``wav_index`` remapping in both train and test. This unifies
    the retrieval database size and the number of training windows across the
    whole segment-length sweep; the audio is still encoded at the actual
    ``window_size``. Requires ``anchor_window >= window_size``.
    """
    window_samples_audio = int(round(window_size * AUDIO_SR))
    meg_window_samples = int(round(window_size * meg_sr))
    if abs(window_size * meg_sr - meg_window_samples) > 1e-9:
        raise ValueError(
            f'window_size * meg_sr must be an integer number of samples; got '
            f'{window_size} * {meg_sr}.'
        )

    if not window_tag:
        raise RuntimeError('window_tag must be non-empty for regeneration.')
    out_emb_path = get_audio_embeddings_path(
        dirprocess=dirprocess,
        split=split,
        embedding_layers=last_n_w2v,
        window_tag=window_tag,
    )
    out_df_path = get_dataframe_path(dirprocess, split, N_SUBJECTS, window_tag=window_tag)
    out_chunks_path = get_chunks_info_path(dirprocess, split, window_tag=window_tag)
    canonical = {
        get_audio_embeddings_path(
            dirprocess=dirprocess, split=split, embedding_layers=last_n_w2v
        ),
        get_dataframe_path(dirprocess, split, N_SUBJECTS),
        get_chunks_info_path(dirprocess, split),
    }
    for path in (out_emb_path, out_df_path, out_chunks_path):
        if path in canonical:
            raise RuntimeError(
                f'Refusing to write to the canonical (3-second) path {path}.'
            )
        if path.exists() and not overwrite:
            raise FileExistsError(
                f'Output already exists: {path}. Pass --overwrite to replace it.'
            )

    chunks_info = pd.read_csv(get_chunks_info_path(dirprocess, split))
    baseline_emb = np.load(
        get_audio_embeddings_path(
            dirprocess=dirprocess, split=split, embedding_layers=last_n_w2v
        )
    )
    if baseline_emb.shape[0] != len(chunks_info):
        raise RuntimeError(
            f'{split}: baseline embeddings have {baseline_emb.shape[0]} rows but '
            f'chunks_info has {len(chunks_info)} rows.'
        )
    expected_wav_index = np.arange(len(chunks_info), dtype=np.int64)
    if not np.array_equal(chunks_info['wav_index'].to_numpy(dtype=np.int64), expected_wav_index):
        raise RuntimeError(
            f'{split}: chunks_info wav_index is not the contiguous range '
            '0..N-1; cannot align embeddings by position.'
        )

    audio_dict = _load_audio_dict(get_audios(mode=split, stimuli_root=stimuli_root))

    # 1) Feasible anchors (window fits inside its own sound's audio), in wav_index
    #    order. When anchor_window is set, gate by that fixed length so the anchor
    #    set is identical across every window in the sweep (common retrieval bank
    #    and equal training-window count); otherwise gate by the window itself.
    if anchor_window is not None:
        if anchor_window < window_size:
            raise RuntimeError(
                f'{split}: anchor_window={anchor_window}s must be >= '
                f'window_size={window_size}s to keep the actual window feasible.'
            )
        gate_samples_audio = int(round(anchor_window * AUDIO_SR))
    else:
        gate_samples_audio = window_samples_audio
    survivors: list[int] = []
    for row in chunks_info.itertuples(index=False):
        audio = audio_dict[(int(row.story_id), int(row.sound_id))]
        if int(row.wav_start) + gate_samples_audio <= len(audio):
            survivors.append(int(row.wav_index))
    if not survivors:
        gate = anchor_window if anchor_window is not None else window_size
        raise RuntimeError(f'{split}: no anchors fit gate window={gate}s.')
    old_to_new = {old: new for new, old in enumerate(survivors)}

    # 2) Re-encode each survivor at the target length.
    embeddings: list[np.ndarray] = []
    emb_shape: tuple[int, int] | None = None
    for old_idx in survivors:
        row = chunks_info.iloc[old_idx]
        audio = audio_dict[(int(row['story_id']), int(row['sound_id']))]
        start = int(row['wav_start'])
        emb = _encode_chunk(
            audio,
            start,
            start + window_samples_audio,
            feature_extractor,
            wav2vec_model,
            device,
            last_n_w2v,
        )
        if emb_shape is None:
            emb_shape = emb.shape
        elif emb.shape != emb_shape:
            raise RuntimeError(
                f'{split}: inconsistent embedding shape {emb.shape} vs {emb_shape} '
                f'at wav_index={old_idx}.'
            )
        embeddings.append(emb)
    embeddings_arr = np.stack(embeddings).astype(baseline_emb.dtype, copy=False)

    audio_time = embeddings_arr.shape[1]
    if check_conv_length:
        expected = _expected_conv_time(meg_window_samples)
        if audio_time != expected:
            raise RuntimeError(
                f'{split}: regenerated audio time length {audio_time} does not match '
                f'the MEG ConvHead output {expected} for a {window_size}s window '
                f'(meg_window={meg_window_samples}). This assumes the paper head '
                '(single_conv, stride 2, kernel 3). CLIP requires matching (F, T); '
                'adjust the window or pass --no-check-conv-length if you use a '
                'different head.'
            )

    # 3) New chunks_info (traceability): filtered, remapped, new wav_stop.
    new_chunks = chunks_info.iloc[survivors].copy()
    new_chunks['wav_stop'] = new_chunks['wav_start'].to_numpy(dtype=np.int64) + window_samples_audio
    new_chunks['wav_index'] = [old_to_new[i] for i in survivors]
    new_chunks = new_chunks.reset_index(drop=True)

    # 4) New MEG dataframe: keep survivor rows, recompute stops, remap wav_index.
    df = pd.read_csv(get_dataframe_path(dirprocess, split, N_SUBJECTS))
    survivor_set = set(survivors)
    keep_mask = df['wav_index'].isin(survivor_set).to_numpy()
    df_new = df.loc[keep_mask].copy()
    start_col = f'meg{meg_sr}_start'
    stop_col = f'meg{meg_sr}_stop'
    if start_col not in df_new.columns:
        raise RuntimeError(f'{split}: dataframe missing column {start_col!r}.')
    df_new['wav_stop'] = df_new['wav_start'].to_numpy(dtype=np.int64) + window_samples_audio
    df_new[stop_col] = df_new[start_col].to_numpy(dtype=np.int64) + meg_window_samples
    df_new['wav_index'] = df_new['wav_index'].map(old_to_new).to_numpy(dtype=np.int64)
    df_new = df_new.reset_index(drop=True)

    # 5) Validate MEG stops stay within the raw MEG arrays (no out-of-bounds slicing).
    #    DoubleDataset shifts the window by meg_offset on both ends, so include it.
    _assert_meg_in_bounds(
        df_new=df_new,
        dirprocess=dirprocess,
        meg_sr=meg_sr,
        meg_offset=meg_offset,
        start_col=start_col,
        stop_col=stop_col,
        split=split,
    )

    # 6) Sanity: embeddings and dataframe agree.
    if embeddings_arr.shape[0] != len(survivors):
        raise RuntimeError(f'{split}: embeddings/survivor count mismatch.')
    max_wav = int(df_new['wav_index'].max())
    if max_wav >= embeddings_arr.shape[0]:
        raise RuntimeError(
            f'{split}: df wav_index max {max_wav} out of bounds for '
            f'{embeddings_arr.shape[0]} embeddings.'
        )

    # 7) Write outputs (window-tagged; never the canonical names).
    out_emb_path.parent.mkdir(parents=True, exist_ok=True)
    out_df_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_emb_path, embeddings_arr)
    df_new.to_csv(out_df_path, index=False)
    new_chunks.to_csv(out_chunks_path, index=False)

    dropped = len(chunks_info) - len(survivors)
    gate_desc = f'{anchor_window}s (common)' if anchor_window is not None else 'self'
    print(
        f'[{split}] window={window_size}s tag={window_tag!r} gate={gate_desc}: '
        f'candidates {len(chunks_info)} -> {len(survivors)} (dropped {dropped}); '
        f'audio T={audio_time}, F={embeddings_arr.shape[2]}; '
        f'df rows {int(keep_mask.sum())}/{len(df)}.'
    )
    print(f'  wrote {out_emb_path}')
    print(f'  wrote {out_df_path}')
    print(f'  wrote {out_chunks_path}')


def _assert_meg_in_bounds(
    *,
    df_new: pd.DataFrame,
    dirprocess: str,
    meg_sr: int,
    meg_offset: float,
    start_col: str,
    stop_col: str,
    split: str,
) -> None:
    """Ensure every recomputed MEG window (incl. meg_offset shift) is in bounds."""
    meg_path = os.path.join(dirprocess, 'meg', f'meg{N_SUBJECTS}_sr{meg_sr}.npz')
    offset_samples = int(round(meg_offset * meg_sr))
    min_start = int((df_new[start_col].to_numpy(dtype=np.int64) + offset_samples).min())
    if min_start < 0:
        raise RuntimeError(
            f'{split}: meg_offset={meg_offset}s pushes a MEG window start to '
            f'{min_start} < 0; regeneration aborted.'
        )
    required_stop: dict[str, int] = {}
    for row in df_new.itertuples(index=False):
        key = _subset_key(
            getattr(row, 'subject_id'),
            getattr(row, 'session_id'),
            getattr(row, 'story_id'),
        )
        stop = int(getattr(row, stop_col)) + offset_samples
        if stop > required_stop.get(key, -1):
            required_stop[key] = stop

    with np.load(meg_path) as meg:
        available = set(meg.files)
        missing = sorted(required_stop.keys() - available)
        if missing:
            raise RuntimeError(
                f'{split}: MEG npz is missing keys referenced by the dataframe: '
                f'{missing[:5]}.'
            )
        for key, stop in required_stop.items():
            length = meg[key].shape[1]
            if stop > length:
                raise RuntimeError(
                    f'{split}: MEG window for {key} needs stop={stop} '
                    f'(incl. meg_offset={meg_offset}s) but the raw MEG array is only '
                    f'{length} samples long. The requested window runs past the '
                    'recording; regeneration aborted.'
                )


def main(
    *,
    window_sizes: list[float],
    window_tags: list[str],
    dirprocess: str,
    stimuli_root: str,
    meg_sr: int,
    meg_offset: float,
    wav2vec_ckpt: str,
    last_n_w2v: int,
    verify_baseline: bool,
    verify_atol: float,
    overwrite: bool,
    check_conv_length: bool,
    anchor_window: float | None,
) -> None:
    """Verify baseline reproduction and/or regenerate embeddings for each window."""
    fix_seed(42, torch_deterministic=True)
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feature_extractor, wav2vec_model = load_wav2vec_model(
        wav2vec_ckpt=wav2vec_ckpt, device=device
    )

    if verify_baseline:
        for split in ('train', 'test'):
            verify_split(
                split=split,
                dirprocess=dirprocess,
                stimuli_root=stimuli_root,
                feature_extractor=feature_extractor,
                wav2vec_model=wav2vec_model,
                device=device,
                last_n_w2v=last_n_w2v,
                atol=verify_atol,
            )
        print('[verify] baseline reproduction OK.')
        return

    if anchor_window is not None:
        max_window = max(window_sizes)
        if anchor_window < max_window:
            raise RuntimeError(
                f'--anchor-window={anchor_window}s must be >= the largest '
                f'--window-sizes entry ({max_window}s) so the common anchor set is '
                'feasible for every window.'
            )
        print(
            f'Common anchor set: gating all windows + both splits by '
            f'{anchor_window:g}s feasibility (unified retrieval bank / train count).'
        )

    print(
        'Regenerating windows: '
        + ', '.join(
            f'{w:g}s->{t!r}'
            for w, t in zip(window_sizes, window_tags, strict=True)
        )
    )
    for window_size, window_tag in zip(window_sizes, window_tags, strict=True):
        print(f'=== window {window_size:g}s -> tag {window_tag!r} ===')
        for split in ('train', 'test'):
            regenerate_split(
                split=split,
                window_size=window_size,
                window_tag=window_tag,
                dirprocess=dirprocess,
                stimuli_root=stimuli_root,
                meg_sr=meg_sr,
                meg_offset=meg_offset,
                feature_extractor=feature_extractor,
                wav2vec_model=wav2vec_model,
                device=device,
                last_n_w2v=last_n_w2v,
                overwrite=overwrite,
                check_conv_length=check_conv_length,
                anchor_window=anchor_window,
            )
        print(f'    -> train with: lisa-train ... --window-tag {window_tag}')


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for window-length embedding regeneration."""
    parser = argparse.ArgumentParser(
        description='Regenerate audio embeddings + metadata at alternative segment '
        'lengths using the frozen 3-second anchors.',
        allow_abbrev=False,
    )
    parser.add_argument(
        '--window-sizes',
        type=float,
        nargs='+',
        default=[1.5, 2.25, 4.0, 5.0],
        help='One or more target segment lengths in seconds. Default: the '
        'recommended ablation grid around the 3-second baseline (1.5 2.25 4 5). '
        'Ignored with --verify-baseline.',
    )
    parser.add_argument(
        '--window-tags',
        type=str,
        nargs='+',
        default=None,
        help='Optional explicit alphanumeric tags, one per --window-sizes. If '
        'omitted, tags are derived from the size (1.5 -> w1p5, 4 -> w4). The tag '
        'is what you pass to lisa-train --window-tag.',
    )
    parser.add_argument(
        '--dirprocess',
        type=str,
        default=PREPROCESSED_DATA_DIR,
        help='Root directory with preprocessed embeddings/metadata (read + write).',
    )
    parser.add_argument(
        '--stimuli-root',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'data', 'MASC-MEG', 'stimuli'),
        help='Root directory of stimuli (expects audio/ under it).',
    )
    parser.add_argument('--meg-sr', type=int, default=100)
    parser.add_argument(
        '--meg-offset',
        type=float,
        default=0.15,
        help='MEG window offset in seconds (must match --meg-offset in training). '
        'Used only to validate that shifted windows stay within the recording.',
    )
    parser.add_argument(
        '--wav2vec-ckpt',
        type=str,
        default='facebook/wav2vec2-base-960h',
        help='Wav2Vec2 checkpoint; MUST match the one used for the baseline embeddings.',
    )
    parser.add_argument(
        '--last-n-w2v',
        type=int,
        default=4,
        help='Number of Wav2Vec2 layers averaged (embedding_layers). Must match '
        'the baseline embeddings and --embedding-layers in training.',
    )
    parser.add_argument(
        '--verify-baseline',
        action='store_true',
        help='Re-encode the frozen 3-second chunks and assert they match the '
        'baseline .npy, then exit without writing anything.',
    )
    parser.add_argument(
        '--verify-atol',
        type=float,
        default=1e-4,
        help='Max allowed absolute difference for --verify-baseline.',
    )
    parser.add_argument(
        '--overwrite',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Overwrite existing window-tagged output files.',
    )
    parser.add_argument(
        '--check-conv-length',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Assert the regenerated audio time length matches the paper-baseline '
        'ConvHead output for the MEG window. Disable only for a custom head.',
    )
    parser.add_argument(
        '--anchor-window',
        type=float,
        default=None,
        help='If set, gate the retained anchors by feasibility at this fixed '
        'window length (seconds) for EVERY --window-sizes entry and both splits, '
        'instead of gating each window by itself. This yields an identical anchor '
        'set / retrieval-database size / training-window count across the whole '
        'sweep (recommended for a clean ablation). Must be >= the largest '
        '--window-sizes entry. Ignored with --verify-baseline.',
    )

    args = parser.parse_args(argv)

    if not args.verify_baseline:
        if not args.window_sizes:
            parser.error('--window-sizes must list at least one length.')
        if any(w <= 0 for w in args.window_sizes):
            parser.error('--window-sizes must all be positive numbers of seconds.')
        if args.window_tags is not None:
            if len(args.window_tags) != len(args.window_sizes):
                parser.error(
                    '--window-tags must have exactly one tag per --window-sizes.'
                )
            tags = list(args.window_tags)
        else:
            tags = [_default_window_tag(w) for w in args.window_sizes]
        for tag in tags:
            if not tag or not tag.isalnum():
                parser.error(f'window tag {tag!r} must be non-empty alphanumeric.')
        if len(set(tags)) != len(tags):
            parser.error(f'window tags must be unique, got {tags}.')
        args.window_tags = tags
        if args.anchor_window is not None:
            if args.anchor_window <= 0:
                parser.error('--anchor-window must be a positive number of seconds.')
            max_window = max(args.window_sizes)
            if args.anchor_window < max_window:
                parser.error(
                    f'--anchor-window ({args.anchor_window}) must be >= the largest '
                    f'--window-sizes entry ({max_window}).'
                )
    if args.last_n_w2v < 1:
        parser.error('--last-n-w2v must be a positive integer.')
    return args


def main_cli(argv: list[str] | None = None) -> None:
    """CLI entry point for window-length embedding regeneration."""
    args = parse_arguments(argv)
    main(
        window_sizes=args.window_sizes,
        window_tags=args.window_tags if args.window_tags is not None else [],
        dirprocess=args.dirprocess,
        stimuli_root=args.stimuli_root,
        meg_sr=args.meg_sr,
        meg_offset=args.meg_offset,
        wav2vec_ckpt=args.wav2vec_ckpt,
        last_n_w2v=args.last_n_w2v,
        verify_baseline=args.verify_baseline,
        verify_atol=args.verify_atol,
        overwrite=args.overwrite,
        check_conv_length=args.check_conv_length,
        anchor_window=args.anchor_window,
    )


if __name__ == '__main__':
    main_cli()
