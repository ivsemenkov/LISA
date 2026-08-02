"""Prepare MEG data arrays and alignment DataFrames."""

import argparse
import ast
import os
import time
import warnings
from typing import Any

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from lisa.data.meg_io import load_raw_meg, preprocess_meg
from lisa.utils.constants import (
    AUDIO_SR,
    CLEAN_DATA_DIR,
    N_SUBJECTS,
    PREPROCESSED_DATA_DIR,
    SUB_SES_COMBOS,
)


def process_single_combo(
    story_id: int,
    sub: int,
    ses: int,
    meg_format: str,
    meg_dir: str,
    target_fs: int,
    sounds_per_split: dict,
    story_dfs: dict,
    df_columns: list,
    ordered_columns: list,
    chunk_size: int,
    allow_test_only: bool,
    profile_log: bool,
) -> dict[str, Any]:
    """Process a single (story, subject, session) combination.

    Args:
        story_id: Story identifier.
        sub: Subject id.
        ses: Session id.
        meg_format: "fif" or "bids".
        meg_dir: MEG source directory.
        target_fs: Target MEG sampling rate (Hz).
        sounds_per_split: Dict mapping mode -> set of (story_id, sound_id, sound_fname).
        story_dfs: Dict mapping mode -> {story_id: DataFrame}.
        df_columns: List of columns from input DataFrame.
        ordered_columns: List of all output columns.
        chunk_size: Chunk size in samples.
        allow_test_only: Allow processing stories that only appear in test.
        profile_log: Whether to collect timing data.

    Returns:
        Dict containing 'subset', 'meg', 'df_entries', and 'timing_rows'.
    """
    modes = ['train', 'test']
    timing_rows = []
    df_entries = {mode: {key: [] for key in ordered_columns} for mode in modes}

    # --- Load raw MEG ---
    t0 = time.perf_counter()
    raw, orig_fs = load_raw_meg(
        meg_format=meg_format,
        data_root=meg_dir,
        sub=sub,
        ses=ses,
        story_id=story_id,
    )
    if profile_log:
        timing_rows.append(
            {
                'story_id': story_id,
                'subject_id': sub,
                'session_id': ses,
                'stage': 'load_raw_meg',
                'seconds': time.perf_counter() - t0,
            }
        )

    # --- Parse onsets from annotations ---
    t0 = time.perf_counter()
    onsets_all = {}
    events, events_dict = mne.events_from_annotations(raw=raw, verbose=False)
    for event, event_id in events_dict.items():
        event = ast.literal_eval(event)
        if event['kind'] == 'sound':
            event_sample = events[events[:, -1] == event_id]
            assert len(event_sample) == 1, event_sample
            event_sample = event_sample[0, 0]
            event_onset = event_sample / orig_fs
            sound_fname = event['sound'].lower()
            assert sound_fname.endswith('.0.wav'), sound_fname
            sound_fname = os.path.basename(sound_fname[: -len('.0.wav')]) + '.wav'
            sound_key = (story_id, int(event['sound_id']), sound_fname)
            if sound_key in sounds_per_split['all']:
                onsets_all[sound_key] = event_onset
    if profile_log:
        timing_rows.append(
            {
                'story_id': story_id,
                'subject_id': sub,
                'session_id': ses,
                'stage': 'parse_onsets',
                'seconds': time.perf_counter() - t0,
            }
        )

    # --- Split onsets into train/test ---
    onsets = {m: {} for m in modes}
    for sound_key, event_onset in onsets_all.items():
        for mode in modes:
            if sound_key in sounds_per_split[mode]:
                onsets[mode][sound_key] = event_onset
    if not onsets['train'] and not onsets['test']:
        raise ValueError(
            f'No train/test onsets found for story {story_id} (sub={sub}, ses={ses}).'
        )

    # --- Determine onset bounds ---
    has_train = bool(onsets['train'])
    has_test = bool(onsets['test'])
    if has_train:
        first_train_onset = min(onsets['train'].values())
        final_train_onset = max(onsets['train'].values())
    else:
        if not allow_test_only:
            raise ValueError(
                f'No train onsets found for story {story_id} '
                f'(sub={sub}, ses={ses}); cannot fit scalers without train.'
            )
        warnings.warn(
            f'No train onsets found for story {story_id} '
            f'(sub={sub}, ses={ses}); fitting scalers on test-only data.'
        )
        first_train_onset = None
        final_train_onset = None

    if has_test:
        first_test_onset = min(onsets['test'].values())
        final_test_onset = max(onsets['test'].values())
    else:
        first_test_onset = None
        final_test_onset = None

    # --- Validate onset ordering ---
    if has_train and has_test:
        assert (
            first_train_onset < final_train_onset < first_test_onset < final_test_onset
        ), (
            story_id,
            first_train_onset,
            final_train_onset,
            first_test_onset,
            final_test_onset,
        )
    elif has_train:
        assert first_train_onset <= final_train_onset, (
            story_id,
            first_train_onset,
            final_train_onset,
        )
    elif has_test:
        assert first_test_onset <= final_test_onset, (
            story_id,
            first_test_onset,
            final_test_onset,
        )

    if has_train:
        first_onset = first_train_onset
        first_test_onset_for_scaler = first_test_onset
    else:
        first_onset = first_test_onset
        first_test_onset_for_scaler = None

    # --- Preprocess MEG ---
    t0 = time.perf_counter()
    meg, _ = preprocess_meg(
        raw=raw,
        target_fs=target_fs,
        first_onset=first_onset,
        robust_scaler=None,
        standard_scaler=None,
        first_test_onset=first_test_onset_for_scaler,
    )
    if profile_log:
        timing_rows.append(
            {
                'story_id': story_id,
                'subject_id': sub,
                'session_id': ses,
                'stage': 'preprocess_meg',
                'seconds': time.perf_counter() - t0,
            }
        )

    subset = f'subject{sub:02d}_session{ses}_story{story_id}'

    # --- Build DataFrame entries ---
    for mode in modes:
        story_df = story_dfs[mode].get(story_id)
        if story_df is None:
            continue
        for row in story_df.itertuples(index=False):
            sound_key = (story_id, row.sound_id, row.sound_fname)
            if sound_key not in onsets[mode]:
                raise KeyError(
                    f'Missing onset for {sound_key} in {mode} (sub={sub}, ses={ses}).'
                )
            for key in df_columns:
                df_entries[mode][key].append(getattr(row, key))

            onset_meg = onsets[mode][sound_key]
            onset = int(onset_meg * target_fs)
            meg_start_samples = onset + int(round(row.wav_start / AUDIO_SR * target_fs))
            meg_stop_samples = onset + int(round(row.wav_stop / AUDIO_SR * target_fs))
            assert (meg_stop_samples - meg_start_samples) == chunk_size, (
                meg_stop_samples,
                meg_start_samples,
                meg_stop_samples - meg_start_samples,
                chunk_size,
            )

            df_entries[mode]['subject_id'].append(sub)
            df_entries[mode]['session_id'].append(ses)
            df_entries[mode]['onset_meg'].append(onset_meg)
            df_entries[mode][f'meg{target_fs}_start'].append(meg_start_samples)
            df_entries[mode][f'meg{target_fs}_stop'].append(meg_stop_samples)

    return {
        'subset': subset,
        'meg': meg,
        'df_entries': df_entries,
        'timing_rows': timing_rows,
    }


def main(
    target_fs: int,
    chunk_size: int,
    preprocessed_dir: str,
    meg_dir: str,
    meg_format: str,
    allow_test_only: bool,
    profile_log: bool,
    n_jobs: int,
):
    """Build preprocessed MEG arrays and metadata DataFrames.

    Args:
        target_fs: Target MEG sampling rate (Hz).
        chunk_size: Chunk size in samples.
        preprocessed_dir: Root directory for outputs.
        meg_dir: MEG source directory.
        meg_format: "fif" or "bids".
        allow_test_only: Allow processing stories that only appear in test.
        profile_log: Whether to write timing logs.
        n_jobs: Number of parallel jobs (-1 for all CPUs, 1 for sequential).
    """

    modes = ['train', 'test']
    dfs_per_split = {}
    story_dfs = {}
    sounds_per_split = {'all': set()}

    for mode in modes:
        df = pd.read_csv(
            os.path.join(preprocessed_dir, 'dataframe', f'chunks_info_{mode}.csv')
        )
        dfs_per_split[mode] = df
        story_dfs[mode] = {
            story_id: story_df for story_id, story_df in df.groupby('story_id')
        }
        split_sounds = set()
        for _, row in df.iterrows():
            split_sounds.add((row['story_id'], row['sound_id'], row['sound_fname']))
        sounds_per_split[mode] = split_sounds
        sounds_per_split['all'] = sounds_per_split['all'].union(split_sounds)
    assert len(sounds_per_split['train'].intersection(sounds_per_split['test'])) == 0, (
        sounds_per_split
    )

    df_columns = list(dfs_per_split[modes[0]].columns)
    for mode in modes[1:]:
        if list(dfs_per_split[mode].columns) != df_columns:
            raise ValueError(
                f'{modes[0]} and {mode} chunk DataFrames have different columns.'
            )

    ordered_columns = [
        'subject_id',
        'session_id',
        'story_id',
        'sound_id',
        'onset_meg',
        'sound_length',
        'sound_fname',
        'segment_start',
        'words',
        f'meg{target_fs}_start',
        f'meg{target_fs}_stop',
        'wav_start',
        'wav_stop',
        'wav_index',
    ]
    required_input_columns = [
        col
        for col in ordered_columns
        if col
        not in {
            'subject_id',
            'session_id',
            'onset_meg',
            f'meg{target_fs}_start',
            f'meg{target_fs}_stop',
        }
    ]
    missing_columns = set(required_input_columns) - set(df_columns)
    if missing_columns:
        raise ValueError(
            f'Missing required columns in chunk DataFrames: {sorted(missing_columns)}'
        )

    stories_needed = sorted({story_id for story_id, _, _ in sounds_per_split['all']})

    # Build list of all (story_id, sub, ses) combinations
    all_combos = [
        (story_id, sub, ses)
        for story_id in stories_needed
        for sub, ses in SUB_SES_COMBOS
    ]
    total_combos = len(all_combos)
    print(f'Processing {total_combos} (story, subject, session) combinations...')

    # Process combinations (parallel or sequential)
    if n_jobs == 1:
        # Sequential processing with progress bar
        results = []
        for story_id, sub, ses in tqdm(all_combos, desc='Processing MEG'):
            result = process_single_combo(
                story_id=story_id,
                sub=sub,
                ses=ses,
                meg_format=meg_format,
                meg_dir=meg_dir,
                target_fs=target_fs,
                sounds_per_split=sounds_per_split,
                story_dfs=story_dfs,
                df_columns=df_columns,
                ordered_columns=ordered_columns,
                chunk_size=chunk_size,
                allow_test_only=allow_test_only,
                profile_log=profile_log,
            )
            results.append(result)
    else:
        # Parallel processing with joblib
        print(f'Using {n_jobs} parallel jobs...')
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_single_combo)(
                story_id=story_id,
                sub=sub,
                ses=ses,
                meg_format=meg_format,
                meg_dir=meg_dir,
                target_fs=target_fs,
                sounds_per_split=sounds_per_split,
                story_dfs=story_dfs,
                df_columns=df_columns,
                ordered_columns=ordered_columns,
                chunk_size=chunk_size,
                allow_test_only=allow_test_only,
                profile_log=profile_log,
            )
            for story_id, sub, ses in all_combos
        )

    # Aggregate results
    global_df_per_mode = {mode: {key: [] for key in ordered_columns} for mode in modes}
    global_meg_recordings = {}
    timing_rows = []

    for result in results:
        global_meg_recordings[result['subset']] = result['meg']
        timing_rows.extend(result['timing_rows'])
        for mode in modes:
            for key in ordered_columns:
                global_df_per_mode[mode][key].extend(result['df_entries'][mode][key])

    # Save DataFrames
    for mode in modes:
        global_df = pd.DataFrame(global_df_per_mode[mode])
        print(f'DataFrame shape: {global_df.shape}')
        global_df.to_csv(
            os.path.join(preprocessed_dir, 'dataframe', f'df_{mode}{N_SUBJECTS}.csv'),
            index=False,
        )

    for subset, megs in global_meg_recordings.items():
        print(f'{subset} shape: {megs.shape}')

    # Save timing logs
    if profile_log and timing_rows:
        logs_dir = os.path.join(preprocessed_dir, 'preprocessing_logs')
        os.makedirs(logs_dir, exist_ok=True)
        profile_log_path = os.path.join(logs_dir, 'meg_preprocessing_timing.csv')
        timing_df = pd.DataFrame(timing_rows)
        timing_df.to_csv(profile_log_path, index=False)

    # Save MEG data
    out_dir = os.path.join(preprocessed_dir, 'meg')
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, f'meg{N_SUBJECTS}_sr{target_fs}.npz'),
        **global_meg_recordings,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for MEG preprocessing."""

    parser = argparse.ArgumentParser(description='Prepare MEG data and DataFrames')

    parser.add_argument(
        '--chunk-size-sec', type=float, default=3.0, help='Chunk size in seconds.'
    )
    parser.add_argument(
        '--target-fs', type=int, default=100, help='Desired MEG sampling rate.'
    )
    parser.add_argument(
        '--meg-dir',
        type=str,
        default=CLEAN_DATA_DIR,
        help='A directory with MEG files.',
    )
    parser.add_argument(
        '--meg-format',
        type=str,
        default='fif',
        choices=('bids', 'fif'),
        help='Which format meg files are in. For raw data bids is used, for clean fif is used.',
    )
    parser.add_argument(
        '--preprocessed-dir',
        type=str,
        default=PREPROCESSED_DATA_DIR,
        help='Root directory of audio chunks DataFrames.',
    )
    parser.add_argument(
        '--allow-test-only',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Allow processing stories that only appear in test; fits scalers on test-only data.',
    )
    parser.add_argument(
        '--profile-log',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Write timing logs to preprocessed_dir/preprocessing_logs/meg_preprocessing_timing.csv.',
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel jobs for MEG preprocessing. Use -1 for all CPUs, 1 for sequential.',
    )

    args = parser.parse_args()
    return args


def main_cli() -> None:
    """CLI entry point for MEG preprocessing."""
    args = parse_arguments()
    chunk_size = int(args.chunk_size_sec * args.target_fs)
    main(
        target_fs=args.target_fs,
        chunk_size=chunk_size,
        preprocessed_dir=args.preprocessed_dir,
        meg_dir=args.meg_dir,
        meg_format=args.meg_format,
        allow_test_only=args.allow_test_only,
        profile_log=args.profile_log,
        n_jobs=args.n_jobs,
    )


if __name__ == '__main__':
    main_cli()
