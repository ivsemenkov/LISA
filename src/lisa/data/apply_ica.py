"""Apply ICA components to raw MEG data and save cleaned files.

This script:
1. Loads raw MEG data from data/MASC-MEG (BIDS format)
2. Loads ICA components from data/clean/sub-{sub:02d}/ses-{ses}/meg/
3. Applies ICA to raw data: clean_raw = ica.apply(raw)
4. Saves cleaned files as sub-{sub:02d}_ses-{ses}_task-{story_id}_desc-clean_meg.fif in data/clean/sub-{sub:02d}/ses-{ses}/meg/
"""

import argparse
import os
from pathlib import Path

import mne
from joblib import Parallel, delayed
from tqdm import tqdm

from lisa.data.meg_io import load_raw_meg
from lisa.utils.constants import CLEAN_DATA_DIR, DATA_ROOT, SUB_SES_COMBOS, STORY_IDS


def apply_ica_to_raw(
    sub: int, ses: int, story_id: int, raw_meg_dir: str, ica_dir: str, output_dir: str
) -> tuple[bool, str]:
    """Apply ICA to raw MEG data for a single subject/session/story combination.

    Args:
        sub: Subject ID
        ses: Session ID
        story_id: Story ID (0-3)
        raw_meg_dir: Directory containing raw MEG data (BIDS format)
        ica_dir: Directory containing ICA files
        output_dir: Directory to save cleaned files

    Returns:
        Tuple of (success: bool, message: str)
    """
    sub_str = f'{sub:02d}'

    # 1. Load raw data from MASC-MEG (BIDS format)
    try:
        raw, _ = load_raw_meg(
            meg_format='bids',
            data_root=raw_meg_dir,
            sub=sub,
            ses=ses,
            story_id=story_id,
        )
    except FileNotFoundError as e:
        return False, f'Raw data not found: sub-{sub_str}_ses-{ses}_task-{story_id}'
    except Exception as e:
        return False, f'Failed to load raw data: {e}'

    # 2. Load ICA component from clean directory
    ica_path = (
        Path(ica_dir)
        / f'sub-{sub_str}'
        / f'ses-{ses}'
        / 'meg'
        / f'sub-{sub_str}_ses-{ses}_task-{story_id}_meg-ica.fif'
    )

    if not ica_path.exists():
        return False, f'ICA file not found: {ica_path.name}'

    try:
        ica = mne.preprocessing.read_ica(ica_path, verbose=False)
    except Exception as e:
        return False, f'Failed to load ICA: {e}'

    if raw.ch_names != ica.ch_names:
        if set(raw.ch_names) == set(ica.ch_names):
            return (
                False,
                f'Channel order mismatch between raw and ICA (n={len(raw.ch_names)})',
            )
        missing = [ch for ch in ica.ch_names if ch not in raw.ch_names]
        extra = [ch for ch in raw.ch_names if ch not in ica.ch_names]
        return (
            False,
            f'Channel set mismatch between raw and ICA: missing={missing} extra={extra}',
        )

    # 3. Apply ICA to raw data
    try:
        ica.apply(raw, verbose=False)
    except Exception as e:
        return False, f'Failed to apply ICA: {e}'

    # 4. Save cleaned file in BIDS-like structure: output_dir/sub-{sub}/ses-{ses}/meg/
    output_filename = f'sub-{sub_str}_ses-{ses}_task-{story_id}_desc-clean_meg.fif'
    output_path = (
        Path(output_dir) / f'sub-{sub_str}' / f'ses-{ses}' / 'meg' / output_filename
    )

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Warn if file already exists
    if output_path.exists():
        print(
            f'WARNING: Overwriting existing file: {output_filename} '
            f'(sub-{sub_str}_ses-{ses}_task-{story_id})'
        )

    try:
        raw.save(output_path, overwrite=True, verbose=False)
        return True, f'Saved: {output_filename}'
    except Exception as e:
        return False, f'Failed to save: {e}'


def main(raw_meg_dir: str, ica_dir: str, output_dir: str, n_jobs: int = 1):
    """Process all combinations of SUB_SES_COMBOS and story_ids.

    Args:
        raw_meg_dir: Directory containing raw MEG data (BIDS format)
        ica_dir: Directory containing ICA files
        output_dir: Directory to save cleaned files
        n_jobs: Number of parallel jobs (-1 for all CPUs, 1 for sequential)
    """
    print(f'Applying ICA to {len(SUB_SES_COMBOS)} subject-session combinations')
    print(
        f'and {len(STORY_IDS)} stories ({len(SUB_SES_COMBOS) * len(STORY_IDS)} total files)'
    )
    print(f'Raw MEG source: {raw_meg_dir}')
    print(f'ICA source: {ica_dir}/sub-*/ses-*/meg/')
    print(f'Output directory: {output_dir}/sub-*/ses-*/meg/')
    print()

    total = len(SUB_SES_COMBOS) * len(STORY_IDS)
    success_count = 0
    error_count = 0
    errors = []

    # Create list of all combinations
    all_combos = [
        (sub, ses, story_id) for sub, ses in SUB_SES_COMBOS for story_id in STORY_IDS
    ]

    # Process combinations (parallel or sequential)
    if n_jobs == 1:
        # Sequential processing with progress bar
        for sub, ses, story_id in tqdm(all_combos, desc='Applying ICA'):
            success, message = apply_ica_to_raw(
                sub, ses, story_id, raw_meg_dir, ica_dir, output_dir
            )

            if success:
                success_count += 1
            else:
                error_count += 1
                errors.append(f'sub-{sub:02d}_ses-{ses}_task-{story_id}: {message}')
    else:
        # Parallel processing with joblib
        print(f'Using {n_jobs} parallel jobs...')
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(apply_ica_to_raw)(
                sub, ses, story_id, raw_meg_dir, ica_dir, output_dir
            )
            for sub, ses, story_id in all_combos
        )

        # Collect results
        for (sub, ses, story_id), (success, message) in zip(all_combos, results):
            if success:
                success_count += 1
            else:
                error_count += 1
                errors.append(f'sub-{sub:02d}_ses-{ses}_task-{story_id}: {message}')

    print()
    print('=' * 60)
    print('ICA application complete!')
    print(f'  Total combinations: {total}')
    print(f'  Successfully processed: {success_count}')
    print(f'  Errors: {error_count}')
    print('=' * 60)

    if errors:
        print('\nErrors encountered:')
        for error in errors[:10]:  # Show first 10 errors
            print(f'  - {error}')
        if len(errors) > 10:
            print(f'  ... and {len(errors) - 10} more errors')

    if error_count > 0:
        raise RuntimeError(
            f'ICA failed for {error_count}/{total} recordings'
        )


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for ICA application."""

    parser = argparse.ArgumentParser(description='Apply ICA components to raw MEG data')

    parser.add_argument(
        '--raw-meg-dir',
        type=str,
        default=os.path.join(DATA_ROOT, 'MASC-MEG'),
        help='Directory containing raw MEG data in BIDS format.',
    )
    parser.add_argument(
        '--ica-dir',
        type=str,
        default=CLEAN_DATA_DIR,
        help='Directory containing ICA files (sub-*/ses-*/meg/ structure).',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=CLEAN_DATA_DIR,
        help='Directory to save cleaned MEG files.',
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel jobs for ICA application. Use -1 for all CPUs, 1 for sequential.',
    )

    args = parser.parse_args()
    return args


def main_cli() -> None:
    """CLI entry point for ICA application."""
    args = parse_arguments()
    main(
        raw_meg_dir=args.raw_meg_dir,
        ica_dir=args.ica_dir,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
    )


if __name__ == '__main__':
    main_cli()
