"""Prepare MASC-MEG stimuli layout for MFA alignment."""

import argparse
import os
import shutil

from lisa.utils.constants import PROJECT_ROOT


def copy_files(
    src_dir: str, dst_dir: str, middle_remove: str | None = None, verbose: bool = False
) -> None:
    """Copy stimuli files to MFA combined directory.

    Args:
        src_dir: Source directory (audio or text_with_wordlists).
        dst_dir: Destination directory (combined).
        middle_remove: If set, remove the first occurrence of _{middle_remove}_ in filenames.
    """
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        if fname.endswith('.ipynb_checkpoints'):
            continue
        if fname.endswith('lw1.wav'):
            # it has split story lw1 that we need, this is a full file which is not aligned with text
            continue
        new_fname = fname
        if middle_remove is not None:
            # Remove _{middle_remove}_ from the first occurrence in the filename
            tag = f'_{middle_remove}_'
            if tag in fname:
                new_fname = fname.replace(tag, '_', 1)
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, new_fname)
        shutil.copy2(src_path, dst_path)
        if verbose:
            print(f'Copied {src_path} to {dst_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Copy stimuli into the same directory for MFA'
    )
    parser.add_argument(
        '--source-dir',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'data', 'MASC-MEG', 'stimuli'),
        help='A source directory with stimuli of MASC-MEG',
    )
    parser.add_argument(
        '--target-dir',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'data', 'MASC-MEG', 'stimuli', 'combined'),
        help='A destination for MFA directory with stimuli on MASC-MEG',
    )
    parser.add_argument(
        '--verbose',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Whether to print verbose logs.',
    )
    args = parser.parse_args()
    src_dir = args.source_dir
    dst_dir = args.target_dir

    copy_files(src_dir=os.path.join(src_dir, 'audio'), dst_dir=dst_dir)
    copy_files(
        src_dir=os.path.join(src_dir, 'text_with_wordlists'),
        dst_dir=dst_dir,
        middle_remove='produced',
    )
