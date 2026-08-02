"""Find out-of-vocabulary words relative to an MFA lexicon."""

import argparse
import os
import re

from lisa.utils.constants import PROJECT_ROOT


# 1. Load known words from MFA dictionary
def load_dict_words(dict_path: str) -> set[str]:
    """Load MFA lexicon words from a dictionary file."""
    words = set()
    with open(dict_path, encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith(';;;'):
                w = line.split()[0].lower()
                words.add(w)
    return words


# 2. Extract all unique words from lab files
def extract_lab_words(lab_dir: str) -> set[str]:
    """Extract unique tokens from .txt lab files."""
    lab_words = set()
    for fname in os.listdir(lab_dir):
        if fname.lower().endswith('.txt'):
            with open(os.path.join(lab_dir, fname), encoding='utf-8') as f:
                text = f.read().lower()
                # Split on non-word characters, but keep apostrophes
                words = re.findall(r"\b[\w']+\b", text)
                lab_words.update(words)
    return lab_words


def extract_oov_words(lab_dir: str, dict_path: str, verbose: bool) -> None:
    """Compute and save OOV words for MFA dictionary extension."""

    dict_words = load_dict_words(dict_path)
    lab_words = extract_lab_words(lab_dir)

    # 3. Compute OOV
    oov = sorted(lab_words - dict_words)
    if verbose:
        print('Words not in dictionary (OOV):')
        for word in oov:
            print(word)

    # 4. Optionally, write to a file for custom dictionary creation
    with open(os.path.join(lab_dir, '../oov_words.txt'), 'w', encoding='utf-8') as f:
        for word in oov:
            f.write(word + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find OOV words from MFA dictionary in lab files'
    )
    parser.add_argument(
        '--lab-dir',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'data', 'MASC-MEG', 'stimuli', 'combined'),
        help='Directory containing .lab files',
    )
    parser.add_argument(
        '--mfa-dict',
        type=str,
        default=None,
        required=True,
        help='Path to MFA dictionary file (e.g. .../english_mfa.dict from MFA pretrained models).',
    )
    parser.add_argument(
        '--verbose',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Whether to print verbose logs.',
    )
    args = parser.parse_args()

    extract_oov_words(
        lab_dir=args.lab_dir, dict_path=args.mfa_dict, verbose=args.verbose
    )
