"""Expand numeric tokens in text files before MFA alignment."""

import argparse
import os
import re

from num2words import num2words

from lisa.utils.constants import PROJECT_ROOT


def expand_numbers_in_text(text: str) -> str:
    """Convert numeric tokens to words (e.g., 12 -> twelve)."""

    def repl(match):
        num_str = match.group(0).replace(',', '')
        try:
            if '.' in num_str:
                left, right = num_str.split('.')
                left_words = num2words(int(left))
                right_words = ' '.join([num2words(int(d)) for d in right])
                return f'{left_words} point {right_words}'
            else:
                return num2words(int(num_str))
        except Exception:
            # If it can't be parsed (e.g., too big for int), leave as-is
            return match.group(0)

    # Match numbers with optional commas and decimals
    return re.sub(r'\d[\d,]*\.?\d*', repl, text)


def process_numbers_all_texts(text_dir: str, verbose: bool = False) -> None:
    """Expand numbers for all .txt files under text_dir."""
    for fname in os.listdir(text_dir):
        if fname.lower().endswith('.txt'):
            fpath = os.path.join(text_dir, fname)
            with open(fpath, encoding='utf-8') as f:
                original = f.read()
            expanded = expand_numbers_in_text(original)
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(expanded)
            if verbose:
                print(f'Processed: {fname}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processing numbers in texts for MFA')
    parser.add_argument(
        '--text-dir',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'data', 'MASC-MEG', 'stimuli', 'combined'),
        help='A directory with text stimuli of MASC-MEG',
    )
    parser.add_argument(
        '--verbose',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Whether to print verbose logs.',
    )
    args = parser.parse_args()
    process_numbers_all_texts(text_dir=args.text_dir, verbose=args.verbose)
