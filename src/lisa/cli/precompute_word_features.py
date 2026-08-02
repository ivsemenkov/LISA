"""Precompute per-word linguistic features for occlusion-feature analysis.

For every MFA TextGrid ``words`` tier this computes, per spoken word:

* ``surprisal_bits``  -- GPT-2 ``-log2 p(word | preceding context)`` (subword
  surprisals summed to the word),
* ``entropy_bits``    -- entropy of GPT-2's next-token distribution just before
  the word (predictive uncertainty),
* ``zipf_freq``       -- word frequency on the Zipf scale (``wordfreq``),
* ``insertion_kind``  -- ground-truth stimulus label, recovered by diffing the
  spoken transcript against the original story text (``text/<story>.txt``):

    - ``none``     : a word of the original narrative,
    - ``single``   : an isolated inserted token (the MEG-MASC "odd word" /
      pseudoword probe of lexical access),
    - ``wordlist`` : part of an inserted contiguous block of >=2 random words
      (the MEG-MASC random word-list probe of context / syntax).

  This diff-based label is reliable where frequency/OOV heuristics were not: real
  but rare story words (``abominations``, ``caliche`` ...) stay ``none`` because
  they are present in the original text, while only genuinely inserted tokens are
  flagged.

Context is the *running story*: chunks of one story (``<story>_<n>.TextGrid``)
are concatenated in numeric order before scoring, so a word's surprisal uses the
real preceding text, not just its own chunk. Output is one CSV consumed later to
build word-level feature traces on the 100 Hz grid; it can be run independently
of the occlusion analysis.
"""

from __future__ import annotations

import argparse
import difflib
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from wordfreq import zipf_frequency

from lisa.utils.constants import DATA_ROOT, PREPROCESSED_DATA_DIR
from lisa.utils.textgrid_io import parse_textgrid_tier_exact

_CHUNK_RE = re.compile(r'^(?P<story>.+)_(?P<idx>\d+)$')
_LN2 = math.log(2.0)


def group_textgrids_by_story(mfa_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    """Group ``<story>_<n>.TextGrid`` files by story, ordered by chunk index."""

    stories: dict[str, list[tuple[int, Path]]] = {}
    for path in sorted(mfa_dir.glob('*.TextGrid')):
        match = _CHUNK_RE.match(path.stem)
        if not match:
            raise ValueError(f'Cannot parse <story>_<chunk> from {path.name!r}.')
        stories.setdefault(match['story'], []).append((int(match['idx']), path))
    for story in stories:
        stories[story].sort(key=lambda item: item[0])
    return stories


def load_story_words(chunks: list[tuple[int, Path]]) -> list[dict]:
    """Concatenate the spoken words of a story's chunks, in reading order."""

    records: list[dict] = []
    for idx, path in chunks:
        sound_fname = path.with_suffix('.wav').name
        word_index = 0
        for interval in parse_textgrid_tier_exact(path, 'words'):
            if interval['label'] == '':
                continue
            records.append(
                {
                    'sound_fname': sound_fname,
                    'chunk': idx,
                    'word_index': word_index,
                    'word': interval['label'],
                    'start_sample': int(interval['start_sample']),
                    'end_sample': int(interval['end_sample']),
                }
            )
            word_index += 1
    return records


def _word_key(word: str) -> str:
    """Lowercased match key: keep letters and internal apostrophes only."""

    return re.sub(r"[^a-z']", '', word.lower().replace('\u2019', "'"))


_CONTRACTION_SUFFIXES = {"n't", "'s", "'re", "'ve", "'ll", "'d", "'m"}


def tokenize_original(text: str) -> list[str]:
    """Tokenize an original story, merging split contractions back onto their word.

    The narrative text writes contractions as separate tokens (``would n't``,
    ``Chad 's``) whereas the aligned transcript keeps them whole (``wouldn't``,
    ``chad's``). Only genuine contraction suffixes (``n't`` / ``'s`` / ``'re`` ...)
    are merged back; a standalone quotation mark or spaced possessive apostrophe is
    dropped and surrounding quotes are stripped. Otherwise those apostrophes would
    glue onto the previous word and corrupt its key, so a quoted or possessive
    story word (``'normally'``, ``leaders '``) would surface as a false insertion.
    """

    tokens: list[str] = []
    for raw in re.findall(r"[A-Za-z']+", text.replace('\u2019', "'")):
        key = _word_key(raw)
        if not key or key == "'":
            continue
        if tokens and key in _CONTRACTION_SUFFIXES:
            tokens[-1] += key
        else:
            key = key.strip("'")
            if key:
                tokens.append(key)
    return tokens


def label_insertions(spoken_words: list[str], original_text: str) -> list[str]:
    """Tag each spoken word as ``none`` / ``single`` / ``wordlist`` vs original.

    Aligns the spoken (TextGrid) word sequence against the original story tokens.
    Words the aligner cannot match to the narrative are insertions: an isolated
    inserted token is ``single`` (pseudoword probe); a contiguous run of >=2 is
    ``wordlist`` (random word-list probe).
    """

    original = tokenize_original(original_text)
    spoken = [_word_key(w) for w in spoken_words]
    kinds = ['none'] * len(spoken)
    matcher = difflib.SequenceMatcher(a=original, b=spoken, autojunk=False)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in ('insert', 'replace'):
            kind = 'single' if (j2 - j1) == 1 else 'wordlist'
            for j in range(j1, j2):
                kinds[j] = kind
    return kinds


@torch.no_grad()
def score_words(
    words: list[str],
    tokenizer: GPT2TokenizerFast,
    model: GPT2LMHeadModel,
    max_len: int = 1024,
    stride: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """GPT-2 per-word surprisal (bits, summed over subwords) and pre-word entropy.

    Long stories are scored with strided windows so every token is predicted with
    as much left context as fits in ``max_len``; only tokens with adequate context
    (>= ``max_len - stride`` preceding tokens, or any in the first window) are
    committed. The story's first token is scored separately from GPT-2's
    document-boundary token, so all returned values are finite.
    """

    token_ids: list[int] = []
    spans: list[tuple[int, int]] = []
    for i, word in enumerate(words):
        piece = word if i == 0 else ' ' + word
        ids = tokenizer.encode(piece)
        if not ids:
            ids = tokenizer.encode(' ' + word) or [tokenizer.eos_token_id]
        spans.append((len(token_ids), len(token_ids) + len(ids)))
        token_ids.extend(ids)

    n_tokens = len(token_ids)
    tokens = torch.tensor(
        token_ids, dtype=torch.long, device=next(model.parameters()).device
    )
    surprisal = np.full(n_tokens, np.nan)  # bits, indexed by predicted token
    entropy = np.full(n_tokens, np.nan)  # bits, uncertainty before that token

    boundary = tokens.new_tensor([[tokenizer.bos_token_id]])
    boundary_logits = model(boundary).logits[0, -1]
    boundary_logp = torch.log_softmax(boundary_logits, dim=-1)
    surprisal[0] = -boundary_logp[token_ids[0]].item() / _LN2
    entropy[0] = -(boundary_logp.exp() * boundary_logp).sum().item() / _LN2

    start = 0
    while start < n_tokens:
        end = min(start + max_len, n_tokens)
        logits = model(tokens[start:end].unsqueeze(0)).logits[0]  # (win, vocab)
        logp = torch.log_softmax(logits, dim=-1)
        ent = -(logp.exp() * logp).sum(dim=-1) / _LN2  # (win,) bits
        commit_from = start + 1 if start == 0 else start + (max_len - stride)
        for j in range(end - start - 1):
            predicted = start + j + 1  # global index of token predicted at pos j
            if predicted < commit_from:
                continue
            surprisal[predicted] = -logp[j, token_ids[predicted]].item() / _LN2
            entropy[predicted] = ent[j].item()
        if end == n_tokens:
            break
        start += stride

    if not np.isfinite(surprisal).all() or not np.isfinite(entropy).all():
        raise RuntimeError('GPT-2 scoring left non-finite token features.')

    word_surprisal = np.full(len(words), np.nan)
    word_entropy = np.full(len(words), np.nan)
    for wi, (lo, hi) in enumerate(spans):
        if hi > lo:
            word_surprisal[wi] = float(np.sum(surprisal[lo:hi]))
        if lo < n_tokens:
            word_entropy[wi] = entropy[lo]
    return word_surprisal, word_entropy


def build_table(mfa_dir: Path, text_dir: Path, device: torch.device) -> pd.DataFrame:
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()

    stories = group_textgrids_by_story(mfa_dir)
    frames: list[pd.DataFrame] = []
    for story, chunks in stories.items():
        records = load_story_words(chunks)
        if not records:
            raise ValueError(f'No spoken words found for story {story!r}.')
        words = [rec['word'] for rec in records]
        surprisal, entropy = score_words(words, tokenizer, model)
        story_text = (text_dir / f'{story}.txt').read_text()
        df = pd.DataFrame(records)
        df.insert(0, 'story', story)
        df['surprisal_bits'] = surprisal
        df['entropy_bits'] = entropy
        df['zipf_freq'] = np.array([zipf_frequency(w.lower(), 'en') for w in words])
        df['insertion_kind'] = label_insertions(words, story_text)
        frames.append(df)
        counts = df['insertion_kind'].value_counts()
        print(
            f'[{story}] words={len(df)} chunks={len(chunks)} '
            f'single={int(counts.get("single", 0))} '
            f'wordlist={int(counts.get("wordlist", 0))} '
            f'mean_surprisal={np.nanmean(surprisal):.2f} bits'
        )
    return pd.concat(frames, ignore_index=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        '--mfa-dir',
        type=Path,
        default=Path(DATA_ROOT) / 'MASC-MEG' / 'stimuli' / 'combined_mfa',
        help='Directory with MFA TextGrid files (words tier).',
    )
    parser.add_argument(
        '--text-dir',
        type=Path,
        default=Path(DATA_ROOT) / 'MASC-MEG' / 'stimuli' / 'text',
        help='Directory with original story texts (<story>.txt) for insertion diff.',
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=Path(PREPROCESSED_DATA_DIR) / 'linguistic' / 'word_features.csv',
        help='Output CSV of per-word linguistic features.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help="Torch device for GPT-2 ('cpu', 'cuda', ...).",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    table = build_table(args.mfa_dir, args.text_dir, device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f'[done] wrote {len(table)} word rows to {args.out}')


def main_cli() -> None:
    main(build_parser().parse_args())


if __name__ == '__main__':
    main_cli()
