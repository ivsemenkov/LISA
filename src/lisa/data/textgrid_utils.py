"""TextGrid parsing utilities shared by chunking and embedding scripts."""

from __future__ import annotations

from textgrid import TextGrid


def parse_textgrid_intervals(
    textgrid_path: str,
    sr: int,
    tier_name: str = 'words',
    include_empty: bool = True,
) -> list[dict]:
    """Parse intervals from a TextGrid tier as sample indices."""

    tg = TextGrid.fromFile(textgrid_path)
    tier = None
    for t in tg.tiers:
        if tier_name.lower() in t.name.lower():
            tier = t
            break
    if not tier:
        raise ValueError(f"No tier with name containing '{tier_name}' found.")

    intervals = []
    for interval in tier.intervals:
        label = interval.mark.strip()
        if label or include_empty:
            intervals.append(
                {
                    'label': label,
                    'start_sample': round(float(interval.minTime) * sr),
                    'end_sample': round(float(interval.maxTime) * sr),
                }
            )
    return intervals


def parse_textgrid_words(
    textgrid_path: str, sr: int, tier_name: str = 'words'
) -> list[dict]:
    """Parse word intervals from a TextGrid file as sample indices.

    Args:
        textgrid_path: Path to the TextGrid file.
        sr: Sample rate for converting times to sample indices.
        tier_name: Name of the tier containing word intervals.

    Returns:
        List of dicts with 'word', 'start_sample', and 'end_sample' keys.
    """
    word_times = []
    for interval in parse_textgrid_intervals(
        textgrid_path, sr=sr, tier_name=tier_name, include_empty=False
    ):
        word = interval['label']
        if word:
            word_times.append(
                {
                    'word': word,
                    'start_sample': interval['start_sample'],
                    'end_sample': interval['end_sample'],
                }
            )
    return word_times


def word_majority_overlap(
    word: dict, chunk_start_sample: int, chunk_end_sample: int
) -> bool:
    """Return True if word overlaps >= 50% in sample space (integer arithmetic).

    Uses integer comparison (overlap * 2 >= duration) to avoid float division
    and ensure deterministic, reproducible results across runs.

    Args:
        word: Dict with 'start_sample' and 'end_sample' keys.
        chunk_start_sample: Start sample index of the chunk.
        chunk_end_sample: End sample index of the chunk.

    Returns:
        True if the word's overlap with the chunk is at least 50% of word duration.
    """
    overlap = max(
        0,
        min(word['end_sample'], chunk_end_sample)
        - max(word['start_sample'], chunk_start_sample),
    )
    duration = word['end_sample'] - word['start_sample']
    # Integer comparison: overlap * 2 >= duration avoids division entirely
    return duration > 0 and overlap * 2 >= duration
