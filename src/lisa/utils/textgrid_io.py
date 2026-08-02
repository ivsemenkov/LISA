"""Lightweight MFA TextGrid parsing shared across analysis CLIs.

Kept dependency-free (only ``textgrid`` + a constant) so both plotting/analysis
modules and standalone CLIs can reuse it without importing heavy plotting code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textgrid import TextGrid

from lisa.utils.constants import AUDIO_SR


def parse_textgrid_tier_exact(textgrid_path: Path, tier_name: str) -> list[dict[str, Any]]:
    tg = TextGrid.fromFile(str(textgrid_path))
    matches = [tier for tier in tg.tiers if tier.name == tier_name]
    if len(matches) != 1:
        tier_names = [tier.name for tier in tg.tiers]
        raise ValueError(
            f'Expected exactly one TextGrid tier named {tier_name!r} in '
            f'{textgrid_path}, found {tier_names}.'
        )

    intervals = []
    for interval in matches[0].intervals:
        intervals.append(
            {
                'label': interval.mark.strip(),
                'start_sample': round(float(interval.minTime) * AUDIO_SR),
                'end_sample': round(float(interval.maxTime) * AUDIO_SR),
            }
        )
    return intervals
