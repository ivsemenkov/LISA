"""Lightweight audio loading shared across analysis CLIs.

Kept dependency-light (only ``librosa`` + a constant) so both the plotting/
analysis modules and feature-trace builders can load stimulus audio without
importing heavy plotting/model code.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from lisa.utils.constants import AUDIO_SR


def load_audio_mono_16k(audio_path: Path) -> np.ndarray:
    """Load mono audio at ``AUDIO_SR``, matching the feature-extraction contract."""

    audio, orig_sr = librosa.load(audio_path, sr=None, mono=True)
    if orig_sr != AUDIO_SR:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=AUDIO_SR)
    return np.asarray(audio, dtype=np.float32)
