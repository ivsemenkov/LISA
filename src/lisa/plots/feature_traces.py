"""Build numeric feature traces on the per-sound feature-frame grid.

Interpretable acoustic features are represented as
``dict[sound_name -> np.ndarray]`` of length ``L_g`` (the sound's frame count)
on a shared feature-frame grid. Word-derived features come from the precomputed
word-features table (``lisa.cli.precompute_word_features``); callers such as
occlusion analysis load that table with ``load_word_features`` and build their
own mask / value traces.

Frame mapping for word spans matches the TextGrid mask pipeline's "any overlap"
convention: frame ``f`` spans audio samples ``[f*hop, (f+1)*hop)`` from the
sound's start.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from lisa.utils.audio_io import load_audio_mono_16k
from lisa.utils.constants import AUDIO_SR

try:
    from scipy.stats import rankdata as _rankdata
except ImportError:  # pragma: no cover - scipy is a project dependency
    _rankdata = None


def load_word_features(csv_path: Path) -> pd.DataFrame:
    """Load the per-word feature table and validate the columns we rely on."""

    df = pd.read_csv(csv_path)
    required = {
        'sound_fname',
        'start_sample',
        'end_sample',
        'insertion_kind',
        'surprisal_bits',
        'entropy_bits',
        'zipf_freq',
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{csv_path}: word-features table is missing columns {sorted(missing)}.')
    return df


# A source may fall a few frames short of the analysis grid for a deterministic,
# benign reason (e.g. librosa's center=False STFT drops a partial trailing frame).
# Padding beyond this fraction would mean running on fabricated data, so it fails.
_MAX_PAD_FRACTION = 0.01


def _fit_length(x: np.ndarray, length: int, fill: float, ctx: str) -> np.ndarray:
    """Trim, or minimally tail-pad, ``x`` to exactly ``length``; fail on gross pads.

    Trimming a longer source is always fine -- those elements fall outside the
    analysed window grid. Padding invents values, so it is capped at
    ``_MAX_PAD_FRACTION`` of the grid (a small, deterministic framing edge); a
    larger shortfall is a hard error rather than silently running on fabricated
    frames.
    """

    n = x.shape[0]
    if n < length:
        pad = length - n
        if pad > max(1, int(_MAX_PAD_FRACTION * length)):
            raise ValueError(
                f'{ctx}: source has {n} elements but the analysis grid needs {length} '
                f'({pad} missing); refusing to pad that many fabricated values (fail-fast).'
            )
    out = np.full(length, fill, dtype=np.float64)
    m = min(n, length)
    out[:m] = x[:m]
    return out


def audio_feature_trace(
    audio_dir: Path,
    sound_names: list[str],
    sound_length: dict[str, int],
    hop: int,
    kind: str,
    rank: bool = False,
) -> dict[str, np.ndarray]:
    """Continuous acoustic feature computed from the stimulus audio on the grid.

    ``loudness`` frames are left-aligned and non-overlapping (``hop`` samples at
    ``AUDIO_SR``), matching the energy-mask frame convention.
    ``acoustic_onset`` uses librosa's centered 128 ms analysis window, so its
    frame ``t`` is timing-compensated to the same audio position as the other
    features (to within half a hop); a non-centered envelope would sit ~6 frames
    (60 ms) early relative to this grid.

    * ``loudness``       -- per-frame log mean-square energy (log-energy; an
      energy-based loudness proxy, not perceptual loudness);
    * ``acoustic_onset`` -- mel-spectral-flux onset strength (librosa), i.e. how
      fast the spectrum is changing -- large at note/phone/word acoustic onsets;
      128 ms window, 10 ms hop, centered-frame timing.

    Pitch (F0) is intentionally not offered here: F0 is undefined on unvoiced /
    silent frames (a large, unavoidable fraction of speech), so any pitch trace
    would be mostly an invented baseline -- exactly the kind of silent fabrication
    this module refuses to do.

    With ``rank=True`` each sound's trace is replaced by its per-sound percentile
    ranks in (0, 1) (robust to the heavy tails of loudness/onset spikes; the
    shared (0, 1) scale keeps sounds of different length comparable, so longer
    sounds do not dominate the pooled correlation).
    """

    if kind not in {'loudness', 'acoustic_onset'}:
        raise ValueError(f"audio kind must be loudness/acoustic_onset, got {kind!r}.")
    if rank and _rankdata is None:  # pragma: no cover
        raise ImportError('scipy is required for rank=True (percentile-rank) traces.')
    out: dict[str, np.ndarray] = {}
    for name in sound_names:
        L = sound_length[name]
        audio = load_audio_mono_16k(audio_dir / name).astype(np.float64)
        if kind == 'loudness':
            samples = _fit_length(
                audio, L * hop, fill=0.0, ctx=f'loudness audio for sound {name!r}'
            )
            energy = np.mean(samples.reshape(L, hop) ** 2, axis=1)
            floor = float(np.max(energy)) * 1e-6 + 1e-12
            trace = np.log(energy + floor)
        elif kind == 'acoustic_onset':
            env = librosa.onset.onset_strength(
                y=audio, sr=AUDIO_SR, hop_length=hop, n_fft=2048, center=True
            )
            trace = _fit_length(
                np.asarray(env, dtype=np.float64), L, fill=0.0,
                ctx=f'acoustic_onset envelope for sound {name!r}',
            )
        # Fail-fast: never let the analysis proceed on non-finite feature values.
        if not np.isfinite(trace).all():
            n_bad = int(np.sum(~np.isfinite(trace)))
            raise ValueError(
                f'audio_feature_trace: {kind} trace for sound {name!r} has {n_bad} '
                f'non-finite value(s) (NaN/Inf); refusing to continue (fail-fast).'
            )
        if rank:
            # Per-sound percentile in (0, 1); raw frame ranks (1..L) would scale
            # with the sound length L and over-weight longer sounds in pooled
            # analyses.
            trace = (_rankdata(trace) - 0.5) / trace.shape[0]
        out[name] = trace.astype(np.float64)
    return out
