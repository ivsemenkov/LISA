"""Small pure numeric helpers shared across plotting and MEG code."""

from __future__ import annotations

import numpy as np


EPS = 1e-12


def assert_finite(name: str, arr: np.ndarray | None) -> None:
    """Guard against NaNs/infs before heavy ops."""
    if arr is None:
        return
    if not np.isfinite(arr).all():
        raise ValueError(
            f"Non-finite values detected in {name}: "
            f"nan={np.isnan(arr).sum()}, "
            f"pos_inf={(arr == np.inf).sum()}, "
            f"neg_inf={(arr == -np.inf).sum()}"
        )


def fft_magnitude(
    h: np.ndarray,
    fs: float,
    square_fft: bool,
    n_fft: int = 4096,
    normalize: bool = True,
    demean: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute magnitude rFFT and return (freqs, magnitude) for 0..Nyquist."""
    if demean:
        h = h - np.mean(h)
    H = np.fft.rfft(h, n=n_fft)
    f = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    magnitude = np.abs(H)
    if square_fft:
        magnitude = magnitude**2
    if normalize:
        magnitude /= magnitude.max() + EPS
    return f, magnitude
