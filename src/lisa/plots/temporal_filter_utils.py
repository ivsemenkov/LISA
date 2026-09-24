"""Temporal-filter preprocessing helpers used by interpretation plots."""

import numpy as np

from lisa.utils.numeric import EPS, assert_finite as _assert_finite


def _convolve_same(
    data: np.ndarray,
    kernel: np.ndarray,
    pad_value: float = 0.0,
) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if pad_value == 0.0:
        return np.convolve(data, kernel, mode='same')
    return (
        np.convolve(data - pad_value, kernel, mode='same')
        + pad_value * kernel.sum()
    )


def filter_data(
    data: np.ndarray,
    temporal_filter: np.ndarray,
    filtfilt: bool,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Apply 1D convolution, optionally forward-backward."""
    if data.ndim != 1:
        raise ValueError(f'Expected 1D data, got {data.shape}.')
    y = _convolve_same(data, temporal_filter, pad_value)
    if filtfilt:
        kernel_sum = float(np.asarray(temporal_filter, dtype=np.float64).sum())
        y = _convolve_same(
            np.flip(y, axis=-1),
            temporal_filter,
            pad_value * kernel_sum,
        )
        y = np.flip(y, axis=-1)
    return y


def demean_temporal_filters(temporal_filters: np.ndarray) -> np.ndarray:
    """Subtract each temporal filter row mean before interpretation analyses."""
    temporal_filters = np.asarray(temporal_filters, dtype=np.float64)
    if temporal_filters.ndim != 2:
        raise ValueError(
            f'Expected temporal filters to be 2D, got {temporal_filters.shape}.'
        )
    _assert_finite('temporal_filters_before_demean', temporal_filters)
    demeaned = temporal_filters - temporal_filters.mean(axis=1, keepdims=True)
    near_zero = np.where(np.linalg.norm(demeaned, axis=1) <= EPS)[0]
    if near_zero.size:
        raise ValueError(
            'Cannot demean temporal filters with near-zero residual rows: '
            f'{near_zero.tolist()}'
        )
    _assert_finite('demeaned_temporal_filters', demeaned)
    return demeaned
