"""Temporal-filter preprocessing helpers used by interpretation plots."""

import numpy as np

from lisa.utils.numeric import EPS, assert_finite as _assert_finite


def filter_data(
    data: np.ndarray,
    temporal_filter: np.ndarray,
    filtfilt: bool,
) -> np.ndarray:
    """Apply 1D convolution, optionally forward-backward."""
    if data.ndim != 1:
        raise ValueError(f'Expected 1D data, got {data.shape}.')
    y = np.convolve(data, temporal_filter, mode='same')
    if filtfilt:
        y = np.convolve(np.flip(y, axis=-1), temporal_filter, mode='same')
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
