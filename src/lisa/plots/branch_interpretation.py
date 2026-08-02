"""Branch-interpretation extraction, helpers, and combined clustering."""

from __future__ import annotations

import re
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.covariance import LedoitWolf
from tqdm import tqdm

from lisa.plots.temporal_filter_utils import demean_temporal_filters, filter_data
from lisa.utils.constants import PREPROCESSED_DATA_DIR
from lisa.utils.numeric import EPS, assert_finite as _assert_finite, fft_magnitude


DEFAULT_OFFSET_GAP_SEC = 10.0
DEFAULT_SPATIAL_ROUGHNESS_NEIGHBORS = 6
DEFAULT_MINIMUM_SIMILARITY = 0.25
DEFAULT_MAIN_SPATIAL_ROUGHNESS_MAX = 1.0
TEMPORAL_FILTER_SUFFIX = {
    True: '_dtf',
    False: '_raw_tf',
}


# ---------------------------------------------------------------------------
# MEG / filter loading
# ---------------------------------------------------------------------------


def load_fixed_meg_batch(
    preprocessed_meg_path: str | Path,
    target_fs: float,
    sub: int,
    ses: int,
    story_id: int,
    offset_gap_sec: float = DEFAULT_OFFSET_GAP_SEC,
) -> np.ndarray:
    """Load a fixed MEG segment (channels, time) for one subject."""
    offset_gap = int(offset_gap_sec * target_fs)
    meg = np.load(preprocessed_meg_path)[
        f'subject{sub:02d}_session{ses}_story{story_id}'
    ]
    meg = meg[:, offset_gap:-offset_gap].astype(np.float64)
    _assert_finite('meg_batch', meg)
    return meg


def load_subject_meg_and_raw(
    sub: int,
    ses: int,
    story_id: int,
    target_fs: float,
    offset_gap: float,
    meg_format: str,
    data_root: str | Path,
    raw_meg: bool,
    preprocessed_meg_path: str | Path,
    preprocess: bool,
    *,
    raw_metadata_only: bool = False,
):
    """Load analysis MEG and its Raw sensor metadata.

    ``raw_metadata_only`` avoids loading and resampling the Raw recording when
    preprocessed MEG supplies the analysis data. Source amplitudes depend on
    sensor geometry, not the Raw sampling frequency.
    """
    from lisa.data.meg_io import load_meg, load_raw_meg

    if raw_meg:
        meg, raw = load_meg(
            target_fs=target_fs,
            sub=sub,
            ses=ses,
            story_id=story_id,
            offset_gap=offset_gap,
            return_raw=True,
            preprocess=preprocess,
        )
    else:
        meg = load_fixed_meg_batch(
            preprocessed_meg_path=preprocessed_meg_path,
            target_fs=target_fs,
            sub=sub,
            ses=ses,
            story_id=story_id,
            offset_gap_sec=offset_gap,
        )
        raw, _ = load_raw_meg(
            meg_format=meg_format,
            data_root=str(data_root),
            sub=sub,
            ses=ses,
            story_id=story_id,
            preload=not raw_metadata_only,
        )
        if not raw_metadata_only:
            raw.resample(target_fs, npad='auto')

    meg = np.asarray(meg, dtype=np.float64)
    _assert_finite(f'meg_subject_{sub}', meg)
    return meg, raw


def extract_run_filters(
    model,
    hyper_params: dict[str, Any],
    *,
    demean_temporal_filters_enabled: bool,
) -> dict[str, Any]:
    """Extract temporal and spatial filters (numpy) and metadata."""
    K = hyper_params['n_channels_unmix']
    temporal_filter_type = hyper_params['temporal_filter_type']
    filtfilt = temporal_filter_type == 'filtfilt'

    temporal_filters = model.extract_temporal_filters()
    temporal_filters = temporal_filters.detach().cpu().numpy().astype(np.float64)
    if demean_temporal_filters_enabled:
        temporal_filters = demean_temporal_filters(temporal_filters)
    spatial_weight, spatial_bias = model.extract_spatial_filters()
    spatial_weight = spatial_weight.detach().cpu().numpy().astype(np.float64)
    _assert_finite('temporal_filters', temporal_filters)
    _assert_finite('spatial_filters_weight', spatial_weight)

    return {
        'temporal_filters': temporal_filters,
        'spatial_filters_weight': spatial_weight,
        'spatial_filters_bias': (
            spatial_bias.detach().cpu().numpy().astype(np.float64)
            if spatial_bias is not None
            else None
        ),
        'K': K,
        'filtfilt': filtfilt,
        'demean_temporal_filters': bool(demean_temporal_filters_enabled),
    }


# ---------------------------------------------------------------------------
# Pattern math
# ---------------------------------------------------------------------------


def calculate_spatial_patterns(
    meg: np.ndarray,
    spatial_filters_weight: np.ndarray,
    temporal_filters: np.ndarray,
    sub: int,
    filtfilt: bool,
    *,
    return_covariances: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute spatial patterns by temporally filtering MEG and estimating covariance.

    Args:
        meg: MEG data (channels, time).
        spatial_filters_weight: Spatial filters (subjects, branches, channels).
        temporal_filters: Temporal filters (branches, kernel).
        sub: Subject id (1-indexed). Converted to 0-indexed for model indexing.
        filtfilt: Apply zero-phase temporal filtering.
        return_covariances: Also return Ledoit-Wolf covariances
            ``(branches, channels, channels)``.

    Returns:
        Spatial patterns (branches, channels), or ``(patterns, covariances)``.
    """
    meg = np.asarray(meg, dtype=np.float64)
    _assert_finite('meg_before_spatial_patterns', meg)
    _assert_finite('spatial_filters_weight', spatial_filters_weight)
    _assert_finite('temporal_filters', temporal_filters)

    patterns = []
    covariances = []
    for branch, temporal_filter in enumerate(temporal_filters):
        filtered = np.empty_like(meg, dtype=np.float64)
        kernel = temporal_filter[::-1]
        for ch in range(meg.shape[0]):
            filtered[ch, :] = filter_data(
                data=meg[ch, :],
                temporal_filter=kernel,
                filtfilt=filtfilt,
            )
        _assert_finite('temporally_filtered_meg', filtered)
        filtered -= filtered.mean(axis=1, keepdims=True)
        if filtered.shape[1] <= 1:
            raise ValueError(
                'Not enough time points to form spatial covariance: '
                f'filtered shape={filtered.shape}'
            )

        covariance = LedoitWolf(store_precision=False).fit(filtered.T).covariance_
        patterns.append(np.dot(covariance, spatial_filters_weight[sub - 1, branch]))
        covariances.append(covariance)

    result = np.stack(patterns, axis=0)
    if return_covariances:
        return result, np.stack(covariances, axis=0)
    return result


def calculate_temporal_patterns(
    meg: np.ndarray,
    spatial_filters_weight: np.ndarray,
    temporal_filters: np.ndarray,
    sub: int,
    spatial_filters_bias: np.ndarray | None = None,
    *,
    return_covariances: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute temporal patterns from spatially filtered MEG.

    Args:
        meg: MEG data (channels, time).
        spatial_filters_weight: Spatial filters (subjects, branches, channels).
        temporal_filters: Temporal filters (branches, kernel).
        sub: Subject id (1-indexed). Converted to 0-indexed for model indexing.
        spatial_filters_bias: Optional bias (subjects, branches, 1).
        return_covariances: Also return Ledoit-Wolf covariances
            ``(branches, kernel, kernel)``.

    Returns:
        Temporal patterns (branches, kernel), or ``(patterns, covariances)``.
    """
    meg = np.asarray(meg, dtype=np.float64)
    _assert_finite('meg_before_temporal_patterns', meg)
    _assert_finite('spatial_filters_weight', spatial_filters_weight)
    _assert_finite('spatial_filters_bias', spatial_filters_bias)
    _assert_finite('temporal_filters', temporal_filters)

    projected = np.dot(spatial_filters_weight[sub - 1], meg)
    _assert_finite('spatially_filtered_meg', projected)
    if spatial_filters_bias is not None:
        projected = projected + np.asarray(
            spatial_filters_bias[sub - 1],
            dtype=np.float64,
        )

    filter_len = temporal_filters.shape[1]
    if projected.shape[1] < filter_len:
        raise ValueError(
            'Not enough temporal segments to form covariance: '
            f'signal_len={projected.shape[1]}, filter_len={filter_len}'
        )

    patterns = []
    covariances = []
    for branch, signal in enumerate(projected):
        lagged = np.lib.stride_tricks.sliding_window_view(signal, filter_len)
        lagged = lagged[:, ::-1].copy()
        lagged -= lagged.mean(axis=0, keepdims=True)
        if lagged.shape[0] <= 1:
            raise ValueError(
                'Not enough temporal segments to form covariance: '
                f'V_m shape={lagged.shape}, filter_len={filter_len}, '
                f'signal_len={signal.shape[0]}'
            )

        covariance = LedoitWolf(store_precision=False).fit(lagged).covariance_
        pattern = covariance @ temporal_filters[branch][::-1]
        patterns.append(pattern[::-1])
        covariances.append(covariance)

    result = np.stack(patterns, axis=0)
    if return_covariances:
        return result, np.stack(covariances, axis=0)
    return result


# ---------------------------------------------------------------------------
# Item-table / similarity helpers
# ---------------------------------------------------------------------------


def load_sensor_xy(dirprocess: str | Path | None = None) -> np.ndarray:
    filename = 'coords208_xy_scaled.npy'
    candidates = []
    if dirprocess is not None:
        candidates.append(Path(dirprocess) / 'coords' / filename)
    default_path = Path(PREPROCESSED_DATA_DIR) / 'coords' / filename
    if default_path not in candidates:
        candidates.append(default_path)

    for path in candidates:
        if path.exists():
            sensor_xy = np.asarray(np.load(path), dtype=np.float64)
            _assert_finite('sensor_xy', sensor_xy)
            if sensor_xy.ndim != 2 or sensor_xy.shape[1] != 2:
                raise ValueError(
                    f'sensor_xy must have shape (n_channels, 2), got {sensor_xy.shape}'
                )
            return sensor_xy

    raise FileNotFoundError(
        'Could not find coords208_xy_scaled.npy in any known coords directory. '
        f'Tried: {[str(path) for path in candidates]}'
    )


def build_sensor_neighbor_index(
    sensor_xy: np.ndarray,
    n_neighbors: int = DEFAULT_SPATIAL_ROUGHNESS_NEIGHBORS,
) -> np.ndarray:
    sensor_xy = np.asarray(sensor_xy, dtype=np.float64)
    if sensor_xy.ndim != 2 or sensor_xy.shape[1] != 2:
        raise ValueError(f'sensor_xy must have shape (n_channels, 2), got {sensor_xy.shape}')
    if sensor_xy.shape[0] < 2:
        raise ValueError('sensor_xy must contain at least two channels')

    n_neighbors = int(min(max(1, n_neighbors), sensor_xy.shape[0] - 1))
    squared_distances = np.sum(
        (sensor_xy[:, None, :] - sensor_xy[None, :, :]) ** 2,
        axis=2,
    )
    order = np.argsort(squared_distances, axis=1)
    return order[:, 1 : n_neighbors + 1].astype(int)


def compute_item_spatial_roughness(
    spatial_patterns: np.ndarray,
    sensor_xy: np.ndarray,
) -> np.ndarray:
    sensor_xy = np.asarray(sensor_xy, dtype=np.float64)
    spatial_patterns = np.asarray(spatial_patterns, dtype=np.float64)
    if sensor_xy.shape[0] != spatial_patterns.shape[1]:
        raise ValueError(
            'sensor_xy channel count does not match spatial pattern channel count.'
        )
    _assert_finite('item_spatial_patterns_for_roughness', spatial_patterns)
    neighbor_index = build_sensor_neighbor_index(sensor_xy)
    local_mse = np.mean(
        (spatial_patterns[:, :, np.newaxis] - spatial_patterns[:, neighbor_index]) ** 2,
        axis=(1, 2),
    )
    roughness = local_mse / (np.var(spatial_patterns, axis=1) + EPS)
    _assert_finite('item_spatial_roughness', roughness)
    return roughness


def save_item_table_npz(
    items: dict[str, Any],
    outpath: str | Path,
) -> None:
    if items['temporal_patterns'] is None:
        raise ValueError(
            'Cannot save item stats NPZ when temporal_patterns are unavailable.'
        )

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outpath,
        item_subjects=np.asarray(items['subjects'], dtype=int),
        item_branches=np.asarray(items['branches'], dtype=int) + 1,
        item_spatial_patterns=np.asarray(items['spatial_patterns'], dtype=np.float64),
        item_source_magnitudes=np.asarray(items['source_magnitudes'], dtype=np.float64),
        item_temporal_patterns=np.asarray(items['temporal_patterns'], dtype=np.float64),
        item_temporal_spectra=np.asarray(
            items['temporal_spectra_mag'],
            dtype=np.float64,
        ),
        temporal_spectrum_freqs=np.asarray(
            items['temporal_spectrum_freqs'],
            dtype=np.float64,
        ),
        demean_temporal_filters=np.asarray(bool(items['demean_temporal_filters'])),
    )


def correlation_similarity(
    features: np.ndarray,
    feature_name: str,
    absolute: bool = True,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f'{feature_name} must be 2D, got shape={features.shape}')
    _assert_finite(feature_name, features)

    centered = features - features.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    bad_rows = np.where(norms[:, 0] <= EPS)[0]
    if bad_rows.size:
        raise ValueError(
            f'{feature_name} contains near-constant rows: {bad_rows[:5].tolist()}'
        )

    similarity = (centered / norms) @ (centered / norms).T
    if similarity.min() < -1.0 - 1e-10 or similarity.max() > 1.0 + 1e-10:
        raise ValueError(
            f'{feature_name} correlation similarity fell outside [-1, 1]. '
            f'Observed range: [{similarity.min()}, {similarity.max()}]'
        )
    similarity = np.clip(similarity, -1.0, 1.0)
    if absolute:
        similarity = np.abs(similarity)
    np.fill_diagonal(similarity, 1.0)
    _assert_finite(f'{feature_name}_similarity', similarity)
    return similarity


def normalize_source_magnitudes(source_data: np.ndarray) -> np.ndarray:
    source_data = np.asarray(source_data, dtype=np.float64)
    _assert_finite('source_data', source_data)
    source_magnitudes = np.abs(source_data)
    norms = np.linalg.norm(source_magnitudes, axis=0, keepdims=True)
    bad = np.where(norms[0] <= EPS)[0]
    if bad.size:
        raise ValueError(
            f'Near-zero source projection norms for branches: {bad.tolist()}'
        )
    normalized = source_magnitudes / norms
    _assert_finite('normalized_source_magnitudes', normalized)
    return normalized


def validate_source_space_compatible(
    reference_stc: Any,
    candidate_stc: Any,
    *,
    context: str,
    expected_subject: str = 'fsaverage',
) -> None:
    reference_subject = reference_stc.subject
    candidate_subject = candidate_stc.subject
    if reference_subject != expected_subject or candidate_subject != expected_subject:
        raise ValueError(
            f'{context}: source estimates must use {expected_subject!r}; '
            f'got reference={reference_subject!r}, candidate={candidate_subject!r}.'
        )

    reference_vertices = reference_stc.vertices
    candidate_vertices = candidate_stc.vertices
    if len(reference_vertices) != len(candidate_vertices):
        raise ValueError(
            f'{context}: source estimates have different hemisphere vertex lists.'
        )
    for hemi_idx, (reference_hemi, candidate_hemi) in enumerate(
        zip(reference_vertices, candidate_vertices)
    ):
        if not np.array_equal(
            np.asarray(reference_hemi, dtype=int),
            np.asarray(candidate_hemi, dtype=int),
        ):
            raise ValueError(
                f'{context}: source vertex ordering differs for hemi index {hemi_idx}.'
            )

    reference_data = np.asarray(reference_stc.data)
    candidate_data = np.asarray(candidate_stc.data)
    if reference_data.shape != candidate_data.shape:
        raise ValueError(
            f'{context}: source estimates have different data shapes: '
            f'reference={reference_data.shape}, candidate={candidate_data.shape}.'
        )


def remap_correlation_to_unit_interval(
    similarity_raw: np.ndarray | float,
    name: str,
) -> np.ndarray:
    similarity = (np.asarray(similarity_raw, dtype=np.float64) + 1.0) / 2.0
    if similarity.min() < -1e-10 or similarity.max() > 1.0 + 1e-10:
        raise ValueError(
            f'{name} fell outside [0, 1] after remapping from correlation. '
            f'Observed range: [{similarity.min()}, {similarity.max()}]'
        )
    similarity = np.clip(similarity, 0.0, 1.0)
    if similarity.ndim == 2:
        np.fill_diagonal(similarity, 1.0)
    _assert_finite(name, similarity)
    return similarity


def build_item_table(
    spatial_patterns_all: np.ndarray,
    temporal_patterns_all: np.ndarray,
    subjects: np.ndarray,
    target_fs: float,
    source_magnitudes_all: np.ndarray,
    *,
    demean_temporal_filters: bool,
) -> dict[str, Any]:
    n_subjects, n_branches, n_channels = spatial_patterns_all.shape
    _, _, temporal_len = temporal_patterns_all.shape

    item_subjects = np.repeat(subjects, n_branches).astype(int)
    item_branches = np.tile(np.arange(n_branches, dtype=int), n_subjects)
    item_spatial_patterns = spatial_patterns_all.reshape(n_subjects * n_branches, n_channels)
    item_temporal_patterns = temporal_patterns_all.reshape(
        n_subjects * n_branches, temporal_len
    )

    spectra_log = []
    spectra_mag = []
    freqs_selected = None
    for pattern in item_temporal_patterns:
        freqs, magnitude_raw = fft_magnitude(
            pattern.astype(np.float64),
            fs=target_fs,
            square_fft=False,
            n_fft=4096,
            normalize=False,
        )
        valid = freqs >= 0.0
        freqs = freqs[valid]
        magnitude_raw = magnitude_raw[valid]
        _assert_finite('temporal_spectrum_magnitude_raw', magnitude_raw)
        spectra_log.append(np.log(magnitude_raw + EPS))
        spectra_mag.append(magnitude_raw)
        if freqs_selected is None:
            freqs_selected = freqs
        elif not np.allclose(freqs_selected, freqs):
            raise ValueError('Temporal spectrum frequencies are inconsistent across items')

    items = {
        'subjects': item_subjects,
        'branches': item_branches,
        'spatial_patterns': item_spatial_patterns,
        'temporal_patterns': item_temporal_patterns,
        'temporal_pattern_times_ms': (
            np.arange(item_temporal_patterns.shape[1], dtype=np.float64)
            / target_fs
            * 1000.0
        ),
        'temporal_spectra_log': np.stack(spectra_log, axis=0),
        'temporal_spectra_mag': np.stack(spectra_mag, axis=0),
        'temporal_spectrum_freqs': freqs_selected,
        'demean_temporal_filters': bool(demean_temporal_filters),
    }
    source_magnitudes_all = np.asarray(source_magnitudes_all, dtype=np.float64)
    expected_shape = (n_subjects, n_branches)
    if source_magnitudes_all.ndim != 3 or source_magnitudes_all.shape[:2] != expected_shape:
        raise ValueError(
            'source_magnitudes_all must have shape '
            f'({n_subjects}, {n_branches}, n_sources), got {source_magnitudes_all.shape}'
        )
    items['source_magnitudes'] = source_magnitudes_all.reshape(
        n_subjects * n_branches,
        source_magnitudes_all.shape[2],
    )
    _assert_finite('item_source_magnitudes', items['source_magnitudes'])
    return items


def get_cluster_output_group(cluster_name: str) -> str:
    cluster_name = str(cluster_name).strip()
    if re.fullmatch(r'[Cc]\d+', cluster_name):
        return 'main'
    if re.fullmatch(r'[Rr]\d+', cluster_name):
        return 'rest'
    raise ValueError(
        'Curated cluster names must be of the form C<number> or R<number>. '
        f'Got {cluster_name!r}.'
    )


def validate_curated_assignments(
    assignments: pd.DataFrame,
    items: dict[str, Any],
) -> pd.DataFrame:
    required_cols = {'item_index', 'subject', 'branch', 'cluster_name'}
    missing_cols = required_cols - set(assignments.columns)
    if missing_cols:
        raise ValueError(
            f'Assignments CSV is missing required columns: {sorted(missing_cols)}'
        )

    assignments = assignments.copy()
    assignments['item_index'] = assignments['item_index'].astype(int)
    assignments['cluster_name'] = assignments['cluster_name'].astype(str)
    for cluster_name in assignments['cluster_name'].drop_duplicates():
        get_cluster_output_group(cluster_name)
    cluster_order = assignments['cluster_name'].drop_duplicates().astype(str).tolist()
    assignments = assignments.sort_values('item_index').reset_index(drop=True)
    assignments.attrs['cluster_order'] = cluster_order

    if assignments['item_index'].duplicated().any():
        duplicates = assignments.loc[
            assignments['item_index'].duplicated(), 'item_index'
        ].tolist()[:5]
        raise ValueError(f'Assignments contain duplicated item_index values: {duplicates}')

    n_items = items['spatial_patterns'].shape[0]
    expected = np.arange(n_items, dtype=int)
    observed = assignments['item_index'].to_numpy(dtype=int)
    if observed.shape[0] != n_items or not np.array_equal(observed, expected):
        raise ValueError(
            'Assignments must contain each item_index exactly once so the curated split '
            'covers the full computed item table.'
        )

    if not np.array_equal(assignments['subject'].to_numpy(dtype=int), items['subjects']):
        raise ValueError(
            'Assignments subject column does not match the computed subject ordering.'
        )
    expected_branches = items['branches'] + 1
    if not np.array_equal(assignments['branch'].to_numpy(dtype=int), expected_branches):
        raise ValueError(
            'Assignments branch column does not match the computed branch ordering.'
        )
    return assignments


def compute_plot_compatible_similarities(
    items: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if 'source_magnitudes' not in items:
        raise ValueError(
            "Source-forward branch clustering requires items['source_magnitudes']."
        )

    source_similarity_raw = correlation_similarity(
        items['source_magnitudes'],
        feature_name='item_source_magnitudes',
        absolute=False,
    )
    source_similarity = remap_correlation_to_unit_interval(
        source_similarity_raw,
        name='source_similarity',
    )
    temporal_similarity_raw = correlation_similarity(
        items['temporal_spectra_log'],
        feature_name='item_temporal_spectra_log',
        absolute=False,
    )
    temporal_similarity = remap_correlation_to_unit_interval(
        temporal_similarity_raw,
        name='temporal_similarity',
    )
    return source_similarity, temporal_similarity


# ---------------------------------------------------------------------------
# Extraction orchestration
# ---------------------------------------------------------------------------


def extract_item_table_from_model(
    model,
    hyper_params: dict[str, Any],
    subjects: list[int],
    ses: int,
    story_id: int,
    offset_gap: float,
    meg_format: str,
    data_root: str | Path,
    raw_meg: bool,
    preprocessed_meg_path: str | Path,
    preprocess: bool,
    demean_temporal_filters: bool,
) -> tuple[dict[str, Any], int]:
    """Extract combined-clustering items while reusing invariant source geometry."""
    from lisa.plots.source_estimation import get_stc, prepare_source_geometry

    if not subjects:
        raise ValueError('At least one subject is required')

    filters = extract_run_filters(
        model,
        hyper_params,
        demean_temporal_filters_enabled=demean_temporal_filters,
    )
    target_fs = float(hyper_params['meg_sr'])
    n_branches = int(filters['K'])
    spatial_weight = filters['spatial_filters_weight']
    spatial_bias = filters['spatial_filters_bias']
    temporal_filters = filters['temporal_filters']
    filtfilt = filters['filtfilt']

    phase_seconds = {
        'source_setup': 0.0,
        'load': 0.0,
        'spatial': 0.0,
        'temporal': 0.0,
        'source': 0.0,
    }
    extraction_started = perf_counter()
    started = perf_counter()
    source_geometry = prepare_source_geometry()
    phase_seconds['source_setup'] = perf_counter() - started

    spatial_patterns_all = []
    source_magnitudes_all = []
    temporal_patterns_all = []
    reference_raw = None
    reference_stc = None

    progress = tqdm(subjects, desc='Extracting subject-branch patterns')
    for sub in progress:
        started = perf_counter()
        meg, raw = load_subject_meg_and_raw(
            sub=sub,
            ses=ses,
            story_id=story_id,
            target_fs=target_fs,
            offset_gap=offset_gap,
            meg_format=meg_format,
            data_root=data_root,
            raw_meg=raw_meg,
            preprocessed_meg_path=preprocessed_meg_path,
            preprocess=preprocess,
            raw_metadata_only=not raw_meg,
        )
        phase_seconds['load'] += perf_counter() - started
        if reference_raw is None:
            reference_raw = raw
        elif raw.ch_names != reference_raw.ch_names:
            raise ValueError(
                f'Channel mismatch for subject {sub}; all subjects must share the '
                'same MEG channel order.'
            )

        started = perf_counter()
        spatial_patterns = calculate_spatial_patterns(
            meg=meg,
            spatial_filters_weight=spatial_weight,
            temporal_filters=temporal_filters,
            sub=sub,
            filtfilt=filtfilt,
        )
        phase_seconds['spatial'] += perf_counter() - started

        started = perf_counter()
        temporal_patterns = calculate_temporal_patterns(
            meg=meg,
            spatial_filters_weight=spatial_weight,
            temporal_filters=temporal_filters,
            sub=sub,
            spatial_filters_bias=spatial_bias,
        )
        phase_seconds['temporal'] += perf_counter() - started
        _assert_finite(f'spatial_patterns_subject_{sub}', spatial_patterns)
        _assert_finite(f'temporal_patterns_subject_{sub}', temporal_patterns)

        started = perf_counter()
        stc = get_stc(
            raw=raw,
            spatial_patterns=spatial_patterns,
            prepared_geometry=source_geometry,
        )
        phase_seconds['source'] += perf_counter() - started
        if reference_stc is None:
            reference_stc = stc.copy()
        validate_source_space_compatible(
            reference_stc,
            stc,
            context=f'subject {sub} source estimate',
        )

        spatial_patterns_all.append(spatial_patterns)
        source_magnitudes_all.append(normalize_source_magnitudes(stc.data).T)
        temporal_patterns_all.append(temporal_patterns)
        progress.set_postfix(
            spatial=f'{phase_seconds["spatial"]:.0f}s',
            source=f'{phase_seconds["source"]:.0f}s',
            temporal=f'{phase_seconds["temporal"]:.0f}s',
        )

    items = build_item_table(
        spatial_patterns_all=np.stack(spatial_patterns_all, axis=0),
        temporal_patterns_all=np.stack(temporal_patterns_all, axis=0),
        subjects=np.asarray(subjects, dtype=int),
        target_fs=target_fs,
        source_magnitudes_all=np.stack(source_magnitudes_all, axis=0),
        demean_temporal_filters=demean_temporal_filters,
    )
    items['sensor_xy'] = load_sensor_xy(hyper_params['dirprocess'])
    if reference_raw is None:
        raise ValueError('No subjects were loaded')
    if len(reference_raw.ch_names) != items['spatial_patterns'].shape[1]:
        raise ValueError(
            'Channel count mismatch between computed spatial patterns and the Raw '
            'info used for subject loading.'
        )

    phase_seconds['total'] = perf_counter() - extraction_started
    timing = ', '.join(
        f'{name}={seconds:.1f}s' for name, seconds in phase_seconds.items()
    )
    items['extraction_timing_seconds'] = {
        name: float(seconds) for name, seconds in phase_seconds.items()
    }
    print(f'Extraction timing: {timing}')
    return items, n_branches


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def cluster_by_minimum_similarity(
    source_similarity: np.ndarray,
    temporal_similarity: np.ndarray,
    item_indices: np.ndarray,
    minimum_similarity_unit: float,
) -> tuple[dict[str, list[int]], np.ndarray]:
    """Run one complete-linkage clustering on unit-interval similarities."""
    source_similarity = np.asarray(source_similarity, dtype=np.float64)
    temporal_similarity = np.asarray(temporal_similarity, dtype=np.float64)
    item_indices = np.asarray(item_indices, dtype=int)
    if source_similarity.shape != temporal_similarity.shape:
        raise ValueError(
            'Source and temporal similarity matrices must have equal shapes.'
        )
    if (
        source_similarity.ndim != 2
        or source_similarity.shape[0] != source_similarity.shape[1]
    ):
        raise ValueError('Similarity matrices must be square.')
    if item_indices.ndim != 1:
        raise ValueError('item_indices must be one-dimensional.')
    if not 0.0 <= minimum_similarity_unit <= 1.0:
        raise ValueError('minimum_similarity_unit must be in [0, 1].')

    joint_similarity = np.minimum(source_similarity, temporal_similarity)
    _assert_finite('joint_minimum_similarity', joint_similarity)
    if item_indices.size == 0:
        return {}, joint_similarity
    if item_indices.min() < 0 or item_indices.max() >= joint_similarity.shape[0]:
        raise ValueError('item_indices contain values outside the similarity matrices.')

    if item_indices.size == 1:
        labels = np.zeros(1, dtype=int)
    else:
        distance = 1.0 - joint_similarity[np.ix_(item_indices, item_indices)]
        np.fill_diagonal(distance, 0.0)
        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0 - float(minimum_similarity_unit),
            metric='precomputed',
            linkage='complete',
        ).fit_predict(distance)

    label_to_items: dict[int, list[int]] = {}
    for item, label in zip(item_indices, labels):
        label_to_items.setdefault(int(label), []).append(int(item))
    ordered = sorted(
        label_to_items.values(),
        key=lambda cluster_items: (-len(cluster_items), min(cluster_items)),
    )
    return {
        f'C{cluster_number}': sorted(cluster_items)
        for cluster_number, cluster_items in enumerate(ordered, start=1)
    }, joint_similarity


def _pair_stats(similarity: np.ndarray, item_indices: np.ndarray) -> dict[str, float]:
    item_indices = np.asarray(item_indices, dtype=int)
    if item_indices.size < 2:
        return {
            statistic: np.nan for statistic in ('min', 'q10', 'median', 'q90', 'max')
        }
    submatrix = similarity[np.ix_(item_indices, item_indices)]
    values = submatrix[np.triu_indices(item_indices.size, k=1)]
    return {
        'min': float(np.min(values)),
        'q10': float(np.percentile(values, 10.0)),
        'median': float(np.median(values)),
        'q90': float(np.percentile(values, 90.0)),
        'max': float(np.max(values)),
    }


def _raw_correlation_stats(
    similarity: np.ndarray,
    item_indices: np.ndarray,
) -> dict[str, float]:
    return {
        name: 2.0 * value - 1.0 if np.isfinite(value) else np.nan
        for name, value in _pair_stats(similarity, item_indices).items()
    }


def build_clustering_outputs(
    items: dict[str, Any],
    *,
    minimum_similarity_raw: float,
    spatial_roughness_threshold: float,
    exclude_rough: bool,
) -> dict[str, Any]:
    """Build plotting assignments and cluster-level diagnostics."""
    if (
        not np.isfinite(spatial_roughness_threshold)
        or spatial_roughness_threshold <= 0.0
    ):
        raise ValueError('spatial_roughness_threshold must be positive.')

    minimum_similarity_raw = float(minimum_similarity_raw)
    if not -1.0 <= minimum_similarity_raw <= 1.0:
        raise ValueError(
            f'minimum_similarity must be in [-1, 1], got {minimum_similarity_raw}.'
        )

    source_similarity, temporal_similarity = compute_plot_compatible_similarities(items)
    minimum_similarity = float(
        remap_correlation_to_unit_interval(minimum_similarity_raw, 'minimum_similarity')
    )
    roughness = compute_item_spatial_roughness(
        items['spatial_patterns'],
        items['sensor_xy'],
    )
    rough_mask = roughness > spatial_roughness_threshold
    clustered_mask = (
        ~rough_mask if exclude_rough else np.ones_like(rough_mask, dtype=bool)
    )
    clustered_indices = np.flatnonzero(clustered_mask)

    clusters, joint_similarity = cluster_by_minimum_similarity(
        source_similarity,
        temporal_similarity,
        clustered_indices,
        minimum_similarity,
    )
    excluded_indices = np.flatnonzero(~clustered_mask)
    partition = dict(clusters)
    if excluded_indices.size:
        partition['R1'] = excluded_indices.tolist()

    rows = []
    for cluster_name, cluster_items in partition.items():
        is_clustered = cluster_name.startswith('C')
        for item in cluster_items:
            rows.append(
                {
                    'item_index': int(item),
                    'subject': int(items['subjects'][item]),
                    'branch': int(items['branches'][item]) + 1,
                    'cluster_name': cluster_name,
                    'is_main': is_clustered,
                    'is_clustered': is_clustered,
                    'spatial_roughness': float(roughness[item]),
                    'is_rough': bool(rough_mask[item]),
                }
            )
    assignments = validate_curated_assignments(pd.DataFrame(rows), items)

    n_items = len(items['subjects'])
    n_clustered = int(clustered_indices.size)
    n_branches = int(np.max(items['branches'])) + 1
    summary_rows = []
    for cluster_name, cluster_items in partition.items():
        item_indices = np.asarray(cluster_items, dtype=int)
        source_stats = _raw_correlation_stats(source_similarity, item_indices)
        temporal_stats = _raw_correlation_stats(temporal_similarity, item_indices)
        joint_stats = _raw_correlation_stats(joint_similarity, item_indices)
        rough_values = roughness[item_indices]
        branch_counts = np.bincount(
            items['branches'][item_indices], minlength=n_branches
        )
        top_branches = ', '.join(
            f'B{branch + 1}:{int(branch_counts[branch])}'
            for branch in np.argsort(branch_counts)[::-1]
            if branch_counts[branch]
        )
        is_clustered = cluster_name.startswith('C')
        summary_rows.append(
            {
                'cluster_name': cluster_name,
                'is_clustered': is_clustered,
                'n': int(item_indices.size),
                'coverage_all_pct': 100.0 * item_indices.size / n_items,
                'coverage_clustered_pct': (
                    100.0 * item_indices.size / n_clustered
                    if is_clustered and n_clustered
                    else np.nan
                ),
                'unique_subject_count': int(
                    np.unique(items['subjects'][item_indices]).size
                ),
                'source_similarity_min': source_stats['min'],
                'source_similarity_q10': source_stats['q10'],
                'source_similarity_median': source_stats['median'],
                'source_similarity_q90': source_stats['q90'],
                'source_similarity_max': source_stats['max'],
                'temporal_similarity_min': temporal_stats['min'],
                'temporal_similarity_q10': temporal_stats['q10'],
                'temporal_similarity_median': temporal_stats['median'],
                'temporal_similarity_q90': temporal_stats['q90'],
                'temporal_similarity_max': temporal_stats['max'],
                'joint_similarity_min': joint_stats['min'],
                'joint_similarity_q10': joint_stats['q10'],
                'joint_similarity_median': joint_stats['median'],
                'joint_similarity_q90': joint_stats['q90'],
                'joint_similarity_max': joint_stats['max'],
                'maximum_joint_clustering_distance': (
                    (1.0 - joint_stats['min']) / 2.0
                    if np.isfinite(joint_stats['min'])
                    else 0.0
                ),
                'meets_minimum_similarity': (
                    bool(
                        item_indices.size < 2
                        or joint_stats['min'] >= minimum_similarity_raw - 1e-12
                    )
                    if is_clustered
                    else False
                ),
                'spatial_roughness_mean': float(np.mean(rough_values)),
                'spatial_roughness_median': float(np.median(rough_values)),
                'spatial_roughness_q90': float(np.percentile(rough_values, 90.0)),
                'spatial_roughness_max': float(np.max(rough_values)),
                'rough_item_count': int(np.sum(rough_mask[item_indices])),
                'rough_item_fraction': float(np.mean(rough_mask[item_indices])),
                'top_branches': top_branches,
            }
        )

    return {
        'assignments': assignments,
        'summary': pd.DataFrame(summary_rows),
        'similarities': {
            'source_correlation': 2.0 * source_similarity - 1.0,
            'temporal_correlation': 2.0 * temporal_similarity - 1.0,
            'joint_minimum_correlation': 2.0 * joint_similarity - 1.0,
            'joint_clustering_distance': 1.0 - joint_similarity,
            'spatial_roughness': roughness,
            'clustered_mask': clustered_mask,
        },
        'meta': {
            'algorithm': 'single_complete_linkage_threshold',
            'fusion': 'minimum',
            'linkage': 'complete',
            'minimum_similarity': float(minimum_similarity_raw),
            'similarity_scale': 'raw_pearson_correlation',
            'source_similarity': (
                'pearson_correlation_l2_normalized_source_magnitudes'
            ),
            'temporal_similarity': 'pearson_correlation_log_fft_magnitude',
            'exclude_rough': bool(exclude_rough),
            'spatial_roughness_threshold': float(spatial_roughness_threshold),
            'n_items': int(n_items),
            'n_clustered_items': n_clustered,
            'n_rough_items': int(np.sum(rough_mask)),
            'n_excluded_items': int(excluded_indices.size),
            'n_clusters': int(len(clusters)),
        },
    }
