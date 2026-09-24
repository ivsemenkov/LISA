"""Plot SVDs of learned spatial filters and covariance-derived patterns."""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from lisa.plots.plot_style import FULL_WIDTH_IN, paper_topomap, publication_style
from lisa.utils.constants import (
    CLEAN_DATA_DIR,
    DATA_ROOT,
    EXPERIMENTS_DIR,
    N_SUBJECTS,
    OUTPUTS_DIR,
    PLOTS_DIR,
    PREPROCESSED_DATA_DIR,
)
from lisa.utils.numeric import assert_finite as _assert_finite
from lisa.utils.validators import validate_run_group


EPS = 1e-12
SVD_NORMALIZATIONS = ('raw', 'row_l2', 'both')
TEMPORAL_FILTER_SUFFIX = {
    True: '_dtf',
    False: '_raw_tf',
}


@dataclass(frozen=True)
class TopographyStack:
    spatial_filters: np.ndarray
    spatial_patterns: np.ndarray
    item_subjects: np.ndarray
    item_branches: np.ndarray
    info: object
    n_branches: int
    demeaned_temporal_spatial_patterns: np.ndarray | None = None


@dataclass(frozen=True)
class SVDResult:
    singular_values: np.ndarray
    right_singular_vectors: np.ndarray
    explained_energy: np.ndarray
    cumulative_energy: np.ndarray
    n_components: int


@dataclass(frozen=True)
class SVDVariantResult:
    key: str
    spatial_filters: np.ndarray
    spatial_patterns: np.ndarray
    filter_result: SVDResult
    pattern_result: SVDResult
    filter_row_norms: np.ndarray | None = None
    pattern_row_norms: np.ndarray | None = None


@dataclass(frozen=True)
class TangentialGainResult:
    gain_2d: np.ndarray
    gain_2d_unnormalized: np.ndarray
    valid_site_indices: np.ndarray
    invalid_site_indices: np.ndarray
    invalid_tangential_norms: np.ndarray
    n_source_sites: int


@dataclass(frozen=True)
class RapMusicResult:
    values: list[float]
    indices: list[int]
    n_source_sites: int
    initial_invalid_site_indices: np.ndarray
    initial_invalid_tangential_norms: np.ndarray
    union_invalid_site_indices: np.ndarray
    iteration_invalid_counts: np.ndarray
    iteration_valid_counts: np.ndarray


def validate_energy_threshold(energy_threshold: float) -> None:
    if not 0.0 < energy_threshold <= 1.0:
        raise ValueError(f'energy_threshold must be in (0, 1], got {energy_threshold}.')


def validate_svd_normalization(normalization: str) -> None:
    if normalization not in SVD_NORMALIZATIONS:
        raise ValueError(
            "svd_normalization must be 'raw', 'row_l2', or 'both', "
            f'got {normalization!r}.'
        )


def row_l2_normalize(topographies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    topographies = np.asarray(topographies, dtype=np.float64)
    _assert_finite('topographies_before_row_l2_normalization', topographies)
    norms = np.linalg.norm(topographies, axis=1, keepdims=True)
    bad = np.where(norms[:, 0] == 0.0)[0]
    if bad.size:
        raise ValueError(
            'Cannot row-L2-normalize zero topographies at item indices: '
            f'{bad.tolist()}'
        )
    normalized = topographies / norms
    _assert_finite('row_l2_normalized_topographies', normalized)
    return normalized, norms[:, 0]


def compute_svd_result(
    topographies: np.ndarray,
    energy_threshold: float,
) -> SVDResult:
    validate_energy_threshold(energy_threshold)
    topographies = np.asarray(topographies, dtype=np.float64)
    if topographies.ndim != 2:
        raise ValueError(f'Expected topographies to be 2D, got {topographies.shape}.')
    _assert_finite('svd_input_topographies', topographies)

    _, singular_values, right_singular_vectors = np.linalg.svd(
        topographies,
        full_matrices=False,
    )
    max_singular_value = (
        float(singular_values[0]) if singular_values.size else 0.0
    )
    if max_singular_value == 0.0:
        raise ValueError('Cannot compute explained energy for a zero matrix.')

    relative_energies = (singular_values / max_singular_value) ** 2
    explained_energy = relative_energies / np.sum(relative_energies)
    cumulative_energy = np.cumsum(explained_energy)
    n_components = int(np.searchsorted(cumulative_energy, energy_threshold) + 1)

    # Fix the arbitrary component sign for deterministic topomap display.
    right_singular_vectors = right_singular_vectors.copy()
    for component_idx in range(right_singular_vectors.shape[0]):
        component = right_singular_vectors[component_idx]
        peak_idx = int(np.argmax(np.abs(component)))
        if component[peak_idx] < 0.0:
            right_singular_vectors[component_idx] *= -1.0

    return SVDResult(
        singular_values=singular_values,
        right_singular_vectors=right_singular_vectors,
        explained_energy=explained_energy,
        cumulative_energy=cumulative_energy,
        n_components=n_components,
    )


def compute_svd_variants(
    stack: TopographyStack,
    energy_threshold: float,
    normalization: str = 'both',
) -> tuple[SVDVariantResult, ...]:
    validate_svd_normalization(normalization)
    include_raw = normalization in ('raw', 'both')
    include_row_l2 = normalization in ('row_l2', 'both')

    variants = []
    if include_raw:
        variants.append(
            SVDVariantResult(
                key='raw',
                spatial_filters=stack.spatial_filters,
                spatial_patterns=stack.spatial_patterns,
                filter_result=compute_svd_result(
                    topographies=stack.spatial_filters,
                    energy_threshold=energy_threshold,
                ),
                pattern_result=compute_svd_result(
                    topographies=stack.spatial_patterns,
                    energy_threshold=energy_threshold,
                ),
            )
        )

    if include_row_l2:
        normalized_filters, filter_row_norms = row_l2_normalize(stack.spatial_filters)
        normalized_patterns, pattern_row_norms = row_l2_normalize(stack.spatial_patterns)
        variants.append(
            SVDVariantResult(
                key='row_l2',
                spatial_filters=normalized_filters,
                spatial_patterns=normalized_patterns,
                filter_result=compute_svd_result(
                    topographies=normalized_filters,
                    energy_threshold=energy_threshold,
                ),
                pattern_result=compute_svd_result(
                    topographies=normalized_patterns,
                    energy_threshold=energy_threshold,
                ),
                filter_row_norms=filter_row_norms,
                pattern_row_norms=pattern_row_norms,
            )
        )

    if stack.demeaned_temporal_spatial_patterns is not None:
        demeaned_patterns = np.asarray(
            stack.demeaned_temporal_spatial_patterns,
            dtype=np.float64,
        )
        expected_shape = stack.spatial_patterns.shape
        if demeaned_patterns.shape != expected_shape:
            raise ValueError(
                'Demeaned-temporal spatial patterns must match original spatial '
                f'patterns shape {expected_shape}, got {demeaned_patterns.shape}.'
            )
        _assert_finite('demeaned_temporal_spatial_patterns', demeaned_patterns)
        if include_raw:
            variants.append(
                SVDVariantResult(
                    key='dtf_raw',
                    spatial_filters=stack.spatial_filters,
                    spatial_patterns=demeaned_patterns,
                    filter_result=compute_svd_result(
                        topographies=stack.spatial_filters,
                        energy_threshold=energy_threshold,
                    ),
                    pattern_result=compute_svd_result(
                        topographies=demeaned_patterns,
                        energy_threshold=energy_threshold,
                    ),
                )
            )
        if include_row_l2:
            normalized_demeaned_patterns, demeaned_pattern_row_norms = row_l2_normalize(
                demeaned_patterns
            )
            variants.append(
                SVDVariantResult(
                    key='dtf_row_l2',
                    spatial_filters=normalized_filters,
                    spatial_patterns=normalized_demeaned_patterns,
                    filter_result=compute_svd_result(
                        topographies=normalized_filters,
                        energy_threshold=energy_threshold,
                    ),
                    pattern_result=compute_svd_result(
                        topographies=normalized_demeaned_patterns,
                        energy_threshold=energy_threshold,
                    ),
                    filter_row_norms=filter_row_norms,
                    pattern_row_norms=demeaned_pattern_row_norms,
                )
            )

    return tuple(variants)


def collect_spatial_topographies(
    model,
    hyper_params: dict,
    subjects: Sequence[int],
    ses: int,
    story_id: int,
    offset_gap: float,
    meg_format: str,
    data_root: str | Path,
    preprocessed_meg_path: str | Path,
    demean_temporal_filters_enabled: bool,
) -> TopographyStack:
    from tqdm import tqdm

    from lisa.plots.branch_interpretation import (
        calculate_spatial_patterns,
        extract_run_filters,
        load_subject_meg_and_raw,
        physical_spatial_affine,
    )

    if not subjects:
        raise ValueError('At least one subject is required.')

    filters = extract_run_filters(
        model,
        hyper_params,
        demean_temporal_filters_enabled=demean_temporal_filters_enabled,
    )
    target_fs = float(hyper_params['meg_sr'])
    n_branches = int(filters['K'])
    spatial_weight = filters['spatial_filters_weight']
    temporal_filters = filters['temporal_filters']
    filtfilt = bool(filters['filtfilt'])

    bad_subjects = [
        int(subject)
        for subject in subjects
        if int(subject) < 1 or int(subject) > spatial_weight.shape[0]
    ]
    if bad_subjects:
        raise ValueError(
            f'Subject ids are outside the trained subject-layer range: {bad_subjects}'
        )

    spatial_filters_all = []
    spatial_patterns_all = []
    reference_info = None
    reference_ch_names = None

    for sub in tqdm(subjects, desc='Extracting subject-branch topographies'):
        meg, raw, scale, offset = load_subject_meg_and_raw(
            sub=int(sub),
            ses=ses,
            story_id=story_id,
            target_fs=target_fs,
            offset_gap=offset_gap,
            meg_format=meg_format,
            data_root=data_root,
            preprocessed_meg_path=preprocessed_meg_path,
            raw_metadata_only=True,
        )
        if reference_info is None:
            reference_info = raw.info.copy()
            reference_ch_names = list(raw.ch_names)
        elif list(raw.ch_names) != reference_ch_names:
            raise ValueError(
                f'Channel mismatch for subject {sub}; all subjects must share the '
                'same MEG channel order.'
            )

        weight, _bias = physical_spatial_affine(
            spatial_weight, None, scale, offset, int(sub)
        )
        spatial_patterns = calculate_spatial_patterns(
            meg=meg,
            spatial_filters_weight=weight,
            temporal_filters=temporal_filters,
            sub=int(sub),
            filtfilt=filtfilt,
            channel_pad=offset,
        )
        expected_shape = (n_branches, spatial_weight.shape[2])
        if spatial_patterns.shape != expected_shape:
            raise ValueError(
                f'Expected spatial patterns with shape {expected_shape} for '
                f'subject {sub}, got {spatial_patterns.shape}.'
            )
        _assert_finite(f'spatial_patterns_subject_{sub}', spatial_patterns)
        subject_filters = weight[int(sub) - 1]
        if subject_filters.shape != expected_shape:
            raise ValueError(
                f'Expected spatial filters with shape {expected_shape} for '
                f'subject {sub}, got {subject_filters.shape}.'
            )
        _assert_finite(f'spatial_filters_subject_{sub}', subject_filters)
        spatial_filters_all.append(subject_filters)
        spatial_patterns_all.append(spatial_patterns)

    if reference_info is None:
        raise ValueError('No subject topographies were extracted.')

    spatial_filters = np.concatenate(spatial_filters_all, axis=0)
    spatial_patterns = np.concatenate(spatial_patterns_all, axis=0)
    item_subjects = np.repeat(np.asarray(subjects, dtype=int), n_branches)
    item_branches = np.tile(
        np.arange(1, n_branches + 1, dtype=int),
        len(subjects),
    )
    _assert_finite('stacked_spatial_filters', spatial_filters)
    _assert_finite('stacked_spatial_patterns', spatial_patterns)

    return TopographyStack(
        spatial_filters=spatial_filters,
        spatial_patterns=spatial_patterns,
        item_subjects=item_subjects,
        item_branches=item_branches,
        info=reference_info,
        n_branches=n_branches,
    )


def exclude_rough_spatial_items(
    stack: TopographyStack,
    *,
    dirprocess: str | Path | None,
) -> tuple[TopographyStack, dict[str, object]]:
    """Drop the same rough participant×branch items used by clustering QC."""
    from lisa.plots.branch_interpretation import (
        DEFAULT_MAIN_SPATIAL_ROUGHNESS_MAX,
        compute_item_spatial_roughness,
        load_sensor_xy,
    )

    n_items = int(stack.spatial_patterns.shape[0])
    if (
        stack.spatial_filters.shape[0] != n_items
        or stack.item_subjects.shape[0] != n_items
        or stack.item_branches.shape[0] != n_items
    ):
        raise ValueError(
            'Participant×branch matrices must share the same item axis before '
            'roughness QC.'
        )
    demeaned = stack.demeaned_temporal_spatial_patterns
    if demeaned is not None and demeaned.shape[0] != n_items:
        raise ValueError(
            'Demeaned-temporal spatial patterns must share the same item axis '
            'before roughness QC.'
        )

    roughness = compute_item_spatial_roughness(
        stack.spatial_patterns,
        load_sensor_xy(dirprocess),
    )
    rough_mask = roughness > DEFAULT_MAIN_SPATIAL_ROUGHNESS_MAX
    keep_mask = ~rough_mask
    filtered = TopographyStack(
        spatial_filters=stack.spatial_filters[keep_mask],
        spatial_patterns=stack.spatial_patterns[keep_mask],
        item_subjects=stack.item_subjects[keep_mask],
        item_branches=stack.item_branches[keep_mask],
        info=stack.info,
        n_branches=stack.n_branches,
        demeaned_temporal_spatial_patterns=(
            None if demeaned is None else demeaned[keep_mask]
        ),
    )
    n_dropped = int(np.sum(rough_mask))
    n_retained = int(np.sum(keep_mask))
    subjects_before = np.unique(stack.item_subjects)
    subjects_after = np.unique(filtered.item_subjects)
    if filtered.item_subjects.size:
        retained_branch_counts = np.bincount(filtered.item_subjects)[subjects_after]
        min_retained_branches = int(np.min(retained_branch_counts))
        max_retained_branches = int(np.max(retained_branch_counts))
    else:
        min_retained_branches = 0
        max_retained_branches = 0
    qc = {
        'n_items': n_items,
        'n_dropped_items': n_dropped,
        'n_rough_items': n_dropped,
        'n_excluded_items': n_dropped,
        'n_retained_items': n_retained,
        'spatial_roughness_threshold': float(DEFAULT_MAIN_SPATIAL_ROUGHNESS_MAX),
        'exclude_rough': True,
        'n_subjects_before': int(subjects_before.size),
        'n_subjects_after': int(subjects_after.size),
        'min_retained_branches_per_participant': min_retained_branches,
        'max_retained_branches_per_participant': max_retained_branches,
    }
    return filtered, qc


def component_table(result: SVDResult, topography_type: str, svd_variant: str):
    import pandas as pd

    singular_vector_numbers = np.arange(
        1, result.singular_values.shape[0] + 1, dtype=int
    )
    return pd.DataFrame(
        {
            'svd_variant': svd_variant,
            'topography_type': topography_type,
            'singular_vector': singular_vector_numbers,
            'singular_value': result.singular_values,
            'explained_energy': result.explained_energy,
            'cumulative_explained_energy': result.cumulative_energy,
            'reaches_energy_threshold': singular_vector_numbers
            == result.n_components,
        }
    )


def topomap_symmetric_vlim(vectors: np.ndarray) -> tuple[float, float] | None:
    limit = float(np.max(np.abs(vectors))) if vectors.size else 0.0
    if limit <= EPS:
        return None
    return -limit, limit


@publication_style
def plot_singular_vectors(
    filter_result: SVDResult,
    pattern_result: SVDResult,
    info: object,
    n_display: int = 10,
):
    import matplotlib.pyplot as plt

    if n_display < 1:
        raise ValueError(f'n_display must be positive, got {n_display}.')
    n_available = min(
        filter_result.right_singular_vectors.shape[0],
        pattern_result.right_singular_vectors.shape[0],
    )
    n_display = min(n_display, n_available)
    n_map_columns = min(5, n_display)
    n_groups = int(math.ceil(n_display / n_map_columns))
    n_grid_rows = 3 * n_groups - 1
    height_ratios = [0.95, 0.95]
    for _ in range(1, n_groups):
        height_ratios.extend([0.18, 0.95, 0.95])

    fig = plt.figure(
        figsize=(FULL_WIDTH_IN, 0.65 + 2.25 * n_groups),
        layout='constrained',
    )
    grid = fig.add_gridspec(
        n_grid_rows,
        n_map_columns + 1,
        width_ratios=[1.15] + [1.0] * n_map_columns,
        height_ratios=height_ratios,
    )
    results = (
        ('Filters', filter_result),
        ('Topographies', pattern_result),
    )
    row_vlims = {
        row_label: topomap_symmetric_vlim(result.right_singular_vectors[:n_display])
        for row_label, result in results
    }
    for group_idx in range(n_groups):
        first_idx = group_idx * n_map_columns
        filter_row = group_idx * 3
        for result_row, (row_label, result) in enumerate(results):
            grid_row = filter_row + result_row
            label_ax = fig.add_subplot(grid[grid_row, 0])
            label_ax.axis('off')
            label_ax.text(
                1.0,
                0.5,
                row_label,
                ha='right',
                va='center',
                fontweight='bold',
            )

            for column_idx in range(n_map_columns):
                singular_vector_idx = first_idx + column_idx
                ax = fig.add_subplot(grid[grid_row, column_idx + 1])
                if singular_vector_idx >= n_display:
                    ax.axis('off')
                    continue
                topomap_kwargs = {
                    'cmap': 'RdBu_r',
                }
                if row_vlims[row_label] is not None:
                    topomap_kwargs['vlim'] = row_vlims[row_label]
                paper_topomap(
                    result.right_singular_vectors[singular_vector_idx],
                    info,
                    kind='topography',
                    axes=ax,
                    show=False,
                    **topomap_kwargs,
                )
                if result_row == 0:
                    ax.set_title(str(singular_vector_idx + 1), pad=2)

    return fig


@publication_style
def plot_explained_energy(
    filter_result: SVDResult,
    pattern_result: SVDResult,
    energy_threshold: float,
    n_display: int = 10,
):
    import matplotlib.pyplot as plt

    if n_display < 1:
        raise ValueError(f'n_display must be positive, got {n_display}.')

    results = (
        ('Filters', filter_result),
        ('Topographies', pattern_result),
    )
    n_available = min(result.cumulative_energy.shape[0] for _, result in results)
    display_marker = min(n_display, n_available)
    x_limit = min(
        n_available,
        max(display_marker, *(result.n_components for _, result in results)) + 2,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FULL_WIDTH_IN, 3.2),
        sharex=True,
        sharey=True,
        layout='constrained',
    )
    threshold_percent = energy_threshold * 100.0
    tick_values = sorted(
        {
            1,
            display_marker,
            x_limit,
            *(result.n_components for _, result in results),
        }
    )
    tick_values = [tick for tick in tick_values if tick <= x_limit]
    for ax, (title, result) in zip(axes, results, strict=True):
        x = np.arange(1, result.cumulative_energy.shape[0] + 1, dtype=int)
        cumulative_percent = result.cumulative_energy * 100.0
        ax.plot(x, cumulative_percent, color='C0', label='Cumulative energy')
        ax.axhline(
            threshold_percent,
            color='black',
            linestyle='--',
            linewidth=1.0,
            label=f'{energy_threshold:.0%} threshold',
        )
        ax.axvline(
            display_marker,
            color='C0',
            linestyle=':',
            linewidth=1.1,
            label=f'{display_marker} components',
        )

        display_energy = cumulative_percent[display_marker - 1]
        threshold_energy = cumulative_percent[result.n_components - 1]
        ax.scatter(display_marker, display_energy, color='C0', s=18, zorder=3)
        ax.axvline(
            result.n_components,
            color='C3',
            linestyle=':',
            linewidth=1.1,
            label=f'{energy_threshold:.0%} components',
        )
        ax.scatter(result.n_components, threshold_energy, color='C3', s=18, zorder=3)
        ax.set_title(title)
        ax.set_xlim(1, x_limit)
        ax.set_ylim(0.0, 102.0)
        ax.set_xticks(tick_values)

    axes[0].legend(loc='lower right', frameon=False)
    fig.supxlabel('Number of components')
    fig.supylabel('Cumulative explained energy (%)')
    return fig


@publication_style
def plot_singular_value_stems(
    filter_result: SVDResult,
    pattern_result: SVDResult,
    n_display: int = 50,
):
    import matplotlib.pyplot as plt

    if n_display < 1:
        raise ValueError(f'n_display must be positive, got {n_display}.')

    results = (
        ('Filters', filter_result),
        ('Topographies', pattern_result),
    )
    n_available = min(result.singular_values.shape[0] for _, result in results)
    n_display = min(n_display, n_available)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FULL_WIDTH_IN, 3.2),
        sharex=True,
        layout='constrained',
    )
    for ax, (title, result) in zip(axes, results, strict=True):
        x = np.arange(1, n_display + 1, dtype=int)
        markerline, stemlines, baseline = ax.stem(
            x,
            result.singular_values[:n_display],
            basefmt=' ',
        )
        markerline.set_markersize(3.5)
        stemlines.set_linewidth(1.0)
        baseline.set_linewidth(0.0)
        ax.set_title(title)
        ax.set_xlim(0.5, n_display + 0.5)
        ax.set_xlabel('Component')

    axes[0].set_ylabel('Singular value')
    return fig


def validate_positive_int(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f'{name} must be positive, got {value}.')


def g3_to_g2(gain: np.ndarray) -> TangentialGainResult:
    gain = np.asarray(gain, dtype=np.float64)
    if gain.ndim != 2:
        raise ValueError(f'Forward gain must be 2D, got {gain.shape}.')
    if gain.shape[1] % 3 != 0:
        raise ValueError(
            f'Forward gain columns must be divisible by 3, got {gain.shape[1]}.'
        )
    _assert_finite('forward_gain', gain)

    n_channels, _ = gain.shape
    n_sites = gain[:, 0::3].shape[1]
    gain_2d_columns = []
    gain_2d_unnormalized_columns = []
    valid_site_indices = []
    invalid_site_indices = []
    invalid_tangential_norms = []

    for site_idx in range(n_sites):
        site_gain = gain[:, 3 * site_idx : 3 * (site_idx + 1)]
        _, _, vh = np.linalg.svd(site_gain, full_matrices=False)
        tangential_gain = site_gain @ vh.T[:, :2]
        norms = np.sqrt(np.sum(tangential_gain**2, axis=0, keepdims=True))
        if np.any(norms <= EPS):
            invalid_site_indices.append(site_idx)
            invalid_tangential_norms.append(norms.ravel())
            continue
        gain_2d_columns.append(tangential_gain / norms)
        gain_2d_unnormalized_columns.append(tangential_gain)
        valid_site_indices.append(site_idx)

    if gain_2d_columns:
        gain_2d = np.concatenate(gain_2d_columns, axis=1)
        gain_2d_unnormalized = np.concatenate(gain_2d_unnormalized_columns, axis=1)
    else:
        gain_2d = np.empty((n_channels, 0), dtype=np.float64)
        gain_2d_unnormalized = np.empty((n_channels, 0), dtype=np.float64)

    return TangentialGainResult(
        gain_2d=gain_2d,
        gain_2d_unnormalized=gain_2d_unnormalized,
        valid_site_indices=np.asarray(valid_site_indices, dtype=int),
        invalid_site_indices=np.asarray(invalid_site_indices, dtype=int),
        invalid_tangential_norms=(
            np.asarray(invalid_tangential_norms, dtype=np.float64).reshape(-1, 2)
            if invalid_tangential_norms
            else np.empty((0, 2), dtype=np.float64)
        ),
        n_source_sites=n_sites,
    )


def music_scan(gain_2d: np.ndarray, subspace: np.ndarray) -> np.ndarray:
    if gain_2d.ndim != 2 or subspace.ndim != 2:
        raise ValueError(
            f'Expected 2D gain and subspace, got {gain_2d.shape} and '
            f'{subspace.shape}.'
        )
    if gain_2d.shape[0] != subspace.shape[0]:
        raise ValueError(
            f'RAP-MUSIC channel mismatch: gain has {gain_2d.shape[0]} rows, '
            f'subspace has {subspace.shape[0]}.'
        )
    if gain_2d.shape[1] % 2 != 0:
        raise ValueError(
            f'Tangential gain columns must be divisible by 2, got {gain_2d.shape[1]}.'
        )

    tmp = subspace.T @ gain_2d
    c11c22 = np.sum(tmp**2, axis=0)
    tmp1 = tmp[:, ::2]
    tmp2 = tmp[:, 1::2]
    c12 = np.sum(tmp1 * tmp2, axis=0)

    tr = c11c22[::2] + c11c22[1::2]
    determinant = c11c22[::2] * c11c22[1::2] - c12**2
    discriminant = tr**2 - 4 * determinant
    if float(np.min(discriminant)) < -1e-9:
        raise ValueError('Negative RAP-MUSIC discriminant beyond numerical tolerance.')
    discriminant = np.maximum(discriminant, 0.0)

    lambda1 = np.sqrt(0.5 * (tr + np.sqrt(discriminant)))
    lambda2_squared = tr - lambda1**2
    if float(np.min(lambda2_squared)) < -1e-9:
        raise ValueError('Negative RAP-MUSIC eigenvalue beyond numerical tolerance.')
    lambda2 = np.sqrt(np.maximum(lambda2_squared, 0.0))
    return np.maximum(lambda1, lambda2)


def signal_subspace(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f'Expected RAP-MUSIC data to be 2D, got {data.shape}.')
    _assert_finite('rap_music_data', data)
    left_vectors, singular_values, _ = np.linalg.svd(data, full_matrices=False)
    max_singular_value = float(np.max(singular_values)) if singular_values.size else 0.0
    if max_singular_value <= EPS:
        raise ValueError('Cannot compute RAP-MUSIC subspace for near-zero data.')
    return left_vectors


def rap_music_scan(
    data: np.ndarray,
    gain: np.ndarray,
    music_threshold: float,
) -> RapMusicResult:
    validate_energy_threshold(music_threshold)
    data = np.asarray(data, dtype=np.float64)
    gain = np.asarray(gain, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f'Expected RAP-MUSIC data to be 2D, got {data.shape}.')
    if gain.ndim != 2:
        raise ValueError(f'Expected forward gain to be 2D, got {gain.shape}.')
    if data.shape[0] != gain.shape[0]:
        raise ValueError(
            f'RAP-MUSIC channel mismatch: data has {data.shape[0]} rows, '
            f'forward gain has {gain.shape[0]}.'
        )
    _assert_finite('rap_music_data', data)
    _assert_finite('rap_music_gain', gain)

    n_channels = gain.shape[0]
    n_sites = gain.shape[1] // 3
    max_dipoles = min(data.shape[1], n_sites)
    projected_data = data
    projected_gain = gain
    values: list[float] = []
    indices: list[int] = []
    initial_invalid_site_indices = np.empty(0, dtype=int)
    initial_invalid_tangential_norms = np.empty((0, 2), dtype=np.float64)
    invalid_site_indices_by_iteration = []
    iteration_invalid_counts = []
    iteration_valid_counts = []

    while len(indices) < max_dipoles:
        tangential_gain = g3_to_g2(projected_gain)
        if not iteration_invalid_counts:
            initial_invalid_site_indices = tangential_gain.invalid_site_indices
            initial_invalid_tangential_norms = tangential_gain.invalid_tangential_norms
        invalid_site_indices_by_iteration.append(tangential_gain.invalid_site_indices)
        iteration_invalid_counts.append(tangential_gain.invalid_site_indices.size)
        iteration_valid_counts.append(tangential_gain.valid_site_indices.size)
        if tangential_gain.valid_site_indices.size == 0:
            if values:
                break
            raise ValueError(
                'No valid RAP-MUSIC source sites remain after excluding near-zero '
                'tangential gains.'
            )
        subspace = signal_subspace(projected_data)
        correlations = music_scan(tangential_gain.gain_2d, subspace)
        value = float(np.max(correlations))
        compressed_index = int(np.argmax(correlations))
        index = int(tangential_gain.valid_site_indices[compressed_index])
        if value <= music_threshold:
            break

        values.append(value)
        indices.append(index)

        active_gain = projected_gain[:, 3 * index : 3 * (index + 1)]
        gram = active_gain.T @ active_gain
        projector = np.eye(n_channels) - active_gain @ np.linalg.pinv(gram) @ active_gain.T
        projected_data = projector @ projected_data
        projected_gain = projector @ projected_gain

    union_invalid_site_indices = (
        np.unique(np.concatenate(invalid_site_indices_by_iteration))
        if invalid_site_indices_by_iteration
        else np.empty(0, dtype=int)
    )
    return RapMusicResult(
        values=values,
        indices=indices,
        n_source_sites=n_sites,
        initial_invalid_site_indices=initial_invalid_site_indices,
        initial_invalid_tangential_norms=initial_invalid_tangential_norms,
        union_invalid_site_indices=union_invalid_site_indices,
        iteration_invalid_counts=np.asarray(iteration_invalid_counts, dtype=int),
        iteration_valid_counts=np.asarray(iteration_valid_counts, dtype=int),
    )


def align_topographies_to_forward(
    topographies: np.ndarray,
    info: object,
    fwd: object,
) -> np.ndarray:
    topographies = np.asarray(topographies, dtype=np.float64)
    info_ch_names = list(info['ch_names'])
    row_names = list(fwd['sol']['row_names'])
    if topographies.ndim != 2:
        raise ValueError(f'Expected topographies to be 2D, got {topographies.shape}.')
    if topographies.shape[1] != len(info_ch_names):
        raise ValueError(
            f'Topography channel count {topographies.shape[1]} does not match '
            f'info channel count {len(info_ch_names)}.'
        )
    missing = [name for name in row_names if name not in info_ch_names]
    if missing:
        raise ValueError(
            'Forward solution contains channels not present in topography info: '
            f'{missing[:10]}'
        )
    indices = np.asarray([info_ch_names.index(name) for name in row_names], dtype=int)
    aligned = topographies[:, indices]
    _assert_finite('forward_aligned_topographies', aligned)
    return aligned


def source_index_hemisphere(fwd: object, source_index: int) -> str:
    n_lh = int(fwd['src'][0]['nuse'])
    n_rh = int(fwd['src'][1]['nuse'])
    if source_index < 0 or source_index >= n_lh + n_rh:
        raise ValueError(
            f'source_index {source_index} is outside the forward source space '
            f'(n_lh={n_lh}, n_rh={n_rh}).'
        )
    return 'lh' if source_index < n_lh else 'rh'


def fsaverage_coords_from_head(fwd: object, indices: np.ndarray, trans: object) -> np.ndarray:
    from mne.transforms import apply_trans

    coords_head = np.atleast_2d(np.asarray(fwd['source_rr'], dtype=np.float64)[indices])
    return np.atleast_2d(apply_trans(trans, coords_head))


def build_participant_fsaverage_forwards(
    *,
    template_info: object,
    scaler_ch_names: Sequence[str],
    subjects: Sequence[int],
    session: int,
    story_id: int,
    bids_root: str | Path,
    n_jobs: int,
    geometry_cache_dir: str | Path,
) -> tuple[object, dict[int, tuple[object, object, object, dict]]]:
    """Shared fsaverage anatomy plus one reconstructed forward per participant."""

    from lisa.data.kit_geometry import prepare_participant_fsaverage_forward
    from lisa.plots.source_estimation import prepare_source_geometry

    validate_positive_int('dipole_n_jobs', n_jobs)
    anatomy = prepare_source_geometry()
    cache_dir = Path(geometry_cache_dir)
    models: dict[int, tuple[object, object, object, dict]] = {}
    for subject in subjects:
        models[int(subject)] = prepare_participant_fsaverage_forward(
            template_info=template_info,
            scaler_ch_names=scaler_ch_names,
            bids_root=bids_root,
            subject=int(subject),
            session=int(session),
            task=story_id,
            src=anatomy.src,
            bem=anatomy.bem,
            subjects_dir=anatomy.subjects_dir,
            cache_dir=cache_dir,
            n_jobs=n_jobs,
        )
    return anatomy, models


def fit_svd_pattern_dipoles(
    variant: SVDVariantResult,
    info: object,
    fwd: object,
    component_count: int,
    music_threshold: float,
) -> dict[str, list]:
    validate_positive_int('dipole_components', component_count)
    n_available = variant.pattern_result.right_singular_vectors.shape[0]
    n_components = min(component_count, n_available)
    topographies = variant.pattern_result.right_singular_vectors[:n_components]
    aligned_topographies = align_topographies_to_forward(topographies, info, fwd)
    scan = rap_music_scan(
        data=aligned_topographies.T,
        gain=fwd['sol']['data'],
        music_threshold=music_threshold,
    )
    coords = (
        fwd['source_rr'][scan.indices]
        if scan.indices
        else np.empty((0, 3), dtype=np.float64)
    )
    return {
        'coords': [np.atleast_2d(coords)],
        'vals': [scan.values],
        'index': [scan.indices],
        'n_components': [n_components],
        'n_source_sites': scan.n_source_sites,
        'initial_invalid_site_indices': scan.initial_invalid_site_indices,
        'initial_invalid_tangential_norms': scan.initial_invalid_tangential_norms,
        'union_invalid_site_indices': scan.union_invalid_site_indices,
        'iteration_invalid_counts': scan.iteration_invalid_counts,
        'iteration_valid_counts': scan.iteration_valid_counts,
    }


def plot_dipoles_2d(
    fit_res: dict[str, list],
    *,
    trans: object,
    subject: str,
    subjects_dir: str,
):
    import mne
    from matplotlib.patches import Patch

    coords = np.vstack(fit_res['coords']) if fit_res['coords'] else np.empty((0, 3))
    if coords.size == 0:
        raise RuntimeError('No dipoles to plot.')
    amplitudes = (
        np.hstack(fit_res['vals'])
        if fit_res['vals'] and len(fit_res['vals'][0]) > 0
        else np.ones(coords.shape[0])
    )
    dipoles = mne.Dipole(
        times=np.zeros(len(coords)),
        pos=coords,
        amplitude=amplitudes,
        ori=np.zeros((len(coords), 3)),
        gof=np.zeros(len(coords)),
    )
    left_color, right_color = 'crimson', 'steelblue'
    fig = dipoles.plot_locations(
        trans=trans,
        subject=subject,
        subjects_dir=subjects_dir,
        mode='outlines',
        color=[left_color if x < 0.0 else right_color for x in coords[:, 0]],
        show_all=False,
        show=False,
    )
    fig.set_size_inches(FULL_WIDTH_IN, 2.6, forward=True)
    for ax in fig.axes:
        ax.grid(False)
        ax.set_title(ax.get_title(), fontsize=9.0, pad=2.0)
        ax.tick_params(axis='both', labelsize=7.0, length=2.0, width=0.6, colors='0.35')
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color('0.55')
        for line in ax.lines:
            if line.get_marker() in ('None', 'none', ''):
                line.set_linewidth(0.7)
                line.set_color('0.45')
    legend = fig.axes[1].legend(
        handles=[
            Patch(facecolor=left_color, edgecolor='none', label='Left'),
            Patch(facecolor=right_color, edgecolor='none', label='Right'),
        ],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        facecolor='white',
        edgecolor='0.65',
        borderpad=0.45,
        handlelength=0.9,
        handleheight=0.7,
        handletextpad=0.5,
        columnspacing=1.1,
        fontsize=8.0,
    )
    legend.get_frame().set_linewidth(0.6)
    legend.get_frame().set_boxstyle('round', pad=0.25, rounding_size=0.6)
    return fig


def save_dipole_outputs(
    variants: Sequence[SVDVariantResult],
    stack: TopographyStack,
    outdir: Path,
    component_count: int,
    music_threshold: float,
    dpi: int,
    n_jobs: int,
    *,
    subjects: Sequence[int],
    session: int,
    story_id: int,
    bids_root: str | Path,
    geometry_cache_dir: str | Path,
) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    from mne.transforms import Transform

    if not subjects:
        raise ValueError('At least one subject is required for dipole localisation.')
    anatomy, participant_models = build_participant_fsaverage_forwards(
        template_info=stack.info,
        scaler_ch_names=list(stack.info['ch_names']),
        subjects=subjects,
        session=session,
        story_id=story_id,
        bids_root=bids_root,
        n_jobs=n_jobs,
        geometry_cache_dir=geometry_cache_dir,
    )
    identity_trans = Transform('head', 'mri', np.eye(4))
    rows = []
    diagnostic_rows = []
    qc_rows = [model[3] for model in participant_models.values()]
    for variant in variants:
        pooled_coords = []
        pooled_values = []
        pooled_indices = []
        pooled_participants = []
        pooled_hemispheres = []
        n_components = None
        for subject in subjects:
            info, trans, fwd, _qc = participant_models[int(subject)]
            fit_res = fit_svd_pattern_dipoles(
                variant=variant,
                info=info,
                fwd=fwd,
                component_count=component_count,
                music_threshold=music_threshold,
            )
            n_components = int(fit_res['n_components'][0])
            n_source_sites = int(fit_res['n_source_sites'])
            initial_invalid_indices = np.asarray(
                fit_res['initial_invalid_site_indices'],
                dtype=int,
            )
            union_invalid_indices = np.asarray(
                fit_res['union_invalid_site_indices'],
                dtype=int,
            )
            iteration_invalid_counts = np.asarray(
                fit_res['iteration_invalid_counts'],
                dtype=int,
            )
            iteration_valid_counts = np.asarray(
                fit_res['iteration_valid_counts'],
                dtype=int,
            )
            initial_invalid_count = int(initial_invalid_indices.size)
            union_invalid_count = int(union_invalid_indices.size)
            max_iteration_invalid_count = (
                int(np.max(iteration_invalid_counts))
                if iteration_invalid_counts.size
                else 0
            )
            min_iteration_valid_count = (
                int(np.min(iteration_valid_counts))
                if iteration_valid_counts.size
                else 0
            )
            initial_invalid_fraction = (
                initial_invalid_count / n_source_sites if n_source_sites else math.nan
            )
            union_invalid_fraction = (
                union_invalid_count / n_source_sites if n_source_sites else math.nan
            )
            max_iteration_invalid_fraction = (
                max_iteration_invalid_count / n_source_sites
                if n_source_sites
                else math.nan
            )
            indices = np.asarray(fit_res['index'][0], dtype=int)
            values = np.asarray(fit_res['vals'][0], dtype=np.float64)
            n_dipoles = int(indices.size)
            diagnostic_rows.append(
                {
                    'participant': int(subject),
                    'session': int(session),
                    'task': str(story_id),
                    'svd_variant': variant.key,
                    'topography_type': 'spatial_pattern',
                    'n_source_sites': n_source_sites,
                    'initial_invalid_source_count': initial_invalid_count,
                    'initial_invalid_source_fraction': initial_invalid_fraction,
                    'initial_invalid_source_percent': initial_invalid_fraction * 100.0,
                    'union_invalid_source_count': union_invalid_count,
                    'union_invalid_source_fraction': union_invalid_fraction,
                    'union_invalid_source_percent': union_invalid_fraction * 100.0,
                    'max_iteration_invalid_source_count': max_iteration_invalid_count,
                    'max_iteration_invalid_source_fraction': (
                        max_iteration_invalid_fraction
                    ),
                    'max_iteration_invalid_source_percent': (
                        max_iteration_invalid_fraction * 100.0
                    ),
                    'min_iteration_valid_source_count': min_iteration_valid_count,
                    'rap_music_iteration_count': int(iteration_invalid_counts.size),
                    'component_count': n_components,
                    'music_threshold': float(music_threshold),
                    'rap_music_subspace': 'full_selected_topography_components',
                }
            )
            print(
                'RAP-MUSIC '
                f'{variant.key} sub-{int(subject):02d}: excluded '
                f'{initial_invalid_count}/{n_source_sites} '
                'initial near-zero tangential-gain source sites '
                f'({initial_invalid_fraction:.2%}); '
                f'n_dipoles={n_dipoles}.'
            )
            if n_dipoles == 0:
                warnings.warn(
                    f'No RAP-MUSIC dipoles found for subject {int(subject)} '
                    f'and SVD variant {variant.key!r} with music threshold '
                    f'{music_threshold}.',
                    UserWarning,
                    stacklevel=2,
                )
                continue
            coords = fsaverage_coords_from_head(fwd, indices, trans)
            for dipole_idx, (coord, value, source_index) in enumerate(
                zip(coords, values, indices, strict=True),
                start=1,
            ):
                hemisphere = source_index_hemisphere(fwd, int(source_index))
                rows.append(
                    {
                        'participant': int(subject),
                        'session': int(session),
                        'task': str(story_id),
                        'svd_variant': variant.key,
                        'topography_type': 'spatial_pattern',
                        'dipole': dipole_idx,
                        'source_index': int(source_index),
                        'hemisphere': hemisphere,
                        'subspace_correlation': float(value),
                        'x_m': float(coord[0]),
                        'y_m': float(coord[1]),
                        'z_m': float(coord[2]),
                        'coordinate_frame': 'fsaverage',
                        'component_count': n_components,
                        'music_threshold': float(music_threshold),
                        'rap_music_subspace': 'full_selected_topography_components',
                        'n_source_sites': n_source_sites,
                        'initial_invalid_source_count': initial_invalid_count,
                        'initial_invalid_source_fraction': initial_invalid_fraction,
                        'union_invalid_source_count': union_invalid_count,
                        'union_invalid_source_fraction': union_invalid_fraction,
                    }
                )
                pooled_coords.append(coord)
                pooled_values.append(float(value))
                pooled_indices.append(int(source_index))
                pooled_participants.append(int(subject))
                pooled_hemispheres.append(hemisphere)

        n_participants_with_dipoles = len(set(pooled_participants))
        print(
            f'RAP-MUSIC {variant.key}: '
            f'{n_participants_with_dipoles}/{len(subjects)} participants '
            'contributed dipoles.'
        )
        dipole_stem = f'dipoles_{variant.key}_k{int(n_components)}'
        if not pooled_coords:
            warnings.warn(
                f'No RAP-MUSIC dipoles found for SVD variant {variant.key!r}; '
                'skipping pooled dipole plot.',
                UserWarning,
                stacklevel=2,
            )
            pooled_coords_m = np.empty((0, 3), dtype=np.float64)
        else:
            pooled_coords_m = np.vstack(pooled_coords)
            plot_fit = {
                'coords': [np.atleast_2d(pooled_coords_m)],
                'vals': [pooled_values],
                'index': [pooled_indices],
                'n_components': [n_components],
            }
            fig = plot_dipoles_2d(
                fit_res=plot_fit,
                trans=identity_trans,
                subject=anatomy.subject,
                subjects_dir=anatomy.subjects_dir,
            )
            for ext in ('pdf', 'png'):
                fig.savefig(
                    outdir / f'{dipole_stem}.{ext}',
                    dpi=dpi,
                    bbox_inches='tight',
                )
            plt.close(fig)
        np.savez_compressed(
            outdir / f'{dipole_stem}.npz',
            coords_m=pooled_coords_m,
            subspace_correlations=np.asarray(pooled_values, dtype=np.float64),
            source_indices=np.asarray(pooled_indices, dtype=int),
            participants=np.asarray(pooled_participants, dtype=int),
            hemispheres=np.asarray(pooled_hemispheres),
            component_count=np.asarray(n_components, dtype=int),
            music_threshold=np.asarray(music_threshold, dtype=np.float64),
            rap_music_subspace=np.asarray('full_selected_topography_components'),
            svd_variant=np.asarray(variant.key),
            topography_type=np.asarray('spatial_pattern'),
            coordinate_frame=np.asarray('fsaverage'),
            session=np.asarray(session, dtype=int),
            task=np.asarray(str(story_id)),
            subject=np.asarray(anatomy.subject),
            subjects_dir=np.asarray(anatomy.subjects_dir),
        )

    pd.DataFrame(rows).to_csv(outdir / 'dipoles.csv', index=False)
    pd.DataFrame(diagnostic_rows).to_csv(outdir / 'dipole_diagnostics.csv', index=False)
    pd.DataFrame(qc_rows).to_csv(outdir / 'coregistration_qc.csv', index=False)


def save_svd_outputs(
    variants: Sequence[SVDVariantResult],
    stack: TopographyStack,
    outdir: Path,
    energy_threshold: float,
    run_group: str,
    run_id: str,
    run_name: str,
    ses: int,
    story_id: int,
    dpi: int,
    demean_temporal_filters_enabled: bool,
    roughness_qc: dict[str, object],
    n_display: int = 10,
) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    tables = []
    for variant in variants:
        display_count = min(
            n_display,
            variant.filter_result.right_singular_vectors.shape[0],
            variant.pattern_result.right_singular_vectors.shape[0],
        )
        vector_fig = plot_singular_vectors(
            filter_result=variant.filter_result,
            pattern_result=variant.pattern_result,
            info=stack.info,
            n_display=n_display,
        )
        energy_fig = plot_explained_energy(
            filter_result=variant.filter_result,
            pattern_result=variant.pattern_result,
            energy_threshold=energy_threshold,
            n_display=n_display,
        )
        singular_value_fig = plot_singular_value_stems(
            filter_result=variant.filter_result,
            pattern_result=variant.pattern_result,
        )
        for ext in ('pdf', 'png'):
            vector_fig.savefig(
                outdir / f'components_{variant.key}_k{display_count}.{ext}',
                dpi=dpi,
                bbox_inches='tight',
            )
            energy_fig.savefig(
                outdir / f'energy_{variant.key}.{ext}',
                dpi=dpi,
                bbox_inches='tight',
            )
            singular_value_fig.savefig(
                outdir / f'singular_values_{variant.key}.{ext}',
                dpi=dpi,
                bbox_inches='tight',
            )
        plt.close(vector_fig)
        plt.close(energy_fig)
        plt.close(singular_value_fig)

        variant_table = pd.concat(
            (
                component_table(
                    variant.filter_result,
                    topography_type='spatial_filter',
                    svd_variant=variant.key,
                ),
                component_table(
                    variant.pattern_result,
                    topography_type='spatial_pattern',
                    svd_variant=variant.key,
                ),
            ),
            ignore_index=True,
        )
        variant_table.to_csv(
            outdir / f'energy_{variant.key}.csv',
            index=False,
        )
        tables.append(variant_table)

    table = pd.concat(tables, ignore_index=True)
    table.to_csv(outdir / 'energy.csv', index=False)

    arrays = {
        'schema_version': np.asarray(2, dtype=int),
        'original_spatial_filters': stack.spatial_filters,
        'original_spatial_patterns': stack.spatial_patterns,
        'demeaned_temporal_spatial_patterns': (
            stack.demeaned_temporal_spatial_patterns
            if stack.demeaned_temporal_spatial_patterns is not None
            else np.asarray([], dtype=np.float64)
        ),
        'has_demeaned_temporal_filter_variants': np.asarray(
            stack.demeaned_temporal_spatial_patterns is not None,
            dtype=bool,
        ),
        'item_subjects': stack.item_subjects,
        'item_branches': stack.item_branches,
        'energy_threshold': np.asarray(energy_threshold, dtype=np.float64),
        'demean_temporal_filters': np.asarray(
            bool(demean_temporal_filters_enabled),
            dtype=bool,
        ),
        'svd_variants': np.asarray([variant.key for variant in variants]),
        'run_group': np.asarray(run_group),
        'run_id': np.asarray(run_id),
        'run_name': np.asarray(run_name),
        'session': np.asarray(ses, dtype=int),
        'story_id': np.asarray(story_id, dtype=int),
        'n_branches': np.asarray(stack.n_branches, dtype=int),
        'n_items': np.asarray(roughness_qc['n_items'], dtype=int),
        'n_dropped_items': np.asarray(roughness_qc['n_dropped_items'], dtype=int),
        'n_rough_items': np.asarray(roughness_qc['n_rough_items'], dtype=int),
        'n_excluded_items': np.asarray(roughness_qc['n_excluded_items'], dtype=int),
        'n_retained_items': np.asarray(roughness_qc['n_retained_items'], dtype=int),
        'spatial_roughness_threshold': np.asarray(
            roughness_qc['spatial_roughness_threshold'],
            dtype=np.float64,
        ),
        'exclude_rough': np.asarray(roughness_qc['exclude_rough'], dtype=bool),
        'n_subjects_before_roughness_qc': np.asarray(
            roughness_qc['n_subjects_before'],
            dtype=int,
        ),
        'n_subjects_after_roughness_qc': np.asarray(
            roughness_qc['n_subjects_after'],
            dtype=int,
        ),
        'min_retained_branches_per_participant': np.asarray(
            roughness_qc['min_retained_branches_per_participant'],
            dtype=int,
        ),
        'max_retained_branches_per_participant': np.asarray(
            roughness_qc['max_retained_branches_per_participant'],
            dtype=int,
        ),
    }
    for variant in variants:
        arrays[f'{variant.key}_spatial_filters'] = variant.spatial_filters
        arrays[f'{variant.key}_spatial_patterns'] = variant.spatial_patterns
        if variant.filter_row_norms is not None:
            arrays[f'{variant.key}_spatial_filter_row_l2_norms'] = (
                variant.filter_row_norms
            )
        if variant.pattern_row_norms is not None:
            arrays[f'{variant.key}_spatial_pattern_row_l2_norms'] = (
                variant.pattern_row_norms
            )

        for topography_type, result in (
            ('spatial_filter', variant.filter_result),
            ('spatial_pattern', variant.pattern_result),
        ):
            prefix = f'{variant.key}_{topography_type}'
            arrays[f'{prefix}_singular_values'] = result.singular_values
            arrays[f'{prefix}_right_singular_vectors'] = result.right_singular_vectors
            arrays[f'{prefix}_displayed_right_singular_vectors'] = (
                result.right_singular_vectors[:n_display]
            )
            arrays[f'{prefix}_explained_energy'] = result.explained_energy
            arrays[f'{prefix}_cumulative_energy'] = result.cumulative_energy
            arrays[f'{prefix}_n_components'] = np.asarray(
                result.n_components,
                dtype=int,
            )
        arrays[f'{variant.key}_displayed_singular_vector_count'] = np.asarray(
            min(
                n_display,
                variant.filter_result.right_singular_vectors.shape[0],
                variant.pattern_result.right_singular_vectors.shape[0],
            ),
            dtype=int,
        )
    np.savez_compressed(outdir / 'data.npz', **arrays)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--run-id', type=str, required=True, help='ID of the trained model run.'
    )
    parser.add_argument(
        '--run-group',
        type=str,
        required=True,
        help='Run group id under experiments_root.',
    )
    parser.add_argument(
        '--experiments-root',
        type=str,
        default=EXPERIMENTS_DIR,
        help='Root directory with experiment logs.',
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default=str(Path(PLOTS_DIR) / 'spatial_topography_svd'),
        help='Output directory for SVD plots and tables.',
    )
    parser.add_argument(
        '--subjects',
        type=int,
        nargs='+',
        default=list(range(1, N_SUBJECTS + 1)),
        help='Subject IDs to include.',
    )
    parser.add_argument('--session', type=int, default=0, help='Session number to use.')
    parser.add_argument('--story-id', type=int, default=1, help='Story ID to use.')
    parser.add_argument(
        '--offset-gap',
        type=float,
        default=10.0,
        help='Seconds removed from the start and end of each recording.',
    )
    parser.add_argument(
        '--energy-threshold',
        type=float,
        default=0.95,
        help='Cumulative explained-energy threshold for selecting right singular vectors.',
    )
    parser.add_argument(
        '--svd-normalization',
        type=str,
        default='both',
        choices=SVD_NORMALIZATIONS,
        help=(
            'Which SVD variants to compute: unnormalized topographies (raw), '
            'row-L2-normalized topographies (row_l2), or both. Default: both.'
        ),
    )
    parser.add_argument(
        '--demean-temporal-filters',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Subtract each learned temporal filter mean before spatial-pattern '
            'covariance and all SVD/dipole outputs. Default: disabled.'
        ),
    )
    parser.add_argument(
        '--meg-files-dir',
        type=str,
        default=CLEAN_DATA_DIR,
        help='Directory containing cleaned raw MEG files.',
    )
    parser.add_argument(
        '--meg-format',
        type=str,
        default='fif',
        choices=('bids', 'fif'),
        help='Format of the MEG files.',
    )
    parser.add_argument(
        '--preprocessed-meg-path',
        type=str,
        default=None,
        help='Path to preprocessed MEG NPZ.',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Output DPI for saved PNG/PDF figures.',
    )
    parser.add_argument(
        '--dipoles',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Fit and plot RAP-MUSIC dipoles for the selected spatial-pattern '
            'SVD variants.'
        ),
    )
    parser.add_argument(
        '--raw-bids-root',
        type=str,
        default=str(Path(DATA_ROOT) / 'MASC-MEG'),
        help='Canonical raw/BIDS root used for participant ELP/HSP/MRK geometry.',
    )
    parser.add_argument(
        '--geometry-cache-dir',
        type=str,
        default=str(Path(OUTPUTS_DIR) / 'source_geometry'),
        help='Directory for cached participant head to fsaverage transforms.',
    )
    parser.add_argument(
        '--dipole-components',
        type=int,
        default=10,
        help='Number of leading spatial-pattern SVD components used for RAP-MUSIC.',
    )
    parser.add_argument(
        '--dipole-thr-music',
        type=float,
        default=0.98,
        help='Minimum RAP-MUSIC subspace correlation for accepting a dipole.',
    )
    parser.add_argument(
        '--dipole-n-jobs',
        type=int,
        default=3,
        help='Number of jobs for building the MNE forward solution.',
    )
    return parser.parse_args()


def main_cli() -> None:
    args = parse_arguments()
    validate_run_group(args.run_group)
    validate_energy_threshold(args.energy_threshold)
    validate_svd_normalization(args.svd_normalization)
    validate_positive_int('dipole_components', args.dipole_components)
    validate_energy_threshold(args.dipole_thr_music)
    validate_positive_int('dipole_n_jobs', args.dipole_n_jobs)

    from lisa.model.load_model import load_model

    model, hyper_params = load_model(
        run_id=args.run_id,
        experiments_root=args.experiments_root,
        run_group=args.run_group,
    )

    if args.preprocessed_meg_path is None:
        args.preprocessed_meg_path = (
            Path(PREPROCESSED_DATA_DIR)
            / 'meg'
            / f'meg{N_SUBJECTS}_sr{int(hyper_params["meg_sr"])}.npz'
        )

    stack = collect_spatial_topographies(
        model=model,
        hyper_params=hyper_params,
        subjects=args.subjects,
        ses=args.session,
        story_id=args.story_id,
        offset_gap=args.offset_gap,
        meg_format=args.meg_format,
        data_root=args.meg_files_dir,
        preprocessed_meg_path=args.preprocessed_meg_path,
        demean_temporal_filters_enabled=args.demean_temporal_filters,
    )
    stack, roughness_qc = exclude_rough_spatial_items(
        stack,
        dirprocess=hyper_params['dirprocess'],
    )
    print(
        'Spatial-roughness QC from '
        'lisa.plots.branch_interpretation.compute_item_spatial_roughness '
        f'(threshold={roughness_qc["spatial_roughness_threshold"]}, '
        'exclude if roughness > threshold) in '
        'lisa.plots.plot_spatial_topography_svd: '
        f'n_items={roughness_qc["n_items"]} '
        f'n_dropped_items={roughness_qc["n_dropped_items"]} '
        f'n_retained_items={roughness_qc["n_retained_items"]} '
        f'n_subjects={roughness_qc["n_subjects_before"]}->'
        f'{roughness_qc["n_subjects_after"]} '
        f'retained_branches_per_participant='
        f'{roughness_qc["min_retained_branches_per_participant"]}..'
        f'{roughness_qc["max_retained_branches_per_participant"]}; '
        'same retain mask applied to spatial_filters, spatial_patterns, '
        'item_subjects, item_branches'
        + (
            ', and demeaned_temporal_spatial_patterns'
            if stack.demeaned_temporal_spatial_patterns is not None
            else ''
        )
    )

    run_name = str(hyper_params['run_name'])
    run_label = f'{run_name}-{args.run_id}'
    outdir = (
        Path(args.outdir)
        / args.run_group
        / run_label
        / (
            f'ses{args.session}-story{args.story_id}'
            f'{TEMPORAL_FILTER_SUFFIX[bool(args.demean_temporal_filters)]}'
        )
    )
    outdir.mkdir(parents=True, exist_ok=True)

    variants = compute_svd_variants(
        stack=stack,
        energy_threshold=args.energy_threshold,
        normalization=args.svd_normalization,
    )
    save_svd_outputs(
        variants=variants,
        stack=stack,
        outdir=outdir,
        energy_threshold=args.energy_threshold,
        run_group=args.run_group,
        run_id=args.run_id,
        run_name=run_name,
        ses=args.session,
        story_id=args.story_id,
        dpi=args.dpi,
        demean_temporal_filters_enabled=args.demean_temporal_filters,
        roughness_qc=roughness_qc,
    )
    if args.dipoles:
        save_dipole_outputs(
            variants=variants,
            stack=stack,
            outdir=outdir,
            component_count=args.dipole_components,
            music_threshold=args.dipole_thr_music,
            dpi=args.dpi,
            n_jobs=args.dipole_n_jobs,
            subjects=args.subjects,
            session=args.session,
            story_id=args.story_id,
            bids_root=args.raw_bids_root,
            geometry_cache_dir=args.geometry_cache_dir,
        )

    variant_keys = ', '.join(variant.key for variant in variants)
    print(f'Saved SVD outputs ({variant_keys}) to {outdir}')


if __name__ == '__main__':
    main_cli()
