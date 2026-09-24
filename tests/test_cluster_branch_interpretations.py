from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from lisa.plots.branch_interpretation import (  # noqa: E402
    DEFAULT_MAIN_SPATIAL_ROUGHNESS_MAX,
    build_clustering_outputs as build_combined_outputs,
    correlation_similarity,
    save_item_table_npz,
    validate_curated_assignments,
)
from lisa.plots.plot_branch_interpretations import (  # noqa: E402
    build_medoid_similarity,
    build_curated_cluster_specs,
    get_cluster_display_label,
    load_cached_item_table,
    with_sequential_display_labels,
)


def _toy_items() -> dict[str, np.ndarray]:
    spatial_patterns = np.array(
        [
            [1.00, 0.00, 0.00, 0.00],
            [0.96, 0.06, 0.00, 0.00],
            [1.04, -0.04, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00],
            [0.04, 0.95, 0.00, 0.00],
            [-0.03, 1.03, 0.00, 0.00],
            [0.00, 0.00, 1.00, 0.20],
            [0.00, 0.00, 0.92, 0.30],
        ],
        dtype=np.float64,
    )
    temporal_spectra_log = np.array(
        [
            [4.0, 3.0, 2.0, 1.0],
            [3.9, 3.1, 2.1, 1.0],
            [4.1, 2.9, 1.9, 1.1],
            [1.0, 2.0, 3.0, 4.0],
            [1.1, 2.1, 2.9, 3.9],
            [0.9, 1.9, 3.1, 4.1],
            [2.8, 1.0, 2.7, 1.1],
            [2.7, 1.1, 2.8, 1.0],
        ],
        dtype=np.float64,
    )
    source_magnitudes = np.abs(spatial_patterns)
    return {
        'subjects': np.arange(1, 9, dtype=int),
        'branches': np.array([0, 0, 1, 0, 1, 1, 0, 1], dtype=int),
        'spatial_patterns': spatial_patterns,
        'source_magnitudes': source_magnitudes,
        'temporal_spectra_log': temporal_spectra_log,
        'sensor_xy': np.column_stack(
            [
                np.arange(spatial_patterns.shape[1], dtype=np.float64),
                np.zeros(spatial_patterns.shape[1], dtype=np.float64),
            ]
        ),
    }


def _residue_promotion_items(
    *,
    noisy_third_cluster: bool,
) -> dict[str, np.ndarray]:
    sensor_xy = np.column_stack(
        [
            np.arange(20, dtype=np.float64),
            np.zeros(20, dtype=np.float64),
        ]
    )

    cluster_a = np.exp(-((np.arange(20, dtype=np.float64) - 4.0) ** 2) / 10.0)
    cluster_b = np.exp(-((np.arange(20, dtype=np.float64) - 15.0) ** 2) / 10.0)
    cluster_c = np.exp(-((np.arange(20, dtype=np.float64) - 10.0) ** 2) / 8.0)
    source_c = cluster_c.copy()
    if noisy_third_cluster:
        cluster_c = np.where(np.arange(20) % 2 == 0, 1.0, -1.0).astype(np.float64)

    spatial_patterns = np.stack(
        [
            cluster_a,
            cluster_a * 0.98 + 0.01,
            cluster_a * 1.02 - 0.01,
            cluster_b,
            cluster_b * 0.99 + 0.01,
            cluster_b * 1.01 - 0.01,
            cluster_c,
            cluster_c * 0.98,
        ],
        axis=0,
    )
    source_magnitudes = np.stack(
        [
            cluster_a,
            cluster_a * 0.98 + 0.01,
            cluster_a * 1.02 - 0.01,
            cluster_b,
            cluster_b * 0.99 + 0.01,
            cluster_b * 1.01 - 0.01,
            source_c,
            source_c * 0.98,
        ],
        axis=0,
    )

    temporal_a = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5], dtype=np.float64)
    temporal_b = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    temporal_c = np.array([1.0, 2.5, 4.5, 4.5, 2.5, 1.0], dtype=np.float64)
    temporal_spectra_log = np.stack(
        [
            temporal_a,
            temporal_a + np.array([0.1, -0.1, 0.0, 0.0, 0.0, 0.0]),
            temporal_a + np.array([-0.1, 0.1, 0.0, 0.0, 0.0, 0.0]),
            temporal_b,
            temporal_b + np.array([0.0, 0.0, 0.1, -0.1, 0.0, 0.0]),
            temporal_b + np.array([0.0, 0.0, -0.1, 0.1, 0.0, 0.0]),
            temporal_c,
            temporal_c + np.array([0.0, 0.1, -0.1, 0.1, -0.1, 0.0]),
        ],
        axis=0,
    )

    return {
        'subjects': np.arange(1, 9, dtype=int),
        'branches': np.zeros(8, dtype=int),
        'spatial_patterns': spatial_patterns,
        'source_magnitudes': source_magnitudes,
        'temporal_spectra_log': temporal_spectra_log,
        'sensor_xy': sensor_xy,
    }


def _nonconstant_feature_matrix() -> np.ndarray:
    return np.array(
        [
            [1.00, 0.20, -0.40, 0.10],
            [0.50, -1.00, 0.25, 0.00],
            [-0.30, 0.60, 0.10, -0.80],
        ],
        dtype=np.float64,
    )


def test_correlation_similarity_rejects_near_constant_rows() -> None:
    features = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match='near-constant rows'):
        correlation_similarity(features, feature_name='toy_features')


def test_correlation_similarity_rejects_constant_physical_unit_rows() -> None:
    features = np.array(
        [
            [1e-13, 1e-13, 1e-13],
            [1e-13, 2e-13, 3e-13],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match='near-constant rows'):
        correlation_similarity(features, feature_name='spatial_haufe')


def test_correlation_similarity_rejects_all_zero_rows() -> None:
    features = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match='near-constant rows'):
        correlation_similarity(features, feature_name='toy_features')


def test_correlation_similarity_is_invariant_to_positive_rescaling() -> None:
    features = _nonconstant_feature_matrix()
    for absolute in (True, False):
        reference = correlation_similarity(
            features,
            feature_name='spatial_haufe',
            absolute=absolute,
        )
        for scale in (1e-20, 1e-15, 1.0, 1e15):
            scaled = correlation_similarity(
                features * scale,
                feature_name='spatial_haufe',
                absolute=absolute,
            )
            np.testing.assert_allclose(scaled, reference, rtol=0.0, atol=1e-12)


def test_correlation_similarity_accepts_physical_unit_spatial_haufe_scale() -> None:
    tesla = 1e-13
    features = _nonconstant_feature_matrix()
    physical = features * tesla
    centered = physical - physical.mean(axis=1, keepdims=True)
    centered_norms = np.linalg.norm(centered, axis=1)
    assert np.all(centered_norms > 0.0)
    assert np.all(centered_norms <= 1e-12)

    reference = correlation_similarity(
        features,
        feature_name='spatial_haufe',
        absolute=True,
    )
    physical_similarity = correlation_similarity(
        physical,
        feature_name='spatial_haufe',
        absolute=True,
    )
    np.testing.assert_allclose(
        physical_similarity,
        reference,
        rtol=0.0,
        atol=1e-12,
    )


def test_validate_cluster_assignments_accepts_plot_compatible_schema() -> None:
    items = _toy_items()
    assignments = pd.DataFrame(
        {
            'item_index': np.arange(8, dtype=int),
            'subject': items['subjects'],
            'branch': items['branches'] + 1,
            'cluster_name': ['C1', 'C1', 'C1', 'C2', 'C2', 'C2', 'R1', 'R1'],
        }
    )

    validated = validate_curated_assignments(assignments, items)

    assert validated.attrs['cluster_order'] == ['C1', 'C2', 'R1']


def test_validate_cluster_assignments_rejects_zero_based_branch_column() -> None:
    items = _toy_items()
    assignments = pd.DataFrame(
        {
            'item_index': np.arange(8, dtype=int),
            'subject': items['subjects'],
            'branch': items['branches'],
            'cluster_name': ['C1', 'C1', 'C1', 'C2', 'C2', 'C2', 'R1', 'R1'],
        }
    )

    with pytest.raises(ValueError, match='branch column'):
        validate_curated_assignments(assignments, items)


def test_build_curated_cluster_specs_sorts_equal_sizes_numerically() -> None:
    assignments = pd.DataFrame(
        {
            'item_index': np.arange(3, dtype=int),
            'cluster_name': ['C10', 'C11', 'C9'],
        }
    )
    assignments.attrs['cluster_order'] = ['C10', 'C11', 'C9']

    cluster_specs, excluded_item_count = build_curated_cluster_specs(
        assignments,
        exclude_rest=False,
    )

    assert excluded_item_count == 0
    assert [cluster['cluster_name'] for cluster in cluster_specs] == [
        'C9',
        'C10',
        'C11',
    ]


def test_sequential_display_labels_preserve_stable_cluster_names() -> None:
    clusters = [
        {'cluster_name': 'C12'},
        {'cluster_name': 'C3'},
        {'cluster_name': 'C8'},
    ]

    labeled = with_sequential_display_labels(clusters)

    assert [cluster['cluster_name'] for cluster in labeled] == ['C12', 'C3', 'C8']
    assert [get_cluster_display_label(cluster) for cluster in labeled] == [
        'Cluster 1',
        'Cluster 2',
        'Cluster 3',
    ]
    assert [get_cluster_display_label(cluster) for cluster in clusters] == [
        'Cluster 12',
        'Cluster 3',
        'Cluster 8',
    ]


def test_combined_clustering_enforces_one_threshold_in_both_views() -> None:
    result = build_combined_outputs(
        _toy_items(),
        minimum_similarity_raw=0.80,
        spatial_roughness_threshold=100.0,
        exclude_rough=True,
    )

    assignments = result['assignments']
    summary = result['summary']
    assert assignments['cluster_name'].str.fullmatch(r'C\d+').all()
    assert summary['is_clustered'].all()
    assert summary['meets_minimum_similarity'].all()
    assert (
        summary.loc[summary['n'] > 1, 'source_similarity_min'] >= 0.80 - 1e-12
    ).all()
    assert (
        summary.loc[summary['n'] > 1, 'temporal_similarity_min'] >= 0.80 - 1e-12
    ).all()
    assert result['meta']['fusion'] == 'minimum'
    assert result['meta']['algorithm'] == 'single_complete_linkage_threshold'


def test_combined_clustering_keeps_rough_items_as_plot_compatible_qc_rest() -> None:
    result = build_combined_outputs(
        _residue_promotion_items(noisy_third_cluster=True),
        minimum_similarity_raw=0.80,
        spatial_roughness_threshold=DEFAULT_MAIN_SPATIAL_ROUGHNESS_MAX,
        exclude_rough=True,
    )

    assignments = result['assignments']
    rough_assignments = assignments.loc[assignments['is_rough']]
    assert not rough_assignments.empty
    assert rough_assignments['cluster_name'].eq('R1').all()
    assert not rough_assignments['is_clustered'].any()
    assert (
        assignments.loc[~assignments['is_rough'], 'cluster_name']
        .str.startswith('C')
        .all()
    )


def test_plot_medoid_similarity_supports_minimum_fusion() -> None:
    source_similarity = np.array([[1.0, 0.9], [0.9, 1.0]])
    temporal_similarity = np.array([[1.0, 0.4], [0.4, 1.0]])

    observed = build_medoid_similarity(
        source_similarity,
        temporal_similarity,
        fusion='minimum',
        source_weight=None,
    )

    assert np.array_equal(observed, temporal_similarity)


def _minimal_item_table_for_cache() -> dict:
    return {
        'subjects': np.array([1], dtype=np.int64),
        'branches': np.array([0], dtype=np.int64),
        'spatial_patterns': np.ones((1, 3), dtype=np.float64),
        'source_magnitudes': np.ones((1, 4), dtype=np.float64),
        'temporal_patterns': np.ones((1, 5), dtype=np.float64),
        'temporal_spectra_mag': np.ones((1, 6), dtype=np.float64),
        'temporal_spectrum_freqs': np.arange(6, dtype=np.float64),
        'demean_temporal_filters': True,
    }


def test_item_stat_cache_requires_physical_sensor_units(tmp_path: Path) -> None:
    cache_path = tmp_path / 'items.npz'
    items = _minimal_item_table_for_cache()
    load_kwargs = dict(
        item_stats_npz=cache_path,
        subjects=[1],
        n_branches=1,
        target_fs=100.0,
        demean_temporal_filters=True,
    )
    rerun = 'lisa-combined-cluster-branch-interpretations'
    save_item_table_npz(items, cache_path)
    loaded = load_cached_item_table(**load_kwargs)
    np.testing.assert_array_equal(loaded['subjects'], [1])
    np.testing.assert_array_equal(loaded['branches'], [0])

    with np.load(cache_path, allow_pickle=False) as data:
        payload = {key: np.array(data[key]) for key in data.files}
    payload.pop('sensor_units')
    np.savez(cache_path, **payload)
    with pytest.raises(ValueError, match=rerun):
        load_cached_item_table(**load_kwargs)

    save_item_table_npz(items, cache_path)
    with np.load(cache_path, allow_pickle=False) as data:
        payload = {key: np.array(data[key]) for key in data.files}
    payload['sensor_units'] = np.asarray('normalized')
    np.savez(cache_path, **payload)
    with pytest.raises(ValueError, match=rerun):
        load_cached_item_table(**load_kwargs)
