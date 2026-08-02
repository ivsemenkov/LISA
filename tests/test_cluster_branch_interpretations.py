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
    validate_curated_assignments,
)
from lisa.plots.plot_branch_interpretations import (  # noqa: E402
    build_medoid_similarity,
    build_curated_cluster_specs,
    get_cluster_display_label,
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
