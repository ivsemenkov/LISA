import matplotlib.pyplot as plt
import numpy as np
import pytest

from lisa.plots.plot_spatial_topography_svd import (
    TopographyStack,
    align_topographies_to_forward,
    compute_svd_variants,
    g3_to_g2,
    plot_singular_value_stems,
    rap_music_scan,
)
from lisa.plots.temporal_filter_utils import demean_temporal_filters


def test_compute_svd_variants_keeps_raw_and_row_l2_separate():
    spatial_filters = np.asarray(
        [
            [10.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    spatial_patterns = np.asarray(
        [
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    stack = TopographyStack(
        spatial_filters=spatial_filters,
        spatial_patterns=spatial_patterns,
        item_subjects=np.asarray([1, 1], dtype=int),
        item_branches=np.asarray([1, 2], dtype=int),
        info={'ch_names': ['a', 'b', 'c']},
        n_branches=2,
    )

    variants = compute_svd_variants(stack=stack, energy_threshold=0.95)
    assert len(variants) == 2
    raw_variant, normalized_variant = variants

    assert raw_variant.key == 'raw'
    assert normalized_variant.key == 'row_l2'
    np.testing.assert_allclose(raw_variant.spatial_filters, spatial_filters)
    np.testing.assert_allclose(raw_variant.spatial_patterns, spatial_patterns)
    np.testing.assert_allclose(
        normalized_variant.spatial_filters,
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    np.testing.assert_allclose(
        normalized_variant.spatial_patterns,
        np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    assert raw_variant.filter_row_norms is None
    assert raw_variant.pattern_row_norms is None
    np.testing.assert_allclose(normalized_variant.filter_row_norms, [10.0, 1.0])
    np.testing.assert_allclose(normalized_variant.pattern_row_norms, [5.0, 2.0])
    assert raw_variant.filter_result.singular_values[0] == pytest.approx(10.0)
    assert normalized_variant.filter_result.singular_values[0] == pytest.approx(1.0)


def test_compute_svd_variants_adds_demeaned_temporal_filter_outputs():
    spatial_filters = np.asarray([[2.0, 0.0], [0.0, 1.0]])
    demeaned_patterns = np.asarray([[0.5, 0.5], [1.0, -1.0]])
    stack = TopographyStack(
        spatial_filters=spatial_filters,
        spatial_patterns=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        demeaned_temporal_spatial_patterns=demeaned_patterns,
        item_subjects=np.asarray([1, 1], dtype=int),
        item_branches=np.asarray([1, 2], dtype=int),
        info={'ch_names': ['a', 'b']},
        n_branches=2,
    )

    variants = compute_svd_variants(stack=stack, energy_threshold=0.95)

    assert [variant.key for variant in variants] == [
        'raw',
        'row_l2',
        'dtf_raw',
        'dtf_row_l2',
    ]
    # DTF variants replace patterns only; filters stay on the original stack.
    np.testing.assert_allclose(variants[2].spatial_filters, spatial_filters)
    np.testing.assert_allclose(variants[2].spatial_patterns, demeaned_patterns)
    np.testing.assert_allclose(
        variants[3].spatial_filters,
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
    )
    np.testing.assert_allclose(
        np.linalg.norm(variants[3].spatial_patterns, axis=1),
        np.ones(2),
    )
    np.testing.assert_allclose(
        variants[3].pattern_row_norms,
        np.linalg.norm(demeaned_patterns, axis=1),
    )


def test_compute_svd_variants_rejects_mismatched_demeaned_pattern_shape():
    stack = TopographyStack(
        spatial_filters=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        spatial_patterns=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        demeaned_temporal_spatial_patterns=np.asarray([[1.0, 0.0]]),
        item_subjects=np.asarray([1, 1], dtype=int),
        item_branches=np.asarray([1, 2], dtype=int),
        info={'ch_names': ['a', 'b']},
        n_branches=2,
    )

    with pytest.raises(ValueError, match='must match original spatial'):
        compute_svd_variants(stack=stack, energy_threshold=0.95)


def test_demean_temporal_filters_forces_zero_dc_and_rejects_constant_rows():
    filters = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])

    demeaned = demean_temporal_filters(filters)

    np.testing.assert_allclose(demeaned.sum(axis=1), [0.0, 0.0])
    np.testing.assert_allclose(demeaned[0], [-1.0, 0.0, 1.0])

    with pytest.raises(ValueError, match='near-zero residual rows'):
        demean_temporal_filters(np.asarray([[1.0, 1.0, 1.0]]))


def test_align_topographies_to_forward_reorders_channels_and_rejects_mismatch():
    topographies = np.asarray([[1.0, 2.0, 3.0]])
    info = {'ch_names': ['a', 'b', 'c']}
    fwd = {'sol': {'row_names': ['c', 'a']}}

    aligned = align_topographies_to_forward(topographies, info, fwd)

    np.testing.assert_allclose(aligned, [[3.0, 1.0]])

    with pytest.raises(ValueError, match='Topography channel count'):
        align_topographies_to_forward(np.asarray([[1.0, 2.0]]), info, fwd)

    with pytest.raises(ValueError, match='channels not present'):
        align_topographies_to_forward(
            topographies,
            info,
            {'sol': {'row_names': ['missing']}},
        )


def test_g3_to_g2_skips_near_zero_tangential_gain_sites():
    gain = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )

    result = g3_to_g2(gain)

    assert result.n_source_sites == 2
    np.testing.assert_array_equal(result.valid_site_indices, [1])
    np.testing.assert_array_equal(result.invalid_site_indices, [0])
    assert result.gain_2d.shape == (2, 2)
    assert result.invalid_tangential_norms.shape == (1, 2)


def test_rap_music_scan_keeps_original_source_indices_after_skipping_sites():
    gain = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )

    result = rap_music_scan(
        data=np.eye(2),
        gain=gain,
        music_threshold=0.1,
    )

    assert result.indices == [1]
    assert result.n_source_sites == 2
    np.testing.assert_array_equal(result.initial_invalid_site_indices, [0])
    np.testing.assert_array_equal(result.union_invalid_site_indices, [0, 1])
    np.testing.assert_array_equal(result.iteration_invalid_counts, [1, 2])
    np.testing.assert_array_equal(result.iteration_valid_counts, [1, 0])


def test_plot_singular_value_stems_has_filter_and_pattern_panels():
    stack = TopographyStack(
        spatial_filters=np.asarray([[1.0, 0.0], [0.0, 0.5]]),
        spatial_patterns=np.asarray([[0.0, 2.0], [0.25, 0.0]]),
        item_subjects=np.asarray([1, 1], dtype=int),
        item_branches=np.asarray([1, 2], dtype=int),
        info={'ch_names': ['a', 'b']},
        n_branches=2,
    )
    raw_variant, _ = compute_svd_variants(stack=stack, energy_threshold=0.95)

    fig = plot_singular_value_stems(
        filter_result=raw_variant.filter_result,
        pattern_result=raw_variant.pattern_result,
        n_display=2,
    )

    assert len(fig.axes) == 2
    assert [ax.get_title() for ax in fig.axes] == ['Filters', 'Topographies']
    assert fig.axes[0].get_ylabel() == 'Singular value'
    plt.close(fig)

    with pytest.raises(ValueError, match='n_display must be positive'):
        plot_singular_value_stems(
            filter_result=raw_variant.filter_result,
            pattern_result=raw_variant.pattern_result,
            n_display=0,
        )
