import matplotlib.pyplot as plt
import numpy as np
import pytest

from lisa.plots.plot_spatial_topography_svd import (
    TopographyStack,
    align_topographies_to_forward,
    compute_svd_result,
    compute_svd_variants,
    fit_svd_pattern_dipoles,
    g3_to_g2,
    plot_singular_value_stems,
    rap_music_scan,
    row_l2_normalize,
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


def test_svd_normalization_is_invariant_at_physical_sensor_scale():
    topographies = np.asarray(
        [
            [1.0, 2.0, -1.0, 0.5],
            [-2.0, 1.0, 3.0, -0.5],
            [0.5, -1.0, 2.0, 1.5],
        ]
    )
    physical = topographies * 1e-14

    normalized, norms = row_l2_normalize(topographies)
    normalized_physical, norms_physical = row_l2_normalize(physical)
    np.testing.assert_allclose(normalized_physical, normalized)
    np.testing.assert_allclose(norms_physical, norms * 1e-14)

    result = compute_svd_result(topographies, energy_threshold=0.95)
    result_physical = compute_svd_result(physical, energy_threshold=0.95)
    np.testing.assert_allclose(
        result_physical.singular_values,
        result.singular_values * 1e-14,
    )
    np.testing.assert_allclose(
        result_physical.right_singular_vectors,
        result.right_singular_vectors,
    )
    np.testing.assert_allclose(result_physical.explained_energy, result.explained_energy)
    np.testing.assert_allclose(result_physical.cumulative_energy, result.cumulative_energy)
    assert result_physical.n_components == result.n_components

    with pytest.raises(ValueError, match='zero topographies'):
        row_l2_normalize(np.zeros((1, 4)))
    with pytest.raises(ValueError, match='zero matrix'):
        compute_svd_result(np.zeros((1, 4)), energy_threshold=0.95)


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


def test_compute_svd_variants_can_select_raw_or_row_l2_only():
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

    raw_only = compute_svd_variants(
        stack=stack,
        energy_threshold=0.95,
        normalization='raw',
    )
    row_l2_only = compute_svd_variants(
        stack=stack,
        energy_threshold=0.95,
        normalization='row_l2',
    )

    assert [variant.key for variant in raw_only] == ['raw', 'dtf_raw']
    assert [variant.key for variant in row_l2_only] == ['row_l2', 'dtf_row_l2']
    with pytest.raises(ValueError, match='svd_normalization must be'):
        compute_svd_variants(stack=stack, energy_threshold=0.95, normalization='l2')


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


def test_pooled_sensor_space_svd_does_not_depend_on_info_geometry():
    spatial_filters = np.asarray([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]])
    spatial_patterns = np.asarray([[4.0, 0.0, 1.0], [0.5, 2.0, 0.0]])
    first = TopographyStack(
        spatial_filters=spatial_filters,
        spatial_patterns=spatial_patterns,
        item_subjects=np.asarray([1, 1], dtype=int),
        item_branches=np.asarray([1, 2], dtype=int),
        info={'ch_names': ['a', 'b', 'c'], 'dig': 'first'},
        n_branches=2,
    )
    second = TopographyStack(
        spatial_filters=spatial_filters.copy(),
        spatial_patterns=spatial_patterns.copy(),
        item_subjects=np.asarray([1, 1], dtype=int),
        item_branches=np.asarray([1, 2], dtype=int),
        info={'ch_names': ['a', 'b', 'c'], 'dig': 'second', 'dev_head_t': 'other'},
        n_branches=2,
    )

    first_variants = compute_svd_variants(stack=first, energy_threshold=0.95)
    second_variants = compute_svd_variants(stack=second, energy_threshold=0.95)

    assert [variant.key for variant in first_variants] == [
        variant.key for variant in second_variants
    ]
    for left, right in zip(first_variants, second_variants, strict=True):
        np.testing.assert_allclose(
            left.pattern_result.singular_values,
            right.pattern_result.singular_values,
        )
        np.testing.assert_allclose(
            left.pattern_result.right_singular_vectors,
            right.pattern_result.right_singular_vectors,
        )
        np.testing.assert_allclose(
            left.filter_result.right_singular_vectors,
            right.filter_result.right_singular_vectors,
        )


def test_localisation_uses_participant_specific_source_rr():
    from mne.transforms import Transform

    from lisa.plots.plot_spatial_topography_svd import (
        fsaverage_coords_from_head,
        source_index_hemisphere,
    )

    gain = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    info = {'ch_names': ['a', 'b']}
    variant = type(
        'Variant',
        (),
        {
            'key': 'raw',
            'pattern_result': type(
                'Result',
                (),
                {'right_singular_vectors': np.eye(2)},
            )(),
        },
    )()
    fwd_one = {
        'sol': {'data': gain, 'row_names': ['a', 'b']},
        'source_rr': np.array([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float64),
        'src': [{'nuse': 1}, {'nuse': 1}],
    }
    fwd_two = {
        'sol': {'data': gain, 'row_names': ['a', 'b']},
        'source_rr': np.array([[0.11, 0.0, 0.0], [0.12, 0.0, 0.0]], dtype=np.float64),
        'src': [{'nuse': 1}, {'nuse': 1}],
    }
    identity = Transform('head', 'mri', np.eye(4))
    first = fit_svd_pattern_dipoles(
        variant=variant,
        info=info,
        fwd=fwd_one,
        component_count=1,
        music_threshold=0.1,
    )
    second = fit_svd_pattern_dipoles(
        variant=variant,
        info=info,
        fwd=fwd_two,
        component_count=1,
        music_threshold=0.1,
    )
    first_coords = fsaverage_coords_from_head(
        fwd_one, np.asarray(first['index'][0], dtype=int), identity
    )
    second_coords = fsaverage_coords_from_head(
        fwd_two, np.asarray(second['index'][0], dtype=int), identity
    )
    assert not np.allclose(first_coords, second_coords)
    assert source_index_hemisphere(fwd_one, 0) == 'lh'
    assert source_index_hemisphere(fwd_one, 1) == 'rh'

    empty = fit_svd_pattern_dipoles(
        variant=variant,
        info=info,
        fwd=fwd_one,
        component_count=1,
        music_threshold=1.0,
    )
    assert empty['index'][0] == []
    assert empty['vals'][0] == []
    assert empty['coords'][0].shape == (0, 3)


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
