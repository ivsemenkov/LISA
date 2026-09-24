from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from lisa.plots.analyze_interpretation_reproducibility import (  # noqa: E402
    FEATURE_KEYS,
    PATTERN_ITEM_KEYS,
    PATTERN_VIEWS,
    SEED_VIEW_MATCHING_CORRELATION,
    SEED_VIEWS,
    SIGN_AMBIGUOUS_VIEWS,
    compare_item_conditions,
    compare_seed_view_correspondence,
    load_physical_item_stats,
    main,
)
from lisa.plots.branch_interpretation import save_item_table_npz  # noqa: E402
from lisa.plots.plot_branch_interpretations import (  # noqa: E402
    load_cached_item_table,
)
import lisa.plots.analyze_interpretation_reproducibility as repro  # noqa: E402


OLD_PATTERN_VIEWS = ('spatial_haufe', 'temporal_pattern', 'log_spectrum')
OLD_STORY_COLUMNS = (
    'spatial_haufe_pearson',
    'temporal_pattern_pearson',
    'log_spectrum_pearson',
)
OLD_STABILITY_COLUMNS = (
    'spatial_haufe_abs_pearson',
    'spatial_haufe_signed_pearson',
    'temporal_pattern_abs_pearson',
    'temporal_pattern_signed_pearson',
    'log_spectrum_pearson',
)


def _signed_pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def _branch_vectors() -> dict[str, np.ndarray]:
    return {
        'spatial': np.array(
            [
                [1.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        'source': np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
            ],
            dtype=np.float64,
        ),
        'temporal': np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [5.0, 4.0, 3.0, 2.0, 1.0],
            ],
            dtype=np.float64,
        ),
        'spectra_mag': np.array(
            [
                [4.0, 3.0, 2.0, 1.0],
                [1.0, 1.0, 2.0, 5.0],
            ],
            dtype=np.float64,
        ),
    }


def _cache_items(
    spatial: np.ndarray,
    source: np.ndarray,
    temporal: np.ndarray,
    spectra_mag: np.ndarray,
    *,
    subject: int = 1,
) -> dict[str, np.ndarray | bool]:
    n_branches = int(spatial.shape[0])
    return {
        'subjects': np.full(n_branches, int(subject), dtype=int),
        'branches': np.arange(n_branches, dtype=int),
        'spatial_patterns': np.asarray(spatial, dtype=np.float64),
        'source_magnitudes': np.asarray(source, dtype=np.float64),
        'temporal_patterns': np.asarray(temporal, dtype=np.float64),
        'temporal_spectra_mag': np.asarray(spectra_mag, dtype=np.float64),
        'temporal_spectrum_freqs': np.arange(
            spectra_mag.shape[1], dtype=np.float64
        ),
        'demean_temporal_filters': True,
    }


def _analysis_items(
    spatial: np.ndarray,
    source: np.ndarray,
    temporal: np.ndarray,
    spectra_log: np.ndarray,
    *,
    subject: int = 1,
) -> dict[str, np.ndarray | bool | int]:
    n_branches = int(spatial.shape[0])
    return {
        'subjects': np.full(n_branches, int(subject), dtype=int),
        'branches': np.arange(n_branches, dtype=int),
        'spatial_patterns': np.asarray(spatial, dtype=np.float64),
        'source_magnitudes': np.asarray(source, dtype=np.float64),
        'temporal_patterns': np.asarray(temporal, dtype=np.float64),
        'temporal_spectra_log': np.asarray(spectra_log, dtype=np.float64),
        'temporal_spectrum_freqs': np.arange(
            spectra_log.shape[1], dtype=np.float64
        ),
        'n_branches': n_branches,
        'n_subjects': 1,
        'demean_temporal_filters': True,
    }


def _record(
    items: dict[str, np.ndarray | bool | int],
    *,
    label: str,
    story_id: int,
    session: int,
    train_seed: int,
) -> dict[str, object]:
    return {
        'label': label,
        'story_id': int(story_id),
        'session': int(session),
        'train_seed': int(train_seed),
        'items': items,
    }


def test_cached_item_table_exposes_producer_source_magnitudes(
    tmp_path: Path,
) -> None:
    vectors = _branch_vectors()
    items = _cache_items(
        vectors['spatial'],
        vectors['source'],
        vectors['temporal'],
        vectors['spectra_mag'],
    )
    cache_path = tmp_path / 'item_stats.npz'
    save_item_table_npz(items, cache_path)
    with np.load(cache_path, allow_pickle=False) as data:
        assert 'item_source_magnitudes' in data.files
        np.testing.assert_array_equal(
            data['item_source_magnitudes'], items['source_magnitudes']
        )

    loaded = load_cached_item_table(
        item_stats_npz=cache_path,
        subjects=[1],
        n_branches=2,
        target_fs=100.0,
        demean_temporal_filters=True,
    )
    np.testing.assert_allclose(
        loaded['source_magnitudes'], items['source_magnitudes']
    )

    physical = load_physical_item_stats(cache_path, meg_sr=100.0)
    np.testing.assert_allclose(
        physical['source_magnitudes'], items['source_magnitudes']
    )


def test_missing_source_magnitudes_in_npz_fails_loudly(tmp_path: Path) -> None:
    vectors = _branch_vectors()
    cache_path = tmp_path / 'item_stats.npz'
    save_item_table_npz(
        _cache_items(
            vectors['spatial'],
            vectors['source'],
            vectors['temporal'],
            vectors['spectra_mag'],
        ),
        cache_path,
    )
    with np.load(cache_path, allow_pickle=False) as data:
        payload = {key: np.array(data[key]) for key in data.files}
    payload.pop('item_source_magnitudes')
    np.savez(cache_path, **payload)

    with pytest.raises(ValueError, match='item_source_magnitudes'):
        load_physical_item_stats(cache_path, meg_sr=100.0)


def test_story_and_session_comparisons_include_signed_source_space() -> None:
    vectors = _branch_vectors()
    spectra_log = np.log(vectors['spectra_mag'] + 1e-12)
    items_a = _analysis_items(
        vectors['spatial'],
        vectors['source'],
        vectors['temporal'],
        spectra_log,
    )
    items_b = _analysis_items(
        vectors['spatial'],
        vectors['source'][:, ::-1],
        vectors['temporal'],
        spectra_log,
    )
    record_a = _record(
        items_a, label='story0', story_id=0, session=0, train_seed=1
    )
    record_b = _record(
        items_b, label='story1', story_id=1, session=0, train_seed=1
    )
    table, summaries = compare_item_conditions(
        record_a, record_b, kind='stories'
    )

    for column in OLD_STORY_COLUMNS:
        assert column in table.columns
        np.testing.assert_allclose(table[column].to_numpy(), 1.0)
    assert 'source_magnitudes_pearson' in table.columns
    expected_source = np.array(
        [
            _signed_pearson(vectors['source'][0], vectors['source'][0, ::-1]),
            _signed_pearson(vectors['source'][1], vectors['source'][1, ::-1]),
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        table.sort_values('branch')['source_magnitudes_pearson'].to_numpy(),
        expected_source,
    )
    np.testing.assert_array_less(table['source_magnitudes_pearson'], 0.0)

    features = [row['feature'] for row in summaries]
    assert features == [
        'spatial_haufe',
        'temporal_pattern',
        'log_spectrum',
        'source_magnitudes',
    ]

    session_table, session_summaries = compare_item_conditions(
        _record(items_a, label='ses0', story_id=0, session=0, train_seed=1),
        _record(items_b, label='ses1', story_id=0, session=1, train_seed=1),
        kind='sessions',
    )
    assert 'source_magnitudes_pearson' in session_table.columns
    np.testing.assert_allclose(
        session_table.sort_values('branch')['spatial_haufe_pearson'].to_numpy(),
        1.0,
    )
    np.testing.assert_allclose(
        session_table.sort_values('branch')['source_magnitudes_pearson'].to_numpy(),
        expected_source,
    )
    assert [row['feature'] for row in session_summaries] == features


def test_story_cli_writes_source_space_columns(tmp_path: Path) -> None:
    vectors = _branch_vectors()
    path_a = tmp_path / 'story0.npz'
    path_b = tmp_path / 'story1.npz'
    save_item_table_npz(
        _cache_items(
            vectors['spatial'],
            vectors['source'],
            vectors['temporal'],
            vectors['spectra_mag'],
        ),
        path_a,
    )
    save_item_table_npz(
        _cache_items(
            vectors['spatial'],
            vectors['source'][:, ::-1],
            vectors['temporal'],
            vectors['spectra_mag'],
        ),
        path_b,
    )
    out_dir = tmp_path / 'out'
    main(
        [
            '--item-stats',
            str(path_a),
            str(path_b),
            '--labels',
            'story0',
            'story1',
            '--story-ids',
            '0',
            '1',
            '--sessions',
            '0',
            '0',
            '--train-seeds',
            '1',
            '1',
            '--compare',
            'stories',
            '--out-dir',
            str(out_dir),
        ]
    )
    table = pd.read_csv(out_dir / 'story_item_similarities.csv')
    summaries = pd.read_csv(out_dir / 'story_summaries.csv')
    for column in OLD_STORY_COLUMNS:
        assert column in table.columns
    assert 'source_magnitudes_pearson' in table.columns
    assert list(summaries['feature']) == [
        'spatial_haufe',
        'temporal_pattern',
        'log_spectrum',
        'source_magnitudes',
    ]


def test_missing_and_incompatible_source_magnitudes_fail_loudly() -> None:
    vectors = _branch_vectors()
    spectra_log = np.log(vectors['spectra_mag'] + 1e-12)
    items_a = _analysis_items(
        vectors['spatial'],
        vectors['source'],
        vectors['temporal'],
        spectra_log,
    )
    items_missing = dict(items_a)
    del items_missing['source_magnitudes']
    with pytest.raises(ValueError, match='required source_magnitudes'):
        compare_item_conditions(
            _record(items_a, label='a', story_id=0, session=0, train_seed=1),
            _record(
                items_missing, label='b', story_id=1, session=0, train_seed=1
            ),
            kind='stories',
        )

    items_wrong_dim = dict(items_a)
    items_wrong_dim['source_magnitudes'] = vectors['source'][:, :-1]
    with pytest.raises(ValueError, match='source-space vertex counts differ'):
        compare_item_conditions(
            _record(items_a, label='a', story_id=0, session=0, train_seed=1),
            _record(
                items_wrong_dim, label='b', story_id=1, session=0, train_seed=1
            ),
            kind='stories',
        )

    items_nan = dict(items_a)
    source_nan = vectors['source'].copy()
    source_nan[0, 0] = np.nan
    items_nan['source_magnitudes'] = source_nan
    with pytest.raises(ValueError, match='Non-finite'):
        compare_item_conditions(
            _record(items_a, label='a', story_id=0, session=0, train_seed=1),
            _record(items_nan, label='b', story_id=1, session=0, train_seed=1),
            kind='stories',
        )


def test_source_seed_view_uses_signed_pearson_and_own_hungarian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(repro, 'N_SUBJECTS', 1)
    vectors = _branch_vectors()
    spectra_log = np.log(vectors['spectra_mag'] + 1e-12)
    items_a = _analysis_items(
        vectors['spatial'],
        vectors['source'],
        vectors['temporal'],
        spectra_log,
    )
    items_b = _analysis_items(
        -vectors['spatial'],
        vectors['source'][::-1],
        vectors['temporal'],
        spectra_log,
    )
    records = [
        _record(items_a, label='s1', story_id=0, session=0, train_seed=1),
        _record(items_b, label='s2', story_id=0, session=0, train_seed=2),
    ]
    learned_filters = {
        1: {
            'spatial': vectors['spatial'][np.newaxis],
            'temporal': vectors['temporal'],
        },
        2: {
            'spatial': -vectors['spatial'][np.newaxis],
            'temporal': vectors['temporal'],
        },
    }
    aggregated, _participant, assignments, quality, agreement, stability = (
        compare_seed_view_correspondence(records, learned_filters)
    )

    assert list(SEED_VIEWS[:5]) == [
        'spatial_filters',
        'temporal_filters',
        'spatial_haufe',
        'temporal_pattern',
        'log_spectrum',
    ]
    assert SEED_VIEWS[-1] == 'source_magnitudes'
    assert 'source_magnitudes' not in SIGN_AMBIGUOUS_VIEWS
    assert tuple(view for _name, view in FEATURE_KEYS[-1:]) == (
        'source_magnitudes',
    )
    assert PATTERN_ITEM_KEYS['source_magnitudes'] == 'source_magnitudes'
    assert PATTERN_VIEWS[-1] == 'source_magnitudes'
    for view in OLD_PATTERN_VIEWS:
        assert view in PATTERN_VIEWS

    source_matrix = aggregated['source_magnitudes__s1__s2']
    signed_source_00 = _signed_pearson(
        vectors['source'][0], vectors['source'][1]
    )
    abs_source_00 = abs(signed_source_00)
    assert signed_source_00 < 0.0
    np.testing.assert_allclose(source_matrix[0, 0], signed_source_00)
    assert not np.isclose(source_matrix[0, 0], abs_source_00)
    assert 'source_magnitudes__signed__s1__s2' not in aggregated
    assert 'log_spectrum__signed__s1__s2' not in aggregated

    spatial_abs = aggregated['spatial_haufe__s1__s2']
    spatial_signed = aggregated['spatial_haufe__signed__s1__s2']
    np.testing.assert_allclose(spatial_abs[0, 0], 1.0)
    np.testing.assert_allclose(spatial_signed[0, 0], -1.0)
    assert SEED_VIEW_MATCHING_CORRELATION['source_magnitudes'] == 'signed'
    assert SEED_VIEW_MATCHING_CORRELATION['spatial_haufe'] == 'absolute'
    assert SEED_VIEW_MATCHING_CORRELATION['log_spectrum'] == 'signed'

    assert set(assignments['view']) == set(SEED_VIEWS)
    source_matches = assignments.loc[
        assignments['view'] == 'source_magnitudes'
    ].sort_values('branch_a')
    spatial_matches = assignments.loc[
        assignments['view'] == 'spatial_haufe'
    ].sort_values('branch_a')
    assert list(source_matches['correlation_kind']) == ['signed', 'signed']
    assert list(spatial_matches['correlation_kind']) == [
        'absolute',
        'absolute',
    ]
    assert list(source_matches['branch_b']) == [2, 1]
    assert list(spatial_matches['branch_b']) == [1, 2]

    source_vs_spatial = agreement.loc[
        (agreement['view_a'] == 'spatial_haufe')
        & (agreement['view_b'] == 'source_magnitudes')
    ]
    assert len(source_vs_spatial) == 1
    assert bool(source_vs_spatial['assignments_equal'].iloc[0]) is False

    for column in OLD_STABILITY_COLUMNS:
        assert column in stability.columns
    assert 'source_magnitudes_pearson' in stability.columns
    source_matched = stability.loc[
        stability['matching_view'] == 'source_magnitudes'
    ].sort_values('branch_a')
    np.testing.assert_allclose(
        source_matched['source_magnitudes_pearson'].to_numpy(),
        [
            _signed_pearson(vectors['source'][0], vectors['source'][0]),
            _signed_pearson(vectors['source'][1], vectors['source'][1]),
        ],
    )
    spatial_matched = stability.loc[
        stability['matching_view'] == 'spatial_haufe'
    ].sort_values('branch_a')
    np.testing.assert_allclose(
        spatial_matched['spatial_haufe_abs_pearson'].to_numpy(), 1.0
    )
    np.testing.assert_allclose(
        spatial_matched['spatial_haufe_signed_pearson'].to_numpy(), -1.0
    )
    assert set(quality['view']) == set(SEED_VIEWS)


def test_physical_unit_spatial_haufe_pearson_is_not_rejected_as_constant() -> None:
    tesla = 1e-13
    vectors = _branch_vectors()
    spatial = vectors['spatial'] * tesla
    centered = spatial - spatial.mean(axis=1, keepdims=True)
    centered_norms = np.linalg.norm(centered, axis=1)
    assert np.all(centered_norms > 0.0)
    assert np.all(centered_norms <= 1e-12)

    spectra_log = np.log(vectors['spectra_mag'] + 1e-12)
    items = _analysis_items(
        spatial,
        vectors['source'],
        vectors['temporal'],
        spectra_log,
    )
    table, _summaries = compare_item_conditions(
        _record(items, label='story0', story_id=0, session=0, train_seed=1),
        _record(items, label='story1', story_id=1, session=0, train_seed=1),
        kind='stories',
    )
    np.testing.assert_allclose(
        table.sort_values('branch')['spatial_haufe_pearson'].to_numpy(),
        1.0,
    )

    tiny = spatial * 1e-7
    large = spatial * 1e20
    items_tiny = _analysis_items(
        tiny,
        vectors['source'],
        vectors['temporal'],
        spectra_log,
    )
    items_large = _analysis_items(
        large,
        vectors['source'],
        vectors['temporal'],
        spectra_log,
    )
    scaled_table, _ = compare_item_conditions(
        _record(items_tiny, label='story0', story_id=0, session=0, train_seed=1),
        _record(items_large, label='story1', story_id=1, session=0, train_seed=1),
        kind='stories',
    )
    np.testing.assert_allclose(
        scaled_table.sort_values('branch')['spatial_haufe_pearson'].to_numpy(),
        1.0,
    )
