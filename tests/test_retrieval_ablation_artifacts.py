from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lisa.plots.retrieval_ablation_artifacts import (
    RunSpec,
    expand_label_overrides,
    expand_run_group_template,
    find_run_dir_and_config,
    load_results,
    parse_args,
    seed_template_fields,
    summarize_multiseed_results,
)


def _base_config(seed: int, **updates) -> dict:
    config = {
        'n_channels_unmix': 25,
        'seed': seed,
        'run_name': f'run-seed{seed}-25branches',
        'use_spatial_attention': '3D',
        'use_unmixing_layer': True,
        'use_subject_layer': True,
        'temporal_filter_kernel_size': 15,
        'n_channels_attention': 24,
    }
    config.update(updates)
    return config


def _write_run(
    root: Path,
    run_group: str,
    config: dict,
    ranks: list[int],
) -> Path:
    run_dir = root / run_group / str(config['run_name'])
    run_dir.mkdir(parents=True)
    (run_dir / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    pd.DataFrame({'epoch': [0], 'top1s_final_test': [0.1]}).to_csv(
        run_dir / 'metrics.csv',
        index=False,
    )
    n = len(ranks)
    pd.DataFrame(
        {
            'rank_of_true': ranks,
            'session_id': [0] * n,
            'subject_id': [1] * n,
        }
    ).to_csv(run_dir / 'final_test_per_window.csv', index=False)
    return run_dir


def _load(
    root: Path,
    *,
    baseline: str,
    conditions: list[str],
    seeds: list[int] | None,
    family: str = 'ablation',
    labels: dict[str, str] | None = None,
):
    specs = [
        RunSpec(family=family, run_group=group, order=idx)
        for idx, group in enumerate(conditions)
    ]
    return load_results(
        experiments_root=root,
        baseline_spec=RunSpec(
            family='baseline',
            run_group=baseline,
            order=-1,
            label='Full model',
            method='Baseline',
            plot_label='Full model',
        ),
        specs=specs,
        n_branches=25,
        label_overrides=labels or {},
        plot_label_overrides={},
        audit_training_artifacts=False,
        seeds=seeds,
    )


def test_seed_template_expansion_accepts_only_seed():
    assert seed_template_fields('2conv-seed{seed}-paper') == ['seed']
    assert seed_template_fields('2conv-seed42-paper') == []
    assert (
        expand_run_group_template('2conv-seed{seed}-paper', 42) == '2conv-seed42-paper'
    )
    assert expand_run_group_template('2conv-seed42-paper', 43) == '2conv-seed42-paper'

    with pytest.raises(ValueError, match=r'only use the \{seed\} placeholder'):
        expand_run_group_template('2conv-seed{foo}-paper', 42)
    with pytest.raises(ValueError, match=r'only use the \{seed\} placeholder'):
        expand_run_group_template('pca{dim}-seed{seed}-paper', 42)
    with pytest.raises(ValueError, match='Invalid run-group template'):
        expand_run_group_template('2conv-seed{seed', 42)


def test_parse_args_expands_seed_templates_and_label_overrides():
    args = parse_args(
        [
            '--baseline-run-group',
            '2conv-seed{seed}-paper',
            '--n-branches',
            '25',
            '--ablation-run-groups',
            'ablate-no3d-2conv-seed{seed}-paper',
            '--seeds',
            '42',
            '43',
            '--label',
            'ablate-no3d-2conv-seed{seed}-paper=No 3D',
        ]
    )
    assert args.seeds == [42, 43]
    assert args.label_overrides['ablate-no3d-2conv-seed{seed}-paper'] == 'No 3D'
    expanded = expand_label_overrides(args.label_overrides, args.seeds)
    assert expanded['ablate-no3d-2conv-seed42-paper'] == 'No 3D'
    assert expanded['ablate-no3d-2conv-seed43-paper'] == 'No 3D'

    with pytest.raises(ValueError, match=r'\[a-zA-Z0-9\._-\]'):
        parse_args(
            [
                '--baseline-run-group',
                '2conv-seed{seed}-paper',
                '--n-branches',
                '25',
                '--ablation-run-groups',
                'ablate-no3d-2conv-seed{seed}-paper',
            ]
        )
    with pytest.raises(SystemExit):
        parse_args(
            [
                '--baseline-run-group',
                '2conv-seed{foo}-paper',
                '--n-branches',
                '25',
                '--ablation-run-groups',
                'ablate-no3d-2conv-seed{seed}-paper',
                '--seeds',
                '42',
            ]
        )


def test_single_run_mode_still_requires_existing_runs(tmp_path: Path):
    _write_run(tmp_path, '2conv-seed42-paper', _base_config(42), [0, 0, 9])
    _write_run(
        tmp_path,
        'ablate-no3d-2conv-seed42-paper',
        _base_config(42, use_spatial_attention='2D'),
        [9, 9, 9],
    )
    df, _diffs = _load(
        tmp_path,
        baseline='2conv-seed42-paper',
        conditions=['ablate-no3d-2conv-seed42-paper'],
        seeds=None,
    )
    assert len(df) == 2
    ablation = df[df['family'] == 'ablation'].iloc[0]
    baseline = df[df['family'] == 'baseline'].iloc[0]
    assert ablation['delta_top1'] == pytest.approx(ablation['top1'] - baseline['top1'])

    with pytest.raises(FileNotFoundError):
        _load(
            tmp_path,
            baseline='2conv-seed42-paper',
            conditions=['missing-ablate-2conv-seed42-paper'],
            seeds=None,
        )


def test_missing_requested_seed_warns_and_skips(tmp_path: Path):
    for seed, ranks in ((42, [0, 0, 9]), (43, [0, 5, 5])):
        _write_run(
            tmp_path,
            f'2conv-seed{seed}-paper',
            _base_config(seed),
            ranks,
        )
        _write_run(
            tmp_path,
            f'ablate-no3d-2conv-seed{seed}-paper',
            _base_config(seed, use_spatial_attention='2D'),
            [9, 9, 9],
        )

    with pytest.warns(UserWarning, match=r'template .* seed=44'):
        df, _diffs = _load(
            tmp_path,
            baseline='2conv-seed{seed}-paper',
            conditions=['ablate-no3d-2conv-seed{seed}-paper'],
            seeds=[42, 43, 44],
        )

    assert sorted(df['seed'].unique()) == [42, 43]
    assert set(df['family']) == {'baseline', 'ablation'}


def test_same_seed_deltas_use_matched_baseline(tmp_path: Path):
    _write_run(tmp_path, '2conv-seed42-paper', _base_config(42), [0, 0, 9])
    _write_run(tmp_path, '2conv-seed43-paper', _base_config(43), [0, 5, 5])
    _write_run(
        tmp_path,
        'ablate-no3d-2conv-seed42-paper',
        _base_config(42, use_spatial_attention='2D'),
        [9, 9, 9],
    )
    _write_run(
        tmp_path,
        'ablate-no3d-2conv-seed43-paper',
        _base_config(43, use_spatial_attention='2D'),
        [0, 0, 0],
    )

    df, diffs = _load(
        tmp_path,
        baseline='2conv-seed{seed}-paper',
        conditions=['ablate-no3d-2conv-seed{seed}-paper'],
        seeds=[42, 43],
    )
    ablation = df[df['family'] == 'ablation'].set_index('seed')
    baseline = df[df['family'] == 'baseline'].set_index('seed')

    assert ablation.loc[42, 'delta_top1'] == pytest.approx(
        ablation.loc[42, 'top1'] - baseline.loc[42, 'top1']
    )
    assert ablation.loc[43, 'delta_top1'] == pytest.approx(
        ablation.loc[43, 'top1'] - baseline.loc[43, 'top1']
    )
    assert ablation.loc[42, 'delta_top1'] != pytest.approx(
        ablation.loc[42, 'top1'] - baseline.loc[43, 'top1']
    )
    assert ablation.loc[43, 'delta_top1'] != pytest.approx(
        ablation.loc[43, 'top1'] - baseline.loc[42, 'top1']
    )
    assert set(diffs['seed']) == {42, 43}


def test_config_mismatch_still_raises(tmp_path: Path):
    _write_run(tmp_path, '2conv-seed42-paper', _base_config(42), [0, 0, 9])
    _write_run(
        tmp_path,
        'ablate-no3d-2conv-seed42-paper',
        _base_config(42, use_spatial_attention='2D', n_temporal_module_blocks=5),
        [9, 9, 9],
    )

    with pytest.raises(RuntimeError, match='n_temporal_module_blocks'):
        _load(
            tmp_path,
            baseline='2conv-seed{seed}-paper',
            conditions=['ablate-no3d-2conv-seed{seed}-paper'],
            seeds=[42],
        )


def test_summarize_multiseed_results_mean_min_max_n_seeds():
    df = pd.DataFrame(
        [
            {
                'family': 'ablation',
                'order': 0,
                'label': 'No 3D',
                'method': 'Ablation',
                'dim': None,
                'seed': 42,
                'top1': 10.0,
                'top10': 20.0,
                'delta_top1': -1.0,
                'delta_top10': -2.0,
            },
            {
                'family': 'ablation',
                'order': 0,
                'label': 'No 3D',
                'method': 'Ablation',
                'dim': None,
                'seed': 43,
                'top1': 30.0,
                'top10': 40.0,
                'delta_top1': -5.0,
                'delta_top10': -8.0,
            },
        ]
    )
    summary = summarize_multiseed_results(df)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row['n_seeds'] == 2
    assert row['top1_mean'] == pytest.approx(20.0)
    assert row['top1_min'] == pytest.approx(10.0)
    assert row['top1_max'] == pytest.approx(30.0)
    assert row['top10_mean'] == pytest.approx(30.0)
    assert row['top10_min'] == pytest.approx(20.0)
    assert row['top10_max'] == pytest.approx(40.0)
    assert row['delta_top1_mean'] == pytest.approx(-3.0)
    assert row['delta_top1_min'] == pytest.approx(-5.0)
    assert row['delta_top1_max'] == pytest.approx(-1.0)
    assert row['delta_top10_mean'] == pytest.approx(-5.0)
    assert row['delta_top10_min'] == pytest.approx(-8.0)
    assert row['delta_top10_max'] == pytest.approx(-2.0)


def test_no_valid_baseline_seed_raises(tmp_path: Path):
    with pytest.warns(UserWarning, match=r'seed=42'):
        with pytest.raises(RuntimeError, match='No valid baseline'):
            _load(
                tmp_path,
                baseline='2conv-seed{seed}-paper',
                conditions=['ablate-no3d-2conv-seed{seed}-paper'],
                seeds=[42],
            )


def test_condition_with_zero_paired_seeds_is_omitted(tmp_path: Path):
    _write_run(tmp_path, '2conv-seed42-paper', _base_config(42), [0, 0, 9])

    with pytest.warns(UserWarning, match='Omitting condition'):
        df, _diffs = _load(
            tmp_path,
            baseline='2conv-seed{seed}-paper',
            conditions=['ablate-no3d-2conv-seed{seed}-paper'],
            seeds=[42],
        )

    assert set(df['family']) == {'baseline'}


def test_feature_missing_seed_uses_remaining_seeds(tmp_path: Path):
    for seed, ranks in ((42, [0, 0, 9]), (43, [0, 5, 5])):
        _write_run(tmp_path, f'2conv-seed{seed}-paper', _base_config(seed), ranks)
    _write_run(
        tmp_path,
        'pca16-2conv-seed42-paper',
        _base_config(42, feature_reduction='pca', feature_reduction_dim=16),
        [9, 9, 9],
    )

    with pytest.warns(UserWarning, match=r'pca16-2conv-seed\{seed\}-paper.*seed=43'):
        df, _diffs = _load(
            tmp_path,
            baseline='2conv-seed{seed}-paper',
            conditions=['pca16-2conv-seed{seed}-paper'],
            seeds=[42, 43],
            family='feature',
        )

    feature = df[df['family'] == 'feature']
    assert list(feature['seed']) == [42]
    assert feature.iloc[0]['delta_top1'] == pytest.approx(
        feature.iloc[0]['top1']
        - df.loc[(df['family'] == 'baseline') & (df['seed'] == 42), 'top1'].iloc[0]
    )
    summary = summarize_multiseed_results(feature)
    assert summary.iloc[0]['n_seeds'] == 1


def test_find_run_dir_selects_requested_seed_in_shared_group(tmp_path: Path):
    group = 'shared-group'
    _write_run(tmp_path, group, _base_config(42), [0, 0, 9])
    _write_run(tmp_path, group, _base_config(43), [0, 5, 5])

    run_dir, config = find_run_dir_and_config(tmp_path, group, 25, seed=43)
    assert config['seed'] == 43
    assert run_dir.name == 'run-seed43-25branches'

    with pytest.raises(RuntimeError, match='found'):
        find_run_dir_and_config(tmp_path, group, 25)
