from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lisa.plots.plot_metrics import load_metrics, parse_arguments


def _write_k_run(root: Path, run_group: str = '2conv-seed42-paper') -> Path:
    run_dir = root / run_group / 'k25-run'
    run_dir.mkdir(parents=True)
    (run_dir / 'config.json').write_text(
        json.dumps({'n_channels_unmix': 25, 'seed': 42}),
        encoding='utf-8',
    )
    pd.DataFrame(
        {
            'epoch': [0, 1, 2],
            'loss_val': [1.0, 0.2, 0.8],
            'top1s_val': [0.10, 0.40, 0.90],
            'top10s_val': [0.20, 0.50, 0.95],
            'top1s_final_test': [np.nan, 0.70, np.nan],
            'top10s_final_test': [np.nan, 0.80, np.nan],
        }
    ).to_csv(run_dir / 'metrics.csv', index=False)
    return run_dir


def test_parse_arguments_validation_metrics_default_false():
    args = parse_arguments(['--run-group', '2conv-seed42-paper'])
    assert args.validation_metrics is False

    args = parse_arguments(
        ['--run-group', '2conv-seed42-paper', '--validation-metrics']
    )
    assert args.validation_metrics is True


def test_best_epoch_uses_min_loss_val_for_test_or_validation(tmp_path: Path):
    _write_k_run(tmp_path)

    default_df, _, _ = load_metrics(
        experiments_root=str(tmp_path),
        run_groups=['2conv-seed42-paper'],
        epochs=('best',),
    )
    assert default_df['Top 1'].iloc[0] == pytest.approx(0.70)
    assert default_df['Top 10'].iloc[0] == pytest.approx(0.80)
    assert default_df['Epoch'].iloc[0] == 'Best validation checkpoint'

    val_df, _, _ = load_metrics(
        experiments_root=str(tmp_path),
        run_groups=['2conv-seed42-paper'],
        epochs=('best',),
        validation_metrics=True,
    )
    assert val_df['Top 1'].iloc[0] == pytest.approx(0.40)
    assert val_df['Top 10'].iloc[0] == pytest.approx(0.50)
    assert val_df['Epoch'].iloc[0] == 'Best validation checkpoint'
    # Must not pick the epoch that maximizes validation Top-1 (epoch 2).
    assert val_df['Top 1'].iloc[0] != pytest.approx(0.90)

    numeric_df, _, _ = load_metrics(
        experiments_root=str(tmp_path),
        run_groups=['2conv-seed42-paper'],
        epochs=(2,),
        validation_metrics=True,
    )
    assert numeric_df['Top 1'].iloc[0] == pytest.approx(0.90)
    assert numeric_df['Top 10'].iloc[0] == pytest.approx(0.95)
