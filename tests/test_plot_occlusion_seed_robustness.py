from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lisa.plots.plot_occlusion_seed_robustness import (
    ANALYSIS_DIR,
    DEFAULT_SPLIT_THRESHOLD,
    build_effect_tables,
    discover_completed_result_dirs,
    load_occlusion_run,
    parse_arguments,
    pattern_output_dirname,
    plot_feature_group,
    split_features,
)


FEATURES = ('large_positive', 'small_positive', 'large_negative')


def test_pattern_is_required_and_other_inputs_have_defaults() -> None:
    with pytest.raises(SystemExit):
        parse_arguments([])

    args = parse_arguments(['initialization-*'])
    assert args.pattern == 'initialization-*'
    assert args.analysis_dir == ANALYSIS_DIR
    assert args.out_dir == ANALYSIS_DIR.parent / 'occlusion_seed_robustness'
    assert args.split_threshold == DEFAULT_SPLIT_THRESHOLD


def test_pattern_becomes_readable_output_subdirectory() -> None:
    assert (
        pattern_output_dirname('family/initialization-[1-5]')
        == 'family__initialization-[1-5]'
    )


def _write_result(
    root: Path,
    *,
    run_group: str,
    effects: tuple[float, float, float],
) -> Path:
    result_dir = root / run_group / 'completed-result'
    result_dir.mkdir(parents=True)
    (result_dir / 'COMPLETED.json').write_text(
        json.dumps({'complete': True}),
        encoding='utf-8',
    )
    pd.DataFrame(
        {
            'feature': FEATURES,
            'label': ('Large positive', 'Small positive', 'Large negative'),
            'mean_rank_effect': effects,
            'p_max_t_fwer': (0.01, 0.20, 0.04),
        }
    ).to_csv(result_dir / 'feature_summary.csv', index=False)
    return result_dir


def test_load_split_and_align_completed_seed_results(tmp_path: Path) -> None:
    first = _write_result(
        tmp_path,
        run_group='initialization-alpha',
        effects=(8.0, 1.0, -6.0),
    )
    second = _write_result(
        tmp_path,
        run_group='initialization-beta',
        effects=(10.0, 2.0, -8.0),
    )
    incomplete = tmp_path / 'initialization-gamma' / 'incomplete-result'
    incomplete.mkdir(parents=True)
    pd.DataFrame({'feature': FEATURES}).to_csv(
        incomplete / 'feature_summary.csv',
        index=False,
    )

    discovered = discover_completed_result_dirs(tmp_path, 'initialization-*')
    assert discovered == [first, second]

    effects, significant, labels = build_effect_tables(
        [load_occlusion_run(path) for path in discovered]
    )
    assert effects.columns.tolist() == [
        'initialization-alpha',
        'initialization-beta',
    ]
    assert labels.loc['small_positive'] == 'Small positive'
    assert significant.loc['large_positive'].tolist() == [True, True]
    assert significant.loc['small_positive'].tolist() == [False, False]

    large, smaller = split_features(effects, threshold=4.0)
    assert large == ['large_positive', 'large_negative']
    assert smaller == ['small_positive']


def test_split_threshold_is_strict() -> None:
    effects = pd.DataFrame(
        {42: [4.0, -4.0], 43: [4.0, -4.0]},
        index=['positive', 'negative'],
    )
    with pytest.raises(ValueError, match='empty plotting group'):
        split_features(effects, threshold=4.0)


def test_plot_feature_group_writes_title_free_png_and_pdf(
    tmp_path: Path,
) -> None:
    effects = pd.DataFrame(
        {42: [8.0, -6.0], 43: [10.0, -8.0]},
        index=['positive', 'negative'],
    )
    significant = pd.DataFrame(
        {42: [True, True], 43: [True, False]},
        index=effects.index,
    )
    labels = pd.Series(
        {'positive': 'Positive', 'negative': 'Negative'},
        name='label',
    )
    output_base = tmp_path / 'large_effects_robustness'

    plot_feature_group(
        features=['positive', 'negative'],
        effects=effects,
        significant=significant,
        labels=labels,
        output_base=output_base,
    )

    assert output_base.with_suffix('.png').is_file()
    assert output_base.with_suffix('.pdf').is_file()
    image = plt_read(output_base.with_suffix('.png'))
    assert np.asarray(image).size > 0


def plt_read(path: Path) -> np.ndarray:
    import matplotlib.pyplot as plt

    return plt.imread(path)
