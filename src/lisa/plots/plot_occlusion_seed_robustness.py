"""Plot occlusion-effect robustness across independently trained model seeds."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from lisa.utils.constants import (
    OCCLUSION_FEATURE_ANALYSIS_DIR,
    PLOTS_DIR,
)


ANALYSIS_DIR = Path(OCCLUSION_FEATURE_ANALYSIS_DIR)
DEFAULT_SPLIT_THRESHOLD = 4.0
SIGNIFICANCE_ALPHA = 0.05
FIGURE_SIZE = (7.2, 4.3)
FIGURE_DPI = 300
MAX_SEED_OFFSET = 0.24
MIN_MEDIAN_CLEARANCE = 0.06

REQUIRED_COLUMNS = frozenset(
    {
        'feature',
        'label',
        'mean_rank_effect',
        'p_max_t_fwer',
    }
)

# Only labels that are too long or inconsistent with the headline occlusion
# figure need overriding. All other labels come from feature_summary.csv.
DISPLAY_NAME_OVERRIDES = {
    'textgrid_silence_transition': 'Silence edges',
    'phoneme_vowels_all': 'Vowels',
    'phoneme_stops_all': 'Stops',
    'phoneme_fricatives_all': 'Fricatives',
    'phoneme_nasals_all': 'Nasals',
    'phoneme_liquids_approximants': 'Liquids/glides',
}

@dataclass(frozen=True)
class OcclusionRun:
    run_group: str
    result_dir: Path
    summary: pd.DataFrame


def discover_completed_result_dirs(
    analysis_dir: Path,
    pattern: str,
) -> list[Path]:
    """Find one completed occlusion result below every matching run group."""

    pattern_path = Path(pattern)
    if pattern_path.is_absolute() or '..' in pattern_path.parts:
        raise ValueError(
            f'Pattern must be relative and cannot contain "..": {pattern!r}'
        )

    analysis_dir = analysis_dir.expanduser()
    if not analysis_dir.is_dir():
        raise FileNotFoundError(f'Analysis directory does not exist: {analysis_dir}')

    matched_groups = sorted(
        path for path in analysis_dir.glob(pattern) if path.is_dir()
    )
    if not matched_groups:
        raise FileNotFoundError(
            f'Pattern {pattern!r} matched no run groups below {analysis_dir}.'
        )

    completed_dirs: list[Path] = []
    for group_dir in matched_groups:
        candidates = sorted(
            marker.parent
            for marker in group_dir.glob('*/COMPLETED.json')
            if (marker.parent / 'feature_summary.csv').is_file()
        )
        if len(candidates) > 1:
            raise ValueError(
                f'Run group {group_dir.name!r} contains multiple completed results: '
                f'{candidates}. Narrow the input pattern or remove the ambiguity.'
            )
        if candidates:
            completed_dirs.append(candidates[0])

    if len(completed_dirs) < 2:
        raise ValueError(
            'Seed robustness requires at least two completed results; '
            f'found {len(completed_dirs)}.'
        )
    return completed_dirs


def pattern_output_dirname(pattern: str) -> str:
    """Preserve a relative input pattern as a readable output directory name."""

    parts = [part for part in Path(pattern).parts if part not in {'', '.'}]
    if not parts:
        raise ValueError('Run pattern must contain a non-empty directory name.')
    return '__'.join(parts)


def load_occlusion_run(result_dir: Path) -> OcclusionRun:
    """Load and validate one completed feature summary."""

    summary_path = result_dir / 'feature_summary.csv'
    summary = pd.read_csv(summary_path)
    missing = REQUIRED_COLUMNS.difference(summary.columns)
    if missing:
        raise ValueError(f'{summary_path} is missing columns: {sorted(missing)}.')
    if summary.empty:
        raise ValueError(f'{summary_path} is empty.')
    if summary['feature'].isna().any() or summary['feature'].duplicated().any():
        raise ValueError(f'{summary_path} has missing or duplicate feature names.')

    summary = summary.copy().set_index('feature', drop=False)
    for column in ('mean_rank_effect', 'p_max_t_fwer'):
        summary[column] = pd.to_numeric(summary[column], errors='raise')
        values = summary[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f'{summary_path} contains non-finite {column} values.')
    p_values = summary['p_max_t_fwer'].to_numpy(dtype=np.float64)
    if np.any((p_values < 0.0) | (p_values > 1.0)):
        raise ValueError(f'{summary_path} contains invalid FWER p-values.')

    return OcclusionRun(
        run_group=result_dir.parent.name,
        result_dir=result_dir,
        summary=summary,
    )


def build_effect_tables(
    runs: Sequence[OcclusionRun],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build aligned feature-by-seed effect and significance tables."""

    runs = sorted(runs, key=lambda run: run.run_group)
    run_groups = [run.run_group for run in runs]
    if len(run_groups) != len(set(run_groups)):
        raise ValueError(f'Run groups must be unique, got {run_groups}.')

    reference_features = runs[0].summary.index
    reference_labels = runs[0].summary['label'].astype(str)
    effect_columns: dict[str, pd.Series] = {}
    significance_columns: dict[str, pd.Series] = {}

    for run in runs:
        if set(run.summary.index) != set(reference_features):
            raise ValueError(
                f'Feature set differs for run group {run.run_group!r} '
                f'in {run.result_dir}.'
            )
        aligned = run.summary.reindex(reference_features)
        labels = aligned['label'].astype(str)
        if not labels.equals(reference_labels):
            raise ValueError(f'Feature labels differ for {run.run_group!r}.')
        effect_columns[run.run_group] = aligned['mean_rank_effect']
        significance_columns[run.run_group] = (
            aligned['p_max_t_fwer'] < SIGNIFICANCE_ALPHA
        )

    effects = pd.DataFrame(effect_columns, index=reference_features)
    significant = pd.DataFrame(significance_columns, index=reference_features)
    return effects, significant, reference_labels


def split_features(
    effects: pd.DataFrame,
    threshold: float,
) -> tuple[list[str], list[str]]:
    """Split ordered features by absolute across-seed median effect."""

    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(
            f'Split threshold must be positive and finite, got {threshold}.'
        )

    medians = effects.median(axis=1).sort_values(ascending=False)
    large = [
        str(feature)
        for feature, value in medians.items()
        if abs(float(value)) > threshold
    ]
    smaller = [
        str(feature)
        for feature, value in medians.items()
        if abs(float(value)) <= threshold
    ]
    if not large or not smaller:
        raise ValueError(
            f'Split threshold {threshold:g} produced an empty plotting group.'
        )
    return large, smaller


def _seed_offsets(n_seeds: int) -> np.ndarray:
    """Spread seed points narrowly while leaving the median marker visible."""

    if n_seeds < 2:
        raise ValueError(f'At least two seeds are required, got {n_seeds}.')
    n_left = (n_seeds + 1) // 2
    n_right = n_seeds // 2
    left = np.linspace(-MAX_SEED_OFFSET, -MIN_MEDIAN_CLEARANCE, n_left)
    right = np.linspace(MIN_MEDIAN_CLEARANCE, MAX_SEED_OFFSET, n_right)
    return np.concatenate((left, right))


def _axis_limits(values: np.ndarray) -> tuple[float, float]:
    """Return linear limits with zero and modest data-dependent padding."""

    data_min = min(float(np.min(values)), 0.0)
    data_max = max(float(np.max(values)), 0.0)
    data_range = data_max - data_min
    if data_range <= 0.0:
        data_range = 1.0
    padding = 0.08 * data_range
    return data_min - padding, data_max + padding


def _display_name(feature: str, labels: pd.Series) -> str:
    return DISPLAY_NAME_OVERRIDES.get(feature, str(labels.loc[feature]))


def plot_feature_group(
    *,
    features: Sequence[str],
    effects: pd.DataFrame,
    significant: pd.DataFrame,
    labels: pd.Series,
    output_base: Path,
) -> None:
    """Write one title-free, half-page robustness figure."""

    medians = effects.median(axis=1)
    positions = np.arange(len(features))
    offsets = _seed_offsets(effects.shape[1])
    values = effects.loc[list(features)].to_numpy(dtype=np.float64)

    font_context = {
        'font.size': 10.5,
        'axes.labelsize': 11.0,
        'xtick.labelsize': 10.5,
        'ytick.labelsize': 10.5,
        'legend.fontsize': 10.0,
    }
    with plt.rc_context(font_context):
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        ax.axhline(0.0, color='black', linewidth=1.0, zorder=0)

        for position, feature in zip(positions, features):
            seed_values = effects.loc[feature].to_numpy(dtype=np.float64)
            ax.scatter(
                position + offsets,
                seed_values,
                s=25,
                color='0.48',
                alpha=0.82,
                edgecolors='white',
                linewidths=0.35,
                zorder=2.5,
            )
            median = float(medians.loc[feature])
            median_color = 'crimson' if median > 0.0 else 'steelblue'
            ax.plot(
                position,
                median,
                marker='D',
                markersize=7.0,
                color=median_color,
                markeredgecolor='black',
                markeredgewidth=0.7,
                linestyle='',
                zorder=4,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [_display_name(feature, labels) for feature in features],
            rotation=45,
            ha='right',
        )
        ax.set_xlim(-0.5, len(features) - 0.5)
        ax.set_ylim(*_axis_limits(values))
        ax.set_ylabel('Rank difference')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        count_transform = blended_transform_factory(ax.transData, ax.transAxes)
        counts = significant.loc[list(features)].sum(axis=1)
        for position, count in zip(positions, counts):
            ax.text(
                position,
                1.025,
                f'{int(count)}/{effects.shape[1]}',
                transform=count_transform,
                ha='center',
                va='bottom',
                fontsize=9.5,
                color='black',
                clip_on=False,
            )

        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='none',
                    markerfacecolor='0.48',
                    markeredgecolor='white',
                    markersize=5.5,
                    label='Seed',
                ),
                Line2D(
                    [0],
                    [0],
                    marker='D',
                    color='none',
                    markerfacecolor='0.55',
                    markeredgecolor='black',
                    markersize=7.0,
                    label='Median',
                ),
            ],
            loc='upper right',
            frameon=False,
        )

        fig.subplots_adjust(left=0.11, right=0.985, bottom=0.29, top=0.90)
        output_base.parent.mkdir(parents=True, exist_ok=True)
        for extension in ('png', 'pdf'):
            output_path = output_base.with_suffix(f'.{extension}')
            fig.savefig(
                output_path,
                dpi=FIGURE_DPI,
                facecolor='white',
                bbox_inches='tight',
            )
            print(output_path)
        plt.close(fig)


def run(
    *,
    analysis_dir: Path,
    pattern: str,
    out_dir: Path,
    split_threshold: float,
) -> None:
    """Load completed seed results and write both robustness figures."""

    result_dirs = discover_completed_result_dirs(analysis_dir, pattern)
    runs = [load_occlusion_run(result_dir) for result_dir in result_dirs]
    effects, significant, labels = build_effect_tables(runs)
    large_features, smaller_features = split_features(effects, split_threshold)
    pattern_out_dir = out_dir / pattern_output_dirname(pattern)

    print(
        f'Run groups: {", ".join(str(name) for name in effects.columns)}; '
        f'split rule: |across-seed median rank difference| > '
        f'{split_threshold:g}.'
    )
    plot_feature_group(
        features=large_features,
        effects=effects,
        significant=significant,
        labels=labels,
        output_base=pattern_out_dir / 'large_effects_robustness',
    )
    plot_feature_group(
        features=smaller_features,
        effects=effects,
        significant=significant,
        labels=labels,
        output_base=pattern_out_dir / 'smaller_effects_robustness',
    )


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Plot occlusion-effect robustness across completed model-seed runs.'
        )
    )
    parser.add_argument(
        'pattern',
        metavar='RUN_PATTERN',
        help=(
            'Required run-group glob below --analysis-dir, for example '
            "'my-model-initialization-*'."
        ),
    )
    parser.add_argument(
        '--analysis-dir',
        type=Path,
        default=ANALYSIS_DIR,
        help='Directory containing occlusion run-group outputs.',
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path(PLOTS_DIR) / 'occlusion_seed_robustness',
        help='Output root; the run pattern becomes a subdirectory.',
    )
    parser.add_argument(
        '--split-threshold',
        type=float,
        default=DEFAULT_SPLIT_THRESHOLD,
        help=(
            'Absolute across-seed median rank difference separating the two '
            f'figures. Default: {DEFAULT_SPLIT_THRESHOLD:g}.'
        ),
    )
    return parser.parse_args(argv)


def main_cli() -> None:
    args = parse_arguments()
    run(
        analysis_dir=args.analysis_dir,
        pattern=args.pattern,
        out_dir=args.out_dir,
        split_threshold=args.split_threshold,
    )


if __name__ == '__main__':
    main_cli()
