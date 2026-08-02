"""Plot segment-duration ablation retrieval metrics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lisa.plots.plot_metrics import GROUP_LINE_COLORS, group_line_style
from lisa.plots.plot_style import panel_figsize, publication_style
from lisa.utils.constants import EXPERIMENTS_DIR, PLOTS_DIR
from lisa.utils.validators import validate_run_group


DEFAULT_DEPTHS = (0, 2, 5)
DEFAULT_WINDOW_TAGS = ('w1p5', 'w2p25', 'w3', 'w4', 'w5')
DEFAULT_WINDOW_SECONDS = (1.5, 2.25, 3.0, 4.0, 5.0)


@dataclass(frozen=True)
class WindowSpec:
    tag: str
    seconds: float
    label: str


@dataclass(frozen=True)
class RunArtifacts:
    run_group: str
    run_dir: Path
    config_path: Path
    metrics_path: Path
    final_test_path: Path
    config: dict


def format_window_label(seconds: float) -> str:
    return f'{seconds:g} s'


def parse_window_specs(
    tags: Sequence[str],
    seconds: Sequence[float],
    labels: Sequence[str] | None,
) -> list[WindowSpec]:
    if len(tags) != len(seconds):
        raise ValueError(
            '--window-tags and --window-seconds must have the same length '
            f'({len(tags)} != {len(seconds)}).'
        )
    if labels is not None and len(labels) != len(tags):
        raise ValueError(
            '--window-labels must have the same length as --window-tags '
            f'({len(labels)} != {len(tags)}).'
        )
    return [
        WindowSpec(
            tag=tag,
            seconds=float(sec),
            label=labels[i] if labels is not None else format_window_label(float(sec)),
        )
        for i, (tag, sec) in enumerate(zip(tags, seconds))
    ]


def read_json(path: Path) -> dict:
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def get_final_metric_row(metrics_path: Path) -> pd.Series:
    metrics = pd.read_csv(metrics_path)
    required = {'top1s_final_test', 'top10s_final_test'}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(
            f'{metrics_path} is missing final-test columns: {sorted(missing)}.'
        )
    final_rows = metrics.dropna(subset=['top1s_final_test', 'top10s_final_test'])
    if final_rows.empty:
        raise ValueError(f'{metrics_path} has no logged final-test metrics.')
    return final_rows.iloc[-1]


def topn_accuracy_from_ranks(final_test_path: Path, n: int) -> float:
    ranks = pd.read_csv(final_test_path, usecols=['rank_of_true'])[
        'rank_of_true'
    ].astype(int)
    return float(((ranks >= 0) & (ranks < n)).mean())


def find_matching_run(
    experiments_root: Path,
    run_group: str,
    depth: int,
    window_tag: str,
    n_channels_unmix: int,
    strict_single_run: bool,
) -> RunArtifacts:
    group_dir = experiments_root / run_group
    if not group_dir.is_dir():
        raise FileNotFoundError(f'Run group directory not found: {group_dir}')

    matches: list[RunArtifacts] = []
    for config_path in sorted(group_dir.glob('*/config.json')):
        run_dir = config_path.parent
        metrics_path = run_dir / 'metrics.csv'
        final_test_path = run_dir / 'final_test_per_window.csv'
        if not metrics_path.is_file() or not final_test_path.is_file():
            continue
        config = read_json(config_path)
        if int(config.get('n_channels_unmix', -1)) != n_channels_unmix:
            continue
        if int(config.get('n_temporal_module_blocks', -1)) != depth:
            continue
        if config.get('window_tag') != window_tag:
            continue
        matches.append(
            RunArtifacts(
                run_group=run_group,
                run_dir=run_dir,
                config_path=config_path,
                metrics_path=metrics_path,
                final_test_path=final_test_path,
                config=config,
            )
        )

    if not matches:
        raise FileNotFoundError(
            f'No matching run found under {group_dir} for depth={depth}, '
            f'window_tag={window_tag!r}, n_channels_unmix={n_channels_unmix}.'
        )
    if strict_single_run and len(matches) != 1:
        dirs = '\n  '.join(str(m.run_dir) for m in matches)
        raise ValueError(
            f'Found {len(matches)} matching runs for {run_group}; expected one:\n  '
            f'{dirs}'
        )

    selected = max(matches, key=lambda m: m.metrics_path.stat().st_mtime)
    if len(matches) > 1:
        print(
            f'Warning: found {len(matches)} matching runs for {run_group}; '
            f'using newest metrics.csv: {selected.run_dir}',
            file=sys.stderr,
        )
    return selected


def collect_summary(
    experiments_root: Path,
    depths: Sequence[int],
    windows: Sequence[WindowSpec],
    run_group_template: str,
    n_channels_unmix: int,
    strict_single_run: bool,
) -> pd.DataFrame:
    rows: list[dict] = []
    for depth in depths:
        for window in windows:
            run_group = run_group_template.format(depth=depth, tag=window.tag)
            validate_run_group(run_group)
            artifacts = find_matching_run(
                experiments_root=experiments_root,
                run_group=run_group,
                depth=depth,
                window_tag=window.tag,
                n_channels_unmix=n_channels_unmix,
                strict_single_run=strict_single_run,
            )
            final_row = get_final_metric_row(artifacts.metrics_path)
            # Fallback to per-window ranks if a future metrics file is incomplete.
            top1 = final_row.get('top1s_final_test')
            top10 = final_row.get('top10s_final_test')
            if pd.isna(top1):
                top1 = topn_accuracy_from_ranks(artifacts.final_test_path, 1)
            if pd.isna(top10):
                top10 = topn_accuracy_from_ranks(artifacts.final_test_path, 10)
            rows.append(
                {
                    'Depth': depth,
                    'Blocks': f'{depth} Blocks',
                    'Window tag': window.tag,
                    'Segment duration': window.seconds,
                    'Segment duration label': window.label,
                    'Top 1': float(top1),
                    'Top 10': float(top10),
                    'Run group': run_group,
                    'Run dir': str(artifacts.run_dir),
                }
            )
    return pd.DataFrame(rows)


def block_color(depth: int):
    if 0 <= depth < len(GROUP_LINE_COLORS):
        return sns.color_palette([GROUP_LINE_COLORS[depth]])[0]
    return sns.color_palette('colorblind', 1)[0]


def save_figure(fig: plt.Figure, outdir: Path, basename: str, formats: Sequence[str]):
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = outdir / f'{basename}.{fmt}'
        fig.savefig(path)
        print(path)


def slugify_component(value: object) -> str:
    slug = re.sub(r'[^A-Za-z0-9._-]+', '-', str(value).strip().lower())
    slug = slug.replace('.', 'p').strip('-_')
    if not slug:
        raise ValueError(f'Cannot build output directory slug from {value!r}.')
    return slug


def build_output_dir(
    base_outdir: Path,
    output_subdir: str | None,
    windows: Sequence[WindowSpec],
    depths: Sequence[int],
    n_channels_unmix: int,
    rank_depth: int,
    rank_max_n: int,
) -> Path:
    if output_subdir is not None:
        return base_outdir / slugify_component(output_subdir)

    depth_slug = '-'.join(slugify_component(depth) for depth in depths)
    window_slug = '-'.join(slugify_component(window.tag) for window in windows)
    return base_outdir / (
        f'{n_channels_unmix}branches_depths{depth_slug}_'
        f'windows-{window_slug}_rank{rank_depth}_topn{rank_max_n}'
    )


@publication_style
def plot_segment_duration_accuracy(
    df: pd.DataFrame,
    windows: Sequence[WindowSpec],
    block_order: Sequence[str],
    n_channels_unmix: int,
    highlight_depth: int,
    outdir: Path,
    output_prefix: str,
    formats: Sequence[str],
) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=panel_figsize(2),
        sharex=True,
        layout='constrained',
    )
    x_ticks = [w.seconds for w in windows]
    x_labels = [w.label for w in windows]

    for ax_idx, metric_name in enumerate(('Top 1', 'Top 10')):
        ax = axes[ax_idx]
        for block_label in block_order:
            curve = (
                df[df['Blocks'] == block_label]
                .sort_values('Segment duration')
                .set_index('Segment duration')
            )
            depth = int(curve['Depth'].iloc[0])
            is_highlight = depth == highlight_depth
            line_idx = block_order.index(block_label)
            ax.plot(
                curve.index.to_numpy(dtype=float),
                curve[metric_name].to_numpy(dtype=float),
                label=block_label if ax_idx == 1 else None,
                color=block_color(depth),
                linestyle=group_line_style(block_label, line_idx),
                marker='o',
                markersize=4.5 if is_highlight else 3.5,
                linewidth=2.2 if is_highlight else 1.6,
                alpha=1.0 if is_highlight else 0.72,
                zorder=3 if is_highlight else 2,
            )

        ax.set_ylabel(
            'Top-10 accuracy' if metric_name == 'Top 10' else 'Top-1 accuracy'
        )
        ax.set_xlabel('' if ax_idx == 0 else 'Segment duration (s)')
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.grid(True, axis='both', alpha=0.35, linestyle='-')
        ax.set_axisbelow(True)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

        y_min_val = df[metric_name].min()
        y_max_val = df[metric_name].max()
        y_range = y_max_val - y_min_val
        pad = max(float(y_range) * 0.1, 0.02)
        ax.set_ylim(max(0.0, y_min_val - pad), min(1.0, y_max_val + pad))

    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[1].legend(
        by_label.values(),
        by_label.keys(),
        title='Temporal blocks',
        loc='lower right',
        framealpha=0.95,
        handlelength=2.2,
        labelspacing=0.6,
    )

    save_figure(
        fig,
        outdir,
        f'{output_prefix}_{n_channels_unmix}branches_accuracy_vs_segment_duration',
        formats,
    )
    plt.close(fig)


def read_rank_curve(final_test_path: Path, max_n: int) -> tuple[np.ndarray, int | None]:
    header = pd.read_csv(final_test_path, nrows=0)
    usecols = ['rank_of_true']
    has_wav_index = 'wav_index' in header.columns
    if has_wav_index:
        usecols.append('wav_index')
    df = pd.read_csv(final_test_path, usecols=usecols)
    ranks = df['rank_of_true'].astype(int)
    n_values = np.arange(1, max_n + 1)
    accuracies = np.array(
        [((ranks >= 0) & (ranks < n)).mean() for n in n_values],
        dtype=float,
    )
    candidate_count = int(df['wav_index'].nunique()) if has_wav_index else None
    return accuracies, candidate_count


def collect_rank_curves(
    experiments_root: Path,
    depth: int,
    windows: Sequence[WindowSpec],
    run_group_template: str,
    n_channels_unmix: int,
    max_n: int,
    strict_single_run: bool,
) -> tuple[pd.DataFrame, int | None]:
    rows: list[dict] = []
    candidate_counts: set[int] = set()
    for window in windows:
        run_group = run_group_template.format(depth=depth, tag=window.tag)
        validate_run_group(run_group)
        artifacts = find_matching_run(
            experiments_root=experiments_root,
            run_group=run_group,
            depth=depth,
            window_tag=window.tag,
            n_channels_unmix=n_channels_unmix,
            strict_single_run=strict_single_run,
        )
        accuracies, candidate_count = read_rank_curve(artifacts.final_test_path, max_n)
        if candidate_count is not None:
            candidate_counts.add(candidate_count)
        for n, acc in enumerate(accuracies, start=1):
            rows.append(
                {
                    'Depth': depth,
                    'Blocks': f'{depth} Blocks',
                    'Window tag': window.tag,
                    'Segment duration label': window.label,
                    'Segment duration': window.seconds,
                    'Retrieval cutoff': n,
                    'Accuracy': acc,
                    'Run group': run_group,
                    'Run dir': str(artifacts.run_dir),
                }
            )
    candidate_count_out = candidate_counts.pop() if len(candidate_counts) == 1 else None
    return pd.DataFrame(rows), candidate_count_out


@publication_style
def plot_rank_curve(
    rank_df: pd.DataFrame,
    windows: Sequence[WindowSpec],
    rank_depth: int,
    max_n: int,
    candidate_count: int | None,
    n_channels_unmix: int,
    outdir: Path,
    output_prefix: str,
    formats: Sequence[str],
) -> None:
    fig, ax = plt.subplots(
        1,
        1,
        figsize=panel_figsize(1),
        layout='constrained',
    )
    palette = dict(
        zip(
            [w.label for w in windows],
            sns.color_palette('viridis', len(windows)),
        )
    )
    for window in windows:
        curve = rank_df[rank_df['Segment duration label'] == window.label].sort_values(
            'Retrieval cutoff'
        )
        ax.plot(
            curve['Retrieval cutoff'].to_numpy(dtype=float),
            curve['Accuracy'].to_numpy(dtype=float),
            label=window.label,
            color=palette[window.label],
            linewidth=2.0 if window.seconds == max(w.seconds for w in windows) else 1.6,
            alpha=0.95,
        )

    x_ticks = [1] + list(range(5, max_n + 1, 5))
    if max_n not in x_ticks:
        x_ticks.append(max_n)
    ax.set_xticks(sorted(set(x_ticks)))
    ax.set_xlim(1, max_n)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Top-n cutoff')
    ax.set_ylabel('Top-n retrieval accuracy')
    ax.grid(True, axis='both', alpha=0.35, linestyle='-')
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.legend(
        title='Segment duration',
        loc='lower right',
        framealpha=0.95,
        handlelength=2.2,
        labelspacing=0.6,
    )
    save_figure(
        fig,
        outdir,
        f'{output_prefix}_{n_channels_unmix}branches_{rank_depth}blocks_topn{max_n}',
        formats,
    )
    plt.close(fig)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot segment-duration ablation retrieval metrics.'
    )
    parser.add_argument(
        '--experiments-root',
        type=Path,
        default=Path(EXPERIMENTS_DIR),
        help='Root directory with experiments/<run_group>/<run>/ artifacts.',
    )
    parser.add_argument(
        '--outdir',
        type=Path,
        default=Path(PLOTS_DIR) / 'segment_duration_ablation',
        help='Parent output directory for generated figures.',
    )
    parser.add_argument(
        '--output-subdir',
        default=None,
        help=(
            'Optional subdirectory under --outdir. By default, a stable '
            'parameter-specific subdirectory is generated.'
        ),
    )
    parser.add_argument(
        '--depths',
        nargs='+',
        type=int,
        default=DEFAULT_DEPTHS,
        help='Temporal-module block counts for the main accuracy plot.',
    )
    parser.add_argument(
        '--window-tags',
        nargs='+',
        default=DEFAULT_WINDOW_TAGS,
        help='Segment-duration tags in plotting order.',
    )
    parser.add_argument(
        '--window-seconds',
        nargs='+',
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help='Segment durations in seconds, same order as --window-tags.',
    )
    parser.add_argument(
        '--window-labels',
        nargs='+',
        default=None,
        help='Optional segment-duration labels, same order as --window-tags.',
    )
    parser.add_argument(
        '--run-group-template',
        default='{depth}conv-seed42-paper-{tag}',
        help='Template with {depth} and {tag}, e.g. {depth}conv-seed42-paper-{tag}.',
    )
    parser.add_argument(
        '--n-channels-unmix',
        type=int,
        default=25,
        help='K / number of branches to select within each run group.',
    )
    parser.add_argument(
        '--rank-depth',
        type=int,
        default=2,
        help='Temporal-module block count for the Acc@n rank-curve plot.',
    )
    parser.add_argument(
        '--rank-max-n',
        type=int,
        default=50,
        help='Maximum retrieval cutoff n for the Acc@n rank-curve plot.',
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        default=('pdf', 'png'),
        choices=('pdf', 'png', 'svg'),
        help='Figure formats to write.',
    )
    parser.add_argument(
        '--output-prefix',
        default='segment_duration_ablation',
        help='Prefix for output figure filenames.',
    )
    parser.add_argument(
        '--strict-single-run',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Fail if multiple matching runs exist in a run group. '
            'Enabled by default; use --no-strict-single-run to use newest.'
        ),
    )
    args = parser.parse_args()
    if args.rank_max_n < 1:
        raise ValueError('--rank-max-n must be positive.')
    validate_run_group(args.output_prefix)
    return args


def main_cli() -> None:
    args = parse_arguments()
    windows = parse_window_specs(
        tags=args.window_tags,
        seconds=args.window_seconds,
        labels=args.window_labels,
    )
    block_order = [f'{depth} Blocks' for depth in args.depths]
    output_dir = build_output_dir(
        base_outdir=args.outdir,
        output_subdir=args.output_subdir,
        windows=windows,
        depths=args.depths,
        n_channels_unmix=args.n_channels_unmix,
        rank_depth=args.rank_depth,
        rank_max_n=args.rank_max_n,
    )

    summary_df = collect_summary(
        experiments_root=args.experiments_root,
        depths=args.depths,
        windows=windows,
        run_group_template=args.run_group_template,
        n_channels_unmix=args.n_channels_unmix,
        strict_single_run=args.strict_single_run,
    )
    plot_segment_duration_accuracy(
        df=summary_df,
        windows=windows,
        block_order=block_order,
        n_channels_unmix=args.n_channels_unmix,
        highlight_depth=args.rank_depth,
        outdir=output_dir,
        output_prefix=args.output_prefix,
        formats=args.formats,
    )

    rank_df, candidate_count = collect_rank_curves(
        experiments_root=args.experiments_root,
        depth=args.rank_depth,
        windows=windows,
        run_group_template=args.run_group_template,
        n_channels_unmix=args.n_channels_unmix,
        max_n=args.rank_max_n,
        strict_single_run=args.strict_single_run,
    )
    if candidate_count is not None and args.rank_max_n > candidate_count:
        raise ValueError(
            f'--rank-max-n={args.rank_max_n} exceeds candidate count {candidate_count}.'
        )
    plot_rank_curve(
        rank_df=rank_df,
        windows=windows,
        rank_depth=args.rank_depth,
        max_n=args.rank_max_n,
        candidate_count=candidate_count,
        n_channels_unmix=args.n_channels_unmix,
        outdir=output_dir,
        output_prefix=args.output_prefix,
        formats=args.formats,
    )


if __name__ == '__main__':
    main_cli()
