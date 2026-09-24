"""Plot accuracy metrics versus K (number of unmix channels)."""

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import numpy as np
import pandas as pd
import seaborn as sns

from lisa.plots.plot_style import panel_figsize, publication_style
from lisa.utils.constants import EXPERIMENTS_DIR, PLOTS_DIR
from lisa.utils.validators import validate_run_group

# Config keys that are allowed to differ across run_groups/runs (paths, machine, seed, logging, names).
# n_channels_unmix (K) is the x-axis of the plot, so it must be allowed to differ per run.
# All other keys (architecture, training hyperparams, etc.) must match so we compare "same model" sweeps.
# See parse_experiment_args.py for full list. Other args you might want to allow:
#   - upload_metrics_csv: logging only
#   - dl_n_workers: runtime, can differ per machine
CONFIG_ALLOWED_DIFFERENCES = frozenset(
    {
        'n_channels_unmix',  # K: the x-axis of the plot, differs per run
        'n_temporal_module_blocks',  # compared as separate run-group lines
        'device',
        'dirprocess',
        'meg_files_dir',
        'meg_format',
        'seed',
        'torch_deterministic',
        'logging_project',
        'run_group',
        'max_branches_to_plot',
        'plot_filter_graphs',
        'experiments_root',
        # Segment-length ablation: selects alt embeddings/df files. Allowed to differ
        # ACROSS run_groups (each window length is its own line), but enforced constant
        # WITHIN a run_group by _assert_constant_within_run_group so different window
        # lengths are never silently averaged into one point.
        'window_tag',
        'image_formats',
        'log_images_to_server',
        'logger',
        'checkpoint',
        'run_name',
        'upload_metrics_csv',
        'dl_n_workers',
        'save_test_top_k',  # only affects final_test_per_window.npz dump width
    }
)

# Keys that may differ across configs as long as they are consistent per K (same value for all runs with the same n_channels_unmix).
# E.g. n_channels_block can be 10 for K=10 and 15 for K=15, but all runs with K=10 must have n_channels_block=10.
CONFIG_ALLOWED_TO_VARY_BY_K = frozenset({'n_channels_block'})

# Keys that select the underlying dataset a run trained/evaluated on and therefore must be
# constant within a single run_group (one run_group == one plotted line). These are in
# CONFIG_ALLOWED_DIFFERENCES so they may differ across run_groups, but mixing them inside
# one run_group would silently average incomparable conditions (e.g. 2s and 5s retrieval)
# into a single point. Enforced by _assert_constant_within_run_group.
CONFIG_CONSTANT_WITHIN_RUN_GROUP = frozenset({'window_tag'})

# Default K values to show on x-axis (only those present in data). Linear vs log.
DEFAULT_X_TICKS_LINEAR = (1, 10, 25, 50, 75, 100, 150, 200, 250, 270)
DEFAULT_X_TICKS_LOG = (1, 5, 10, 25, 50, 75, 100, 150, 200, 250)
GROUP_LINE_COLORS = (
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#009E73',  # green
    '#CC79A7',  # purple
    '#E69F00',  # orange
    '#000000',  # black
    '#56B4E9',  # sky blue
    '#F0E442',  # yellow
)
GROUP_LINE_STYLES = ('-', '--', '-.', ':', (0, (6, 2)), (0, (3, 1, 1, 1)))
BLOCK_LINE_STYLES_BY_DEPTH = {
    0: '--',
    1: ':',
    2: '-',
    3: '-.',
    4: (0, (6, 2)),
    5: (0, (3, 1, 1, 1)),
}


def group_line_style(line_label: str, line_idx: int):
    """Choose line style, keeping the 2-block model solid in block comparisons."""
    parts = str(line_label).strip().split()
    if (
        parts
        and parts[0].isdigit()
        and any('block' in part.lower() for part in parts[1:])
    ):
        depth = int(parts[0])
        if depth in BLOCK_LINE_STYLES_BY_DEPTH:
            return BLOCK_LINE_STYLES_BY_DEPTH[depth]
    return GROUP_LINE_STYLES[line_idx % len(GROUP_LINE_STYLES)]


def _config_value_for_comparison(v):
    """Normalize config value for in-memory equality check only (dict key order, list vs tuple).
    Does not write or modify any config files."""
    if isinstance(v, dict):
        return tuple(
            sorted((k, _config_value_for_comparison(v2)) for k, v2 in v.items())
        )
    if isinstance(v, list):
        return tuple(_config_value_for_comparison(x) for x in v)
    if isinstance(v, tuple):
        return tuple(_config_value_for_comparison(x) for x in v)
    if hasattr(v, 'item'):  # numpy scalar
        return v.item()
    return v


def _assert_configs_aligned(configs: list[dict], config_paths: list[str]) -> None:
    """Raise if configs differ in a way that is not allowed.

    - CONFIG_ALLOWED_DIFFERENCES: key may differ arbitrarily.
    - CONFIG_ALLOWED_TO_VARY_BY_K: key may differ across configs but must be constant within each K
      (all runs with the same n_channels_unmix must have the same value).
    - All other keys: must be identical across all configs.
    """
    if len(configs) <= 1:
        return
    all_keys = set()
    for c in configs:
        all_keys.update(c.keys())
    for key in sorted(all_keys):
        if key in CONFIG_ALLOWED_DIFFERENCES:
            continue
        if key in CONFIG_ALLOWED_TO_VARY_BY_K:
            # Group by K (n_channels_unmix); within each K, require same value for this key
            by_k: dict[int, list[tuple[dict, str, int]]] = defaultdict(list)
            for i, c in enumerate(configs):
                k = c.get('n_channels_unmix')
                if k is None:
                    raise ValueError(
                        f"Config {config_paths[i]!r} has no 'n_channels_unmix'; required for per-K checks."
                    )
                by_k[k].append((c, config_paths[i], i))
            for k, group in by_k.items():
                if len(group) <= 1:
                    continue
                ref_norm = _config_value_for_comparison(group[0][0].get(key))
                for c, path, idx in group[1:]:
                    if _config_value_for_comparison(c.get(key)) != ref_norm:
                        raise ValueError(
                            f'Config mismatch on key {key!r} for K={k}: '
                            f'{group[0][1]!r} has {group[0][0].get(key)!r} vs {path!r} has {c.get(key)!r}. '
                            'For keys in CONFIG_ALLOWED_TO_VARY_BY_K, all runs with the same K must agree.'
                        )
            continue
        ref = configs[0].get(key)
        ref_norm = _config_value_for_comparison(ref)
        for i, c in enumerate(configs[1:], start=1):
            val = c.get(key)
            if _config_value_for_comparison(val) != ref_norm:
                raise ValueError(
                    f'Config mismatch on key {key!r}: '
                    f'{config_paths[0]!r} has {ref!r} vs {config_paths[i]!r} has {val!r}. '
                    'Only keys in CONFIG_ALLOWED_DIFFERENCES may differ across run_groups.'
                )


def _assert_constant_within_run_group(
    runs: list[tuple[str, str, str, dict, str]],
) -> None:
    """Raise if a dataset-selecting key varies within a single run_group.

    Runs sharing a (run_group, K) are averaged together when plotting, so a key
    that changes the data a run was trained/evaluated on (currently window_tag,
    the segment-length ablation) must be constant within each run_group. Different
    values belong in different run_groups (separate lines). Without this guard,
    e.g. a 2s and a 5s run placed in the same run_group would be silently averaged
    into one incomparable point.
    """
    for key in CONFIG_CONSTANT_WITHIN_RUN_GROUP:
        seen: dict[str, tuple[Any, Any, str]] = {}
        for run_group, config_path, _metrics_path, hyper_params, _task_dir in runs:
            raw = hyper_params.get(key)
            norm = _config_value_for_comparison(raw)
            if run_group not in seen:
                seen[run_group] = (norm, raw, config_path)
            elif seen[run_group][0] != norm:
                _, first_raw, first_path = seen[run_group]
                raise ValueError(
                    f'Run group {run_group!r} mixes {key!r} values: '
                    f'{first_path!r} has {first_raw!r} vs {config_path!r} has {raw!r}. '
                    f'Each {key} must have its own run_group so the conditions are '
                    'plotted as separate lines instead of being averaged together.'
                )


def load_metrics(
    experiments_root: str,
    run_groups: list[str],
    epochs: Iterable[int | str] | None,
    k_special: int = 270,
    validation_metrics: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load metrics across runs in one or more run-groups and return dataframes for plotting.

    Finds run dirs under experiments_root/<run_group>/*/config.json for each run_group.
    Configs must be identical except for CONFIG_ALLOWED_DIFFERENCES (e.g. same architecture).
    Returns long-format df (one row per run per Epoch per K) for seaborn lineplot.
    For epoch='best', the checkpoint is always the min-loss_val epoch; plotted values
    are final-test metrics unless validation_metrics is True.

    Returns:
        df_long: DataFrame with Run group, Epoch, K, Top 1, Top 10 (one row per run).
        df_k_special: One row per Epoch with mean Top 1 / Top 10 at K=k_special (for horizontal lines).
        epoch_order: Ordered list of epoch labels.
    """
    validation_cols = {'top1s_val': 'Top 1', 'top10s_val': 'Top 10'}
    final_cols = {
        'top1s_final_test': 'Top 1',
        'top10s_final_test': 'Top 10',
    }
    runs: list[tuple[str, str, dict, str, str]] = []

    for run_group in run_groups:
        group_dir = os.path.join(experiments_root, run_group)
        if not os.path.isdir(group_dir):
            raise FileNotFoundError(f'Run group directory not found: {group_dir}')

        config_globs = sorted(glob.glob(os.path.join(group_dir, '*', 'config.json')))
        if not config_globs:
            raise FileNotFoundError(
                f'No run dirs (with config.json) found under {group_dir}'
            )

        for config_path in config_globs:
            task_dir = os.path.dirname(config_path)
            metrics_path = os.path.join(task_dir, 'metrics.csv')
            if not os.path.isfile(metrics_path):
                raise FileNotFoundError(
                    f'Run dir has config.json but no metrics.csv: {task_dir}'
                )

            with open(config_path, encoding='utf-8') as f:
                hyper_params = json.load(f)
            if 'n_channels_unmix' not in hyper_params:
                raise ValueError(
                    f"config.json in {task_dir} is missing 'n_channels_unmix' (required for metrics vs K)."
                )
            K = hyper_params['n_channels_unmix']
            if not (isinstance(K, (int, float)) and K > 0):
                raise ValueError(
                    f'config.json in {task_dir} has n_channels_unmix={K!r}; K must be a positive number.'
                )
            runs.append((run_group, config_path, metrics_path, hyper_params, task_dir))

    _assert_configs_aligned([r[3] for r in runs], [r[1] for r in runs])
    _assert_constant_within_run_group(runs)

    epoch_iter = ('best',) if epochs is None else epochs

    run_rows: list[dict] = []
    for run_group, config_path, metrics_path, hyper_params, task_dir in runs:
        K = hyper_params['n_channels_unmix']
        assert isinstance(K, (int, float)) and K > 0  # validated above
        metrics_df = pd.read_csv(metrics_path, index_col='epoch')
        for epoch in epoch_iter:
            if epoch == 'best':
                epoch_name = 'Best validation checkpoint'
                epoch_metric_cols = (
                    validation_cols if validation_metrics else final_cols
                )
            else:
                epoch_name = f'Epoch {epoch}'
                epoch_metric_cols = validation_cols
            row = {'Run group': run_group, 'Epoch': epoch_name, 'K': K}
            for metric_id, metric_name in epoch_metric_cols.items():
                if epoch == 'best':
                    required_cols = {'loss_val', *epoch_metric_cols}
                    missing_cols = required_cols - set(metrics_df.columns)
                    if missing_cols:
                        raise ValueError(
                            f'Missing required best-checkpoint metrics in {metrics_path}: '
                            f'{sorted(missing_cols)}.'
                        )
                    valid_loss = metrics_df['loss_val'].dropna()
                    if valid_loss.empty:
                        raise ValueError(f'No loss_val values found in {metrics_path}.')
                    best_epoch = valid_loss.astype(float).idxmin()
                    val = metrics_df.loc[best_epoch, metric_id]
                    if pd.isna(val):
                        raise ValueError(
                            f'{metric_id!r} is missing at best validation epoch '
                            f'{best_epoch} in {metrics_path}.'
                        )
                else:
                    if metric_id not in metrics_df.columns:
                        raise ValueError(
                            f'Missing validation metric {metric_id!r} in {metrics_path}.'
                        )
                    if int(epoch) not in metrics_df.index:
                        raise ValueError(f'Epoch {epoch} is missing in {metrics_path}.')
                    val = metrics_df[metric_id].loc[int(epoch)]
                row[metric_name] = float(val)
            run_rows.append(row)

    df_long = pd.DataFrame(run_rows)
    if df_long.empty:
        raise ValueError('No metrics rows collected from any run_group.')

    # Keep epoch order as in data
    epoch_order = []
    for e in df_long['Epoch']:
        if e not in epoch_order:
            epoch_order.append(e)

    # For horizontal lines: one row per Epoch with mean at K=k_special
    at_k = df_long[df_long['K'] == k_special]
    if at_k.empty:
        df_k_special = pd.DataFrame(
            columns=['Run group', 'Epoch', 'K', 'Top 1', 'Top 10']
        )
    else:
        df_k_special = (
            at_k.groupby(['Run group', 'Epoch'], sort=False)
            .agg({'Top 1': 'mean', 'Top 10': 'mean'})
            .reset_index()
        )
        df_k_special['K'] = k_special

    return df_long, df_k_special, epoch_order


@publication_style
def plot_metrics(
    df_long: pd.DataFrame,
    df_k_special: pd.DataFrame,
    epoch_order: list[str],
    outdir: str | Path,
    run_group_label: str,
    line_order: Sequence[str],
    legend_title: str | None = None,
    highlight_run_group: str | None = None,
    k_special: int = 270,
    use_log_scale: bool = True,
    x_ticks: Sequence[int] | None = None,
    validation_metrics: bool = False,
) -> None:
    """Plot Top-1 and Top-10 accuracy vs K and save as PDF.

    Uses line plots without error bars. K_special is on the main curve and also
    highlighted with horizontal dashed lines for single-run-group plots.

    Args:
        df_long: Long-format DataFrame (Epoch, K, Top 1, Top 10), one row per run.
        df_k_special: One row per Epoch with mean at K=k_special (for horizontal lines).
        epoch_order: Ordered epoch labels.
        outdir: Output directory for plots.
        run_group_label: Label for output filename (path-safe).
        line_order: Ordered line labels for multi-run-group plots.
        legend_title: Legend title. Defaults to line labels for multi-group plots and
            Epoch for single-group plots.
        highlight_run_group: Optional run group to draw above other group lines.
        k_special: K value to highlight with horizontal lines.
        use_log_scale: If True (default), use log scale for K on x-axis; else linear.
        x_ticks: K values to show as x-axis tick labels. Only values present in the data
            are shown. If None (default), use the scale's default set (DEFAULT_X_TICKS_*).
    """
    draw_by_group = len(line_order) > 1
    if draw_by_group and len(epoch_order) > 1:
        raise ValueError(
            'Multi-run-group plots draw one line per run group and support one '
            'epoch at a time. Pass a single value to --epochs.'
        )
    hue_col = 'Line label' if draw_by_group else 'Epoch'
    hue_order = list(line_order) if draw_by_group else epoch_order
    legend_heading = legend_title or ('Temporal blocks' if draw_by_group else 'Epoch')
    highlight_line_labels: set[str] = set()
    if draw_by_group and highlight_run_group is not None:
        highlight_line_labels = set(
            df_long.loc[df_long['Run group'] == highlight_run_group, 'Line label']
            .dropna()
            .unique()
        )
        if not highlight_line_labels:
            raise ValueError(
                f'--highlight-run-group={highlight_run_group!r} was not found '
                'in the plotted data.'
            )

    # Colorblind-friendly palette; use a higher-contrast fixed set for group plots.
    n_lines = len(hue_order)
    if draw_by_group and n_lines <= len(GROUP_LINE_COLORS):
        palette = sns.color_palette(GROUP_LINE_COLORS[:n_lines])
    elif n_lines <= 10:
        palette = sns.color_palette('colorblind', n_lines)
    else:
        palette = sns.color_palette('muted', n_lines)
    hue_palette = dict(zip(hue_order, palette))
    k_vals = np.sort(df_long['K'].unique())
    k_set = set(k_vals)
    n_k = len(k_vals)

    if x_ticks is not None:
        preferred = tuple(x_ticks)
        missing = [k for k in preferred if k not in k_set]
        if missing:
            raise ValueError(
                f'Requested x_ticks include K values not present in the data: {missing}. '
                f'Available K: {sorted(k_set)}.'
            )
        tick_positions = np.array(preferred, dtype=float)
    else:
        preferred = DEFAULT_X_TICKS_LOG if use_log_scale else DEFAULT_X_TICKS_LINEAR
        tick_positions = np.array([k for k in preferred if k in k_set], dtype=float)
        if len(tick_positions) == 0:
            n_show = min(8, n_k)
            indices = np.linspace(0, n_k - 1, n_show, dtype=int)
            tick_positions = k_vals[indices]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=panel_figsize(2),
        sharex=True,
        layout='constrained',
    )

    k_min, k_max = float(k_vals.min()), float(k_vals.max())
    if use_log_scale:
        x_lim = (k_min / 1.15, k_max * 1.15)
    else:
        x_margin = max(0.5, (k_max - k_min) * 0.02)
        x_lim = (k_min - x_margin, k_max + x_margin)

    for ax_idx, metric_name in enumerate(['Top 1', 'Top 10']):
        ax = axes[ax_idx]
        if draw_by_group:
            for line_idx, line_label in enumerate(hue_order):
                is_highlight = line_label in highlight_line_labels
                curve = (
                    df_long[df_long[hue_col] == line_label]
                    .groupby('K', sort=True)[metric_name]
                    .mean()
                )
                ax.plot(
                    curve.index.to_numpy(dtype=float),
                    curve.to_numpy(dtype=float),
                    label=line_label if ax_idx == 1 else None,
                    color=hue_palette[line_label],
                    linestyle=group_line_style(line_label, line_idx),
                    linewidth=2.2 if is_highlight else 1.6,
                    alpha=1.0 if is_highlight else 0.95,
                    zorder=4 if is_highlight else 2,
                )
        else:
            sns.lineplot(
                data=df_long,
                x='K',
                y=metric_name,
                hue=hue_col,
                hue_order=hue_order,
                palette=hue_palette,
                errorbar=None,
                marker=None,
                ax=ax,
                legend=(ax_idx == 1),
            )
            if ax_idx == 0 and ax.legend_ is not None:
                ax.legend_.remove()

        # Horizontal K=k_special lines are useful for epoch plots, but clutter group comparisons.
        if not draw_by_group:
            for _, row in df_k_special.iterrows():
                y_val = row[metric_name]
                line_label = row['Epoch']
                color = hue_palette.get(line_label, 'gray')
                ax.axhline(
                    y=y_val,
                    linestyle='--',
                    color=color,
                    label=f'{line_label}, K={k_special}' if ax_idx == 1 else None,
                    alpha=0.7,
                )

        ylabel = 'Top-10 accuracy' if '10' in metric_name else 'Top-1 accuracy'
        if validation_metrics:
            ylabel = f'{ylabel} (validation)'
        ax.set_ylabel(ylabel)
        ax.set_xlabel('' if ax_idx == 0 else 'K (unmix channels)')
        if use_log_scale:
            ax.set_xscale('log')
            ax.xaxis.set_minor_locator(mpl_ticker.NullLocator())
        ax.set_xlim(x_lim)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(int(k)) for k in tick_positions])
        plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
        ax.grid(True, axis='both', alpha=0.35, linestyle='-')
        ax.set_axisbelow(True)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

        # Y-axis scale focused on data range (with padding) so curves are visible
        y_min_val = df_long[metric_name].min()
        y_max_val = df_long[metric_name].max()
        y_range = y_max_val - y_min_val
        pad = max(y_range * 0.1, 0.02)
        ax.set_ylim(max(0.0, y_min_val - pad), min(1.0, y_max_val + pad))

    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend_kwargs = {
        'title': legend_heading,
        'framealpha': 0.95,
        'handlelength': 2.2,
        'columnspacing': 1.0,
        'labelspacing': 0.6,
    }
    if draw_by_group:
        axes[1].legend(
            by_label.values(),
            by_label.keys(),
            loc='lower right',
            ncol=2,
            **legend_kwargs,
        )
    else:
        axes[1].legend(
            by_label.values(),
            by_label.keys(),
            loc='best',
            **legend_kwargs,
        )

    scale_suffix = 'log' if use_log_scale else 'linear'
    save_path = Path(outdir) / f'metrics_vs_k_{run_group_label}-{scale_suffix}.pdf'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for metrics plotting."""

    def epoch_or_best(v: str):
        v = v.strip().lower()
        if v == 'best':
            return 'best'
        try:
            return int(v)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                "epochs must be integers and/or 'best'"
            ) from e

    p = argparse.ArgumentParser()
    p.add_argument(
        '--run-group',
        type=str,
        nargs='+',
        required=True,
        metavar='GROUP',
        help='One or more run group ids (experiments under experiments_root/<run_group>/). '
        'Configs must match except CONFIG_ALLOWED_DIFFERENCES (e.g. same architecture).',
    )
    p.add_argument(
        '--experiments-root',
        type=str,
        default=EXPERIMENTS_DIR,
        help='Root directory where experiments logs are stored',
    )
    p.add_argument(
        '--outdir',
        type=str,
        default=os.path.join(PLOTS_DIR, 'unmix_k_analysis'),
        help='Output directory for metrics plots',
    )
    p.add_argument(
        '--epochs',
        nargs='+',
        type=epoch_or_best,
        default=('best',),
        help="Epochs to plot (integers and/or 'best'). Default: best only.",
    )
    p.add_argument(
        '--output-name',
        type=str,
        default=None,
        metavar='NAME',
        help='Short name for the output file (metrics_vs_k_<name>-<scale>.pdf). '
        'If not set, run groups are joined with "-", which can be long for many groups.',
    )
    p.add_argument(
        '--line-labels',
        type=str,
        nargs='+',
        default=None,
        metavar='LABEL',
        help='Labels for run-group lines, in the same order as --run-group. '
        'Default: use run group ids.',
    )
    p.add_argument(
        '--legend-title',
        type=str,
        default=None,
        metavar='TITLE',
        help='Legend title. Example: "Temporal blocks".',
    )
    p.add_argument(
        '--highlight-run-group',
        type=str,
        default=None,
        metavar='GROUP',
        help='Optional run group id to draw above other run-group lines.',
    )
    p.add_argument(
        '--k-special',
        type=int,
        default=270,
        help='Special K value to highlight with horizontal lines (kept on main curve too)',
    )
    p.add_argument(
        '--log-scale',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use log scale for K on x-axis (default: True). Use --no-log-scale for linear.',
    )
    p.add_argument(
        '--x-ticks',
        nargs='*',
        type=int,
        default=None,
        metavar='K',
        help='K values to show on x-axis (only those present in data are shown). '
        'Default: scale-specific set (e.g. 1,10,25,... for linear; 1,5,10,... for log).',
    )
    p.add_argument(
        '--validation-metrics',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'For --epochs best, plot top1s_val/top10s_val at the min-loss_val '
            'checkpoint instead of final-test metrics. Numeric --epochs are unchanged.'
        ),
    )
    args = p.parse_args(argv)
    for rg in args.run_group:
        validate_run_group(rg)
    if args.highlight_run_group is not None:
        validate_run_group(args.highlight_run_group)
        if args.highlight_run_group not in args.run_group:
            raise ValueError(
                '--highlight-run-group must be one of the values passed to --run-group.'
            )
    if args.output_name is not None:
        validate_run_group(args.output_name)
    if args.line_labels is not None and len(args.line_labels) != len(args.run_group):
        raise ValueError(
            '--line-labels must contain exactly one label per --run-group '
            f'({len(args.run_group)} expected, got {len(args.line_labels)}).'
        )
    return args


def main_cli():
    """CLI entry point for metrics plotting."""
    args = parse_arguments()

    df_long, df_k_special, epoch_order = load_metrics(
        experiments_root=args.experiments_root,
        run_groups=args.run_group,
        epochs=args.epochs,
        k_special=args.k_special,
        validation_metrics=args.validation_metrics,
    )
    run_group_label = (
        args.output_name if args.output_name is not None else '-'.join(args.run_group)
    )
    if args.validation_metrics and 'validation' not in run_group_label.lower():
        run_group_label = f'{run_group_label}-validation'
    line_labels = args.line_labels if args.line_labels is not None else args.run_group
    label_by_group = dict(zip(args.run_group, line_labels))
    df_long['Line label'] = df_long['Run group'].map(label_by_group)
    df_k_special['Line label'] = df_k_special['Run group'].map(label_by_group)

    plot_metrics(
        df_long=df_long,
        df_k_special=df_k_special,
        epoch_order=epoch_order,
        outdir=args.outdir,
        run_group_label=run_group_label,
        line_order=line_labels,
        legend_title=args.legend_title,
        highlight_run_group=args.highlight_run_group,
        k_special=args.k_special,
        use_log_scale=args.log_scale,
        x_ticks=tuple(args.x_ticks) if args.x_ticks else None,
        validation_metrics=args.validation_metrics,
    )


if __name__ == '__main__':
    main_cli()
