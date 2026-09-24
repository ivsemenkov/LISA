"""Publication artifacts for the temporal-filter length ablation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lisa.plots.plot_style import FULL_WIDTH_IN, publication_style
from lisa.plots.temporal_filter_utils import demean_temporal_filters
from lisa.utils.constants import EXPERIMENTS_DIR
from lisa.utils.numeric import EPS


DEFAULT_KERNEL = 15
N_BRANCHES = 25
SEEDS = (42, 43, 44)
# The spectral and per-subject artifacts describe one trained model each, so
# they stay on a single seed; widening them is a separate decision.
SPECTRAL_SEED = 42
N_FFT = 4096
TOP1_COLOR = '#4C78A8'
TOP10_COLOR = '#F58518'

RUN_GROUPS = {
    1: 'ablate-tfkernel1-2conv-seed{seed}-paper',
    5: 'tfk5-2conv-seed{seed}-paper',
    9: 'tfk9-2conv-seed{seed}-paper',
    15: '2conv-seed{seed}-paper',
    19: 'tfk19-2conv-seed{seed}-paper',
    25: 'tfk25-2conv-seed{seed}-paper',
    29: 'tfk29-2conv-seed{seed}-paper',
    49: 'tfk49-2conv-seed{seed}-paper',
    65: 'tfk65-2conv-seed{seed}-paper',
    81: 'tfk81-2conv-seed{seed}-paper',
    101: 'tfk101-2conv-seed{seed}-paper',
    151: 'tfk151-2conv-seed{seed}-paper',
    201: 'tfk201-2conv-seed{seed}-paper',
    251: 'tfk251-2conv-seed{seed}-paper',
    299: 'tfk299-2conv-seed{seed}-paper',
}

WINDOW_ID_COLUMNS = [
    'dataset_idx',
    'subject_id',
    'session_id',
    'story_id',
    'sound_id',
    'segment_start',
]


@dataclass(frozen=True)
class RunData:
    kernel_size: int
    duration_ms: int
    seed: int
    run_dir: Path
    selected_epoch: int
    early_stop_epoch: int
    top1: float
    top10: float
    windows: pd.DataFrame
    subject_scores: pd.DataFrame
    temporal_filters: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build temporal-filter length ablation artifacts.'
    )
    parser.add_argument(
        '--experiments-root',
        type=Path,
        default=Path(EXPERIMENTS_DIR),
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path('outputs/plots/temporal_filter_length_ablation'),
    )
    return parser.parse_args()


def find_run_dir(
    experiments_root: Path,
    run_group: str,
    kernel_size: int,
    seed: int,
) -> tuple[Path, dict]:
    group_dir = experiments_root / run_group
    if not group_dir.is_dir():
        raise FileNotFoundError(f'Missing experiment group: {group_dir}')

    matches: list[tuple[Path, dict]] = []
    for run_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
        config_path = run_dir / 'config.json'
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding='utf-8'))
        if (
            config.get('temporal_filter_kernel_size') == kernel_size
            and config.get('n_channels_unmix') == N_BRANCHES
            and config.get('n_channels_block') == N_BRANCHES
            and config.get('seed') == seed
        ):
            matches.append((run_dir, config))

    if len(matches) != 1:
        raise ValueError(
            f'Expected one {N_BRANCHES}-branch K={kernel_size} seed-{seed} run '
            f'in {group_dir}, found {len(matches)}.'
        )
    return matches[0]


def validate_config(config: dict, kernel_size: int, seed: int, run_dir: Path) -> None:
    expected = {
        'seed': seed,
        'meg_sr': 100,
        'temporal_filter_kernel_size': kernel_size,
        'temporal_filter_type': 'conv',
        'n_channels_unmix': N_BRANCHES,
        'n_channels_block': N_BRANCHES,
        'n_temporal_module_blocks': 2,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f'Unexpected config values in {run_dir}: {mismatches}')


def load_run(
    experiments_root: Path,
    kernel_size: int,
    run_group: str,
    seed: int,
) -> RunData:
    run_dir, config = find_run_dir(experiments_root, run_group, kernel_size, seed)
    validate_config(config, kernel_size, seed, run_dir)

    metrics = pd.read_csv(run_dir / 'metrics.csv')
    selected_rows = metrics[metrics['loss_final_test'].notna()]
    if len(selected_rows) != 1:
        raise ValueError(
            f'Expected one final-test row in {run_dir / "metrics.csv"}, '
            f'found {len(selected_rows)}.'
        )
    selected = selected_rows.iloc[0]
    early_stop_rows = metrics[metrics['early_stop_epoch'].notna()]
    if len(early_stop_rows) != 1:
        raise ValueError(
            f'Expected one early-stop row in {run_dir / "metrics.csv"}, '
            f'found {len(early_stop_rows)}.'
        )

    window_columns = [*WINDOW_ID_COLUMNS, 'rank_of_true']
    windows = pd.read_csv(
        run_dir / 'final_test_per_window.csv',
        usecols=window_columns,
    )
    windows['top1'] = (windows['rank_of_true'] == 0).astype(float)
    windows['top10'] = (windows['rank_of_true'] < 10).astype(float)
    top1 = float(windows['top1'].mean() * 100.0)
    top10 = float(windows['top10'].mean() * 100.0)
    if not np.isclose(top1 / 100.0, float(selected['top1s_final_test'])):
        raise ValueError(f'Top-1 mismatch between metrics and windows in {run_dir}.')
    if not np.isclose(top10 / 100.0, float(selected['top10s_final_test'])):
        raise ValueError(f'Top-10 mismatch between metrics and windows in {run_dir}.')

    subject_scores = (
        windows.groupby('subject_id', sort=True)[['top1', 'top10']].mean() * 100.0
    )

    filter_paths = sorted(run_dir.glob('filters_*.npz'))
    if len(filter_paths) != 1:
        raise ValueError(
            f'Expected one filters NPZ in {run_dir}, found {len(filter_paths)}.'
        )
    with np.load(filter_paths[0], allow_pickle=False) as data:
        temporal_filters = np.asarray(
            data['temporal_filters_weight'],
            dtype=np.float64,
        )
    expected_shape = (N_BRANCHES, kernel_size)
    if temporal_filters.shape != expected_shape:
        raise ValueError(
            f'Expected temporal filters {expected_shape}, got '
            f'{temporal_filters.shape} in {filter_paths[0]}.'
        )

    return RunData(
        kernel_size=kernel_size,
        duration_ms=kernel_size * 10,
        seed=seed,
        run_dir=run_dir,
        selected_epoch=int(selected['epoch']),
        early_stop_epoch=int(early_stop_rows.iloc[0]['early_stop_epoch']),
        top1=top1,
        top10=top10,
        windows=windows,
        subject_scores=subject_scores,
        temporal_filters=temporal_filters,
    )


def validate_paired_queries(runs: list[RunData]) -> None:
    reference = runs[0].windows[WINDOW_ID_COLUMNS].reset_index(drop=True)
    reference_subjects = runs[0].subject_scores.index
    for run in runs[1:]:
        candidate = run.windows[WINDOW_ID_COLUMNS].reset_index(drop=True)
        if not reference.equals(candidate):
            raise ValueError(
                f'Test queries differ between {runs[0].run_dir} and {run.run_dir}.'
            )
        if not reference_subjects.equals(run.subject_scores.index):
            raise ValueError(f'Subject sets differ in {run.run_dir}.')


def build_result_table(runs: list[RunData]) -> pd.DataFrame:
    # Deltas are paired within seed: every run is compared with the default
    # kernel trained from the same seed, never with a pooled baseline.
    baselines = {run.seed: run for run in runs if run.kernel_size == DEFAULT_KERNEL}
    if set(baselines) != set(SEEDS):
        raise ValueError(
            f'Missing K={DEFAULT_KERNEL} runs for seeds '
            f'{sorted(set(SEEDS) - set(baselines))}.'
        )
    rows = []
    for run in runs:
        baseline = baselines[run.seed]
        rows.append(
            {
                'kernel_size': run.kernel_size,
                'duration_ms': run.duration_ms,
                'seed': run.seed,
                'no_temporal_context': run.kernel_size == 1,
                'is_default': run.kernel_size == DEFAULT_KERNEL,
                'top1_percent': run.top1,
                'delta_top1_pp': run.top1 - baseline.top1,
                'top10_percent': run.top10,
                'delta_top10_pp': run.top10 - baseline.top10,
                'n_subjects': len(run.subject_scores),
                'n_windows': len(run.windows),
                'selected_epoch': run.selected_epoch,
                'early_stop_epoch': run.early_stop_epoch,
                'run_dir': str(run.run_dir),
            }
        )
    return (
        pd.DataFrame(rows).sort_values(['kernel_size', 'seed']).reset_index(drop=True)
    )


def build_summary_table(table: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-seed rows to one row per kernel size."""
    summary = table.groupby(
        ['kernel_size', 'duration_ms', 'no_temporal_context', 'is_default'],
        as_index=False,
    ).agg(
        top1_percent_mean=('top1_percent', 'mean'),
        top1_percent_min=('top1_percent', 'min'),
        top1_percent_max=('top1_percent', 'max'),
        top10_percent_mean=('top10_percent', 'mean'),
        top10_percent_min=('top10_percent', 'min'),
        top10_percent_max=('top10_percent', 'max'),
        delta_top1_pp_mean=('delta_top1_pp', 'mean'),
        delta_top1_pp_min=('delta_top1_pp', 'min'),
        delta_top1_pp_max=('delta_top1_pp', 'max'),
        delta_top10_pp_mean=('delta_top10_pp', 'mean'),
        delta_top10_pp_min=('delta_top10_pp', 'min'),
        delta_top10_pp_max=('delta_top10_pp', 'max'),
        n_seeds=('seed', 'nunique'),
    )
    if not (summary['n_seeds'] == len(SEEDS)).all():
        raise ValueError(f'Every kernel size needs {len(SEEDS)} seeds:\n{summary}')
    return summary.sort_values('kernel_size').reset_index(drop=True)


def build_subject_delta_table(runs: list[RunData]) -> pd.DataFrame:
    baseline = next(run for run in runs if run.kernel_size == DEFAULT_KERNEL)
    rows = []
    for run in runs:
        deltas = run.subject_scores - baseline.subject_scores
        for subject_id, values in deltas.iterrows():
            rows.append(
                {
                    'subject_id': int(subject_id),
                    'kernel_size': run.kernel_size,
                    'duration_ms': run.duration_ms,
                    'delta_top1_pp': float(values['top1']),
                    'delta_top10_pp': float(values['top10']),
                }
            )
    return pd.DataFrame(rows)


def latex_metric_table(
    table: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    metric: str,
    metric_label: str,
    tex_label: str,
) -> str:
    per_seed = table.pivot(
        index='kernel_size', columns='seed', values=f'{metric}_percent'
    )
    mean_column = f'{metric}_percent_mean'
    delta_column = f'delta_{metric}_pp_mean'
    best_mean = float(summary[mean_column].max())
    seed_headers = ' & '.join(rf'Seed {seed} (\%)' for seed in SEEDS)
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        (
            rf'\caption{{Effect of temporal-filter support on {metric_label} '
            r'retrieval accuracy, for three training seeds. All models use 25 '
            r'branches. Deltas are paired within seed, each run against the '
            r'150-ms run of the same seed.}'
        ),
        rf'\label{{{tex_label}}}',
        r'\begin{tabular}{rr' + 'r' * len(SEEDS) + r'rr}',
        r'\toprule',
        (
            rf'Kernel & Support (ms) & {seed_headers} & Mean (\%) '
            r'& $\Delta$Mean (pp) \\'
        ),
        r'\midrule',
    ]
    for row in summary.itertuples(index=False):
        kernel = str(row.kernel_size)
        if row.is_default:
            kernel += r' (default)'
        support = str(row.duration_ms)
        if row.no_temporal_context:
            support += r'\textsuperscript{a}'
        seed_values = ' & '.join(
            f'{per_seed.loc[row.kernel_size, seed]:.2f}' for seed in SEEDS
        )
        mean_value = getattr(row, mean_column)
        mean_text = f'{mean_value:.2f}'
        if np.isclose(mean_value, best_mean):
            mean_text = rf'\textbf{{{mean_text}}}'
        lines.append(
            f'{kernel} & {support} & {seed_values} & {mean_text} '
            f'& {getattr(row, delta_column):+.2f} \\\\'
        )
    lines.extend(
        [
            r'\bottomrule',
            r'\end{tabular}',
            (
                r'\vspace{2pt}\parbox{\linewidth}{\footnotesize '
                r'\textsuperscript{a}A one-tap filter has no temporal context; '
                r'it only rescales each branch.}'
            ),
            r'\end{table}',
        ]
    )
    return '\n'.join(lines)


def write_latex_table(table: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    blocks = [
        latex_metric_table(
            table,
            summary,
            metric='top1',
            metric_label='Top-1',
            tex_label='tab:temporal-filter-length-top1',
        ),
        latex_metric_table(
            table,
            summary,
            metric='top10',
            metric_label='Top-10',
            tex_label='tab:temporal-filter-length-top10',
        ),
    ]
    path.write_text('\n\n'.join(blocks) + '\n', encoding='utf-8')


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for suffix in ['.pdf', '.png']:
        fig.savefig(
            out_dir / f'{stem}{suffix}',
            bbox_inches='tight',
            dpi=300,
        )
    plt.close(fig)


def x_tick_labels(table: pd.DataFrame) -> list[str]:
    return [
        '1 tap' if row.no_temporal_context else str(row.duration_ms)
        for row in table.itertuples(index=False)
    ]


@publication_style
def plot_performance_delta(summary: pd.DataFrame) -> plt.Figure:
    """Mean change per kernel size, with the seed range shaded behind it."""
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 3.5))
    x = summary['duration_ms'].to_numpy()
    for metric, label, color, marker in [
        ('delta_top1_pp', 'Top-1', TOP1_COLOR, 'o'),
        ('delta_top10_pp', 'Top-10', TOP10_COLOR, 's'),
    ]:
        # With three seeds the band edges are the best and worst runs.
        ax.fill_between(
            x,
            summary[f'{metric}_min'],
            summary[f'{metric}_max'],
            color=color,
            alpha=0.18,
            linewidth=0,
            label='_nolegend_',
            zorder=2,
        )
        ax.plot(
            x,
            summary[f'{metric}_mean'],
            color=color,
            marker=marker,
            markersize=5.5,
            linewidth=1.8,
            label=label,
            zorder=3,
        )
    ax.axhline(0.0, color='0.35', linestyle='--', linewidth=1.0, zorder=1)
    ax.set_xscale('log')
    ax.set_xticks(x, x_tick_labels(summary), rotation=90, ha='center')
    ax.minorticks_off()
    ax.set_xlabel('Temporal-filter support (ms; 100 Hz)')
    ax.set_ylabel('Change vs 150-ms default (pp)')
    ax.legend(frameon=False, ncol=2, loc='upper left')
    fig.tight_layout()
    return fig


@publication_style
def plot_subject_spread(
    table: pd.DataFrame,
    subject_deltas: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(FULL_WIDTH_IN, 5.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    positions = np.arange(len(table), dtype=float)
    position_by_kernel = dict(zip(table['kernel_size'], positions, strict=True))
    rng = np.random.default_rng(42)
    subject_ids = sorted(subject_deltas['subject_id'].unique())
    jitter_by_subject = {
        subject_id: jitter
        for subject_id, jitter in zip(
            subject_ids,
            rng.uniform(-0.13, 0.13, len(subject_ids)),
            strict=True,
        )
    }

    for ax, metric, label, color in [
        (axes[0], 'delta_top1_pp', 'Top-1 change (pp)', TOP1_COLOR),
        (axes[1], 'delta_top10_pp', 'Top-10 change (pp)', TOP10_COLOR),
    ]:
        for row in table.itertuples(index=False):
            values = subject_deltas[subject_deltas['kernel_size'] == row.kernel_size]
            xpos = np.asarray(
                [
                    position_by_kernel[row.kernel_size] + jitter_by_subject[subject_id]
                    for subject_id in values['subject_id']
                ]
            )
            ax.scatter(
                xpos,
                values[metric],
                s=9,
                color=color,
                alpha=0.24,
                linewidths=0,
                zorder=2,
            )
            q25, median, q75 = np.quantile(values[metric], [0.25, 0.5, 0.75])
            center = position_by_kernel[row.kernel_size]
            ax.vlines(center, q25, q75, color='0.2', linewidth=2.2, zorder=3)
            ax.hlines(
                median,
                center - 0.10,
                center + 0.10,
                color='0.1',
                linewidth=1.4,
                zorder=4,
            )
        ax.plot(
            positions,
            table[metric],
            color=color,
            marker='D',
            markersize=4.5,
            linewidth=1.6,
            zorder=5,
        )
        ax.axhline(0.0, color='0.45', linestyle='--', linewidth=1.0, zorder=1)
        ax.set_ylabel(label)

    max_abs_delta = float(
        np.abs(subject_deltas[['delta_top1_pp', 'delta_top10_pp']].to_numpy()).max()
    )
    shared_limit = max(1.0, 1.05 * max_abs_delta)
    axes[0].set_ylim(-shared_limit, shared_limit)

    axes[-1].set_xticks(positions, x_tick_labels(table), rotation=90, ha='center')
    axes[-1].set_xlabel('Temporal-filter support (ms; 100 Hz)')
    return fig


def normalized_magnitude_spectra(
    temporal_filters: np.ndarray,
    *,
    demean: bool,
    fs: float = 100.0,
    n_fft: int = N_FFT,
) -> tuple[np.ndarray, np.ndarray] | None:
    filters = np.asarray(temporal_filters, dtype=np.float64)
    if demean:
        if filters.shape[1] == 1:
            return None
        filters = demean_temporal_filters(filters)
    spectra = np.abs(np.fft.rfft(filters, n=n_fft, axis=1))
    spectra /= np.maximum(spectra.max(axis=1, keepdims=True), EPS)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    return frequencies, spectra


def sort_spectra_by_centroid(
    frequencies: np.ndarray,
    spectra: np.ndarray,
) -> np.ndarray:
    centroid = (spectra * frequencies[None, :]).sum(axis=1) / np.maximum(
        spectra.sum(axis=1),
        EPS,
    )
    return spectra[np.argsort(centroid)]


def validate_one_tap_response(run: RunData) -> None:
    if run.kernel_size != 1:
        raise ValueError(f'Expected a one-tap run, got K={run.kernel_size}.')
    raw_result = normalized_magnitude_spectra(
        run.temporal_filters,
        demean=False,
    )
    if raw_result is None:
        raise ValueError('Raw one-tap response must be defined.')
    _, raw_spectra = raw_result
    if not np.allclose(raw_spectra, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(
            'Peak-normalized one-tap responses must equal one at every frequency.'
        )
    demeaned_filters = run.temporal_filters - run.temporal_filters.mean(
        axis=1, keepdims=True
    )
    if not np.allclose(demeaned_filters, 0.0, rtol=0.0, atol=1e-12):
        raise ValueError('Mean-subtracted one-tap kernels must be identically zero.')
    if normalized_magnitude_spectra(run.temporal_filters, demean=True) is not None:
        raise ValueError('Mean-subtracted one-tap response must be undefined.')


@publication_style
def plot_frequency_medians(
    runs: list[RunData],
    *,
    demean: bool,
) -> plt.Figure:
    spectral_runs = [run for run in runs if run.kernel_size > 1]
    fig, ax = plt.subplots(
        figsize=(FULL_WIDTH_IN, 3.8),
        constrained_layout=True,
    )
    colors = plt.get_cmap('viridis')(np.linspace(0.05, 0.95, len(spectral_runs)))
    for run, color in zip(spectral_runs, colors, strict=True):
        result = normalized_magnitude_spectra(
            run.temporal_filters,
            demean=demean,
        )
        if result is None:
            raise ValueError(f'Unexpected undefined response for {run.run_dir}.')
        frequencies, spectra = result
        ax.plot(
            frequencies,
            np.median(spectra, axis=0),
            color=color,
            label=f'{run.kernel_size} taps ({run.duration_ms} ms)',
        )
    ax.set_xlim(0.0, 50.0)
    ax.set_ylim(0.0, 1.03)
    ax.set_xticks(np.arange(0.0, 51.0, 10.0))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(r'Median normalized amplitude $|H(f)|$')
    ax.legend(
        frameon=False,
        ncol=3,
        loc='upper right',
    )
    return fig


def _heatmap_grid(n_panels: int) -> tuple[int, int]:
    n_cols = 4
    n_rows = int(np.ceil(n_panels / n_cols))
    return n_rows, n_cols


@publication_style
def plot_frequency_heatmaps(
    runs: list[RunData],
    *,
    demean: bool,
) -> plt.Figure:
    spectral_runs = [run for run in runs if run.kernel_size > 1]
    n_rows, n_cols = _heatmap_grid(len(spectral_runs))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(FULL_WIDTH_IN, 2.35 * n_rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    image = None
    flat_axes = axes.ravel()
    if len(spectral_runs) > len(flat_axes):
        raise ValueError(
            f'Heatmap grid {n_rows}x{n_cols} cannot hold {len(spectral_runs)} runs.'
        )
    for ax, run in zip(flat_axes, spectral_runs, strict=False):
        result = normalized_magnitude_spectra(
            run.temporal_filters,
            demean=demean,
        )
        if result is None:
            raise ValueError(f'Unexpected undefined response for {run.run_dir}.')
        frequencies, spectra = result
        spectra = sort_spectra_by_centroid(frequencies, spectra)
        ax.set_title(f'{run.kernel_size} taps ({run.duration_ms} ms)')
        image = ax.imshow(
            spectra,
            origin='lower',
            aspect='auto',
            extent=[0.0, 50.0, 0.5, N_BRANCHES + 0.5],
            cmap='magma',
            vmin=0.0,
            vmax=1.0,
            interpolation='nearest',
        )
        ax.set_xlim(0.0, 50.0)
        ax.set_xticks(np.arange(0.0, 51.0, 10.0))
        ax.set_yticks([1, 13, 25])
    for ax in axes[-1, :]:
        ax.set_xlabel('Frequency (Hz)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Branch')
    for ax in flat_axes[len(spectral_runs) :]:
        ax.set_axis_off()
    if image is not None:
        colorbar = fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            shrink=0.88,
            pad=0.02,
        )
        colorbar.set_label(r'Normalized amplitude $|H(f)|$')
    return fig


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        load_run(args.experiments_root, kernel_size, run_group.format(seed=seed), seed)
        for kernel_size, run_group in RUN_GROUPS.items()
        for seed in SEEDS
    ]
    runs.sort(key=lambda run: (run.kernel_size, run.seed))
    validate_paired_queries(runs)
    for seed in SEEDS:
        validate_one_tap_response(
            next(run for run in runs if run.kernel_size == 1 and run.seed == seed)
        )

    table = build_result_table(runs)
    summary = build_summary_table(table)

    # Per-subject and spectral artifacts describe one model each, so they use a
    # single seed and keep the meaning they had before.
    spectral_runs = [run for run in runs if run.seed == SPECTRAL_SEED]
    spectral_table = table[table['seed'] == SPECTRAL_SEED].reset_index(drop=True)
    subject_deltas = build_subject_delta_table(spectral_runs)

    table.to_csv(
        args.out_dir / 'temporal_filter_ablation_table.csv',
        index=False,
        float_format='%.6f',
    )
    summary.to_csv(
        args.out_dir / 'temporal_filter_ablation_summary.csv',
        index=False,
        float_format='%.6f',
    )
    write_latex_table(
        table,
        summary,
        args.out_dir / 'temporal_filter_ablation_table.txt',
    )
    subject_deltas.to_csv(
        args.out_dir / 'temporal_filter_subject_deltas.csv',
        index=False,
        float_format='%.6f',
    )

    save_figure(
        plot_performance_delta(summary),
        args.out_dir,
        'temporal_filter_performance_delta',
    )
    save_figure(
        plot_subject_spread(spectral_table, subject_deltas),
        args.out_dir,
        'temporal_filter_subject_spread',
    )
    save_figure(
        plot_frequency_medians(spectral_runs, demean=False),
        args.out_dir,
        'temporal_filter_frequency_medians_raw',
    )
    save_figure(
        plot_frequency_medians(spectral_runs, demean=True),
        args.out_dir,
        'temporal_filter_frequency_medians_demeaned',
    )
    save_figure(
        plot_frequency_heatmaps(spectral_runs, demean=False),
        args.out_dir,
        'temporal_filter_frequency_heatmaps_raw',
    )
    save_figure(
        plot_frequency_heatmaps(spectral_runs, demean=True),
        args.out_dir,
        'temporal_filter_frequency_heatmaps_demeaned',
    )

    print(f'Wrote temporal-filter ablation artifacts to {args.out_dir}')


if __name__ == '__main__':
    main()
