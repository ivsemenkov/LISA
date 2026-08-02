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
from lisa.utils.numeric import EPS


DEFAULT_KERNEL = 15
N_BRANCHES = 25
TARGET_SEED = 42
N_FFT = 4096
TOP1_COLOR = '#4C78A8'
TOP10_COLOR = '#F58518'

RUN_GROUPS = {
    1: 'ablate-tfkernel1-2conv-seed42-paper',
    5: 'tfk5-2conv-seed42-paper',
    9: 'tfk9-2conv-seed42-paper',
    15: '2conv-seed42-paper',
    19: 'tfk19-2conv-seed42-paper',
    25: 'tfk25-2conv-seed42-paper',
    29: 'tfk29-2conv-seed42-paper',
    49: 'tfk49-2conv-seed42-paper',
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
        default=Path('outputs/experiments'),
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
        ):
            matches.append((run_dir, config))

    if len(matches) != 1:
        raise ValueError(
            f'Expected one {N_BRANCHES}-branch K={kernel_size} run in '
            f'{group_dir}, found {len(matches)}.'
        )
    return matches[0]


def validate_config(config: dict, kernel_size: int, run_dir: Path) -> None:
    expected = {
        'seed': TARGET_SEED,
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
) -> RunData:
    run_dir, config = find_run_dir(experiments_root, run_group, kernel_size)
    validate_config(config, kernel_size, run_dir)

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
    baseline = next(run for run in runs if run.kernel_size == DEFAULT_KERNEL)
    rows = []
    for run in runs:
        rows.append(
            {
                'kernel_size': run.kernel_size,
                'duration_ms': run.duration_ms,
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
    return pd.DataFrame(rows).sort_values('kernel_size').reset_index(drop=True)


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


def write_latex_table(table: pd.DataFrame, path: Path) -> None:
    best_top1 = float(table['top1_percent'].max())
    best_top10 = float(table['top10_percent'].max())
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        (
            r'\caption{Effect of temporal-filter support on retrieval accuracy. '
            r'All models use 25 branches and the same deterministic training seed.}'
        ),
        r'\label{tab:temporal-filter-length}',
        r'\begin{tabular}{rrrrrr}',
        r'\toprule',
        (
            r'Kernel & Support (ms) & Top-1 (\%) & $\Delta$Top-1 (pp) '
            r'& Top-10 (\%) & $\Delta$Top-10 (pp) \\'
        ),
        r'\midrule',
    ]
    for row in table.itertuples(index=False):
        kernel = str(row.kernel_size)
        if row.is_default:
            kernel += r' (default)'
        support = str(row.duration_ms)
        if row.no_temporal_context:
            support += r'\textsuperscript{a}'
        top1 = f'{row.top1_percent:.2f}'
        top10 = f'{row.top10_percent:.2f}'
        if np.isclose(row.top1_percent, best_top1):
            top1 = rf'\textbf{{{top1}}}'
        if np.isclose(row.top10_percent, best_top10):
            top10 = rf'\textbf{{{top10}}}'
        lines.append(
            f'{kernel} & {support} & {top1} & {row.delta_top1_pp:+.2f} '
            f'& {top10} & {row.delta_top10_pp:+.2f} \\\\'
        )
    lines.extend(
        [
            r'\bottomrule',
            r'\end{tabular}',
            (
                r'\vspace{2pt}\parbox{\linewidth}{\footnotesize '
                r'\textsuperscript{a}A one-tap filter has no temporal context; '
                r'it only rescales each branch. Deltas are relative to the '
                r'150-ms default.}'
            ),
            r'\end{table}',
        ]
    )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


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
def plot_performance_delta(table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 3.5))
    x = table['duration_ms'].to_numpy()
    for metric, label, color, marker in [
        ('delta_top1_pp', 'Top-1', TOP1_COLOR, 'o'),
        ('delta_top10_pp', 'Top-10', TOP10_COLOR, 's'),
    ]:
        ax.plot(
            x,
            table[metric],
            color=color,
            marker=marker,
            markersize=5.5,
            linewidth=1.8,
            label=label,
            zorder=3,
        )
    ax.axhline(0.0, color='0.35', linestyle='--', linewidth=1.0, zorder=1)
    ax.axvline(
        DEFAULT_KERNEL * 10,
        color='0.55',
        linestyle=':',
        linewidth=1.0,
        zorder=1,
    )
    ax.set_xticks(x, x_tick_labels(table))
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

    axes[-1].set_xticks(positions, x_tick_labels(table))
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
        figsize=(FULL_WIDTH_IN, 3.4),
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
        ncol=2,
        loc='upper right',
    )
    return fig


@publication_style
def plot_frequency_heatmaps(
    runs: list[RunData],
    *,
    demean: bool,
) -> plt.Figure:
    spectral_runs = [run for run in runs if run.kernel_size > 1]
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(FULL_WIDTH_IN, 4.7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    flat_axes = axes.ravel()
    for ax, run in zip(flat_axes, spectral_runs):
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
    for ax in flat_axes[len(spectral_runs) :]:
        ax.set_axis_off()

    for ax in axes[-1, :]:
        ax.set_xlabel('Frequency (Hz)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Branch')
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
        load_run(args.experiments_root, kernel_size, run_group)
        for kernel_size, run_group in RUN_GROUPS.items()
    ]
    runs.sort(key=lambda run: run.kernel_size)
    validate_paired_queries(runs)
    one_tap_run = next(run for run in runs if run.kernel_size == 1)
    validate_one_tap_response(one_tap_run)

    table = build_result_table(runs)
    subject_deltas = build_subject_delta_table(runs)
    table.to_csv(
        args.out_dir / 'temporal_filter_ablation_table.csv',
        index=False,
        float_format='%.6f',
    )
    write_latex_table(
        table,
        args.out_dir / 'temporal_filter_ablation_table.txt',
    )
    subject_deltas.to_csv(
        args.out_dir / 'temporal_filter_subject_deltas.csv',
        index=False,
        float_format='%.6f',
    )

    save_figure(
        plot_performance_delta(table),
        args.out_dir,
        'temporal_filter_performance_delta',
    )
    save_figure(
        plot_subject_spread(table, subject_deltas),
        args.out_dir,
        'temporal_filter_subject_spread',
    )
    save_figure(
        plot_frequency_medians(runs, demean=False),
        args.out_dir,
        'temporal_filter_frequency_medians_raw',
    )
    save_figure(
        plot_frequency_medians(runs, demean=True),
        args.out_dir,
        'temporal_filter_frequency_medians_demeaned',
    )
    save_figure(
        plot_frequency_heatmaps(runs, demean=False),
        args.out_dir,
        'temporal_filter_frequency_heatmaps_raw',
    )
    save_figure(
        plot_frequency_heatmaps(runs, demean=True),
        args.out_dir,
        'temporal_filter_frequency_heatmaps_demeaned',
    )

    print(f'Wrote temporal-filter ablation artifacts to {args.out_dir}')


if __name__ == '__main__':
    main()
