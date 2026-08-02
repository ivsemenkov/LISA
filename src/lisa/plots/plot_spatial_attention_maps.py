"""Plot overall spatial-attention maps as a grid of MEG topographies.

For each run, the spatial-attention layer exposes a softmax attention matrix of
shape (n_attention, n_channels). Averaging over the attention outputs produces a
per-sensor percentage distribution that is rendered as a topomap.

The grid has the number of unmixing branches (n_channels_unmix) on the y-axis and
the number of temporal-module blocks on the x-axis. Each blocks value maps to a
run group via a name template (e.g. "{blocks}conv-seed42-paper"); within a group,
runs are selected by their n_channels_unmix.
"""

import argparse
import json
import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import torch

from lisa.data.meg_io import load_raw_meg
from lisa.model.load_model import load_from_config
from lisa.plots.plot_style import (
    PUBLICATION_RC,
    grid_geometry,
    paper_topomap,
    publication_style,
)
from lisa.utils.constants import CLEAN_DATA_DIR, EXPERIMENTS_DIR, PLOTS_DIR
from lisa.utils.validators import validate_run_group


def discover_branch_runs(group_dir: str, expected_blocks: int) -> dict[int, str]:
    """Map n_channels_unmix -> run_dir for one group.

    Asserts every run's n_temporal_module_blocks equals expected_blocks so a
    mislabeled group name (or a wrong --blocks value) fails loudly instead of
    silently plotting the wrong model.
    """
    mapping: dict[int, str] = {}
    for config_path in glob(os.path.join(group_dir, '*', 'config.json')):
        with open(config_path, encoding='utf-8') as f:
            cfg = json.load(f)
        run_dir = os.path.dirname(config_path)
        blocks = int(cfg['n_temporal_module_blocks'])
        if blocks != expected_blocks:
            raise ValueError(
                f'Run {run_dir} has n_temporal_module_blocks={blocks} but the group '
                f'{group_dir} is expected to hold blocks={expected_blocks}. '
                'Check the --group-template / --blocks mapping.'
            )
        branch = int(cfg['n_channels_unmix'])
        if branch in mapping:
            raise ValueError(
                f'Multiple runs in {group_dir} share n_channels_unmix={branch}: '
                f'{mapping[branch]} and {run_dir}.'
            )
        mapping[branch] = run_dir
    if not mapping:
        raise FileNotFoundError(f'No runs with config.json found under {group_dir}.')
    return mapping


def load_attention_matrix(run_dir: str) -> np.ndarray:
    """Load a run and return its softmax spatial-attention matrix.

    Returns:
        Attention matrix with shape (n_attention, n_channels).
    """
    with open(os.path.join(run_dir, 'config.json'), encoding='utf-8') as f:
        hyper_params = json.load(f)
    checkpoint_path = os.path.join(run_dir, f'{hyper_params["checkpoint"]}.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    model, _ = load_from_config(
        hyper_params=hyper_params,
        checkpoint_path=checkpoint_path,
        device=torch.device('cpu'),
    )
    model.eval()

    attention = model.spatial_module.self_attention
    if attention is None:
        raise ValueError(
            f'Run {run_dir} has no spatial attention layer '
            f'(use_spatial_attention={hyper_params["use_spatial_attention"]!r}).'
        )

    # (n_attention, n_channels), softmax over channels.
    return attention.get_spatial_filter().numpy()


def _output_stem(
    name: str,
    group_template: str,
    blocks: list[int],
    branches: list[int],
) -> str:
    """Build a filesystem-safe stem that identifies the run group and grid axes."""
    template_slug = group_template.replace('{blocks}', 'N').replace('/', '_')
    name_slug = name.replace('/', '_')
    blocks_slug = '-'.join(str(block) for block in blocks)
    branches_slug = '-'.join(str(branch) for branch in branches)
    return f'{name_slug}__{template_slug}__blocks{blocks_slug}__branches{branches_slug}'


_NORM_LABELS = {
    'linear': 'Mean attention weight (%)',
    'sqrt': 'Mean attention weight (square-root scale)',
    'log': 'Mean attention weight (log scale)',
}


def apply_norm(vec: np.ndarray, norm: str) -> np.ndarray:
    """Apply the display transform to a (non-negative) attention vector."""
    if norm == 'linear':
        return vec
    if norm == 'sqrt':
        return np.sqrt(vec)
    if norm == 'log':
        return np.log1p(vec)
    raise ValueError(f'Unknown norm: {norm!r}')


def effective_sensor_count(vec: np.ndarray) -> float:
    """Return the inverse-Simpson effective number of sensors."""
    weights = np.asarray(vec, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError(f'Expected a 1D attention vector, got shape {weights.shape}.')
    if weights.size == 0:
        raise ValueError('Cannot compute spatial metrics for an empty vector.')
    if np.any(weights < 0):
        raise ValueError('Spatial metrics expect non-negative attention weights.')

    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError('Spatial metrics expect a positive total attention mass.')

    probabilities = weights / total
    hhi = float(np.sum(probabilities**2))
    return 1.0 / hhi


def _topomap_metric_label(vec: np.ndarray) -> str:
    return rf'$N_{{\mathrm{{eff}}}} = {effective_sensor_count(vec):.0f}$'


@publication_style
def plot_attention_grid(
    maps: dict[tuple[int, int], np.ndarray],
    blocks: list[int],
    branches: list[int],
    info: mne.Info,
    cmap: str,
    scale: str,
    norm: str,
    vmax_percentile: float,
    metric_basis: str,
) -> plt.Figure:
    """Render the (blocks x branches) grid of attention topographies.

    Args:
        scale: 'shared' for one color scale across the grid (comparable across
            cells) or 'per_cell' to autoscale each topomap independently.
        norm: 'linear', 'sqrt', or 'log' (log1p) transform applied before display.
        vmax_percentile: Upper color limit percentile (e.g. 99 clips the brightest
            sensors so a few hotspots do not wash out the map; 100 uses the max).
        metric_basis: 'display' to compute effective sensor count after display
            transform and color-limit clipping, or 'raw' to compute it from the
            untransformed mean attention weights.
    """
    present = [vec for vec in maps.values() if vec is not None]
    if not present:
        raise ValueError('No attention maps available to plot.')
    if present[0].shape[0] != info['nchan']:
        raise ValueError(
            f'Attention vector has {present[0].shape[0]} channels but the MEG info '
            f'has {info["nchan"]}. The Raw used for topomap positions must match the '
            'model channel order.'
        )

    transformed = {
        key: apply_norm(vec, norm) for key, vec in maps.items() if vec is not None
    }
    shared_vmax = float(
        np.percentile(
            np.concatenate([t.ravel() for t in transformed.values()]), vmax_percentile
        )
    )

    nrows, ncols = len(branches), len(blocks)
    geometry = grid_geometry(nrows, ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=geometry.figsize,
        squeeze=False,
        layout='constrained',
    )
    fig.set_constrained_layout_pads(
        w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02
    )

    last_im = None
    for i, branch in enumerate(branches):
        for j, block in enumerate(blocks):
            ax = axes[i][j]
            raw_vec = maps.get((block, branch))
            vec = transformed.get((block, branch))
            if vec is None:
                ax.text(0.5, 0.5, 'Not available', ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                if scale == 'shared':
                    vmax = shared_vmax
                else:
                    vmax = float(np.percentile(vec, vmax_percentile))
                vmax = max(vmax, 1e-12)
                metric_vec = (
                    raw_vec if metric_basis == 'raw' else np.clip(vec, 0.0, vmax)
                )
                last_im, _ = paper_topomap(
                    vec,
                    info,
                    kind='attention',
                    axes=ax,
                    show=False,
                    cmap=cmap,
                    vlim=(0.0, vmax),
                )
                ax.text(
                    0.5,
                    -0.045,
                    _topomap_metric_label(metric_vec),
                    transform=ax.transAxes,
                    ha='center',
                    va='top',
                    fontsize=geometry.annotation_fontsize * 0.85,
                    clip_on=False,
                )
                if i == 0:
                    ax.set_title(str(block), pad=4)
            if vec is None and i == 0:
                ax.set_title(str(block), pad=4)
            if j == 0:
                ax.set_ylabel(str(branch), rotation=0, labelpad=12)
                ax.yaxis.set_label_coords(-0.15, 0.5)

    # A single colorbar is only meaningful when all cells share one scale.
    if scale == 'shared' and last_im is not None:
        fig.colorbar(last_im, ax=axes, shrink=0.8, label=_NORM_LABELS[norm])
    fig.supxlabel('Number of temporal blocks')
    fig.supylabel(r'Number of branches ($K$)')
    return fig


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--group-template',
        type=str,
        required=True,
        help=(
            'Run group name, optionally with a "{blocks}" placeholder substituted by '
            'each --blocks value (e.g. "{blocks}conv-seed42-paper" or a literal '
            '"ablate-2dfull-2conv-seed42-paper").'
        ),
    )
    parser.add_argument(
        '--blocks',
        type=int,
        nargs='+',
        required=True,
        help='n_temporal_module_blocks values (grid columns); also fill the template placeholder.',
    )
    parser.add_argument(
        '--branches',
        type=int,
        nargs='+',
        required=True,
        help='n_channels_unmix values (grid rows) to select within each group.',
    )
    parser.add_argument(
        '--experiments-root',
        type=str,
        default=EXPERIMENTS_DIR,
        help='Root directory with experiment logs.',
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Output subfolder and filename identifier. Default: derived from --group-template.',
    )
    parser.add_argument(
        '--meg-files-dir',
        type=str,
        default=CLEAN_DATA_DIR,
        help='Directory with cleaned raw MEG files (used only for topomap sensor positions).',
    )
    parser.add_argument(
        '--meg-format',
        type=str,
        default='fif',
        choices=('bids', 'fif'),
        help='Format of the MEG files used for topomap positions.',
    )
    parser.add_argument(
        '--info-subject',
        type=int,
        default=1,
        help='Subject id of the Raw loaded only to get MEG sensor positions.',
    )
    parser.add_argument(
        '--info-session',
        type=int,
        default=0,
        help='Session id of the Raw loaded only to get MEG sensor positions.',
    )
    parser.add_argument(
        '--info-story-id',
        type=int,
        default=1,
        help='Story id of the Raw loaded only to get MEG sensor positions.',
    )
    parser.add_argument(
        '--cmap',
        type=str,
        default='viridis',
        help='Colormap for the non-negative mean attention weights.',
    )
    parser.add_argument(
        '--scale',
        type=str,
        default='shared',
        choices=('shared', 'per_cell'),
        help='Shared color scale across the grid (comparable) or per-cell autoscaling.',
    )
    parser.add_argument(
        '--norm',
        type=str,
        default='linear',
        choices=('linear', 'sqrt', 'log'),
        help='Value transform before display (log uses log1p) to tame hotspots.',
    )
    parser.add_argument(
        '--vmax-percentile',
        type=float,
        default=99.0,
        help='Upper color-limit percentile; <100 clips bright hotspots (100 = max).',
    )
    parser.add_argument(
        '--metric-basis',
        type=str,
        default='raw',
        choices=('display', 'raw'),
        help=(
            'Values used for effective sensor counts. "display" uses the same '
            'norm and color-limit clipping as the topomap; "raw" uses mean '
            'attention weights before visualization transforms.'
        ),
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default=str(Path(PLOTS_DIR) / 'spatial_attention_maps'),
        help='Output directory for the attention-map grid.',
    )
    return parser.parse_args()


def main_cli() -> None:
    args = parse_arguments()
    blocks = args.blocks
    branches = args.branches
    name = args.name or args.group_template.replace('{blocks}', 'Nblocks')

    raw, _ = load_raw_meg(
        meg_format=args.meg_format,
        data_root=args.meg_files_dir,
        sub=args.info_subject,
        ses=args.info_session,
        story_id=args.info_story_id,
    )
    info = raw.info

    maps: dict[tuple[int, int], np.ndarray] = {}
    for block in blocks:
        group = args.group_template.format(blocks=block)
        validate_run_group(group)
        group_dir = os.path.join(args.experiments_root, group)
        branch_runs = discover_branch_runs(group_dir, expected_blocks=block)
        for branch in branches:
            run_dir = branch_runs.get(branch)
            if run_dir is None:
                print(
                    f'No run for blocks={block}, branches={branch} in {group}; cell left empty.'
                )
                continue
            matrix = load_attention_matrix(run_dir)
            maps[(block, branch)] = matrix.mean(axis=0) * 100.0
            print(
                f'Loaded attention map for blocks={block}, branches={branch} ({run_dir}).'
            )

    fig = plot_attention_grid(
        maps=maps,
        blocks=blocks,
        branches=branches,
        info=info,
        cmap=args.cmap,
        scale=args.scale,
        norm=args.norm,
        vmax_percentile=args.vmax_percentile,
        metric_basis=args.metric_basis,
    )

    outdir = Path(args.outdir) / name
    outdir.mkdir(parents=True, exist_ok=True)
    stem = _output_stem(name, args.group_template, blocks, branches)
    metric_slug = f'metrics-{args.metric_basis}'
    topomap_basename = f'{stem}_{metric_slug}_topomap_grid'
    with plt.rc_context(PUBLICATION_RC):
        for ext in ('png', 'pdf'):
            fig.savefig(outdir / f'{topomap_basename}.{ext}', dpi=300)
    plt.close(fig)

    print(f'Saved {topomap_basename}.{{png,pdf}} to {outdir}')


if __name__ == '__main__':
    main_cli()
