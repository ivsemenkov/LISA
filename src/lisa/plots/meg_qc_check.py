#!/usr/bin/env python3
"""MEG QC report generation: snapshots, PSDs, and validation checks."""

from __future__ import annotations
import argparse
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mne.time_frequency import psd_array_welch as mne_psd_welch
from tqdm import tqdm

from lisa.utils.constants import N_MEG_CHANNELS, PLOTS_DIR, PREPROCESSED_DATA_DIR


KEY_RE = re.compile(r'^subject(\d+)_session(\d+)_story(.+)$')


@dataclass
class Config:
    sfreq: float
    expected_channels: int
    chunk_seconds: float
    snapshot_mode: str
    psd_mode: str
    outdir: Path
    stacked_per_page: (
        int  # 0 -> disabled; >0 -> channels per page in stacked raw-like view
    )
    psd_fmin: float
    psd_fmax: float
    powerline_hz: float
    overlay_guides: bool = False
    representative_channels: int = 16
    heatmap_per_channel_normalize: bool = False
    heatmap_clip_percentile: Optional[float] = None
    show_median_on_all: bool = False
    continue_on_error: bool = False
    stacked_spacing_factor: float = (
        3.0  # vertical offset = factor * max(|signal|) across shown channels
    )


def parse_key(key: str) -> Tuple[int, int, str]:
    """Parse NPZ key into (subject, session, story_id)."""
    m = KEY_RE.match(key)
    if not m:
        raise RuntimeError(
            f"Key '{key}' does not match 'subject{{sub}}session{{ses}}story{{story}}'."
        )
    return int(m.group(1)), int(m.group(2)), m.group(3)


def validate_array(data: np.ndarray, ident: str, cfg: Config) -> None:
    """Validate array shape, dtype, and basic sanity constraints."""
    if data.ndim != 2:
        raise RuntimeError(f'{ident}: expected 2D array, got {data.shape}')
    n_ch, n_samp = data.shape
    if n_ch != cfg.expected_channels:
        raise RuntimeError(
            f'{ident}: expected {cfg.expected_channels} channels, got {n_ch}'
        )
    if np.iscomplexobj(data):
        raise RuntimeError(f'{ident}: complex dtype not allowed')
    if not np.issubdtype(data.dtype, np.floating):
        raise RuntimeError(f'{ident}: dtype must be float, got {data.dtype}')
    if np.isnan(data).any():
        coords = np.argwhere(np.isnan(data))
        raise RuntimeError(f'{ident}: NaNs detected (e.g. {coords[:5].tolist()}...)')
    if np.isinf(data).any():
        coords = np.argwhere(np.isinf(data))
        raise RuntimeError(f'{ident}: Infs detected (e.g. {coords[:5].tolist()}...)')
    stds = np.std(data, axis=1)
    zero_std_idx = np.where(stds == 0.0)[0]
    if zero_std_idx.size:
        raise RuntimeError(f'{ident}: zero-std channels: {zero_std_idx.tolist()}')
    if n_samp < int(cfg.chunk_seconds * cfg.sfreq):
        raise RuntimeError(
            f'{ident}: too short for {cfg.chunk_seconds}s snapshot (samples={n_samp})'
        )


def pick_center_chunk(data: np.ndarray, cfg: Config) -> np.ndarray:
    """Select a centered time chunk of length cfg.chunk_seconds."""
    n_ch, n_samp = data.shape
    L = int(round(cfg.chunk_seconds * cfg.sfreq))
    start = max(0, (n_samp - L) // 2)
    end = start + L
    return data[:, start:end]


def compute_psd_all(data: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (f, psds) with psds shape (n_channels, n_freqs).
    Strict checks: frequency vector is 1D, strictly increasing, finite; identical across channels;
    psd values finite; shapes consistent.
    """
    n_ch, n_samp = data.shape
    n_per_seg = int(
        min(max(round(8 * cfg.sfreq), int(cfg.sfreq)), n_samp)
    )  # ~8s segments, but <= n_samp

    # Important: psd_array_welch in MNE has no 'detrend' argument in many versions.
    # Keep the call minimal and portable.
    n_fft = int(2 ** np.ceil(np.log2(max(1, n_per_seg))))
    psd, f = mne_psd_welch(
        data,
        sfreq=cfg.sfreq,
        fmin=0.0,
        fmax=cfg.sfreq / 2,
        n_per_seg=n_per_seg,
        n_overlap=n_per_seg // 2,
        n_fft=n_fft,
        window='hann',
        average='mean',  # average Welch segments within each channel
        verbose=False,
    )
    # Validate
    if f.ndim != 1 or f.size == 0:
        raise RuntimeError('PSD: invalid frequency vector.')
    if not np.all(np.diff(f) > 0):
        raise RuntimeError('PSD: frequency vector not strictly increasing.')
    if psd.shape != (n_ch, f.size):
        raise RuntimeError(
            f'PSD: unexpected shape {psd.shape}; expected {(n_ch, f.size)}.'
        )
    if not np.all(np.isfinite(f)):
        raise RuntimeError('PSD: non-finite values in frequency vector.')
    if not np.all(np.isfinite(psd)):
        bad = np.where(~np.all(np.isfinite(psd), axis=1))[0].tolist()
        raise RuntimeError(f'PSD: non-finite spectral values on channels {bad}.')
    return f, psd


# ---------- Plotting ----------
def plot_snapshot(ax, chunk: np.ndarray, cfg: Config):
    """Plot a time-domain snapshot (butterfly or heatmap)."""
    n_ch, n_samp = chunk.shape
    t = np.arange(n_samp) / cfg.sfreq
    if cfg.snapshot_mode == 'butterfly':
        ax.plot(t, chunk.T, linewidth=0.35, alpha=0.35)
        if cfg.overlay_guides and cfg.representative_channels > 0:
            idx = np.linspace(
                0, n_ch - 1, num=min(cfg.representative_channels, n_ch), dtype=int
            )
            for i in idx:
                ax.plot(t, chunk[i], linewidth=0.9)
        ax.set_ylabel('Amplitude (raw)')
        ax.set_title(f'{int(round(n_samp / cfg.sfreq))} s snapshot (butterfly)')
    elif cfg.snapshot_mode == 'heatmap':
        data = chunk.copy()
        if cfg.heatmap_per_channel_normalize:
            mu = np.mean(data, axis=1, keepdims=True)
            sd = np.std(data, axis=1, keepdims=True)
            sd[sd == 0.0] = 1.0
            data = (data - mu) / sd
        if cfg.heatmap_clip_percentile is None:
            vmin, vmax = np.min(data), np.max(data)
        else:
            lim = np.percentile(np.abs(data), cfg.heatmap_clip_percentile)
            vmin, vmax = -lim, lim
        im = ax.imshow(
            data,
            aspect='auto',
            origin='lower',
            extent=[0, n_samp / cfg.sfreq, 0, n_ch],
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest',
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_ylabel('Channel')
        ax.set_title(f'{int(round(n_samp / cfg.sfreq))} s snapshot (heatmap)')
    else:
        raise RuntimeError(f'Unknown snapshot_mode={cfg.snapshot_mode}')
    ax.set_xlabel('Time (s)')
    ax.grid(True, linewidth=0.3, alpha=0.4)


def plot_psd(ax, f: np.ndarray, psds: np.ndarray, cfg: Config):
    """Plot PSD curves for channels or summary statistics."""
    sel = (f >= cfg.psd_fmin) & (f <= cfg.psd_fmax)
    if cfg.psd_mode == 'all':
        ax.plot(f[sel], psds[:, sel].T, linewidth=0.4, alpha=0.35)
        if cfg.show_median_on_all:
            ax.plot(f[sel], np.median(psds, axis=0)[sel], linewidth=1.0, label='Median')
    elif cfg.psd_mode == 'median_iqr':
        med = np.median(psds, axis=0)
        q25 = np.quantile(psds, 0.25, axis=0)
        q75 = np.quantile(psds, 0.75, axis=0)
        ax.plot(f[sel], med[sel], linewidth=1.0, label='Median')
        ax.fill_between(f[sel], q25[sel], q75[sel], alpha=0.25, label='IQR')
    else:
        raise RuntimeError(f'Unknown psd_mode={cfg.psd_mode}')
    if cfg.powerline_hz and cfg.psd_fmin <= cfg.powerline_hz <= cfg.psd_fmax:
        ax.axvline(cfg.powerline_hz, linestyle='--', linewidth=0.8)
    ax.set_xlim(cfg.psd_fmin, cfg.psd_fmax)
    ax.set_yscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.grid(True, which='both', linewidth=0.3, alpha=0.4)
    ax.set_title('PSD')


def plot_stacked_subset(ax, data_subset: np.ndarray, cfg: Config, sfreq: float):
    """
    Raw-like stacked plot for a subset of channels (data_subset shape: n_subch x n_samp).
    Preserves raw amplitudes; uses vertical offsets so traces don't overlap.
    No normalization or clipping is applied.
    """
    n_subch, n_samp = data_subset.shape
    t = np.arange(n_samp) / sfreq

    # Use global max abs amplitude across the subset to set spacing (no normalization)
    global_amp = np.max(np.abs(data_subset))
    if not np.isfinite(global_amp) or global_amp == 0:
        global_amp = 1.0
    step = cfg.stacked_spacing_factor * global_amp

    offsets = np.arange(n_subch) * step
    for i in range(n_subch):
        ax.plot(t, data_subset[i] + offsets[i], linewidth=0.6)

    ax.set_yticks(offsets)
    ax.set_yticklabels([f'Ch {i}' for i in range(n_subch)])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channels (offset)')
    ax.grid(True, linewidth=0.3, alpha=0.4)


def run_qc(npz_path: Path, cfg: Config):
    """Run QC and emit per-subject PDF reports and a CSV summary."""
    npz = np.load(str(npz_path), allow_pickle=True)
    entries = []
    for key in npz.files:
        sub, ses, story = parse_key(key)
        data = np.asarray(npz[key])
        ident = f'[{key}] subj {sub} ses {ses} story {story}'
        validate_array(data, ident, cfg)
        entries.append((sub, ses, story, data))

    by_subject: Dict[int, List[Tuple[int, int, str, np.ndarray]]] = {}
    for e in entries:
        by_subject.setdefault(e[0], []).append(e)
    for s in by_subject:
        by_subject[s].sort(key=lambda e: (e[1], e[2]))

    cfg.outdir.mkdir(parents=True, exist_ok=True)
    summary = []
    for subject, sub_entries in tqdm(sorted(by_subject.items())):
        pdf_path = cfg.outdir / f'qc_subject{subject:02d}.pdf'
        with PdfPages(str(pdf_path)) as pdf:
            for sub, ses, story, data in sub_entries:
                f, psds = compute_psd_all(data, cfg)
                if cfg.stacked_per_page > 0:
                    # Paged raw-like stacked plots on the left; PSD on the right (repeated per page)
                    n_ch, n_samp = data.shape
                    chunk = pick_center_chunk(data, cfg)
                    per_page = max(1, int(cfg.stacked_per_page))
                    for i0 in range(0, n_ch, per_page):
                        i1 = min(i0 + per_page, n_ch)
                        sub_chunk = chunk[i0:i1, :]
                        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150)
                        # Left: stacked subset (raw-like)
                        plot_stacked_subset(axes[0], sub_chunk, cfg, cfg.sfreq)
                        axes[0].set_title(
                            f'{int(round(sub_chunk.shape[1] / cfg.sfreq))} s stacked (Ch {i0}-{i1 - 1})'
                        )
                        # Right: PSD (all channels, unchanged)
                        plot_psd(axes[1], f, psds, cfg)
                        fig.suptitle(
                            f'Subject {sub:02d}, Session {ses}, Story {story}', y=0.98
                        )
                        plt.tight_layout(rect=[0, 0, 1, 0.97])
                        pdf.savefig(fig)
                        plt.close(fig)
                else:
                    # Original single-page behavior: butterfly/heatmap + PSD
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150)
                    chunk = pick_center_chunk(data, cfg)
                    plot_snapshot(axes[0], chunk, cfg)
                    plot_psd(axes[1], f, psds, cfg)
                    fig.suptitle(
                        f'Subject {sub:02d}, Session {ses}, Story {story}', y=0.98
                    )
                    plt.tight_layout(rect=[0, 0, 1, 0.97])
                    pdf.savefig(fig)
                    plt.close(fig)
                summary.append(
                    {
                        'subject': sub,
                        'session': ses,
                        'story': story,
                        'n_channels': data.shape[0],
                        'n_samples': data.shape[1],
                        'implied_duration_s': data.shape[1] / cfg.sfreq,
                    }
                )
        print(f'[OK] {pdf_path}')

    pd.DataFrame(summary).to_csv(cfg.outdir / 'qc_summary.csv', index=False)


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for MEG QC."""
    p = argparse.ArgumentParser()
    p.add_argument(
        '--meg',
        type=str,
        default=os.path.join(PREPROCESSED_DATA_DIR, 'meg', 'meg27_sr100.npz'),
        help='Path to input .npz file with MEG data',
    )
    p.add_argument(
        '--outdir',
        type=str,
        default=os.path.join(PLOTS_DIR, 'meg_qc'),
        help='Output directory for QC reports',
    )
    p.add_argument(
        '--sfreq', type=float, default=100.0, help='MEG sampling frequency (Hz)'
    )
    p.add_argument(
        '--n-channels',
        type=int,
        default=N_MEG_CHANNELS,
        help='Expected number of MEG channels',
    )
    p.add_argument(
        '--snapshot',
        choices=['butterfly', 'heatmap'],
        default='butterfly',
        help='Type of snapshot plot',
    )
    p.add_argument(
        '--psd-mode',
        choices=['all', 'median_iqr'],
        default='all',
        help='Type of PSD plot',
    )
    p.add_argument(
        '--overlay-guides',
        action='store_true',
        help='Overlay representative channel traces on butterfly plot',
    )
    p.add_argument(
        '--stacked-per-page',
        type=int,
        default=16,
        help='If >0, draw raw-like stacked view with this many channels per page',
    )
    p.add_argument(
        '--chunk-size',
        type=float,
        default=10.0,
        help='Length of data chunk (seconds) for snapshot plots',
    )
    p.add_argument(
        '--psd-fmin',
        type=float,
        default=0.5,
        help='Low frequency limit for PSD plots (Hz)',
    )
    p.add_argument(
        '--psd-fmax',
        type=float,
        default=60.0,
        help='High frequency limit for PSD plots (Hz)',
    )
    p.add_argument(
        '--powerline',
        type=float,
        default=50.0,
        help='Powerline frequency (Hz); set to 0 to disable vertical line in PSD plots',
    )
    return p.parse_args()


def main() -> None:
    """CLI entry point for MEG QC."""
    args = parse_arguments()

    cfg = Config(
        sfreq=args.sfreq,
        expected_channels=args.n_channels,
        snapshot_mode=args.snapshot,
        psd_mode=args.psd_mode,
        overlay_guides=args.overlay_guides,
        outdir=Path(args.outdir),
        stacked_per_page=args.stacked_per_page,
        chunk_seconds=args.chunk_size,
        psd_fmin=args.psd_fmin,
        psd_fmax=args.psd_fmax,
        powerline_hz=args.powerline,
    )

    run_qc(Path(args.meg), cfg)


if __name__ == '__main__':
    main()
