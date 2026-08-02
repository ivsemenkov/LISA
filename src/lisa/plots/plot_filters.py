"""Utility plotting functions for spatial and temporal filters."""

import math
from pathlib import Path
from typing import Optional, Literal

import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy.signal as sg

from lisa.data.meg_io import load_raw_meg
from lisa.utils.numeric import fft_magnitude


def filter_data(
    data: np.ndarray, temporal_filter: np.ndarray, filtfilt: bool
) -> np.ndarray:
    """Apply 1D convolution (optionally forward-backward) to a signal."""
    assert data.ndim == 1, data.shape
    y = np.convolve(data, temporal_filter, mode='same')
    if filtfilt:
        y = np.convolve(np.flip(y, axis=-1), temporal_filter, mode='same')
        y = np.flip(y, axis=-1)
    return y
def plot_filters_fft(
    filters: np.ndarray,
    fs: float,
    out_path: Path | None = None,
    return_fig: bool = False,
    x_temp_in: np.ndarray | None = None,
    x_temp_out: np.ndarray | None = None,
    plot_filters: bool = True,
    low_freq_cut: float = 0.0,
    plot_kwargs: dict | None = None,
    filters_label: str = 'Filter',
    square_fft: bool = True,
):
    """Plot rFFT magnitudes for all filters with shared y; full frame; x ticks only.

    Args:
        filters: Temporal filters (K, L).
        fs: Sampling rate (Hz).
        out_path: Output path for PNG/PDF; if None, show interactively.
        return_fig: If True, return the matplotlib Figure.
        x_temp_in: Optional temporal filter input traces (K, T).
        x_temp_out: Optional temporal filter output traces (K, T).
        plot_filters: Whether to plot filter FFTs.
        low_freq_cut: Lower frequency bound for plotting.
        plot_kwargs: Matplotlib kwargs for plot().
        filters_label: Legend label for filters.
        square_fft: If True, plot |H(w)|^2.
    """
    assert low_freq_cut >= 0.0, low_freq_cut
    if plot_kwargs is None:
        plot_kwargs = {'linewidth': 2}
    K, L = filters.shape
    if x_temp_in is not None:
        assert x_temp_in.shape[0] == K, (K, x_temp_in.shape)
        assert x_temp_in.ndim == 2, x_temp_in.shape
    if x_temp_out is not None:
        assert x_temp_out.shape[0] == K, (K, x_temp_out.shape)
        assert x_temp_out.ndim == 2, x_temp_out.shape
    ncols = 2 if K > 1 else 1
    nrows = math.ceil(K / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(14, 2.2 * nrows), sharex=True
    )  # , sharey=True)
    if K == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])

    freq_max = fs / 2.0
    xticks = np.arange(5.0, freq_max + 1e-9, 5.0)
    xticks = xticks[xticks <= freq_max]

    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            if idx < K:
                if plot_filters:
                    f, magnitude = fft_magnitude(
                        filters[idx], fs, square_fft=square_fft
                    )
                    sel = f <= freq_max
                    ax.plot(f[sel], magnitude[sel], label=filters_label, **plot_kwargs)

                Pxx_in = None
                if x_temp_in is not None:
                    x_in_seg = x_temp_in[idx]
                    f_in, Pxx_in = sg.welch(
                        x_in_seg,
                        fs,
                        window='hann',
                        nperseg=min(int(fs * 4), x_in_seg.shape[-1]),
                        noverlap=min(int(fs * 4), x_in_seg.shape[-1]) // 2,
                        nfft=4096,
                        detrend=False,
                        return_onesided=True,
                        scaling='density',
                    )
                    freq_mask_in = f_in >= low_freq_cut
                    f_in = f_in[freq_mask_in]
                    Pxx_in = Pxx_in[freq_mask_in]
                    ax.plot(
                        f_in,
                        Pxx_in / np.linalg.norm(Pxx_in),
                        label='Input',
                        **plot_kwargs,
                    )

                if x_temp_out is not None:
                    x_out_seg = x_temp_out[idx]
                    f_out, Pxx_out = sg.welch(
                        x_out_seg,
                        fs,
                        window='hann',
                        nperseg=min(int(fs * 4), x_out_seg.shape[-1]),
                        noverlap=min(int(fs * 4), x_out_seg.shape[-1]) // 2,
                        nfft=4096,
                        detrend=False,
                        return_onesided=True,
                        scaling='density',
                    )
                    freq_mask_out = f_out >= low_freq_cut
                    f_out = f_out[freq_mask_out]
                    Pxx_out = Pxx_out[freq_mask_out]
                    ax.plot(
                        f_out,
                        Pxx_out / np.linalg.norm(Pxx_out),
                        label='Output',
                        **plot_kwargs,
                    )

                for spine in ('right', 'top', 'left', 'bottom'):
                    ax.spines[spine].set_visible(True)

                ax.set_yticks([])
                ax.set_yticklabels([])

                if r == nrows - 1:
                    ax.set_xticks(xticks)
                    ax.set_xticklabels([str(int(t)) for t in xticks])
                    ax.set_xlabel('Frequency (Hz)')
                else:
                    ax.set_xticks([])
                    ax.set_xticklabels([])

                ax.set_xlim(low_freq_cut, freq_max)
                ax.set_title(f'Branch: {idx + 1}')
            else:
                ax.axis('off')
            idx += 1
    ax.legend()

    plt.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
    else:
        plt.show()
    if return_fig:
        return fig
    else:
        plt.close(fig)


def _pick_single_meg_type(
    raw: mne.io.BaseRaw, meg_type: Literal['mag', 'grad']
) -> mne.io.BaseRaw:
    """Keep only one MEG sensor type; error if none found."""
    picks = mne.pick_types(
        raw.info, meg=meg_type, eeg=False, eog=False, ecg=False, stim=False, misc=False
    )
    if len(picks) == 0:
        raise RuntimeError(f'No MEG {meg_type} channels found in recording.')
    return raw.pick(picks, verbose=False, exclude=())


def _strict_layout_alignment(info: mne.Info) -> tuple[list[str], np.ndarray]:
    """
    Get a strict layout for the current info.
    Returns (layout_names_in_order, pos_xy_scaled).
    Errors if any channel name mismatch (extra/missing) vs layout.
    """
    layout = mne.channels.find_layout(info)
    if layout is None:
        raise RuntimeError('Could not compute a MEG layout for these channels.')
    layout_names = list(layout.names)
    pos = layout.pos  # shape (n_layout, 2) in [0, 1] layout coords

    # Strict set equality between layout names and info channel names
    info_names = [ch['ch_name'] for ch in info['chs']]

    set_layout = set(layout_names)
    set_info = set(info_names)

    missing_in_layout = sorted(set_info - set_layout)
    extra_in_layout = sorted(set_layout - set_info)
    if missing_in_layout or extra_in_layout:
        msg = []
        if missing_in_layout:
            msg.append(
                f'Missing in layout ({len(missing_in_layout)}): {missing_in_layout[:10]}{" ..." if len(missing_in_layout) > 10 else ""}'
            )
        if extra_in_layout:
            msg.append(
                f'Extra in layout ({len(extra_in_layout)}): {extra_in_layout[:10]}{" ..." if len(extra_in_layout) > 10 else ""}'
            )
        raise ValueError(
            'Channel/layout mismatch.\n'
            + '\n'.join(msg)
            + '\nTip: ensure you plotted only one sensor type (mag OR grad) and that channels were not dropped.'
        )

    # Now require identical ORDER as well
    if layout_names != info_names:
        # Provide a compact diff hint
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(layout_names, info_names)) if a != b),
            None,
        )
        raise ValueError(
            'Channel order mismatch between layout and raw.info.\n'
            f"First difference at index {first_diff}: layout='{layout_names[first_diff]}', info='{info_names[first_diff]}'.\n"
            'Refuse to reorder silently. Align your filter vectors to the layout order (or adjust your plotting code explicitly).'
        )

    # Scale layout coords to ~[-1, 1] for plot_topomap
    pos_scaled = (pos - 0.5) * 2.0
    return layout_names, pos_scaled


def plot_spatial_filters(
    filters: np.ndarray,
    bids_root: str,
    subject: str | int,
    session: str | int,
    task: str,
    meg_format: Literal['bids', 'fif'],
    meg_type: Literal['mag', 'grad'] = 'mag',
    out_path: Optional[Path] = None,
    return_fig: bool = False,
    add_colorbar: bool = False,
):
    """
    Plot spatial filter weights as strict MEG topomaps.

    STRICTNESS:
    - Errors if 'filters.shape[1]' != number of picked MEG channels.
    - Errors if layout channels != picked channels (both set and order).
    - No silent reordering or dropping of channels.

    Args:
        filters: Array (K, n_channels) in EXACTLY the same order as raw.pick(meg=meg_type).ch_names.
        bids_root: Path to MEG data root.
        subject: Subject id.
        session: Session id.
        task: Story/task id.
        meg_format: "bids" or "fif".
        meg_type: "mag" or "grad".
        out_path: Output path; if None, show interactively.
        return_fig: If True, return the Figure.
        add_colorbar: If True, add a colorbar to the last plot.
    """

    raw, _ = load_raw_meg(
        meg_format=meg_format,
        data_root=bids_root,
        sub=subject,
        ses=session,
        story_id=task,
    )
    try:
        raw = _pick_single_meg_type(raw, meg_type)

        K, L = filters.shape
        n_ch = len(raw.ch_names)
        if L != n_ch:
            raise ValueError(
                f'filters.shape[1] ({L}) != number of picked channels ({n_ch}).\n'
                f'Expected order: raw.ch_names = {raw.ch_names[:10]}{" ..." if n_ch > 10 else ""}'
            )

        # Ensure layout matches channels STRICTLY (set and order)
        _, pos = _strict_layout_alignment(raw.info)
        # At this point, the layout channel order has been validated against
        # raw.ch_names.
    finally:
        raw.close()

    # Shared symmetric color scaling
    # vmax = np.max(np.abs(filters))
    # vmin = -vmax if vmax > 0 else 0.0

    ncols = 2 if K > 1 else 1
    nrows = math.ceil(K / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.6 * nrows))
    if K == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])

    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            if idx < K:
                vec = filters[idx]
                im, _ = mne.viz.plot_topomap(
                    vec,
                    pos,
                    axes=ax,
                    show=False,
                    # vmin=vmin, vmax=vmax,
                    cmap='RdBu_r',
                    contours=0,
                    sphere=1.0,
                )
                ax.set_title(f'Branch {idx + 1}', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                if add_colorbar and (idx == K - 1):
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.axis('off')
            idx += 1

    plt.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
    else:
        plt.show()

    if return_fig:
        return fig
    plt.close(fig)
