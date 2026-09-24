"""Source-space estimation and rendering helpers for branch interpretations."""

import io
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import apply_inverse, make_inverse_operator
from mne.utils._logging import use_log_level
from PIL import Image


SOURCE_RENDER_FIGSIZE_IN = (4.0, 3.0)
SOURCE_RENDER_LIMIT_PAD_FRAC = 0.15
SOURCE_RENDER_OUTPUT_PAD_FRAC = 0.08
FSAVERAGE_BEM_ICO = 4
FSAVERAGE_BEM_CONDUCTIVITY = (0.3,)
INVERSE_LOOSE = 0.5
INVERSE_DEPTH = 0.5
INVERSE_LAMBDA2 = 1.0 / 3
INVERSE_METHOD = 'MNE'


@dataclass(frozen=True)
class PreparedSourceGeometry:
    """Subject-independent fsaverage source space and single-layer MEG BEM."""

    subject: str
    subjects_dir: str
    src: Any
    bem: Any


def prepare_source_geometry() -> PreparedSourceGeometry:
    """Prepare the fsaverage source space and BEM shared by all subjects."""
    fs_dir = Path(fetch_fsaverage(verbose=False))
    subjects_dir = str(fs_dir.parent)
    subject = 'fsaverage'
    src = mne.setup_source_space(
        subject,
        spacing='ico4',
        add_dist=False,
        subjects_dir=subjects_dir,
        verbose=False,
    )
    model = mne.make_bem_model(
        subject=subject,
        ico=FSAVERAGE_BEM_ICO,
        conductivity=FSAVERAGE_BEM_CONDUCTIVITY,
        subjects_dir=subjects_dir,
        verbose=False,
    )
    bem = mne.make_bem_solution(model, verbose=False)
    return PreparedSourceGeometry(
        subject=subject,
        subjects_dir=subjects_dir,
        src=src,
        bem=bem,
    )


def fsaverage_stc_template(
    *,
    n_times: int = 1,
    prepared_geometry: PreparedSourceGeometry | None = None,
) -> mne.SourceEstimate:
    """Empty fsaverage STC used only as a plotting/source-space container."""

    geometry = prepared_geometry or prepare_source_geometry()
    vertices = [geometry.src[0]['vertno'], geometry.src[1]['vertno']]
    n_vertices = int(sum(len(hemi) for hemi in vertices))
    data = np.zeros((n_vertices, n_times), dtype=np.float64)
    return mne.SourceEstimate(
        data,
        vertices=vertices,
        tmin=0.0,
        tstep=1.0,
        subject=geometry.subject,
    )


def get_stc(
    spatial_patterns: np.ndarray,
    *,
    info: Any,
    trans: Any,
    prepared_geometry: PreparedSourceGeometry | None = None,
    fwd: Any | None = None,
    n_jobs: int = 3,
) -> mne.SourceEstimate:
    """Project sensor-space patterns through participant-specific fsaverage geometry.

    Args:
        spatial_patterns: Array (branches, channels) in physical sensor units.
        info: Participant-specific reconstructed KIT Info.
        trans: Participant-specific head to fsaverage transform.
        prepared_geometry: Shared fsaverage source space and BEM.
        fwd: Optional precomputed participant forward. When omitted, a forward
            is built from ``info`` and ``trans``.
        n_jobs: Workers for ``make_forward_solution``.
    """
    geometry = prepared_geometry or prepare_source_geometry()
    if fwd is None:
        meg_types = info.get_channel_types(picks='data', unique=True)
        fwd = mne.make_forward_solution(
            info,
            trans=trans,
            src=geometry.src,
            bem=geometry.bem,
            eeg=False,
            meg=meg_types[0],
            n_jobs=n_jobs,
            verbose=False,
        )

    ad_hoc_cov = mne.make_ad_hoc_cov(info)
    inverse_operator = make_inverse_operator(
        info,
        fwd,
        ad_hoc_cov,
        loose=INVERSE_LOOSE,
        depth=INVERSE_DEPTH,
        verbose=False,
    )
    evoked = mne.EvokedArray(spatial_patterns.T, info)
    stc = apply_inverse(
        evoked,
        inverse_operator,
        lambda2=INVERSE_LAMBDA2,
        method=INVERSE_METHOD,
        verbose=False,
    )

    return stc


def crop_bottom_fraction(img: Image.Image, frac: float = 0.2) -> Image.Image:
    """Crop bottom fraction of a PIL image."""
    w, h = img.size
    new_h = int(h * (1 - frac))
    return img.crop((0, 0, w, new_h))


def pad_image(
    img: Image.Image,
    *,
    left: int,
    right: int,
    top: int = 0,
    bottom: int = 0,
) -> Image.Image:
    w, h = img.size
    canvas = Image.new(img.mode, (w + left + right, h + top + bottom), 'white')
    canvas.paste(img, (left, top))
    return canvas


def crop_to_nonwhite_content(
    img: Image.Image,
    *,
    threshold: int = 248,
    padding_frac: float = SOURCE_RENDER_OUTPUT_PAD_FRAC,
) -> Image.Image:
    """Crop empty canvas while retaining controlled whitespace around content."""
    rgb = img.convert('RGB')
    pixels = np.asarray(rgb)
    nonwhite = np.any(pixels < threshold, axis=2)
    rows = np.flatnonzero(np.any(nonwhite, axis=1))
    columns = np.flatnonzero(np.any(nonwhite, axis=0))
    if rows.size == 0 or columns.size == 0:
        raise ValueError('Rendered source image contains no non-white content.')

    content = rgb.crop(
        (
            int(columns[0]),
            int(rows[0]),
            int(columns[-1]) + 1,
            int(rows[-1]) + 1,
        )
    )
    pad_x = max(12, int(round(content.width * padding_frac)))
    pad_y = max(12, int(round(content.height * padding_frac)))
    return pad_image(
        content,
        left=pad_x,
        right=pad_x,
        top=pad_y,
        bottom=pad_y,
    )


def pad_3d_axes_limits(ax, padding_frac: float) -> None:
    """Zoom out a 3D axes by expanding each data limit symmetrically."""
    for axis_name in ('x', 'y', 'z'):
        get_limit = getattr(ax, f'get_{axis_name}lim')
        set_limit = getattr(ax, f'set_{axis_name}lim')
        lower, upper = get_limit()
        padding = (upper - lower) * padding_frac
        set_limit(lower - padding, upper + padding)


@contextmanager
def temp_backend(backend_name: str):
    """Context manager for temporary backend switching"""
    old_backend = plt.get_backend()
    plt.switch_backend(backend_name)
    try:
        yield
    finally:
        plt.switch_backend(old_backend)


def get_stc_hemi(
    stc: mne.SourceEstimate,
    hemi: str,
    subject: str = 'fsaverage',
    subjects_dir: str | None = None,
    dpi: int = 150,
    clim: str | dict = 'auto',
):
    """Render a single-hemisphere image for a SourceEstimate."""
    # for some reason stc.plot() in mne has plt.show() baked in and is always interactive
    with temp_backend('Agg'), use_log_level(False):
        fig = stc.plot(
            subject=subject,
            subjects_dir=subjects_dir,
            hemi=hemi,
            views='lat',
            backend='matplotlib',
            time_viewer=False,
            colorbar=False,
            clim=clim,
            background='white',
        )
        fig.set_size_inches(*SOURCE_RENDER_FIGSIZE_IN, forward=True)
        brain_ax, *extra_axes = fig.axes
        for extra_ax in extra_axes:
            extra_ax.remove()
        pad_3d_axes_limits(brain_ax, SOURCE_RENDER_LIMIT_PAD_FRAC)

        with io.BytesIO() as buf:
            fig.savefig(
                buf,
                format='png',
                dpi=dpi,
                bbox_inches=None,
                facecolor='white',
                edgecolor='white',
            )
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            img.load()  # fully detach from buffer
            return crop_to_nonwhite_content(img)


def _render_lh_rh_pil(stc: mne.SourceEstimate) -> tuple[Image.Image, Image.Image]:
    img_lh = get_stc_hemi(stc, hemi='lh')
    img_rh = get_stc_hemi(stc, hemi='rh')
    return img_lh.convert('RGB'), img_rh.convert('RGB')


def _compose_lh_rh_side(img_lh: Image.Image, img_rh: Image.Image) -> Image.Image:
    img_lh = img_lh.convert('RGB')
    img_rh = img_rh.convert('RGB')

    w = img_lh.width + img_rh.width
    h = max(img_lh.height, img_rh.height)

    canvas = Image.new('RGB', (w, h), 'white')
    canvas.paste(img_lh, (0, 0))
    canvas.paste(img_rh, (img_lh.width, 0))
    return canvas


def render_lh_rh_views(stc: mne.SourceEstimate) -> dict[str, np.ndarray]:
    """Render left, right, and side-by-side hemisphere images without cropping."""
    img_lh, img_rh = _render_lh_rh_pil(stc)
    side = _compose_lh_rh_side(img_lh, img_rh)
    return {
        'lh': np.asarray(img_lh),
        'rh': np.asarray(img_rh),
        'side': np.asarray(side),
    }


def render_lh_rh_side(stc: mne.SourceEstimate) -> np.ndarray:
    """Render left/right hemisphere images side-by-side without cropping."""
    return render_lh_rh_views(stc)['side']
