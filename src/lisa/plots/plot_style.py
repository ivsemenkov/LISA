"""Shared publication typography and responsive figure geometry."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

import matplotlib as mpl


FULL_WIDTH_IN = 7.0

BASE_FONT_SIZE = 9.0
LABEL_FONT_SIZE = 10.0
TITLE_FONT_SIZE = 10.0
SUPTITLE_FONT_SIZE = 11.0
ANNOTATION_FONT_SIZE = 8.0
MIN_ANNOTATION_FONT_SIZE = 7.0

PUBLICATION_RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': BASE_FONT_SIZE,
    'axes.labelsize': LABEL_FONT_SIZE,
    'axes.titlesize': TITLE_FONT_SIZE,
    'xtick.labelsize': BASE_FONT_SIZE,
    'ytick.labelsize': BASE_FONT_SIZE,
    'legend.fontsize': BASE_FONT_SIZE - 0.5,
    'legend.title_fontsize': BASE_FONT_SIZE,
    'figure.titlesize': SUPTITLE_FONT_SIZE,
    'lines.linewidth': 1.6,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'mathtext.fontset': 'dejavusans',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300,
}

_P = ParamSpec('_P')
_R = TypeVar('_R')


def publication_style(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Run a plotting function with the shared style without changing global state."""

    @wraps(function)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with mpl.rc_context(PUBLICATION_RC):
            return function(*args, **kwargs)

    return wrapped


def panel_figsize(nrows: int = 1) -> tuple[float, float]:
    """Return a full-width canvas for vertically stacked chart panels."""
    if nrows < 1:
        raise ValueError(f'nrows must be positive, got {nrows}.')
    height = max(3.2, 0.8 + 2.1 * nrows)
    return FULL_WIDTH_IN, height


def paper_topomap(
    data: Any,
    pos: Any,
    *,
    kind: str = 'topography',
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Render MNE topomaps with the paper-wide sensor/contour convention."""
    from mne.viz import plot_topomap

    if kind == 'attention':
        kwargs.setdefault('contours', 0)
        kwargs.setdefault('sensors', False)
    elif kind == 'topography':
        kwargs.setdefault('sensors', False)
    else:
        raise ValueError(f'Unknown paper topomap kind: {kind!r}.')

    image, contours = plot_topomap(data, pos, **kwargs)
    if kind == 'attention':
        axes = kwargs.get('axes')
        if axes is not None:
            for line in axes.lines:
                line.set_linewidth(0.55)
    return image, contours


@dataclass(frozen=True)
class GridGeometry:
    """Physical geometry derived from the number of grid cells."""

    figsize: tuple[float, float]
    cell_width: float
    annotation_fontsize: float


def grid_geometry(
    nrows: int,
    ncols: int,
    *,
    cell_aspect: float = 1.0,
) -> GridGeometry:
    """Size a full-width image grid while preserving readable cells and text.

    ``cell_aspect`` is cell width divided by cell height. Very dense grids fail
    explicitly so callers can split the figure instead of silently shrinking it.
    """
    if nrows < 1 or ncols < 1:
        raise ValueError(f'nrows and ncols must be positive, got {nrows} and {ncols}.')
    if cell_aspect <= 0:
        raise ValueError(f'cell_aspect must be positive, got {cell_aspect}.')

    left_margin = 0.65
    right_margin = 0.35
    top_margin = 0.55
    bottom_margin = 0.50
    column_gap = 0.10
    row_gap = 0.18

    cell_width = (
        FULL_WIDTH_IN - left_margin - right_margin - column_gap * (ncols - 1)
    ) / ncols
    minimum_cell_width = 0.65
    if cell_width < minimum_cell_width:
        raise ValueError(
            f'{ncols} grid columns leave only {cell_width:.2f} inches per cell; '
            'split the figure or use a landscape layout.'
        )

    cell_height = cell_width / cell_aspect
    figure_height = (
        top_margin + bottom_margin + nrows * cell_height + row_gap * (nrows - 1)
    )
    font_scale = min(1.0, max(0.85, cell_width))
    annotation_fontsize = max(
        MIN_ANNOTATION_FONT_SIZE,
        ANNOTATION_FONT_SIZE * font_scale,
    )
    return GridGeometry(
        figsize=(FULL_WIDTH_IN, figure_height),
        cell_width=cell_width,
        annotation_fontsize=annotation_fontsize,
    )
