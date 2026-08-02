import matplotlib as mpl
import pytest

from lisa.plots.plot_style import (
    BASE_FONT_SIZE,
    FULL_WIDTH_IN,
    grid_geometry,
    panel_figsize,
    publication_style,
)


def test_panel_figsize_grows_with_rows() -> None:
    assert panel_figsize(1) == (FULL_WIDTH_IN, 3.2)
    assert panel_figsize(2) == (FULL_WIDTH_IN, 5.0)


def test_grid_geometry_keeps_common_grids_readable() -> None:
    attention = grid_geometry(nrows=3, ncols=6)
    larger_grid = grid_geometry(nrows=5, ncols=4)

    assert attention.figsize[0] == FULL_WIDTH_IN
    assert attention.cell_width > 0.9
    assert attention.annotation_fontsize >= 7.0
    assert larger_grid.figsize[1] > attention.figsize[1]


def test_grid_geometry_rejects_unreadably_dense_grid() -> None:
    with pytest.raises(ValueError, match='split the figure'):
        grid_geometry(nrows=2, ncols=10)


def test_publication_style_does_not_leak_global_rcparams() -> None:
    original_font_size = mpl.rcParams['font.size']

    @publication_style
    def styled_font_size() -> float:
        return float(mpl.rcParams['font.size'])

    assert styled_font_size() == BASE_FONT_SIZE
    assert mpl.rcParams['font.size'] == original_font_size
