from lisa.plots.plot_metrics import _assert_configs_aligned
from lisa.plots.retrieval_ablation_artifacts import semantic_config_diffs


def test_metrics_configs_allow_filter_plotting_to_differ():
    configs = [
        {'n_channels_unmix': 25, 'plot_filter_graphs': False},
        {'n_channels_unmix': 25, 'plot_filter_graphs': True},
    ]

    _assert_configs_aligned(
        configs,
        ['without-plots/config.json', 'with-plots/config.json'],
    )


def test_retrieval_configs_allow_filter_plotting_to_differ():
    baseline = {'n_channels_unmix': 25, 'plot_filter_graphs': False}
    plotted = {'n_channels_unmix': 25, 'plot_filter_graphs': True}

    assert semantic_config_diffs(baseline, plotted) == {}
