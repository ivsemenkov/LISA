import pytest

from lisa.cli.parse_experiment_args import (
    parse_experiment_arguments,
    parse_story_sounds,
)


def test_continue_training_is_removed_from_parser():
    with pytest.raises(SystemExit):
        parse_experiment_arguments(
            [
                '--run-group',
                'test-publication',
                '--device',
                'cpu',
                '--continue-training',
                'checkpoint.pt',
            ]
        )


@pytest.mark.parametrize(
    'legacy_option', ['--plot-subject', '--plot-session', '--plot-task']
)
def test_legacy_plot_selection_options_are_removed_from_parser(legacy_option):
    with pytest.raises(SystemExit):
        parse_experiment_arguments(
            [
                '--run-group',
                'test-publication',
                '--device',
                'cpu',
                legacy_option,
                '0',
            ]
        )


def test_spatial_dropout_number_must_be_integer():
    with pytest.raises(SystemExit):
        parse_experiment_arguments(
            [
                '--run-group',
                'test-publication',
                '--device',
                'cpu',
                '--spatial-dropout-number',
                '1.5',
            ]
        )


def test_training_defaults_match_paper_baseline():
    args = parse_experiment_arguments(
        ['--run-group', 'test-publication', '--device', 'cpu']
    )

    assert args['n_channels_unmix'] == 25
    assert args['n_channels_block'] == 25
    assert args['head_pool'] == 'single_conv'
    assert args['n_temporal_module_blocks'] == 2
    assert args['nepoch'] == 50
    assert args['early_stopping_patience'] == 7
    assert args['seed'] == 42
    assert args['torch_deterministic'] is False
    assert args['plot_filter_graphs'] is False
    assert args['logger'] == 'local'
    assert not {'plot_subject', 'plot_session', 'plot_task'} & args.keys()


def test_filter_graph_plotting_is_opt_in():
    args = parse_experiment_arguments(
        [
            '--run-group',
            'test-publication',
            '--device',
            'cpu',
            '--plot-filter-graphs',
        ]
    )

    assert args['plot_filter_graphs'] is True


def test_training_defaults_allow_semantic_none_and_auto_values():
    args = parse_experiment_arguments(
        [
            '--run-group',
            'test-publication',
            '--device',
            'cpu',
            '--n-channels-unmix',
            '13',
            '--n-channels-block',
            'auto',
            '--seed',
            'none',
            '--early-stopping-patience',
            'none',
        ]
    )

    assert args['n_channels_block'] == 13
    assert args['seed'] is None
    assert args['early_stopping_patience'] is None


def test_parse_story_sounds_rejects_duplicates():
    with pytest.raises(Exception, match='duplicate'):
        parse_story_sounds('3:4,3:4')
