"""Tests for the unified audio reduction pipeline."""

from pathlib import Path

import numpy as np
import pytest
import torch

from lisa.cli.parse_experiment_args import parse_experiment_arguments
from lisa.cli.train_model import (
    load_audio_embeddings_for_training,
    normalize_determinism_for_time_reduction,
)
from lisa.data.audio_reduction import precompute_pca
from lisa.model.load_model import build_audio_model_from_config, load_audio_model_weights
from lisa.model.nn_modules import TimeReduction
from lisa.utils.constants import N_FEATURES


def _write_audio_embeddings(
    root: Path,
    *,
    embedding_layers: int = 4,
    train: np.ndarray,
    test: np.ndarray,
    suffix: str = '',
) -> None:
    audio_dir = root / 'audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    np.save(audio_dir / f'extract_features_train{embedding_layers}{suffix}.npy', train)
    np.save(audio_dir / f'extract_features_test{embedding_layers}{suffix}.npy', test)


def test_parse_feature_reduction_pca_derives_dimensions():
    args = parse_experiment_arguments(
        [
            '--run-group',
            'test-reduction',
            '--device',
            'cpu',
            '--feature-reduction',
            'pca',
            '--feature-reduction-dim',
            '256',
        ]
    )

    assert args['n_features'] == 256


def test_parse_time_reduction_linear_keeps_feature_size():
    args = parse_experiment_arguments(
        [
            '--run-group',
            'test-reduction',
            '--device',
            'cpu',
            '--time-reduction',
            'linear',
            '--time-reduction-dim',
            '64',
        ]
    )

    assert args['n_features'] == N_FEATURES


def test_parse_time_reduction_query_attention_accepts_shared_args():
    args = parse_experiment_arguments(
        [
            '--run-group',
            'test-reduction',
            '--device',
            'cpu',
            '--time-reduction',
            'query_attn',
            '--time-reduction-dim',
            '8',
            '--time-reduction-num-heads',
            '4',
        ]
    )

    assert args['time_reduction'] == 'query_attn'
    assert args['time_reduction_dim'] == 8
    assert args['time_reduction_num_heads'] == 4


def test_parse_rejects_feature_and_time_reduction_together():
    with pytest.raises(SystemExit):
        parse_experiment_arguments(
            [
                '--run-group',
                'test-reduction',
                '--device',
                'cpu',
                '--feature-reduction',
                'linear',
                '--feature-reduction-dim',
                '256',
                '--time-reduction',
                'linear',
                '--time-reduction-dim',
                '64',
            ]
        )


def test_parse_rejects_hidden_dim_for_non_attention_time_reduction():
    with pytest.raises(SystemExit):
        parse_experiment_arguments(
            [
                '--run-group',
                'test-reduction',
                '--device',
                'cpu',
                '--time-reduction',
                'linear',
                '--time-reduction-dim',
                '64',
                '--time-reduction-hidden-dim',
                '32',
            ]
        )


def test_parse_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    with pytest.raises(SystemExit):
        parse_experiment_arguments(
            [
                '--run-group',
                'test-reduction',
                '--device',
                'cuda',
            ]
        )


def test_adaptive_time_reduction_disables_torch_deterministic_with_warning():
    args = parse_experiment_arguments(
        [
            '--run-group',
            'test-reduction',
            '--device',
            'cpu',
            '--time-reduction',
            'adaptive_avg',
            '--time-reduction-dim',
            '32',
            '--torch-deterministic',
        ]
    )

    with pytest.warns(UserWarning, match='Disabling torch_deterministic'):
        normalize_determinism_for_time_reduction(args)

    assert args['torch_deterministic'] is False


def test_precompute_pca_writes_feature_and_time_outputs(tmp_path: Path):
    train = np.random.RandomState(0).randn(4, 6, 5).astype(np.float32)
    test = np.random.RandomState(1).randn(3, 6, 5).astype(np.float32)
    _write_audio_embeddings(tmp_path, train=train, test=test)

    feature_outputs = precompute_pca(
        dirprocess=tmp_path,
        embedding_layers=4,
        axis='feature',
        n_components=3,
        overwrite=False,
    )
    time_outputs = precompute_pca(
        dirprocess=tmp_path,
        embedding_layers=4,
        axis='time',
        n_components=2,
        overwrite=False,
    )

    feature_train = np.load(feature_outputs['train_embeddings'])
    time_train = np.load(time_outputs['train_embeddings'])
    feature_artifact = np.load(feature_outputs['artifact'])
    time_artifact = np.load(time_outputs['artifact'])

    assert feature_train.shape == (4, 6, 3)
    assert time_train.shape == (4, 2, 5)
    assert feature_artifact['axis'].item() == 'feature'
    assert time_artifact['axis'].item() == 'time'
    assert int(feature_artifact['source_feature_size']) == 5
    assert int(time_artifact['source_time_size']) == 6


def test_time_pca_loading_uses_reduced_audio_only(tmp_path: Path):
    train = np.random.RandomState(2).randn(4, 32, N_FEATURES).astype(np.float32)
    test = np.random.RandomState(3).randn(3, 32, N_FEATURES).astype(np.float32)
    _write_audio_embeddings(tmp_path, train=train, test=test, suffix='_TPCA32')

    args = parse_experiment_arguments(
        [
            '--run-group',
            'test-reduction',
            '--device',
            'cpu',
            '--dirprocess',
            str(tmp_path),
            '--time-reduction',
            'pca',
            '--time-reduction-dim',
            '32',
        ]
    )
    hidden_train, hidden_test = load_audio_embeddings_for_training(args)

    assert hidden_train.shape == (4, 32, N_FEATURES)
    assert hidden_test.shape == (3, 32, N_FEATURES)


def test_time_linear_audio_model_reduces_time_axis(tmp_path: Path):
    train = np.random.RandomState(4).randn(4, 149, N_FEATURES).astype(np.float32)
    test = np.random.RandomState(5).randn(3, 149, N_FEATURES).astype(np.float32)
    _write_audio_embeddings(tmp_path, train=train, test=test)

    args = parse_experiment_arguments(
        [
            '--run-group',
            'test-reduction',
            '--device',
            'cpu',
            '--dirprocess',
            str(tmp_path),
            '--time-reduction',
            'linear',
            '--time-reduction-dim',
            '32',
        ]
    )
    load_audio_embeddings_for_training(args)
    args['audio_time_reduction_input_length'] = 149
    audio_model = build_audio_model_from_config(
        hyper_params=args,
        device=torch.device('cpu'),
    )

    x = torch.randn(2, N_FEATURES, 149)
    y = audio_model(x)
    assert y.shape == (2, N_FEATURES, 32)


@pytest.mark.parametrize(
    'reduction',
    ['linear', 'adaptive_avg', 'adaptive_max', 'query_attn'],
)
def test_explicit_length_time_reductions_reject_output_longer_than_input(
    tmp_path: Path,
    reduction: str,
):
    train = np.random.RandomState(8).randn(4, 16, N_FEATURES).astype(np.float32)
    test = np.random.RandomState(9).randn(3, 16, N_FEATURES).astype(np.float32)
    _write_audio_embeddings(tmp_path, train=train, test=test)

    args = parse_experiment_arguments(
        [
            '--run-group',
            'test-reduction',
            '--device',
            'cpu',
            '--dirprocess',
            str(tmp_path),
            '--time-reduction',
            reduction,
            '--time-reduction-dim',
            '32',
        ]
    )

    with pytest.raises(ValueError):
        load_audio_embeddings_for_training(args)


@pytest.mark.parametrize('reduction', ['mean', 'max', 'attn', 'gated_attn'])
def test_single_token_time_reducers_keep_singleton_time_axis(reduction: str):
    reducer = TimeReduction(
        reduction=reduction,
        n_features=8,
        hidden_dim=4,
    )

    x = torch.randn(2, 8, 5)
    y = reducer(x)

    assert y.shape == (2, 8, 1)


def test_feature_pca_audio_model_is_identity(tmp_path: Path):
    train = np.random.RandomState(6).randn(4, 149, 256).astype(np.float32)
    test = np.random.RandomState(7).randn(3, 149, 256).astype(np.float32)
    _write_audio_embeddings(tmp_path, train=train, test=test, suffix='_PCA256')

    args = parse_experiment_arguments(
        [
            '--run-group',
            'test-reduction',
            '--device',
            'cpu',
            '--dirprocess',
            str(tmp_path),
            '--feature-reduction',
            'pca',
            '--feature-reduction-dim',
            '256',
        ]
    )
    load_audio_embeddings_for_training(args)
    audio_model = build_audio_model_from_config(
        hyper_params=args,
        device=torch.device('cpu'),
    )

    assert isinstance(audio_model, torch.nn.Identity)

    reloaded = load_audio_model_weights(
        audio_model=audio_model,
        run_dir=str(tmp_path / 'missing-run'),
        device=torch.device('cpu'),
    )
    assert isinstance(reloaded, torch.nn.Identity)
