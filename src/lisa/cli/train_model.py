"""CLI entry point for training LISA models and producing artifacts."""

import os
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from lisa.cli.parse_experiment_args import parse_experiment_arguments
from lisa.data.audio_reduction import (
    get_audio_embeddings_path,
    get_dataframe_path,
)
from lisa.data.datasets import DoubleDataset
from lisa.model.load_model import (
    build_audio_model_from_config,
    load_from_config,
)
from lisa.model.nn_modules import ConvHead
from lisa.training.trainer import Trainer
from lisa.utils.constants import N_FEATURES, N_SUBJECTS
from lisa.utils.determinism import fix_seed
from lisa.utils.experiment_logger import ExperimentProtocol, create_experiment


def load_audio_embeddings_for_training(
    hyper_params: Dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Load audio embeddings according to the active reduction configuration."""
    feature_reduction = hyper_params['feature_reduction']
    feature_reduction_dim = hyper_params['feature_reduction_dim']
    time_reduction = hyper_params['time_reduction']
    time_reduction_dim = hyper_params['time_reduction_dim']
    axis = None
    n_components = None
    if feature_reduction == 'pca':
        axis = 'feature'
        n_components = feature_reduction_dim
    elif time_reduction == 'pca':
        axis = 'time'
        n_components = time_reduction_dim

    window_tag = hyper_params['window_tag']
    train_path = get_audio_embeddings_path(
        dirprocess=hyper_params['dirprocess'],
        split='train',
        embedding_layers=hyper_params['embedding_layers'],
        axis=axis,
        n_components=n_components,
        window_tag=window_tag,
    )
    test_path = get_audio_embeddings_path(
        dirprocess=hyper_params['dirprocess'],
        split='test',
        embedding_layers=hyper_params['embedding_layers'],
        axis=axis,
        n_components=n_components,
        window_tag=window_tag,
    )

    hidden_train = np.load(train_path)
    hidden_test = np.load(test_path)
    if hidden_train.ndim != 3 or hidden_test.ndim != 3:
        raise ValueError(
            'Expected audio embeddings with shape (N, T, F) for both train and test.'
        )
    if hidden_train.shape[-1] != hidden_test.shape[-1]:
        raise ValueError(
            'Audio embedding feature dimensions do not match between train and test: '
            f'train={hidden_train.shape[-1]}, test={hidden_test.shape[-1]}.'
        )
    if hidden_train.shape[1] != hidden_test.shape[1]:
        raise ValueError(
            'Audio embedding time dimensions do not match between train and test: '
            f'train={hidden_train.shape[1]}, test={hidden_test.shape[1]}.'
        )

    if feature_reduction == 'pca':
        if hidden_train.shape[-1] != feature_reduction_dim:
            raise ValueError(
                'Loaded feature-PCA embedding dimension does not match --feature-reduction-dim: '
                f'loaded={hidden_train.shape[-1]}, expected={feature_reduction_dim}.'
            )
    elif hidden_train.shape[-1] != N_FEATURES:
        raise ValueError(
            'Non-PCA feature reduction expects full wav2vec embeddings before any learned reduction: '
            f'loaded={hidden_train.shape[-1]}, expected={N_FEATURES}.'
        )

    if (
        feature_reduction == 'linear'
        and hyper_params['n_features'] != feature_reduction_dim
    ):
        raise ValueError(
            'Model output size n_features must match --feature-reduction-dim in feature linear mode.'
        )

    if time_reduction == 'pca':
        if hidden_train.shape[-1] != N_FEATURES:
            raise ValueError(
                'Time PCA embeddings must preserve the original wav2vec feature size.'
            )
        if hidden_train.shape[1] != time_reduction_dim:
            raise ValueError(
                'Loaded time-PCA embedding length does not match --time-reduction-dim: '
                f'loaded={hidden_train.shape[1]}, expected={time_reduction_dim}.'
            )
    elif time_reduction != 'none':
        if hidden_train.shape[-1] != N_FEATURES:
            raise ValueError(
                'Non-PCA time reduction expects full wav2vec feature size before temporal reduction.'
            )
        if time_reduction in ('linear', 'adaptive_avg', 'adaptive_max', 'query_attn'):
            if time_reduction_dim is None:
                raise ValueError(
                    f'time_reduction={time_reduction!r} requires time_reduction_dim.'
                )
            if time_reduction_dim > hidden_train.shape[1]:
                raise ValueError(
                    '--time-reduction-dim cannot exceed the loaded audio time length '
                    f'for time_reduction={time_reduction!r}.'
                )

    return hidden_train, hidden_test


def _int_column(df: pd.DataFrame, column: str, split_name: str) -> np.ndarray:
    if column not in df.columns:
        raise ValueError(f'{split_name} dataframe is missing required column {column}.')
    if df[column].isna().any():
        raise ValueError(f'{split_name} dataframe contains null {column} values.')
    values = df[column].to_numpy()
    try:
        int_values = values.astype(np.int64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'{split_name} dataframe contains non-integer {column} values.'
        ) from exc
    if not np.array_equal(values, int_values):
        raise ValueError(f'{split_name} dataframe contains non-integer {column} values.')
    return int_values


def _validate_wav_indices(df: pd.DataFrame, hidden: np.ndarray, split_name: str) -> None:
    wav_indices = _int_column(df, 'wav_index', split_name)
    if len(wav_indices) == 0:
        raise ValueError(f'{split_name} dataframe is empty.')
    if int(wav_indices.min()) < 0 or int(wav_indices.max()) >= hidden.shape[0]:
        raise ValueError(
            f'{split_name} dataframe wav_index values are out of bounds for '
            f'{hidden.shape[0]} embeddings.'
        )


def _split_train_val_indices(
    df_train: pd.DataFrame, val_story_sounds: list[tuple[int, int]]
) -> tuple[list[int], list[int]]:
    story_ids = _int_column(df_train, 'story_id', 'train')
    sound_ids = _int_column(df_train, 'sound_id', 'train')
    requested_val = set(val_story_sounds)
    present = set(zip(story_ids.tolist(), sound_ids.tolist()))
    missing = requested_val - present
    if missing:
        raise ValueError(
            f'Requested --val-story-sounds are not present in train: {sorted(missing)}.'
        )
    val_mask = np.array(
        [
            (story_id, sound_id) in requested_val
            for story_id, sound_id in zip(story_ids, sound_ids)
        ],
        dtype=bool,
    )
    train_indices = np.flatnonzero(~val_mask).tolist()
    val_indices = np.flatnonzero(val_mask).tolist()
    if not train_indices or not val_indices:
        raise ValueError('Train and validation splits must both be non-empty.')

    train_wav = set(_int_column(df_train.iloc[train_indices], 'wav_index', 'train fit'))
    val_wav = set(_int_column(df_train.iloc[val_indices], 'wav_index', 'validation'))
    overlap = train_wav & val_wav
    if overlap:
        raise ValueError(f'Train/validation wav_index overlap: {sorted(overlap)[:20]}.')
    return train_indices, val_indices


def prepare_training_datasets(
    hyper_params: Dict[str, Any],
    hidden_train: np.ndarray,
    hidden_test: np.ndarray,
) -> tuple[Subset, Subset, DoubleDataset, np.ndarray, np.ndarray]:
    """Load MEG/dataframes once, create datasets, and populate derived time lengths."""
    hyper_params['audio_time_reduction_input_length'] = None
    hyper_params['time_reduction_input_length'] = None

    dirprocess = hyper_params['dirprocess']
    window_tag = hyper_params['window_tag']
    df_train = pd.read_csv(
        get_dataframe_path(dirprocess, 'train', N_SUBJECTS, window_tag=window_tag)
    )
    df_test = pd.read_csv(
        get_dataframe_path(dirprocess, 'test', N_SUBJECTS, window_tag=window_tag)
    )
    _validate_wav_indices(df_train, hidden_train, 'train')
    _validate_wav_indices(df_test, hidden_test, 'test')

    train_indices, val_indices = _split_train_val_indices(
        df_train, hyper_params['val_story_sounds']
    )
    df_train_fit = df_train.iloc[train_indices]
    df_val = df_train.iloc[val_indices]
    train_candidate_ids = np.array(
        sorted(set(_int_column(df_train_fit, 'wav_index', 'train fit').tolist())),
        dtype=np.int64,
    )
    val_candidate_ids = np.array(
        sorted(set(_int_column(df_val, 'wav_index', 'validation').tolist())),
        dtype=np.int64,
    )

    meg = dict(
        np.load(
            os.path.join(
                dirprocess, 'meg', f'meg{N_SUBJECTS}_sr{hyper_params["meg_sr"]}.npz'
            )
        )
    )
    dataset_train_full = DoubleDataset(
        meg=meg,
        hidden=hidden_train,
        df=df_train,
        meg_sr=hyper_params['meg_sr'],
        meg_offset=hyper_params['meg_offset'],
    )
    dataset_train = Subset(dataset_train_full, train_indices)
    dataset_val = Subset(dataset_train_full, val_indices)
    dataset_test = DoubleDataset(
        meg=meg,
        hidden=hidden_test,
        df=df_test,
        meg_sr=hyper_params['meg_sr'],
        meg_offset=hyper_params['meg_offset'],
    )

    if hyper_params['time_reduction'] in ('linear', 'pca'):
        if len(dataset_train) == 0:
            raise ValueError(
                'Cannot infer MEG time reduction input length from an empty training dataset.'
            )
        meg_sample, _, _, _, _ = dataset_train[0]
        head = ConvHead(
            n_channels=1,
            n_features=1,
            pool=hyper_params['head_pool'],
            head_stride=hyper_params['head_stride'],
        )
        hyper_params['time_reduction_input_length'] = head.output_length(
            input_length=int(meg_sample.shape[-1])
        )
    if hyper_params['time_reduction'] == 'linear':
        hyper_params['audio_time_reduction_input_length'] = int(hidden_train.shape[1])
    return dataset_train, dataset_val, dataset_test, train_candidate_ids, val_candidate_ids


def normalize_determinism_for_time_reduction(hyper_params: Dict[str, Any]) -> None:
    """Disable unsupported deterministic mode for adaptive pooling reducers."""
    if (
        hyper_params['time_reduction'] in ('adaptive_avg', 'adaptive_max')
        and hyper_params['torch_deterministic']
    ):
        warnings.warn(
            'time_reduction='
            f'{hyper_params["time_reduction"]!r} is incompatible with '
            'torch_deterministic=True in training. '
            'Disabling torch_deterministic for this run.',
            UserWarning,
            stacklevel=2,
        )
        hyper_params['torch_deterministic'] = False


def train(
    hyper_params: Dict[str, Any],
    experiment: ExperimentProtocol,
    hidden_train: np.ndarray,
    hidden_test: np.ndarray,
    dataset_train: Subset,
    dataset_val: Subset,
    dataset_test: DoubleDataset,
    train_candidate_ids: np.ndarray,
    val_candidate_ids: np.ndarray,
) -> Path:
    """Run training and return the experiment directory."""
    # Single place for RNG seeding: after experiment (e.g. ClearML) init, before model/data.
    seed = hyper_params['seed']
    fix_seed(
        seed=seed,
        torch_deterministic=hyper_params['torch_deterministic'],
    )

    experiment_dir = experiment.dir
    if len(dataset_train) < hyper_params['batch_size']:
        raise ValueError(
            'Training dataset is smaller than --batch-size while drop_last=True: '
            f'train_size={len(dataset_train)}, batch_size={hyper_params["batch_size"]}.'
        )
    for split_name, split_size in (
        ('validation', len(dataset_val)),
        ('test', len(dataset_test)),
    ):
        if split_size < 2:
            raise ValueError(
                f'{split_name} split must contain at least two samples for CLIP loss.'
            )
        remainder = split_size % hyper_params['test_batch_size']
        if remainder == 1:
            raise ValueError(
                f'{split_name} split would produce a final batch with one sample '
                'and no negatives. Change --test-batch-size or the split. '
                f'split_size={split_size}, test_batch_size={hyper_params["test_batch_size"]}.'
            )

    # Create dataloaders. Shuffle order is deterministic when generator is seeded.
    # DoubleDataset does not use RNG in __getitem__, so worker_init_fn is not needed.
    train_generator = None
    if seed is not None:
        train_generator = torch.Generator()
        train_generator.manual_seed(seed)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=hyper_params['batch_size'],
        num_workers=hyper_params['dl_n_workers'],
        shuffle=True,
        drop_last=True,
        generator=train_generator,
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=hyper_params['test_batch_size'],
        num_workers=hyper_params['dl_n_workers'],
        shuffle=False,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=hyper_params['test_batch_size'],
        num_workers=hyper_params['dl_n_workers'],
        shuffle=False,
    )
    # Create model
    model, _ = load_from_config(
        hyper_params=hyper_params,
        checkpoint_path=None,
        device=hyper_params['device'],
    )
    audio_model = build_audio_model_from_config(
        hyper_params=hyper_params,
        device=hyper_params['device'],
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        audio_model=audio_model,
        clip_temperature=hyper_params['clip_temperature'],
        lr_fe=hyper_params['lr_fe'],
        weight_decay=hyper_params['weight_decay'],
        clip_temperature_lr=hyper_params['clip_temperature_lr'],
        optim=hyper_params['optim'],
        checkpoint=hyper_params['checkpoint'],
        experiment=experiment,
        bids_root=hyper_params['meg_files_dir'],
        meg_format=hyper_params['meg_format'],
        sampling_rate=hyper_params['meg_sr'],
        save_test_top_k=hyper_params['save_test_top_k'],
        max_branches_to_plot=hyper_params['max_branches_to_plot'],
        plot_filter_graphs=hyper_params['plot_filter_graphs'],
        tf_unfreeze_epoch=hyper_params['tf_unfreeze_epoch'],
        lr_temporal_mult=hyper_params['lr_temporal_mult'],
        wd_temporal=hyper_params['wd_temporal'],
        tf_smooth=hyper_params['tf_smooth'],
    )

    # Train
    trainer.fit(
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        dataloader_test=dataloader_test,
        hidden_train_bank=hidden_train[train_candidate_ids],
        hidden_val_bank=hidden_train[val_candidate_ids],
        hidden_test_bank=hidden_test,
        train_candidate_ids=train_candidate_ids,
        val_candidate_ids=val_candidate_ids,
        test_candidate_ids=np.arange(hidden_test.shape[0], dtype=np.int64),
        train_size=len(dataset_train),
        validation_size=len(dataset_val),
        test_size=len(dataset_test),
        nepoch=hyper_params['nepoch'],
        early_stopping_patience=hyper_params['early_stopping_patience'],
        device=hyper_params['device'],
    )

    return experiment_dir


def save_filters(
    model: torch.nn.Module, hyper_params: Dict[str, Any], experiment_dir: str | Path
) -> Dict[str, np.ndarray]:
    """Save spatial/temporal filters to a .npz file.

    Args:
        model: Trained model instance.
        hyper_params: Dict of parsed CLI arguments.
        experiment_dir: Path to experiment directory.

    Returns:
        Dict with saved filter arrays.
    """
    filters = {}

    # Extract and save spatial filters
    spatial_filter_weight, spatial_filter_bias = model.extract_spatial_filters()
    filters['overall_spatial_filter_weight'] = spatial_filter_weight.numpy()
    if spatial_filter_bias is not None:
        filters['overall_spatial_filter_bias'] = spatial_filter_bias.numpy()

    # Extract and save temporal filters
    temporal_filters = model.extract_temporal_filters()
    filters['temporal_filters_weight'] = temporal_filters.numpy()

    np.savez(
        os.path.join(experiment_dir, f'filters_{hyper_params["run_name"]}.npz'),
        **filters,
    )
    return filters


def main():
    """Parse CLI args, run training, save filters, and plot diagnostics."""
    hyper_params = parse_experiment_arguments()
    normalize_determinism_for_time_reduction(hyper_params)
    hidden_train, hidden_test = load_audio_embeddings_for_training(hyper_params)
    (
        dataset_train,
        dataset_val,
        dataset_test,
        train_candidate_ids,
        val_candidate_ids,
    ) = prepare_training_datasets(
        hyper_params=hyper_params,
        hidden_train=hidden_train,
        hidden_test=hidden_test,
    )
    experiment = create_experiment(hyper_params)
    try:
        experiment_dir = train(
            hyper_params=hyper_params,
            experiment=experiment,
            hidden_train=hidden_train,
            hidden_test=hidden_test,
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            dataset_test=dataset_test,
            train_candidate_ids=train_candidate_ids,
            val_candidate_ids=val_candidate_ids,
        )

        model, _ = load_from_config(
            hyper_params=hyper_params,
            checkpoint_path=os.path.join(
                experiment_dir, f'{hyper_params["checkpoint"]}.pt'
            ),
            device=torch.device('cpu'),
        )
        model.eval()

        save_filters(
            model=model, hyper_params=hyper_params, experiment_dir=experiment_dir
        )
    except Exception:
        traceback.print_exc()
        raise
    finally:
        experiment.finish()


if __name__ == '__main__':
    main()
