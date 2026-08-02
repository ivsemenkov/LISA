"""Model loading utilities for LISA experiments."""

import json
import os
import warnings
from glob import glob
from typing import Any, Dict, Tuple

import numpy as np
import torch

from lisa.model.nn_modules import AudioEmbeddingModel, LISA
from lisa.utils.constants import (
    N_MEG_CHANNELS,
    N_FEATURES,
    N_SUBJECTS,
    PREPROCESSED_DATA_DIR,
)

def load_from_config(
    hyper_params: Dict[str, Any],
    checkpoint_path: str | None,
    device: torch.device,
) -> Tuple[LISA, Dict[str, Any]]:
    """Instantiate a model from config and optionally load weights.

    Args:
        hyper_params: Training config dictionary (parsed CLI/config.json).
            Must contain all required parameters in their final format, including
            'seed' (int or None; used for deterministic spatial attention init; missing
            key will raise KeyError).
            - use_spatial_attention: None, '2D', or '3D' (already normalized)
            - use_unmixing_layer: bool
            - use_subject_layer: bool
            - temporal_filter_bands: list of tuples or None (must be present in config)
            - All other parameters must be present and correctly typed.
        checkpoint_path: Path to checkpoint .pt file; if None, returns uninitialized weights.
        device: Torch device for the model.

    Returns:
        Model instance moved to the specified device.
        Hyperparameters dictionary.

    Raises:
        KeyError: If any required parameter is missing from hyper_params.
        FileNotFoundError: If coordinate files are not found.
    """
    dirprocess = hyper_params['dirprocess']
    if not os.path.exists(dirprocess):
        dirprocess_default = PREPROCESSED_DATA_DIR
        warnings.warn(
            f'dirprocess {dirprocess} does not exist, using default path: {dirprocess_default}.',
            UserWarning,
        )
        dirprocess = dirprocess_default

    temporal_filter_type = hyper_params['temporal_filter_type']
    assert temporal_filter_type in ('conv', 'filtfilt'), (
        f'temporal_filter_type must be "conv" or "filtfilt", got {temporal_filter_type}'
    )
    meg_time_reduction = hyper_params['time_reduction']
    if meg_time_reduction == 'pca':
        meg_time_reduction = 'linear'
    checkpoint = None
    state_dict = None
    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False
        )
        state_dict = checkpoint['model_state_dict']

    model = LISA(
        n_channels_input=N_MEG_CHANNELS,
        n_channels_attention=hyper_params['n_channels_attention'],
        n_channels_unmix=hyper_params['n_channels_unmix'],
        use_spatial_attention=hyper_params['use_spatial_attention'],
        n_spatial_harmonics=hyper_params['n_spatial_harmonics'],
        coords_xyz=np.load(os.path.join(dirprocess, 'coords', 'sensor_xyz.npy')),
        coords_xy_scaled=np.load(
            os.path.join(dirprocess, 'coords', 'coords208_xy_scaled.npy')
        ),
        spatial_dropout_number=hyper_params['spatial_dropout_number'],
        spatial_dropout_radius=hyper_params['spatial_dropout_radius'],
        n_subjects=N_SUBJECTS,
        n_channels_block=hyper_params['n_channels_block'],
        n_features=hyper_params['n_features'],
        head_pool=hyper_params['head_pool'],
        head_stride=hyper_params['head_stride'],
        time_reduction=meg_time_reduction,
        time_reduction_dim=hyper_params['time_reduction_dim'],
        time_reduction_input_length=hyper_params['time_reduction_input_length'],
        time_reduction_hidden_dim=hyper_params['time_reduction_hidden_dim'],
        time_reduction_num_heads=hyper_params['time_reduction_num_heads'],
        meg_sr=hyper_params['meg_sr'],
        temporal_filter_type=temporal_filter_type,
        temporal_filter_bands=hyper_params['temporal_filter_bands'],
        temporal_filter_freeze=hyper_params['temporal_filter_freeze'],
        temporal_filter_kernel_size=hyper_params['temporal_filter_kernel_size'],
        temporal_filter_padding_mode=hyper_params['temporal_filter_padding_mode'],
        tf_gated=hyper_params['tf_gated'],
        tf_gate_per_channel=hyper_params['tf_gate_per_channel'],
        tf_gate_init_alpha=hyper_params['tf_gate_init_alpha'],
        tf_gelu=hyper_params['tf_gelu'],
        n_temporal_module_blocks=hyper_params['n_temporal_module_blocks'],
        use_unmixing_layer=hyper_params['use_unmixing_layer'],
        use_subject_layer=hyper_params['use_subject_layer'],
        dropout_center_low=hyper_params['dropout_center_low'],
        dropout_center_high=hyper_params['dropout_center_high'],
        seed=hyper_params['seed'],
    )

    if state_dict is not None:
        model = model.to('cpu')
        model.load_state_dict(state_dict)
    model = model.to(device)
    return model, hyper_params


def build_audio_model_from_config(
    hyper_params: Dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """Instantiate the separate audio embedding model from config."""
    n_features = hyper_params['n_features']
    feature_reduction = hyper_params['feature_reduction']
    feature_reduction_dim = hyper_params['feature_reduction_dim']
    audio_feature_reduction = 'linear' if feature_reduction == 'linear' else 'none'
    audio_time_reduction = hyper_params['time_reduction']
    if audio_time_reduction == 'pca':
        audio_time_reduction = 'none'

    if feature_reduction == 'linear':
        if feature_reduction_dim != n_features:
            raise ValueError(
                'Config is invalid: n_features must match feature_reduction_dim when feature_reduction="linear".'
            )
        n_input_features = N_FEATURES
    elif feature_reduction == 'pca':
        if feature_reduction_dim != n_features:
            raise ValueError(
                'Config is invalid: n_features must match feature_reduction_dim when feature_reduction="pca".'
            )
        n_input_features = n_features
    elif feature_reduction == 'none':
        n_input_features = n_features
    else:
        raise ValueError(
            f'Config is invalid: unknown feature_reduction {feature_reduction!r}.'
        )

    if audio_feature_reduction == 'none' and audio_time_reduction == 'none':
        model = torch.nn.Identity()
        return model.to(device)

    model = AudioEmbeddingModel(
        n_input_features=n_input_features,
        n_output_features=n_features,
        feature_reduction=audio_feature_reduction,
        time_reduction=audio_time_reduction,
        time_reduction_dim=hyper_params['time_reduction_dim'],
        time_reduction_input_length=hyper_params['audio_time_reduction_input_length'],
        time_reduction_hidden_dim=hyper_params['time_reduction_hidden_dim'],
        time_reduction_num_heads=hyper_params['time_reduction_num_heads'],
    )
    return model.to(device)


def load_audio_model_weights(
    audio_model: torch.nn.Module,
    run_dir: str,
    device: torch.device,
    *,
    strict_if_present: bool = True,
) -> torch.nn.Module:
    """Load audio model weights from a run directory."""
    audio_model_path = os.path.join(run_dir, 'audio_model_best.pt')

    if os.path.exists(audio_model_path):
        checkpoint = torch.load(audio_model_path, map_location=device, weights_only=False)
        state_dict = checkpoint['audio_model_state_dict']
        audio_model.load_state_dict(state_dict, strict=strict_if_present)
        return audio_model

    if any(p.numel() > 0 for p in audio_model.parameters()):
        raise FileNotFoundError(
            f'Audio model checkpoint not found in {run_dir}. Expected {audio_model_path}.'
        )

    return audio_model


def _find_run_dirs(group_dir: str, run_id: str) -> list[str]:
    """Find run directories for a run_id, supporting normal and ClearML offline dirs.

    Run dirs are named either *_{run_id} (normal) or *_offline-{run_id} (ClearML offline).
    """
    patterns = [
        os.path.join(group_dir, f'*_{run_id}', 'config.json'),
        os.path.join(group_dir, f'*_offline-{run_id}', 'config.json'),
    ]
    run_dirs: set[str] = set()
    for pat in patterns:
        run_dirs.update(os.path.dirname(path) for path in glob(pat))
    return sorted(run_dirs)


def _load_run_dir_and_config(
    run_id: str, experiments_root: str, run_group: str
) -> Tuple[str, Dict[str, Any]]:
    """Resolve a unique run directory and load its config."""
    group_dir = os.path.join(experiments_root, run_group)
    if not os.path.isdir(group_dir):
        raise FileNotFoundError(f'Run group directory not found: {group_dir}')

    run_dirs = _find_run_dirs(group_dir, run_id)
    if len(run_dirs) == 0:
        raise FileNotFoundError(
            f'No run dir matching *_{{run_id}} or *_offline-{{run_id}} under {group_dir} (run_id={run_id!r})'
        )
    if len(run_dirs) > 1:
        raise ValueError(
            f'Multiple run dirs match run_id {run_id!r} under {group_dir}: {run_dirs}'
        )

    run_dir = run_dirs[0]
    with open(os.path.join(run_dir, 'config.json'), encoding='utf-8') as f:
        hyper_params = json.load(f)

    return run_dir, hyper_params


def load_model(
    run_id: str, experiments_root: str, run_group: str
) -> Tuple[LISA, Dict[str, Any]]:
    """Load a trained model and metadata for plotting/analysis.

    Args:
        run_id: ClearML run id.
        experiments_root: Root directory containing experiment folders.
        run_group: Run group id (experiment dir is experiments_root/run_group/*_run_id/).

    Returns:
        model: Loaded model (eval mode).
        hyper_params: Hyperparameters dictionary.

    Raises:
        FileNotFoundError: If run_group dir does not exist or no run dir matches run_id.
        ValueError: If more than one run dir matches run_id.
        KeyError: If required parameters are missing from config.json.
    """
    run_dir, hyper_params = _load_run_dir_and_config(
        run_id=run_id,
        experiments_root=experiments_root,
        run_group=run_group,
    )
    checkpoint_path = os.path.join(run_dir, f'{hyper_params["checkpoint"]}.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f'Main model checkpoint not found: {checkpoint_path}'
        )

    model, hyper_params = load_from_config(
        hyper_params=hyper_params,
        checkpoint_path=checkpoint_path,
        device=torch.device('cpu'),
    )
    model.eval()
    return model, hyper_params


def load_retrieval_models(
    run_id: str, experiments_root: str, run_group: str
) -> Tuple[LISA, torch.nn.Module, Dict[str, Any]]:
    """Load the eval-mode MEG model and separate eval-mode audio model for retrieval."""
    run_dir, hyper_params = _load_run_dir_and_config(
        run_id=run_id,
        experiments_root=experiments_root,
        run_group=run_group,
    )
    checkpoint_path = os.path.join(run_dir, f'{hyper_params["checkpoint"]}.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f'Main model checkpoint not found: {checkpoint_path}'
        )
    model, hyper_params = load_from_config(
        hyper_params=hyper_params,
        checkpoint_path=checkpoint_path,
        device=torch.device('cpu'),
    )
    model.eval()

    audio_model = build_audio_model_from_config(
        hyper_params=hyper_params,
        device=torch.device('cpu'),
    )
    audio_model = load_audio_model_weights(
        audio_model=audio_model,
        run_dir=run_dir,
        device=torch.device('cpu'),
        strict_if_present=True,
    )
    audio_model.eval()
    return model, audio_model, hyper_params
