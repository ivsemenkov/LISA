"""Helpers for PCA preprocessing and reduction-aware audio file loading."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from lisa.utils.constants import PREPROCESSED_DATA_DIR


def get_pca_array_suffix(axis: str, n_components: int) -> str:
    """Return the on-disk suffix for PCA-reduced audio arrays."""
    if axis == 'feature':
        return f'_PCA{n_components}'
    if axis == 'time':
        return f'_TPCA{n_components}'
    raise ValueError(f'Unsupported PCA axis: {axis!r}.')


def get_window_suffix(window_tag: str | None) -> str:
    """Return the on-disk filename suffix for a segment-length ablation.

    Empty/None tag returns '' so that default (3-second) paths are byte-for-byte
    identical to the baseline layout. A non-empty tag like 'w5' maps to '_w5'.
    """
    if not window_tag:
        return ''
    return f'_{window_tag}'


def get_audio_embeddings_path(
    dirprocess: str | Path,
    split: str,
    embedding_layers: int,
    *,
    axis: str | None = None,
    n_components: int | None = None,
    window_tag: str | None = None,
) -> Path:
    """Build the path to raw or PCA-reduced audio embeddings."""
    if split not in {'train', 'test'}:
        raise ValueError(f'Unsupported split: {split!r}.')
    suffix = ''
    if axis is not None:
        if n_components is None:
            raise ValueError('n_components is required when axis is provided.')
        suffix = get_pca_array_suffix(axis=axis, n_components=n_components)
    window_suffix = get_window_suffix(window_tag)

    return (
        Path(dirprocess)
        / 'audio'
        / f'extract_features_{split}{embedding_layers}{window_suffix}{suffix}.npy'
    )


def get_dataframe_path(
    dirprocess: str | Path,
    split: str,
    n_subjects: int,
    *,
    window_tag: str | None = None,
) -> Path:
    """Build the path to the per-window MEG/embedding metadata dataframe."""
    if split not in {'train', 'test'}:
        raise ValueError(f'Unsupported split: {split!r}.')
    window_suffix = get_window_suffix(window_tag)
    return (
        Path(dirprocess)
        / 'dataframe'
        / f'df_{split}{n_subjects}{window_suffix}.csv'
    )


def get_chunks_info_path(
    dirprocess: str | Path,
    split: str,
    *,
    window_tag: str | None = None,
) -> Path:
    """Build the path to the audio-chunk info table for a split."""
    if split not in {'train', 'test'}:
        raise ValueError(f'Unsupported split: {split!r}.')
    window_suffix = get_window_suffix(window_tag)
    return (
        Path(dirprocess)
        / 'dataframe'
        / f'chunks_info_{split}{window_suffix}.csv'
    )


def get_pca_artifact_path(
    dirprocess: str | Path,
    embedding_layers: int,
    axis: str,
    n_components: int,
) -> Path:
    """Build the path to the PCA artifact metadata file."""
    stem = 'n' if axis == 'feature' else 'k'
    return (
        Path(dirprocess)
        / 'audio'
        / 'pca_artifacts'
        / axis
        / f'layers{embedding_layers}_{stem}{n_components}.npz'
    )


def _reshape_for_pca(axis: str, embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 3:
        raise ValueError(
            f'Expected audio embeddings with shape (N, T, F), got {embeddings.shape}.'
        )

    if axis == 'feature':
        return embeddings.reshape(-1, embeddings.shape[-1])
    if axis == 'time':
        return np.transpose(embeddings, (0, 2, 1)).reshape(-1, embeddings.shape[1])
    raise ValueError(f'Unsupported PCA axis: {axis!r}.')


def _restore_from_pca(
    axis: str,
    reduced: np.ndarray,
    original_shape: tuple[int, int, int],
) -> np.ndarray:
    n_samples, time_steps, n_features = original_shape
    if axis == 'feature':
        return reduced.reshape(n_samples, time_steps, reduced.shape[-1])
    if axis == 'time':
        restored = reduced.reshape(n_samples, n_features, reduced.shape[-1])
        return np.transpose(restored, (0, 2, 1))
    raise ValueError(f'Unsupported PCA axis: {axis!r}.')


def fit_pca(
    axis: str,
    embeddings: np.ndarray,
    n_components: int,
) -> PCA:
    """Fit PCA on train audio embeddings for the selected axis."""
    flattened = _reshape_for_pca(axis=axis, embeddings=embeddings)
    pca = PCA(n_components=n_components)
    pca.fit(flattened)
    return pca


def transform_embeddings(
    axis: str,
    embeddings: np.ndarray,
    pca: PCA,
) -> np.ndarray:
    """Apply a fitted PCA transform and restore the expected array shape."""
    flattened = _reshape_for_pca(axis=axis, embeddings=embeddings)
    reduced = pca.transform(flattened).astype(embeddings.dtype, copy=False)
    return _restore_from_pca(axis=axis, reduced=reduced, original_shape=embeddings.shape)


def precompute_pca(
    *,
    dirprocess: str | Path,
    embedding_layers: int,
    axis: str,
    n_components: int,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Fit PCA on train audio and save reduced train/test arrays plus metadata."""
    train_path = get_audio_embeddings_path(
        dirprocess=dirprocess,
        split='train',
        embedding_layers=embedding_layers,
    )
    test_path = get_audio_embeddings_path(
        dirprocess=dirprocess,
        split='test',
        embedding_layers=embedding_layers,
    )
    output_train_path = get_audio_embeddings_path(
        dirprocess=dirprocess,
        split='train',
        embedding_layers=embedding_layers,
        axis=axis,
        n_components=n_components,
    )
    output_test_path = get_audio_embeddings_path(
        dirprocess=dirprocess,
        split='test',
        embedding_layers=embedding_layers,
        axis=axis,
        n_components=n_components,
    )
    artifact_path = get_pca_artifact_path(
        dirprocess=dirprocess,
        embedding_layers=embedding_layers,
        axis=axis,
        n_components=n_components,
    )

    outputs = (output_train_path, output_test_path, artifact_path)
    if not overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(
                f'Output files already exist: {", ".join(str(path) for path in existing)}'
            )

    train_embeddings = np.load(train_path)
    test_embeddings = np.load(test_path)
    if train_embeddings.ndim != 3 or test_embeddings.ndim != 3:
        raise ValueError(
            'Expected raw audio embeddings to have shape (N, T, F) for both train and test.'
        )
    if train_embeddings.shape[-1] != test_embeddings.shape[-1]:
        raise ValueError(
            'Train/test feature dimensions do not match: '
            f'{train_embeddings.shape[-1]} vs {test_embeddings.shape[-1]}.'
        )
    if train_embeddings.shape[1] != test_embeddings.shape[1]:
        raise ValueError(
            'Train/test time dimensions do not match: '
            f'{train_embeddings.shape[1]} vs {test_embeddings.shape[1]}.'
        )

    pca = fit_pca(axis=axis, embeddings=train_embeddings, n_components=n_components)
    reduced_train = transform_embeddings(axis=axis, embeddings=train_embeddings, pca=pca)
    reduced_test = transform_embeddings(axis=axis, embeddings=test_embeddings, pca=pca)

    output_train_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_train_path, reduced_train)
    np.save(output_test_path, reduced_test)
    np.savez(
        artifact_path,
        components=pca.components_,
        mean=pca.mean_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        singular_values=pca.singular_values_,
        axis=axis,
        embedding_layers=embedding_layers,
        n_components=n_components,
        source_time_size=train_embeddings.shape[1],
        source_feature_size=train_embeddings.shape[2],
    )

    return {
        'train_embeddings': output_train_path,
        'test_embeddings': output_test_path,
        'artifact': artifact_path,
    }


def main_cli(argv: list[str] | None = None) -> None:
    """CLI entry point for precomputing audio PCA reductions."""
    parser = argparse.ArgumentParser(
        description='Precompute PCA-reduced audio embeddings for feature or time axes.',
        allow_abbrev=False,
    )
    parser.add_argument(
        '--dirprocess',
        type=str,
        default=PREPROCESSED_DATA_DIR,
        help='Root directory with preprocessed embeddings/metadata.',
    )
    parser.add_argument('--embedding-layers', type=int, required=True)
    parser.add_argument(
        '--axis',
        type=str,
        choices=('feature', 'time'),
        required=True,
    )
    parser.add_argument('--n-components', type=int, required=True)
    parser.add_argument(
        '--overwrite',
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    args = parser.parse_args(argv)
    if args.n_components <= 0:
        parser.error('--n-components must be a positive integer.')

    precompute_pca(
        dirprocess=args.dirprocess,
        embedding_layers=args.embedding_layers,
        axis=args.axis,
        n_components=args.n_components,
        overwrite=args.overwrite,
    )
