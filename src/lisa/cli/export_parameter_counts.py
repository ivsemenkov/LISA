"""Export per-run model parameter counts and architecture metadata to CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm import tqdm

from lisa.model.load_model import build_audio_model_from_config, load_from_config
from lisa.training.criteria import CLIPLoss
from lisa.utils.constants import OUTPUTS_DIR


ARCHITECTURE_FIELDS = (
    'n_channels_attention',
    'n_channels_unmix',
    'use_spatial_attention',
    'n_spatial_harmonics',
    'n_channels_block',
    'n_features',
    'head_pool',
    'head_stride',
    'temporal_filter_type',
    'temporal_filter_bands',
    'temporal_filter_freeze',
    'temporal_filter_kernel_size',
    'tf_gated',
    'tf_gate_per_channel',
    'tf_gelu',
    'n_temporal_module_blocks',
    'use_unmixing_layer',
    'use_subject_layer',
    'feature_reduction',
    'feature_reduction_dim',
    'time_reduction',
    'time_reduction_dim',
    'time_reduction_input_length',
    'audio_time_reduction_input_length',
    'time_reduction_hidden_dim',
    'time_reduction_num_heads',
)

CONFIG_COLUMNS = (
    'seed',
    'n_temporal_module_blocks',
    'n_channels_unmix',
    'n_channels_block',
    'n_channels_attention',
    'use_spatial_attention',
    'use_unmixing_layer',
    'use_subject_layer',
    'temporal_filter_type',
    'temporal_filter_kernel_size',
    'temporal_filter_freeze',
    'head_pool',
    'head_stride',
    'n_features',
    'feature_reduction',
    'feature_reduction_dim',
    'time_reduction',
    'time_reduction_dim',
    'time_reduction_input_length',
    'audio_time_reduction_input_length',
    'time_reduction_hidden_dim',
    'time_reduction_num_heads',
    'window_tag',
)

COUNT_COLUMNS = (
    'trainable_params',
    'total_params',
    'meg_model_trainable_params',
    'meg_model_total_params',
    'meg_core_trainable_params',
    'meg_core_total_params',
    'meg_projection_resampler_trainable_params',
    'meg_projection_resampler_total_params',
    'meg_time_reducer_trainable_params',
    'meg_time_reducer_total_params',
    'audio_adapter_trainable_params',
    'audio_adapter_total_params',
    'audio_feature_reducer_trainable_params',
    'audio_feature_reducer_total_params',
    'audio_time_reducer_trainable_params',
    'audio_time_reducer_total_params',
    'audio_other_trainable_params',
    'audio_other_total_params',
    'criterion_trainable_params',
    'criterion_total_params',
    'optimizer_trainable_params',
)

METRIC_COLUMNS = (
    'metrics_status',
    'metrics_error',
    'metrics_path',
    'best_epoch',
    'loss_final_test',
    'top1s_final_test',
    'top10s_final_test',
    'top1_final_test_pct',
    'top10_final_test_pct',
)

CSV_COLUMNS = (
    'status',
    'error',
    'model_family',
    'model_subfamily',
    'ablation',
    'architecture_id',
    'architecture_label',
    'run_group',
    'run_name',
    'run_id',
    'experiment_dir',
    'config_path',
    *CONFIG_COLUMNS,
    'window_duration_tag',
    'window_compute_variant',
    'audio_precomputed_reduction',
    *COUNT_COLUMNS,
    *METRIC_COLUMNS,
)


def _count_parameters(
    parameters: Iterable[torch.nn.Parameter],
) -> tuple[int, int]:
    parameters = list(parameters)
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    return trainable, total


def _partition_parameters(
    parent: torch.nn.Module,
    named_parts: dict[str, torch.nn.Module | None],
) -> dict[str, tuple[int, int]]:
    """Count disjoint child modules and the remaining parent parameters."""
    parent_parameters = {id(parameter): parameter for parameter in parent.parameters()}
    assigned: set[int] = set()
    counts: dict[str, tuple[int, int]] = {}

    for name, module in named_parts.items():
        part_parameters = [] if module is None else list(module.parameters())
        part_ids = {id(parameter) for parameter in part_parameters}
        shared_ids = assigned.intersection(part_ids)
        if shared_ids:
            raise ValueError(
                f'Parameter partition {name!r} shares {len(shared_ids)} parameter(s) '
                'with an earlier partition.'
            )
        unknown_ids = part_ids.difference(parent_parameters)
        if unknown_ids:
            raise ValueError(
                f'Parameter partition {name!r} contains parameters outside its parent.'
            )
        assigned.update(part_ids)
        counts[name] = _count_parameters(part_parameters)

    remaining = [
        parameter
        for parameter_id, parameter in parent_parameters.items()
        if parameter_id not in assigned
    ]
    counts['other'] = _count_parameters(remaining)
    return counts


def _parse_run_id(run_dir_name: str) -> str:
    if '_offline-' in run_dir_name:
        return run_dir_name.rsplit('_offline-', 1)[1]
    if '_' in run_dir_name:
        return run_dir_name.rsplit('_', 1)[1]
    return ''


def _read_final_test_metrics(run_dir: Path) -> dict[str, Any]:
    """Read metrics produced after restoring the best-validation checkpoint."""
    metrics_path = run_dir / 'metrics.csv'
    result: dict[str, Any] = {
        'metrics_status': 'missing',
        'metrics_error': '',
        'metrics_path': str(metrics_path),
        'best_epoch': '',
        'loss_final_test': '',
        'top1s_final_test': '',
        'top10s_final_test': '',
        'top1_final_test_pct': '',
        'top10_final_test_pct': '',
    }
    if not metrics_path.exists():
        result['metrics_error'] = 'metrics.csv not found'
        return result

    required_columns = {
        'epoch',
        'loss_final_test',
        'top1s_final_test',
        'top10s_final_test',
    }
    try:
        with metrics_path.open(encoding='utf-8', newline='') as file:
            reader = csv.DictReader(file)
            missing_columns = required_columns.difference(reader.fieldnames or ())
            if missing_columns:
                raise ValueError(
                    'metrics.csv is missing required columns: '
                    + ', '.join(sorted(missing_columns))
                )
            final_rows = [
                row
                for row in reader
                if all(
                    row[column] is not None
                    and row[column].strip()
                    and row[column].strip().lower() not in {'nan', 'none'}
                    for column in required_columns
                )
            ]

        if not final_rows:
            result['metrics_error'] = (
                'metrics.csv has no complete final-test metric row'
            )
            return result

        final = final_rows[-1]
        best_epoch_value = float(final['epoch'])
        loss = float(final['loss_final_test'])
        top1 = float(final['top1s_final_test'])
        top10 = float(final['top10s_final_test'])
        if not best_epoch_value.is_integer():
            raise ValueError(f'Final-test epoch is not an integer: {best_epoch_value}')
        if not all(math.isfinite(value) for value in (loss, top1, top10)):
            raise ValueError('Final-test metrics contain non-finite values')
        if not 0.0 <= top1 <= 1.0 or not 0.0 <= top10 <= 1.0:
            raise ValueError(
                f'Final-test accuracies must be in [0, 1], got top1={top1}, '
                f'top10={top10}'
            )

        result.update(
            {
                'metrics_status': 'ok',
                'metrics_error': '',
                'best_epoch': int(best_epoch_value),
                'loss_final_test': loss,
                'top1s_final_test': top1,
                'top10s_final_test': top10,
                'top1_final_test_pct': 100.0 * top1,
                'top10_final_test_pct': 100.0 * top10,
            }
        )
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        result['metrics_status'] = 'error'
        result['metrics_error'] = f'{type(error).__name__}: {error}'
    return result


def _parse_window_tag(window_tag: str) -> tuple[str, str]:
    match = re.fullmatch(r'(.+?)(cpu|gpu)', window_tag)
    if match is None:
        return window_tag, ''
    return match.group(1), match.group(2)


def _ablation_name(run_group: str, config: dict[str, Any]) -> str:
    match = re.match(r'^ablate-(.+?)-\d+conv(?:-|$)', run_group)
    if match is not None:
        return match.group(1)

    changes = []
    if config.get('use_spatial_attention') != '3D':
        changes.append(f'attention-{config.get("use_spatial_attention") or "none"}')
    if not config.get('use_unmixing_layer', True):
        changes.append('no-unmix')
    if not config.get('use_subject_layer', True):
        changes.append('no-subject')
    if config.get('temporal_filter_kernel_size', 15) != 15:
        changes.append(f'tf-kernel-{config.get("temporal_filter_kernel_size")}')
    return '+'.join(changes)


def classify_run(
    config: dict[str, Any],
    run_group: str,
) -> tuple[str, str, str]:
    """Return (family, subfamily, ablation) for filtering paper runs."""
    window_tag = str(config.get('window_tag') or '')
    feature_reduction = str(config.get('feature_reduction') or 'none')
    time_reduction = str(config.get('time_reduction') or 'none')
    temporal_depth = config.get('n_temporal_module_blocks')

    if window_tag:
        _, compute_variant = _parse_window_tag(window_tag)
        subfamily = f'duration_{compute_variant}' if compute_variant else 'duration'
        return 'segment_duration', subfamily, ''
    if feature_reduction != 'none':
        return 'feature_reduction', f'feature_{feature_reduction}', ''
    if time_reduction != 'none':
        return 'time_reduction', f'time_{time_reduction}', ''

    ablation = _ablation_name(run_group, config)
    if run_group.startswith('ablate-') or ablation:
        return 'architecture_ablation', ablation or 'other', ablation

    depth_subfamily = f'{temporal_depth}conv'
    if config.get('seed') != 42:
        return 'seed_replication', depth_subfamily, ''
    if re.match(r'^\d+conv-seed42(?:-|$)', run_group):
        return 'branch_depth_sweep', depth_subfamily, ''
    return 'other', depth_subfamily, ''


def _architecture_id(config: dict[str, Any]) -> str:
    architecture = {field: config.get(field) for field in ARCHITECTURE_FIELDS}
    payload = json.dumps(architecture, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]


def _architecture_label(config: dict[str, Any]) -> str:
    attention = config.get('use_spatial_attention') or 'none'
    feature_mode = config.get('feature_reduction') or 'none'
    feature_dim = config.get('feature_reduction_dim')
    time_mode = config.get('time_reduction') or 'none'
    time_dim = config.get('time_reduction_dim')
    feature = feature_mode if feature_dim is None else f'{feature_mode}{feature_dim}'
    time = time_mode if time_dim is None else f'{time_mode}{time_dim}'
    return (
        f'd{config.get("n_temporal_module_blocks")}'
        f'_k{config.get("n_channels_unmix")}'
        f'_attn{attention}'
        f'_unmix{int(bool(config.get("use_unmixing_layer")))}'
        f'_subject{int(bool(config.get("use_subject_layer")))}'
        f'_tfk{config.get("temporal_filter_kernel_size")}'
        f'_feature-{feature}'
        f'_time-{time}'
    )


def _set_counts(
    row: dict[str, Any],
    prefix: str,
    counts: tuple[int, int],
) -> None:
    row[f'{prefix}_trainable_params'] = counts[0]
    row[f'{prefix}_total_params'] = counts[1]


def summarize_run(config_path: Path) -> dict[str, Any]:
    """Build one parameter-count row from a completed run config."""
    config = json.loads(config_path.read_text(encoding='utf-8'))
    run_dir = config_path.parent
    run_group = str(config.get('run_group') or run_dir.parent.name)
    family, subfamily, ablation = classify_run(config, run_group)
    window_tag = str(config.get('window_tag') or '')
    window_duration_tag, window_compute_variant = _parse_window_tag(window_tag)

    row: dict[str, Any] = {
        'status': 'ok',
        'error': '',
        'model_family': family,
        'model_subfamily': subfamily,
        'ablation': ablation,
        'architecture_id': _architecture_id(config),
        'architecture_label': _architecture_label(config),
        'run_group': run_group,
        'run_name': config.get('run_name', ''),
        'run_id': _parse_run_id(run_dir.name),
        'experiment_dir': str(run_dir),
        'config_path': str(config_path),
        'window_duration_tag': window_duration_tag,
        'window_compute_variant': window_compute_variant,
        'audio_precomputed_reduction': (
            config.get('feature_reduction') == 'pca'
            or config.get('time_reduction') == 'pca'
        ),
    }
    row.update({column: config.get(column, '') for column in CONFIG_COLUMNS})
    row.update(_read_final_test_metrics(run_dir))

    device = torch.device('cpu')
    meg_model, _ = load_from_config(
        hyper_params=config,
        checkpoint_path=None,
        device=device,
    )
    audio_adapter = build_audio_model_from_config(
        hyper_params=config,
        device=device,
    )
    criterion = CLIPLoss(clip_temperature=config['clip_temperature'])

    meg_parts = _partition_parameters(
        meg_model,
        {
            'projection_resampler': meg_model.feature_projection,
            'time_reducer': meg_model.time_reducer,
        },
    )
    audio_parts = _partition_parameters(
        audio_adapter,
        {
            'feature_reducer': getattr(audio_adapter, 'feature_reducer', None),
            'time_reducer': getattr(audio_adapter, 'time_reducer', None),
        },
    )

    meg_counts = _count_parameters(meg_model.parameters())
    audio_counts = _count_parameters(audio_adapter.parameters())
    criterion_counts = _count_parameters(criterion.parameters())
    network_counts = (
        meg_counts[0] + audio_counts[0],
        meg_counts[1] + audio_counts[1],
    )

    row['trainable_params'], row['total_params'] = network_counts
    _set_counts(row, 'meg_model', meg_counts)
    _set_counts(row, 'meg_core', meg_parts['other'])
    _set_counts(row, 'meg_projection_resampler', meg_parts['projection_resampler'])
    _set_counts(row, 'meg_time_reducer', meg_parts['time_reducer'])
    _set_counts(row, 'audio_adapter', audio_counts)
    _set_counts(row, 'audio_feature_reducer', audio_parts['feature_reducer'])
    _set_counts(row, 'audio_time_reducer', audio_parts['time_reducer'])
    _set_counts(row, 'audio_other', audio_parts['other'])
    _set_counts(row, 'criterion', criterion_counts)
    row['optimizer_trainable_params'] = network_counts[0] + criterion_counts[0]
    return row


def collect_rows(experiments_root: Path) -> list[dict[str, Any]]:
    config_paths = sorted(experiments_root.glob('*/*/config.json'))
    rows = []
    for config_path in tqdm(config_paths, desc='Counting model parameters'):
        try:
            rows.append(summarize_run(config_path))
        except Exception as error:
            error_row = {
                'status': 'error',
                'error': f'{type(error).__name__}: {error}',
                'run_group': config_path.parent.parent.name,
                'run_name': config_path.parent.name,
                'run_id': _parse_run_id(config_path.parent.name),
                'experiment_dir': str(config_path.parent),
                'config_path': str(config_path),
            }
            error_row.update(_read_final_test_metrics(config_path.parent))
            rows.append(error_row)
    return rows


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(
            {column: row.get(column, '') for column in CSV_COLUMNS} for row in rows
        )


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Rebuild every configured MEG/audio model on CPU and export parameter '
            'counts, final test metrics, and filterable architecture metadata.'
        )
    )
    parser.add_argument(
        '--experiments-root',
        default=os.path.join(OUTPUTS_DIR, 'experiments'),
        help='Root containing <run-group>/<run>/config.json directories.',
    )
    parser.add_argument(
        '--out',
        default=os.path.join(OUTPUTS_DIR, 'model_parameter_counts.csv'),
        help='Output CSV path.',
    )
    parser.add_argument(
        '--strict',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Exit with an error after writing CSV if any parameter count or '
            'final-test metric row is unavailable.'
        ),
    )
    return parser.parse_args(argv)


def main_cli(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    rows = collect_rows(Path(args.experiments_root))
    out_path = Path(args.out)
    write_csv(rows, out_path)

    parameter_errors = [row for row in rows if row['status'] != 'ok']
    metric_errors = [row for row in rows if row.get('metrics_status') != 'ok']
    print(
        f'Wrote {len(rows)} rows to {out_path} '
        f'({len(parameter_errors)} parameter errors, '
        f'{len(metric_errors)} metric errors)'
    )
    if args.strict and (parameter_errors or metric_errors):
        raise RuntimeError(
            'Incomplete export: '
            f'{len(parameter_errors)} parameter error(s), '
            f'{len(metric_errors)} metric error(s); see status/error columns.'
        )


if __name__ == '__main__':
    main_cli()
