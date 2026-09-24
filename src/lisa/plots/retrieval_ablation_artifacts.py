"""Generate ablation/reduction artifacts for retrieval experiments."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from string import Formatter
from typing import Any, Sequence

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lisa.utils.constants import EXPERIMENTS_DIR
from lisa.utils.validators import validate_run_group


FAMILY_ORDER = {'baseline': -1, 'window': 0, 'ablation': 1, 'feature': 2, 'time': 3}

CONFIG_ALLOWED_DIFFERENCES = frozenset(
    {
        'checkpoint',
        'device',
        'dirprocess',
        'dl_n_workers',
        'experiments_root',
        'image_formats',
        'logger',
        'logging_project',
        'log_images_to_server',
        'max_branches_to_plot',
        'plot_filter_graphs',
        'meg_files_dir',
        'meg_format',
        'run_group',
        'run_name',
        'save_test_top_k',
        'torch_deterministic',
        'upload_metrics_csv',
    }
)

WINDOW_ALLOWED_DIFFS = frozenset({'window_tag'})
FEATURE_ALLOWED_DIFFS = frozenset(
    {'feature_reduction', 'feature_reduction_dim', 'n_features'}
)
TIME_ALLOWED_DIFFS = frozenset(
    {
        'audio_time_reduction_input_length',
        'time_reduction',
        'time_reduction_dim',
        'time_reduction_hidden_dim',
        'time_reduction_input_length',
        'time_reduction_num_heads',
    }
)
ABLATION_ALLOWED_DIFFS = frozenset(
    {
        'n_channels_attention',
        'temporal_filter_kernel_size',
        'use_spatial_attention',
        'use_subject_layer',
        'use_unmixing_layer',
    }
)

FAMILY_ALLOWED_DIFFS = {
    'window': WINDOW_ALLOWED_DIFFS,
    'ablation': ABLATION_ALLOWED_DIFFS,
    'feature': FEATURE_ALLOWED_DIFFS,
    'time': TIME_ALLOWED_DIFFS,
}


@dataclass(frozen=True)
class RunSpec:
    family: str
    run_group: str
    order: int
    label: str | None = None
    method: str | None = None
    dim: int | None = None
    plot_label: str | None = None


def is_none_like(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.lower() in {'none', ''})


def _config_value_for_comparison(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            sorted((k, _config_value_for_comparison(v)) for k, v in value.items())
        )
    if isinstance(value, list):
        return tuple(_config_value_for_comparison(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_config_value_for_comparison(v) for v in value)
    if hasattr(value, 'item'):
        return value.item()
    return value


def config_values_equal(actual: Any, expected: Any) -> bool:
    return _config_value_for_comparison(actual) == _config_value_for_comparison(
        expected
    )


def format_config_value(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def method_label(raw_method: Any, mapping: dict[str, str]) -> str:
    if raw_method is None:
        return ''
    raw = str(raw_method)
    return mapping.get(raw, raw.replace('_', ' ').title())


def seed_template_fields(template: str) -> list[str]:
    """Return placeholder names; only exactly ``{seed}`` is allowed."""
    try:
        parsed = list(Formatter().parse(template))
    except ValueError as exc:
        raise ValueError(f'Invalid run-group template {template!r}: {exc}') from exc
    fields = []
    for _literal, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if field_name != 'seed' or format_spec or conversion:
            raise ValueError(
                f'Run-group template {template!r} may only use the {{seed}} '
                'placeholder.'
            )
        fields.append(field_name)
    return fields


def expand_run_group_template(template: str, seed: int) -> str:
    seed_template_fields(template)
    run_group = template.format(seed=seed)
    validate_run_group(run_group)
    return run_group


def expand_label_overrides(
    overrides: dict[str, str],
    seeds: list[int] | None,
) -> dict[str, str]:
    if seeds is None:
        return overrides
    expanded: dict[str, str] = {}
    for key, value in overrides.items():
        if seed_template_fields(key):
            for seed in seeds:
                expanded[expand_run_group_template(key, seed)] = value
        else:
            expanded[key] = value
    return expanded


def parse_key_value_overrides(
    entries: list[str],
    arg_name: str,
    *,
    allow_seed_template: bool = False,
) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for entry in entries:
        if '=' not in entry:
            raise argparse.ArgumentTypeError(
                f'{arg_name} expects RUN_GROUP=VALUE entries, got {entry!r}.'
            )
        run_group, value = entry.split('=', 1)
        run_group = run_group.strip()
        if not run_group:
            raise argparse.ArgumentTypeError(
                f'{arg_name} has an empty run group in {entry!r}.'
            )
        if allow_seed_template:
            try:
                fields = seed_template_fields(run_group)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(str(exc)) from exc
            if not fields:
                validate_run_group(run_group)
        else:
            validate_run_group(run_group)
        overrides[run_group] = value.strip()
    return overrides


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate ablation/reduction artifacts from completed retrieval runs.'
    )
    parser.add_argument(
        '--experiments-root',
        type=Path,
        default=Path(EXPERIMENTS_DIR),
        help='Root directory with experiment run groups.',
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=None,
        help=(
            'Directory where tables and figures will be written. Default: '
            'outputs/plots/retrieval_reductions_ablations/<baseline>-<K>branches.'
        ),
    )
    parser.add_argument(
        '--baseline-run-group',
        type=str,
        required=True,
        help='Baseline run group.',
    )
    parser.add_argument(
        '--n-branches',
        type=int,
        required=True,
        help='Expected number of unmix channels/branches for all selected runs.',
    )
    parser.add_argument(
        '--window-run-groups',
        nargs='*',
        default=[],
        metavar='GROUP',
        help='Run groups to treat as segment-length/window ablations.',
    )
    parser.add_argument(
        '--ablation-run-groups',
        nargs='*',
        default=[],
        metavar='GROUP',
        help='Run groups to treat as architecture/config ablations.',
    )
    parser.add_argument(
        '--feature-run-groups',
        nargs='*',
        default=[],
        metavar='GROUP',
        help='Run groups to treat as feature reduction runs.',
    )
    parser.add_argument(
        '--time-run-groups',
        nargs='*',
        default=[],
        metavar='GROUP',
        help='Run groups to treat as time reduction runs.',
    )
    parser.add_argument(
        '--label',
        action='append',
        default=[],
        metavar='GROUP=LABEL',
        help='Override table/display label for a run group. Can be repeated.',
    )
    parser.add_argument(
        '--plot-label',
        action='append',
        default=[],
        metavar='GROUP=LABEL',
        help='Override compact plot label for a run group. Can be repeated.',
    )
    parser.add_argument(
        '--audit-training-artifacts',
        action='store_true',
        help='Also require checkpoints, filters, and NPZ final-test dump.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Allow writing into a non-empty output directory.',
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=None,
        metavar='SEED',
        help=(
            'If given, expand {seed} in run-group templates and pair each '
            'condition with the same-seed full model. Omitted: current '
            'single-run lookup.'
        ),
    )
    args = parser.parse_args(argv)

    all_run_groups = (
        [args.baseline_run_group]
        + args.window_run_groups
        + args.ablation_run_groups
        + args.feature_run_groups
        + args.time_run_groups
    )
    if args.seeds is not None:
        if len(set(args.seeds)) != len(args.seeds):
            parser.error('--seeds must be unique.')
        if any(seed < 0 for seed in args.seeds):
            parser.error('--seeds must be non-negative.')
        for run_group in all_run_groups:
            try:
                for seed in args.seeds:
                    expand_run_group_template(run_group, seed)
            except ValueError as exc:
                parser.error(str(exc))
    else:
        for run_group in all_run_groups:
            validate_run_group(run_group)

    if not (
        args.window_run_groups
        or args.ablation_run_groups
        or args.feature_run_groups
        or args.time_run_groups
    ):
        parser.error(
            'At least one of --window-run-groups, --ablation-run-groups, '
            '--feature-run-groups, or --time-run-groups must be provided.'
        )
    if args.n_branches <= 0:
        parser.error('--n-branches must be positive.')

    try:
        allow_seed_template = args.seeds is not None
        args.label_overrides = parse_key_value_overrides(
            args.label,
            '--label',
            allow_seed_template=allow_seed_template,
        )
        args.plot_label_overrides = parse_key_value_overrides(
            args.plot_label,
            '--plot-label',
            allow_seed_template=allow_seed_template,
        )
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    duplicate_run_groups = sorted(
        {run_group for run_group in all_run_groups if all_run_groups.count(run_group) > 1}
    )
    if duplicate_run_groups:
        parser.error(
            'Run groups must be unique across baseline and family arguments; '
            f'duplicates: {", ".join(duplicate_run_groups)}.'
        )

    selected_run_groups = set(all_run_groups)
    unknown_label_groups = sorted(
        (set(args.label_overrides) | set(args.plot_label_overrides)) - selected_run_groups
    )
    if unknown_label_groups:
        parser.error(
            'Label overrides refer to run groups that were not selected: '
            f'{", ".join(unknown_label_groups)}.'
        )
    return args


def build_run_specs(args: argparse.Namespace) -> list[RunSpec]:
    specs = []
    order = 0
    for family, run_groups in [
        ('window', args.window_run_groups),
        ('ablation', args.ablation_run_groups),
        ('feature', args.feature_run_groups),
        ('time', args.time_run_groups),
    ]:
        for run_group in run_groups:
            specs.append(RunSpec(family=family, run_group=run_group, order=order))
            order += 1
    return specs


def selected_families(specs: list[RunSpec]) -> set[str]:
    return {spec.family for spec in specs}


def default_out_dir(args: argparse.Namespace) -> Path:
    baseline = args.baseline_run_group.replace('{seed}', 'multiseed')
    return (
        Path('outputs/plots/retrieval_reductions_ablations')
        / f'{baseline}-{args.n_branches}branches'
    )


def prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise RuntimeError(
            f'Output directory already exists and is not empty: {out_dir}. '
            'Choose a new --out-dir or pass --overwrite.'
        )
    out_dir.mkdir(parents=True, exist_ok=True)


def read_config(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding='utf-8'))


def find_run_dir_and_config(
    experiments_root: Path,
    run_group: str,
    n_branches: int,
    seed: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    group_dir = experiments_root / run_group
    if not group_dir.is_dir():
        raise FileNotFoundError(f'Missing run group directory: {group_dir}')

    branch_token = f'{n_branches}branches'
    matches = []
    for config_path in sorted(group_dir.glob('*/config.json')):
        config = read_config(config_path)
        if config.get('n_channels_unmix') != n_branches:
            continue
        if seed is not None:
            if 'seed' not in config or config['seed'] is None:
                continue
            try:
                config_seed = int(config['seed'])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f'Invalid seed in {config_path}: {config["seed"]!r}'
                ) from exc
            if config_seed != int(seed):
                continue
        run_name = str(config.get('run_name') or '')
        run_dir = config_path.parent
        if branch_token not in run_dir.name and branch_token not in run_name:
            continue
        matches.append((run_dir, config))

    seed_msg = f', seed={seed}' if seed is not None else ''
    if not matches:
        raise FileNotFoundError(
            f'No run with n_channels_unmix={n_branches} and {branch_token!r} '
            f'in run directory name or config run_name{seed_msg} found in '
            f'group: {group_dir}'
        )
    if len(matches) > 1:
        names = ', '.join(path.name for path, _ in matches)
        raise RuntimeError(
            f'Expected one {n_branches}-branch{seed_msg} run directory in '
            f'{group_dir}, found: {names}'
        )
    return matches[0]


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f'Missing required file: {path}')
    if path.stat().st_size == 0:
        raise RuntimeError(f'Required file is empty: {path}')


def require_csv_columns(path: Path, columns: set[str]) -> None:
    actual = set(pd.read_csv(path, nrows=0).columns)
    missing = columns - actual
    if missing:
        raise RuntimeError(f'Missing required columns in {path}: {sorted(missing)}')


def semantic_config_diffs(
    baseline_config: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, tuple[Any, Any]]:
    diffs = {}
    for key in sorted(set(baseline_config) | set(run_config)):
        if key in CONFIG_ALLOWED_DIFFERENCES:
            continue
        baseline_value = baseline_config.get(key)
        run_value = run_config.get(key)
        if (
            key == 'window_tag'
            and is_none_like(baseline_value)
            and is_none_like(run_value)
        ):
            continue
        if not config_values_equal(baseline_value, run_value):
            diffs[key] = (baseline_value, run_value)
    return diffs


def validate_family_diffs(
    *,
    family: str,
    run_group: str,
    diffs: dict[str, tuple[Any, Any]],
    run_config: dict[str, Any],
) -> None:
    allowed = FAMILY_ALLOWED_DIFFS[family]
    unexpected = sorted(set(diffs) - allowed)
    if unexpected:
        raise RuntimeError(
            f'Run {run_group} was passed as {family}, but also differs in '
            f'{", ".join(unexpected)}. Allowed {family} diffs: '
            f'{", ".join(sorted(allowed))}.'
        )

    if family == 'window':
        if is_none_like(run_config.get('window_tag')):
            raise RuntimeError(
                f'Run {run_group} was passed as window, but window_tag is empty.'
            )
        if not (set(diffs) & WINDOW_ALLOWED_DIFFS):
            raise RuntimeError(
                f'Run {run_group} was passed as window, but window_tag does not '
                'differ from the baseline.'
            )
    elif family == 'feature':
        if is_none_like(run_config.get('feature_reduction')):
            raise RuntimeError(
                f'Run {run_group} was passed as feature, but feature_reduction '
                'is none/null.'
            )
        if run_config.get('feature_reduction_dim') is None:
            raise RuntimeError(
                f'Run {run_group} was passed as feature, but '
                'feature_reduction_dim is missing.'
            )
    elif family == 'time':
        if is_none_like(run_config.get('time_reduction')):
            raise RuntimeError(
                f'Run {run_group} was passed as time, but time_reduction is none/null.'
            )
    elif family == 'ablation' and not (set(diffs) & ABLATION_ALLOWED_DIFFS):
        raise RuntimeError(
            f'Run {run_group} was passed as ablation, but none of the allowed '
            f'ablation keys differ: {", ".join(sorted(ABLATION_ALLOWED_DIFFS))}.'
        )


def validate_training_artifacts(run_dir: Path) -> None:
    for filename in [
        'lisa_checkpoint.pt',
        'criterion_best.pt',
        'final_test_per_window.npz',
    ]:
        require_file(run_dir / filename)

    if not list(run_dir.glob('filters_*.npz')):
        raise FileNotFoundError(f'Missing filters_*.npz in {run_dir}')


def compute_final_metrics(final_test_path: Path) -> tuple[float, float, int]:
    df = pd.read_csv(final_test_path, usecols=['rank_of_true'])
    ranks = df['rank_of_true'].dropna().astype(int)
    if ranks.empty:
        raise RuntimeError(f'No scored rows found in {final_test_path}')

    top1 = 100.0 * (ranks == 0).mean()
    top10 = 100.0 * ((ranks >= 0) & (ranks < 10)).mean()
    return float(top1), float(top10), int(len(ranks))


def read_candidate_count(run_dir: Path) -> int | None:
    """Read retrieval database size saved by Trainer, if available."""
    npz_path = run_dir / 'final_test_per_window.npz'
    if not npz_path.is_file() or npz_path.stat().st_size == 0:
        return None
    with np.load(npz_path) as data:
        if 'candidate_ids' not in data:
            return None
        return int(np.asarray(data['candidate_ids']).shape[0])


def read_epoch_info(metrics_path: Path) -> tuple[int | None, int | None, int | None]:
    df = pd.read_csv(metrics_path)
    last_epoch = None
    early_stop_epoch = None
    best_epoch = None

    if 'epoch' in df.columns:
        epochs = df['epoch'].dropna()
        if not epochs.empty:
            last_epoch = int(epochs.iloc[-1])

    if 'early_stop_epoch' in df.columns:
        early_stops = df['early_stop_epoch'].dropna()
        if not early_stops.empty:
            early_stop_epoch = int(early_stops.iloc[-1])

    if {'epoch', 'top1s_final_test'}.issubset(df.columns):
        final_rows = df[df['top1s_final_test'].notna()]
        final_epochs = final_rows['epoch'].dropna()
        if not final_epochs.empty:
            best_epoch = int(final_epochs.iloc[-1])

    return last_epoch, early_stop_epoch, best_epoch


def infer_feature_labels(config: dict[str, Any]) -> tuple[str, int, str, str]:
    dim = int(config['feature_reduction_dim'])
    method = method_label(
        config.get('feature_reduction'),
        {'linear': 'LinearDR', 'pca': 'PCA'},
    )
    label = f'{method} {dim}'
    return label, dim, method, label


def infer_time_labels(config: dict[str, Any]) -> tuple[str, int | None, str, str]:
    raw_method = config.get('time_reduction')
    method = method_label(
        raw_method,
        {
            'adaptive_avg': 'Adaptive Avg',
            'adaptive_max': 'Adaptive Max',
            'attn': 'Attention',
            'gated_attn': 'Gated Attention',
            'linear': 'LinearDR',
            'max': 'Max Pool',
            'mean': 'Mean Pool',
            'pca': 'PCA',
            'query_attn': 'Query Attention',
        },
    )
    dim = config.get('time_reduction_dim')
    hidden_dim = config.get('time_reduction_hidden_dim')
    heads = config.get('time_reduction_num_heads')
    dim = int(dim) if dim is not None else None
    hidden_dim = int(hidden_dim) if hidden_dim is not None else None

    if raw_method in {'pca', 'linear'} and dim is not None:
        label = f'Time {method} ({dim})'
        plot_label = f'{method.replace("DR", "")}{dim}'
    elif raw_method in {'adaptive_avg', 'adaptive_max'} and dim is not None:
        label = f'{method} Pool (k={dim})'
        prefix = 'Avg' if raw_method == 'adaptive_avg' else 'Max'
        plot_label = f'{prefix}{dim}'
    elif raw_method == 'query_attn' and dim is not None:
        label = f'Query Attention (k={dim}, heads={heads})'
        plot_label = f'QAttn{dim}'
    elif raw_method == 'attn' and hidden_dim is not None:
        label = f'Attention Pool (h={hidden_dim})'
        plot_label = f'Attn{hidden_dim}'
    elif raw_method == 'gated_attn' and hidden_dim is not None:
        label = f'Gated Attention Pool (h={hidden_dim})'
        plot_label = f'Gated{hidden_dim}'
    elif raw_method == 'mean':
        label = 'Mean Pool'
        plot_label = 'Mean'
    elif raw_method == 'max':
        label = 'Max Pool'
        plot_label = 'Max'
    elif dim is not None:
        label = f'{method} ({dim})'
        plot_label = f'{method}{dim}'
    else:
        label = method
        plot_label = method
    return label, dim, method, plot_label


def spatial_attention_label(config: dict[str, Any]) -> str:
    value = config.get('use_spatial_attention')
    if is_none_like(value):
        return 'No attention'
    if str(value) in {'2D', '3D'}:
        return f'{value} attention'
    return f'{value} attention'


def infer_ablation_labels(config: dict[str, Any]) -> tuple[str, None, str, str]:
    parts = [spatial_attention_label(config)]
    if config.get('use_unmixing_layer', True):
        parts.append('Unmix')
    if config.get('use_subject_layer', True):
        parts.append('Subject layer')

    if config.get('temporal_filter_kernel_size') == 1:
        parts.append('No temporal filter')
    else:
        parts.append('Temporal filter')

    label = ' + '.join(parts)
    return label, None, 'Ablation', label


def infer_window_labels(config: dict[str, Any]) -> tuple[str, None, str, str]:
    tag = str(config.get('window_tag') or '')
    label = f'Window {tag}'
    return label, None, 'Window length', label


def infer_run_metadata(spec: RunSpec, config: dict[str, Any]) -> RunSpec:
    if spec.family == 'window':
        label, dim, method, plot_label = infer_window_labels(config)
    elif spec.family == 'feature':
        label, dim, method, plot_label = infer_feature_labels(config)
    elif spec.family == 'time':
        label, dim, method, plot_label = infer_time_labels(config)
    elif spec.family == 'ablation':
        label, dim, method, plot_label = infer_ablation_labels(config)
    else:
        label, dim, method, plot_label = 'Full model', None, 'Baseline', 'Full model'

    return RunSpec(
        family=spec.family,
        run_group=spec.run_group,
        order=spec.order,
        label=spec.label or label,
        method=spec.method or method,
        dim=spec.dim if spec.dim is not None else dim,
        plot_label=spec.plot_label or plot_label,
    )


def apply_label_overrides(
    spec: RunSpec,
    label_overrides: dict[str, str],
    plot_label_overrides: dict[str, str],
) -> RunSpec:
    return RunSpec(
        family=spec.family,
        run_group=spec.run_group,
        order=spec.order,
        label=label_overrides.get(spec.run_group, spec.label),
        method=spec.method,
        dim=spec.dim,
        plot_label=plot_label_overrides.get(spec.run_group, spec.plot_label),
    )


def load_result_row(
    experiments_root: Path,
    spec: RunSpec,
    n_branches: int,
    baseline_config: dict[str, Any],
    label_overrides: dict[str, str],
    plot_label_overrides: dict[str, str],
    baseline_top1: float | None = None,
    baseline_top10: float | None = None,
    baseline_n_windows: int | None = None,
    audit_training_artifacts: bool = False,
    seed: int | None = None,
    run_dir: Path | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if run_dir is None or config is None:
        run_dir, config = find_run_dir_and_config(
            experiments_root,
            spec.run_group,
            n_branches,
            seed=seed,
        )
    config_path = run_dir / 'config.json'
    metrics_path = run_dir / 'metrics.csv'
    final_test_path = run_dir / 'final_test_per_window.csv'

    for path in [config_path, metrics_path, final_test_path]:
        require_file(path)
    require_csv_columns(final_test_path, {'rank_of_true', 'session_id', 'subject_id'})

    diffs = semantic_config_diffs(baseline_config, config)
    if spec.family != 'baseline':
        validate_family_diffs(
            family=spec.family,
            run_group=spec.run_group,
            diffs=diffs,
            run_config=config,
        )
    spec = infer_run_metadata(spec, config)
    spec = apply_label_overrides(spec, label_overrides, plot_label_overrides)

    if audit_training_artifacts:
        validate_training_artifacts(run_dir)

    top1, top10, n_windows = compute_final_metrics(final_test_path)
    enforce_n_windows = spec.family != 'window'
    if (
        enforce_n_windows
        and baseline_n_windows is not None
        and n_windows != baseline_n_windows
    ):
        raise RuntimeError(
            f'{final_test_path} has {n_windows} scored rows, '
            f'expected {baseline_n_windows}.'
        )
    n_candidates = read_candidate_count(run_dir)

    last_epoch, early_stop_epoch, best_epoch = read_epoch_info(metrics_path)
    delta_top1 = 0.0 if baseline_top1 is None else top1 - baseline_top1
    delta_top10 = 0.0 if baseline_top10 is None else top10 - baseline_top10
    row_seed = seed if seed is not None else config.get('seed')

    row = {
        'family': spec.family,
        'run_group': spec.run_group,
        'seed': row_seed,
        'label': spec.label,
        'method': spec.method,
        'dim': spec.dim,
        'plot_label': spec.plot_label,
        'order': spec.order,
        'top1': top1,
        'top10': top10,
        'delta_top1': delta_top1,
        'delta_top10': delta_top10,
        'n_windows': n_windows,
        'n_candidates': n_candidates,
        'last_epoch': last_epoch,
        'early_stop_epoch': early_stop_epoch,
        'best_epoch': best_epoch,
        'run_dir': str(run_dir),
    }
    diff_rows = [
        {
            'family': spec.family,
            'run_group': spec.run_group,
            'seed': row_seed,
            'key': key,
            'baseline_value': format_config_value(baseline_value),
            'run_value': format_config_value(run_value),
        }
        for key, (baseline_value, run_value) in diffs.items()
    ]
    return row, diff_rows


def _resolved_run_group(template: str, seed: int | None) -> str:
    if seed is None:
        return template
    return expand_run_group_template(template, seed)


def _warn_missing_run(kind: str, template: str, seed: int, exc: FileNotFoundError) -> None:
    warnings.warn(
        f'Missing {kind} for template {template!r} seed={seed} ({exc}). '
        'Skipping this seed.',
        UserWarning,
        stacklevel=3,
    )


def load_results(
    experiments_root: Path,
    baseline_spec: RunSpec,
    specs: list[RunSpec],
    n_branches: int,
    label_overrides: dict[str, str],
    plot_label_overrides: dict[str, str],
    audit_training_artifacts: bool,
    seeds: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    requested_seeds: list[int | None] = list(seeds) if seeds is not None else [None]
    baseline_by_seed: dict[int | None, tuple[dict[str, Any], dict[str, Any]]] = {}

    for seed in requested_seeds:
        run_group = _resolved_run_group(baseline_spec.run_group, seed)
        spec = replace(baseline_spec, run_group=run_group)
        try:
            run_dir, config = find_run_dir_and_config(
                experiments_root,
                run_group,
                n_branches,
                seed=seed,
            )
        except FileNotFoundError as exc:
            if seed is None:
                raise
            _warn_missing_run('baseline', baseline_spec.run_group, seed, exc)
            continue
        baseline_row, _ = load_result_row(
            experiments_root=experiments_root,
            spec=spec,
            n_branches=n_branches,
            baseline_config=config,
            label_overrides=label_overrides,
            plot_label_overrides=plot_label_overrides,
            audit_training_artifacts=audit_training_artifacts,
            seed=seed,
            run_dir=run_dir,
            config=config,
        )
        baseline_row['run_dir'] = str(run_dir)
        baseline_by_seed[seed] = (baseline_row, config)

    if not baseline_by_seed:
        raise RuntimeError(
            f'No valid baseline runs found for template {baseline_spec.run_group!r} '
            f'and seeds {list(seeds) if seeds is not None else []}.'
        )

    rows = [row for row, _ in baseline_by_seed.values()]
    diff_rows: list[dict[str, str]] = []
    for spec in specs:
        n_loaded = 0
        for seed, (baseline_row, baseline_config) in baseline_by_seed.items():
            run_group = _resolved_run_group(spec.run_group, seed)
            resolved_spec = replace(spec, run_group=run_group)
            try:
                run_dir, config = find_run_dir_and_config(
                    experiments_root,
                    run_group,
                    n_branches,
                    seed=seed,
                )
            except FileNotFoundError as exc:
                if seed is None:
                    raise
                _warn_missing_run('run', spec.run_group, seed, exc)
                continue
            row, run_diff_rows = load_result_row(
                experiments_root=experiments_root,
                spec=resolved_spec,
                n_branches=n_branches,
                baseline_config=baseline_config,
                label_overrides=label_overrides,
                plot_label_overrides=plot_label_overrides,
                baseline_top1=baseline_row['top1'],
                baseline_top10=baseline_row['top10'],
                baseline_n_windows=baseline_row['n_windows'],
                audit_training_artifacts=audit_training_artifacts,
                seed=seed,
                run_dir=run_dir,
                config=config,
            )
            rows.append(row)
            diff_rows.extend(run_diff_rows)
            n_loaded += 1
        if n_loaded == 0 and seeds is not None:
            warnings.warn(
                f'Omitting condition {spec.run_group!r}: no valid paired seeds remain.',
                UserWarning,
                stacklevel=2,
            )

    df = pd.DataFrame(rows)
    df['_family_order'] = df['family'].map(FAMILY_ORDER)
    df = df.sort_values(['_family_order', 'order', 'seed'], kind='mergesort')
    df = df.drop(columns=['_family_order'])
    diff_df = pd.DataFrame(
        diff_rows,
        columns=['family', 'run_group', 'seed', 'key', 'baseline_value', 'run_value'],
    )
    return df.reset_index(drop=True), diff_df


def summarize_multiseed_results(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ['family', 'order', 'label', 'method', 'dim']
    summary = (
        df.groupby(group_cols, dropna=False, sort=False)
        .agg(
            n_seeds=('seed', 'nunique'),
            top1_mean=('top1', 'mean'),
            top1_min=('top1', 'min'),
            top1_max=('top1', 'max'),
            top10_mean=('top10', 'mean'),
            top10_min=('top10', 'min'),
            top10_max=('top10', 'max'),
            delta_top1_mean=('delta_top1', 'mean'),
            delta_top1_min=('delta_top1', 'min'),
            delta_top1_max=('delta_top1', 'max'),
            delta_top10_mean=('delta_top10', 'mean'),
            delta_top10_min=('delta_top10', 'min'),
            delta_top10_max=('delta_top10', 'max'),
        )
        .reset_index()
    )
    summary['_family_order'] = summary['family'].map(FAMILY_ORDER)
    summary = summary.sort_values(['_family_order', 'order'], kind='mergesort')
    return summary.drop(columns=['_family_order']).reset_index(drop=True)


def write_table(df: pd.DataFrame, path_base: Path) -> None:
    columns = [
        'label',
        'top1',
        'top10',
        'delta_top1',
        'delta_top10',
        'n_windows',
        'n_candidates',
    ]
    table = df.sort_values(['top1', 'top10'], ascending=False)[columns].copy()
    table = table.rename(
        columns={
            'label': 'Run',
            'top1': 'Top-1 (%)',
            'top10': 'Top-10 (%)',
            'delta_top1': 'Delta Top-1 (pp)',
            'delta_top10': 'Delta Top-10 (pp)',
            'n_windows': 'Queries',
            'n_candidates': 'Candidates',
        }
    )
    table.to_csv(path_base.with_suffix('.csv'), index=False, float_format='%.2f')
    table.to_latex(
        path_base.with_suffix('.tex'),
        index=False,
        escape=True,
        float_format='%.2f',
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for suffix in ['.pdf', '.png']:
        fig.savefig(out_dir / f'{stem}{suffix}', bbox_inches='tight', dpi=300)
    plt.close(fig)


def prepare_plot_style() -> None:
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.05)
    plt.rcParams.update(
        {
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.dpi': 120,
            'savefig.dpi': 300,
        }
    )


def _minmax_xerr(mean: pd.Series, lo: pd.Series, hi: pd.Series) -> np.ndarray:
    lower = np.asarray(mean - lo, dtype=float)
    upper = np.asarray(hi - mean, dtype=float)
    return np.vstack([np.maximum(lower, 0.0), np.maximum(upper, 0.0)])


def plot_ablation_delta(df: pd.DataFrame, out_dir: Path) -> None:
    subset = df[df['family'] == 'ablation'].copy()
    if subset.empty:
        return
    multi = subset.groupby(['order', 'label']).size().max() > 1
    if multi:
        grouped = subset.groupby(['order', 'label'], as_index=False, sort=False).agg(
            delta_top1=('delta_top1', 'mean'),
            delta_top1_min=('delta_top1', 'min'),
            delta_top1_max=('delta_top1', 'max'),
            delta_top10=('delta_top10', 'mean'),
            delta_top10_min=('delta_top10', 'min'),
            delta_top10_max=('delta_top10', 'max'),
            top1=('top1', 'mean'),
        )
        grouped['mean_delta'] = (grouped['delta_top1'] + grouped['delta_top10']) / 2.0
        grouped = grouped.sort_values(['mean_delta', 'top1'], ascending=False)
        xerr_top1 = _minmax_xerr(
            grouped['delta_top1'],
            grouped['delta_top1_min'],
            grouped['delta_top1_max'],
        )
        xerr_top10 = _minmax_xerr(
            grouped['delta_top10'],
            grouped['delta_top10_min'],
            grouped['delta_top10_max'],
        )
        error_kw = {'ecolor': '0.25', 'capsize': 2.5, 'elinewidth': 1.0}
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        y_positions = range(len(grouped))
        height = 0.36
        ax.barh(
            [y - height / 2 for y in y_positions],
            grouped['delta_top1'],
            height=height,
            xerr=xerr_top1,
            error_kw=error_kw,
            label='Top-1',
            color='#4C78A8',
        )
        ax.barh(
            [y + height / 2 for y in y_positions],
            grouped['delta_top10'],
            height=height,
            xerr=xerr_top10,
            error_kw=error_kw,
            label='Top-10',
            color='#F58518',
        )
        ax.axvline(0, color='0.35', linewidth=1.0, linestyle='--')
        ax.set_yticks(list(y_positions))
        ax.set_yticklabels(grouped['label'])
        ax.invert_yaxis()
        ax.set_xlabel('Change vs full model (percentage points)')
        ax.set_ylabel('')
        ax.legend(frameon=False, loc='upper left', fontsize=12)
        save_figure(fig, out_dir, 'ablation_delta_barplot')
        return

    subset['mean_delta'] = (subset['delta_top1'] + subset['delta_top10']) / 2.0
    subset = subset.sort_values(['mean_delta', 'top1'], ascending=False)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    y_positions = range(len(subset))
    height = 0.36

    ax.barh(
        [y - height / 2 for y in y_positions],
        subset['delta_top1'],
        height=height,
        label='Top-1',
        color='#4C78A8',
    )
    ax.barh(
        [y + height / 2 for y in y_positions],
        subset['delta_top10'],
        height=height,
        label='Top-10',
        color='#F58518',
    )
    ax.axvline(0, color='0.35', linewidth=1.0, linestyle='--')
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(subset['label'])
    ax.invert_yaxis()
    ax.set_xlabel('Change vs full model (percentage points)')
    ax.set_ylabel('')
    ax.legend(frameon=False, loc='upper left', fontsize=12)
    save_figure(fig, out_dir, 'ablation_delta_barplot')


def plot_feature_reduction_curve(df: pd.DataFrame, out_dir: Path) -> None:
    baseline_df = df[df['family'] == 'baseline']
    if baseline_df.empty:
        return
    subset = df[df['family'] == 'feature'].dropna(subset=['dim']).copy()
    if subset.empty:
        return
    subset['dim'] = subset['dim'].astype(int)
    dims = sorted(subset['dim'].unique())
    dim_to_x = {dim: idx for idx, dim in enumerate(dims)}
    methods = list(dict.fromkeys(subset.sort_values('order')['method']))
    colors = dict(zip(methods, sns.color_palette('colorblind', len(methods))))
    multi = (
        len(baseline_df) > 1 or subset.groupby(['method', 'dim']).size().max() > 1
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.7), sharex=True)
    metric_info = [('top1', 'Top-1 (%)'), ('top10', 'Top-10 (%)')]

    if multi:
        agg = subset.groupby(['method', 'dim'], as_index=False, sort=False).agg(
            top1_mean=('top1', 'mean'),
            top1_min=('top1', 'min'),
            top1_max=('top1', 'max'),
            top10_mean=('top10', 'mean'),
            top10_min=('top10', 'min'),
            top10_max=('top10', 'max'),
            order=('order', 'min'),
        )
        for ax, (metric, ylabel) in zip(axes, metric_info, strict=True):
            for method in methods:
                method_df = agg[agg['method'] == method].sort_values('dim').copy()
                method_df['xpos'] = method_df['dim'].map(dim_to_x)
                color = colors[method]
                lo = method_df[f'{metric}_min']
                hi = method_df[f'{metric}_max']
                if (hi > lo).any():
                    ax.fill_between(
                        method_df['xpos'],
                        lo,
                        hi,
                        color=color,
                        alpha=0.18,
                        linewidth=0,
                        label='_nolegend_',
                        zorder=2,
                    )
                ax.plot(
                    method_df['xpos'],
                    method_df[f'{metric}_mean'],
                    marker='o',
                    linewidth=2.0,
                    label=method,
                    color=color,
                    zorder=3,
                )
            base_vals = baseline_df[metric]
            ax.axhline(
                float(base_vals.mean()),
                color='0.35',
                linewidth=1.2,
                linestyle='--',
                label='Full model' if metric == 'top1' else None,
            )
            base_min = float(base_vals.min())
            base_max = float(base_vals.max())
            if base_max > base_min:
                ax.axhspan(base_min, base_max, color='0.35', alpha=0.10, zorder=0)
            ax.set_xlim(-0.4, len(dims) - 0.6)
            ax.set_xticks(range(len(dims)))
            ax.set_xticklabels([str(dim) for dim in dims], rotation=0, ha='center')
            ax.set_xlabel('Embedding dimension')
            ax.set_ylabel(ylabel)
            ax.margins(x=0.04)
    else:
        baseline = baseline_df.iloc[0]
        for ax, (metric, ylabel) in zip(axes, metric_info, strict=True):
            for method in methods:
                method_df = subset[subset['method'] == method].sort_values('dim').copy()
                method_df['xpos'] = method_df['dim'].map(dim_to_x)
                ax.plot(
                    method_df['xpos'],
                    method_df[metric],
                    marker='o',
                    linewidth=2.0,
                    label=method,
                    color=colors[method],
                )
            ax.axhline(
                baseline[metric],
                color='0.35',
                linewidth=1.2,
                linestyle='--',
                label='Full model' if metric == 'top1' else None,
            )
            ax.set_xlim(-0.4, len(dims) - 0.6)
            ax.set_xticks(range(len(dims)))
            ax.set_xticklabels([str(dim) for dim in dims], rotation=0, ha='center')
            ax.set_xlabel('Embedding dimension')
            ax.set_ylabel(ylabel)
            ax.margins(x=0.04)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.subplots_adjust(top=0.9, bottom=0.16, wspace=0.22)
    save_figure(fig, out_dir, 'feature_reduction_curve')


def time_plot_label(row: pd.Series) -> str:
    plot_label = row.get('plot_label')
    if isinstance(plot_label, str) and plot_label:
        return plot_label
    return row['label']


def plot_time_reduction_barplots(df: pd.DataFrame, out_dir: Path) -> None:
    baseline = df[df['family'] == 'baseline'].iloc[0]
    subset = df[df['family'] == 'time'].copy()
    subset['mean_metric'] = (subset['top1'] + subset['top10']) / 2.0
    subset = subset.sort_values(['mean_metric', 'top1'], ascending=False)
    methods = list(dict.fromkeys(subset['method']))
    palette = dict(zip(methods, sns.color_palette('colorblind', len(methods))))

    subset['plot_label'] = subset.apply(time_plot_label, axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 5.6), sharex=True)
    x_positions = list(range(len(subset)))
    metric_info = [('top1', 'Top-1 (%)'), ('top10', 'Top-10 (%)')]

    for ax, (metric, ylabel) in zip(axes, metric_info, strict=True):
        ax.bar(
            x_positions,
            subset[metric],
            color=[palette[method] for method in subset['method']],
            width=0.78,
        )
        ax.axhline(
            baseline[metric],
            color='0.35',
            linewidth=1.2,
            linestyle='--',
        )
        ax.set_ylabel(ylabel)
        ax.margins(x=0.01)

    axes[0].tick_params(axis='x', labelbottom=False)
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(subset['plot_label'], rotation=45, ha='right')

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker='o',
            color='none',
            markerfacecolor=palette[method],
            markeredgecolor='white',
            markersize=7,
            label=method,
        )
        for method in methods
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc='center left',
        bbox_to_anchor=(0.765, 0.5),
    )
    fig.subplots_adjust(left=0.08, right=0.75, bottom=0.2, hspace=0.18)
    save_figure(fig, out_dir, 'time_reduction_barplots')


def write_selected_tables_and_figures(
    df: pd.DataFrame,
    families: set[str],
    out_dir: Path,
) -> None:
    baseline = df[df['family'] == 'baseline']

    if 'ablation' in families:
        ablation_df = pd.concat(
            [baseline, df[df['family'] == 'ablation'].sort_values('order')],
            ignore_index=True,
        )
        write_table(ablation_df, out_dir / 'ablation_table')
        plot_ablation_delta(df, out_dir)

    if 'window' in families:
        window_df = pd.concat(
            [baseline, df[df['family'] == 'window'].sort_values('order')],
            ignore_index=True,
        )
        write_table(window_df, out_dir / 'window_ablation_table')

    if 'feature' in families:
        feature_df = pd.concat(
            [baseline, df[df['family'] == 'feature'].sort_values('order')],
            ignore_index=True,
        )
        write_table(feature_df, out_dir / 'feature_reduction_table')
        plot_feature_reduction_curve(df, out_dir)

    if 'time' in families:
        time_df = pd.concat(
            [baseline, df[df['family'] == 'time'].sort_values('order')],
            ignore_index=True,
        )
        write_table(time_df, out_dir / 'time_reduction_table')
        plot_time_reduction_barplots(df, out_dir)


def print_summary(df: pd.DataFrame, families: set[str], out_dir: Path) -> None:
    selected_df = df[df['family'].isin(families)]
    counts = selected_df.groupby('family').size().to_dict()
    baseline = df[df['family'] == 'baseline'].iloc[0]
    print(
        'Baseline '
        f'Top1={baseline["top1"]:.2f}, Top10={baseline["top10"]:.2f}, '
        f'n_windows={int(baseline["n_windows"])}'
    )
    print(
        'Selected runs: '
        + ', '.join(f'{family}={counts.get(family, 0)}' for family in sorted(families))
    )
    print(f'Wrote artifacts to {out_dir}')


def main() -> None:
    args = parse_args()
    specs = build_run_specs(args)
    families = selected_families(specs)
    baseline_spec = RunSpec(
        family='baseline',
        run_group=args.baseline_run_group,
        order=-1,
        label='Full model',
        method='Baseline',
        plot_label='Full model',
    )
    out_dir = args.out_dir or default_out_dir(args)
    prepare_out_dir(out_dir, args.overwrite)

    prepare_plot_style()
    label_overrides = expand_label_overrides(args.label_overrides, args.seeds)
    plot_label_overrides = expand_label_overrides(
        args.plot_label_overrides,
        args.seeds,
    )
    df, diff_df = load_results(
        experiments_root=args.experiments_root,
        baseline_spec=baseline_spec,
        specs=specs,
        n_branches=args.n_branches,
        label_overrides=label_overrides,
        plot_label_overrides=plot_label_overrides,
        audit_training_artifacts=args.audit_training_artifacts,
        seeds=args.seeds,
    )

    output_columns = [
        'family',
        'run_group',
        'seed',
        'label',
        'method',
        'dim',
        'plot_label',
        'top1',
        'top10',
        'delta_top1',
        'delta_top10',
        'n_windows',
        'n_candidates',
        'best_epoch',
        'last_epoch',
        'early_stop_epoch',
        'run_dir',
    ]
    df[output_columns].to_csv(
        out_dir / 'all_results.csv',
        index=False,
        float_format='%.6f',
    )
    diff_df.to_csv(out_dir / 'config_diffs.csv', index=False)
    if args.seeds is not None:
        summarize_multiseed_results(df).to_csv(
            out_dir / 'summary_results.csv',
            index=False,
            float_format='%.6f',
        )
    write_selected_tables_and_figures(df, families, out_dir)
    print_summary(df, families, out_dir)


if __name__ == '__main__':
    main()
