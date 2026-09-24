"""Story, session, and seed reproducibility of physical-unit interpretations.

Story and session comparisons pair the same ``(participant, branch)`` item.

Seed comparisons do not assume branch-index identity. The primary seed
analysis computes six independent branch-similarity matrices per seed pair
(learned spatial filters, learned temporal filters, spatial Haufe patterns,
temporal Haufe patterns, temporal log-spectra, and source-space magnitude maps)
and a Hungarian one-to-one assignment for each view. No weighted composite of
views is used as the matching. Matched-pair stability is reported separately for
spatial Haufe patterns, temporal patterns, spectra, and source-space
magnitudes. Subspace (principal-angle) scores remain secondary diagnostics.

Optional ``--haufe-geometry`` reports sensor-space roughness; smoothness is not
neuronal validity. Cluster medoids stay the displayed representatives.

Run as ``python3 -m lisa.plots.analyze_interpretation_reproducibility``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment

from lisa.plots.branch_interpretation import (
    compute_item_spatial_roughness,
    compute_plot_compatible_similarities,
    correlation_similarity,
    load_sensor_xy,
    validate_curated_assignments,
)
from lisa.plots.combined_cluster_branch_interpretations import resolve_output_paths
from lisa.plots.plot_branch_interpretations import (
    build_curated_cluster_specs,
    build_medoid_similarity,
    choose_medoid_index,
    load_cached_item_table,
    resolve_item_stats_npz,
    resolve_medoid_config,
    validate_clustering_metadata,
)
from lisa.utils.constants import (
    BRANCH_INTERPRETATION_ASSIGNMENTS_DIR,
    EXPERIMENTS_DIR,
    N_MEG_CHANNELS,
    N_SUBJECTS,
    PLOTS_DIR,
    SUB_SES_COMBOS,
)
from lisa.utils.validators import validate_run_group
from lisa.utils.numeric import assert_finite as _assert_finite

PHYSICAL_SENSOR_UNITS = 'physical'
EXPECTED_REPEATED_SESSION_SUBJECTS = 22
SMOOTHNESS_NOT_NEURONAL_VALIDITY = (
    'Spatial roughness/smoothness is a sensor-topology statistic of the '
    'physical-unit Haufe pattern. It is not evidence of a neuronal source.'
)
SUBSPACE_RANK_TOLERANCE = (
    's_max * max(n_rows, n_cols) * eps(dtype); NumPy matrix_rank default '
    'tolerance.'
)
SUBSPACE_RANK_POLICY = (
    'Secondary subspace diagnostics use the numerical row-space rank of each '
    'view: the number of singular values strictly greater than '
    's_max * max(n_rows, n_cols) * eps(dtype) (NumPy matrix_rank default '
    'tolerance). The comparison uses '
    'the smaller of the two numerical ranks. Near-null directions are excluded '
    'because they are an arbitrary orthonormal completion, not invariant under '
    'invertible left (branch) mixing, and not comparable across seeds. '
    '--subspace-rank, if given, overrides that default for both views and must '
    'fit each matrix. When T <= K, any full-column-rank K x T matrix has row '
    'space equal to all of R^T, so Grassmann overlap is identically 1 and is '
    'not a stability diagnostic; seed_summaries.csv does not report '
    'temporal_subspace_overlap_rms as a stability score in that regime. '
    'These subspace scores are not the primary seed result. Orthogonal '
    'Procrustes aligned Pearson remains supporting, not a stability verdict. '
    'Differing T is rejected, not interpolated.'
)
TEMPORAL_ROW_SPACE_DEGENERACY = (
    'NOT A SEED-STABILITY SCORE. When T <= K, each temporal pattern matrix is '
    'K x T with T <= K. Any full-column-rank such matrix has row space equal '
    'to all of R^T, so subspace overlap is identically 1 and principal angles '
    'are 0 for any two full-column-rank K x T matrices, including unrelated '
    'ones. seed_summaries.csv therefore does not aggregate '
    'temporal_subspace_overlap_rms as a stability score in this regime and '
    'does not report a mean for it. Temporal row-space overlap is not the '
    'primary seed analysis. Orthogonal Procrustes aligned Pearson remains '
    'a supporting number, not a stability verdict: when T <= K a K x K '
    'orthogonal rotation fitted in a T-dimensional ambient space has enough '
    'degrees of freedom that this quantity has a high chance level, so it '
    'must not be read as evidence of reproduction without a null distribution. '
    'This script does not compute one. Per-row overlap, principal angles, '
    'numerical ranks, and temporal_fills_ambient remain in '
    'seed_subspace_scores.csv and seed_principal_angles.csv; those columns '
    'record whether a given matrix fills R^T and are not a claim that the '
    'paper temporal matrices have been shown to do so.'
)
FEATURE_KEYS = (
    ('spatial_haufe', 'spatial_patterns'),
    ('temporal_pattern', 'temporal_patterns'),
    ('log_spectrum', 'temporal_spectra_log'),
    ('source_magnitudes', 'source_magnitudes'),
)
SEED_VIEWS = (
    'spatial_filters',
    'temporal_filters',
    'spatial_haufe',
    'temporal_pattern',
    'log_spectrum',
    'source_magnitudes',
)
PATTERN_VIEWS = (
    'spatial_haufe',
    'temporal_pattern',
    'log_spectrum',
    'source_magnitudes',
)
PATTERN_ITEM_KEYS = {
    'spatial_haufe': 'spatial_patterns',
    'temporal_pattern': 'temporal_patterns',
    'log_spectrum': 'temporal_spectra_log',
    'source_magnitudes': 'source_magnitudes',
}
PAPER_N_BRANCHES = 25
PAPER_RUN_GROUP_TEMPLATE = '2conv-seed{seed}-paper'
SIGN_AMBIGUOUS_VIEWS = (
    'spatial_filters',
    'temporal_filters',
    'spatial_haufe',
    'temporal_pattern',
)
SEED_VIEW_MATCHING_CORRELATION = {
    'spatial_filters': 'absolute',
    'temporal_filters': 'absolute',
    'spatial_haufe': 'absolute',
    'temporal_pattern': 'absolute',
    'log_spectrum': 'signed',
    'source_magnitudes': 'signed',
}
SEED_VIEW_SIGN_CONVENTION = {
    'spatial_filters': (
        'Absolute Pearson. Branch sign is not identifiable across independently '
        'trained seeds, so the matching matrix quotients out sign.'
    ),
    'temporal_filters': (
        'Absolute Pearson. Branch sign is not identifiable across independently '
        'trained seeds, so the matching matrix quotients out sign.'
    ),
    'spatial_haufe': (
        'Absolute Pearson. Branch sign is not identifiable across independently '
        'trained seeds, so the matching matrix quotients out sign.'
    ),
    'temporal_pattern': (
        'Absolute Pearson. Branch sign is not identifiable across independently '
        'trained seeds, so the matching matrix quotients out sign.'
    ),
    'log_spectrum': (
        'Signed Pearson, not remapped to the unit interval. A magnitude '
        'spectrum has no branch-sign ambiguity; a strong negative spectral '
        'correlation is a poor match, not a strong one.'
    ),
    'source_magnitudes': (
        'Signed Pearson, not remapped to the unit interval. A source-space '
        'magnitude map has no branch-sign ambiguity; a strong negative '
        'correlation is a poor match, not a strong one.'
    ),
}
SEED_VIEW_SIGNED_DIAGNOSTICS = (
    'For the four sign-ambiguous views, seed_view_similarities.npz also '
    'stores {view}__signed__s{seed_a}__s{seed_b} aggregated signed Pearson '
    'matrices. Matching uses the unsuffixed matrices, which are absolute '
    'for those views. Log-spectrum and source-space magnitude matching use '
    'the unsuffixed matrices, which are signed. Participant-level NPZ '
    'stores only the matching matrices. Story/session comparisons are '
    'unrelated and remain signed same-item Pearson.'
)
SEED_VIEW_AGGREGATION_REDUCER = (
    'Nested arithmetic mean: story observations within session, sessions within '
    'participant, then participants equally weighted. Arithmetic mean is the '
    'equal-weight reducer and is recorded so it can be changed if a different '
    'reducer is chosen. View 1 has no story/session axis and means over '
    'participants only. View 2 is not aggregated.'
)
VIEW_AXIS_NOTES = {
    'spatial_filters': (
        'Learned spatial filters from filters_*.npz key '
        'overall_spatial_filter_weight, shape (N_SUBJECTS, K, N_MEG_CHANNELS). '
        'Has a participant axis and no story/session axis (model weights). '
        'The story->session->participant hierarchy does not apply; the matrix '
        'is already one value per participant. Participants are equally '
        'weighted by arithmetic mean of the per-participant KxK matrices. '
        'Subject index i in the filter tensor is participant id i+1.'
    ),
    'temporal_filters': (
        'Learned temporal filters from filters_*.npz key '
        'temporal_filters_weight, shape (K, kernel_size). No participant axis '
        'and no story/session axis. There is exactly one KxK matrix per seed '
        'pair. This view is not participant-aggregated.'
    ),
    'spatial_haufe': (
        'Spatial Haufe patterns from cached item-stats. Same participant only. '
        'Nested arithmetic mean: stories within session, sessions within '
        'participant, then participants equally weighted.'
    ),
    'temporal_pattern': (
        'Temporal Haufe patterns from cached item-stats. Same participant only. '
        'Nested arithmetic mean: stories within session, sessions within '
        'participant, then participants equally weighted.'
    ),
    'log_spectrum': (
        'Temporal log-spectra from cached item-stats. Same participant only. '
        'Nested arithmetic mean: stories within session, sessions within '
        'participant, then participants equally weighted.'
    ),
    'source_magnitudes': (
        'Source-space magnitude maps from cached item-stats. Same participant '
        'only. Nested arithmetic mean: stories within session, sessions '
        'within participant, then participants equally weighted.'
    ),
}
PRIMARY_SEED_ANALYSIS = (
    'Primary seed result: six independent Hungarian assignments, one per '
    'view. Matching uses absolute Pearson for the four sign-ambiguous vector '
    'views and signed Pearson for temporal log-spectra and source-space '
    'magnitude maps. No 0.5/0.5 or other weighted composite similarity is used '
    'for matching. Matched-pair stability is reported separately for spatial '
    'Haufe, temporal pattern, log-spectrum, and source-space magnitudes. '
    'Subspace scores are secondary.'
)


def repeated_session_subjects() -> tuple[int, ...]:
    sessions: dict[int, set[int]] = {}
    for subject, session in SUB_SES_COMBOS:
        sessions.setdefault(int(subject), set()).add(int(session))
    repeated = tuple(
        sorted(
            subject for subject, seen in sessions.items() if len(seen) > 1
        )
    )
    if len(repeated) != EXPECTED_REPEATED_SESSION_SUBJECTS:
        raise ValueError(
            'Expected '
            f'{EXPECTED_REPEATED_SESSION_SUBJECTS} repeated-session '
            f'participants in SUB_SES_COMBOS, found {len(repeated)}: '
            f'{repeated}.'
        )
    return repeated


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def summarize_values(values: np.ndarray, *, name: str) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    _assert_finite(name, values)
    if values.size == 0:
        raise ValueError(f'{name} has no values to summarize.')
    std: float | None
    if values.size > 1:
        std = float(np.std(values, ddof=1))
    else:
        std = None
    return {
        'n': int(values.size),
        'mean': float(np.mean(values)),
        'std': std,
        'min': float(np.min(values)),
        'q10': float(np.percentile(values, 10.0)),
        'q25': float(np.percentile(values, 25.0)),
        'median': float(np.median(values)),
        'q75': float(np.percentile(values, 75.0)),
        'q90': float(np.percentile(values, 90.0)),
        'max': float(np.max(values)),
    }


def condition_label(seed: int, story_id: int, session: int) -> str:
    return f'seed{seed}_story{story_id}_ses{session}'


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        '--item-stats',
        nargs='+',
        type=Path,
        default=None,
        help='Explicit physical-unit item-statistics NPZ paths, one per condition.',
    )
    parser.add_argument(
        '--labels',
        nargs='+',
        default=None,
        help='Condition labels, one per --item-stats path. Explicit-path mode only.',
    )
    parser.add_argument(
        '--story-ids',
        nargs='+',
        type=int,
        default=None,
        help=(
            'Explicit-path mode: story id of each --item-stats path. '
            'High-level mode: story ids to resolve (Cartesian with --sessions).'
        ),
    )
    parser.add_argument(
        '--sessions',
        nargs='+',
        type=int,
        default=None,
        help=(
            'Explicit-path mode: session of each --item-stats path. '
            'High-level mode: sessions to resolve (Cartesian with --story-ids).'
        ),
    )
    parser.add_argument(
        '--train-seeds',
        nargs='+',
        type=int,
        default=None,
        help='Training seed of each --item-stats path. Explicit-path mode only.',
    )
    parser.add_argument(
        '--experiments-root',
        type=Path,
        default=Path(EXPERIMENTS_DIR),
        help='Experiment root used by the high-level run resolver.',
    )
    parser.add_argument(
        '--interpretations-root',
        type=Path,
        default=Path(BRANCH_INTERPRETATION_ASSIGNMENTS_DIR),
        help='Root of combined_cluster_branch_interpretations outputs.',
    )
    parser.add_argument(
        '--run-group-template',
        default=None,
        help=(
            'High-level run-group template. Use {seed} when each seed has its '
            f'own group. Default when --seeds is given: {PAPER_RUN_GROUP_TEMPLATE}.'
        ),
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=None,
        help='High-level training seeds. Mutually exclusive with --item-stats.',
    )
    parser.add_argument(
        '--n-branches',
        type=int,
        default=PAPER_N_BRANCHES,
        help='Expected n_channels_unmix used when resolving runs.',
    )
    parser.add_argument(
        '--demean-temporal-filters',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Temporal-filter-demeaning suffix used to locate item-stats.',
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path(PLOTS_DIR) / 'analyze_interpretation_reproducibility',
        help='Directory for CSV/JSON outputs.',
    )
    parser.add_argument(
        '--compare',
        nargs='*',
        default=[],
        choices=('stories', 'sessions', 'seeds'),
        help='Pairwise comparisons to run.',
    )
    parser.add_argument(
        '--assignments-csv',
        type=Path,
        default=None,
        help='Clustering assignments CSV for medoid representativeness.',
    )
    parser.add_argument(
        '--cluster-label',
        default=None,
        help='--labels entry whose NPZ matches --assignments-csv.',
    )
    parser.add_argument(
        '--clusters',
        nargs='+',
        default=None,
        metavar='NAME',
        help='Displayed cluster names. Default: all main (C*) clusters.',
    )
    parser.add_argument(
        '--include-rest',
        action='store_true',
        help='Also report rest (R*) clusters.',
    )
    parser.add_argument(
        '--hungarian',
        action='store_true',
        help=(
            'Accepted for compatibility. Per-view Hungarian matching always '
            'runs with --compare seeds and is not a 0.5/0.5 composite.'
        ),
    )
    parser.add_argument(
        '--haufe-geometry',
        action='store_true',
        help='Optional spatial roughness of physical-unit Haufe patterns.',
    )
    parser.add_argument(
        '--sensor-xy',
        type=Path,
        default=None,
        help='Sensor XY NPY for --haufe-geometry. Default: load_sensor_xy().',
    )
    parser.add_argument(
        '--subspace-rank',
        type=int,
        default=None,
        help='Override numerical row-space rank for both seed views.',
    )
    parser.add_argument(
        '--meg-sr',
        type=float,
        default=100.0,
        help='MEG sampling rate used only to reconstruct pattern time axes.',
    )
    parser.add_argument(
        '--rng-seed',
        type=int,
        default=0,
        help='Deterministic RNG seed stored in metadata.',
    )
    args = parser.parse_args(argv)
    explicit = args.item_stats is not None
    high_level = args.seeds is not None or args.run_group_template is not None
    if explicit and high_level:
        parser.error(
            'Explicit --item-stats cannot be mixed with high-level --seeds / '
            '--run-group-template.'
        )
    if explicit and args.train_seeds is None:
        parser.error('Explicit --item-stats requires --train-seeds.')
    if high_level and args.train_seeds is not None:
        parser.error('--train-seeds is explicit-path only.')
    if high_level and args.labels is not None:
        parser.error('--labels is explicit-path only.')
    if not explicit and not high_level:
        parser.error(
            'Pass either --item-stats (explicit paths) or --seeds / '
            '--run-group-template (high-level).'
        )
    if args.n_branches < 1:
        parser.error('--n-branches must be positive.')
    if explicit:
        n_conditions = len(args.item_stats)
        if n_conditions < 1:
            parser.error('At least one --item-stats path is required.')
        if args.labels is None:
            parser.error('Explicit --item-stats requires --labels.')
        if args.story_ids is None or args.sessions is None:
            parser.error(
                'Explicit --item-stats requires --story-ids and --sessions.'
            )
        for name in ('labels', 'story_ids', 'sessions', 'train_seeds'):
            if len(getattr(args, name)) != n_conditions:
                parser.error(
                    f'--{name.replace("_", "-")} must match --item-stats.'
                )
        if len(set(args.labels)) != len(args.labels):
            parser.error('--labels must be unique.')
    else:
        if args.seeds is None:
            parser.error('High-level mode requires --seeds.')
        if args.story_ids is None or args.sessions is None:
            parser.error('High-level mode requires --story-ids and --sessions.')
        if len(args.seeds) < 1:
            parser.error('--seeds must be non-empty.')
        if len(set(args.seeds)) != len(args.seeds):
            parser.error('--seeds must be unique.')
        if len(set(args.story_ids)) != len(args.story_ids):
            parser.error('--story-ids must be unique in high-level mode.')
        if len(set(args.sessions)) != len(args.sessions):
            parser.error('--sessions must be unique in high-level mode.')
        if args.run_group_template is None:
            args.run_group_template = PAPER_RUN_GROUP_TEMPLATE
        n_conditions = (
            len(args.seeds) * len(args.story_ids) * len(args.sessions)
        )
        args.labels = [
            condition_label(seed, story_id, session)
            for seed in args.seeds
            for story_id in args.story_ids
            for session in args.sessions
        ]
    args.input_mode = 'explicit' if explicit else 'high_level'
    if args.subspace_rank is not None and args.subspace_rank < 1:
        parser.error('--subspace-rank must be positive.')
    if 'seeds' in args.compare and args.input_mode != 'high_level':
        parser.error(
            '--compare seeds requires the high-level interface (--seeds).'
        )
    if args.hungarian and 'seeds' not in args.compare:
        parser.error('--hungarian requires --compare seeds.')
    if not args.compare and args.assignments_csv is None:
        parser.error('Pass --compare and/or --assignments-csv.')
    if args.assignments_csv is not None:
        if args.cluster_label is None:
            if n_conditions != 1:
                parser.error(
                    '--cluster-label is required when multiple conditions '
                    'are given with --assignments-csv.'
                )
            args.cluster_label = args.labels[0]
        if args.cluster_label not in args.labels:
            parser.error('--cluster-label must be one of --labels.')
    if args.rng_seed < 0:
        parser.error('--rng-seed must be non-negative.')
    return args


def load_physical_item_stats(path: Path, *, meg_sr: float) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if 'sensor_units' not in data.files:
            raise ValueError(
                f'{path} is missing sensor_units; expected '
                f"sensor_units={PHYSICAL_SENSOR_UNITS!r}."
            )
        sensor_units = str(np.asarray(data['sensor_units']).item())
        if sensor_units != PHYSICAL_SENSOR_UNITS:
            raise ValueError(
                f'{path} has sensor_units={sensor_units!r}, expected '
                f'{PHYSICAL_SENSOR_UNITS!r}.'
            )
        item_subjects = np.asarray(data['item_subjects'], dtype=int)
        item_branches = np.asarray(data['item_branches'], dtype=int)
        demean = bool(np.asarray(data['demean_temporal_filters']).item())

    subjects: list[int] = []
    for subject in item_subjects:
        subject_id = int(subject)
        if subject_id not in subjects:
            subjects.append(subject_id)
    unique_branches = np.unique(item_branches)
    n_branches = int(unique_branches.size)
    expected_branches = np.arange(1, n_branches + 1, dtype=int)
    if not np.array_equal(unique_branches, expected_branches):
        raise ValueError(
            f'{path} branch labels are {unique_branches.tolist()}, expected '
            f'{expected_branches.tolist()}.'
        )

    items = load_cached_item_table(
        item_stats_npz=path,
        subjects=subjects,
        n_branches=n_branches,
        target_fs=float(meg_sr),
        demean_temporal_filters=demean,
    )
    items['path'] = str(path.resolve())
    items['sensor_units'] = PHYSICAL_SENSOR_UNITS
    items['n_branches'] = n_branches
    items['n_subjects'] = len(subjects)
    items['subjects_list'] = subjects
    return items


def item_key_index(items: dict[str, Any]) -> dict[tuple[int, int], int]:
    keys: dict[tuple[int, int], int] = {}
    subjects = np.asarray(items['subjects'], dtype=int)
    branches = np.asarray(items['branches'], dtype=int)
    for index, (subject, branch) in enumerate(
        zip(subjects, branches, strict=True)
    ):
        key = (int(subject), int(branch))
        if key in keys:
            raise ValueError(f'Duplicate (subject, branch) item {key}.')
        keys[key] = int(index)
    return keys


def paired_pearson(
    a: np.ndarray,
    b: np.ndarray,
    feature_name: str,
) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f'{feature_name} paired arrays have different shapes: '
            f'{a.shape} vs {b.shape}.'
        )
    if a.ndim != 2:
        raise ValueError(f'{feature_name} must be 2D, got {a.shape}.')
    n_items = a.shape[0]
    if n_items == 0:
        raise ValueError(f'{feature_name} has no items to pair.')
    similarity = correlation_similarity(
        np.concatenate([a, b], axis=0),
        feature_name,
        absolute=False,
    )
    return np.diag(similarity[:n_items, n_items:]).astype(np.float64)


def assert_pairable_patterns(
    items_a: dict[str, Any],
    items_b: dict[str, Any],
    *,
    context: str,
) -> None:
    spatial_a = np.asarray(items_a['spatial_patterns'])
    spatial_b = np.asarray(items_b['spatial_patterns'])
    if spatial_a.shape[1] != spatial_b.shape[1]:
        raise ValueError(
            f'{context}: spatial channel counts differ: '
            f'{spatial_a.shape[1]} vs {spatial_b.shape[1]}.'
        )
    t_a = int(np.asarray(items_a['temporal_patterns']).shape[1])
    t_b = int(np.asarray(items_b['temporal_patterns']).shape[1])
    if t_a != t_b:
        raise ValueError(
            f'{context}: temporal pattern length differs ({t_a} vs {t_b}). '
            'Conditions with different T are not interpolated.'
        )
    freqs_a = np.asarray(items_a['temporal_spectrum_freqs'], dtype=np.float64)
    freqs_b = np.asarray(items_b['temporal_spectrum_freqs'], dtype=np.float64)
    if freqs_a.shape != freqs_b.shape or not np.allclose(freqs_a, freqs_b):
        raise ValueError(
            f'{context}: temporal_spectrum_freqs differ and will not be '
            'resampled.'
        )
    if 'source_magnitudes' not in items_a or 'source_magnitudes' not in items_b:
        missing = [
            side
            for side, items in (('a', items_a), ('b', items_b))
            if 'source_magnitudes' not in items
        ]
        raise ValueError(
            f'{context}: required source_magnitudes missing from item '
            f'table(s) {missing}.'
        )
    source_a = np.asarray(items_a['source_magnitudes'])
    source_b = np.asarray(items_b['source_magnitudes'])
    if source_a.ndim != 2 or source_b.ndim != 2:
        raise ValueError(
            f'{context}: source_magnitudes must be 2D, got '
            f'{source_a.shape} vs {source_b.shape}.'
        )
    if source_a.shape[1] != source_b.shape[1]:
        raise ValueError(
            f'{context}: source-space vertex counts differ: '
            f'{source_a.shape[1]} vs {source_b.shape[1]}.'
        )


def pair_story_indices(
    items_a: dict[str, Any],
    items_b: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys_a = item_key_index(items_a)
    keys_b = item_key_index(items_b)
    if set(keys_a) != set(keys_b):
        raise ValueError(
            'Story comparison requires identical (participant, branch) items.'
        )
    subjects = np.empty(len(keys_a), dtype=int)
    branches = np.empty(len(keys_a), dtype=int)
    index_a = np.empty(len(keys_a), dtype=int)
    index_b = np.empty(len(keys_a), dtype=int)
    for row, key in enumerate(sorted(keys_a)):
        subjects[row] = key[0]
        branches[row] = key[1]
        index_a[row] = keys_a[key]
        index_b[row] = keys_b[key]
    return subjects, branches, index_a, index_b


def pair_session_indices(
    items_a: dict[str, Any],
    items_b: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    repeated = set(repeated_session_subjects())
    keys_a = item_key_index(items_a)
    keys_b = item_key_index(items_b)
    subjects_a = {subject for subject, _branch in keys_a} & repeated
    subjects_b = {subject for subject, _branch in keys_b} & repeated
    if subjects_a != subjects_b:
        raise ValueError(
            'Repeated-session participants present in one session item table '
            'are missing from the other: '
            f'only_a={sorted(subjects_a - subjects_b)}, '
            f'only_b={sorted(subjects_b - subjects_a)}.'
        )
    if not subjects_a:
        raise ValueError(
            'Session comparison found no repeated-session participants.'
        )

    rows: list[tuple[int, int, int, int]] = []
    for subject in sorted(subjects_a):
        branches_a = {branch for sub, branch in keys_a if sub == subject}
        branches_b = {branch for sub, branch in keys_b if sub == subject}
        if branches_a != branches_b:
            raise ValueError(
                f'Subject {subject} has different branches across sessions: '
                f'{sorted(branches_a)} vs {sorted(branches_b)}.'
            )
        for branch in sorted(branches_a):
            rows.append(
                (
                    subject,
                    branch,
                    keys_a[(subject, branch)],
                    keys_b[(subject, branch)],
                )
            )
    subjects = np.asarray([row[0] for row in rows], dtype=int)
    branches = np.asarray([row[1] for row in rows], dtype=int)
    index_a = np.asarray([row[2] for row in rows], dtype=int)
    index_b = np.asarray([row[3] for row in rows], dtype=int)
    return subjects, branches, index_a, index_b


def feature_pair_similarities(
    items_a: dict[str, Any],
    items_b: dict[str, Any],
    index_a: np.ndarray,
    index_b: np.ndarray,
    *,
    context: str,
) -> dict[str, np.ndarray]:
    assert_pairable_patterns(items_a, items_b, context=context)
    similarities: dict[str, np.ndarray] = {}
    for feature_name, key in FEATURE_KEYS:
        similarities[feature_name] = paired_pearson(
            np.asarray(items_a[key], dtype=np.float64)[index_a],
            np.asarray(items_b[key], dtype=np.float64)[index_b],
            f'{context}_{feature_name}',
        )
    return similarities


def condition_pairs(
    records: Sequence[dict[str, Any]],
    kind: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record_a, record_b in combinations(records, 2):
        if kind == 'stories':
            match = (
                record_a['session'] == record_b['session']
                and record_a['train_seed'] == record_b['train_seed']
                and record_a['story_id'] != record_b['story_id']
            )
        elif kind == 'sessions':
            match = (
                record_a['story_id'] == record_b['story_id']
                and record_a['train_seed'] == record_b['train_seed']
                and record_a['session'] != record_b['session']
            )
        elif kind == 'seeds':
            match = (
                record_a['story_id'] == record_b['story_id']
                and record_a['session'] == record_b['session']
                and record_a['train_seed'] != record_b['train_seed']
            )
        else:
            raise ValueError(f'Unknown comparison kind {kind!r}.')
        if not match:
            continue
        if record_a['label'] <= record_b['label']:
            pairs.append((record_a, record_b))
        else:
            pairs.append((record_b, record_a))
    if not pairs:
        raise ValueError(f'No {kind} pairs among the supplied conditions.')
    return pairs


def format_run_group(template: str, seed: int) -> str:
    run_group = template.format(seed=seed)
    validate_run_group(run_group)
    return run_group


def find_run_dir(
    experiments_root: Path,
    run_group: str,
    seed: int,
    n_branches: int,
) -> tuple[Path, dict]:
    group_dir = experiments_root / run_group
    if not group_dir.is_dir():
        raise FileNotFoundError(f'Missing run group directory: {group_dir}')

    matches: list[tuple[Path, dict]] = []
    for config_path in sorted(group_dir.glob('*/config.json')):
        config = json.loads(config_path.read_text(encoding='utf-8'))
        if int(config.get('seed', -1)) != seed:
            continue
        if int(config.get('n_channels_unmix', -1)) != n_branches:
            continue
        matches.append((config_path.parent, config))

    if len(matches) != 1:
        raise ValueError(
            f'Expected one seed-{seed} {n_branches}-branch run in {group_dir}, '
            f'found {len(matches)}.'
        )
    return matches[0]


def load_spatial_filters(run_dir: Path, n_branches: int) -> np.ndarray:
    filter_paths = sorted(run_dir.glob('filters_*.npz'))
    if len(filter_paths) != 1:
        raise ValueError(
            f'Expected one filters NPZ in {run_dir}, found {len(filter_paths)}.'
        )
    with np.load(filter_paths[0], allow_pickle=False) as data:
        weights = np.asarray(data['overall_spatial_filter_weight'], dtype=np.float64)
    expected = (N_SUBJECTS, n_branches, N_MEG_CHANNELS)
    if weights.shape != expected:
        raise ValueError(
            f'Expected spatial filters {expected} in {filter_paths[0]}, '
            f'got {weights.shape}.'
        )
    _assert_finite('overall_spatial_filter_weight', weights)
    return weights


def subspace_bases(filters: np.ndarray, rank: int) -> np.ndarray:
    n_seeds, n_subjects, n_branches, n_channels = filters.shape
    if rank > n_branches:
        raise ValueError(f'rank {rank} exceeds n_branches {n_branches}.')
    bases = np.empty((n_seeds, n_subjects, n_channels, rank), dtype=np.float64)
    for seed_idx in range(n_seeds):
        for subject_idx in range(n_subjects):
            _, _, right_vectors = np.linalg.svd(
                filters[seed_idx, subject_idx],
                full_matrices=False,
            )
            bases[seed_idx, subject_idx] = right_vectors[:rank].T
    return bases


def pair_score(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    rank = basis_a.shape[1]
    return float(np.linalg.norm(basis_a.T @ basis_b, ord='fro') / np.sqrt(rank))


def numerical_subspace_rank(matrix: np.ndarray) -> int:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f'branch matrix must be 2D, got {matrix.shape}.')
    _assert_finite('branch_matrix', matrix)
    if min(matrix.shape) < 1:
        raise ValueError(
            f'branch matrix has no singular values: {matrix.shape}.'
        )
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    # NumPy matrix_rank default tolerance: s_max * max(shape) * eps(dtype).
    eps = float(np.finfo(singular_values.dtype).eps)
    tolerance = float(singular_values[0] * max(matrix.shape) * eps)
    rank = int(np.sum(singular_values > tolerance))
    if rank < 1:
        raise ValueError(
            f'Numerical rank is zero for matrix of shape {matrix.shape}.'
        )
    return rank


def resolved_subspace_rank(matrix: np.ndarray, requested: int | None) -> int:
    matrix = np.asarray(matrix, dtype=np.float64)
    full_rank = min(matrix.shape)
    if requested is None:
        return numerical_subspace_rank(matrix)
    if requested > full_rank:
        raise ValueError(
            f'subspace rank {requested} exceeds matrix shape {matrix.shape}.'
        )
    return requested


def branch_subspace_basis(matrix: np.ndarray, rank: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f'branch matrix must be 2D, got {matrix.shape}.')
    _assert_finite('branch_matrix', matrix)
    if rank > min(matrix.shape):
        raise ValueError(
            f'rank {rank} exceeds branch matrix shape {matrix.shape}.'
        )
    return subspace_bases(matrix[np.newaxis, np.newaxis], rank)[0, 0]


def principal_angles(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    if basis_a.shape != basis_b.shape:
        raise ValueError(
            f'bases have different shapes: {basis_a.shape} vs {basis_b.shape}.'
        )
    cosines = np.linalg.svd(basis_a.T @ basis_b, compute_uv=False)
    return np.arccos(np.clip(cosines, 0.0, 1.0))


def procrustes_aligned_pearson(a: np.ndarray, b: np.ndarray, name: str) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f'{name} Procrustes arrays have different shapes: '
            f'{a.shape} vs {b.shape}.'
        )
    rotation, _scale = orthogonal_procrustes(a.T, b.T)
    aligned = rotation.T @ a
    return float(np.mean(paired_pearson(aligned, b, name)))


def subspace_metrics(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    *,
    rank: int,
    name: str,
) -> dict[str, Any]:
    if matrix_a.shape != matrix_b.shape:
        raise ValueError(
            f'{name} matrices have different shapes: '
            f'{matrix_a.shape} vs {matrix_b.shape}.'
        )
    basis_a = branch_subspace_basis(matrix_a, rank)
    basis_b = branch_subspace_basis(matrix_b, rank)
    angles = principal_angles(basis_a, basis_b)
    cosines = np.cos(angles)
    overlap = float(pair_score(basis_a, basis_b))
    return {
        'rank': int(rank),
        'n_branches': int(matrix_a.shape[0]),
        'ambient_dim': int(matrix_a.shape[1]),
        'fills_ambient': int(rank) == int(matrix_a.shape[1]),
        'subspace_overlap_rms': overlap,
        'mean_canonical_correlation': float(np.mean(cosines)),
        'mean_principal_angle_rad': float(np.mean(angles)),
        'max_principal_angle_rad': float(np.max(angles)),
        'principal_angles_rad': angles.astype(np.float64),
        'procrustes_mean_pearson': procrustes_aligned_pearson(
            matrix_a, matrix_b, name
        ),
    }


def subject_branch_matrix(
    items: dict[str, Any],
    subject: int,
    key: str,
) -> np.ndarray:
    mask = np.asarray(items['subjects'], dtype=int) == int(subject)
    if not np.any(mask):
        raise ValueError(f'subject {subject} is absent from the item table.')
    branches = np.asarray(items['branches'], dtype=int)[mask]
    order = np.argsort(branches, kind='stable')
    ordered_branches = branches[order]
    expected = np.arange(ordered_branches.size, dtype=int)
    if not np.array_equal(ordered_branches, expected):
        raise ValueError(
            f'subject {subject} branches are {ordered_branches.tolist()}, '
            f'expected {expected.tolist()}.'
        )
    matrix = np.asarray(items[key], dtype=np.float64)[mask][order]
    n_branches = int(items['n_branches'])
    if matrix.shape[0] != n_branches:
        raise ValueError(
            f'subject {subject} {key} has {matrix.shape[0]} rows, expected '
            f'{n_branches}.'
        )
    _assert_finite(f'subject_{subject}_{key}', matrix)
    return matrix


def parse_run_id(run_dir_name: str) -> str:
    # ClearML offline dirs are <run_name>_offline-<run_id>; local runs use
    # <run_name>_<run_id>. The run id excludes the offline- prefix.
    name = str(run_dir_name)
    if '_offline-' in name:
        run_id = name.rsplit('_offline-', 1)[1]
    elif '_' in name:
        run_id = name.rsplit('_', 1)[1]
    else:
        run_id = ''
    if not run_id:
        raise ValueError(f'Cannot parse run id from directory name {name!r}.')
    return run_id


def run_id_from_dir(run_dir: Path) -> str:
    return parse_run_id(Path(run_dir).name)


def resolve_trained_run(
    experiments_root: Path,
    run_group_template: str,
    seed: int,
    n_branches: int,
) -> tuple[Path, dict[str, Any], str]:
    run_group = format_run_group(run_group_template, int(seed))
    run_dir, config = find_run_dir(
        Path(experiments_root),
        run_group,
        int(seed),
        int(n_branches),
    )
    return run_dir, config, run_group


def load_temporal_filters(run_dir: Path, n_branches: int) -> np.ndarray:
    filter_paths = sorted(Path(run_dir).glob('filters_*.npz'))
    if len(filter_paths) != 1:
        raise ValueError(
            f'Expected one filters NPZ in {run_dir}, found {len(filter_paths)}.'
        )
    with np.load(filter_paths[0], allow_pickle=False) as data:
        weights = np.asarray(data['temporal_filters_weight'], dtype=np.float64)
    if weights.ndim != 2 or int(weights.shape[0]) != int(n_branches):
        raise ValueError(
            f'Expected temporal filters with {n_branches} rows in '
            f'{filter_paths[0]}, got {weights.shape}.'
        )
    _assert_finite('temporal_filters_weight', weights)
    return weights


def load_learned_filters(
    run_dir: Path,
    config: dict[str, Any],
    n_branches: int,
) -> dict[str, Any]:
    spatial = load_spatial_filters(run_dir, int(n_branches))
    temporal = load_temporal_filters(run_dir, int(n_branches))
    kernel_size = int(temporal.shape[1])
    if 'temporal_filter_kernel_size' in config:
        configured = int(config['temporal_filter_kernel_size'])
        if configured != kernel_size:
            raise ValueError(
                f'{run_dir} temporal_filter_kernel_size={configured} does not '
                f'match temporal_filters_weight shape {temporal.shape}.'
            )
    return {
        'spatial': spatial,
        'temporal': temporal,
        'kernel_size': kernel_size,
    }


def resolve_high_level_item_stats(
    *,
    interpretations_root: Path,
    run_group: str,
    run_name: str,
    run_id: str,
    session: int,
    story_id: int,
    n_branches: int,
    demean_temporal_filters: bool,
) -> tuple[Path, dict[str, Any], Path]:
    paths = resolve_output_paths(
        interpretations_root,
        run_group=run_group,
        run_name=run_name,
        run_id=run_id,
        session=int(session),
        story_id=int(story_id),
        demean_temporal_filters=bool(demean_temporal_filters),
    )
    _fusion, _source_weight, meta_path, meta = resolve_medoid_config(
        paths['assignments_csv']
    )
    subjects = (
        [int(subject) for subject in meta['subjects']]
        if 'subjects' in meta
        else []
    )
    validate_clustering_metadata(
        meta,
        meta_path,
        run_name=str(run_name),
        run_id=str(run_id),
        run_group=str(run_group),
        session=int(session),
        story_id=int(story_id),
        subjects=subjects,
        preprocessed_meg_path=meta.get('preprocessed_meg_path'),
        data_root=meta.get('meg_files_dir', ''),
        meg_format=str(meta.get('meg_format', '')),
        offset_gap=(
            float(meta['offset_gap']) if 'offset_gap' in meta else float('nan')
        ),
        demean_temporal_filters=bool(demean_temporal_filters),
        n_items=len(subjects) * int(n_branches),
        n_branches=int(n_branches),
    )
    item_stats = resolve_item_stats_npz(paths['assignments_csv'])
    if not item_stats.is_file():
        raise FileNotFoundError(
            f'Missing item-stats NPZ for run_group={run_group!r} '
            f'run_name={run_name!r} run_id={run_id!r} session={session} '
            f'story_id={story_id}: {item_stats}.'
        )
    return item_stats, meta, meta_path


def branch_similarity_matrix(
    a: np.ndarray,
    b: np.ndarray,
    feature_name: str,
    *,
    absolute: bool,
) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f'{feature_name} matrices have different shapes: '
            f'{a.shape} vs {b.shape}.'
        )
    if a.ndim != 2:
        raise ValueError(f'{feature_name} must be 2D, got {a.shape}.')
    k = a.shape[0]
    similarity = correlation_similarity(
        np.concatenate([a, b], axis=0),
        feature_name,
        absolute=bool(absolute),
    )[:k, k:]
    return np.asarray(similarity, dtype=np.float64)


def nested_mean_observation_matrices(
    observations: Sequence[tuple[int, int, int, np.ndarray]],
    *,
    name: str,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    if not observations:
        raise ValueError(f'{name} has no same-participant observations.')
    by_subject_session: dict[tuple[int, int], list[np.ndarray]] = {}
    for subject, session, _story, matrix in observations:
        matrix = np.asarray(matrix, dtype=np.float64)
        _assert_finite(f'{name}_observation', matrix)
        by_subject_session.setdefault((int(subject), int(session)), []).append(
            matrix
        )
    by_subject: dict[int, list[np.ndarray]] = {}
    for (subject, _session), matrices in by_subject_session.items():
        stacked = np.stack(matrices, axis=0)
        by_subject.setdefault(int(subject), []).append(
            np.mean(stacked, axis=0)
        )
    participant: dict[int, np.ndarray] = {}
    for subject, matrices in by_subject.items():
        stacked = np.stack(matrices, axis=0)
        participant[int(subject)] = np.mean(stacked, axis=0)
        _assert_finite(f'{name}_subject_{subject}', participant[subject])
    ordered_subjects = sorted(participant)
    aggregated = np.mean(
        np.stack([participant[subject] for subject in ordered_subjects], axis=0),
        axis=0,
    )
    _assert_finite(name, aggregated)
    return aggregated, participant


def hungarian_assignment(
    similarity: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    similarity = np.asarray(similarity, dtype=np.float64)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError(
            f'Hungarian similarity must be square, got {similarity.shape}.'
        )
    _assert_finite('hungarian_similarity', similarity)
    k = similarity.shape[0]
    row_ind, col_ind = linear_sum_assignment(-similarity)
    match_rows = []
    gaps = []
    for row, col in zip(row_ind, col_ind, strict=True):
        assigned = float(similarity[row, col])
        if k == 1:
            gap = None
        else:
            others = np.delete(similarity[row], col)
            gap = float(assigned - np.max(others))
            gaps.append(gap)
        match_rows.append(
            {
                'branch_a': int(row) + 1,
                'branch_b': int(col) + 1,
                'similarity': assigned,
                'match_gap': gap,
            }
        )
    matches = pd.DataFrame(match_rows)
    matched = matches['similarity'].to_numpy(dtype=np.float64)
    quality = {
        'n_branches': int(k),
        'matched_similarity_mean': float(np.mean(matched)),
        'matched_similarity_min': float(np.min(matched)),
        'mean_match_gap': (
            None if not gaps else float(np.mean(np.asarray(gaps, dtype=np.float64)))
        ),
    }
    return matches, quality


def _records_by_story_session(
    records: Sequence[dict[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:
    by_obs: dict[tuple[int, int], dict[str, Any]] = {}
    for record in records:
        key = (int(record['story_id']), int(record['session']))
        if key in by_obs:
            raise ValueError(
                'Duplicate (story_id, session) '
                f'{key} for train_seed={record["train_seed"]}.'
            )
        by_obs[key] = record
    return by_obs


def pattern_view_observations(
    records_a: Sequence[dict[str, Any]],
    records_b: Sequence[dict[str, Any]],
    view: str,
    *,
    absolute: bool,
) -> list[tuple[int, int, int, np.ndarray]]:
    key = PATTERN_ITEM_KEYS[view]
    by_obs_a = _records_by_story_session(records_a)
    by_obs_b = _records_by_story_session(records_b)
    shared = sorted(set(by_obs_a) & set(by_obs_b))
    if not shared:
        raise ValueError(
            f'{view}: no shared (story, session) observations across seeds.'
        )
    observations: list[tuple[int, int, int, np.ndarray]] = []
    for story_id, session in shared:
        record_a = by_obs_a[(story_id, session)]
        record_b = by_obs_b[(story_id, session)]
        assert_pairable_patterns(
            record_a['items'],
            record_b['items'],
            context=f'seeds_{view}_{record_a["label"]}_{record_b["label"]}',
        )
        subjects_a = {
            int(subject) for subject in np.asarray(record_a['items']['subjects'])
        }
        subjects_b = {
            int(subject) for subject in np.asarray(record_b['items']['subjects'])
        }
        shared_subjects = sorted(subjects_a & subjects_b)
        if not shared_subjects:
            raise ValueError(
                f'{view}: no shared participants for story {story_id} '
                f'session {session}.'
            )
        for subject in shared_subjects:
            matrix_a = subject_branch_matrix(record_a['items'], subject, key)
            matrix_b = subject_branch_matrix(record_b['items'], subject, key)
            similarity = branch_similarity_matrix(
                matrix_a,
                matrix_b,
                f'{view}_s{subject}_story{story_id}_ses{session}',
                absolute=absolute,
            )
            observations.append(
                (int(subject), int(session), int(story_id), similarity)
            )
    return observations


def compare_seed_view_correspondence(
    records: Sequence[dict[str, Any]],
    learned_filters: dict[int, dict[str, Any]],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[int, np.ndarray]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    records_by_seed: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        records_by_seed.setdefault(int(record['train_seed']), []).append(record)
    seeds = sorted(records_by_seed)
    missing_filters = [seed for seed in seeds if seed not in learned_filters]
    if missing_filters:
        raise ValueError(
            f'Missing learned filters for seeds {missing_filters}.'
        )
    if len(seeds) < 2:
        raise ValueError('Seed correspondence requires at least two seeds.')

    n_branches = int(next(iter(learned_filters.values()))['spatial'].shape[1])
    aggregated: dict[str, np.ndarray] = {}
    participant_level: dict[str, dict[int, np.ndarray]] = {}
    assignment_tables: list[pd.DataFrame] = []
    quality_rows: list[dict[str, Any]] = []
    assignment_lookup: dict[tuple[int, int, str], np.ndarray] = {}
    stability_rows: list[dict[str, Any]] = []

    for seed_a, seed_b in combinations(seeds, 2):
        filters_a = learned_filters[seed_a]
        filters_b = learned_filters[seed_b]
        if int(filters_a['spatial'].shape[1]) != n_branches:
            raise ValueError('Spatial filter branch counts differ across seeds.')
        if int(filters_b['spatial'].shape[1]) != n_branches:
            raise ValueError('Spatial filter branch counts differ across seeds.')
        if filters_a['temporal'].shape != filters_b['temporal'].shape:
            raise ValueError(
                'Temporal filter shapes differ: '
                f'{filters_a["temporal"].shape} vs {filters_b["temporal"].shape}.'
            )

        spatial_participant: dict[int, np.ndarray] = {}
        spatial_abs_mats = []
        spatial_signed_mats = []
        for index in range(N_SUBJECTS):
            subject = index + 1
            signed = branch_similarity_matrix(
                filters_a['spatial'][index],
                filters_b['spatial'][index],
                f'spatial_filters_s{subject}',
                absolute=False,
            )
            absolute = np.abs(signed)
            spatial_participant[subject] = absolute
            spatial_abs_mats.append(absolute)
            spatial_signed_mats.append(signed)
        spatial_agg = np.mean(np.stack(spatial_abs_mats, axis=0), axis=0)
        spatial_signed_agg = np.mean(np.stack(spatial_signed_mats, axis=0), axis=0)
        _assert_finite(f'spatial_filters_{seed_a}_{seed_b}', spatial_agg)
        aggregated[f'spatial_filters__s{seed_a}__s{seed_b}'] = spatial_agg
        aggregated[f'spatial_filters__signed__s{seed_a}__s{seed_b}'] = (
            spatial_signed_agg
        )
        participant_level[f'spatial_filters__s{seed_a}__s{seed_b}'] = (
            spatial_participant
        )

        temporal_signed = branch_similarity_matrix(
            filters_a['temporal'],
            filters_b['temporal'],
            f'temporal_filters_{seed_a}_{seed_b}',
            absolute=False,
        )
        temporal_agg = np.abs(temporal_signed)
        aggregated[f'temporal_filters__s{seed_a}__s{seed_b}'] = temporal_agg
        aggregated[f'temporal_filters__signed__s{seed_a}__s{seed_b}'] = (
            temporal_signed
        )

        pattern_agg: dict[str, np.ndarray] = {}
        for view in PATTERN_VIEWS:
            signed_observations = pattern_view_observations(
                records_by_seed[seed_a],
                records_by_seed[seed_b],
                view,
                absolute=False,
            )
            matching_kind = SEED_VIEW_MATCHING_CORRELATION[view]
            if matching_kind == 'signed':
                view_agg, participant = nested_mean_observation_matrices(
                    signed_observations,
                    name=f'{view}_{seed_a}_{seed_b}',
                )
            elif matching_kind == 'absolute':
                abs_observations = [
                    (subject, session, story, np.abs(matrix))
                    for subject, session, story, matrix in signed_observations
                ]
                view_agg, participant = nested_mean_observation_matrices(
                    abs_observations,
                    name=f'{view}_{seed_a}_{seed_b}',
                )
                signed_agg, _signed_part = nested_mean_observation_matrices(
                    signed_observations,
                    name=f'{view}_signed_{seed_a}_{seed_b}',
                )
                aggregated[f'{view}__signed__s{seed_a}__s{seed_b}'] = signed_agg
            else:
                raise ValueError(
                    f'{view} matching correlation {matching_kind!r} is not '
                    "one of ('absolute', 'signed')."
                )
            if view_agg.shape != (n_branches, n_branches):
                raise ValueError(
                    f'{view} aggregated matrix has shape {view_agg.shape}, '
                    f'expected {(n_branches, n_branches)}.'
                )
            aggregated[f'{view}__s{seed_a}__s{seed_b}'] = view_agg
            participant_level[f'{view}__s{seed_a}__s{seed_b}'] = participant
            pattern_agg[view] = view_agg
            if matching_kind == 'absolute':
                pattern_agg[f'{view}_signed'] = aggregated[
                    f'{view}__signed__s{seed_a}__s{seed_b}'
                ]

        view_matrices = {
            'spatial_filters': spatial_agg,
            'temporal_filters': temporal_agg,
            'spatial_haufe': pattern_agg['spatial_haufe'],
            'temporal_pattern': pattern_agg['temporal_pattern'],
            'log_spectrum': pattern_agg['log_spectrum'],
            'source_magnitudes': pattern_agg['source_magnitudes'],
        }
        for view in SEED_VIEWS:
            matches, quality = hungarian_assignment(view_matrices[view])
            matches.insert(0, 'view', view)
            matches.insert(1, 'train_seed_a', int(seed_a))
            matches.insert(2, 'train_seed_b', int(seed_b))
            matches.insert(
                3, 'correlation_kind', SEED_VIEW_MATCHING_CORRELATION[view]
            )
            assignment_tables.append(matches)
            quality_rows.append(
                {
                    'view': view,
                    'train_seed_a': int(seed_a),
                    'train_seed_b': int(seed_b),
                    'correlation_kind': SEED_VIEW_MATCHING_CORRELATION[view],
                    **quality,
                }
            )
            mapping = np.full(n_branches, -1, dtype=int)
            for _, row in matches.iterrows():
                mapping[int(row['branch_a']) - 1] = int(row['branch_b']) - 1
            if np.any(mapping < 0):
                raise ValueError(f'Incomplete Hungarian assignment for {view}.')
            assignment_lookup[(seed_a, seed_b, view)] = mapping
            for branch_a in range(n_branches):
                branch_b = int(mapping[branch_a])
                stability_rows.append(
                    {
                        'train_seed_a': int(seed_a),
                        'train_seed_b': int(seed_b),
                        'matching_view': view,
                        'branch_a': int(branch_a) + 1,
                        'branch_b': int(branch_b) + 1,
                        'spatial_haufe_abs_pearson': float(
                            pattern_agg['spatial_haufe'][branch_a, branch_b]
                        ),
                        'spatial_haufe_signed_pearson': float(
                            pattern_agg['spatial_haufe_signed'][branch_a, branch_b]
                        ),
                        'temporal_pattern_abs_pearson': float(
                            pattern_agg['temporal_pattern'][branch_a, branch_b]
                        ),
                        'temporal_pattern_signed_pearson': float(
                            pattern_agg['temporal_pattern_signed'][
                                branch_a, branch_b
                            ]
                        ),
                        'log_spectrum_pearson': float(
                            pattern_agg['log_spectrum'][branch_a, branch_b]
                        ),
                        'source_magnitudes_pearson': float(
                            pattern_agg['source_magnitudes'][branch_a, branch_b]
                        ),
                    }
                )

    agreement_rows: list[dict[str, Any]] = []
    for seed_a, seed_b in combinations(seeds, 2):
        for view_a, view_b in combinations(SEED_VIEWS, 2):
            map_a = assignment_lookup[(seed_a, seed_b, view_a)]
            map_b = assignment_lookup[(seed_a, seed_b, view_b)]
            n_agree = int(np.sum(map_a == map_b))
            agreement_rows.append(
                {
                    'train_seed_a': int(seed_a),
                    'train_seed_b': int(seed_b),
                    'view_a': view_a,
                    'view_b': view_b,
                    'n_agree': n_agree,
                    'n_branches': n_branches,
                    'fraction_agree': float(n_agree) / float(n_branches),
                    'assignments_equal': n_agree == n_branches,
                }
            )

    return (
        aggregated,
        participant_level,
        pd.concat(assignment_tables, ignore_index=True),
        pd.DataFrame(quality_rows),
        pd.DataFrame(agreement_rows),
        pd.DataFrame(stability_rows),
    )


def compare_item_conditions(
    record_a: dict[str, Any],
    record_b: dict[str, Any],
    *,
    kind: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if kind == 'stories':
        subjects, branches, index_a, index_b = pair_story_indices(
            record_a['items'], record_b['items']
        )
    elif kind == 'sessions':
        subjects, branches, index_a, index_b = pair_session_indices(
            record_a['items'], record_b['items']
        )
    else:
        raise ValueError(f'Unsupported item-pair kind {kind!r}.')
    context = f'{kind}_{record_a["label"]}_{record_b["label"]}'
    similarities = feature_pair_similarities(
        record_a['items'],
        record_b['items'],
        index_a,
        index_b,
        context=context,
    )
    rows = []
    for row, (subject, branch) in enumerate(
        zip(subjects, branches, strict=True)
    ):
        rows.append(
            {
                'comparison': kind,
                'label_a': record_a['label'],
                'label_b': record_b['label'],
                'story_id_a': record_a['story_id'],
                'story_id_b': record_b['story_id'],
                'session_a': record_a['session'],
                'session_b': record_b['session'],
                'train_seed_a': record_a['train_seed'],
                'train_seed_b': record_b['train_seed'],
                'subject': int(subject),
                'branch': int(branch) + 1,
                'spatial_haufe_pearson': float(
                    similarities['spatial_haufe'][row]
                ),
                'temporal_pattern_pearson': float(
                    similarities['temporal_pattern'][row]
                ),
                'log_spectrum_pearson': float(
                    similarities['log_spectrum'][row]
                ),
                'source_magnitudes_pearson': float(
                    similarities['source_magnitudes'][row]
                ),
            }
        )
    table = pd.DataFrame(rows)
    summaries = []
    for feature_name in (
        'spatial_haufe',
        'temporal_pattern',
        'log_spectrum',
        'source_magnitudes',
    ):
        summary = summarize_values(
            similarities[feature_name],
            name=f'{context}_{feature_name}',
        )
        summaries.append(
            {
                'comparison': kind,
                'label_a': record_a['label'],
                'label_b': record_b['label'],
                'feature': feature_name,
                **summary,
            }
        )
    return table, summaries


def compare_seed_conditions(
    record_a: dict[str, Any],
    record_b: dict[str, Any],
    *,
    subspace_rank: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    items_a = record_a['items']
    items_b = record_b['items']
    assert_pairable_patterns(
        items_a,
        items_b,
        context=f'seeds_{record_a["label"]}_{record_b["label"]}',
    )
    subjects_a = np.unique(np.asarray(items_a['subjects'], dtype=int))
    subjects_b = np.unique(np.asarray(items_b['subjects'], dtype=int))
    if not np.array_equal(subjects_a, subjects_b):
        raise ValueError('Seed comparison requires identical participant sets.')
    if int(items_a['n_branches']) != int(items_b['n_branches']):
        raise ValueError(
            'Seed comparison requires identical branch counts: '
            f'{items_a["n_branches"]} vs {items_b["n_branches"]}.'
        )

    score_rows: list[dict[str, Any]] = []
    angle_rows: list[dict[str, Any]] = []
    for subject in subjects_a.tolist():
        spatial_a = subject_branch_matrix(items_a, subject, 'spatial_patterns')
        spatial_b = subject_branch_matrix(items_b, subject, 'spatial_patterns')
        temporal_a = subject_branch_matrix(items_a, subject, 'temporal_patterns')
        temporal_b = subject_branch_matrix(items_b, subject, 'temporal_patterns')
        spatial_num_a = numerical_subspace_rank(spatial_a)
        spatial_num_b = numerical_subspace_rank(spatial_b)
        temporal_num_a = numerical_subspace_rank(temporal_a)
        temporal_num_b = numerical_subspace_rank(temporal_b)
        if subspace_rank is None:
            spatial_rank = min(spatial_num_a, spatial_num_b)
            temporal_rank = min(temporal_num_a, temporal_num_b)
        else:
            spatial_rank = min(
                resolved_subspace_rank(spatial_a, subspace_rank),
                resolved_subspace_rank(spatial_b, subspace_rank),
            )
            temporal_rank = min(
                resolved_subspace_rank(temporal_a, subspace_rank),
                resolved_subspace_rank(temporal_b, subspace_rank),
            )
        spatial = subspace_metrics(
            spatial_a, spatial_b, rank=spatial_rank, name=f'spatial_s{subject}'
        )
        temporal = subspace_metrics(
            temporal_a,
            temporal_b,
            rank=temporal_rank,
            name=f'temporal_s{subject}',
        )
        score_rows.append(
            {
                'label_a': record_a['label'],
                'label_b': record_b['label'],
                'story_id': record_a['story_id'],
                'session': record_a['session'],
                'train_seed_a': record_a['train_seed'],
                'train_seed_b': record_b['train_seed'],
                'subject': int(subject),
                'n_branches': spatial['n_branches'],
                'spatial_rank': spatial['rank'],
                'temporal_rank': temporal['rank'],
                'spatial_numerical_rank_a': spatial_num_a,
                'spatial_numerical_rank_b': spatial_num_b,
                'temporal_numerical_rank_a': temporal_num_a,
                'temporal_numerical_rank_b': temporal_num_b,
                'spatial_n_near_null_a': min(spatial_a.shape) - spatial_num_a,
                'spatial_n_near_null_b': min(spatial_b.shape) - spatial_num_b,
                'temporal_n_near_null_a': min(temporal_a.shape) - temporal_num_a,
                'temporal_n_near_null_b': min(temporal_b.shape) - temporal_num_b,
                'spatial_ambient_dim': spatial['ambient_dim'],
                'temporal_ambient_dim': temporal['ambient_dim'],
                'spatial_fills_ambient': bool(spatial['fills_ambient']),
                'temporal_fills_ambient': bool(temporal['fills_ambient']),
                'spatial_subspace_overlap_rms': spatial['subspace_overlap_rms'],
                'temporal_subspace_overlap_rms': temporal['subspace_overlap_rms'],
                'spatial_mean_canonical_correlation': spatial[
                    'mean_canonical_correlation'
                ],
                'temporal_mean_canonical_correlation': temporal[
                    'mean_canonical_correlation'
                ],
                'spatial_mean_principal_angle_rad': spatial[
                    'mean_principal_angle_rad'
                ],
                'temporal_mean_principal_angle_rad': temporal[
                    'mean_principal_angle_rad'
                ],
                'spatial_max_principal_angle_rad': spatial[
                    'max_principal_angle_rad'
                ],
                'temporal_max_principal_angle_rad': temporal[
                    'max_principal_angle_rad'
                ],
                'spatial_procrustes_mean_pearson': spatial[
                    'procrustes_mean_pearson'
                ],
                'temporal_procrustes_mean_pearson': temporal[
                    'procrustes_mean_pearson'
                ],
            }
        )
        for view, metrics in (('spatial', spatial), ('temporal', temporal)):
            angles = np.asarray(metrics['principal_angles_rad'], dtype=np.float64)
            for component, angle in enumerate(angles):
                angle_rows.append(
                    {
                        'label_a': record_a['label'],
                        'label_b': record_b['label'],
                        'subject': int(subject),
                        'view': view,
                        'component': int(component),
                        'principal_angle_rad': float(angle),
                        'canonical_correlation': float(np.cos(angle)),
                    }
                )
    return pd.DataFrame(score_rows), pd.DataFrame(angle_rows)


def cluster_medoid_representativeness(
    items: dict[str, Any],
    assignments_csv: Path,
    *,
    cluster_names: Sequence[str] | None,
    include_rest: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fusion, source_weight, meta_path, meta = resolve_medoid_config(
        assignments_csv=assignments_csv
    )
    assignments = validate_curated_assignments(
        pd.read_csv(assignments_csv),
        items,
    )
    source_similarity, temporal_similarity = compute_plot_compatible_similarities(
        items
    )
    medoid_similarity = build_medoid_similarity(
        source_similarity=source_similarity,
        temporal_similarity=temporal_similarity,
        fusion=fusion,
        source_weight=source_weight,
    )
    specs, _excluded = build_curated_cluster_specs(
        assignments,
        exclude_rest=not include_rest,
    )
    if cluster_names is not None:
        requested = [str(name) for name in cluster_names]
        if len(requested) != len(set(requested)):
            raise ValueError(f'--clusters contains duplicates: {requested}.')
        by_name = {str(spec['cluster_name']): spec for spec in specs}
        missing = [name for name in requested if name not in by_name]
        if missing:
            raise ValueError(
                f'Requested clusters are unavailable: {missing}. '
                f'Available: {sorted(by_name)}.'
            )
        specs = [by_name[name] for name in requested]
    if not specs:
        raise ValueError('No clusters remain to summarize.')

    member_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for spec in specs:
        cluster_name = str(spec['cluster_name'])
        item_idx = np.asarray(spec['item_idx'], dtype=int)
        medoid_item = choose_medoid_index(
            medoid_similarity,
            item_idx,
            name=f'cluster_{cluster_name}_medoid_similarity',
        )
        if medoid_item not in set(item_idx.tolist()):
            raise ValueError(
                f'Cluster {cluster_name} medoid {medoid_item} is not a member.'
            )
        source_to_medoid = source_similarity[item_idx, medoid_item]
        temporal_to_medoid = temporal_similarity[item_idx, medoid_item]
        member_subjects = np.asarray(items['subjects'], dtype=int)[item_idx]
        member_branches = np.asarray(items['branches'], dtype=int)[item_idx]
        unique_subjects = np.unique(member_subjects)
        for local, item in enumerate(item_idx):
            member_rows.append(
                {
                    'cluster_name': cluster_name,
                    'item_index': int(item),
                    'subject': int(member_subjects[local]),
                    'branch': int(member_branches[local]) + 1,
                    'is_medoid': int(item) == int(medoid_item),
                    'source_similarity_to_medoid': float(source_to_medoid[local]),
                    'temporal_spectrum_similarity_to_medoid': float(
                        temporal_to_medoid[local]
                    ),
                    'source_pearson_to_medoid': float(
                        2.0 * source_to_medoid[local] - 1.0
                    ),
                    'temporal_spectrum_pearson_to_medoid': float(
                        2.0 * temporal_to_medoid[local] - 1.0
                    ),
                }
            )
        others = item_idx != medoid_item
        source_others = source_to_medoid[others]
        temporal_others = temporal_to_medoid[others]
        source_all = summarize_values(
            source_to_medoid, name=f'{cluster_name}_source_to_medoid'
        )
        temporal_all = summarize_values(
            temporal_to_medoid, name=f'{cluster_name}_temporal_to_medoid'
        )
        summary_rows.append(
            {
                'cluster_name': cluster_name,
                'n': int(item_idx.size),
                'n_subjects': int(unique_subjects.size),
                'subjects': ','.join(str(int(sub)) for sub in unique_subjects),
                'medoid_item_index': int(medoid_item),
                'medoid_subject': int(items['subjects'][medoid_item]),
                'medoid_branch': int(items['branches'][medoid_item]) + 1,
                'medoid_is_observed_item': True,
                'medoid_replaced_by_mean': False,
                'source_similarity_to_medoid_mean': source_all['mean'],
                'source_similarity_to_medoid_std': source_all['std'],
                'source_similarity_to_medoid_min': source_all['min'],
                'source_similarity_to_medoid_median': source_all['median'],
                'temporal_spectrum_similarity_to_medoid_mean': temporal_all['mean'],
                'temporal_spectrum_similarity_to_medoid_std': temporal_all['std'],
                'temporal_spectrum_similarity_to_medoid_min': temporal_all['min'],
                'temporal_spectrum_similarity_to_medoid_median': temporal_all[
                    'median'
                ],
                'source_similarity_others_mean': (
                    None
                    if source_others.size == 0
                    else float(np.mean(source_others))
                ),
                'temporal_spectrum_similarity_others_mean': (
                    None
                    if temporal_others.size == 0
                    else float(np.mean(temporal_others))
                ),
            }
        )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(member_rows),
        {
            'fusion': fusion,
            'source_weight': source_weight,
            'meta_path': str(meta_path),
            'clustering_algorithm': meta.get('algorithm'),
            'clustering_source_similarity': meta.get('source_similarity'),
            'clustering_temporal_similarity': meta.get('temporal_similarity'),
            'medoid_selection': (
                'choose_medoid_index on the clustering fusion of source '
                'magnitudes and log-spectrum Pearson similarities; the medoid '
                'is an observed item, not a mean pattern.'
            ),
        },
    )


def haufe_geometry_table(
    records: Sequence[dict[str, Any]],
    sensor_xy: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for record in records:
        items = record['items']
        roughness = compute_item_spatial_roughness(
            items['spatial_patterns'],
            sensor_xy,
        )
        subjects = np.asarray(items['subjects'], dtype=int)
        branches = np.asarray(items['branches'], dtype=int)
        for index, value in enumerate(roughness):
            rows.append(
                {
                    'label': record['label'],
                    'story_id': record['story_id'],
                    'session': record['session'],
                    'train_seed': record['train_seed'],
                    'subject': int(subjects[index]),
                    'branch': int(branches[index]) + 1,
                    'spatial_roughness': float(value),
                    'spatial_smoothness': float(1.0 / value),
                }
            )
    return pd.DataFrame(rows)


def write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False)


def condition_record(
    *,
    label: str,
    path: Path | str,
    story_id: int,
    session: int,
    train_seed: int,
    items: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        'label': str(label),
        'path': str(Path(path).resolve()),
        'story_id': int(story_id),
        'session': int(session),
        'train_seed': int(train_seed),
        'items': items,
        'n_items': int(np.asarray(items['subjects']).shape[0]),
        'n_subjects': int(items['n_subjects']),
        'n_branches': int(items['n_branches']),
        'n_channels': int(items['spatial_patterns'].shape[1]),
        'temporal_len': int(items['temporal_patterns'].shape[1]),
        'demean_temporal_filters': bool(items['demean_temporal_filters']),
        'sensor_units': PHYSICAL_SENSOR_UNITS,
    }
    if extra:
        record.update(extra)
    return record


def load_explicit_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = []
    for path, label, story_id, session, train_seed in zip(
        args.item_stats,
        args.labels,
        args.story_ids,
        args.sessions,
        args.train_seeds,
        strict=True,
    ):
        items = load_physical_item_stats(path, meg_sr=args.meg_sr)
        records.append(
            condition_record(
                label=label,
                path=path,
                story_id=story_id,
                session=session,
                train_seed=train_seed,
                items=items,
            )
        )
    return records


def load_high_level_records(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    learned_filters: dict[int, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for seed in args.seeds:
        run_dir, config, run_group = resolve_trained_run(
            args.experiments_root,
            args.run_group_template,
            int(seed),
            int(args.n_branches),
        )
        run_name = str(config.get('run_name', '')).strip()
        if not run_name:
            raise ValueError(f'{run_dir} config.json has no usable run_name.')
        run_id = run_id_from_dir(run_dir)
        if 'seed' in config and int(config['seed']) != int(seed):
            raise ValueError(
                f'{run_dir} config seed={config["seed"]} does not match {seed}.'
            )
        if 'meg_sr' in config and float(config['meg_sr']) != float(args.meg_sr):
            raise ValueError(
                f'{run_dir} config meg_sr={config["meg_sr"]} does not match '
                f'--meg-sr {args.meg_sr}.'
            )
        filters = load_learned_filters(run_dir, config, int(args.n_branches))
        learned_filters[int(seed)] = {
            **filters,
            'run_dir': str(run_dir.resolve()),
            'run_id': run_id,
            'run_name': run_name,
            'run_group': run_group,
        }
        for story_id in args.story_ids:
            for session in args.sessions:
                item_stats, meta, meta_path = resolve_high_level_item_stats(
                    interpretations_root=args.interpretations_root,
                    run_group=run_group,
                    run_name=run_name,
                    run_id=run_id,
                    session=int(session),
                    story_id=int(story_id),
                    n_branches=int(args.n_branches),
                    demean_temporal_filters=bool(args.demean_temporal_filters),
                )
                items = load_physical_item_stats(item_stats, meg_sr=args.meg_sr)
                if int(items['n_branches']) != int(args.n_branches):
                    raise ValueError(
                        f'{item_stats} has n_branches={items["n_branches"]}, '
                        f'expected {args.n_branches}.'
                    )
                records.append(
                    condition_record(
                        label=condition_label(int(seed), int(story_id), int(session)),
                        path=item_stats,
                        story_id=int(story_id),
                        session=int(session),
                        train_seed=int(seed),
                        items=items,
                        extra={
                            'run_group': run_group,
                            'run_name': run_name,
                            'run_id': run_id,
                            'run_dir': str(run_dir.resolve()),
                            'meta_path': str(meta_path.resolve()),
                            'meta_n_items': meta.get('n_items'),
                        },
                    )
                )
    return records, learned_filters


def write_seed_view_outputs(
    out_dir: Path,
    aggregated: dict[str, np.ndarray],
    participant_level: dict[str, dict[int, np.ndarray]],
    assignments: pd.DataFrame,
    quality: pd.DataFrame,
    agreement: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    np.savez_compressed(out_dir / 'seed_view_similarities.npz', **aggregated)
    participant_payload: dict[str, np.ndarray] = {}
    for key, by_subject in participant_level.items():
        subjects = np.asarray(sorted(by_subject), dtype=int)
        participant_payload[f'{key}__subjects'] = subjects
        for subject, matrix in by_subject.items():
            participant_payload[f'{key}__sub{int(subject)}'] = matrix
    np.savez_compressed(
        out_dir / 'seed_view_participant_similarities.npz',
        **participant_payload,
    )
    write_csv(assignments, out_dir / 'seed_view_assignments.csv')
    write_csv(quality, out_dir / 'seed_view_assignment_quality.csv')
    write_csv(agreement, out_dir / 'seed_view_assignment_agreement.csv')
    write_csv(stability, out_dir / 'seed_matched_pair_stability.csv')


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    learned_filters: dict[int, dict[str, Any]] = {}
    if args.input_mode == 'explicit':
        records = load_explicit_records(args)
    else:
        records, learned_filters = load_high_level_records(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        'rng_seed': int(args.rng_seed),
        'meg_sr': float(args.meg_sr),
        'compare': list(args.compare),
        'input_mode': args.input_mode,
        'n_branches_expected': int(args.n_branches),
        'hungarian': bool(args.hungarian),
        'haufe_geometry': bool(args.haufe_geometry),
        'subspace_rank': args.subspace_rank,
        'subspace_rank_policy': SUBSPACE_RANK_POLICY,
        'subspace_rank_tolerance': SUBSPACE_RANK_TOLERANCE,
        'temporal_row_space_degeneracy': TEMPORAL_ROW_SPACE_DEGENERACY,
        'sensor_units_required': PHYSICAL_SENSOR_UNITS,
        'repeated_session_subjects': list(repeated_session_subjects()),
        'conditions': [
            {key: value for key, value in record.items() if key != 'items'}
            for record in records
        ],
        'primary_seed_metrics': PRIMARY_SEED_ANALYSIS,
        'primary_seed_analysis': PRIMARY_SEED_ANALYSIS,
        'seed_view_matching_correlation': dict(SEED_VIEW_MATCHING_CORRELATION),
        'seed_view_sign_convention': dict(SEED_VIEW_SIGN_CONVENTION),
        'seed_view_signed_diagnostics': SEED_VIEW_SIGNED_DIAGNOSTICS,
        'seed_view_aggregation_reducer': SEED_VIEW_AGGREGATION_REDUCER,
        'seed_view_axis_notes': VIEW_AXIS_NOTES,
        'seed_views': list(SEED_VIEWS),
        'sign_ambiguous_views': list(SIGN_AMBIGUOUS_VIEWS),
        'subspace_metrics_are_secondary': True,
        'hungarian_is_optional': False,
        'hungarian_not_a_stability_metric': (
            'Per-view Hungarian assignments are correspondence diagnostics, '
            'not a collapsed seed-stability scalar. Weak one-to-one matches '
            'must not be read as subspace instability.'
        ),
        'composite_similarity_used_for_matching': False,
    }
    if args.input_mode == 'high_level':
        metadata['run_group_template'] = args.run_group_template
        metadata['seeds'] = [int(seed) for seed in args.seeds]
        metadata['story_ids'] = [int(story) for story in args.story_ids]
        metadata['sessions'] = [int(session) for session in args.sessions]
        metadata['demean_temporal_filters'] = bool(args.demean_temporal_filters)
        metadata['experiments_root'] = str(Path(args.experiments_root).resolve())
        metadata['interpretations_root'] = str(
            Path(args.interpretations_root).resolve()
        )
        metadata['learned_filters'] = {
            str(seed): {
                key: value
                for key, value in payload.items()
                if key not in ('spatial', 'temporal')
            }
            for seed, payload in learned_filters.items()
        }

    if 'stories' in args.compare:
        item_tables = []
        summaries: list[dict[str, Any]] = []
        for record_a, record_b in condition_pairs(records, 'stories'):
            table, pair_summaries = compare_item_conditions(
                record_a, record_b, kind='stories'
            )
            item_tables.append(table)
            summaries.extend(pair_summaries)
        write_csv(
            pd.concat(item_tables, ignore_index=True),
            out_dir / 'story_item_similarities.csv',
        )
        write_csv(pd.DataFrame(summaries), out_dir / 'story_summaries.csv')

    if 'sessions' in args.compare:
        item_tables = []
        summaries = []
        for record_a, record_b in condition_pairs(records, 'sessions'):
            table, pair_summaries = compare_item_conditions(
                record_a, record_b, kind='sessions'
            )
            item_tables.append(table)
            summaries.extend(pair_summaries)
        write_csv(
            pd.concat(item_tables, ignore_index=True),
            out_dir / 'session_item_similarities.csv',
        )
        write_csv(pd.DataFrame(summaries), out_dir / 'session_summaries.csv')

    seed_score_tables: list[pd.DataFrame] = []
    if 'seeds' in args.compare:
        (
            aggregated,
            participant_level,
            assignments,
            quality,
            agreement,
            stability,
        ) = compare_seed_view_correspondence(records, learned_filters)
        write_seed_view_outputs(
            out_dir,
            aggregated,
            participant_level,
            assignments,
            quality,
            agreement,
            stability,
        )
        metadata['temporal_filters_participant_aggregated'] = False
        metadata['spatial_filters_story_session_aggregated'] = False

        angle_tables = []
        for record_a, record_b in condition_pairs(records, 'seeds'):
            scores, angles = compare_seed_conditions(
                record_a, record_b, subspace_rank=args.subspace_rank
            )
            seed_score_tables.append(scores)
            angle_tables.append(angles)
        seed_scores = pd.concat(seed_score_tables, ignore_index=True)
        write_csv(seed_scores, out_dir / 'seed_subspace_scores.csv')
        write_csv(
            pd.concat(angle_tables, ignore_index=True),
            out_dir / 'seed_principal_angles.csv',
        )
        temporal_t_le_k = bool(
            np.all(
                seed_scores['temporal_ambient_dim'].to_numpy()
                <= seed_scores['n_branches'].to_numpy()
            )
        )
        summary_metrics: list[tuple[str, str, str]] = [
            (
                'spatial_subspace_overlap_rms',
                'secondary',
                (
                    'Secondary spatial subspace diagnostic: numerical '
                    'row-space overlap (RMS canonical correlation from principal '
                    'angles). Not the primary seed result.'
                ),
            ),
            (
                'spatial_procrustes_mean_pearson',
                'supporting',
                'Supporting spatial aligned similarity (orthogonal Procrustes).',
            ),
        ]
        if temporal_t_le_k:
            summary_metrics.append(
                (
                    'temporal_procrustes_mean_pearson',
                    'supporting',
                    (
                        'Supporting temporal aligned similarity '
                        '(orthogonal Procrustes). Not a seed-stability '
                        'verdict. When T <= K this quantity has a high chance '
                        'level and must not be read as evidence of '
                        'reproduction without a null distribution; this '
                        'script does not compute one.'
                    ),
                )
            )
        else:
            summary_metrics.append(
                (
                    'temporal_subspace_overlap_rms',
                    'secondary',
                    (
                        'Secondary temporal subspace diagnostic: numerical '
                        'row-space overlap (RMS canonical correlation from '
                        'principal angles). Not the primary seed result.'
                    ),
                )
            )
            summary_metrics.append(
                (
                    'temporal_procrustes_mean_pearson',
                    'supporting',
                    (
                        'Supporting temporal aligned similarity '
                        '(orthogonal Procrustes).'
                    ),
                )
            )
        seed_summaries = []
        for column, role, note in summary_metrics:
            seed_summaries.append(
                {
                    'metric': column,
                    'stability_role': role,
                    'is_stability_score': False,
                    'degenerate': False,
                    'note': note,
                    **summarize_values(
                        seed_scores[column].to_numpy(dtype=np.float64),
                        name=column,
                    ),
                }
            )
        if temporal_t_le_k:
            empty_stats = {
                key: None
                for key in (
                    'mean',
                    'std',
                    'min',
                    'q10',
                    'q25',
                    'median',
                    'q75',
                    'q90',
                    'max',
                )
            }
            seed_summaries.append(
                {
                    'metric': 'temporal_subspace_overlap_rms',
                    'stability_role': 'not_a_stability_score',
                    'is_stability_score': False,
                    'degenerate': True,
                    'note': TEMPORAL_ROW_SPACE_DEGENERACY,
                    'n': len(seed_scores),
                    **empty_stats,
                }
            )
        metadata['temporal_t_le_k'] = temporal_t_le_k
        metadata['temporal_subspace_overlap_is_stability_score'] = False
        metadata['reported_temporal_seed_stability_metric'] = None
        write_csv(pd.DataFrame(seed_summaries), out_dir / 'seed_summaries.csv')

    if args.assignments_csv is not None:
        cluster_record = next(
            record for record in records if record['label'] == args.cluster_label
        )
        summary, members, cluster_meta = cluster_medoid_representativeness(
            cluster_record['items'],
            args.assignments_csv,
            cluster_names=args.clusters,
            include_rest=args.include_rest,
        )
        write_csv(summary, out_dir / 'cluster_medoid_summary.csv')
        write_csv(members, out_dir / 'cluster_member_to_medoid.csv')
        metadata['cluster_medoid'] = cluster_meta
        metadata['cluster_label'] = args.cluster_label

    if args.haufe_geometry:
        if args.sensor_xy is None:
            sensor_xy = load_sensor_xy()
            sensor_xy_path = 'load_sensor_xy()'
        else:
            sensor_xy = np.asarray(np.load(args.sensor_xy), dtype=np.float64)
            sensor_xy_path = str(Path(args.sensor_xy).resolve())
        roughness_table = haufe_geometry_table(records, sensor_xy)
        write_csv(roughness_table, out_dir / 'haufe_geometry_roughness.csv')
        metadata['haufe_geometry_report'] = {
            'sensor_xy': sensor_xy_path,
            'smoothness_is_not_neuronal_validity': SMOOTHNESS_NOT_NEURONAL_VALIDITY,
            'cross_seed_pattern_reproducibility': (
                'See seed_view_assignments.csv and seed_matched_pair_stability.csv '
                'when --compare seeds is used. seed_subspace_scores.csv is a '
                'secondary diagnostic and does not assume branch-index identity.'
            ),
        }

    (out_dir / 'metadata.json').write_text(
        json.dumps(json_safe(metadata), indent=2, sort_keys=True),
        encoding='utf-8',
    )
    print(f'Saved interpretation reproducibility outputs to {out_dir}')


if __name__ == '__main__':
    main()
