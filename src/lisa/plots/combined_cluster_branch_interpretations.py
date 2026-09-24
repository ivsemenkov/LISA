#!/usr/bin/env python3
"""Cluster branch interpretations once using their weakest-view similarity.

Non-rough items are clustered by complete linkage. The clustering distance is
the maximum of source-topography and temporal-spectrum correlation distances,
so a threshold cut enforces the same minimum correlation in both views.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lisa.plots.branch_interpretation import (
    DEFAULT_MAIN_SPATIAL_ROUGHNESS_MAX,
    DEFAULT_MINIMUM_SIMILARITY,
    TEMPORAL_FILTER_SUFFIX,
    build_clustering_outputs,
    extract_item_table_from_model,
    save_item_table_npz,
)
from lisa.utils.constants import (
    BRANCH_INTERPRETATION_ASSIGNMENTS_DIR,
    CLEAN_DATA_DIR,
    DATA_ROOT,
    EXPERIMENTS_DIR,
    N_SUBJECTS,
    OUTPUTS_DIR,
    PREPROCESSED_DATA_DIR,
)
from lisa.utils.validators import validate_run_group


def resolve_output_paths(
    outdir: str | Path,
    *,
    run_group: str,
    run_name: str,
    run_id: str,
    session: int,
    story_id: int,
    demean_temporal_filters: bool,
) -> dict[str, Path]:
    suffix = TEMPORAL_FILTER_SUFFIX[bool(demean_temporal_filters)]
    base_dir = Path(outdir) / run_group / f'{run_name}-{run_id}{suffix}'
    stem = f'ses{session}-story{story_id}{suffix}'
    return {
        'base_dir': base_dir,
        'assignments_csv': base_dir / f'{stem}.csv',
        'summary_csv': base_dir / f'{stem}_summary.csv',
        'meta_json': base_dir / f'{stem}_meta.json',
        'item_stats_npz': base_dir / f'{stem}_item_stats.npz',
        'similarities_npz': base_dir / f'{stem}_similarities.npz',
    }


def save_outputs(
    result: dict[str, Any],
    paths: dict[str, Path],
    metadata: dict[str, Any],
) -> None:
    paths['base_dir'].mkdir(parents=True, exist_ok=True)
    result['assignments'].to_csv(paths['assignments_csv'], index=False)
    result['summary'].to_csv(paths['summary_csv'], index=False)
    np.savez_compressed(paths['similarities_npz'], **result['similarities'])

    meta = {**metadata, **result['meta']}
    meta['outputs'] = {
        name: str(path) for name, path in paths.items() if name != 'base_dir'
    }
    with paths['meta_json'].open('w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--run-group', required=True)
    parser.add_argument('--experiments-root', default=EXPERIMENTS_DIR)
    parser.add_argument('--outdir', default=BRANCH_INTERPRETATION_ASSIGNMENTS_DIR)
    parser.add_argument('--session', type=int, default=0)
    parser.add_argument('--story-id', type=int, default=1)
    parser.add_argument(
        '--subjects', type=int, nargs='+', default=list(range(1, N_SUBJECTS + 1))
    )
    parser.add_argument('--offset-gap', type=float, default=10.0)
    parser.add_argument('--meg-files-dir', default=CLEAN_DATA_DIR)
    parser.add_argument('--meg-format', choices=('bids', 'fif'), default='fif')
    parser.add_argument('--preprocessed-meg-path', default=None)
    parser.add_argument(
        '--raw-bids-root',
        default=str(Path(DATA_ROOT) / 'MASC-MEG'),
        help='Canonical raw/BIDS root used for participant ELP/HSP/MRK geometry.',
    )
    parser.add_argument(
        '--geometry-cache-dir',
        default=str(Path(OUTPUTS_DIR) / 'source_geometry'),
        help='Directory for cached participant head to fsaverage transforms.',
    )
    parser.add_argument(
        '--demean-temporal-filters',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Subtract each temporal-filter mean before feature extraction.',
    )
    parser.add_argument(
        '--minimum-similarity',
        type=float,
        default=DEFAULT_MINIMUM_SIMILARITY,
        help='Minimum raw Pearson correlation required in both views.',
    )
    parser.add_argument(
        '--spatial-roughness-threshold',
        type=float,
        default=DEFAULT_MAIN_SPATIAL_ROUGHNESS_MAX,
    )
    parser.add_argument(
        '--exclude-rough',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Exclude rough sensor topographies from clustering and assign them to R1.',
    )
    parser.add_argument(
        '--save-item-stats',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from lisa.model.load_model import load_model

    validate_run_group(args.run_group)
    model, hyper_params = load_model(
        run_id=args.run_id,
        experiments_root=args.experiments_root,
        run_group=args.run_group,
    )
    if args.preprocessed_meg_path is None:
        args.preprocessed_meg_path = (
            Path(PREPROCESSED_DATA_DIR)
            / 'meg'
            / f'meg{N_SUBJECTS}_sr{int(hyper_params["meg_sr"])}.npz'
        )

    items, n_branches = extract_item_table_from_model(
        model=model,
        hyper_params=hyper_params,
        subjects=args.subjects,
        ses=args.session,
        story_id=args.story_id,
        offset_gap=args.offset_gap,
        meg_format=args.meg_format,
        data_root=args.meg_files_dir,
        preprocessed_meg_path=args.preprocessed_meg_path,
        demean_temporal_filters=args.demean_temporal_filters,
        raw_bids_root=args.raw_bids_root,
        geometry_cache_dir=args.geometry_cache_dir,
    )
    metadata = {
        'item_source': 'model',
        'run_name': str(hyper_params['run_name']),
        'run_id': str(args.run_id),
        'run_group': str(args.run_group),
        'session': int(args.session),
        'story_id': int(args.story_id),
        'subjects': [int(subject) for subject in args.subjects],
        'meg_files_dir': str(args.meg_files_dir),
        'meg_format': str(args.meg_format),
        'offset_gap': float(args.offset_gap),
        'preprocessed_meg_path': str(args.preprocessed_meg_path),
        'raw_bids_root': str(args.raw_bids_root),
        'geometry_cache_dir': str(args.geometry_cache_dir),
        'demean_temporal_filters': bool(args.demean_temporal_filters),
        'n_branches': int(n_branches),
        'extraction_timing_seconds': items['extraction_timing_seconds'],
        'coregistration_qc': items['coregistration_qc'],
    }
    paths = resolve_output_paths(
        args.outdir,
        run_group=args.run_group,
        run_name=metadata['run_name'],
        run_id=args.run_id,
        session=args.session,
        story_id=args.story_id,
        demean_temporal_filters=args.demean_temporal_filters,
    )
    if args.save_item_stats:
        save_item_table_npz(items, paths['item_stats_npz'])

    result = build_clustering_outputs(
        items,
        minimum_similarity_raw=args.minimum_similarity,
        spatial_roughness_threshold=args.spatial_roughness_threshold,
        exclude_rough=args.exclude_rough,
    )
    save_outputs(result, paths, metadata)
    for name in ('assignments_csv', 'summary_csv', 'similarities_npz', 'meta_json'):
        print(f'Saved {name} to {paths[name]}')
    if args.save_item_stats:
        print(f'Saved item_stats_npz to {paths["item_stats_npz"]}')


if __name__ == '__main__':
    main()
