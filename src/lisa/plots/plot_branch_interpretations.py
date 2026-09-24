"""Plot branch-interpretation clusters from model-derived patterns."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from lisa.data.meg_io import load_raw_meg
from lisa.model.load_model import load_model
from lisa.plots.branch_interpretation import (
    _assert_finite,
    compute_plot_compatible_similarities,
    extract_run_filters,
    get_cluster_output_group,
    validate_curated_assignments,
)
from lisa.plots.plot_style import paper_topomap, publication_style
from lisa.plots.source_estimation import fsaverage_stc_template, render_lh_rh_views
from lisa.utils.constants import (
    BRANCH_INTERPRETATION_ASSIGNMENTS_DIR,
    CLEAN_DATA_DIR,
    EXPERIMENTS_DIR,
    N_SUBJECTS,
    PLOTS_DIR,
    PREPROCESSED_DATA_DIR,
)
from lisa.utils.validators import validate_run_group


EPS = 1e-12
TEMPORAL_FILTER_SUFFIX = {
    True: "_dtf",
    False: "_raw_tf",
}
CLUSTERS_PER_FIGURE = 6
FIGURE_ROW_LABELS = (
    "Topography",
    "Temporal pattern",
    "Spectrum",
    "LH",
    "RH",
)


def choose_medoid_index(
    similarity: np.ndarray,
    item_idx: np.ndarray,
    name: str,
) -> int:
    similarity = np.asarray(similarity, dtype=np.float64)
    item_idx = np.asarray(item_idx, dtype=int)
    submatrix = similarity[np.ix_(item_idx, item_idx)]
    _assert_finite(f"{name}_submatrix", submatrix)

    if submatrix.shape[0] == 1:
        return int(item_idx[0])
    return int(item_idx[int(np.argmax(submatrix.mean(axis=1)))])


def build_medoid_similarity(
    source_similarity: np.ndarray,
    temporal_similarity: np.ndarray,
    source_weight: float | None,
    fusion: str = "weighted_mean",
) -> np.ndarray:
    if fusion == "weighted_mean":
        if source_weight is None or not 0.0 <= source_weight <= 1.0:
            raise ValueError(f"source_weight must be in [0, 1], got {source_weight}")
        medoid_similarity = (
            source_weight * source_similarity
            + (1.0 - source_weight) * temporal_similarity
        )
    elif fusion == "minimum":
        medoid_similarity = np.minimum(source_similarity, temporal_similarity)
    else:
        raise ValueError(f"Unsupported medoid fusion {fusion!r}.")

    medoid_similarity = np.clip(np.asarray(medoid_similarity, dtype=np.float64), 0.0, 1.0)
    np.fill_diagonal(medoid_similarity, 1.0)
    _assert_finite("medoid_similarity", medoid_similarity)
    return medoid_similarity


def resolve_medoid_config(
    assignments_csv: str | Path,
) -> tuple[str, float | None, Path, dict[str, Any]]:
    assignments_path = Path(assignments_csv)
    meta_path = assignments_path.with_name(f"{assignments_path.stem}_meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing clustering metadata JSON for {assignments_path}: {meta_path}. "
            "Use a complete output from combined_cluster_branch_interpretations.py."
        )

    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    if "fusion" not in meta:
        raise ValueError(f"Clustering metadata is missing required key 'fusion': {meta_path}")
    fusion = str(meta["fusion"])
    if fusion not in {"weighted_mean", "minimum"}:
        raise ValueError(
            f"Unsupported medoid fusion {fusion!r} in {meta_path}. "
            "Expected 'weighted_mean' or 'minimum'."
        )

    if fusion == "weighted_mean" and "source_weight" not in meta:
        raise ValueError(
            f"Clustering metadata is missing required key 'source_weight': {meta_path}"
        )
    source_weight = float(meta["source_weight"]) if fusion == "weighted_mean" else None
    return fusion, source_weight, meta_path, meta


def validate_clustering_metadata(
    meta: dict[str, Any],
    meta_path: Path,
    *,
    run_name: str,
    run_id: str,
    run_group: str,
    session: int,
    story_id: int,
    subjects: list[int],
    preprocessed_meg_path: str | Path | None,
    data_root: str | Path,
    meg_format: str,
    offset_gap: float,
    demean_temporal_filters: bool,
    n_items: int,
    n_branches: int,
) -> None:
    expected_values: dict[str, Any] = {
        "run_name": str(run_name),
        "run_id": str(run_id),
        "run_group": str(run_group),
        "session": int(session),
        "story_id": int(story_id),
        "subjects": [int(subject) for subject in subjects],
        "preprocessed_meg_path": (
            str(preprocessed_meg_path) if preprocessed_meg_path is not None else None
        ),
        "meg_files_dir": str(data_root),
        "meg_format": str(meg_format),
        "offset_gap": float(offset_gap),
        "demean_temporal_filters": bool(demean_temporal_filters),
        "n_items": int(n_items),
        "n_branches": int(n_branches),
    }
    for key, expected in expected_values.items():
        if key not in meta:
            raise ValueError(f"Clustering metadata is missing required key {key!r}: {meta_path}")
        observed = meta[key]
        if isinstance(expected, float):
            matches = np.isclose(float(observed), expected)
        else:
            matches = observed == expected
        if not matches:
            raise ValueError(
                f"Clustering metadata mismatch for {key!r} in {meta_path}: "
                f"expected {expected!r}, got {observed!r}."
            )


def summarize_pairwise_similarity(
    similarity: np.ndarray,
    item_idx: np.ndarray,
    name: str,
) -> dict[str, float | None]:
    similarity = np.asarray(similarity, dtype=np.float64)
    item_idx = np.asarray(item_idx, dtype=int)
    submatrix = similarity[np.ix_(item_idx, item_idx)]
    _assert_finite(f"{name}_submatrix", submatrix)

    if submatrix.shape[0] <= 1:
        return {
            "min": None,
            "q10": None,
            "median": None,
            "q90": None,
            "max": None,
        }

    pair_values = submatrix[np.triu_indices(submatrix.shape[0], k=1)]
    _assert_finite(f"{name}_pair_values", pair_values)
    return {
        "min": float(np.min(pair_values)),
        "q10": float(np.percentile(pair_values, 10.0)),
        "median": float(np.median(pair_values)),
        "q90": float(np.percentile(pair_values, 90.0)),
        "max": float(np.max(pair_values)),
    }


def build_curated_cluster_specs(
    assignments: pd.DataFrame,
    exclude_rest: bool,
) -> tuple[list[dict[str, Any]], int]:
    cluster_specs = []
    excluded_item_count = 0
    if "cluster_order" not in assignments.attrs:
        raise ValueError(
            "Assignments must be validated before building cluster specs; "
            "missing assignments.attrs['cluster_order']."
        )
    cluster_order = assignments.attrs["cluster_order"]

    for cluster_name in cluster_order:
        cluster_mask = assignments["cluster_name"] == cluster_name
        item_idx = assignments.loc[cluster_mask, "item_index"].to_numpy(dtype=int)
        if item_idx.size == 0:
            continue
        if exclude_rest and get_cluster_output_group(cluster_name) == "rest":
            excluded_item_count += int(item_idx.size)
            continue
        cluster_specs.append(
            {
                "cluster_name": str(cluster_name),
                "item_idx": item_idx,
            }
        )

    def cluster_sort_key(cluster: dict[str, Any]) -> tuple[int, str, int]:
        cluster_name = str(cluster["cluster_name"]).strip()
        return (
            -int(cluster["item_idx"].size),
            cluster_name[0].lower(),
            int(cluster_name[1:]),
        )

    cluster_specs.sort(key=cluster_sort_key)
    return cluster_specs, excluded_item_count


def resolve_assignments_csv(
    assignments_csv: str | Path | None,
    run_group: str,
    run_id: str,
    run_name: str,
    session: int,
    story_id: int,
    demean_temporal_filters: bool,
) -> Path:
    if assignments_csv is not None:
        path = Path(assignments_csv)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    assignments_root = Path(BRANCH_INTERPRETATION_ASSIGNMENTS_DIR)
    run_dir_name = f"{run_name}-{run_id}"
    stem = f"ses{session}-story{story_id}"
    suffix = TEMPORAL_FILTER_SUFFIX[bool(demean_temporal_filters)]
    run_dir_name += suffix
    stem += suffix
    csv_path = (
        assignments_root
        / run_group
        / run_dir_name
        / f"{stem}.csv"
    )

    if csv_path.exists():
        return csv_path

    raise FileNotFoundError(
        "Could not infer assignments CSV. Tried: "
        f"{csv_path}. Pass --assignments-csv explicitly if your file uses a different name."
    )


def resolve_item_stats_npz(assignments_csv: str | Path) -> Path:
    assignments_path = Path(assignments_csv)
    return assignments_path.with_name(f"{assignments_path.stem}_item_stats.npz")


def load_cached_item_table(
    item_stats_npz: str | Path,
    *,
    subjects: list[int],
    n_branches: int,
    target_fs: float,
    demean_temporal_filters: bool,
) -> dict[str, Any]:
    item_stats_npz = Path(item_stats_npz)
    with np.load(item_stats_npz, allow_pickle=False) as data:
        required = {
            "item_subjects",
            "item_branches",
            "item_spatial_patterns",
            "item_source_magnitudes",
            "item_temporal_patterns",
            "item_temporal_spectra",
            "temporal_spectrum_freqs",
            "demean_temporal_filters",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"Cached item stats {item_stats_npz} is missing keys: {sorted(missing)}"
            )
        if "sensor_units" not in data.files:
            raise ValueError(
                f"Cached item stats {item_stats_npz} is missing "
                "sensor_units. Rerun "
                "lisa-combined-cluster-branch-interpretations."
            )
        sensor_units = str(np.asarray(data["sensor_units"]).item())
        if sensor_units != "physical":
            raise ValueError(
                f"Cached item stats {item_stats_npz} has "
                f"sensor_units={sensor_units!r}, expected 'physical'. "
                "Rerun lisa-combined-cluster-branch-interpretations."
            )

        item_subjects = np.asarray(data["item_subjects"], dtype=int)
        item_branches_one_based = np.asarray(data["item_branches"], dtype=int)
        spatial_patterns = np.asarray(data["item_spatial_patterns"], dtype=np.float64)
        source_magnitudes = np.asarray(data["item_source_magnitudes"], dtype=np.float64)
        temporal_patterns = np.asarray(data["item_temporal_patterns"], dtype=np.float64)
        temporal_spectra_mag = np.asarray(data["item_temporal_spectra"], dtype=np.float64)
        temporal_spectrum_freqs = np.asarray(
            data["temporal_spectrum_freqs"], dtype=np.float64
        )
        cached_demean = bool(np.asarray(data["demean_temporal_filters"]).item())

    expected_subjects = np.repeat(np.asarray(subjects, dtype=int), n_branches)
    expected_branches = np.tile(
        np.arange(1, n_branches + 1, dtype=int),
        len(subjects),
    )
    if not np.array_equal(item_subjects, expected_subjects):
        raise ValueError(
            f"Cached item subjects in {item_stats_npz} do not match requested subjects."
        )
    if not np.array_equal(item_branches_one_based, expected_branches):
        raise ValueError(
            f"Cached item branches in {item_stats_npz} do not match requested branches."
        )
    if cached_demean != bool(demean_temporal_filters):
        raise ValueError(
            f"Cached item stats demean_temporal_filters={cached_demean} does not match "
            f"requested demean_temporal_filters={bool(demean_temporal_filters)}."
        )

    for name, array in [
        ("cached_spatial_patterns", spatial_patterns),
        ("cached_source_magnitudes", source_magnitudes),
        ("cached_temporal_patterns", temporal_patterns),
        ("cached_temporal_spectra_mag", temporal_spectra_mag),
        ("cached_temporal_spectrum_freqs", temporal_spectrum_freqs),
    ]:
        _assert_finite(name, array)

    if spatial_patterns.ndim != 2:
        raise ValueError(
            f"Cached spatial patterns must be 2D, got {spatial_patterns.shape}."
        )
    if source_magnitudes.ndim != 2:
        raise ValueError(
            f"Cached source magnitudes must be 2D, got {source_magnitudes.shape}."
        )
    if temporal_patterns.ndim != 2:
        raise ValueError(
            f"Cached temporal patterns must be 2D, got {temporal_patterns.shape}."
        )
    n_items = len(subjects) * n_branches
    if spatial_patterns.shape[0] != n_items:
        raise ValueError(
            f"Cached spatial patterns have {spatial_patterns.shape[0]} items, "
            f"expected {n_items}."
        )
    if source_magnitudes.shape[0] != n_items:
        raise ValueError(
            f"Cached source magnitudes have {source_magnitudes.shape[0]} items, "
            f"expected {n_items}."
        )
    if temporal_patterns.shape[0] != n_items:
        raise ValueError(
            f"Cached temporal patterns have {temporal_patterns.shape[0]} items, "
            f"expected {n_items}."
        )
    if temporal_spectra_mag.shape[0] != n_items:
        raise ValueError(
            f"Cached temporal spectra have {temporal_spectra_mag.shape[0]} items, "
            f"expected {n_items}."
        )

    return {
        "subjects": item_subjects,
        "branches": item_branches_one_based - 1,
        "spatial_patterns": spatial_patterns,
        "source_magnitudes": source_magnitudes,
        "temporal_patterns": temporal_patterns,
        "temporal_pattern_times_ms": (
            np.arange(temporal_patterns.shape[1], dtype=np.float64)
            / float(target_fs)
            * 1000.0
        ),
        "temporal_spectra_log": np.log(temporal_spectra_mag + EPS),
        "temporal_spectra_mag": temporal_spectra_mag,
        "temporal_spectrum_freqs": temporal_spectrum_freqs,
        "demean_temporal_filters": cached_demean,
    }


def load_reference_raw_for_geometry(
    *,
    sub: int,
    ses: int,
    story_id: int,
    meg_format: str,
    data_root: str | Path,
) -> mne.io.BaseRaw:
    raw, _ = load_raw_meg(
        meg_format=meg_format,
        data_root=str(data_root),
        sub=sub,
        ses=ses,
        story_id=story_id,
        preload=False,
    )
    return raw


def attach_cached_source_geometry(
    items: dict[str, Any],
    *,
    reference_raw: mne.io.BaseRaw,
    subjects: list[int],
    n_branches: int,
) -> None:
    first_subject = int(subjects[0])
    subject_idx = np.flatnonzero(items["subjects"] == first_subject)
    if subject_idx.size != n_branches:
        raise ValueError(
            f"Cached item table has {subject_idx.size} items for subject "
            f"{first_subject}, expected {n_branches}."
        )
    if len(reference_raw.ch_names) != items["spatial_patterns"].shape[1]:
        raise ValueError(
            "Channel count mismatch between cached spatial patterns and Raw info."
        )
    items["source_geometry_stc"] = fsaverage_stc_template(n_times=1)


def summarize_clusters(
    items: dict[str, Any],
    cluster_specs: list[dict[str, Any]],
    n_branches: int,
    source_similarity: np.ndarray,
    temporal_similarity: np.ndarray,
    medoid_fusion: str,
    medoid_source_weight: float | None,
) -> list[dict[str, Any]]:
    if not cluster_specs:
        raise ValueError("No clusters remain after filtering")

    clusters = []
    n_items_total = items["subjects"].shape[0]
    if "source_geometry_stc" not in items:
        raise ValueError("items['source_geometry_stc'] is required for source plotting.")
    source_geometry_stc = items["source_geometry_stc"]
    medoid_similarity = build_medoid_similarity(
        source_similarity=source_similarity,
        temporal_similarity=temporal_similarity,
        fusion=medoid_fusion,
        source_weight=medoid_source_weight,
    )

    for cluster_idx, cluster_spec in enumerate(
        tqdm(cluster_specs, desc="Summarizing clusters"), start=1
    ):
        cluster_name = str(cluster_spec["cluster_name"])
        item_idx = np.asarray(cluster_spec["item_idx"], dtype=int)
        medoid_item_idx = choose_medoid_index(
            medoid_similarity,
            item_idx,
            name=f"cluster_{cluster_name}_medoid_similarity",
        )
        medoid_spatial_pattern = items["spatial_patterns"][medoid_item_idx].astype(
            np.float64
        )
        medoid_temporal_pattern = items["temporal_patterns"][medoid_item_idx].astype(
            np.float64
        )
        medoid_temporal_spectrum = items["temporal_spectra_mag"][
            medoid_item_idx
        ].astype(np.float64)
        medoid_source_magnitude = items["source_magnitudes"][medoid_item_idx].astype(
            np.float64
        )
        _assert_finite(
            f"cluster_{cluster_name}_medoid_spatial_pattern",
            medoid_spatial_pattern,
        )
        _assert_finite(
            f"cluster_{cluster_name}_medoid_source_magnitude",
            medoid_source_magnitude,
        )

        branch_counts = np.bincount(items["branches"][item_idx], minlength=n_branches)
        dominant_branches = [
            (int(branch + 1), int(branch_counts[branch]))
            for branch in np.argsort(branch_counts)[::-1]
            if branch_counts[branch] > 0
        ]
        dominant_branches_text = ", ".join(
            [f"B{branch}:{count}" for branch, count in dominant_branches[:3]]
        )
        source_similarity_stats = summarize_pairwise_similarity(
            source_similarity,
            item_idx,
            name=f"cluster_{cluster_name}_source_similarity",
        )
        temporal_similarity_stats = summarize_pairwise_similarity(
            temporal_similarity,
            item_idx,
            name=f"cluster_{cluster_name}_temporal_similarity",
        )

        clusters.append(
            {
                "cluster_id": cluster_idx,
                "cluster_name": cluster_name,
                "size": int(item_idx.size),
                "coverage_pct": 100.0 * float(item_idx.size) / float(n_items_total),
                "unique_subject_count": int(np.unique(items["subjects"][item_idx]).size),
                "dominant_branches_text": dominant_branches_text,
                "source_similarity_stats": source_similarity_stats,
                "temporal_similarity_stats": temporal_similarity_stats,
                "medoid_fusion": medoid_fusion,
                "medoid_source_weight": medoid_source_weight,
                "medoid_item_index": medoid_item_idx,
                "medoid_spatial_pattern": medoid_spatial_pattern,
                "medoid_source_magnitude": medoid_source_magnitude,
                "medoid_temporal_pattern": medoid_temporal_pattern,
                "medoid_temporal_spectrum": medoid_temporal_spectrum,
            }
        )

    for cluster in tqdm(clusters, desc="Rendering source maps"):
        stc_cluster = source_geometry_stc.copy()
        stc_cluster._data = cluster["medoid_source_magnitude"][:, np.newaxis]
        source_views = render_lh_rh_views(stc_cluster)
        cluster["source_image_lh"] = source_views["lh"]
        cluster["source_image_rh"] = source_views["rh"]
        cluster["source_image"] = source_views["side"]

    return clusters


def build_analysis_summary(
    model,
    hyper_params: dict[str, Any],
    assignments_csv: str | Path,
    run_id: str,
    run_group: str,
    subjects: list[int],
    ses: int,
    story_id: int,
    offset_gap: float,
    meg_format: str,
    data_root: str | Path,
    preprocessed_meg_path: str | Path,
    exclude_rest: bool,
    demean_temporal_filters: bool,
) -> dict[str, Any]:
    if not subjects:
        raise ValueError("At least one subject is required")

    filters = extract_run_filters(
        model,
        hyper_params,
        demean_temporal_filters_enabled=demean_temporal_filters,
    )
    target_fs = hyper_params["meg_sr"]
    n_branches = int(filters["K"])

    medoid_fusion, resolved_medoid_source_weight, medoid_meta_path, clustering_meta = (
        resolve_medoid_config(assignments_csv=assignments_csv)
    )
    validate_clustering_metadata(
        meta=clustering_meta,
        meta_path=medoid_meta_path,
        run_name=str(hyper_params["run_name"]),
        run_id=run_id,
        run_group=run_group,
        session=ses,
        story_id=story_id,
        subjects=subjects,
        preprocessed_meg_path=preprocessed_meg_path,
        data_root=data_root,
        meg_format=meg_format,
        offset_gap=offset_gap,
        demean_temporal_filters=demean_temporal_filters,
        n_items=len(subjects) * n_branches,
        n_branches=n_branches,
    )

    item_stats_npz = resolve_item_stats_npz(assignments_csv)
    if not item_stats_npz.exists():
        raise FileNotFoundError(
            "Incomplete clustering output: required item statistics are missing at "
            f"{item_stats_npz}. Re-run the clustering command with item-stat saving "
            "enabled (the default), or pass --assignments-csv for a complete output. "
            "The plotter does not recompute model-derived patterns."
        )
    print(f"Using cached item stats from {item_stats_npz}")
    items = load_cached_item_table(
        item_stats_npz=item_stats_npz,
        subjects=subjects,
        n_branches=n_branches,
        target_fs=target_fs,
        demean_temporal_filters=demean_temporal_filters,
    )
    reference_raw = load_reference_raw_for_geometry(
        sub=int(subjects[0]),
        ses=ses,
        story_id=story_id,
        meg_format=meg_format,
        data_root=data_root,
    )
    attach_cached_source_geometry(
        items,
        reference_raw=reference_raw,
        subjects=subjects,
        n_branches=n_branches,
    )
    if len(reference_raw.ch_names) != items["spatial_patterns"].shape[1]:
        raise ValueError(
            "Channel count mismatch between computed spatial patterns and Raw info used "
            "for topomap and dipole rendering."
        )
    assignments = validate_curated_assignments(pd.read_csv(assignments_csv), items)
    cluster_specs, excluded_item_count = build_curated_cluster_specs(
        assignments=assignments,
        exclude_rest=exclude_rest,
    )
    if not cluster_specs:
        raise ValueError(
            "No clusters remain after filtering. Disable --exclude-rest or "
            "provide non-rest assignments."
        )

    source_similarity, temporal_similarity = compute_plot_compatible_similarities(items)
    clusters = summarize_clusters(
        items=items,
        cluster_specs=cluster_specs,
        n_branches=n_branches,
        source_similarity=source_similarity,
        temporal_similarity=temporal_similarity,
        medoid_fusion=medoid_fusion,
        medoid_source_weight=resolved_medoid_source_weight,
    )
    print(
        f"Loaded assignments from {assignments_csv}: "
        f"{len(clusters)} plotted clusters, {excluded_item_count} excluded rest items"
    )
    print(
        "Using cluster medoid selection: "
        f"fusion={medoid_fusion}, "
        f"source_weight={resolved_medoid_source_weight} from {medoid_meta_path}"
    )

    return {
        "run_name": str(hyper_params["run_name"]),
        "session": ses,
        "story_id": story_id,
        "assignments_csv": Path(assignments_csv),
        "medoid_fusion": medoid_fusion,
        "medoid_source_weight": resolved_medoid_source_weight,
        "reference_raw": reference_raw,
        "temporal_pattern_times_ms": items["temporal_pattern_times_ms"],
        "temporal_spectrum_freqs": items["temporal_spectrum_freqs"],
        "demean_temporal_filters": bool(demean_temporal_filters),
        "clusters": clusters,
    }


@publication_style
def plot_cluster_summary_figure(
    summary: dict[str, Any],
    clusters: list[dict[str, Any]],
) -> plt.Figure:
    if not clusters:
        raise ValueError("No clusters available to plot")

    info = summary["reference_raw"].info
    time_ms = summary["temporal_pattern_times_ms"]
    freqs = summary["temporal_spectrum_freqs"]
    n_columns = len(clusters)
    fig = plt.figure(figsize=(11.69, 6.3), layout="constrained")
    grid = fig.add_gridspec(
        len(FIGURE_ROW_LABELS),
        n_columns + 1,
        height_ratios=[0.95, 0.72, 0.72, 0.45, 0.45],
        width_ratios=[0.18, *([1.0] * n_columns)],
    )
    axes = np.empty((len(FIGURE_ROW_LABELS), n_columns), dtype=object)
    for row, label in enumerate(FIGURE_ROW_LABELS):
        label_ax = fig.add_subplot(grid[row, 0])
        label_ax.text(
            0.5, 0.5, label, rotation=90, ha="center", va="center"
        )
        label_ax.set_axis_off()
        for column in range(n_columns):
            axes[row, column] = fig.add_subplot(grid[row, column + 1])

    time_min = float(np.min(time_ms))
    time_max = float(np.max(time_ms))
    xticks_time = MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(
        time_min,
        time_max,
    )
    xticks_time = xticks_time[
        (xticks_time >= time_min - 1e-9) & (xticks_time <= time_max + 1e-9)
    ]
    xticks = np.arange(0.0, np.max(freqs) + 1e-9, 25.0)
    xticks = xticks[xticks <= np.max(freqs)]

    for column, cluster in enumerate(clusters):
        medoid_temporal_pattern = cluster["medoid_temporal_pattern"][::-1]
        axes[0, column].set_title(get_cluster_display_label(cluster), pad=4)

        paper_topomap(
            cluster["medoid_spatial_pattern"],
            info,
            kind="topography",
            axes=axes[0, column],
            show=False,
        )

        axes[1, column].plot(
            time_ms,
            medoid_temporal_pattern,
            color="C0",
        )
        axes[1, column].set_xlim(time_ms[0], time_ms[-1])
        axes[1, column].set_xticks(xticks_time)
        temporal_pattern_limit = max(
            float(np.max(np.abs(medoid_temporal_pattern))),
            EPS,
        )
        axes[1, column].set_ylim(
            -1.05 * temporal_pattern_limit, 1.05 * temporal_pattern_limit
        )
        axes[1, column].set_box_aspect(0.55)
        axes[1, column].tick_params(axis="both", labelsize=7, pad=1)
        axes[1, column].grid(True, axis="x", alpha=0.2)
        axes[1, column].set_xlabel("Time (ms)", fontsize=8, labelpad=2)

        axes[2, column].plot(
            freqs,
            cluster["medoid_temporal_spectrum"],
            color="C1",
        )
        axes[2, column].set_xlim(freqs[0], freqs[-1])
        axes[2, column].set_xticks(xticks)
        temporal_spectrum_limit = max(
            float(np.max(cluster["medoid_temporal_spectrum"])),
            EPS,
        )
        axes[2, column].set_ylim(0.0, 1.05 * temporal_spectrum_limit)
        axes[2, column].set_box_aspect(0.55)
        axes[2, column].tick_params(axis="both", labelsize=7, pad=1)
        axes[2, column].grid(True, axis="x", alpha=0.2)
        axes[2, column].set_xlabel("Frequency (Hz)", fontsize=8, labelpad=2)

        axes[3, column].imshow(cluster["source_image_lh"])
        axes[3, column].axis("off")

        axes[4, column].imshow(cluster["source_image_rh"])
        axes[4, column].axis("off")

    return fig


def select_clusters(
    clusters: list[dict[str, Any]],
    cluster_names: list[str] | None,
) -> list[dict[str, Any]]:
    """Select clusters by stable name while preserving the requested order."""
    if cluster_names is None:
        return list(clusters)
    if len(cluster_names) != len(set(cluster_names)):
        raise ValueError(f"--clusters contains duplicate names: {cluster_names}")

    cluster_by_name = {str(cluster["cluster_name"]): cluster for cluster in clusters}
    missing = [name for name in cluster_names if name not in cluster_by_name]
    if missing:
        raise ValueError(
            f"Requested clusters are unavailable: {missing}. "
            f"Available clusters: {list(cluster_by_name)}"
        )
    return [cluster_by_name[name] for name in cluster_names]


def with_sequential_display_labels(
    clusters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return shallow copies labeled sequentially within main/rest groups."""
    group_counts = {"main": 0, "rest": 0}
    labeled_clusters = []
    for cluster in clusters:
        labeled_cluster = dict(cluster)
        group = get_cluster_output_group(cluster["cluster_name"])
        group_counts[group] += 1
        prefix = "Cluster" if group == "main" else "Rest cluster"
        labeled_cluster["display_label"] = f"{prefix} {group_counts[group]}"
        labeled_clusters.append(labeled_cluster)
    return labeled_clusters


def get_cluster_display_label(cluster: dict[str, Any]) -> str:
    if "display_label" in cluster:
        return str(cluster["display_label"])
    cluster_name = str(cluster["cluster_name"]).strip()
    prefix = (
        "Cluster"
        if get_cluster_output_group(cluster_name) == "main"
        else "Rest cluster"
    )
    return f"{prefix} {int(cluster_name[1:])}"


def save_cluster_arrays(
    summary: dict[str, Any],
    outdir: Path,
    filename_suffix: str = "",
) -> Path:
    """Save every cluster-level array needed to reproduce the figure content."""
    clusters = summary["clusters"]
    if not clusters:
        raise ValueError("No clusters available to save")

    info = summary["reference_raw"].info
    path = outdir / f"cluster_interpretation_data{filename_suffix}.npz"
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=int),
        run_name=np.asarray(summary["run_name"]),
        session=np.asarray(summary["session"], dtype=int),
        story_id=np.asarray(summary["story_id"], dtype=int),
        demean_temporal_filters=np.asarray(
            summary["demean_temporal_filters"], dtype=bool
        ),
        cluster_ids=np.asarray(
            [cluster["cluster_id"] for cluster in clusters], dtype=int
        ),
        cluster_names=np.asarray([cluster["cluster_name"] for cluster in clusters]),
        cluster_groups=np.asarray(
            [get_cluster_output_group(cluster["cluster_name"]) for cluster in clusters]
        ),
        cluster_sizes=np.asarray([cluster["size"] for cluster in clusters], dtype=int),
        cluster_coverage_pct=np.asarray(
            [cluster["coverage_pct"] for cluster in clusters], dtype=np.float64
        ),
        cluster_subject_counts=np.asarray(
            [cluster["unique_subject_count"] for cluster in clusters], dtype=int
        ),
        medoid_item_indices=np.asarray(
            [cluster["medoid_item_index"] for cluster in clusters], dtype=int
        ),
        medoid_spatial_patterns=np.stack(
            [cluster["medoid_spatial_pattern"] for cluster in clusters]
        ),
        medoid_temporal_patterns=np.stack(
            [cluster["medoid_temporal_pattern"] for cluster in clusters]
        ),
        plotted_temporal_patterns=np.stack(
            [cluster["medoid_temporal_pattern"][::-1] for cluster in clusters]
        ),
        medoid_temporal_spectra=np.stack(
            [cluster["medoid_temporal_spectrum"] for cluster in clusters]
        ),
        medoid_source_magnitudes=np.stack(
            [cluster["medoid_source_magnitude"] for cluster in clusters]
        ),
        source_images=np.stack([cluster["source_image"] for cluster in clusters]),
        source_images_lh=np.stack(
            [cluster["source_image_lh"] for cluster in clusters]
        ),
        source_images_rh=np.stack(
            [cluster["source_image_rh"] for cluster in clusters]
        ),
        temporal_pattern_times_ms=np.asarray(
            summary["temporal_pattern_times_ms"], dtype=np.float64
        ),
        temporal_spectrum_freqs=np.asarray(
            summary["temporal_spectrum_freqs"], dtype=np.float64
        ),
        sensor_names=np.asarray(info["ch_names"]),
        sensor_locs=np.stack(
            [np.asarray(channel["loc"], dtype=np.float64) for channel in info["chs"]]
        ),
    )
    return path


def save_cluster_table(
    outdir: Path,
    group_name: str,
    clusters: list[dict[str, Any]],
    filename_suffix: str = "",
) -> None:
    pd.DataFrame(
        [
            {
                "cluster_id": cluster["cluster_id"],
                "cluster_name": cluster["cluster_name"],
                "size": cluster["size"],
                "coverage_pct": cluster["coverage_pct"],
                "unique_subject_count": cluster["unique_subject_count"],
                "source_similarity_min": cluster["source_similarity_stats"]["min"],
                "source_similarity_q10": cluster["source_similarity_stats"]["q10"],
                "source_similarity_median": cluster["source_similarity_stats"][
                    "median"
                ],
                "source_similarity_q90": cluster["source_similarity_stats"]["q90"],
                "source_similarity_max": cluster["source_similarity_stats"]["max"],
                "temporal_similarity_min": cluster["temporal_similarity_stats"]["min"],
                "temporal_similarity_q10": cluster["temporal_similarity_stats"]["q10"],
                "temporal_similarity_median": cluster["temporal_similarity_stats"][
                    "median"
                ],
                "temporal_similarity_q90": cluster["temporal_similarity_stats"]["q90"],
                "temporal_similarity_max": cluster["temporal_similarity_stats"]["max"],
                "top_branches": cluster["dominant_branches_text"],
                "medoid_fusion": cluster["medoid_fusion"],
                "medoid_source_weight": cluster["medoid_source_weight"],
                "medoid_item_index": cluster["medoid_item_index"],
            }
            for cluster in clusters
        ]
    ).to_csv(outdir / f"cluster_summary_{group_name}{filename_suffix}.csv", index=False)


def save_figure_column_table(
    outdir: Path,
    group_name: str,
    clusters: list[dict[str, Any]],
    page: int,
    filename_suffix: str = "",
) -> None:
    pd.DataFrame(
        [
            {
                "page": page,
                "column": column,
                "display_label": get_cluster_display_label(cluster),
                "cluster_name": cluster["cluster_name"],
                "size": cluster["size"],
                "unique_subject_count": cluster["unique_subject_count"],
                "coverage_pct": cluster["coverage_pct"],
            }
            for column, cluster in enumerate(clusters, start=1)
        ]
    ).to_csv(
        outdir
        / f"cluster_interpretation_summary_{group_name}{filename_suffix}_columns.csv",
        index=False,
    )


def save_cluster_figure(
    summary: dict[str, Any],
    outdir: Path,
    group_name: str,
    clusters: list[dict[str, Any]],
    filename_suffix: str = "",
) -> None:
    fig = plot_cluster_summary_figure(summary=summary, clusters=clusters)
    fig.savefig(
        outdir / f"cluster_interpretation_summary_{group_name}{filename_suffix}.pdf",
        dpi=300,
    )
    fig.savefig(
        outdir / f"cluster_interpretation_summary_{group_name}{filename_suffix}.png",
        dpi=300,
    )
    plt.close(fig)


def save_outputs(
    summary: dict[str, Any],
    outdir: str | Path,
    cluster_names: list[str] | None = None,
    renumber_plot_labels: bool = False,
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    filename_suffix = TEMPORAL_FILTER_SUFFIX[bool(summary["demean_temporal_filters"])]

    all_grouped_clusters = {"main": [], "rest": []}
    for cluster in summary["clusters"]:
        all_grouped_clusters[get_cluster_output_group(cluster["cluster_name"])].append(
            cluster
        )

    arrays_path = save_cluster_arrays(
        summary=summary,
        outdir=outdir,
        filename_suffix=filename_suffix,
    )
    print(f"Saved cluster arrays to {arrays_path}")

    for group_name, clusters in all_grouped_clusters.items():
        if clusters:
            save_cluster_table(
                outdir=outdir,
                group_name=group_name,
                clusters=clusters,
                filename_suffix=filename_suffix,
            )

    selected_clusters = select_clusters(summary["clusters"], cluster_names)
    if renumber_plot_labels:
        selected_clusters = with_sequential_display_labels(selected_clusters)
    selected_grouped_clusters = {"main": [], "rest": []}
    for cluster in selected_clusters:
        selected_grouped_clusters[
            get_cluster_output_group(cluster["cluster_name"])
        ].append(cluster)

    for group_name, clusters in selected_grouped_clusters.items():
        if not clusters:
            continue
        selection_suffix = ""
        if cluster_names is not None:
            selected_names = "-".join(
                str(cluster["cluster_name"]) for cluster in clusters
            )
            selection_suffix = f"_clusters-{selected_names}"
        for page, start in enumerate(
            range(0, len(clusters), CLUSTERS_PER_FIGURE), start=1
        ):
            page_clusters = clusters[start : start + CLUSTERS_PER_FIGURE]
            page_suffix = f"_page{page:02d}"
            figure_suffix = f"{selection_suffix}{page_suffix}{filename_suffix}"
            save_cluster_figure(
                summary=summary,
                outdir=outdir,
                group_name=group_name,
                clusters=page_clusters,
                filename_suffix=figure_suffix,
            )
            save_figure_column_table(
                outdir=outdir,
                group_name=group_name,
                clusters=page_clusters,
                page=page,
                filename_suffix=figure_suffix,
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, required=True, help="ID of the trained model run.")
    parser.add_argument(
        "--run-group",
        type=str,
        required=True,
        help="Run group id (experiment dir is experiments_root/run_group/*_run_id/).",
    )
    parser.add_argument(
        "--experiments-root",
        type=str,
        default=EXPERIMENTS_DIR,
        help="Root directory with ClearML experiment logs",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(Path(PLOTS_DIR) / "branch_interpretations"),
        help="Output directory for interpretation plots",
    )
    parser.add_argument(
        "--assignments-csv",
        type=str,
        default=None,
        help=(
            "Assignments CSV. Default: "
            "outputs/motif_clusters/<run_group>/<run_name>-<run_id>/"
            "ses<session>-story<story>.csv."
        ),
    )
    parser.add_argument(
        "--clusters",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Cluster names to plot, in the requested column order (for example: "
            "--clusters C1 C4 C9). All clusters are still saved to NPZ and CSV."
        ),
    )
    parser.add_argument(
        "--renumber-plot-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Label plotted clusters sequentially in their requested display order "
            "without changing their stable cluster IDs."
        ),
    )
    parser.add_argument("--session", type=int, default=0, help="Session number to use")
    parser.add_argument("--story-id", type=int, default=1, help="Story ID to use")
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=list(range(1, N_SUBJECTS + 1)),
        help="Subject IDs to summarize.",
    )
    parser.add_argument(
        "--offset-gap",
        type=float,
        default=10.0,
        help="Seconds removed from the start and end of each recording",
    )
    parser.add_argument(
        "--meg-files-dir",
        type=str,
        default=CLEAN_DATA_DIR,
        help="Directory containing cleaned raw MEG files",
    )
    parser.add_argument(
        "--meg-format",
        type=str,
        default="fif",
        choices=("bids", "fif"),
        help="Format of the MEG files",
    )
    parser.add_argument(
        "--preprocessed-meg-path",
        type=str,
        default=None,
        help="Path to preprocessed MEG NPZ.",
    )
    parser.add_argument(
        "--exclude-rest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Exclude curated rest clusters named R<number> from all outputs.',
    )
    parser.add_argument(
        "--demean-temporal-filters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Subtract each learned temporal filter mean before all pattern, covariance, "
            "source, spectrum, and plotting computations. Default: disabled."
        ),
    )
    return parser.parse_args()


def main_cli() -> None:
    args = parse_arguments()
    validate_run_group(args.run_group)
    model, hyper_params = load_model(
        run_id=args.run_id,
        experiments_root=args.experiments_root,
        run_group=args.run_group,
    )

    if args.preprocessed_meg_path is None:
        args.preprocessed_meg_path = (
            Path(PREPROCESSED_DATA_DIR)
            / "meg"
            / f'meg{N_SUBJECTS}_sr{int(hyper_params["meg_sr"])}.npz'
        )

    assignments_csv = resolve_assignments_csv(
        assignments_csv=args.assignments_csv,
        run_group=args.run_group,
        run_id=args.run_id,
        run_name=str(hyper_params["run_name"]),
        session=args.session,
        story_id=args.story_id,
        demean_temporal_filters=args.demean_temporal_filters,
    )
    print(f"Using assignments from {assignments_csv}")
    outdir = Path(args.outdir) / args.run_group / f'{hyper_params["run_name"]}-{args.run_id}'
    output_name = f"ses{args.session}-story{args.story_id}"
    output_name += TEMPORAL_FILTER_SUFFIX[bool(args.demean_temporal_filters)]
    outdir = outdir / output_name

    summary = build_analysis_summary(
        model=model,
        hyper_params=hyper_params,
        assignments_csv=assignments_csv,
        run_id=args.run_id,
        run_group=args.run_group,
        subjects=args.subjects,
        ses=args.session,
        story_id=args.story_id,
        offset_gap=args.offset_gap,
        meg_format=args.meg_format,
        data_root=args.meg_files_dir,
        preprocessed_meg_path=args.preprocessed_meg_path,
        exclude_rest=args.exclude_rest,
        demean_temporal_filters=args.demean_temporal_filters,
    )
    save_outputs(
        summary=summary,
        outdir=outdir,
        cluster_names=args.clusters,
        renumber_plot_labels=args.renumber_plot_labels,
    )

    print(f"Saved interpretation plots to {outdir}")


if __name__ == "__main__":
    main_cli()
