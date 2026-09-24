"""Participant-specific KIT digitization and fsaverage coregistration."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Sequence

import numpy as np


MM_TO_M = 0.001
M_TO_MM = 1000.0
FSAVERAGE_SUBJECT = 'fsaverage'
# Bump when device-head HPI correspondence/QC policy changes so cached
# head-to-fsaverage transforms cannot be reused under a different policy.
GEOMETRY_POLICY_VERSION = 2
N_KIT_HPI = 5
N_ELP_FIDUCIALS = 3
N_ELP_HPI_POINTS = N_ELP_FIDUCIALS + N_KIT_HPI
IDENTITY_HPI_PERMUTATION = (0, 1, 2, 3, 4)

# Conservative HPI decision thresholds, millimetres.
HPI_SEVERE_RMS_MM = 20.0
HPI_PERMUTATION_PLAUSIBLE_RMS_MM = 12.0
HPI_PERMUTATION_MIN_IMPROVEMENT_MM = 15.0
HPI_PERMUTATION_MIN_SEPARATION_MM = 0.5
HPI_BAD_COIL_PLAUSIBLE_RMS_MM = 8.0
HPI_BAD_COIL_MIN_IMPROVEMENT_MM = 12.0


@dataclass(frozen=True)
class BIDSDigitizationPaths:
    subject: int
    session: int
    task: str
    meg_dir: Path
    elp: Path
    hsp: Path
    mrk: Path


@dataclass(frozen=True)
class ParticipantCoregistration:
    info: Any
    trans: Any
    qc: dict[str, Any]
    cache_path: Path


class UnresolvedHPIGeometryError(RuntimeError):
    """Severe KIT HPI geometry that no conservative automatic correction repairs."""


@dataclass
class HPICorrespondenceResult:
    correction_mode: str
    permutation: tuple[int, ...]
    rejected_coil: int | None
    original_rms_mm: float
    final_rms_mm: float
    original_coil_residuals_mm: tuple[float, ...]
    final_retained_coil_residuals_mm: tuple[float, ...]
    n_retained_coils: int
    mrk: np.ndarray
    elp: np.ndarray


def bids_subject_token(subject: int) -> str:
    return f'{int(subject):02d}'


def bids_task_token(task: int | str) -> str:
    return str(task)


def resolve_bids_meg_dir(
    bids_root: str | Path,
    subject: int,
    session: int,
) -> Path:
    return (
        Path(bids_root)
        / f'sub-{bids_subject_token(subject)}'
        / f'ses-{int(session)}'
        / 'meg'
    )


def resolve_bids_digitization_paths(
    bids_root: str | Path,
    subject: int,
    session: int,
    task: int | str,
) -> BIDSDigitizationPaths:
    """Resolve session ELP/HSP and the task-specific KIT marker file."""

    subject_token = bids_subject_token(subject)
    session_token = str(int(session))
    task_token = bids_task_token(task)
    meg_dir = resolve_bids_meg_dir(bids_root, subject, session)
    paths = BIDSDigitizationPaths(
        subject=int(subject),
        session=int(session),
        task=task_token,
        meg_dir=meg_dir,
        elp=meg_dir / f'sub-{subject_token}_ses-{session_token}_acq-ELP_headshape.pos',
        hsp=meg_dir / f'sub-{subject_token}_ses-{session_token}_acq-HSP_headshape.pos',
        mrk=meg_dir
        / f'sub-{subject_token}_ses-{session_token}_task-{task_token}_markers.mrk',
    )
    missing = [
        str(path)
        for path in (paths.elp, paths.hsp, paths.mrk)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            'Canonical BIDS digitization files are missing for '
            f'sub-{subject_token}/ses-{session_token}/task-{task_token}: {missing}.'
        )
    return paths


def read_kit_pos_millimetres(path: str | Path) -> np.ndarray:
    """Read XYZ rows from a BIDS KIT ``.pos`` file, keeping millimetre units."""

    path = Path(path)
    rows: list[list[float]] = []
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] in {'#', '/', '%'}:
            continue
        parts = stripped.replace(',', ' ').split()
        values: list[float] = []
        for part in parts:
            try:
                values.append(float(part))
            except ValueError:
                values = []
                break
        if len(values) >= 3:
            rows.append(values[:3])
    if not rows:
        raise ValueError(f'No XYZ coordinates found in {path}.')
    return np.asarray(rows, dtype=np.float64)


def read_kit_pos_metres(path: str | Path) -> np.ndarray:
    return read_kit_pos_millimetres(path) * MM_TO_M


def _assert_channel_names_equal(
    observed: Sequence[str],
    expected: Sequence[str],
    *,
    context: str,
) -> None:
    observed_names = [str(name) for name in observed]
    expected_names = [str(name) for name in expected]
    if observed_names != expected_names:
        missing = [name for name in expected_names if name not in observed_names]
        extra = [name for name in observed_names if name not in expected_names]
        raise ValueError(
            f'{context}: channel names/order do not match the scaler sidecar. '
            f'missing={missing[:10]}, extra={extra[:10]}.'
        )


def reorder_info_to_channel_names(info: Any, ch_names: Sequence[str]) -> Any:
    """Return a copy of ``info`` with exactly the scaler channel names/order."""

    expected = [str(name) for name in ch_names]
    if not expected:
        raise ValueError('scaler channel names must be non-empty.')
    if len(expected) != len(set(expected)):
        raise ValueError('scaler channel names must be unique.')
    observed = [str(name) for name in info['ch_names']]
    missing = [name for name in expected if name not in observed]
    extra = [name for name in observed if name not in expected]
    if missing or extra or len(observed) != len(expected):
        raise ValueError(
            'Channel mismatch between KIT Info and scaler sidecar: '
            f'missing={missing[:10]}, extra={extra[:10]}.'
        )
    copied = info.copy()
    if observed != expected:
        import mne

        selection = mne.pick_channels(observed, include=expected, ordered=True)
        copied = mne.pick_info(copied, selection)
    _assert_channel_names_equal(
        copied['ch_names'],
        expected,
        context='after reorder',
    )
    return copied


def _assign_digitization(info: Any, dig: Any, dev_head_t: Any) -> Any:
    unlock = getattr(info, '_unlock', None)
    if unlock is None:
        info['dig'] = dig
        info['dev_head_t'] = dev_head_t
        return info
    with info._unlock():
        info['dig'] = dig
        info['dev_head_t'] = dev_head_t
    return info


def canonical_kit_device_info(template_info: Any, scaler_ch_names: Sequence[str]) -> Any:
    """Copy fixed KIT device-sensor Info and put channels in scaler order."""

    return reorder_info_to_channel_names(template_info, scaler_ch_names)


def _read_kit_mrk(path: str | Path) -> np.ndarray:
    from mne.io.kit import read_mrk

    return np.asarray(read_mrk(path), dtype=np.float64)


def _residuals_mm(residuals_m: np.ndarray) -> tuple[float, ...]:
    return tuple(
        float(value * M_TO_MM) for value in np.asarray(residuals_m, dtype=np.float64)
    )


def _hpi_fit_residuals(mrk: np.ndarray, elp: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-coil residuals and RMS in metres using MNE's KIT device-head fit.

    Mirrors ``mne.io.kit.coreg._set_dig_kit``: MRK is converted ALS->RAS, ELP
    is converted to Neuromag head coordinates via the three fiducials, then a
    rigid transform maps MRK onto the ELP HPI rows ``elp[3:]``.
    """

    from mne.coreg import fit_matched_points
    from mne.io.kit.coreg import als_ras_trans
    from mne.transforms import apply_trans, get_ras_to_neuromag_trans

    mrk = np.asarray(mrk, dtype=np.float64)
    elp = np.asarray(elp, dtype=np.float64)
    if mrk.ndim != 2 or mrk.shape[1] != 3:
        raise ValueError(f'MRK must have shape (n, 3); got {mrk.shape}.')
    if (
        elp.ndim != 2
        or elp.shape[1] != 3
        or elp.shape[0] < N_ELP_FIDUCIALS + mrk.shape[0]
    ):
        raise ValueError(
            'ELP must contain three fiducials plus one HPI row per MRK row; '
            f'got elp {elp.shape} and mrk {mrk.shape}.'
        )
    if mrk.shape[0] != elp.shape[0] - N_ELP_FIDUCIALS:
        raise ValueError(
            'MRK rows must match ELP HPI rows; '
            f'got mrk {mrk.shape[0]} vs elp HPI {elp.shape[0] - N_ELP_FIDUCIALS}.'
        )
    mrk_ras = apply_trans(als_ras_trans, mrk)
    nasion, lpa, rpa = elp[:N_ELP_FIDUCIALS]
    nmtrans = get_ras_to_neuromag_trans(nasion, lpa, rpa)
    elp_nm = apply_trans(nmtrans, elp)
    trans = fit_matched_points(
        tgt_pts=elp_nm[N_ELP_FIDUCIALS:], src_pts=mrk_ras, out='trans'
    )
    fitted = apply_trans(trans, mrk_ras)
    residuals_m = np.linalg.norm(fitted - elp_nm[N_ELP_FIDUCIALS:], axis=1)
    rms_m = float(np.sqrt(np.mean(np.square(residuals_m))))
    return residuals_m, rms_m


def _recording_label(
    subject: int | None,
    session: int | None,
    task: int | str | None,
) -> str:
    subject_token = 'unknown' if subject is None else bids_subject_token(subject)
    session_token = 'unknown' if session is None else str(int(session))
    task_token = 'unknown' if task is None else bids_task_token(task)
    return f'sub-{subject_token}/ses-{session_token}/task-{task_token}'


def _elp_with_hpi(elp: np.ndarray, hpi: np.ndarray) -> np.ndarray:
    return np.vstack(
        [
            np.asarray(elp[:N_ELP_FIDUCIALS], dtype=np.float64),
            np.asarray(hpi, dtype=np.float64),
        ]
    )


def _hpi_qc_dict(result: HPICorrespondenceResult) -> dict[str, Any]:
    return {
        'hpi_original_rms_mm': float(result.original_rms_mm),
        'hpi_final_rms_mm': float(result.final_rms_mm),
        'hpi_original_coil_residuals_mm': [
            float(value) for value in result.original_coil_residuals_mm
        ],
        'hpi_final_retained_coil_residuals_mm': [
            float(value) for value in result.final_retained_coil_residuals_mm
        ],
        'hpi_mrk_permutation': [int(index) for index in result.permutation],
        'hpi_rejected_coil': (
            None if result.rejected_coil is None else int(result.rejected_coil)
        ),
        'hpi_correction_mode': str(result.correction_mode),
        'hpi_n_retained_coils': int(result.n_retained_coils),
        'geometry_policy_version': int(GEOMETRY_POLICY_VERSION),
    }


def resolve_hpi_correspondence(
    mrk: np.ndarray,
    elp: np.ndarray,
    *,
    subject: int | None = None,
    session: int | None = None,
    task: int | str | None = None,
) -> HPICorrespondenceResult:
    """Resolve KIT HPI marker-to-digitizer correspondence conservatively.

    Identity is kept for ordinary recordings. A non-identity permutation is
    accepted only for an unambiguous catastrophic order mismatch. At most one
    HPI coil may be dropped, and only when a single coil is clearly catastrophic.
    Otherwise, severe geometry fails closed.
    """

    mrk = np.asarray(mrk, dtype=np.float64)
    elp = np.asarray(elp, dtype=np.float64)
    if mrk.shape != (N_KIT_HPI, 3):
        raise ValueError(f'KIT MRK must have shape (5, 3); got {mrk.shape}.')
    if elp.ndim != 2 or elp.shape[1] != 3 or elp.shape[0] < N_ELP_HPI_POINTS:
        raise ValueError(
            'KIT ELP must contain three fiducials plus five HPI points; '
            f'got shape {elp.shape}.'
        )
    elp_fids_hpi = np.asarray(elp[:N_ELP_HPI_POINTS], dtype=np.float64)
    hpi = elp_fids_hpi[N_ELP_FIDUCIALS:]

    identity_residuals_m, identity_rms_m = _hpi_fit_residuals(mrk, elp_fids_hpi)
    identity_rms_mm = float(identity_rms_m * M_TO_MM)
    identity_residuals_mm = _residuals_mm(identity_residuals_m)

    perm_candidates: list[tuple[float, tuple[int, ...], np.ndarray]] = []
    for perm in permutations(range(N_KIT_HPI)):
        residuals_m, rms_m = _hpi_fit_residuals(mrk[list(perm)], elp_fids_hpi)
        perm_candidates.append(
            (float(rms_m * M_TO_MM), tuple(int(i) for i in perm), residuals_m)
        )
    perm_candidates.sort(key=lambda item: (item[0], item[1]))
    best_perm_rms_mm, best_perm, best_perm_residuals_m = perm_candidates[0]
    second_perm_rms_mm = (
        perm_candidates[1][0] if len(perm_candidates) > 1 else float('inf')
    )

    loo_candidates: list[tuple[float, int, np.ndarray]] = []
    for coil in range(N_KIT_HPI):
        keep = [index for index in range(N_KIT_HPI) if index != coil]
        residuals_m, rms_m = _hpi_fit_residuals(
            mrk[keep],
            _elp_with_hpi(elp_fids_hpi, hpi[keep]),
        )
        loo_candidates.append((float(rms_m * M_TO_MM), coil, residuals_m))
    loo_candidates.sort(key=lambda item: (item[0], item[1]))
    best_loo_rms_mm, best_loo_coil, best_loo_residuals_m = loo_candidates[0]

    identity_severe = identity_rms_mm >= HPI_SEVERE_RMS_MM
    permutation_ok = (
        identity_severe
        and best_perm != IDENTITY_HPI_PERMUTATION
        and best_perm_rms_mm <= HPI_PERMUTATION_PLAUSIBLE_RMS_MM
        and (identity_rms_mm - best_perm_rms_mm) >= HPI_PERMUTATION_MIN_IMPROVEMENT_MM
        and (second_perm_rms_mm - best_perm_rms_mm) >= HPI_PERMUTATION_MIN_SEPARATION_MM
    )
    qualifying_bad_coils = [
        coil
        for rms_mm, coil, _residuals in loo_candidates
        if (
            rms_mm <= HPI_BAD_COIL_PLAUSIBLE_RMS_MM
            and (identity_rms_mm - rms_mm) >= HPI_BAD_COIL_MIN_IMPROVEMENT_MM
        )
    ]
    bad_coil_ok = identity_severe and len(qualifying_bad_coils) == 1

    if permutation_ok:
        resolved_mrk = np.array(mrk[list(best_perm)], dtype=np.float64, copy=True)
        resolved_elp = np.array(elp_fids_hpi, dtype=np.float64, copy=True)
        return HPICorrespondenceResult(
            correction_mode='permutation',
            permutation=best_perm,
            rejected_coil=None,
            original_rms_mm=identity_rms_mm,
            final_rms_mm=float(best_perm_rms_mm),
            original_coil_residuals_mm=identity_residuals_mm,
            final_retained_coil_residuals_mm=_residuals_mm(best_perm_residuals_m),
            n_retained_coils=N_KIT_HPI,
            mrk=resolved_mrk,
            elp=resolved_elp,
        )

    if not identity_severe:
        resolved_mrk = np.array(mrk, dtype=np.float64, copy=True)
        resolved_elp = np.array(elp_fids_hpi, dtype=np.float64, copy=True)
        return HPICorrespondenceResult(
            correction_mode='identity',
            permutation=IDENTITY_HPI_PERMUTATION,
            rejected_coil=None,
            original_rms_mm=identity_rms_mm,
            final_rms_mm=identity_rms_mm,
            original_coil_residuals_mm=identity_residuals_mm,
            final_retained_coil_residuals_mm=identity_residuals_mm,
            n_retained_coils=N_KIT_HPI,
            mrk=resolved_mrk,
            elp=resolved_elp,
        )

    if bad_coil_ok:
        rejected = int(qualifying_bad_coils[0])
        keep = [index for index in range(N_KIT_HPI) if index != rejected]
        loo_residuals_m = next(
            residuals for rms_mm, coil, residuals in loo_candidates if coil == rejected
        )
        resolved_mrk = np.array(mrk[keep], dtype=np.float64, copy=True)
        resolved_elp = _elp_with_hpi(elp_fids_hpi, hpi[keep])
        return HPICorrespondenceResult(
            correction_mode='bad_coil',
            permutation=IDENTITY_HPI_PERMUTATION,
            rejected_coil=rejected,
            original_rms_mm=identity_rms_mm,
            final_rms_mm=float(
                next(rms for rms, coil, _ in loo_candidates if coil == rejected)
            ),
            original_coil_residuals_mm=identity_residuals_mm,
            final_retained_coil_residuals_mm=_residuals_mm(loo_residuals_m),
            n_retained_coils=N_KIT_HPI - 1,
            mrk=resolved_mrk,
            elp=resolved_elp,
        )

    coil_text = ', '.join(f'{value:.2f}' for value in identity_residuals_mm)
    raise UnresolvedHPIGeometryError(
        'Unresolved KIT HPI geometry for '
        f'{_recording_label(subject, session, task)}: '
        f'identity RMS {identity_rms_mm:.2f} mm; '
        f'best permutation {best_perm} RMS {best_perm_rms_mm:.2f} mm; '
        f'per-coil residuals [{coil_text}] mm; '
        f'best leave-one-out drop coil {best_loo_coil} '
        f'RMS {best_loo_rms_mm:.2f} mm; '
        'automatic correction was refused because the all-five registration '
        'is severely invalid and no conservative permutation or single-coil '
        'exclusion met the acceptance criteria.'
    )


def reconstruct_participant_kit_info(
    *,
    template_info: Any,
    scaler_ch_names: Sequence[str],
    bids_root: str | Path,
    subject: int,
    session: int,
    task: int | str,
) -> Any:
    """Rebuild participant head geometry onto canonical KIT device sensors.

    Device channel locations come from ``template_info``. Digitization and
    ``dev_head_t`` are reconstructed from canonical BIDS ELP/HSP/MRK files and
    replace any geometry stored on a cleaned FIF.
    """

    info, _hpi_qc = _reconstruct_participant_kit_info(
        template_info=template_info,
        scaler_ch_names=scaler_ch_names,
        bids_root=bids_root,
        subject=subject,
        session=session,
        task=task,
    )
    return info


def _reconstruct_participant_kit_info(
    *,
    template_info: Any,
    scaler_ch_names: Sequence[str],
    bids_root: str | Path,
    subject: int,
    session: int,
    task: int | str,
) -> tuple[Any, dict[str, Any]]:
    """Rebuild KIT Info and return HPI correspondence QC."""

    from mne.io.kit.coreg import _set_dig_kit

    paths = resolve_bids_digitization_paths(bids_root, subject, session, task)
    info = canonical_kit_device_info(template_info, scaler_ch_names)
    elp = read_kit_pos_metres(paths.elp)
    hsp = read_kit_pos_metres(paths.hsp)
    mrk = _read_kit_mrk(paths.mrk)
    resolved = resolve_hpi_correspondence(
        mrk,
        elp,
        subject=subject,
        session=session,
        task=task,
    )
    # Correspondence and any coil deletion are already applied to the arrays.
    # Do not pass bad_coils: MNE only deletes matching ELP HPI rows when ELP
    # is path-like, so array ELP plus bad_coils would desynchronize MRK/ELP.
    dig, dev_head_t, _hpi_results = _set_dig_kit(
        resolved.mrk,
        resolved.elp,
        hsp,
        eeg={},
    )
    _assign_digitization(info, dig, dev_head_t)
    _assert_channel_names_equal(
        info['ch_names'],
        scaler_ch_names,
        context=f'sub-{bids_subject_token(subject)} reconstructed Info',
    )
    if info['dig'] is None or info['dev_head_t'] is None:
        raise RuntimeError('Reconstructed KIT Info is missing dig or dev_head_t.')
    return info, _hpi_qc_dict(resolved)


def _coregistration_qc(
    *,
    subject: int,
    session: int,
    task: int | str,
    coreg: Any,
) -> dict[str, Any]:
    distances = np.asarray(coreg.compute_dig_mri_distances(), dtype=np.float64)
    retained = int(distances.size)
    if retained == 0:
        mean = median = p95 = maximum = float('nan')
    else:
        mean = float(np.mean(distances))
        median = float(np.median(distances))
        p95 = float(np.quantile(distances, 0.95))
        maximum = float(np.max(distances))
    return {
        'participant': int(subject),
        'session': int(session),
        'task': bids_task_token(task),
        'n_retained_hsp_points': retained,
        'mean_dig_mri_distance_m': mean,
        'median_dig_mri_distance_m': median,
        'p95_dig_mri_distance_m': p95,
        'max_dig_mri_distance_m': maximum,
    }


def fit_fsaverage_coregistration(info: Any, subjects_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    """Fit a participant-specific head to fsaverage transform."""

    import mne

    coreg = mne.coreg.Coregistration(
        info,
        FSAVERAGE_SUBJECT,
        subjects_dir=str(subjects_dir),
        fiducials='estimated',
    )
    coreg.fit_fiducials()
    coreg.fit_icp(n_iterations=20, nasion_weight=2.0)
    coreg.omit_head_shape_points(distance=0.010)
    coreg.fit_icp(n_iterations=30, nasion_weight=2.0)
    return coreg.trans, coreg


def coregistration_cache_stem(subject: int, session: int, task: int | str) -> str:
    return (
        f'sub-{bids_subject_token(subject)}_ses-{int(session)}'
        f'_task-{bids_task_token(task)}_geopolicy-{GEOMETRY_POLICY_VERSION}'
    )


def load_or_fit_fsaverage_coregistration(
    *,
    info: Any,
    subjects_dir: str | Path,
    subject: int,
    session: int,
    task: int | str,
    cache_dir: str | Path,
) -> ParticipantCoregistration:
    """Fit once per participant/session/task and reuse the saved transform."""

    import json

    import mne

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    stem = coregistration_cache_stem(subject, session, task)
    cache_path = cache_root / f'{stem}_trans-fsaverage.fif'
    qc_path = cache_root / f'{stem}_coreg_qc.json'
    trans_exists = cache_path.is_file()
    qc_exists = qc_path.is_file()
    if trans_exists != qc_exists:
        raise FileNotFoundError(
            'Incomplete coregistration cache for '
            f'{stem}: trans exists={trans_exists}, qc exists={qc_exists}.'
        )
    if trans_exists:
        trans = mne.read_trans(cache_path)
        qc = json.loads(qc_path.read_text(encoding='utf-8'))
        cached_version = qc.get('geometry_policy_version')
        if cached_version != GEOMETRY_POLICY_VERSION:
            trans_exists = False
        else:
            return ParticipantCoregistration(
                info=info,
                trans=trans,
                qc=qc,
                cache_path=cache_path,
            )

    trans, coreg = fit_fsaverage_coregistration(info, subjects_dir)
    qc = _coregistration_qc(
        subject=subject,
        session=session,
        task=task,
        coreg=coreg,
    )
    qc['geometry_policy_version'] = int(GEOMETRY_POLICY_VERSION)
    mne.write_trans(cache_path, trans, overwrite=True)
    qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True), encoding='utf-8')
    return ParticipantCoregistration(
        info=info,
        trans=trans,
        qc=qc,
        cache_path=cache_path,
    )


def _meg_channels_inside_bem(
    info: Any, trans: Any, bem: Any
) -> tuple[list[str], list[float]]:
    """Inside inner-skull MEG names and unsigned surface distances in metres."""

    import mne
    from mne.bem import _bem_find_surface
    from mne.surface import _CheckInside
    from mne.transforms import apply_trans

    picks = mne.pick_types(info, meg=True, ref_meg=False, exclude=())
    if len(picks) == 0:
        return [], []
    locs_mri = apply_trans(
        trans,
        apply_trans(
            info['dev_head_t'],
            np.asarray([info['chs'][i]['loc'][:3] for i in picks], dtype=np.float64),
        ),
    )
    inside = np.asarray(
        _CheckInside(_bem_find_surface(bem, 'inner_skull'))(locs_mri),
        dtype=bool,
    )
    names = [info['ch_names'][i] for i, flag in zip(picks, inside) if flag]
    if not names:
        return [], []
    depths_m = np.atleast_1d(
        np.asarray(
            mne.bem.distance_to_bem(locs_mri[inside], bem, trans=None, verbose=False),
            dtype=np.float64,
        )
    )
    return names, [float(depth) for depth in depths_m]


def prepare_participant_fsaverage_forward(
    *,
    template_info: Any,
    scaler_ch_names: Sequence[str],
    bids_root: str | Path,
    subject: int,
    session: int,
    task: int | str,
    src: Any,
    bem: Any,
    subjects_dir: str | Path,
    cache_dir: str | Path,
    n_jobs: int = 3,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Reconstruct Info, fit/load coregistration, and build one fsaverage forward."""

    import mne

    info, hpi_qc = _reconstruct_participant_kit_info(
        template_info=template_info,
        scaler_ch_names=scaler_ch_names,
        bids_root=bids_root,
        subject=subject,
        session=session,
        task=task,
    )
    coregistration = load_or_fit_fsaverage_coregistration(
        info=info,
        subjects_dir=subjects_dir,
        subject=subject,
        session=session,
        task=task,
        cache_dir=cache_dir,
    )
    meg_types = info.get_channel_types(picks='data', unique=True)
    if not meg_types:
        raise ValueError('Reconstructed Info has no MEG data channels.')
    inside_names, inside_depths_m = _meg_channels_inside_bem(
        info, coregistration.trans, bem
    )
    max_shallow_m = 2 * MM_TO_M
    too_many = len(inside_names) > 2
    too_deep = any(depth > max_shallow_m for depth in inside_depths_m)
    if too_many or too_deep:
        details = ', '.join(
            f'{name}={depth / MM_TO_M:.3f} mm'
            for name, depth in zip(inside_names, inside_depths_m)
        )
        raise RuntimeError(
            f'Found {len(inside_names)} MEG sensors inside the inner-skull '
            f'BEM ({details}). This is not a marginal surrogate-template contact.'
        )
    fwd_info = info
    if inside_names:
        # A marginally invalid sensor is excluded from surrogate-template
        # source localisation only.
        fwd_info = mne.pick_info(
            info,
            [i for i, name in enumerate(info['ch_names']) if name not in inside_names],
        )
    fwd = mne.make_forward_solution(
        fwd_info,
        trans=coregistration.trans,
        src=src,
        bem=bem,
        eeg=False,
        meg=meg_types[0],
        n_jobs=n_jobs,
        verbose=False,
    )
    qc = dict(coregistration.qc)
    qc.update(hpi_qc)
    qc['source_localization_excluded_channels'] = list(inside_names)
    qc['source_localization_excluded_depths_m'] = list(inside_depths_m)
    return info, coregistration.trans, fwd, qc
