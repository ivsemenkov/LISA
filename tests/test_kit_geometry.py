from pathlib import Path
import json
import sys
from types import SimpleNamespace

import mne
import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from lisa.data import kit_geometry  # noqa: E402


def _canonical_elp_metres() -> np.ndarray:
    return np.array(
        [
            [0.000, 0.100, 0.000],
            [-0.070, 0.000, 0.000],
            [0.070, 0.000, 0.000],
            [0.000, 0.090, 0.060],
            [-0.060, 0.040, 0.050],
            [0.060, 0.040, 0.050],
            [-0.040, -0.030, 0.070],
            [0.040, -0.030, 0.070],
        ],
        dtype=np.float64,
    )


def _corresponding_mrk(
    elp: np.ndarray,
    hpi_noise_m: np.ndarray | None = None,
) -> np.ndarray:
    from mne.io.kit.coreg import als_ras_trans
    from mne.transforms import apply_trans, get_ras_to_neuromag_trans

    elp = np.asarray(elp, dtype=np.float64)
    nmtrans = get_ras_to_neuromag_trans(*elp[:3])
    hpi_nm = apply_trans(nmtrans, elp[3:8])
    mrk_ras = hpi_nm
    if hpi_noise_m is not None:
        mrk_ras = hpi_nm + np.asarray(hpi_noise_m, dtype=np.float64)
    return apply_trans(np.linalg.inv(als_ras_trans), mrk_ras)


def _write_pos(path: Path, points: np.ndarray) -> None:
    lines = ['# millimetres']
    for row in points:
        lines.append(' '.join(f'{value:.3f}' for value in row))
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _bids_tree(tmp_path: Path, subject: int, session: int, task: int) -> Path:
    meg_dir = (
        tmp_path
        / f'sub-{subject:02d}'
        / f'ses-{session}'
        / 'meg'
    )
    meg_dir.mkdir(parents=True, exist_ok=True)
    elp = _canonical_elp_metres() * kit_geometry.M_TO_MM
    hsp = np.arange(30, dtype=np.float64).reshape(10, 3) + 100 * session
    _write_pos(
        meg_dir / f'sub-{subject:02d}_ses-{session}_acq-ELP_headshape.pos',
        elp,
    )
    _write_pos(
        meg_dir / f'sub-{subject:02d}_ses-{session}_acq-HSP_headshape.pos',
        hsp,
    )
    (meg_dir / f'sub-{subject:02d}_ses-{session}_task-{task}_markers.mrk').write_bytes(
        b'mrk'
    )
    return tmp_path


def test_resolves_session_elp_hsp_and_task_specific_marker(tmp_path: Path) -> None:
    bids_root = _bids_tree(tmp_path, subject=12, session=1, task=3)
    _bids_tree(tmp_path, subject=12, session=1, task=0)
    paths = kit_geometry.resolve_bids_digitization_paths(
        bids_root, subject=12, session=1, task=3
    )

    assert paths.elp.name == 'sub-12_ses-1_acq-ELP_headshape.pos'
    assert paths.hsp.name == 'sub-12_ses-1_acq-HSP_headshape.pos'
    assert paths.mrk.name == 'sub-12_ses-1_task-3_markers.mrk'
    other = kit_geometry.resolve_bids_digitization_paths(
        bids_root, subject=12, session=1, task=0
    )
    assert other.mrk.name == 'sub-12_ses-1_task-0_markers.mrk'
    assert paths.mrk != other.mrk
    assert paths.elp == other.elp


def test_missing_task_marker_is_rejected(tmp_path: Path) -> None:
    bids_root = _bids_tree(tmp_path, subject=1, session=0, task=1)
    with pytest.raises(FileNotFoundError, match='task-2'):
        kit_geometry.resolve_bids_digitization_paths(
            bids_root, subject=1, session=0, task=2
        )


def test_pos_files_are_converted_from_millimetres_to_metres(tmp_path: Path) -> None:
    path = tmp_path / 'points.pos'
    _write_pos(path, np.array([[100.0, 0.0, 50.0], [0.0, 20.0, 0.0]]))
    metres = kit_geometry.read_kit_pos_metres(path)
    np.testing.assert_allclose(metres, [[0.1, 0.0, 0.05], [0.0, 0.02, 0.0]])


def test_channel_reorder_and_mismatch_rejection() -> None:
    info = mne.create_info(['MEG 003', 'MEG 001', 'MEG 002'], sfreq=1000.0, ch_types='mag')
    reordered = kit_geometry.reorder_info_to_channel_names(
        info, ['MEG 001', 'MEG 002', 'MEG 003']
    )
    assert list(reordered['ch_names']) == ['MEG 001', 'MEG 002', 'MEG 003']

    with pytest.raises(ValueError, match='Channel mismatch'):
        kit_geometry.reorder_info_to_channel_names(info, ['MEG 001', 'MEG 004'])


def test_reconstructed_info_replaces_fif_dig_with_participant_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bids_root = _bids_tree(tmp_path, subject=2, session=0, task=1)
    template = mne.create_info(['MEG 001', 'MEG 002'], sfreq=1000.0, ch_types='mag')
    with template._unlock():
        template['dig'] = [{'kind': 'from-fif'}]
        template['dev_head_t'] = mne.transforms.Transform('meg', 'head', np.eye(4))

    captured = {}
    elp = _canonical_elp_metres()
    mrk = _corresponding_mrk(elp)
    read_paths: list[str] = []

    def fake_read_mrk(path):
        read_paths.append(str(path))
        return mrk.copy()

    def fake_set_dig_kit(mrk_points, elp_points, hsp, eeg, *, bad_coils=()):
        captured['mrk'] = np.asarray(mrk_points)
        captured['elp'] = np.asarray(elp_points)
        captured['hsp'] = np.asarray(hsp)
        captured['eeg'] = eeg
        captured['bad_coils'] = bad_coils
        trans = np.eye(4)
        trans[0, 3] = 0.02
        return (
            [{'kind': 'reconstructed', 'subject': 2}],
            mne.transforms.Transform('meg', 'head', trans),
            None,
        )

    monkeypatch.setattr(kit_geometry, '_read_kit_mrk', fake_read_mrk)
    monkeypatch.setattr('mne.io.kit.coreg._set_dig_kit', fake_set_dig_kit)

    info = kit_geometry.reconstruct_participant_kit_info(
        template_info=template,
        scaler_ch_names=['MEG 001', 'MEG 002'],
        bids_root=bids_root,
        subject=2,
        session=0,
        task=1,
    )

    assert info['dig'] == [{'kind': 'reconstructed', 'subject': 2}]
    np.testing.assert_allclose(info['dev_head_t']['trans'][0, 3], 0.02)
    assert captured['eeg'] == {}
    assert captured['bad_coils'] == ()
    assert captured['mrk'].shape == (5, 3)
    np.testing.assert_allclose(captured['mrk'], mrk)
    np.testing.assert_allclose(captured['elp'][0], [0.0, 0.1, 0.0])
    assert read_paths[0].endswith('task-1_markers.mrk')


def test_different_participants_receive_different_transforms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bids_tree(tmp_path, subject=1, session=0, task=1)
    _bids_tree(tmp_path, subject=4, session=0, task=1)
    template = mne.create_info(['MEG 001'], sfreq=1000.0, ch_types='mag')
    elp = _canonical_elp_metres()
    mrk = _corresponding_mrk(elp)
    monkeypatch.setattr(kit_geometry, '_read_kit_mrk', lambda _path: mrk.copy())
    n_calls = []

    def fake_set_dig_kit(mrk_points, elp_points, hsp, eeg, *, bad_coils=()):
        trans = np.eye(4)
        trans[2, 3] = 0.01 * (len(n_calls) + 1)
        n_calls.append(np.asarray(mrk_points).copy())
        return (
            [{'call': len(n_calls)}],
            mne.transforms.Transform('meg', 'head', trans),
            None,
        )

    monkeypatch.setattr('mne.io.kit.coreg._set_dig_kit', fake_set_dig_kit)
    first = kit_geometry.reconstruct_participant_kit_info(
        template_info=template,
        scaler_ch_names=['MEG 001'],
        bids_root=tmp_path,
        subject=1,
        session=0,
        task=1,
    )
    second = kit_geometry.reconstruct_participant_kit_info(
        template_info=template,
        scaler_ch_names=['MEG 001'],
        bids_root=tmp_path,
        subject=4,
        session=0,
        task=1,
    )
    assert not np.allclose(first['dev_head_t']['trans'], second['dev_head_t']['trans'])
    assert first['dig'] != second['dig']


def test_prepare_participant_forward_is_not_shared(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _bids_tree(tmp_path, subject=1, session=0, task=1)
    _bids_tree(tmp_path, subject=2, session=0, task=1)
    template = mne.create_info(['MEG 001'], sfreq=1000.0, ch_types='mag')
    forwards = []

    def fake_reconstruct(**kwargs):
        info = template.copy()
        trans = np.eye(4)
        trans[1, 3] = float(kwargs['subject'])
        with info._unlock():
            info['dev_head_t'] = mne.transforms.Transform('meg', 'head', trans)
            info['dig'] = [{'subject': kwargs['subject']}]
        return info

    def fake_coreg(**kwargs):
        return kit_geometry.ParticipantCoregistration(
            info=kwargs['info'],
            trans=kwargs['info']['dev_head_t'],
            qc={'participant': kwargs['subject']},
            cache_path=tmp_path / 'unused_trans.fif',
        )

    def fake_forward(info, **kwargs):
        fwd = SimpleNamespace(info=info, trans=kwargs['trans'], id=id(info))
        forwards.append(fwd)
        return fwd

    monkeypatch.setattr(
        kit_geometry,
        '_reconstruct_participant_kit_info',
        lambda **kwargs: (fake_reconstruct(**kwargs), {}),
    )
    monkeypatch.setattr(
        kit_geometry,
        'load_or_fit_fsaverage_coregistration',
        lambda **kwargs: fake_coreg(**kwargs),
    )
    monkeypatch.setattr(
        kit_geometry,
        '_meg_channels_inside_bem',
        lambda *_args, **_kwargs: ([], []),
    )
    monkeypatch.setattr(mne, 'make_forward_solution', fake_forward)

    first = kit_geometry.prepare_participant_fsaverage_forward(
        template_info=template,
        scaler_ch_names=['MEG 001'],
        bids_root=tmp_path,
        subject=1,
        session=0,
        task=1,
        src=object(),
        bem=object(),
        subjects_dir=tmp_path,
        cache_dir=tmp_path / 'geometry_cache',
    )
    second = kit_geometry.prepare_participant_fsaverage_forward(
        template_info=template,
        scaler_ch_names=['MEG 001'],
        bids_root=tmp_path,
        subject=2,
        session=0,
        task=1,
        src=object(),
        bem=object(),
        subjects_dir=tmp_path,
        cache_dir=tmp_path / 'geometry_cache',
    )

    assert first[2] is not second[2]
    assert first[1] is not second[1]
    assert first[3]['participant'] != second[3]['participant']
    assert len(forwards) == 2


def test_inside_bem_sensor_is_omitted_from_forward_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _bids_tree(tmp_path, subject=1, session=0, task=1)
    names = ['MEG 001', 'MEG 002', 'MEG 003']
    template = mne.create_info(names, sfreq=1000.0, ch_types='mag')
    captured = {}

    def fake_reconstruct(**kwargs):
        return template.copy()

    def fake_coreg(**kwargs):
        return kit_geometry.ParticipantCoregistration(
            info=kwargs['info'],
            trans=kwargs['info']['dev_head_t'],
            qc={'participant': kwargs['subject']},
            cache_path=tmp_path / 'unused_trans.fif',
        )

    def fake_forward(info, **kwargs):
        captured['ch_names'] = list(info['ch_names'])
        return SimpleNamespace(info=info, trans=kwargs['trans'])

    monkeypatch.setattr(
        kit_geometry,
        '_reconstruct_participant_kit_info',
        lambda **kwargs: (fake_reconstruct(**kwargs), {}),
    )
    monkeypatch.setattr(
        kit_geometry,
        'load_or_fit_fsaverage_coregistration',
        lambda **kwargs: fake_coreg(**kwargs),
    )
    monkeypatch.setattr(
        kit_geometry,
        '_meg_channels_inside_bem',
        lambda *_args, **_kwargs: (['MEG 002'], [0.00043]),
    )
    monkeypatch.setattr(mne, 'make_forward_solution', fake_forward)

    info, _trans, _fwd, qc = kit_geometry.prepare_participant_fsaverage_forward(
        template_info=template,
        scaler_ch_names=names,
        bids_root=tmp_path,
        subject=1,
        session=0,
        task=1,
        src=object(),
        bem=object(),
        subjects_dir=tmp_path,
        cache_dir=tmp_path / 'geometry_cache',
    )

    assert list(info['ch_names']) == names
    assert captured['ch_names'] == ['MEG 001', 'MEG 003']
    assert qc['source_localization_excluded_channels'] == ['MEG 002']
    assert qc['source_localization_excluded_depths_m'] == pytest.approx([0.00043])
    assert qc['participant'] == 1


def test_many_inside_bem_sensors_still_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _bids_tree(tmp_path, subject=1, session=0, task=1)
    names = ['MEG 001', 'MEG 002', 'MEG 003']
    template = mne.create_info(names, sfreq=1000.0, ch_types='mag')

    monkeypatch.setattr(
        kit_geometry,
        '_reconstruct_participant_kit_info',
        lambda **kwargs: (template.copy(), {}),
    )
    monkeypatch.setattr(
        kit_geometry,
        'load_or_fit_fsaverage_coregistration',
        lambda **kwargs: kit_geometry.ParticipantCoregistration(
            info=kwargs['info'],
            trans=kwargs['info']['dev_head_t'],
            qc={'participant': kwargs['subject']},
            cache_path=tmp_path / 'unused_trans.fif',
        ),
    )
    monkeypatch.setattr(
        kit_geometry,
        '_meg_channels_inside_bem',
        lambda *_args, **_kwargs: (names, [0.0004, 0.0004, 0.0004]),
    )
    monkeypatch.setattr(
        mne,
        'make_forward_solution',
        lambda *_args, **_kwargs: pytest.fail('forward should not run'),
    )

    with pytest.raises(RuntimeError, match='not a marginal surrogate-template'):
        kit_geometry.prepare_participant_fsaverage_forward(
            template_info=template,
            scaler_ch_names=names,
            bids_root=tmp_path,
            subject=1,
            session=0,
            task=1,
            src=object(),
            bem=object(),
            subjects_dir=tmp_path,
            cache_dir=tmp_path / 'geometry_cache',
        )


def test_deep_inside_bem_sensor_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _bids_tree(tmp_path, subject=1, session=0, task=1)
    names = ['MEG 001', 'MEG 002', 'MEG 003']
    template = mne.create_info(names, sfreq=1000.0, ch_types='mag')

    monkeypatch.setattr(
        kit_geometry,
        '_reconstruct_participant_kit_info',
        lambda **kwargs: (template.copy(), {}),
    )
    monkeypatch.setattr(
        kit_geometry,
        'load_or_fit_fsaverage_coregistration',
        lambda **kwargs: kit_geometry.ParticipantCoregistration(
            info=kwargs['info'],
            trans=kwargs['info']['dev_head_t'],
            qc={'participant': kwargs['subject']},
            cache_path=tmp_path / 'unused_trans.fif',
        ),
    )
    monkeypatch.setattr(
        kit_geometry,
        '_meg_channels_inside_bem',
        lambda *_args, **_kwargs: (['MEG 002'], [0.003]),
    )
    monkeypatch.setattr(
        mne,
        'make_forward_solution',
        lambda *_args, **_kwargs: pytest.fail('forward should not run'),
    )

    with pytest.raises(RuntimeError, match='not a marginal surrogate-template'):
        kit_geometry.prepare_participant_fsaverage_forward(
            template_info=template,
            scaler_ch_names=names,
            bids_root=tmp_path,
            subject=1,
            session=0,
            task=1,
            src=object(),
            bem=object(),
            subjects_dir=tmp_path,
            cache_dir=tmp_path / 'geometry_cache',
        )


def test_source_localisation_requires_participant_bids_geometry() -> None:
    import inspect

    from lisa.plots.branch_interpretation import extract_item_table_from_model
    from lisa.plots.plot_spatial_topography_svd import save_dipole_outputs

    parameters = inspect.signature(save_dipole_outputs).parameters
    assert parameters['bids_root'].default is inspect.Parameter.empty
    assert parameters['geometry_cache_dir'].default is inspect.Parameter.empty

    extract = inspect.signature(extract_item_table_from_model).parameters
    assert extract['raw_bids_root'].default is inspect.Parameter.empty
    assert extract['geometry_cache_dir'].default is inspect.Parameter.empty

    forward = inspect.signature(
        kit_geometry.prepare_participant_fsaverage_forward
    ).parameters
    assert forward['bids_root'].default is inspect.Parameter.empty
    assert forward['cache_dir'].default is inspect.Parameter.empty

    coreg = inspect.signature(
        kit_geometry.load_or_fit_fsaverage_coregistration
    ).parameters
    assert coreg['cache_dir'].default is inspect.Parameter.empty


def test_identity_hpi_correspondence_is_preserved() -> None:
    elp = _canonical_elp_metres()
    mrk = _corresponding_mrk(elp)
    result = kit_geometry.resolve_hpi_correspondence(mrk, elp)

    assert result.correction_mode == 'identity'
    assert result.permutation == (0, 1, 2, 3, 4)
    assert result.rejected_coil is None
    assert result.n_retained_coils == 5
    np.testing.assert_allclose(result.mrk, mrk)
    np.testing.assert_allclose(result.elp, elp)
    assert result.original_rms_mm < 1.0
    assert result.final_rms_mm == pytest.approx(result.original_rms_mm)


def test_sub20_like_hpi_permutation_is_repaired() -> None:
    elp = _canonical_elp_metres()
    mrk = _corresponding_mrk(elp)
    scrambled = mrk.copy()
    scrambled[[3, 4]] = mrk[[4, 3]]
    result = kit_geometry.resolve_hpi_correspondence(
        scrambled, elp, subject=20, session=0, task=1
    )

    assert result.correction_mode == 'permutation'
    assert result.permutation == (0, 1, 2, 4, 3)
    assert result.rejected_coil is None
    assert result.n_retained_coils == 5
    np.testing.assert_allclose(result.mrk, mrk)
    np.testing.assert_allclose(result.elp, elp)
    assert result.original_rms_mm >= kit_geometry.HPI_SEVERE_RMS_MM
    assert result.final_rms_mm < 2.0
    assert result.final_rms_mm < result.original_rms_mm / 4.0


def test_one_bad_hpi_coil_is_dropped_from_mrk_and_elp() -> None:
    elp = _canonical_elp_metres()
    mrk = _corresponding_mrk(elp)
    broken = mrk.copy()
    broken[1] = broken[1] + np.array([0.080, 0.0, 0.0])
    result = kit_geometry.resolve_hpi_correspondence(
        broken, elp, subject=11, session=1, task=0
    )

    assert result.correction_mode == 'bad_coil'
    assert result.permutation == (0, 1, 2, 3, 4)
    assert result.rejected_coil == 1
    assert result.n_retained_coils == 4
    assert result.mrk.shape == (4, 3)
    assert result.elp.shape == (7, 3)
    np.testing.assert_allclose(result.elp[:3], elp[:3])
    np.testing.assert_allclose(result.mrk, broken[[0, 2, 3, 4]])
    np.testing.assert_allclose(result.elp[3:], elp[3:][[0, 2, 3, 4]])
    assert result.original_rms_mm >= kit_geometry.HPI_SEVERE_RMS_MM
    assert result.final_rms_mm <= kit_geometry.HPI_BAD_COIL_PLAUSIBLE_RMS_MM


def test_modest_leave_one_out_improvement_is_not_applied() -> None:
    elp = _canonical_elp_metres()
    noise = np.zeros((5, 3), dtype=np.float64)
    noise[1] = np.array([0.018, 0.0, 0.0])
    mrk = _corresponding_mrk(elp, hpi_noise_m=noise)
    result = kit_geometry.resolve_hpi_correspondence(
        mrk, elp, subject=4, session=1, task=2
    )

    assert result.original_rms_mm < kit_geometry.HPI_SEVERE_RMS_MM
    assert result.correction_mode == 'identity'
    assert result.rejected_coil is None
    assert result.permutation == (0, 1, 2, 3, 4)
    np.testing.assert_allclose(result.mrk, mrk)


def test_unresolved_severe_hpi_fails_closed() -> None:
    elp = _canonical_elp_metres()
    mrk = _corresponding_mrk(elp)
    broken = mrk.copy()
    broken[1] = broken[1] + np.array([0.080, 0.0, 0.0])
    broken[2] = broken[2] + np.array([0.0, 0.080, 0.0])

    with pytest.raises(
        kit_geometry.UnresolvedHPIGeometryError,
        match='automatic correction was refused',
    ) as raised:
        kit_geometry.resolve_hpi_correspondence(
            broken, elp, subject=99, session=0, task=1
        )

    message = str(raised.value)
    assert 'sub-99/ses-0/task-1' in message
    assert 'identity RMS' in message
    assert 'best permutation' in message
    assert 'best leave-one-out' in message
    assert 'per-coil residuals' in message


def test_hpi_qc_propagates_through_prepare_participant_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bids_tree(tmp_path, subject=1, session=0, task=1)
    template = mne.create_info(['MEG 001'], sfreq=1000.0, ch_types='mag')
    elp = _canonical_elp_metres()
    mrk = _corresponding_mrk(elp)
    monkeypatch.setattr(kit_geometry, '_read_kit_mrk', lambda _path: mrk.copy())

    def fake_set_dig_kit(mrk_points, elp_points, hsp, eeg, *, bad_coils=()):
        trans = np.eye(4)
        trans[0, 3] = 0.03
        return (
            [{'kind': 'reconstructed'}],
            mne.transforms.Transform('meg', 'head', trans),
            None,
        )

    monkeypatch.setattr('mne.io.kit.coreg._set_dig_kit', fake_set_dig_kit)
    monkeypatch.setattr(
        kit_geometry,
        'load_or_fit_fsaverage_coregistration',
        lambda **kwargs: kit_geometry.ParticipantCoregistration(
            info=kwargs['info'],
            trans=kwargs['info']['dev_head_t'],
            qc={'participant': kwargs['subject']},
            cache_path=tmp_path / 'unused_trans.fif',
        ),
    )
    monkeypatch.setattr(
        kit_geometry,
        '_meg_channels_inside_bem',
        lambda *_args, **_kwargs: ([], []),
    )
    monkeypatch.setattr(
        mne,
        'make_forward_solution',
        lambda info, **kwargs: SimpleNamespace(info=info, trans=kwargs['trans']),
    )

    _info, _trans, _fwd, qc = kit_geometry.prepare_participant_fsaverage_forward(
        template_info=template,
        scaler_ch_names=['MEG 001'],
        bids_root=tmp_path,
        subject=1,
        session=0,
        task=1,
        src=object(),
        bem=object(),
        subjects_dir=tmp_path,
        cache_dir=tmp_path / 'geometry_cache',
    )

    assert qc['participant'] == 1
    assert qc['hpi_correction_mode'] == 'identity'
    assert qc['hpi_mrk_permutation'] == [0, 1, 2, 3, 4]
    assert qc['hpi_rejected_coil'] is None
    assert qc['hpi_n_retained_coils'] == 5
    assert qc['geometry_policy_version'] == kit_geometry.GEOMETRY_POLICY_VERSION
    assert len(qc['hpi_original_coil_residuals_mm']) == 5
    assert len(qc['hpi_final_retained_coil_residuals_mm']) == 5
    assert qc['hpi_original_rms_mm'] == pytest.approx(qc['hpi_final_rms_mm'])


def test_old_geometry_policy_cache_is_not_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = mne.create_info(['MEG 001'], sfreq=1000.0, ch_types='mag')
    old_trans = np.eye(4)
    old_trans[0, 3] = 0.111
    mne.write_trans(
        tmp_path / 'sub-01_ses-0_task-1_trans-fsaverage.fif',
        mne.transforms.Transform('head', 'mri', old_trans),
        overwrite=True,
    )
    (tmp_path / 'sub-01_ses-0_task-1_coreg_qc.json').write_text(
        json.dumps({'participant': 1, 'geometry_policy_version': 1}),
        encoding='utf-8',
    )
    fit_calls = []
    new_trans = np.eye(4)
    new_trans[1, 3] = 0.222

    def fake_fit(_info, _subjects_dir):
        fit_calls.append(1)
        trans = mne.transforms.Transform('head', 'mri', new_trans)
        coreg = SimpleNamespace(
            compute_dig_mri_distances=lambda: np.array([0.001, 0.002])
        )
        return trans, coreg

    monkeypatch.setattr(kit_geometry, 'fit_fsaverage_coregistration', fake_fit)
    first = kit_geometry.load_or_fit_fsaverage_coregistration(
        info=info,
        subjects_dir=tmp_path,
        subject=1,
        session=0,
        task=1,
        cache_dir=tmp_path,
    )
    second = kit_geometry.load_or_fit_fsaverage_coregistration(
        info=info,
        subjects_dir=tmp_path,
        subject=1,
        session=0,
        task=1,
        cache_dir=tmp_path,
    )

    assert len(fit_calls) == 1
    np.testing.assert_allclose(first.trans['trans'][1, 3], 0.222)
    np.testing.assert_allclose(second.trans['trans'][1, 3], 0.222)
    assert 'geopolicy-2' in first.cache_path.name
    assert first.qc['geometry_policy_version'] == kit_geometry.GEOMETRY_POLICY_VERSION
    assert kit_geometry.GEOMETRY_POLICY_VERSION == 2


def test_versioned_cache_with_stale_policy_json_is_not_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = mne.create_info(['MEG 001'], sfreq=1000.0, ch_types='mag')
    stem = kit_geometry.coregistration_cache_stem(1, 0, 1)
    stale_trans = np.eye(4)
    stale_trans[0, 3] = 0.333
    mne.write_trans(
        tmp_path / f'{stem}_trans-fsaverage.fif',
        mne.transforms.Transform('head', 'mri', stale_trans),
        overwrite=True,
    )
    (tmp_path / f'{stem}_coreg_qc.json').write_text(
        json.dumps({'participant': 1, 'geometry_policy_version': 1}),
        encoding='utf-8',
    )
    new_trans = np.eye(4)
    new_trans[2, 3] = 0.444

    def fake_fit(_info, _subjects_dir):
        trans = mne.transforms.Transform('head', 'mri', new_trans)
        coreg = SimpleNamespace(
            compute_dig_mri_distances=lambda: np.array([0.001])
        )
        return trans, coreg

    monkeypatch.setattr(kit_geometry, 'fit_fsaverage_coregistration', fake_fit)
    result = kit_geometry.load_or_fit_fsaverage_coregistration(
        info=info,
        subjects_dir=tmp_path,
        subject=1,
        session=0,
        task=1,
        cache_dir=tmp_path,
    )

    np.testing.assert_allclose(result.trans['trans'][2, 3], 0.444)
    assert result.qc['geometry_policy_version'] == kit_geometry.GEOMETRY_POLICY_VERSION


def test_bad_coil_arrays_are_passed_to_set_dig_kit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bids_tree(tmp_path, subject=11, session=1, task=0)
    template = mne.create_info(['MEG 001'], sfreq=1000.0, ch_types='mag')
    elp = _canonical_elp_metres()
    mrk = _corresponding_mrk(elp)
    broken = mrk.copy()
    broken[1] = broken[1] + np.array([0.080, 0.0, 0.0])
    captured = {}

    def fake_set_dig_kit(mrk_points, elp_points, hsp, eeg, *, bad_coils=()):
        captured['mrk'] = np.asarray(mrk_points)
        captured['elp'] = np.asarray(elp_points)
        captured['bad_coils'] = bad_coils
        return (
            [{'kind': 'reconstructed'}],
            mne.transforms.Transform('meg', 'head', np.eye(4)),
            None,
        )

    monkeypatch.setattr(kit_geometry, '_read_kit_mrk', lambda _path: broken.copy())
    monkeypatch.setattr('mne.io.kit.coreg._set_dig_kit', fake_set_dig_kit)

    kit_geometry.reconstruct_participant_kit_info(
        template_info=template,
        scaler_ch_names=['MEG 001'],
        bids_root=tmp_path,
        subject=11,
        session=1,
        task=0,
    )

    assert captured['bad_coils'] == ()
    assert captured['mrk'].shape == (4, 3)
    assert captured['elp'].shape == (7, 3)
    np.testing.assert_allclose(captured['elp'][:3], elp[:3])
    np.testing.assert_allclose(captured['mrk'], broken[[0, 2, 3, 4]])
    np.testing.assert_allclose(captured['elp'][3:], elp[3:][[0, 2, 3, 4]])
