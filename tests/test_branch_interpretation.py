from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from lisa.plots.branch_interpretation import (  # noqa: E402
    calculate_spatial_patterns,
    calculate_temporal_patterns,
    load_subject_meg_and_raw,
)
from lisa.plots import branch_interpretation  # noqa: E402
from lisa.plots import source_estimation  # noqa: E402
from lisa.data import meg_io  # noqa: E402


pytestmark = pytest.mark.filterwarnings(
    'ignore:(divide by zero|overflow|invalid value) encountered in matmul:RuntimeWarning'
)


@pytest.fixture
def pattern_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(234)
    return {
        'meg': rng.normal(size=(9, 120)),
        'spatial_filters_weight': rng.normal(size=(2, 4, 9)),
        'temporal_filters': rng.normal(size=(4, 11)),
        'spatial_filters_bias': rng.normal(size=(2, 4, 1)),
    }


@pytest.mark.parametrize('filtfilt', [False, True])
def test_spatial_patterns_are_finite(
    pattern_inputs: dict[str, np.ndarray],
    filtfilt: bool,
) -> None:
    patterns, covariances = calculate_spatial_patterns(
        meg=pattern_inputs['meg'],
        spatial_filters_weight=pattern_inputs['spatial_filters_weight'],
        temporal_filters=pattern_inputs['temporal_filters'],
        sub=2,
        filtfilt=filtfilt,
        return_covariances=True,
    )

    assert patterns.shape == (4, 9)
    assert covariances.shape == (4, 9, 9)
    assert np.isfinite(patterns).all()
    assert np.isfinite(covariances).all()


@pytest.mark.parametrize('with_bias', [False, True])
def test_temporal_patterns_are_finite(
    pattern_inputs: dict[str, np.ndarray],
    with_bias: bool,
) -> None:
    bias = pattern_inputs['spatial_filters_bias'] if with_bias else None
    patterns, covariances = calculate_temporal_patterns(
        meg=pattern_inputs['meg'],
        spatial_filters_weight=pattern_inputs['spatial_filters_weight'],
        temporal_filters=pattern_inputs['temporal_filters'],
        sub=2,
        spatial_filters_bias=bias,
        return_covariances=True,
    )

    assert patterns.shape == (4, 11)
    assert covariances.shape == (4, 11, 11)
    assert np.isfinite(patterns).all()
    assert np.isfinite(covariances).all()


def test_get_stc_reuses_prepared_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry = source_estimation.PreparedSourceGeometry(
        subject='fsaverage',
        subjects_dir='/subjects',
        src=object(),
        bem=object(),
        trans='/coords/trans.fif',
    )
    raw = SimpleNamespace(
        info=object(),
        get_channel_types=lambda **_kwargs: ['mag'],
    )
    expected_stc = object()
    calls = {}

    def fake_forward(info, **kwargs):
        calls['forward'] = (info, kwargs)
        return object()

    monkeypatch.setattr(
        source_estimation,
        'prepare_source_geometry',
        lambda: pytest.fail('prepared geometry should not be rebuilt'),
    )
    monkeypatch.setattr(source_estimation.mne, 'make_forward_solution', fake_forward)
    monkeypatch.setattr(
        source_estimation.mne, 'make_ad_hoc_cov', lambda _info: object()
    )
    monkeypatch.setattr(
        source_estimation,
        'make_inverse_operator',
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        source_estimation.mne,
        'EvokedArray',
        lambda data, info: (data, info),
    )
    monkeypatch.setattr(
        source_estimation,
        'apply_inverse',
        lambda *_args, **_kwargs: expected_stc,
    )

    result = source_estimation.get_stc(
        raw=raw,
        spatial_patterns=np.ones((3, 5), dtype=np.float64),
        prepared_geometry=geometry,
    )

    assert result is expected_stc
    _, forward_kwargs = calls['forward']
    assert forward_kwargs['src'] is geometry.src
    assert forward_kwargs['bem'] is geometry.bem
    assert forward_kwargs['trans'] == geometry.trans


@pytest.mark.parametrize(
    ('raw_metadata_only', 'expected_preload', 'expected_resample_calls'),
    [(True, False, 0), (False, True, 1)],
)
def test_preprocessed_meg_loading_controls_raw_data_work(
    monkeypatch: pytest.MonkeyPatch,
    raw_metadata_only: bool,
    expected_preload: bool,
    expected_resample_calls: int,
) -> None:
    expected_meg = np.arange(12, dtype=np.float32).reshape(3, 4)
    calls = {}

    class FakeRaw:
        ch_names = ['MEG1', 'MEG2', 'MEG3']

        def __init__(self):
            self.resample_calls = []

        def resample(self, sfreq, *, npad):
            self.resample_calls.append((sfreq, npad))
            return self

    raw = FakeRaw()

    def fake_load_fixed_meg_batch(**kwargs):
        calls['meg_kwargs'] = kwargs
        return expected_meg

    monkeypatch.setattr(
        branch_interpretation,
        'load_fixed_meg_batch',
        fake_load_fixed_meg_batch,
    )

    def fake_load_raw_meg(**kwargs):
        calls['raw_kwargs'] = kwargs
        return raw, 1000.0

    monkeypatch.setattr(
        meg_io,
        'load_raw_meg',
        fake_load_raw_meg,
    )

    meg, observed_raw = load_subject_meg_and_raw(
        sub=3,
        ses=0,
        story_id=1,
        target_fs=100.0,
        offset_gap=10.0,
        meg_format='fif',
        data_root='/clean',
        raw_meg=False,
        preprocessed_meg_path='/preprocessed/meg.npz',
        preprocess=True,
        raw_metadata_only=raw_metadata_only,
    )

    assert observed_raw is raw
    assert meg.dtype == np.float64
    np.testing.assert_array_equal(meg, expected_meg)
    assert calls['raw_kwargs']['preload'] is expected_preload
    assert calls['raw_kwargs']['data_root'] == '/clean'
    assert len(raw.resample_calls) == expected_resample_calls
    if expected_resample_calls:
        assert raw.resample_calls == [(100.0, 'auto')]


@pytest.mark.parametrize(
    ('preload', 'expected_load_calls'),
    [(False, 0), (True, 1)],
)
def test_load_raw_meg_preload_control(
    monkeypatch: pytest.MonkeyPatch,
    preload: bool,
    expected_load_calls: int,
) -> None:
    class FakeRaw:
        def __init__(self):
            self.info = {'sfreq': 1000.0}
            self.load_calls = 0
            self.picks = None

        def pick(self, picks):
            self.picks = picks
            return self

        def load_data(self, *, verbose):
            assert verbose is False
            self.load_calls += 1
            return self

    raw = FakeRaw()
    monkeypatch.setattr(
        meg_io.mne.io,
        'read_raw_fif',
        lambda *_args, **_kwargs: raw,
    )
    monkeypatch.setattr(
        meg_io.mne,
        'pick_types',
        lambda *_args, **_kwargs: np.array([0, 2], dtype=int),
    )

    observed_raw, orig_fs = meg_io.load_raw_meg(
        meg_format='fif',
        data_root='/clean',
        sub=3,
        ses=0,
        story_id=1,
        preload=preload,
    )

    assert observed_raw is raw
    assert orig_fs == 1000.0
    np.testing.assert_array_equal(raw.picks, [0, 2])
    assert raw.load_calls == expected_load_calls
