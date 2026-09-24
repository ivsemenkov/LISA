from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from lisa.plots.branch_interpretation import (  # noqa: E402
    assert_valid_composed_scaling,
    calculate_spatial_patterns,
    calculate_temporal_patterns,
    compute_item_spatial_roughness,
    load_subject_meg_and_raw,
    normalize_source_magnitudes,
    physical_spatial_affine,
)
from lisa.plots import branch_interpretation  # noqa: E402
from lisa.plots import source_estimation  # noqa: E402
from lisa.plots.temporal_filter_utils import filter_data  # noqa: E402
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
        channel_pad=np.zeros(pattern_inputs['meg'].shape[0]),
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
    )
    info = SimpleNamespace(
        ch_names=['MEG1'],
        get_channel_types=lambda **_kwargs: ['mag'],
    )
    trans = object()
    expected_stc = object()
    calls = {}

    def fake_forward(observed_info, **kwargs):
        calls['forward'] = (observed_info, kwargs)
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
        lambda data, observed_info: (data, observed_info),
    )
    monkeypatch.setattr(
        source_estimation,
        'apply_inverse',
        lambda *_args, **_kwargs: expected_stc,
    )

    result = source_estimation.get_stc(
        spatial_patterns=np.ones((3, 5), dtype=np.float64),
        info=info,
        trans=trans,
        prepared_geometry=geometry,
    )

    assert result is expected_stc
    observed_info, forward_kwargs = calls['forward']
    assert observed_info is info
    assert forward_kwargs['src'] is geometry.src
    assert forward_kwargs['bem'] is geometry.bem
    assert forward_kwargs['trans'] is trans
    assert 'trans' not in source_estimation.PreparedSourceGeometry.__dataclass_fields__


def test_get_stc_uses_supplied_forward_and_inverse_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = source_estimation.PreparedSourceGeometry(
        subject='fsaverage',
        subjects_dir='/subjects',
        src=object(),
        bem=object(),
    )
    info = SimpleNamespace(get_channel_types=lambda **_kwargs: ['mag'])
    supplied_fwd = object()
    inverse_kwargs = {}

    monkeypatch.setattr(
        source_estimation.mne,
        'make_forward_solution',
        lambda *_args, **_kwargs: pytest.fail('forward should be reused'),
    )
    monkeypatch.setattr(
        source_estimation.mne, 'make_ad_hoc_cov', lambda _info: object()
    )

    def fake_inverse(observed_info, fwd, _cov, **kwargs):
        inverse_kwargs['fwd'] = fwd
        inverse_kwargs['info'] = observed_info
        inverse_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(source_estimation, 'make_inverse_operator', fake_inverse)
    monkeypatch.setattr(
        source_estimation.mne, 'EvokedArray', lambda data, observed_info: data
    )
    monkeypatch.setattr(
        source_estimation,
        'apply_inverse',
        lambda *_args, **kwargs: kwargs,
    )

    apply_kwargs = source_estimation.get_stc(
        spatial_patterns=np.ones((2, 3), dtype=np.float64),
        info=info,
        trans=object(),
        prepared_geometry=geometry,
        fwd=supplied_fwd,
    )

    assert inverse_kwargs['fwd'] is supplied_fwd
    assert inverse_kwargs['info'] is info
    assert inverse_kwargs['loose'] == 0.5
    assert inverse_kwargs['depth'] == 0.5
    assert apply_kwargs['lambda2'] == pytest.approx(1.0 / 3)
    assert apply_kwargs['method'] == 'MNE'


def test_prepare_source_geometry_uses_single_layer_ico4_bem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    monkeypatch.setattr(
        source_estimation,
        'fetch_fsaverage',
        lambda verbose=False: Path('/subjects/fsaverage'),
    )
    monkeypatch.setattr(
        source_estimation.mne,
        'setup_source_space',
        lambda *_args, **kwargs: {'spacing': kwargs.get('spacing')},
    )

    def fake_bem_model(**kwargs):
        captured['bem'] = kwargs
        return object()

    monkeypatch.setattr(source_estimation.mne, 'make_bem_model', fake_bem_model)
    monkeypatch.setattr(
        source_estimation.mne, 'make_bem_solution', lambda model, **_kwargs: model
    )

    geometry = source_estimation.prepare_source_geometry()

    assert geometry.subject == 'fsaverage'
    assert geometry.subjects_dir == str(Path('/subjects'))
    assert captured['bem']['ico'] == 4
    assert captured['bem']['conductivity'] == (0.3,)
    assert not hasattr(geometry, 'trans')


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
        n_channels = expected_meg.shape[0]
        return (
            expected_meg,
            np.ones(n_channels, dtype=np.float64),
            np.zeros(n_channels, dtype=np.float64),
            {'ch_names': np.array(raw.ch_names)},
        )

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

    meg, observed_raw, scale, offset = load_subject_meg_and_raw(
        sub=3,
        ses=0,
        story_id=1,
        target_fs=100.0,
        offset_gap=10.0,
        meg_format='fif',
        data_root='/clean',
        preprocessed_meg_path='/preprocessed/meg.npz',
        raw_metadata_only=raw_metadata_only,
    )

    assert observed_raw is raw
    assert meg.dtype == np.float64
    np.testing.assert_array_equal(meg, expected_meg)
    np.testing.assert_array_equal(scale, np.ones(expected_meg.shape[0]))
    np.testing.assert_array_equal(offset, np.zeros(expected_meg.shape[0]))
    assert calls['raw_kwargs']['preload'] is expected_preload
    assert calls['raw_kwargs']['data_root'] == '/clean'
    assert len(raw.resample_calls) == expected_resample_calls
    if expected_resample_calls:
        assert raw.resample_calls == [(100.0, 'auto')]


def test_assert_valid_composed_scaling_rejects_sampling_rate_mismatch() -> None:
    with pytest.raises(ValueError, match='does not match requested MEG sampling rate'):
        assert_valid_composed_scaling(
            np.ones(2),
            np.zeros(2),
            n_channels=2,
            scaler_target_fs=50.0,
            requested_target_fs=100.0,
        )


def test_assert_valid_composed_scaling_rejects_invalid_values() -> None:
    kwargs = dict(n_channels=2, scaler_target_fs=100.0, requested_target_fs=100.0)
    with pytest.raises(ValueError, match='scale must have shape'):
        assert_valid_composed_scaling(np.ones((2, 1)), np.zeros(2), **kwargs)
    with pytest.raises(ValueError, match='offset must have shape'):
        assert_valid_composed_scaling(np.ones(2), np.zeros((2, 1)), **kwargs)
    with pytest.raises(ValueError, match='must be finite'):
        assert_valid_composed_scaling(np.array([1.0, np.nan]), np.zeros(2), **kwargs)
    with pytest.raises(ValueError, match='must be finite'):
        assert_valid_composed_scaling(np.ones(2), np.array([0.0, np.inf]), **kwargs)
    with pytest.raises(ValueError, match='strictly positive'):
        assert_valid_composed_scaling(np.array([1.0, 0.0]), np.zeros(2), **kwargs)
    with pytest.raises(ValueError, match='strictly positive'):
        assert_valid_composed_scaling(np.array([1.0, -1.0]), np.zeros(2), **kwargs)


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


def test_physical_affine_preserves_normalized_branch_activation(
    pattern_inputs: dict[str, np.ndarray],
) -> None:
    z = pattern_inputs['meg']
    weight_z = pattern_inputs['spatial_filters_weight']
    bias_z = pattern_inputs['spatial_filters_bias']
    scale = np.geomspace(1e-13, 8e-13, z.shape[0])
    offset = np.linspace(1e-14, 4e-14, z.shape[0])
    x = scale[:, None] * z + offset[:, None]
    weight_phys, bias_phys = physical_spatial_affine(
        weight_z, bias_z, scale, offset, sub=2
    )

    projected_z = weight_z[1] @ z + bias_z[1]
    projected_x = weight_phys[1] @ x + bias_phys[1]
    np.testing.assert_allclose(projected_x, projected_z)

    temporal_z = calculate_temporal_patterns(
        meg=z,
        spatial_filters_weight=weight_z,
        temporal_filters=pattern_inputs['temporal_filters'],
        sub=2,
        spatial_filters_bias=bias_z,
    )
    temporal_x = calculate_temporal_patterns(
        meg=x,
        spatial_filters_weight=weight_phys,
        temporal_filters=pattern_inputs['temporal_filters'],
        sub=2,
        spatial_filters_bias=bias_phys,
    )
    np.testing.assert_allclose(temporal_x, temporal_z)


def test_offset_padding_matches_zero_pad_of_centered_signal(
    pattern_inputs: dict[str, np.ndarray],
) -> None:
    rng = np.random.default_rng(21)
    x = rng.normal(size=80)
    offset = 0.37
    for kernel_len in (4, 5, 8, 9):
        kernel = rng.normal(size=kernel_len)
        padded = filter_data(x, kernel, filtfilt=False, pad_value=offset)
        centered = filter_data(x - offset, kernel, filtfilt=False, pad_value=0.0)
        np.testing.assert_allclose(padded, centered + offset * kernel.sum())

    z = pattern_inputs['meg']
    scale = np.geomspace(1e-13, 8e-13, z.shape[0])
    offset_vec = np.linspace(1e-14, 4e-14, z.shape[0])
    x_phys = scale[:, None] * z + offset_vec[:, None]
    weight_phys, _bias = physical_spatial_affine(
        pattern_inputs['spatial_filters_weight'],
        None,
        scale,
        offset_vec,
        sub=2,
    )
    patterns_padded = calculate_spatial_patterns(
        meg=x_phys,
        spatial_filters_weight=weight_phys,
        temporal_filters=pattern_inputs['temporal_filters'],
        sub=2,
        filtfilt=False,
        channel_pad=offset_vec,
    )
    patterns_centered = calculate_spatial_patterns(
        meg=x_phys - offset_vec[:, None],
        spatial_filters_weight=weight_phys,
        temporal_filters=pattern_inputs['temporal_filters'],
        sub=2,
        filtfilt=False,
        channel_pad=np.zeros(z.shape[0]),
    )
    np.testing.assert_allclose(patterns_padded, patterns_centered)


def test_filtfilt_offset_padding_matches_normalized_affine_transform() -> None:
    rng = np.random.default_rng(7)
    z = rng.normal(size=96)
    scale = 2.5e-13
    offset = 1.7e-14
    kernel = rng.normal(size=11)
    kernel = kernel - kernel.mean() + 0.4
    assert not np.isclose(kernel.sum(), 1.0)
    assert not np.isclose(kernel.sum(), 0.0)
    x = scale * z + offset
    y_z = filter_data(z, kernel, filtfilt=True, pad_value=0.0)
    y_x = filter_data(x, kernel, filtfilt=True, pad_value=offset)
    np.testing.assert_allclose(y_x, scale * y_z + offset * kernel.sum() ** 2)


def test_physical_affine_no_bias_preserves_activation_identity(
    pattern_inputs: dict[str, np.ndarray],
) -> None:
    z = pattern_inputs['meg']
    weight_z = pattern_inputs['spatial_filters_weight']
    scale = np.geomspace(1e-13, 8e-13, z.shape[0])
    offset = np.linspace(1e-14, 4e-14, z.shape[0])
    x = scale[:, None] * z + offset[:, None]
    weight_phys, bias_phys = physical_spatial_affine(
        weight_z, None, scale, offset, sub=2
    )
    assert bias_phys is not None
    np.testing.assert_allclose(
        weight_z[1] @ z, weight_phys[1] @ x + bias_phys[1]
    )


def test_spatial_roughness_is_invariant_at_physical_sensor_scale() -> None:
    rng = np.random.default_rng(41)
    spatial_patterns = rng.normal(size=(3, 9))
    sensor_xy = rng.normal(size=(9, 2))

    expected = compute_item_spatial_roughness(spatial_patterns, sensor_xy)
    observed = compute_item_spatial_roughness(spatial_patterns * 1e-14, sensor_xy)

    np.testing.assert_allclose(observed, expected)
    with pytest.raises(ValueError, match='zero-variance patterns'):
        compute_item_spatial_roughness(np.ones((1, 9)), sensor_xy)


def test_source_normalization_is_invariant_at_physical_scale() -> None:
    source_data = np.asarray(
        [
            [1.0, -2.0],
            [-3.0, 1.0],
            [2.0, 4.0],
        ]
    )

    expected = normalize_source_magnitudes(source_data)
    observed = normalize_source_magnitudes(source_data * 1e-14)

    np.testing.assert_allclose(observed, expected)
    with pytest.raises(ValueError, match='Zero source projection norms'):
        normalize_source_magnitudes(np.zeros((3, 1)))
