from pathlib import Path
import sys

import mne
import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from lisa.data.meg_io import (  # noqa: E402
    assert_inverts_to_physical,
    invert_meg_scaling,
    meg_checksum,
    preprocess_meg,
    read_scaler_states,
    scaler_state_for,
    write_scaler_states,
)


SCALER_STATE_FIELDS = {
    'ch_names',
    'baseline',
    'robust_center',
    'robust_scale',
    'standard_mean',
    'standard_std',
    'n_clipped',
    'n_times',
    'n_fit_samples',
    'first_onset',
    'first_test_onset',
    'target_fs',
    'quantile_range',
    'clip_abs',
    'scalers_reused',
    'data_crc32',
}
PER_CHANNEL_FIELDS = (
    'baseline',
    'robust_center',
    'robust_scale',
    'standard_mean',
    'standard_std',
)
N_CHANNELS = 8
N_TIMES = 3000
FIRST_ONSET = 2.0
TARGET_FS = 100.0


def make_raw(seed: int) -> mne.io.RawArray:
    rng = np.random.default_rng(seed)
    info = mne.create_info(
        ch_names=[f'MEG{index:03d}' for index in range(N_CHANNELS)],
        sfreq=TARGET_FS,
        ch_types='mag',
    )
    data = rng.normal(scale=1e-12, size=(N_CHANNELS, N_TIMES))
    return mne.io.RawArray(data, info, verbose=False)


def baseline_of(physical: np.ndarray) -> np.ndarray:
    """Recompute the baseline the way preprocess_meg does, from the raw data."""
    window = slice(
        int(round((FIRST_ONSET - 0.5) * TARGET_FS)),
        int(round(FIRST_ONSET * TARGET_FS)),
    )
    return physical[:, window].mean(axis=1)


def assert_physical(restored: np.ndarray, expected: np.ndarray) -> None:
    tolerance = 1e-6 * float(np.abs(expected).max())
    assert np.allclose(restored, expected, rtol=1e-4, atol=tolerance)


@pytest.fixture
def raw() -> mne.io.RawArray:
    return make_raw(seed=7)


@pytest.fixture
def saved_recordings(tmp_path: Path) -> dict:
    """Two preprocessed recordings and the scalers file that describes them."""
    recordings = {}
    states = []
    for key, seed in (('rec1', 7), ('rec2', 8)):
        meg, processed, state, _ = preprocess_meg(
            raw=make_raw(seed), target_fs=TARGET_FS, first_onset=FIRST_ONSET
        )
        physical = processed.get_data()
        recordings[key] = {
            'meg': meg,
            'physical': physical - baseline_of(physical)[:, None],
        }
        states.append(state)

    path = tmp_path / 'scalers.npz'
    write_scaler_states(path, list(recordings), states)
    return {'path': path, 'recordings': recordings}


def test_inversion_recovers_physical_units(raw: mne.io.RawArray) -> None:
    """Inversion returns the baseline-corrected recording, in Tesla."""
    meg, raw, state, excerpt = preprocess_meg(
        raw=raw, target_fs=TARGET_FS, first_onset=FIRST_ONSET
    )
    assert state['n_clipped'].sum() == 0

    physical = raw.get_data()
    baseline = baseline_of(physical)
    assert np.allclose(state['baseline'], baseline, rtol=1e-12, atol=0.0)

    corrected = physical - baseline[:, None]
    assert_physical(invert_meg_scaling(meg, state), corrected)
    assert np.allclose(excerpt, corrected[:, : excerpt.shape[1]], rtol=1e-12, atol=0.0)


def test_inversion_rejects_wrong_channel_count(raw: mne.io.RawArray) -> None:
    meg, _, state, _ = preprocess_meg(
        raw=raw, target_fs=TARGET_FS, first_onset=FIRST_ONSET
    )
    with pytest.raises(ValueError, match='channels'):
        invert_meg_scaling(meg[:-1], state)


def test_state_describes_the_returned_array(raw: mne.io.RawArray) -> None:
    meg, _, state, _ = preprocess_meg(
        raw=raw, target_fs=TARGET_FS, first_onset=FIRST_ONSET
    )

    assert set(state) == SCALER_STATE_FIELDS
    assert state['n_clipped'].shape == (N_CHANNELS,)
    for name in PER_CHANNEL_FIELDS:
        assert state[name].shape == (N_CHANNELS,)
        assert np.all(np.isfinite(state[name]))
    assert np.all(state['robust_scale'] > 0)
    assert np.all(state['standard_std'] > 0)
    assert np.array_equal(state['ch_names'], np.array(raw.ch_names))
    assert state['n_times'] == meg.shape[1]
    assert state['n_fit_samples'] == meg.shape[1]
    assert np.isnan(state['first_test_onset'])
    assert state['scalers_reused'] is False
    assert state['data_crc32'] == meg_checksum(meg)


def test_checksum_covers_the_whole_recording(raw: mne.io.RawArray) -> None:
    """A changed sample must be detected wherever it sits in the recording."""
    meg, _, _, _ = preprocess_meg(raw=raw, target_fs=TARGET_FS, first_onset=FIRST_ONSET)
    for sample in (0, 1500, meg.shape[1] - 1):
        altered = meg.copy()
        altered[0, sample] += 1.0
        assert meg_checksum(altered) != meg_checksum(meg)


def test_clipped_artifact_does_not_loosen_the_check(raw: mne.io.RawArray) -> None:
    """A wrong stored scale must fail even next to a huge clipped artifact."""
    data = raw.get_data()
    data[0, 100] = 1e-6  # about a million times the signal, so it clips
    spiked = mne.io.RawArray(data, raw.info, verbose=False)
    meg, _, state, excerpt = preprocess_meg(
        raw=spiked, target_fs=TARGET_FS, first_onset=FIRST_ONSET
    )
    assert state['n_clipped'].sum() > 0

    excerpt_meg = meg[:, : excerpt.shape[1]]
    assert_inverts_to_physical(excerpt_meg, state, excerpt, 'The recording')

    wrong = dict(state, robust_scale=state['robust_scale'] * 1.001)
    with pytest.raises(ValueError, match='does not invert'):
        assert_inverts_to_physical(excerpt_meg, wrong, excerpt, 'The recording')


def test_clipped_positions_are_compared_too(raw: mne.io.RawArray) -> None:
    """A clipped sample stored past the threshold must be caught, not skipped."""
    data = raw.get_data()
    data[0, 100] = 1e-6
    spiked = mne.io.RawArray(data, raw.info, verbose=False)
    meg, _, state, excerpt = preprocess_meg(
        raw=spiked, target_fs=TARGET_FS, first_onset=FIRST_ONSET
    )
    excerpt_meg = meg[:, : excerpt.shape[1]].copy()
    clipped = np.flatnonzero(np.abs(excerpt_meg[0]) >= state['clip_abs'])
    assert clipped.size == 1

    excerpt_meg[0, clipped[0]] = state['clip_abs'] + 5.0
    with pytest.raises(ValueError, match='does not invert'):
        assert_inverts_to_physical(excerpt_meg, state, excerpt, 'The recording')


def test_check_needs_an_unclipped_sample_in_every_channel(
    raw: mne.io.RawArray,
) -> None:
    """A channel clipped throughout would go unchecked, so it must raise."""
    meg, _, state, excerpt = preprocess_meg(
        raw=raw, target_fs=TARGET_FS, first_onset=FIRST_ONSET
    )
    excerpt_meg = meg[:, : excerpt.shape[1]].copy()
    excerpt_meg[0] = state['clip_abs']

    with pytest.raises(ValueError, match=r'channels \[0\] are clipped'):
        assert_inverts_to_physical(excerpt_meg, state, excerpt, 'The recording')


def test_scalers_fitted_on_train_portion_only(raw: mne.io.RawArray) -> None:
    _, _, state, _ = preprocess_meg(
        raw=raw,
        target_fs=TARGET_FS,
        first_onset=FIRST_ONSET,
        first_test_onset=20.0,
    )
    assert state['n_fit_samples'] == int(round(20.0 * TARGET_FS))
    assert state['first_test_onset'] == 20.0


def test_clipping_is_counted_per_channel(raw: mne.io.RawArray) -> None:
    data = raw.get_data()
    data[0, 100] = 1e-6
    spiked = mne.io.RawArray(data, raw.info, verbose=False)
    _, _, state, _ = preprocess_meg(
        raw=spiked, target_fs=TARGET_FS, first_onset=FIRST_ONSET
    )
    assert state['n_clipped'][0] >= 1
    assert state['n_clipped'][1:].sum() == 0


def test_state_saves_and_loads_without_pickle(
    raw: mne.io.RawArray, tmp_path: Path
) -> None:
    _, _, state, _ = preprocess_meg(
        raw=raw, target_fs=TARGET_FS, first_onset=FIRST_ONSET
    )
    path = tmp_path / 'scalers.npz'
    np.savez(path, **state)

    with np.load(path, allow_pickle=False) as loaded:
        assert set(loaded.files) == SCALER_STATE_FIELDS
        assert np.array_equal(loaded['ch_names'], state['ch_names'])
        for name in PER_CHANNEL_FIELDS:
            assert np.array_equal(loaded[name], state[name])


def test_saved_state_inverts_the_recording_it_belongs_to(
    saved_recordings: dict,
) -> None:
    saved = read_scaler_states(saved_recordings['path'])
    for subset, recording in saved_recordings['recordings'].items():
        state = scaler_state_for(saved, subset, recording['meg'])
        assert set(state) == SCALER_STATE_FIELDS
        assert_physical(
            invert_meg_scaling(recording['meg'], state), recording['physical']
        )


def test_saved_state_rejects_another_recording(saved_recordings: dict) -> None:
    saved = read_scaler_states(saved_recordings['path'])
    other = saved_recordings['recordings']['rec1']['meg']
    with pytest.raises(ValueError, match='does not belong'):
        scaler_state_for(saved, 'rec2', other)


def test_saved_state_rejects_wrong_length(saved_recordings: dict) -> None:
    saved = read_scaler_states(saved_recordings['path'])
    truncated = saved_recordings['recordings']['rec2']['meg'][:, :-1]
    with pytest.raises(ValueError, match='samples'):
        scaler_state_for(saved, 'rec2', truncated)


def test_saved_state_rejects_unknown_subset(saved_recordings: dict) -> None:
    saved = read_scaler_states(saved_recordings['path'])
    meg = saved_recordings['recordings']['rec1']['meg']
    with pytest.raises(ValueError, match='0 scaler rows'):
        scaler_state_for(saved, 'rec3', meg)


def test_writer_rejects_mismatched_inputs(
    saved_recordings: dict, tmp_path: Path
) -> None:
    saved = read_scaler_states(saved_recordings['path'])
    states = [
        scaler_state_for(saved, subset, recording['meg'])
        for subset, recording in saved_recordings['recordings'].items()
    ]
    with pytest.raises(ValueError, match='keys'):
        write_scaler_states(tmp_path / 'bad.npz', ['rec1'], states)
    with pytest.raises(ValueError, match='unique'):
        write_scaler_states(tmp_path / 'bad.npz', ['rec1', 'rec1'], states)


def test_unpreprocessed_call_returns_no_state(raw: mne.io.RawArray) -> None:
    data, _, state, excerpt = preprocess_meg(
        raw=raw,
        target_fs=TARGET_FS,
        first_onset=FIRST_ONSET,
        preprocess=False,
    )
    assert state is None
    assert excerpt is None
    assert data.shape[0] == N_CHANNELS
