"""MEG I/O and preprocessing helpers."""

import ast
import os
import warnings
import zlib

import mne
import mne_bids
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler

from lisa.utils.numeric import assert_finite as _assert_finite
from lisa.utils.constants import CLEAN_DATA_DIR


warnings.filterwarnings(
    'ignore',
    category=RuntimeWarning,
    message=(
        r'^Unable to map the following column\(s\) to to MNE:\n'
        r'task_order: .*\n'
        r'n_sessions: .*\n'
        r'mri: .*\n'
        r'native_english_speaker: .*$'
    ),
)

CLIP_ABS = 20.0


def meg_checksum(meg: np.ndarray) -> int:
    """CRC32 over every sample, binding scaler state to one exact recording.

    The bytes are hashed as they are stored, so the checksum also depends on
    dtype: convert an array to float64 and the checksum changes. Look the
    scaler state up with the array as saved, then convert.
    """
    return zlib.crc32(np.ascontiguousarray(meg))


def composed_scaling(scaler_state: dict) -> tuple[np.ndarray, np.ndarray]:
    """Express both scalers as one affine map per channel: meg * scale + offset.

    Args:
        scaler_state: State from preprocess_meg or scaler_state_for.

    Returns:
        scale: Physical units per normalized unit, one per channel.
        offset: Physical value a normalized zero corresponds to, per channel.
    """
    scale = scaler_state['standard_std'] * scaler_state['robust_scale']
    offset = (
        scaler_state['standard_mean'] * scaler_state['robust_scale']
        + scaler_state['robust_center']
    )
    return scale, offset


def invert_meg_scaling(meg: np.ndarray, scaler_state: dict) -> np.ndarray:
    """Undo the scaler normalization, returning MEG to its recorded units.

    ICA cleaning, resampling, baseline correction and clipping define the
    signal being analysed and stay applied. Only the per-channel scalers are
    undone, because they are the one step that makes amplitudes across sensors
    non-physical.

    Args:
        meg: Normalized data, (n_channels, n_times).
        scaler_state: State from preprocess_meg or scaler_state_for.

    Returns:
        The baseline-corrected recording in its physical units (T for
        magnetometers). Clipped samples stay clipped; n_clipped records how
        many there were per channel.
    """
    n_channels = scaler_state['robust_scale'].shape[0]
    if meg.shape[0] != n_channels:
        raise ValueError(
            f'MEG has {meg.shape[0]} channels but the scaler state has {n_channels}.'
        )
    scale, offset = composed_scaling(scaler_state)
    return meg * scale[:, None] + offset[:, None]


def assert_inverts_to_physical(
    meg: np.ndarray, scaler_state: dict, physical: np.ndarray, name: str
) -> None:
    """Raise unless inverting meg reproduces the physical recording.

    The physical reference is clipped directly at each channel's physical
    bounds, ``offset +/- clip_abs * scale``, so every sample takes part in the
    comparison. Clipped samples confirm the threshold and sign, but cannot pin
    the scale down because they only say the value lay beyond the threshold;
    every channel therefore has to keep at least one unclipped sample.

    Each sample is allowed only the error its stored form can explain. meg is
    written as float32, so a sample already carries up to half an ulp of
    rounding, which the inverse multiplies by that channel's scale; the offset
    contributes its own float64 rounding. Allowing a few times that, per
    sample, keeps the check meaningful in the presence of a large artifact: a
    tolerance read off the signal amplitude instead would be inflated by the
    artifact and would accept a scale that is wrong by a tenth of a percent.

    Args:
        meg: Normalized data, (n_channels, n_times).
        scaler_state: State describing meg.
        physical: Baseline-corrected data in physical units, same shape as meg.
        name: What is being checked, used in the error message.
    """
    if meg.shape != physical.shape:
        raise ValueError(
            f'{name}: MEG is {meg.shape} but the physical data is {physical.shape}.'
        )
    clip_abs = scaler_state['clip_abs']
    unchecked = np.flatnonzero(~(np.abs(meg) < clip_abs).any(axis=1))
    if unchecked.size:
        raise ValueError(
            f'{name}: channels {unchecked.tolist()} are clipped at every sample, '
            'so their scaling cannot be pinned down.'
        )
    scale, offset = composed_scaling(scaler_state)
    reference = np.clip(
        physical,
        (offset - clip_abs * scale)[:, None],
        (offset + clip_abs * scale)[:, None],
    )
    rounding = np.spacing(np.abs(meg).astype(np.float32)).astype(np.float64)
    allowed = 8.0 * (rounding * scale[:, None] + np.spacing(np.abs(offset))[:, None])
    error = np.abs(invert_meg_scaling(meg, scaler_state) - reference)
    if not np.all(error <= allowed):
        raise ValueError(f'{name} does not invert to its physical units.')


def write_scaler_states(scalers_path: str, keys: list, states: list) -> None:
    """Save the scaler states of several recordings as stacked arrays.

    Args:
        scalers_path: Destination NPZ, written alongside the preprocessed MEG.
        keys: Recording keys, in the same order as the saved MEG arrays.
        states: Scaler state per recording, in the order of keys.
    """
    if len(keys) != len(states):
        raise ValueError(f'Got {len(keys)} keys but {len(states)} scaler states.')
    if len(set(keys)) != len(keys):
        raise ValueError('Recording keys are not unique.')
    for key, state in zip(keys, states):
        if not np.array_equal(state['ch_names'], states[0]['ch_names']):
            raise ValueError(f'Channel names differ for {key}.')
    np.savez(
        scalers_path,
        keys=np.array(keys),
        ch_names=states[0]['ch_names'],
        baseline=np.stack([s['baseline'] for s in states]),
        robust_center=np.stack([s['robust_center'] for s in states]),
        robust_scale=np.stack([s['robust_scale'] for s in states]),
        standard_mean=np.stack([s['standard_mean'] for s in states]),
        standard_std=np.stack([s['standard_std'] for s in states]),
        n_clipped=np.stack([s['n_clipped'] for s in states]),
        n_times=np.array([s['n_times'] for s in states]),
        n_fit_samples=np.array([s['n_fit_samples'] for s in states]),
        first_onset=np.array([s['first_onset'] for s in states]),
        first_test_onset=np.array([s['first_test_onset'] for s in states]),
        scalers_reused=np.array([s['scalers_reused'] for s in states]),
        data_crc32=np.array([s['data_crc32'] for s in states], dtype=np.int64),
        target_fs=states[0]['target_fs'],
        clip_abs=states[0]['clip_abs'],
        quantile_range=states[0]['quantile_range'],
    )


def read_scaler_states(scalers_path: str) -> dict:
    """Load the whole scalers file as one snapshot of arrays.

    Args:
        scalers_path: Scalers NPZ written alongside the preprocessed MEG.

    Returns:
        Every stored array, keyed by name, with the file already closed.
    """
    with np.load(scalers_path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def scaler_state_for(saved: dict, subset: str, meg: np.ndarray) -> dict:
    """Take one recording's scaler state, checking that it belongs to meg.

    Args:
        saved: Snapshot from read_scaler_states.
        subset: Recording key, e.g. "subject01_session0_story1".
        meg: Normalized recording the state must belong to.

    Returns:
        scaler_state: Same fields as returned by preprocess_meg.
    """
    rows = np.flatnonzero(saved['keys'] == subset)
    if rows.size != 1:
        raise ValueError(f'Found {rows.size} scaler rows for {subset}.')
    row = int(rows[0])
    if meg.shape[0] != saved['robust_scale'].shape[1]:
        raise ValueError(
            f'MEG has {meg.shape[0]} channels but the scaler state has '
            f'{saved["robust_scale"].shape[1]} for {subset}.'
        )
    if meg.shape[1] != saved['n_times'][row]:
        raise ValueError(
            f'MEG has {meg.shape[1]} samples but the scaler state has '
            f'{saved["n_times"][row]} for {subset}.'
        )
    if meg_checksum(meg) != saved['data_crc32'][row]:
        raise ValueError(f'Scaler state does not belong to the given MEG for {subset}.')
    return {
        'ch_names': saved['ch_names'],
        'baseline': saved['baseline'][row],
        'robust_center': saved['robust_center'][row],
        'robust_scale': saved['robust_scale'][row],
        'standard_mean': saved['standard_mean'][row],
        'standard_std': saved['standard_std'][row],
        'n_clipped': saved['n_clipped'][row],
        'n_times': int(saved['n_times'][row]),
        'n_fit_samples': int(saved['n_fit_samples'][row]),
        'first_onset': float(saved['first_onset'][row]),
        'first_test_onset': float(saved['first_test_onset'][row]),
        'target_fs': float(saved['target_fs']),
        'quantile_range': saved['quantile_range'],
        'clip_abs': float(saved['clip_abs']),
        'scalers_reused': bool(saved['scalers_reused'][row]),
        'data_crc32': int(saved['data_crc32'][row]),
    }


def load_raw_meg(
    meg_format: str,
    data_root: str,
    sub: int,
    ses: int,
    story_id: int,
    *,
    preload: bool = True,
):
    """Load raw MEG recording in FIF or BIDS format.

    Args:
        preload: Load the full recording into memory. Disable when only
            channel and sensor metadata are required.

    Returns:
        raw: MNE Raw object.
        orig_fs: Original sampling rate (Hz).
    """
    sub = f'{sub:02d}'
    if meg_format == 'bids':
        bids_path = mne_bids.BIDSPath(
            subject=sub,
            session=str(ses),
            task=str(story_id),
            datatype='meg',
            root=data_root,
        )
        raw = mne_bids.read_raw_bids(bids_path, verbose=False)
    elif meg_format == 'fif':
        raw = mne.io.read_raw_fif(
            os.path.join(
                data_root,
                f'sub-{sub}',
                f'ses-{ses}',
                'meg',
                f'sub-{sub}_ses-{ses}_task-{story_id}_desc-clean_meg.fif',
            ),
            verbose=False,
        )
    else:
        raise ValueError(f'Format {meg_format} is not available for processing')
    # Pick only MEG channels
    picks = mne.pick_types(
        raw.info, meg=True, misc=False, eeg=False, eog=False, ecg=False
    )
    raw.pick(picks)
    if preload:
        raw.load_data(verbose=False)
    orig_fs = raw.info['sfreq']
    assert orig_fs == 1000, (sub, ses, story_id, orig_fs)
    return raw, orig_fs


def preprocess_meg(
    raw: mne.io.BaseRaw,
    target_fs: float,
    first_onset: float,
    robust_scaler: RobustScaler | None = None,
    standard_scaler: StandardScaler | None = None,
    first_test_onset: float | None = None,
    preprocess: bool = True,
):
    """Resample and normalize MEG data, with optional scaler reuse.

    Args:
        raw: MNE Raw object.
        target_fs: Target sampling rate (Hz).
        first_onset: First stimulus onset (s) for baseline window.
        robust_scaler: Optional pre-fit RobustScaler.
        standard_scaler: Optional pre-fit StandardScaler.
        first_test_onset: If provided, fit scalers only on train portion.
        preprocess: If False, return raw data without normalization.

    Returns:
        meg: Preprocessed data, (n_channels, n_times).
        raw: MNE Raw object, resampled in place.
        scaler_state: Every scaling parameter applied, or None when
            preprocess is False. Inverting it returns data to physical units.
        physical_excerpt: First samples of the baseline-corrected recording in
            physical units, for checking saved files without reading the
            recording again. None when preprocess is False.
    """
    scalers_reused = robust_scaler is not None or standard_scaler is not None

    # --- 1) resample to target_fs Hz ---
    raw.resample(target_fs, npad='auto')

    # --- 2) baseline: subtract per-channel mean over first 0.5 s ---
    # Note: raw already contains only MEG channels (filtered in load_raw_meg)
    data = raw.get_data()  # (n_channels, n_times)
    if not preprocess:
        return data, raw, None, None
    sfreq = raw.info['sfreq']
    if first_onset > 0.5:
        n_bl_start = int(round((first_onset - 0.5) * sfreq))
        n_bl_stop = int(round(first_onset * sfreq))
    else:
        n_bl_start = 0
        n_bl_stop = int(round(0.5 * sfreq))
    baseline = data[:, n_bl_start:n_bl_stop].mean(axis=1, keepdims=True)
    data -= baseline

    n_check = min(1000, data.shape[1])
    physical_excerpt = data[:, :n_check].copy()

    first_test_onset_sec = first_test_onset
    if first_test_onset is not None:
        first_test_onset = int(round(first_test_onset * sfreq))

    # --- 3) RobustScaler (median/IQR) per channel ---
    X = data.T  # sklearn expects (samples, features) -> (time, channels)
    n_fit_samples = X.shape[0] if first_test_onset is None else first_test_onset

    if robust_scaler is None:
        X_fit = X if first_test_onset is None else X[:first_test_onset, :]
        robust_scaler = RobustScaler(
            with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)
        )
        robust_scaler = robust_scaler.fit(X_fit)
    X = robust_scaler.transform(X)

    # --- 4) Standardize to zero-mean, unit-var per channel ---
    if standard_scaler is None:
        X_fit = X if first_test_onset is None else X[:first_test_onset, :]
        standard_scaler = StandardScaler(with_mean=True, with_std=True)
        standard_scaler = standard_scaler.fit(X_fit)
    X = standard_scaler.transform(X)

    if not np.all(robust_scaler.scale_ > 0) or not np.all(standard_scaler.scale_ > 0):
        raise ValueError('Scaler scales must be strictly positive to be invertible.')

    # --- 5) clamp to ±20 SD ---
    n_clipped = np.count_nonzero(np.abs(X) > CLIP_ABS, axis=0)
    np.clip(X, -CLIP_ABS, CLIP_ABS, out=X)

    meg = X.T.astype(np.float32)
    scaler_state = {
        'ch_names': np.array(raw.ch_names),
        'baseline': baseline[:, 0],
        'robust_center': robust_scaler.center_,
        'robust_scale': robust_scaler.scale_,
        'standard_mean': standard_scaler.mean_,
        'standard_std': standard_scaler.scale_,
        'n_clipped': n_clipped,
        'n_times': meg.shape[1],
        'n_fit_samples': n_fit_samples,
        'first_onset': float(first_onset),
        'first_test_onset': (
            np.nan if first_test_onset_sec is None else float(first_test_onset_sec)
        ),
        'target_fs': float(sfreq),
        'quantile_range': np.array(robust_scaler.quantile_range),
        'clip_abs': CLIP_ABS,
        'scalers_reused': scalers_reused,
        'data_crc32': meg_checksum(meg),
    }
    for name in (
        'baseline',
        'robust_center',
        'robust_scale',
        'standard_mean',
        'standard_std',
    ):
        _assert_finite(name, scaler_state[name])

    assert_inverts_to_physical(
        meg[:, :n_check], scaler_state, physical_excerpt, 'The processed recording'
    )

    return meg, raw, scaler_state, physical_excerpt


def load_meg(
    target_fs: float,
    sub: int,
    ses: int,
    story_id: int,
    offset_gap: float = 10,
    return_raw: bool = False,
    preprocess: bool = True,
):
    """Load and preprocess cleaned MEG data for a subject/session/story.

    Args:
        target_fs: Target sampling rate (Hz).
        sub: Subject id.
        ses: Session id.
        story_id: Story id.
        offset_gap: Seconds to trim from start/end.
        return_raw: If True, return (meg, raw).
        preprocess: If False, return raw data without normalization.
    """
    raw, orig_fs = load_raw_meg(
        meg_format='fif', data_root=CLEAN_DATA_DIR, sub=sub, ses=ses, story_id=story_id
    )
    _assert_finite('raw_data', raw.get_data())

    onsets = {}
    events, events_dict = mne.events_from_annotations(raw=raw, verbose=False)
    for event, event_id in events_dict.items():
        event = ast.literal_eval(event)
        if event['kind'] == 'sound':
            event_sample = events[events[:, -1] == event_id]
            assert len(event_sample) == 1, event_sample
            event_sample = event_sample[0, 0]
            event_onset = event_sample / orig_fs
            sound_fname = event['sound'].lower()
            assert sound_fname.endswith('.0.wav'), sound_fname
            sound_fname = os.path.basename(sound_fname[: -len('.0.wav')]) + '.wav'
            onsets[(int(event['sound_id']), sound_fname)] = event_onset

    first_onset = min(onsets.values())
    meg, raw, _, _ = preprocess_meg(
        raw=raw, target_fs=target_fs, first_onset=first_onset, preprocess=preprocess
    )
    offset_gap = int(offset_gap * target_fs)
    meg = meg[
        :, offset_gap:-offset_gap
    ]  # skip first/final 10 seconds to avoid onset/offset effects
    _assert_finite('meg_preprocessed', meg)
    if return_raw:
        return meg, raw
    return meg
