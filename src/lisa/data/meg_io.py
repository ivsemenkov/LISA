"""MEG I/O and preprocessing helpers."""

import ast
import os
import warnings

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
    """
    # --- 1) resample to target_fs Hz ---
    raw.resample(target_fs, npad='auto')

    # --- 2) baseline: subtract per-channel mean over first 0.5 s ---
    # Note: raw already contains only MEG channels (filtered in load_raw_meg)
    data = raw.get_data()  # (n_channels, n_times)
    if not preprocess:
        return data, raw
    sfreq = raw.info['sfreq']
    if first_onset > 0.5:
        n_bl_start = int(round((first_onset - 0.5) * sfreq))
        n_bl_stop = int(round(first_onset * sfreq))
    else:
        n_bl_start = 0
        n_bl_stop = int(round(0.5 * sfreq))
    data -= data[:, n_bl_start:n_bl_stop].mean(axis=1, keepdims=True)

    if first_test_onset is not None:
        first_test_onset = int(round(first_test_onset * sfreq))

    # --- 3) RobustScaler (median/IQR) per channel ---
    X = data.T  # sklearn expects (samples, features) -> (time, channels)

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

    # --- 5) clamp to ±20 SD ---
    np.clip(X, -20.0, 20.0, out=X)

    return X.T.astype(np.float32), raw


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
    meg, raw = preprocess_meg(
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
