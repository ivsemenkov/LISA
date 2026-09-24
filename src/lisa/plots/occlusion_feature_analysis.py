"""Paired raw-MEG occlusion analysis for interpretable audio features.


For feature ``f``, every occurrence in a three-second test query is replaced in
two matched ways:

``f_to_absent``
    real MEG from an interval where ``f`` is absent;
``f_to_present``
    real MEG from an interval where ``f`` is present.

The primary query-level effect is the strict one-based retrieval-rank contrast

    rank(f_to_absent) - rank(f_to_present).

Positive values mean that annotation-present donor MEG preserves retrieval
better than an equally sized annotation-absent replacement.  All ranks are
always computed against the complete, unchanged test candidate bank.  Query
eligibility is applied only after ranking, and the identical eligibility mask
is applied to baseline and both occlusion arms.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import zipfile
from bisect import bisect_left
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import einops
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as torch_f
from scipy import stats as scipy_stats
from tqdm import tqdm

from lisa.data.audio_reduction import (
    get_audio_embeddings_path,
    get_dataframe_path,
)
from lisa.model.load_model import _load_run_dir_and_config, load_retrieval_models
from lisa.plots.feature_traces import audio_feature_trace, load_word_features
from lisa.utils.constants import (
    AUDIO_SR,
    DATA_ROOT,
    EXPERIMENTS_DIR,
    N_FEATURES,
    N_SUBJECTS,
    OCCLUSION_FEATURE_ANALYSIS_DIR,
    PREPROCESSED_DATA_DIR,
    SUB_SES_COMBOS,
)
from lisa.utils.determinism import fix_seed
from lisa.utils.textgrid_io import parse_textgrid_tier_exact
from lisa.utils.validators import validate_run_group


ANALYSIS_NAME = 'occlusion_feature_analysis'
DEFAULT_OUTPUT_ROOT = Path(OCCLUSION_FEATURE_ANALYSIS_DIR)
DEFAULT_DONOR_SOUND_IDS = (0, 1, 2, 3, 4)
DEFAULT_TARGET_STORY_ID = 3
DEFAULT_CUDA_INFERENCE_BATCH_SIZE = 1000
MAIN_EFFECTS_WIDTH_IN = 9.5
MAIN_EFFECTS_FONT_SIZE = 15.0


# Short labels for the wide headline figure.  The full contrast definitions
# remain in feature_summary.csv and analysis_config.json.
PLOT_DISPLAY_NAMES: dict[str, str] = {
    'textgrid_silence_period': 'Silence',
    'textgrid_silence_onset': 'Silence onset',
    'textgrid_silence_offset': 'Silence offset',
    'textgrid_silence_transition': 'Silence edges',
    'phoneme_vowels_all': 'Vowels',
    'phoneme_stops_all': 'Stops',
    'phoneme_fricatives_all': 'Fricatives',
    'phoneme_sibilants': 'Sibilants',
    'phoneme_nasals_all': 'Nasals',
    'phoneme_liquids_approximants': 'Liquids/glides',
    'phoneme_schwa': 'Schwa',
    'word_onset': 'Word onset',
    'pseudoword': 'Pseudoword',
    'wordlist': 'Word list',
    'surprisal': 'High surprisal',
    'entropy': 'High entropy',
    'word_frequency': 'Rare word',
    'loudness': 'High loudness',
    'acoustic_onset': 'Strong acoustic onset',
}


# Local phoneme inventories for the occlusion feature battery.
PHONEME_GROUPS: dict[str, tuple[str, ...]] = {
    'vowels_all': tuple(
        (
            'a i ə ɪ ɛ aj ɔ ɐ e iː əw aw ɑ æ ɒː ʊ eː ɚ ʉː uː '
            'ɒ ɔj ɑː aː ɜː ej oː ow ʉ ɛː u ɝ o ɜ'
        ).split()
    ),
    'stops_all': tuple(
        ('p b t d k ɡ ʈ ɖ c tʲ dʲ d̪ t̪ bʲ cʰ ʈʲ tʰ pʰ ɟ kʰ pʲ tʷ kʷ cʷ ɟʷ').split()
    ),
    'fricatives_all': tuple('s z f ð v ç ʃ θ fʲ h vʲ ʒ'.split()),
    'sibilants': tuple('s z ʃ ʒ'.split()),
    'nasals_all': tuple('n m ŋ ɲ mʲ m̩'.split()),
    'liquids_approximants': tuple('ɹ l ʋ ɫ ʎ j w ɾ'.split()),
    'schwa': ('ə',),
}


@dataclass(frozen=True)
class FeatureSpec:
    """One predeclared feature contrast."""

    name: str
    builder: str
    event: str = 'period'
    phonemes: tuple[str, ...] = ()
    insertion_kind: str | None = None
    column: str | None = None
    high_is_present: bool = True
    label: str | None = None


DEFAULT_FEATURE_BATTERY: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        'textgrid_silence_period',
        'textgrid_silence',
        event='period',
        label='Silence',
    ),
    FeatureSpec(
        'textgrid_silence_onset',
        'textgrid_silence',
        event='onset',
        label='Silence onset',
    ),
    FeatureSpec(
        'textgrid_silence_offset',
        'textgrid_silence',
        event='offset',
        label='Silence offset',
    ),
    FeatureSpec(
        'textgrid_silence_transition',
        'textgrid_silence',
        event='transition',
        label='Silence transition',
    ),
    *(
        FeatureSpec(
            f'phoneme_{name}',
            'phoneme',
            phonemes=phones,
            label=name.replace('_', ' ').title(),
        )
        for name, phones in PHONEME_GROUPS.items()
    ),
    FeatureSpec('word_onset', 'word_onset', event='point', label='Word onset'),
    FeatureSpec(
        'pseudoword',
        'insertion',
        insertion_kind='single',
        label='Pseudoword',
    ),
    FeatureSpec(
        'wordlist',
        'insertion',
        insertion_kind='wordlist',
        label='Word list',
    ),
    FeatureSpec(
        'surprisal',
        'word_value',
        column='surprisal_bits',
        high_is_present=True,
        label='High surprisal',
    ),
    FeatureSpec(
        'entropy',
        'word_value',
        column='entropy_bits',
        high_is_present=True,
        label='High entropy',
    ),
    FeatureSpec(
        'word_frequency',
        'word_value',
        column='zipf_freq',
        high_is_present=False,
        label='Rare word',
    ),
    FeatureSpec(
        'loudness',
        'audio_value',
        column='loudness',
        high_is_present=True,
        label='High loudness',
    ),
    FeatureSpec(
        'acoustic_onset',
        'audio_value',
        column='acoustic_onset',
        high_is_present=True,
        label='Strong acoustic onset',
    ),
)


@dataclass
class FeatureState:
    """Binary present/absent state traces on each sound's MEG-frame grid."""

    spec: FeatureSpec
    present: dict[str, np.ndarray]
    absent: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DonorInterval:
    """Sound-local half-open donor interval in MEG frames."""

    sound_fname: str
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class DonorPair:
    """Matched feature-absent and feature-present donor chunks."""

    absent: DonorInterval
    present: DonorInterval


@dataclass(frozen=True)
class DonorCandidatePool:
    """Reusable donor starts for one feature and target-component length."""

    length: int
    by_sound: dict[str, tuple[tuple[int, ...], tuple[int, ...]]]
    total_present: int
    total_absent: int
    same_file_capacity: int


@dataclass
class ComponentPlan:
    """Donors for one connected target-mask component."""

    start: int
    stop: int
    alpha: np.ndarray
    donor_pairs: tuple[DonorPair, ...]


@dataclass
class WindowPlan:
    """Complete simultaneous intervention plan for one test query."""

    wav_index: int
    sound_fname: str
    sound_start_frame: int
    support_fraction: float
    weighted_fraction: float
    components: tuple[ComponentPlan, ...]


@dataclass
class EligibilityRecord:
    """Auditable disposition of one feature/query pair."""

    feature: str
    wav_index: int
    sound_fname: str
    status: str
    reason: str
    support_fraction: float
    weighted_fraction: float
    n_components: int
    detail: str = ''


@dataclass
class AnalysisInputs:
    """Loaded frozen models, data, and fixed retrieval-bank context."""

    model: torch.nn.Module
    audio_model: torch.nn.Module
    hyper_params: dict[str, Any]
    run_dir: Path
    configured_dirprocess: Path
    dirprocess: Path
    audio_embeddings_path: Path
    train_dataframe_path: Path
    test_dataframe_path: Path
    meg_path: Path
    hidden_test: np.ndarray
    df_train: pd.DataFrame
    df_test: pd.DataFrame
    test_windows: pd.DataFrame
    candidate_ids: np.ndarray
    device: torch.device


def combo_key(subject: int, session: int) -> str:
    return f'sub{subject}-session{session}'


def meg_subset_key(subject: int, session: int, story_id: int) -> str:
    return f'subject{subject:02d}_session{session}_story{story_id}'


def default_analysis_output_dir(
    output_root: Path,
    *,
    run_group: str,
    run_name: str,
    run_id: str,
) -> Path:
    """Use the same run-group/run-name layout as the other plot analyses."""

    if not run_name.strip():
        raise ValueError('The selected run config has no usable run_name.')
    return output_root / run_group / f'{run_name}-{run_id}'


def stable_seed(seed: int, *parts: object) -> int:
    """Derive a deterministic uint64 seed without Python hash randomization."""

    payload = '|'.join([str(seed), *(str(part) for part in parts)]).encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder='little', signed=False)


def resolve_device(value: str) -> torch.device:
    if value == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if value.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('A CUDA device was requested, but CUDA is unavailable.')
    return torch.device(value)


def select_evaluation_batch_sizes(
    *,
    device_type: str,
    training_batch_size: int,
    test_batch_size: int,
    inference_override: int | None,
) -> tuple[int, int, str]:
    """Return canonical-baseline and optimized-perturbation batch sizes."""

    if training_batch_size <= 0 or test_batch_size <= 0:
        raise ValueError('Recorded training and test batch sizes must be positive.')
    canonical_baseline_batch_size = int(test_batch_size)
    if inference_override is not None:
        if inference_override <= 0:
            raise ValueError('Inference batch size must be positive.')
        return (
            canonical_baseline_batch_size,
            int(inference_override),
            'explicit_cli_override',
        )
    if device_type == 'cuda':
        return (
            canonical_baseline_batch_size,
            DEFAULT_CUDA_INFERENCE_BATCH_SIZE,
            'optimized_cuda_default',
        )
    return (
        canonical_baseline_batch_size,
        max(int(training_batch_size), canonical_baseline_batch_size),
        'optimized_cpu_default',
    )


def strict_one_based_ranks(
    similarity: torch.Tensor,
    labels: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    """Rank each true candidate against the complete supplied candidate bank.

    The bank is never filtered by query eligibility.  Ties with the true
    similarity do not outrank it, matching ``1 + count(score > true_score)``.
    """

    if similarity.ndim != 2:
        raise ValueError(f'similarity must be 2D, got {tuple(similarity.shape)}.')
    if not torch.isfinite(similarity).all():
        raise ValueError('similarity contains non-finite values.')
    if labels.ndim != 1 or labels.numel() != similarity.shape[0]:
        raise ValueError('labels must be 1D with one label per query.')
    if candidate_ids.ndim != 1 or candidate_ids.numel() != similarity.shape[1]:
        raise ValueError('candidate_ids must match the complete candidate bank.')
    if torch.unique(candidate_ids).numel() != candidate_ids.numel():
        raise ValueError('candidate_ids must be unique.')

    matches = labels[:, None] == candidate_ids[None, :]
    counts = matches.sum(dim=1)
    if not torch.all(counts == 1):
        bad = torch.where(counts != 1)[0].detach().cpu().tolist()
        raise ValueError(
            'Every query label must occur exactly once in the candidate bank. '
            f'Bad query rows: {bad[:10]}.'
        )
    true_similarity = similarity[matches]
    return 1 + torch.sum(similarity > true_similarity[:, None], dim=1)


def fixed_bank_ranks(
    embeddings: torch.Tensor,
    candidate_bank_normalized: torch.Tensor,
    labels: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    """Cosine ranks against a once-normalized complete candidate bank.

    The operations and reduction order intentionally match
    ``training.criteria.get_clip_similarity``.  Only the redundant candidate
    normalization is cached.
    """

    if embeddings.shape[1:] != candidate_bank_normalized.shape[1:]:
        raise ValueError(
            'Query and candidate embeddings must have matching non-batch '
            f'shapes, got query={tuple(embeddings.shape)} and '
            f'candidate={tuple(candidate_bank_normalized.shape)}.'
        )
    if embeddings.ndim == 3:
        query_normalized = torch_f.normalize(embeddings, dim=(-2, -1))
        similarity = torch.einsum(
            'Bef,bef->Bb',
            query_normalized,
            candidate_bank_normalized,
        )
    elif embeddings.ndim == 2:
        query_normalized = torch_f.normalize(embeddings, dim=-1)
        similarity = torch.einsum(
            'Bf,bf->Bb',
            query_normalized,
            candidate_bank_normalized,
        )
    else:
        raise NotImplementedError(
            'Expected 2D or 3D embeddings, got '
            f'query={tuple(embeddings.shape)} and '
            f'candidate={tuple(candidate_bank_normalized.shape)}.'
        )
    return strict_one_based_ranks(similarity, labels, candidate_ids)


def connected_components(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open runs of True values."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError(f'mask must be 1D, got {values.shape}.')
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation along time without wrapping."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError(f'mask must be 1D, got {values.shape}.')
    if radius < 0:
        raise ValueError(f'radius must be non-negative, got {radius}.')
    if radius == 0:
        return values.copy()
    kernel = np.ones(2 * radius + 1, dtype=np.int16)
    return np.convolve(values.astype(np.int16), kernel, mode='same') > 0


def inner_raised_cosine_alpha(mask: np.ndarray, taper_radius: int) -> np.ndarray:
    """Blend weights supported only inside ``mask``.

    For a five-frame point mask and a two-frame taper this yields
    ``[0.25, 0.75, 1, 0.75, 0.25]``.  Tapering after unioning occurrences avoids
    creating artificial internal splice boundaries.
    """

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError(f'mask must be 1D, got {values.shape}.')
    if taper_radius < 0:
        raise ValueError('taper_radius must be non-negative.')
    alpha = values.astype(np.float64)
    if taper_radius == 0:
        return alpha
    ramp = 0.5 - 0.5 * np.cos(
        np.pi * np.arange(1, taper_radius + 1) / (taper_radius + 1)
    )
    for start, stop in connected_components(values):
        length = stop - start
        for offset in range(min(taper_radius, length)):
            weight = float(ramp[offset])
            alpha[start + offset] = min(alpha[start + offset], weight)
            alpha[stop - 1 - offset] = min(alpha[stop - 1 - offset], weight)
        if length % 2 == 1:
            alpha[start + length // 2] = max(
                alpha[start + length // 2],
                float(ramp[min(taper_radius - 1, length // 2)]),
            )
    return alpha


def derive_event_mask(period: np.ndarray, event: str) -> np.ndarray:
    """Derive point-like onset/offset masks on a continuous sound timeline."""

    values = np.asarray(period, dtype=bool)
    if values.ndim != 1:
        raise ValueError(f'period must be 1D, got {values.shape}.')
    if event in {'period', 'point'}:
        return values.copy()
    previous = np.zeros_like(values)
    previous[1:] = values[:-1]
    if event == 'onset':
        return values & ~previous
    if event == 'offset':
        return ~values & previous
    if event == 'transition':
        return values != previous
    raise ValueError(f'Unknown event type {event!r}.')


def rasterize_intervals(
    intervals: Iterable[dict[str, Any]],
    length: int,
    hop_samples: int,
) -> np.ndarray:
    """Any-overlap rasterization of sound-relative audio intervals."""

    mask = np.zeros(length, dtype=bool)
    for interval in intervals:
        start_sample = int(interval['start_sample'])
        stop_sample = int(interval['end_sample'])
        if stop_sample <= start_sample:
            continue
        start = max(0, start_sample // hop_samples)
        stop = min(length, -(-stop_sample // hop_samples))
        if stop > start:
            mask[start:stop] = True
    return mask


def sound_frame_lengths(
    sound_names: Sequence[str],
    audio_dir: Path,
    hop_samples: int,
) -> dict[str, int]:
    """Use complete audio files while refusing to fabricate a trailing frame."""

    lengths: dict[str, int] = {}
    for name in sound_names:
        info = sf.info(audio_dir / name)
        if int(info.samplerate) <= 0:
            raise ValueError(f'{name!r} has invalid sample rate {info.samplerate}.')
        # ``load_audio_mono_16k``/librosa uses ceil for the resampled length.
        n_samples = (int(info.frames) * AUDIO_SR + int(info.samplerate) - 1) // int(
            info.samplerate
        )
        length = n_samples // hop_samples
        if length <= 0:
            raise ValueError(f'Audio file {name!r} is shorter than one analysis frame.')
        lengths[name] = length
    return lengths


def slice_window_trace(
    trace: np.ndarray,
    wav_start: int,
    wav_stop: int,
    hop_samples: int,
) -> np.ndarray:
    """Slice one test window, requiring exact alignment to the MEG grid."""

    if wav_start % hop_samples or wav_stop % hop_samples:
        raise ValueError(
            'Audio window boundaries must align exactly to the MEG grid. '
            f'Got wav_start={wav_start}, wav_stop={wav_stop}, hop={hop_samples}.'
        )
    start = wav_start // hop_samples
    stop = wav_stop // hop_samples
    if start < 0 or stop > trace.shape[0] or stop <= start:
        raise ValueError(
            f'Window [{start}, {stop}) is outside trace length {trace.shape[0]}.'
        )
    return trace[start:stop]


def nonoverlapping_candidate_starts(mask: np.ndarray, length: int) -> list[int]:
    """Tile state runs into genuinely non-overlapping length-matched donors."""

    if length <= 0:
        raise ValueError(f'length must be positive, got {length}.')
    starts: list[int] = []
    for run_start, run_stop in connected_components(mask):
        count = (run_stop - run_start) // length
        if count <= 0:
            continue
        slack = (run_stop - run_start) - count * length
        first = run_start + slack // 2
        starts.extend(first + index * length for index in range(count))
    return starts


def build_donor_candidate_pool(
    present: dict[str, np.ndarray],
    absent: dict[str, np.ndarray],
    length: int,
) -> DonorCandidatePool:
    """Scan donor traces once for one target-component length."""

    candidates: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    total_present = 0
    total_absent = 0
    for sound in sorted(set(present) & set(absent)):
        present_starts = tuple(nonoverlapping_candidate_starts(present[sound], length))
        absent_starts = tuple(nonoverlapping_candidate_starts(absent[sound], length))
        total_present += len(present_starts)
        total_absent += len(absent_starts)
        if present_starts and absent_starts:
            candidates[sound] = (present_starts, absent_starts)
    return DonorCandidatePool(
        length=length,
        by_sound=candidates,
        total_present=total_present,
        total_absent=total_absent,
        same_file_capacity=sum(
            min(len(present_starts), len(absent_starts))
            for present_starts, absent_starts in candidates.values()
        ),
    )


def sample_donor_candidate_pool(
    pool: DonorCandidatePool,
    n_pairs: int,
    rng: np.random.Generator,
) -> tuple[tuple[DonorPair, ...] | None, str, str]:
    """Choose K distinct pairs from a cached pool, preferring proximity."""

    if n_pairs <= 0:
        raise ValueError('n_pairs must be positive.')
    if pool.total_present < n_pairs:
        return (
            None,
            'insufficient_feature_present_donors',
            f'needed={n_pairs}, available_nonoverlap={pool.total_present}, '
            f'length={pool.length}',
        )
    if pool.total_absent < n_pairs:
        return (
            None,
            'insufficient_feature_absent_donors',
            f'needed={n_pairs}, available_nonoverlap={pool.total_absent}, '
            f'length={pool.length}',
        )
    if pool.same_file_capacity < n_pairs:
        return (
            None,
            'insufficient_same_file_donor_pairs',
            f'needed={n_pairs}, same_file_capacity={pool.same_file_capacity}, '
            f'length={pool.length}',
        )

    # K is small, while a donor trace can contain many thousands of starts.
    # Track only the selected tuple indices instead of copying every candidate
    # list afresh for every target component.
    used_present: dict[str, set[int]] = {sound: set() for sound in pool.by_sound}
    used_absent: dict[str, set[int]] = {sound: set() for sound in pool.by_sound}

    def nth_unused_index(size: int, used: set[int], rank: int) -> int:
        """Map an index in the remaining sequence to its original tuple index."""

        candidate = rank
        for removed in sorted(used):
            if removed <= candidate:
                candidate += 1
            else:
                break
        if candidate >= size:
            raise RuntimeError('Failed to map a donor index into its cached pool.')
        return candidate

    pairs: list[DonorPair] = []
    for _ in range(n_pairs):
        available_sounds = [
            sound
            for sound, (present_starts, absent_starts) in pool.by_sound.items()
            if len(present_starts) > len(used_present[sound])
            and len(absent_starts) > len(used_absent[sound])
        ]
        weights = np.array(
            [
                min(
                    len(pool.by_sound[sound][0]) - len(used_present[sound]),
                    len(pool.by_sound[sound][1]) - len(used_absent[sound]),
                )
                for sound in available_sounds
            ],
            dtype=np.float64,
        )
        sound = str(rng.choice(available_sounds, p=weights / weights.sum()))
        present_starts, absent_starts = pool.by_sound[sound]
        present_rank = int(rng.integers(len(present_starts) - len(used_present[sound])))
        present_index = nth_unused_index(
            len(present_starts),
            used_present[sound],
            present_rank,
        )
        used_present[sound].add(present_index)
        present_start = present_starts[present_index]
        insertion = bisect_left(absent_starts, present_start)
        left = insertion - 1
        while left >= 0 and left in used_absent[sound]:
            left -= 1
        right = insertion
        while right < len(absent_starts) and right in used_absent[sound]:
            right += 1
        nearest_indices: list[int] = []
        if left >= 0:
            nearest_indices.append(left)
        if right < len(absent_starts):
            nearest_indices.append(right)
        if not nearest_indices:
            raise RuntimeError('A feasible donor pool unexpectedly became empty.')
        minimum_distance = min(
            abs(absent_starts[index] - present_start) for index in nearest_indices
        )
        nearest = [
            index
            for index in nearest_indices
            if abs(absent_starts[index] - present_start) == minimum_distance
        ]
        absent_index = int(rng.choice(nearest))
        used_absent[sound].add(absent_index)
        absent_start = absent_starts[absent_index]
        pairs.append(
            DonorPair(
                absent=DonorInterval(
                    sound,
                    absent_start,
                    absent_start + pool.length,
                ),
                present=DonorInterval(
                    sound,
                    present_start,
                    present_start + pool.length,
                ),
            )
        )
    return tuple(pairs), '', ''


def pair_donor_candidates(
    present: dict[str, np.ndarray],
    absent: dict[str, np.ndarray],
    length: int,
    n_pairs: int,
    rng: np.random.Generator,
) -> tuple[tuple[DonorPair, ...] | None, str, str]:
    """Compatibility wrapper that builds then samples one donor pool."""

    pool = build_donor_candidate_pool(present, absent, length)
    return sample_donor_candidate_pool(pool, n_pairs, rng)


def apply_component_plan(
    target: torch.Tensor,
    plan: WindowPlan,
    replicate: int,
    donor_reader: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create one complete f->0/f->f query pair without modifying ``target``."""

    absent_query = target.clone()
    present_query = target.clone()
    for component in plan.components:
        pair = component.donor_pairs[replicate]
        donor_absent = donor_reader(pair.absent)
        donor_present = donor_reader(pair.present)
        expected = (target.shape[0], component.stop - component.start)
        if (
            tuple(donor_absent.shape) != expected
            or tuple(donor_present.shape) != expected
        ):
            raise ValueError(
                'Donor shape mismatch: '
                f'expected={expected}, absent={tuple(donor_absent.shape)}, '
                f'present={tuple(donor_present.shape)}.'
            )
        alpha = torch.as_tensor(
            component.alpha,
            dtype=target.dtype,
            device=target.device,
        )[None, :]
        original = target[:, component.start : component.stop]
        absent_query[:, component.start : component.stop] = (
            1.0 - alpha
        ) * original + alpha * donor_absent
        present_query[:, component.start : component.stop] = (
            1.0 - alpha
        ) * original + alpha * donor_present
    return absent_query, present_query


def paired_feature_metrics(
    baseline_rank: np.ndarray,
    rank_absent: np.ndarray,
    rank_present: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, float]:
    """Compute every arm on the identical eligible-query subset.

    ``rank_absent`` and ``rank_present`` are shaped ``(windows, donor_pairs)``.
    The candidate bank used to obtain those ranks is unrelated to this filter
    and must already be complete.
    """

    base = np.asarray(baseline_rank)
    absent = np.asarray(rank_absent)
    present = np.asarray(rank_present)
    keep = np.asarray(eligible, dtype=bool)
    if base.ndim != 1:
        raise ValueError('baseline_rank must be 1D.')
    if absent.ndim != 2 or present.shape != absent.shape:
        raise ValueError('Occlusion ranks must have matching 2D shapes.')
    if absent.shape[0] != base.shape[0] or keep.shape != base.shape:
        raise ValueError('All rank arrays and eligibility must share window axis.')
    if not np.any(keep):
        raise ValueError('No eligible queries remain.')
    if not np.isfinite(absent[keep]).all() or not np.isfinite(present[keep]).all():
        raise ValueError('Eligible occlusion ranks contain missing/non-finite values.')

    base_kept = base[keep].astype(np.float64)
    absent_kept = absent[keep].astype(np.float64)
    present_kept = present[keep].astype(np.float64)
    effect = absent_kept - present_kept
    absent_damage = absent_kept - base_kept[:, None]
    present_damage = present_kept - base_kept[:, None]
    return {
        'n_queries': int(keep.sum()),
        'baseline_mean_rank': float(base_kept.mean()),
        'f_to_absent_mean_rank': float(absent_kept.mean()),
        'f_to_present_mean_rank': float(present_kept.mean()),
        'effect_rank_absent_minus_present': float(effect.mean()),
        'damage_rank_absent_minus_baseline': float(absent_damage.mean()),
        'damage_rank_present_minus_baseline': float(present_damage.mean()),
    }


def feature_specs_from_json(path: Path) -> tuple[FeatureSpec, ...]:
    """Load an optional predeclared battery without accepting unknown fields."""

    raw = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f'{path}: expected a non-empty JSON list.')
    specs: list[FeatureSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f'{path}: every feature specification must be an object.')
        if 'phonemes' in item:
            item = {**item, 'phonemes': tuple(item['phonemes'])}
        specs.append(FeatureSpec(**item))
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(f'{path}: feature names must be unique.')
    return tuple(specs)


def _validate_trace_dict(
    traces: dict[str, np.ndarray],
    sound_lengths: dict[str, int],
    context: str,
) -> dict[str, np.ndarray]:
    if set(traces) != set(sound_lengths):
        missing = sorted(set(sound_lengths) - set(traces))
        extra = sorted(set(traces) - set(sound_lengths))
        raise ValueError(
            f'{context}: sound mismatch; missing={missing}, extra={extra}.'
        )
    validated: dict[str, np.ndarray] = {}
    for sound, length in sound_lengths.items():
        trace = np.asarray(traces[sound], dtype=bool)
        if trace.shape != (length,):
            raise ValueError(
                f'{context}: {sound!r} has shape {trace.shape}, expected {(length,)}.'
            )
        validated[sound] = trace
    return validated


def validate_word_feature_sound_coverage(
    word_features: pd.DataFrame,
    expected_sounds: Sequence[str],
) -> None:
    """Refuse to turn missing annotation files into feature-absent windows."""

    if 'sound_fname' not in word_features.columns:
        raise ValueError('Word-feature table is missing the sound_fname column.')
    observed = set(word_features['sound_fname'].astype(str).unique())
    missing = sorted(set(str(sound) for sound in expected_sounds) - observed)
    if missing:
        raise ValueError(
            f'Word-feature table has no rows for expected audio files: {missing}.'
        )


def _textgrid_period_masks(
    spec: FeatureSpec,
    sound_lengths: dict[str, int],
    mfa_dir: Path,
    hop_samples: int,
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    observed_labels: set[str] = set()
    for sound, length in sound_lengths.items():
        path = mfa_dir / Path(sound).with_suffix('.TextGrid').name
        if not path.exists():
            raise FileNotFoundError(f'TextGrid not found: {path}')
        if spec.builder == 'textgrid_silence':
            intervals = [
                interval
                for interval in parse_textgrid_tier_exact(path, 'words')
                if interval['label'] == ''
            ]
        elif spec.builder == 'phoneme':
            selected = set(spec.phonemes)
            phone_intervals = parse_textgrid_tier_exact(path, 'phones')
            observed_labels.update(
                str(interval['label'])
                for interval in phone_intervals
                if interval['label'] != ''
            )
            intervals = [
                interval
                for interval in phone_intervals
                if interval['label'] in selected
            ]
        else:  # pragma: no cover - caller validates builders
            raise ValueError(f'Unsupported TextGrid builder {spec.builder!r}.')
        masks[sound] = rasterize_intervals(intervals, length, hop_samples)
    if spec.builder == 'phoneme':
        missing_labels = sorted(set(spec.phonemes) - observed_labels)
        if missing_labels:
            raise ValueError(
                f'{spec.name}: configured phoneme labels do not occur in the '
                f'selected TextGrids: {missing_labels}.'
            )
    return masks


def _other_phoneme_masks(
    spec: FeatureSpec,
    sound_lengths: dict[str, int],
    mfa_dir: Path,
    hop_samples: int,
) -> dict[str, np.ndarray]:
    """Other labeled phones, excluding silence, form the phoneme-class control."""

    selected = set(spec.phonemes)
    masks: dict[str, np.ndarray] = {}
    for sound, length in sound_lengths.items():
        path = mfa_dir / Path(sound).with_suffix('.TextGrid').name
        intervals = [
            interval
            for interval in parse_textgrid_tier_exact(path, 'phones')
            if interval['label'] != '' and interval['label'] not in selected
        ]
        masks[sound] = rasterize_intervals(intervals, length, hop_samples)
    return masks


def _word_binary_masks(
    spec: FeatureSpec,
    word_features: pd.DataFrame,
    sound_lengths: dict[str, int],
    hop_samples: int,
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for sound, length in sound_lengths.items():
        rows = word_features[word_features['sound_fname'] == sound]
        if spec.builder == 'insertion':
            rows = rows[rows['insertion_kind'] == spec.insertion_kind]
        intervals = rows[['start_sample', 'end_sample']].to_dict(orient='records')
        period = rasterize_intervals(intervals, length, hop_samples)
        if spec.builder == 'word_onset':
            onset = np.zeros_like(period)
            for row in rows.itertuples(index=False):
                frame = min(length - 1, max(0, int(row.start_sample) // hop_samples))
                onset[frame] = True
            masks[sound] = onset
        else:
            masks[sound] = period
    return masks


def _regular_word_masks(
    word_features: pd.DataFrame,
    sound_lengths: dict[str, int],
    hop_samples: int,
) -> dict[str, np.ndarray]:
    """Unmanipulated narrative words for pseudoword/word-list controls."""

    masks: dict[str, np.ndarray] = {}
    for sound, length in sound_lengths.items():
        rows = word_features[
            (word_features['sound_fname'] == sound)
            & (word_features['insertion_kind'] == 'none')
        ]
        masks[sound] = rasterize_intervals(
            rows[['start_sample', 'end_sample']].to_dict(orient='records'),
            length,
            hop_samples,
        )
    return masks


def _word_value_masks(
    spec: FeatureSpec,
    word_features: pd.DataFrame,
    sound_lengths: dict[str, int],
    donor_sounds: set[str],
    hop_samples: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    if spec.column is None:
        raise ValueError(f'{spec.name}: word_value requires column.')
    training_rows = word_features[
        word_features['sound_fname'].astype(str).isin(donor_sounds)
        & (word_features['insertion_kind'].astype(str) == 'none')
    ]
    values = training_rows[spec.column].to_numpy(dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f'{spec.name}: training word values are empty or non-finite.')
    q25, q75 = np.quantile(values, [0.25, 0.75])
    if not q25 < q75:
        raise ValueError(
            f'{spec.name}: lower/upper quartiles are not separated: {q25}, {q75}.'
        )
    present: dict[str, np.ndarray] = {}
    absent: dict[str, np.ndarray] = {}
    for sound, length in sound_lengths.items():
        rows = word_features[
            (word_features['sound_fname'] == sound)
            & (word_features['insertion_kind'].astype(str) == 'none')
        ]
        sound_values = rows[spec.column].to_numpy(dtype=np.float64)
        if sound_values.size and not np.isfinite(sound_values).all():
            raise ValueError(f'{spec.name}: non-finite word values in {sound!r}.')
        if spec.high_is_present:
            present_rows = rows[sound_values >= q75]
            absent_rows = rows[sound_values <= q25]
            direction = 'upper_quartile_vs_lower_quartile'
        else:
            present_rows = rows[sound_values <= q25]
            absent_rows = rows[sound_values >= q75]
            direction = 'lower_quartile_vs_upper_quartile'
        present[sound] = rasterize_intervals(
            present_rows[['start_sample', 'end_sample']].to_dict(orient='records'),
            length,
            hop_samples,
        )
        absent[sound] = rasterize_intervals(
            absent_rows[['start_sample', 'end_sample']].to_dict(orient='records'),
            length,
            hop_samples,
        )
    return (
        present,
        absent,
        {
            'lower_quartile': float(q25),
            'upper_quartile': float(q75),
            'threshold_source': 'donor_audio_files_only',
            'contrast': direction,
            'threshold_unit': 'word',
            'word_scope': 'unmanipulated_narrative_words_only',
        },
    )


def _audio_value_masks(
    spec: FeatureSpec,
    sound_lengths: dict[str, int],
    donor_sounds: set[str],
    audio_dir: Path,
    hop_samples: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    if spec.column is None:
        raise ValueError(f'{spec.name}: audio_value requires column.')
    names = sorted(sound_lengths)
    traces = audio_feature_trace(
        audio_dir=audio_dir,
        sound_names=names,
        sound_length=sound_lengths,
        hop=hop_samples,
        kind=spec.column,
        rank=False,
    )
    train_values = np.concatenate(
        [np.asarray(traces[sound], dtype=np.float64) for sound in sorted(donor_sounds)]
    )
    if train_values.size == 0 or not np.isfinite(train_values).all():
        raise ValueError(f'{spec.name}: invalid donor-audio feature values.')
    q25, q75 = np.quantile(train_values, [0.25, 0.75])
    if not q25 < q75:
        raise ValueError(
            f'{spec.name}: lower/upper quartiles are not separated: {q25}, {q75}.'
        )
    present: dict[str, np.ndarray] = {}
    absent: dict[str, np.ndarray] = {}
    for sound, trace in traces.items():
        values = np.asarray(trace, dtype=np.float64)
        if spec.high_is_present:
            present[sound] = values >= q75
            absent[sound] = values <= q25
            direction = 'upper_quartile_vs_lower_quartile'
        else:
            present[sound] = values <= q25
            absent[sound] = values >= q75
            direction = 'lower_quartile_vs_upper_quartile'
    return (
        present,
        absent,
        {
            'lower_quartile': float(q25),
            'upper_quartile': float(q75),
            'threshold_source': 'donor_audio_files_only',
            'contrast': direction,
            'threshold_unit': 'meg_frame',
        },
    )


def build_feature_state(
    spec: FeatureSpec,
    sound_lengths: dict[str, int],
    donor_sounds: set[str],
    *,
    padding_frames: int,
    hop_samples: int,
    audio_dir: Path,
    mfa_dir: Path,
    word_features: pd.DataFrame | None,
) -> FeatureState:
    """Build train-defined present/absent states, then apply one fixed dilation."""

    metadata: dict[str, Any] = {}
    if spec.builder in {'textgrid_silence', 'phoneme'}:
        base = _textgrid_period_masks(spec, sound_lengths, mfa_dir, hop_samples)
        present_base = {
            sound: derive_event_mask(mask, spec.event) for sound, mask in base.items()
        }
        if spec.builder == 'phoneme':
            absent_base = _other_phoneme_masks(
                spec,
                sound_lengths,
                mfa_dir,
                hop_samples,
            )
            metadata['absent_pool'] = 'other_labeled_phonemes'
        else:
            absent_base = None
            metadata['absent_pool'] = 'non_feature_time'
    elif spec.builder in {'word_onset', 'insertion'}:
        if word_features is None:
            raise ValueError(
                f'{spec.name} requires the precomputed word-features table.'
            )
        present_base = _word_binary_masks(
            spec, word_features, sound_lengths, hop_samples
        )
        if spec.builder == 'insertion':
            absent_base = _regular_word_masks(
                word_features,
                sound_lengths,
                hop_samples,
            )
            metadata['absent_pool'] = 'unmanipulated_narrative_words'
        else:
            absent_base = None
            metadata['absent_pool'] = 'non_feature_time'
    elif spec.builder == 'word_value':
        if word_features is None:
            raise ValueError(
                f'{spec.name} requires the precomputed word-features table.'
            )
        present_base, absent_base, metadata = _word_value_masks(
            spec,
            word_features,
            sound_lengths,
            donor_sounds,
            hop_samples,
        )
    elif spec.builder == 'audio_value':
        present_base, absent_base, metadata = _audio_value_masks(
            spec,
            sound_lengths,
            donor_sounds,
            audio_dir,
            hop_samples,
        )
    else:
        raise ValueError(f'{spec.name}: unknown builder {spec.builder!r}.')

    present: dict[str, np.ndarray] = {}
    absent: dict[str, np.ndarray] = {}
    for sound in sorted(sound_lengths):
        expanded_present = dilate_mask(present_base[sound], padding_frames)
        if absent_base is None:
            expanded_absent = ~expanded_present
        else:
            expanded_absent = dilate_mask(absent_base[sound], padding_frames)
            overlap = expanded_present & expanded_absent
            if np.any(overlap):
                expanded_absent = expanded_absent & ~overlap
        present[sound] = expanded_present
        absent[sound] = expanded_absent

    present = _validate_trace_dict(present, sound_lengths, f'{spec.name}/present')
    absent = _validate_trace_dict(absent, sound_lengths, f'{spec.name}/absent')
    metadata.update(
        {
            'padding_frames': int(padding_frames),
            'padding_ms': float(1000.0 * padding_frames * hop_samples / AUDIO_SR),
            'event': spec.event,
            'builder': spec.builder,
        }
    )
    return FeatureState(
        spec=spec,
        present=present,
        absent=absent,
        metadata=metadata,
    )


def build_test_windows(df_test: pd.DataFrame) -> pd.DataFrame:
    """Return one canonical row per test candidate and validate shared stimuli."""

    required = {
        'subject_id',
        'session_id',
        'story_id',
        'sound_fname',
        'wav_start',
        'wav_stop',
        'wav_index',
    }
    missing = required - set(df_test.columns)
    if missing:
        raise ValueError(f'Test dataframe is missing columns {sorted(missing)}.')
    grouped = df_test.groupby('wav_index', sort=True)
    columns = ['story_id', 'sound_fname', 'wav_start', 'wav_stop']
    for column in columns:
        bad = grouped[column].nunique(dropna=False)
        if np.any(bad.to_numpy() != 1):
            ids = bad[bad != 1].index.tolist()
            raise ValueError(f'wav_index maps to multiple {column} values: {ids[:10]}.')
    windows = grouped[columns].first().reset_index()
    expected = np.arange(len(windows), dtype=np.int64)
    observed = windows['wav_index'].to_numpy(dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise ValueError('Test wav_index must be contiguous 0..N-1 for the fixed bank.')
    combo_counts = df_test.groupby(['subject_id', 'session_id']).size()
    if not np.all(combo_counts.to_numpy() == len(windows)):
        raise ValueError('Every subject-session must contain the complete test bank.')
    return windows


def select_donor_sounds(
    df_train: pd.DataFrame,
    story_id: int,
    donor_sound_ids: Sequence[int],
) -> set[str]:
    rows = df_train[
        (df_train['story_id'].astype(int) == story_id)
        & df_train['sound_id']
        .astype(int)
        .isin([int(value) for value in donor_sound_ids])
    ]
    found_ids = set(rows['sound_id'].astype(int).unique().tolist())
    expected_ids = set(int(value) for value in donor_sound_ids)
    if found_ids != expected_ids:
        raise ValueError(
            f'Donor sound-id mismatch: expected={sorted(expected_ids)}, '
            f'found={sorted(found_ids)}.'
        )
    return set(rows['sound_fname'].astype(str).unique().tolist())


def build_window_plans(
    state: FeatureState,
    test_windows: pd.DataFrame,
    donor_sounds: set[str],
    *,
    hop_samples: int,
    taper_frames: int,
    n_donor_pairs: int,
    seed: int,
    donor_candidate_cache: dict[int, DonorCandidatePool] | None = None,
) -> tuple[dict[int, WindowPlan], list[EligibilityRecord]]:
    """Audit and plan every test window before running any model forward."""

    if donor_candidate_cache is None:
        donor_candidate_cache = {}
    donor_present = {sound: state.present[sound] for sound in donor_sounds}
    donor_absent = {sound: state.absent[sound] for sound in donor_sounds}
    alpha_by_sound = {
        sound: inner_raised_cosine_alpha(mask, taper_frames)
        for sound, mask in state.present.items()
    }
    plans: dict[int, WindowPlan] = {}
    records: list[EligibilityRecord] = []
    for row in test_windows.itertuples(index=False):
        wav_index = int(row.wav_index)
        sound = str(row.sound_fname)
        support = slice_window_trace(
            state.present[sound],
            int(row.wav_start),
            int(row.wav_stop),
            hop_samples,
        )
        alpha = slice_window_trace(
            alpha_by_sound[sound],
            int(row.wav_start),
            int(row.wav_stop),
            hop_samples,
        )
        support_fraction = float(np.mean(support))
        weighted_fraction = float(np.mean(alpha))
        components = connected_components(support)
        base = {
            'feature': state.spec.name,
            'wav_index': wav_index,
            'sound_fname': sound,
            'support_fraction': support_fraction,
            'weighted_fraction': weighted_fraction,
            'n_components': len(components),
        }
        if not components:
            records.append(
                EligibilityRecord(
                    **base,
                    status='discarded',
                    reason='feature_absent',
                )
            )
            continue
        if bool(np.all(support)):
            records.append(
                EligibilityRecord(
                    **base,
                    status='discarded',
                    reason='full_mask',
                )
            )
            continue

        component_plans: list[ComponentPlan] = []
        failure_reason = ''
        failure_detail = ''
        for component_index, (start, stop) in enumerate(components):
            length = stop - start
            rng = np.random.default_rng(
                stable_seed(seed, state.spec.name, wav_index, component_index)
            )
            pool = donor_candidate_cache.get(length)
            if pool is None:
                pool = build_donor_candidate_pool(
                    donor_present,
                    donor_absent,
                    length,
                )
                donor_candidate_cache[length] = pool
            pairs, reason, detail = sample_donor_candidate_pool(
                pool,
                n_pairs=n_donor_pairs,
                rng=rng,
            )
            if pairs is None:
                failure_reason = reason
                failure_detail = (
                    f'component={component_index}, target=[{start},{stop}); {detail}'
                )
                break
            component_plans.append(
                ComponentPlan(
                    start=start,
                    stop=stop,
                    alpha=np.asarray(alpha[start:stop], dtype=np.float64),
                    donor_pairs=pairs,
                )
            )
        if failure_reason:
            records.append(
                EligibilityRecord(
                    **base,
                    status='discarded',
                    reason=failure_reason,
                    detail=failure_detail,
                )
            )
            continue

        sound_start_frame = int(row.wav_start) // hop_samples
        plans[wav_index] = WindowPlan(
            wav_index=wav_index,
            sound_fname=sound,
            sound_start_frame=sound_start_frame,
            support_fraction=support_fraction,
            weighted_fraction=weighted_fraction,
            components=tuple(component_plans),
        )
        records.append(
            EligibilityRecord(
                **base,
                status='eligible',
                reason='',
            )
        )
    if len(records) != len(test_windows):
        raise RuntimeError('Eligibility audit did not produce one record per window.')
    return plans, records


def build_participant_feature_plans(
    *,
    subject: int,
    base_feature_plans: dict[str, dict[int, WindowPlan]],
    feature_donor_candidate_caches: dict[str, dict[int, DonorCandidatePool]],
    feature_names: Sequence[str],
    n_donor_pairs: int,
    analysis_seed: int,
) -> dict[str, dict[int, WindowPlan]]:
    """Redraw only donor timestamps, reusing audited geometry and pools."""

    participant_seed = stable_seed(analysis_seed, 'participant_donors', subject)
    result: dict[str, dict[int, WindowPlan]] = {}
    for name in feature_names:
        plans: dict[int, WindowPlan] = {}
        for wav_index, base_plan in base_feature_plans[name].items():
            components: list[ComponentPlan] = []
            for component_index, component in enumerate(base_plan.components):
                length = component.stop - component.start
                pool = feature_donor_candidate_caches[name][length]
                rng = np.random.default_rng(
                    stable_seed(
                        participant_seed,
                        name,
                        wav_index,
                        component_index,
                    )
                )
                pairs, reason, detail = sample_donor_candidate_pool(
                    pool,
                    n_pairs=n_donor_pairs,
                    rng=rng,
                )
                if pairs is None:
                    raise RuntimeError(
                        'An audited donor pool became infeasible while redrawing '
                        f'{name!r}, subject={subject}, wav_index={wav_index}, '
                        f'component={component_index}: {reason}; {detail}'
                    )
                components.append(
                    ComponentPlan(
                        start=component.start,
                        stop=component.stop,
                        alpha=component.alpha,
                        donor_pairs=pairs,
                    )
                )
            plans[wav_index] = WindowPlan(
                wav_index=base_plan.wav_index,
                sound_fname=base_plan.sound_fname,
                sound_start_frame=base_plan.sound_start_frame,
                support_fraction=base_plan.support_fraction,
                weighted_fraction=base_plan.weighted_fraction,
                components=tuple(components),
            )
        result[name] = plans
    return result


def summarize_eligibility(records: Sequence[EligibilityRecord]) -> dict[str, Any]:
    """Counts reconcile by construction and are persisted per feature."""

    if not records:
        raise ValueError('Cannot summarize empty eligibility records.')
    statuses = Counter(record.status for record in records)
    reasons = Counter(
        record.reason for record in records if record.status == 'discarded'
    )
    total = len(records)
    eligible = int(statuses.get('eligible', 0))
    discarded = int(statuses.get('discarded', 0))
    if eligible + discarded != total or sum(reasons.values()) != discarded:
        raise RuntimeError('Eligibility counts do not reconcile.')
    eligible_records = [record for record in records if record.status == 'eligible']
    return {
        'n_total_windows': total,
        'n_eligible_windows': eligible,
        'n_discarded_windows': discarded,
        'discard_reasons': dict(sorted(reasons.items())),
        'eligible_fraction': float(eligible / total),
        'eligible_support_fraction_mean': (
            float(np.mean([record.support_fraction for record in eligible_records]))
            if eligible_records
            else float('nan')
        ),
        'eligible_support_fraction_median': (
            float(np.median([record.support_fraction for record in eligible_records]))
            if eligible_records
            else float('nan')
        ),
        'eligible_weighted_fraction_mean': (
            float(np.mean([record.weighted_fraction for record in eligible_records]))
            if eligible_records
            else float('nan')
        ),
    }


def _mean_or_nan(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float('nan')
    return float(np.mean(array))


def concatenate_test_present_mask(
    present: dict[str, np.ndarray],
    test_windows: pd.DataFrame,
) -> np.ndarray:
    """Exact dilated present mask on test sounds, concatenated in first-seen order."""

    if 'sound_fname' not in test_windows.columns:
        raise ValueError('test_windows must include sound_fname.')
    sounds = list(dict.fromkeys(test_windows['sound_fname'].astype(str).tolist()))
    if not sounds:
        raise ValueError('test_windows does not name any test sounds.')
    missing = [sound for sound in sounds if sound not in present]
    if missing:
        raise KeyError(f'Present masks missing test sounds: {missing[:10]}.')
    parts = [np.asarray(present[sound], dtype=bool).reshape(-1) for sound in sounds]
    return np.concatenate(parts)


def pairwise_jaccard(
    masks: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise Jaccard and directed containment for equal-length binary masks.

    Jaccard of two empty masks is 1. Directed containment is NaN when the
    source mask has empty support.
    """

    names = list(masks)
    if not names:
        raise ValueError('pairwise_jaccard requires at least one mask.')
    arrays = [np.asarray(masks[name], dtype=bool).reshape(-1) for name in names]
    lengths = {array.size for array in arrays}
    if len(lengths) != 1:
        raise ValueError('All masks must have the same length for Jaccard overlap.')
    n_features = len(names)
    jaccard = np.zeros((n_features, n_features), dtype=np.float64)
    containment = np.full((n_features, n_features), np.nan, dtype=np.float64)
    supports = [int(array.sum()) for array in arrays]
    for i, left in enumerate(arrays):
        for j, right in enumerate(arrays):
            intersection = int(np.logical_and(left, right).sum())
            union = int(np.logical_or(left, right).sum())
            jaccard[i, j] = 1.0 if union == 0 else intersection / union
            if supports[i] > 0:
                containment[i, j] = intersection / supports[i]
    return (
        pd.DataFrame(jaccard, index=names, columns=names),
        pd.DataFrame(containment, index=names, columns=names),
    )


def plot_jaccard_heatmap(jaccard: pd.DataFrame, output_path: Path) -> None:
    """Labelled square heatmap of pairwise feature-mask Jaccard overlap."""

    if jaccard.empty:
        raise ValueError('Cannot plot an empty Jaccard matrix.')
    if jaccard.shape[0] != jaccard.shape[1]:
        raise ValueError('Jaccard heatmap requires a square matrix.')
    labels = [
        PLOT_DISPLAY_NAMES.get(str(name), str(name)) for name in jaccard.columns
    ]
    values = jaccard.to_numpy(dtype=np.float64)
    n_features = len(labels)
    figure_size = max(4.5, 0.38 * n_features + 2.4)
    fig, ax = plt.subplots(figsize=(figure_size, figure_size * 0.92))
    image = ax.imshow(values, vmin=0.0, vmax=1.0, cmap='viridis', origin='upper')
    ax.set_xticks(np.arange(n_features))
    ax.set_yticks(np.arange(n_features))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_title('Feature-mask Jaccard overlap')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label='Jaccard')
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def mask_interval_lengths(mask: np.ndarray) -> list[int]:
    return [stop - start for start, stop in connected_components(mask)]


def summarize_feature_diagnostics(
    state: FeatureState,
    *,
    test_windows: pd.DataFrame,
    donor_sounds: set[str],
    plans: dict[int, WindowPlan],
    donor_cache: dict[int, DonorCandidatePool],
    eligibility: dict[str, Any],
) -> dict[str, Any]:
    """Feature diagnostics derived from the canonical occlusion objects."""

    test_mask = concatenate_test_present_mask(state.present, test_windows)
    test_interval_lengths = mask_interval_lengths(test_mask)
    donor_present_intervals = 0
    donor_absent_intervals = 0
    donor_present_frames = 0
    donor_absent_frames = 0
    for sound in sorted(donor_sounds):
        present = np.asarray(state.present[sound], dtype=bool)
        absent = np.asarray(state.absent[sound], dtype=bool)
        donor_present_frames += int(present.sum())
        donor_absent_frames += int(absent.sum())
        donor_present_intervals += len(connected_components(present))
        donor_absent_intervals += len(connected_components(absent))
    component_lengths = [
        component.stop - component.start
        for plan in plans.values()
        for component in plan.components
    ]
    pool_present = int(sum(pool.total_present for pool in donor_cache.values()))
    pool_absent = int(sum(pool.total_absent for pool in donor_cache.values()))
    n_test_frames = int(test_mask.size)
    n_test_support_frames = int(test_mask.sum())
    return {
        'n_eligible_test_queries': int(eligibility['n_eligible_windows']),
        'n_test_support_frames': n_test_support_frames,
        'test_support_fraction': (
            float(n_test_support_frames / n_test_frames) if n_test_frames else float('nan')
        ),
        'n_test_feature_intervals': len(test_interval_lengths),
        'mean_test_interval_length': _mean_or_nan(test_interval_lengths),
        'mean_eligible_component_length': _mean_or_nan(component_lengths),
        'n_eligible_components': int(len(component_lengths)),
        'donor_present_frames': donor_present_frames,
        'donor_absent_frames': donor_absent_frames,
        'donor_present_intervals': donor_present_intervals,
        'donor_absent_intervals': donor_absent_intervals,
        'donor_present_pool_starts': pool_present,
        'donor_absent_pool_starts': pool_absent,
        'mean_meg_replaced_fraction': eligibility[
            'eligible_weighted_fraction_mean'
        ],
        'donor_pools_by_length': {
            str(length): {
                'total_present': int(pool.total_present),
                'total_absent': int(pool.total_absent),
                'same_file_capacity': int(pool.same_file_capacity),
            }
            for length, pool in sorted(donor_cache.items())
        },
    }


def load_reduction_axis(
    hyper_params: dict[str, Any],
) -> tuple[str | None, int | None]:
    if hyper_params['feature_reduction'] == 'pca':
        return 'feature', int(hyper_params['feature_reduction_dim'])
    if hyper_params['time_reduction'] == 'pca':
        return 'time', int(hyper_params['time_reduction_dim'])
    return None, None


def load_analysis_inputs(
    *,
    run_id: str,
    run_group: str,
    experiments_root: Path,
    device: torch.device,
) -> AnalysisInputs:
    """Load metadata and models without materializing the 16 GB MEG archive."""

    run_dir, recorded_hyper_params = _load_run_dir_and_config(
        run_id=run_id,
        experiments_root=str(experiments_root),
        run_group=run_group,
    )
    model, audio_model, hyper_params = load_retrieval_models(
        run_id=run_id,
        experiments_root=str(experiments_root),
        run_group=run_group,
    )
    if hyper_params != recorded_hyper_params:
        raise RuntimeError('Loaded model configuration differs from config.json.')
    configured_dirprocess = Path(hyper_params['dirprocess'])
    dirprocess = configured_dirprocess
    if not dirprocess.exists():
        dirprocess = Path(PREPROCESSED_DATA_DIR)
    axis, n_components = load_reduction_axis(hyper_params)
    window_tag = hyper_params.get('window_tag', '')
    audio_embeddings_path = Path(
        get_audio_embeddings_path(
            dirprocess=dirprocess,
            split='test',
            embedding_layers=int(hyper_params['embedding_layers']),
            axis=axis,
            n_components=n_components,
            window_tag=window_tag,
        )
    )
    hidden_test = np.load(
        audio_embeddings_path,
        mmap_mode='r',
    )
    train_dataframe_path = Path(
        get_dataframe_path(
            dirprocess,
            'train',
            N_SUBJECTS,
            window_tag=window_tag,
        )
    )
    test_dataframe_path = Path(
        get_dataframe_path(
            dirprocess,
            'test',
            N_SUBJECTS,
            window_tag=window_tag,
        )
    )
    df_train = pd.read_csv(train_dataframe_path)
    df_test = pd.read_csv(test_dataframe_path)
    if hidden_test.ndim != 3:
        raise ValueError(
            'Expected test audio embeddings with shape (N, T, F), got '
            f'{hidden_test.shape}.'
        )
    expected_audio_features = (
        int(hyper_params['feature_reduction_dim'])
        if hyper_params['feature_reduction'] == 'pca'
        else N_FEATURES
    )
    if hidden_test.shape[-1] != expected_audio_features:
        raise ValueError(
            'Test audio embedding feature dimension disagrees with the run config: '
            f'loaded={hidden_test.shape[-1]}, expected={expected_audio_features}.'
        )
    if hidden_test.shape[1] <= 0:
        raise ValueError('Test audio embeddings have an empty time dimension.')
    if hyper_params['time_reduction'] == 'pca':
        expected_audio_time = int(hyper_params['time_reduction_dim'])
        if hidden_test.shape[1] != expected_audio_time:
            raise ValueError(
                'Test audio embedding time dimension disagrees with time PCA: '
                f'loaded={hidden_test.shape[1]}, expected={expected_audio_time}.'
            )
    test_windows = build_test_windows(df_test)
    candidate_ids = test_windows['wav_index'].to_numpy(dtype=np.int64)
    if hidden_test.shape[0] != candidate_ids.shape[0]:
        raise ValueError(
            'Test embedding count and fixed candidate bank disagree: '
            f'{hidden_test.shape[0]} vs {candidate_ids.shape[0]}.'
        )
    meg_sr = int(hyper_params['meg_sr'])
    if AUDIO_SR % meg_sr:
        raise ValueError(f'meg_sr={meg_sr} must divide AUDIO_SR={AUDIO_SR}.')
    meg_path = dirprocess / 'meg' / f'meg{N_SUBJECTS}_sr{hyper_params["meg_sr"]}.npz'
    if not meg_path.exists():
        raise FileNotFoundError(f'MEG archive not found: {meg_path}')

    model = model.to(device).eval()
    audio_model = audio_model.to(device).eval()
    for module in (model, audio_model):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return AnalysisInputs(
        model=model,
        audio_model=audio_model,
        hyper_params=hyper_params,
        run_dir=Path(run_dir),
        configured_dirprocess=configured_dirprocess,
        dirprocess=dirprocess,
        audio_embeddings_path=audio_embeddings_path,
        train_dataframe_path=train_dataframe_path,
        test_dataframe_path=test_dataframe_path,
        meg_path=meg_path,
        hidden_test=hidden_test,
        df_train=df_train,
        df_test=df_test,
        test_windows=test_windows,
        candidate_ids=candidate_ids,
        device=device,
    )


@torch.inference_mode()
def encode_candidate_bank(
    audio_model: torch.nn.Module,
    hidden_test: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Encode and normalize the full bank once; never eligibility-filter it."""

    raw = torch.as_tensor(
        np.array(hidden_test, dtype=np.float32, copy=True),
        dtype=torch.float32,
        device=device,
    )
    raw = einops.rearrange(raw, 'b t f -> b f t')
    embedded = audio_model(raw)
    if embedded.ndim == 3:
        bank_normalized = torch_f.normalize(embedded, dim=(-2, -1))
    elif embedded.ndim == 2:
        bank_normalized = torch_f.normalize(embedded, dim=-1)
    else:
        raise NotImplementedError(
            'Expected the audio model to return 2D or 3D embeddings, got '
            f'{tuple(embedded.shape)}.'
        )
    if bank_normalized.shape[0] != hidden_test.shape[0]:
        raise RuntimeError('Encoded candidate bank has the wrong size.')
    return bank_normalized


def derive_sound_origins(
    combo_train: pd.DataFrame,
    *,
    story_id: int,
    donor_sounds: set[str],
    meg_sr: int,
) -> dict[str, int]:
    """Recover exact raw-MEG file origins from integer dataframe coordinates."""

    meg_start_column = f'meg{meg_sr}_start'
    required = {'story_id', 'sound_fname', 'wav_start', meg_start_column}
    missing = required - set(combo_train.columns)
    if missing:
        raise ValueError(f'Train dataframe is missing columns {sorted(missing)}.')
    hop_samples = AUDIO_SR // meg_sr
    rows = combo_train[
        (combo_train['story_id'].astype(int) == story_id)
        & combo_train['sound_fname'].astype(str).isin(donor_sounds)
    ].copy()
    if rows.empty:
        raise ValueError('No donor rows exist for this subject-session.')
    if np.any(rows['wav_start'].to_numpy(dtype=np.int64) % hop_samples):
        raise ValueError('Donor wav_start is not aligned to the MEG frame grid.')
    rows['_origin'] = (
        rows[meg_start_column].to_numpy(dtype=np.int64)
        - rows['wav_start'].to_numpy(dtype=np.int64) // hop_samples
    )
    origins: dict[str, int] = {}
    for sound, sound_rows in rows.groupby('sound_fname'):
        values = np.unique(sound_rows['_origin'].to_numpy(dtype=np.int64))
        if values.size != 1:
            raise ValueError(
                f'{sound!r} has inconsistent raw-MEG origins: {values.tolist()}.'
            )
        origins[str(sound)] = int(values[0])
    if set(origins) != donor_sounds:
        raise ValueError(
            f'Donor origin mismatch; missing={sorted(donor_sounds - set(origins))}, '
            f'extra={sorted(set(origins) - donor_sounds)}.'
        )
    return origins


class ComboDonorReader:
    """Read sound-local donor chunks from one loaded subject-session story."""

    def __init__(
        self,
        raw_story: np.ndarray,
        sound_origins: dict[str, int],
        meg_offset_frames: int,
    ) -> None:
        self.raw_story = raw_story
        self.sound_origins = sound_origins
        self.meg_offset_frames = int(meg_offset_frames)

    def __call__(self, interval: DonorInterval) -> torch.Tensor:
        if interval.sound_fname not in self.sound_origins:
            raise KeyError(f'No MEG origin for donor sound {interval.sound_fname!r}.')
        origin = self.sound_origins[interval.sound_fname]
        start = origin + self.meg_offset_frames + interval.start
        stop = origin + self.meg_offset_frames + interval.stop
        if start < 0 or stop > self.raw_story.shape[-1] or stop <= start:
            raise ValueError(
                f'Donor raw interval [{start}, {stop}) is invalid for '
                f'{interval.sound_fname!r}, story length={self.raw_story.shape[-1]}.'
            )
        # Copy protects the result from non-writeable compressed-NPZ buffers.
        return torch.from_numpy(
            np.array(self.raw_story[:, start:stop], dtype=np.float32, copy=True)
        )


def load_combo_targets(
    raw_story: np.ndarray,
    combo_test: pd.DataFrame,
    *,
    meg_sr: int,
    meg_offset_frames: int,
    expected_candidate_ids: np.ndarray,
) -> torch.Tensor:
    """Load all original query rows in fixed-candidate order."""

    meg_start_column = f'meg{meg_sr}_start'
    meg_stop_column = f'meg{meg_sr}_stop'
    rows = combo_test.sort_values('wav_index')
    observed = rows['wav_index'].to_numpy(dtype=np.int64)
    if not np.array_equal(observed, expected_candidate_ids):
        raise ValueError('Subject-session test rows do not match the fixed bank.')
    windows: list[np.ndarray] = []
    expected_length: int | None = None
    for row in rows.itertuples(index=False):
        start = int(getattr(row, meg_start_column)) + meg_offset_frames
        stop = int(getattr(row, meg_stop_column)) + meg_offset_frames
        if start < 0 or stop > raw_story.shape[-1] or stop <= start:
            raise ValueError(f'Invalid target MEG interval [{start}, {stop}).')
        window = np.asarray(raw_story[:, start:stop], dtype=np.float32)
        if expected_length is None:
            expected_length = int(window.shape[-1])
        if window.shape[-1] != expected_length:
            raise ValueError('Test MEG queries do not share one duration.')
        windows.append(window)
    return torch.from_numpy(np.stack(windows, axis=0))


@torch.inference_mode()
def rank_query_batches(
    model: torch.nn.Module,
    queries: torch.Tensor,
    subject_index: int,
    labels: np.ndarray,
    candidate_bank: torch.Tensor,
    candidate_ids_device: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Forward and rank a CPU query tensor against the complete bank."""

    if queries.shape[0] != len(labels):
        raise ValueError('Query and label counts differ.')
    ranks: list[np.ndarray] = []
    for start in range(0, queries.shape[0], batch_size):
        stop = min(queries.shape[0], start + batch_size)
        try:
            query = queries[start:stop].to(device=device, dtype=torch.float32)
            subjects = torch.full(
                (stop - start,),
                int(subject_index),
                dtype=torch.long,
                device=device,
            )
            interpretable, embedded = model((query, subjects))
            del interpretable
            label_tensor = torch.as_tensor(
                labels[start:stop], dtype=torch.long, device=device
            )
            rank = fixed_bank_ranks(
                embedded,
                candidate_bank,
                label_tensor,
                candidate_ids_device,
            )
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError(
                f'CUDA ran out of memory for MEG inference batch {stop - start}. '
                'Re-run with a lower --inference-batch-size; no partial '
                'feature/session cell will be reused.'
            ) from error
        ranks.append(rank.detach().cpu().numpy().astype(np.int16, copy=False))
        del query, subjects, embedded, label_tensor, rank
    return np.concatenate(ranks)


@torch.inference_mode()
def evaluate_occlusion_queries(
    *,
    model: torch.nn.Module,
    targets: torch.Tensor,
    plans: dict[int, WindowPlan],
    donor_reader: ComboDonorReader,
    n_donor_pairs: int,
    subject_index: int,
    candidate_bank: torch.Tensor,
    candidate_ids: np.ndarray,
    device: torch.device,
    batch_size: int,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate only eligible queries, preserving all bank columns."""

    n_windows = targets.shape[0]
    rank_absent = np.full((n_windows, n_donor_pairs), -1, dtype=np.int16)
    rank_present = np.full((n_windows, n_donor_pairs), -1, dtype=np.int16)
    candidate_ids_device = torch.as_tensor(
        candidate_ids, dtype=torch.long, device=device
    )

    query_buffer: list[torch.Tensor] = []
    label_buffer: list[int] = []
    index_buffer: list[tuple[int, int, int]] = []

    def flush() -> None:
        if not query_buffer:
            return
        completed_queries = len(query_buffer)
        batch = torch.stack(query_buffer, dim=0)
        labels = np.asarray(label_buffer, dtype=np.int64)
        ranks = rank_query_batches(
            model,
            batch,
            subject_index,
            labels,
            candidate_bank,
            candidate_ids_device,
            device,
            batch_size=len(query_buffer),
        )
        for rank, (wav_index, replicate, arm) in zip(ranks, index_buffer):
            if arm == 0:
                rank_absent[wav_index, replicate] = rank
            else:
                rank_present[wav_index, replicate] = rank
        query_buffer.clear()
        label_buffer.clear()
        index_buffer.clear()
        if progress_callback is not None:
            progress_callback(completed_queries)

    for wav_index in sorted(plans):
        target = targets[wav_index]
        for replicate in range(n_donor_pairs):
            absent_query, present_query = apply_component_plan(
                target,
                plans[wav_index],
                replicate,
                donor_reader,
            )
            for arm, query in enumerate((absent_query, present_query)):
                if len(query_buffer) == batch_size:
                    flush()
                query_buffer.append(query)
                label_buffer.append(wav_index)
                index_buffer.append((wav_index, replicate, arm))
    flush()

    eligible = np.zeros(n_windows, dtype=bool)
    eligible[list(plans)] = True
    if np.any(rank_absent[eligible] < 1) or np.any(rank_present[eligible] < 1):
        raise RuntimeError('An eligible occlusion rank was not filled.')
    if np.any(rank_absent[~eligible] != -1) or np.any(rank_present[~eligible] != -1):
        raise RuntimeError('A discarded query unexpectedly received an occlusion rank.')
    return rank_absent, rank_present


def studentized_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError('studentized_mean requires at least two finite values.')
    standard_error = float(np.std(values, ddof=1) / math.sqrt(values.size))
    mean = float(np.mean(values))
    if standard_error == 0.0:
        if mean > 0:
            return float('inf')
        if mean < 0:
            return float('-inf')
        return 0.0
    return mean / standard_error


def two_sided_signflip_max_abs_t(
    participant_effects: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Shared participant sign flips and two-sided single-step max-|T| FWER."""

    effects = np.asarray(participant_effects, dtype=np.float64)
    if effects.ndim != 2 or effects.shape[0] < 2:
        raise ValueError(
            'participant_effects must have shape (participants, features).'
        )
    if not np.isfinite(effects).all():
        raise ValueError('participant_effects contains non-finite values.')
    if n_permutations <= 0:
        raise ValueError('n_permutations must be positive.')
    n_subjects = effects.shape[0]
    observed = np.array(
        [studentized_mean(effects[:, feature]) for feature in range(effects.shape[1])]
    )
    rng = np.random.default_rng(seed)
    signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=(n_permutations, n_subjects),
        replace=True,
    )
    means = signs @ effects / n_subjects
    sum_squares = np.sum(effects**2, axis=0)[None, :]
    variances = (sum_squares - n_subjects * means**2) / (n_subjects - 1)
    variances = np.maximum(variances, 0.0)
    standard_errors = np.sqrt(variances / n_subjects)
    null_t = np.divide(
        means,
        standard_errors,
        out=np.zeros_like(means),
        where=standard_errors > 0,
    )
    null_t[(standard_errors == 0) & (means > 0)] = np.inf
    null_t[(standard_errors == 0) & (means < 0)] = -np.inf
    observed_abs = np.abs(observed)
    null_abs = np.abs(null_t)
    raw_p = (1 + np.sum(null_abs >= observed_abs[None, :], axis=0)) / (
        n_permutations + 1
    )
    max_null = np.max(null_abs, axis=1)
    corrected_p = (1 + np.sum(max_null[:, None] >= observed_abs[None, :], axis=0)) / (
        n_permutations + 1
    )
    return {
        'observed_t': observed,
        'observed_abs_t': observed_abs,
        'raw_p': raw_p,
        'max_t_fwer_p': corrected_p,
        'null_mean': means,
        'null_t': null_t,
        'null_abs_t': null_abs,
        'max_abs_t_null': max_null,
        'signs': signs,
    }


def participant_t_interval(
    values: np.ndarray, confidence: float = 0.95
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError('A participant interval requires at least two finite values.')
    mean = float(np.mean(values))
    standard_error = float(scipy_stats.sem(values))
    if standard_error == 0.0:
        return mean, mean
    critical = float(scipy_stats.t.ppf((1.0 + confidence) / 2.0, df=values.size - 1))
    return mean - critical * standard_error, mean + critical * standard_error


def classify_feature_result(
    *,
    mean_effect: float,
    corrected_p: float,
    feature_present_control_mean_rank: float,
    candidate_bank_size: int,
    alpha: float = 0.05,
) -> tuple[str, float, bool]:
    """Apply the control-validity gate before interpreting the primary test."""

    values = np.asarray(
        [
            mean_effect,
            corrected_p,
            feature_present_control_mean_rank,
            alpha,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError('Feature-result classification inputs must be finite.')
    if candidate_bank_size <= 0:
        raise ValueError('candidate_bank_size must be positive.')
    if not 0.0 <= corrected_p <= 1.0 or not 0.0 < alpha < 1.0:
        raise ValueError('Probabilities must lie in their valid ranges.')

    random_rank_expectation = float((candidate_bank_size + 1) / 2.0)
    control_saturated = feature_present_control_mean_rank >= random_rank_expectation
    if control_saturated:
        conclusion = 'control_saturated'
    elif corrected_p < alpha and mean_effect > 0:
        conclusion = 'significant_positive_effect'
    elif corrected_p < alpha and mean_effect < 0:
        conclusion = 'significant_negative_effect'
    else:
        conclusion = 'inconclusive'
    return conclusion, random_rank_expectation, control_saturated


def aggregate_sessions_to_participants(
    combo_table: pd.DataFrame,
    value_columns: Sequence[str],
) -> pd.DataFrame:
    """Equal session weight within each participant."""

    required = {'subject_id', 'session_id', *value_columns}
    missing = required - set(combo_table.columns)
    if missing:
        raise ValueError(f'Combo table is missing columns {sorted(missing)}.')
    grouped = (
        combo_table.groupby('subject_id', sort=True)[list(value_columns)]
        .mean()
        .reset_index()
    )
    expected_subjects = sorted({subject for subject, _ in SUB_SES_COMBOS})
    if grouped['subject_id'].astype(int).tolist() != expected_subjects:
        raise ValueError(
            'Participant aggregation does not cover the expected subjects.'
        )
    return grouped


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish a JSON artifact within its destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding='utf-8',
    )
    temporary.replace(path)


def atomic_write_csv(path: Path, table: pd.DataFrame) -> None:
    """Atomically publish a CSV artifact within its destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    """Stream a file digest without materializing large artifacts."""

    digest = hashlib.sha256()
    with path.open('rb') as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(
    path: Path,
    *,
    content_hash: bool = False,
) -> dict[str, Any]:
    """Artifact provenance fingerprint, optionally including a content digest."""

    resolved = path.resolve()
    stat = resolved.stat()
    identity = {
        'path': str(resolved),
        'size_bytes': int(stat.st_size),
        'mtime_ns': int(stat.st_mtime_ns),
    }
    if content_hash:
        identity['sha256'] = sha256_file(resolved)
    return identity


def installed_package_versions(names: Sequence[str]) -> dict[str, str]:
    """Record libraries that can affect masks, tensors, or numerical ranks."""

    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not-installed'
    return versions


def analysis_configuration_id(configuration: dict[str, Any]) -> str:
    """Stable short ID for one complete scientific/runtime configuration."""

    serialized = json.dumps(
        _json_safe(configuration),
        sort_keys=True,
        separators=(',', ':'),
    ).encode()
    return hashlib.sha256(serialized).hexdigest()[:16]


def prepare_output_directory(
    output_dir: Path,
    *,
    configuration_id: str,
    configuration: dict[str, Any],
) -> None:
    """Create or safely resume an output directory with the exact same config."""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'analysis_config.json'
    expected = {
        'analysis': ANALYSIS_NAME,
        'configuration_id': configuration_id,
        'configuration': configuration,
    }
    if manifest_path.exists():
        observed = json.loads(manifest_path.read_text(encoding='utf-8'))
        if observed != _json_safe(expected):
            raise ValueError(
                'Output directory belongs to a different analysis configuration: '
                f'{output_dir}. Choose another --output-dir.'
            )
        return
    existing = [path.name for path in output_dir.iterdir()]
    if existing:
        raise ValueError(
            'Refusing to mix occlusion outputs into a nonempty directory without '
            f'a matching analysis_config.json: {output_dir}; entries={existing[:10]}.'
        )
    atomic_write_json(manifest_path, expected)


def load_completed_result(
    output_dir: Path,
    *,
    configuration_id: str,
    save_ranks: bool,
) -> dict[str, Any] | None:
    """Return an intact completed run, or invalidate a stale completion marker."""

    marker_path = output_dir / 'COMPLETED.json'
    if not marker_path.exists():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding='utf-8'))
        stats_path = output_dir / 'stats.json'
        stats = json.loads(stats_path.read_text(encoding='utf-8'))
        if (
            marker.get('configuration_id') != configuration_id
            or stats.get('configuration_id') != configuration_id
            or marker.get('status') != stats.get('status')
        ):
            raise ValueError('Completion marker/configuration mismatch.')
        required = [
            output_dir / 'window_eligibility.csv',
            output_dir / 'feature_summary.csv',
            output_dir / 'combo_metrics.csv',
            output_dir / 'participant_metrics.csv',
        ]
        if stats.get('status') == 'complete':
            required.extend(
                [
                    output_dir / 'feature_effects.png',
                    output_dir / 'feature_effects.pdf',
                    output_dir / 'feature_mask_jaccard.csv',
                    output_dir / 'feature_mask_jaccard.png',
                    output_dir / 'feature_mask_jaccard.pdf',
                    output_dir / 'feature_mask_containment.csv',
                ]
            )
            if save_ranks:
                required.append(output_dir / 'ranks.npz')
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f'Completed run is missing artifacts: {missing}.')
        if save_ranks and stats.get('status') == 'complete':
            with np.load(output_dir / 'ranks.npz') as ranks:
                if 'baseline_rank_full_bank' not in ranks.files:
                    raise ValueError('ranks.npz is missing baseline_rank_full_bank.')

        def read_table(path: Path) -> pd.DataFrame:
            try:
                return pd.read_csv(path)
            except pd.errors.EmptyDataError:
                return pd.DataFrame()

        return {
            'output_dir': output_dir,
            'stats': stats,
            'feature_summary': read_table(output_dir / 'feature_summary.csv'),
            'combo_metrics': read_table(output_dir / 'combo_metrics.csv'),
            'participant_metrics': read_table(output_dir / 'participant_metrics.csv'),
            'eligibility': read_table(output_dir / 'window_eligibility.csv'),
        }
    except (
        OSError,
        ValueError,
        KeyError,
        EOFError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ):
        invalid_path = output_dir / (
            f'INVALID_COMPLETED.{marker_path.stat().st_mtime_ns}.{os.getpid()}.json'
        )
        marker_path.replace(invalid_path)
        return None


def open_rank_memmap(
    path: Path,
    shape: tuple[int, ...],
) -> np.memmap:
    """Open a resumable int16 rank array, initializing missing cells to -1."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            array = np.load(path, mmap_mode='r+')
        except (OSError, ValueError, EOFError):
            corrupt = path.with_name(
                f'CORRUPT_{path.name}.{path.stat().st_mtime_ns}.{os.getpid()}'
            )
            path.replace(corrupt)
            return open_rank_memmap(path, shape)
        if not isinstance(array, np.memmap):
            raise TypeError(f'Checkpoint is not memory-mapped: {path}')
        if array.dtype != np.dtype(np.int16) or array.shape != shape:
            raise ValueError(
                f'Checkpoint shape/dtype mismatch for {path}: '
                f'{array.shape}/{array.dtype} vs {shape}/int16.'
            )
        if np.any(array == 0):
            del array
            corrupt = path.with_name(
                f'CORRUPT_{path.name}.{path.stat().st_mtime_ns}.{os.getpid()}'
            )
            path.replace(corrupt)
            return open_rank_memmap(path, shape)
        return array
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.initializing')
    array = np.lib.format.open_memmap(
        temporary,
        mode='w+',
        dtype=np.int16,
        shape=shape,
    )
    array.fill(-1)
    array.flush()
    del array
    temporary.replace(path)
    loaded = np.load(path, mmap_mode='r+')
    if not isinstance(loaded, np.memmap):  # pragma: no cover - NumPy contract
        raise TypeError(f'Initialized checkpoint is not memory-mapped: {path}')
    return loaded


def baseline_checkpoint_complete(
    ranks: np.ndarray,
    combo_index: int,
    *,
    candidate_bank_size: int,
) -> bool:
    """Validate one baseline row and identify complete versus partial state."""

    row = ranks[combo_index]
    invalid = (row < -1) | (row > candidate_bank_size) | (row == 0)
    if np.any(invalid):
        raise ValueError(f'Baseline checkpoint row {combo_index} is corrupt.')
    complete = row >= 1
    if np.all(complete):
        return True
    if np.any(complete):
        row[:] = -1
        if isinstance(ranks, np.memmap):
            ranks.flush()
    return False


def occlusion_checkpoint_complete(
    absent: np.ndarray,
    present: np.ndarray,
    combo_index: int,
    eligible: np.ndarray,
    *,
    candidate_bank_size: int,
) -> bool:
    """Validate one feature/combo checkpoint, resetting only partial rows."""

    absent_row = absent[combo_index]
    present_row = present[combo_index]
    for name, row in (('absent', absent_row), ('present', present_row)):
        invalid = (row < -1) | (row > candidate_bank_size) | (row == 0)
        if np.any(invalid):
            raise ValueError(
                f'Occlusion {name} checkpoint row {combo_index} is corrupt.'
            )
        if np.any(row[~eligible] != -1):
            raise ValueError(
                f'Occlusion {name} checkpoint filled an ineligible query in '
                f'row {combo_index}.'
            )
    absent_filled = absent_row[eligible] >= 1
    present_filled = present_row[eligible] >= 1
    if np.all(absent_filled) and np.all(present_filled):
        return True
    if np.any(absent_filled) or np.any(present_filled):
        absent_row[:] = -1
        present_row[:] = -1
        if isinstance(absent, np.memmap):
            absent.flush()
        if isinstance(present, np.memmap):
            present.flush()
    return False


def occlusion_progress_totals(
    cell_complete: dict[tuple[int, str], bool],
    eligible_window_counts: dict[str, int],
    n_donor_pairs: int,
) -> tuple[int, int]:
    """Return total and already-checkpointed perturbed-query evaluations."""

    if n_donor_pairs <= 0:
        raise ValueError('n_donor_pairs must be positive.')
    total = 0
    completed = 0
    for (_combo_index, feature), is_complete in cell_complete.items():
        if feature not in eligible_window_counts:
            raise KeyError(f'No eligible-window count for feature {feature!r}.')
        eligible_count = int(eligible_window_counts[feature])
        if eligible_count < 0:
            raise ValueError('Eligible-window counts cannot be negative.')
        work = 2 * n_donor_pairs * eligible_count
        total += work
        if is_complete:
            completed += work
    return total, completed


def pending_occlusion_forward_batches(
    cell_complete: dict[tuple[int, str], bool],
    eligible_window_counts: dict[str, int],
    n_donor_pairs: int,
    batch_size: int,
) -> int:
    """Count the exact pending MEG forward batches at feature-cell boundaries."""

    if n_donor_pairs <= 0:
        raise ValueError('n_donor_pairs must be positive.')
    if batch_size <= 0:
        raise ValueError('batch_size must be positive.')
    pending = 0
    for (_combo_index, feature), is_complete in cell_complete.items():
        if feature not in eligible_window_counts:
            raise KeyError(f'No eligible-window count for feature {feature!r}.')
        eligible_count = int(eligible_window_counts[feature])
        if eligible_count < 0:
            raise ValueError('Eligible-window counts cannot be negative.')
        if not is_complete and eligible_count:
            pending += math.ceil(2 * n_donor_pairs * eligible_count / batch_size)
    return pending


def plot_feature_summary(
    summary: pd.DataFrame,
    participant_metrics: pd.DataFrame,
    output_path: Path,
    *,
    null_mean_effects: np.ndarray | None = None,
    preview_label: str | None = None,
) -> None:
    """Wide paper figure: participant effects and their group means.

    Features run left-to-right, grey points show participants, diamonds show
    group means, and a star is the sole inferential annotation. Coverage,
    confidence intervals, p-values, and exclusion diagnostics remain available
    in the machine-readable outputs instead of crowding the headline figure.
    """

    if summary.empty:
        return
    required_summary = {'feature', 'mean_rank_effect', 'p_max_t_fwer'}
    missing_summary = required_summary.difference(summary.columns)
    if missing_summary:
        raise ValueError(
            f'Summary is missing plotting columns: {sorted(missing_summary)}'
        )
    required_participant = {
        'feature',
        'subject_id',
        'effect_rank_absent_minus_present',
    }
    missing_participant = required_participant.difference(participant_metrics.columns)
    if missing_participant:
        raise ValueError(
            'Participant metrics are missing plotting columns: '
            f'{sorted(missing_participant)}'
        )

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.transforms import blended_transform_factory

    means = summary['mean_rank_effect'].to_numpy(dtype=np.float64)
    feature_names = summary['feature'].astype(str).tolist()
    if null_mean_effects is not None:
        null_mean_effects = np.asarray(null_mean_effects, dtype=np.float64)
        expected_shape = (len(summary),)
        if (
            null_mean_effects.ndim != 2
            or null_mean_effects.shape[1:] != expected_shape
            or null_mean_effects.shape[0] < 2
            or not np.isfinite(null_mean_effects).all()
        ):
            raise ValueError(
                'Null means must be a finite (permutations, features) array; '
                f'got {null_mean_effects.shape}.'
            )
    order = np.argsort(-means)
    positions = np.arange(len(summary))
    labels = [PLOT_DISPLAY_NAMES.get(feature_names[index], feature_names[index])
              for index in order]
    figure_width = 0.62 * len(summary) + 3.0
    font_context = {
        'font.size': MAIN_EFFECTS_FONT_SIZE,
        'axes.titlesize': MAIN_EFFECTS_FONT_SIZE,
        'axes.labelsize': MAIN_EFFECTS_FONT_SIZE,
        'xtick.labelsize': MAIN_EFFECTS_FONT_SIZE,
        'ytick.labelsize': MAIN_EFFECTS_FONT_SIZE,
        'legend.fontsize': MAIN_EFFECTS_FONT_SIZE,
    }

    with plt.rc_context(font_context):
        fig, ax = plt.subplots(
            figsize=(figure_width, MAIN_EFFECTS_WIDTH_IN)
        )

        # Display the feature-wise sign-flip null in the same rank-difference
        # units as the observed participant effects. Inference still uses all
        # permutations and the max-|T| null; only the KDE rendering is thinned to
        # keep figure generation fast for the default 100,000 permutations.
        if null_mean_effects is not None:
            visual_stride = max(1, math.ceil(null_mean_effects.shape[0] / 5_000))
            visual_null = null_mean_effects[::visual_stride]
            vp = ax.violinplot(
                [visual_null[:, index] for index in order],
                positions=positions,
                orientation='vertical',
                showextrema=False,
                showmeans=True,
                quantiles=[[0.025, 0.975] for _ in positions],
                widths=0.82,
            )
            for body in vp['bodies']:
                body.set_facecolor('#b9dfba')
                body.set_edgecolor('#2ca02c')
                body.set_alpha(0.45)
                body.set_zorder(1)
            for part in ('cmeans', 'cquantiles'):
                if part in vp:
                    vp[part].set_edgecolor('#2ca02c')
                    vp[part].set_linewidth(1.2)
                    vp[part].set_zorder(1)
        ax.axhline(0.0, color='#2ca02c', linewidth=1.6, zorder=1)

        jitter = np.random.default_rng(0)
        star_transform = blended_transform_factory(ax.transData, ax.transAxes)
        for rank, summary_index in enumerate(order):
            feature = feature_names[summary_index]
            values = participant_metrics.loc[
                participant_metrics['feature'].astype(str) == feature,
                'effect_rank_absent_minus_present',
            ].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                raise ValueError(
                    f'No finite participant effects available for {feature!r}.'
                )
            offsets = (jitter.random(values.size) - 0.5) * 0.30
            ax.scatter(
                rank + offsets,
                values,
                s=14,
                color='0.45',
                alpha=0.55,
                linewidths=0,
                zorder=2.5,
            )
            mean = means[summary_index]
            mean_color = 'crimson' if mean > 0 else 'steelblue'
            ax.plot(
                rank,
                mean,
                marker='D',
                color=mean_color,
                markersize=10,
                markeredgecolor='black',
                markeredgewidth=0.8,
                linestyle='',
                zorder=4,
            )
            if float(summary.iloc[summary_index]['p_max_t_fwer']) < 0.05:
                ax.plot(
                    rank,
                    1.02,
                    marker='*',
                    markersize=13,
                    color='#2ca02c',
                    linestyle='',
                    transform=star_transform,
                    clip_on=False,
                )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_xlim(-0.6, len(summary) - 0.4)
        ax.set_ylabel('Rank difference (removal − control)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        handles = [
            Line2D(
                [0],
                [0],
                marker='D',
                color='crimson',
                linestyle='',
                markeredgecolor='black',
                label='Group effect (+)',
            ),
            Line2D(
                [0],
                [0],
                marker='D',
                color='steelblue',
                linestyle='',
                markeredgecolor='black',
                label='Group effect (−)',
            ),
            Line2D(
                [0],
                [0],
                marker='o',
                color='0.45',
                linestyle='',
                label='Subject',
            ),
            Patch(
                facecolor='#b9dfba',
                edgecolor='#2ca02c',
                alpha=0.45,
                label='Sign-flip null',
            ),
            Line2D(
                [0],
                [0],
                marker='*',
                color='#2ca02c',
                linestyle='',
                markersize=12,
                label='pFWER < 0.05',
            ),
        ]
        ax.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.28),
            ncol=5,
            framealpha=0.9,
        )
        if preview_label is not None:
            fig.text(
                0.5,
                0.52,
                preview_label,
                ha='center',
                va='center',
                fontsize=28,
                color='black',
                alpha=0.09,
                rotation=28,
                weight='bold',
            )
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches='tight')
        plt.close(fig)


def write_plot_style_preview(output_path: Path) -> None:
    """Render the real figure code immediately using conspicuously fake values."""

    specs = DEFAULT_FEATURE_BATTERY
    position = np.arange(len(specs), dtype=np.float64)
    requested_means = 4.5 * np.sin(0.72 * position) + 0.25 * position - 1.5
    rng = np.random.default_rng(7)
    participant_rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        values = requested_means[index] + rng.normal(0.0, 3.2, size=27)
        for subject_id, value in enumerate(values, start=1):
            participant_rows.append(
                {
                    'feature': spec.name,
                    'subject_id': subject_id,
                    'effect_rank_absent_minus_present': value,
                }
            )
    participants = pd.DataFrame(participant_rows)
    means = np.asarray(
        [
            participants.loc[
                participants['feature'] == spec.name,
                'effect_rank_absent_minus_present',
            ].mean()
            for spec in specs
        ]
    )
    effect_matrix = np.column_stack(
        [
            participants.loc[
                participants['feature'] == spec.name,
                'effect_rank_absent_minus_present',
            ].to_numpy(dtype=np.float64)
            for spec in specs
        ]
    )
    preview_inference = two_sided_signflip_max_abs_t(
        effect_matrix,
        n_permutations=100_000,
        seed=1701,
    )
    preview = pd.DataFrame(
        {
            'feature': [spec.name for spec in specs],
            'label': [spec.label or spec.name for spec in specs],
            'mean_rank_effect': means,
            'p_max_t_fwer': preview_inference['max_t_fwer_p'],
        }
    )
    plot_feature_summary(
        preview,
        participants,
        output_path,
        null_mean_effects=preview_inference['null_mean'],
        preview_label='STYLE PREVIEW — FAKE VALUES',
    )


def run_analysis(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Execute the complete standalone experiment and persist auditable outputs."""

    validate_run_group(args.run_group)
    _preview_run_dir, preview_hyper_params = _load_run_dir_and_config(
        run_id=args.run_id,
        experiments_root=str(args.experiments_root),
        run_group=args.run_group,
    )
    if args.seed is None:
        if preview_hyper_params.get('seed') is None:
            raise ValueError(
                'The selected run has seed=null. Supply --seed so donor selection '
                'and permutation inference are reproducible.'
            )
        analysis_seed = int(preview_hyper_params['seed'])
    else:
        analysis_seed = int(args.seed)
    configured_torch_deterministic = bool(
        preview_hyper_params.get('torch_deterministic', False)
    )
    configured_device = str(preview_hyper_params.get('device', '')).strip()
    if not configured_device:
        raise ValueError('The selected run config has no usable device setting.')
    requested_device = configured_device if args.device is None else args.device
    device_source = (
        'run_config.device' if args.device is None else 'explicit_cli_override'
    )
    if configured_torch_deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    fix_seed(
        analysis_seed,
        torch_deterministic=configured_torch_deterministic,
    )
    if not configured_torch_deterministic:
        torch.use_deterministic_algorithms(False)
    device = resolve_device(requested_device)
    inputs = load_analysis_inputs(
        run_id=args.run_id,
        run_group=args.run_group,
        experiments_root=args.experiments_root,
        device=device,
    )
    hyper_params = inputs.hyper_params
    configured_test_batch_size = int(hyper_params['test_batch_size'])
    (
        canonical_baseline_batch_size,
        inference_batch_size,
        inference_batch_source,
    ) = select_evaluation_batch_sizes(
        device_type=device.type,
        training_batch_size=int(hyper_params['batch_size']),
        test_batch_size=configured_test_batch_size,
        inference_override=args.inference_batch_size,
    )
    using_default_runtime_settings = (
        args.device is None and args.inference_batch_size is None
    )
    runtime_configuration_class = (
        'optimized_default_runtime'
        if using_default_runtime_settings
        else 'explicit_runtime_override'
    )
    tqdm.write(
        'Inference configuration: '
        f'canonical baseline batch_size={canonical_baseline_batch_size}; '
        f'device={device}, MEG batch_size={inference_batch_size} '
        f'({inference_batch_source}); candidate-bank normalization is cached.'
    )
    if int(args.n_permutations) <= 0:
        raise ValueError('--n-permutations must be positive.')
    meg_sr = int(hyper_params['meg_sr'])
    hop_samples = AUDIO_SR // meg_sr
    meg_offset_frames = int(float(hyper_params['meg_offset']) * meg_sr)
    padding_frames = int(round(args.mask_padding_ms * meg_sr / 1000.0))
    taper_frames = int(round(args.taper_ms * meg_sr / 1000.0))
    if padding_frames < 0 or taper_frames < 0:
        raise ValueError('Mask padding and taper must be non-negative.')
    if taper_frames > padding_frames:
        raise ValueError(
            'The taper is applied inside the padded mask and cannot exceed '
            'the padding width.'
        )
    if args.n_donor_pairs <= 0:
        raise ValueError('--n-donor-pairs must be positive.')
    donor_sound_ids = [int(value) for value in args.donor_sound_ids]
    if (
        int(args.donor_story_id) < 0
        or any(value < 0 for value in donor_sound_ids)
        or len(set(donor_sound_ids)) != len(donor_sound_ids)
    ):
        raise ValueError(
            'Donor story/sound IDs must be non-negative and sound IDs unique.'
        )

    feature_specs = (
        feature_specs_from_json(args.features_json)
        if args.features_json is not None
        else DEFAULT_FEATURE_BATTERY
    )
    needs_word_features = any(
        spec.builder in {'word_onset', 'insertion', 'word_value'}
        for spec in feature_specs
    )
    word_features: pd.DataFrame | None = None
    word_features_path = (
        args.word_features
        if args.word_features is not None
        else inputs.dirprocess / 'linguistic' / 'word_features.csv'
    )
    if needs_word_features:
        if not word_features_path.exists():
            raise FileNotFoundError(
                f'Word-feature table not found: {word_features_path}. '
                'Create it with lisa-precompute-word-features, or provide a '
                'feature JSON containing only non-word features.'
            )
        word_features = load_word_features(word_features_path)

    donor_sounds = select_donor_sounds(
        inputs.df_train,
        story_id=args.donor_story_id,
        donor_sound_ids=args.donor_sound_ids,
    )
    target_story_ids = set(
        inputs.test_windows['story_id'].astype(int).unique().tolist()
    )
    if target_story_ids != {args.donor_story_id}:
        raise ValueError(
            'This experiment requires donor and target files from the same story. '
            f'Donor story={args.donor_story_id}, test stories={sorted(target_story_ids)}.'
        )
    target_sounds = set(inputs.test_windows['sound_fname'].astype(str).unique())
    if donor_sounds & target_sounds:
        raise ValueError(
            'Donor and target audio files must be disjoint. '
            f'Overlap={sorted(donor_sounds & target_sounds)}.'
        )
    all_sounds = sorted(donor_sounds | target_sounds)
    if word_features is not None:
        validate_word_feature_sound_coverage(word_features, all_sounds)
    sound_lengths = sound_frame_lengths(
        all_sounds,
        args.audio_dir,
        hop_samples,
    )
    annotation_artifacts: dict[str, Any] = {
        'raw_audio': [
            file_identity(args.audio_dir / sound, content_hash=True)
            for sound in all_sounds
        ],
    }
    if any(spec.builder in {'textgrid_silence', 'phoneme'} for spec in feature_specs):
        annotation_artifacts['mfa_textgrids'] = [
            file_identity(
                args.mfa_dir / Path(sound).with_suffix('.TextGrid').name,
                content_hash=True,
            )
            for sound in all_sounds
        ]
    if needs_word_features:
        annotation_artifacts['word_features'] = file_identity(
            word_features_path,
            content_hash=True,
        )

    implementation_path = Path(__file__).resolve()
    lisa_source_root = implementation_path.parents[1]
    relevant_source_paths = [
        implementation_path,
        lisa_source_root / 'model' / 'load_model.py',
        lisa_source_root / 'model' / 'nn_modules.py',
        lisa_source_root / 'plots' / 'feature_traces.py',
        lisa_source_root / 'training' / 'criteria.py',
        lisa_source_root / 'data' / 'audio_reduction.py',
        lisa_source_root / 'utils' / 'audio_io.py',
        lisa_source_root / 'utils' / 'constants.py',
        lisa_source_root / 'utils' / 'determinism.py',
        lisa_source_root / 'utils' / 'textgrid_io.py',
    ]
    source_manifest = {
        str(path.relative_to(lisa_source_root)): file_identity(
            path,
            content_hash=True,
        )
        for path in relevant_source_paths
    }
    package_versions = installed_package_versions(
        (
            'torch',
            'numpy',
            'pandas',
            'scipy',
            'librosa',
            'soundfile',
            'einops',
            'matplotlib',
            'textgrid',
        )
    )
    device_description = (
        torch.cuda.get_device_name(device)
        if device.type == 'cuda'
        else f'{platform.system()}-{platform.machine()}-{platform.processor()}'
    )
    analysis_configuration: dict[str, Any] = {
        'run_id': args.run_id,
        'run_group': args.run_group,
        'run_config': file_identity(
            inputs.run_dir / 'config.json',
            content_hash=True,
        ),
        'main_checkpoint': file_identity(
            inputs.run_dir / f'{hyper_params["checkpoint"]}.pt',
            content_hash=True,
        ),
        'audio_model_checkpoint': (
            file_identity(
                inputs.run_dir / 'audio_model_best.pt',
                content_hash=True,
            )
            if (inputs.run_dir / 'audio_model_best.pt').exists()
            else None
        ),
        'implementation_path': str(implementation_path),
        'source_manifest': source_manifest,
        'package_versions': package_versions,
        'ordered_subject_session_combos': [
            [int(subject), int(session)] for subject, session in SUB_SES_COMBOS
        ],
        'feature_specs': [asdict(spec) for spec in feature_specs],
        'mask_padding_ms': float(args.mask_padding_ms),
        'mask_padding_frames': int(padding_frames),
        'taper_ms': float(args.taper_ms),
        'taper_frames': int(taper_frames),
        'donor_story_id': int(args.donor_story_id),
        'donor_sound_ids': [int(value) for value in args.donor_sound_ids],
        'n_donor_pairs': int(args.n_donor_pairs),
        'analysis_seed': analysis_seed,
        'n_permutations': int(args.n_permutations),
        'save_ranks': bool(args.save_ranks),
        'meg_inference_batch_size': inference_batch_size,
        'meg_inference_batch_source': inference_batch_source,
        'baseline_reproduction_batch_size': canonical_baseline_batch_size,
        'audio_candidate_bank_batch_size': int(len(inputs.hidden_test)),
        'candidate_similarity_engine': 'cached_normalized_full_bank_trainer_einsum',
        'configured_device': configured_device,
        'device': str(device),
        'device_source': device_source,
        'device_description': device_description,
        'torch_deterministic': bool(torch.are_deterministic_algorithms_enabled()),
        'candidate_ids_sha256': hashlib.sha256(
            inputs.candidate_ids.astype(np.int64, copy=False).tobytes()
        ).hexdigest(),
        'meg_archive': file_identity(inputs.meg_path),
        'audio_embeddings': file_identity(
            inputs.audio_embeddings_path,
            content_hash=True,
        ),
        'train_dataframe': file_identity(
            inputs.train_dataframe_path,
            content_hash=True,
        ),
        'test_dataframe': file_identity(
            inputs.test_dataframe_path,
            content_hash=True,
        ),
        'annotation_artifacts': annotation_artifacts,
    }
    configuration_id = analysis_configuration_id(analysis_configuration)
    run_name = str(hyper_params.get('run_name', '')).strip()
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else default_analysis_output_dir(
            DEFAULT_OUTPUT_ROOT,
            run_group=args.run_group,
            run_name=run_name,
            run_id=args.run_id,
        )
    )
    prepare_output_directory(
        output_dir,
        configuration_id=configuration_id,
        configuration=analysis_configuration,
    )
    completed_result = load_completed_result(
        output_dir,
        configuration_id=configuration_id,
        save_ranks=bool(args.save_ranks),
    )
    if completed_result is not None:
        return completed_result

    feature_states: dict[str, FeatureState] = {}
    feature_plans: dict[str, dict[int, WindowPlan]] = {}
    feature_donor_candidate_caches: dict[str, dict[int, DonorCandidatePool]] = {}
    eligibility_records: list[EligibilityRecord] = []
    feature_eligibility: dict[str, dict[str, Any]] = {}
    for spec in tqdm(feature_specs, desc='Building feature/donor audits'):
        state = build_feature_state(
            spec,
            sound_lengths,
            donor_sounds,
            padding_frames=padding_frames,
            hop_samples=hop_samples,
            audio_dir=args.audio_dir,
            mfa_dir=args.mfa_dir,
            word_features=word_features,
        )
        donor_candidate_cache: dict[int, DonorCandidatePool] = {}
        plans, records = build_window_plans(
            state,
            inputs.test_windows,
            donor_sounds,
            hop_samples=hop_samples,
            taper_frames=taper_frames,
            n_donor_pairs=args.n_donor_pairs,
            seed=analysis_seed,
            donor_candidate_cache=donor_candidate_cache,
        )
        summary = summarize_eligibility(records)
        summary['n_total_combo_queries'] = summary['n_total_windows'] * len(
            SUB_SES_COMBOS
        )
        summary['n_eligible_combo_queries'] = summary['n_eligible_windows'] * len(
            SUB_SES_COMBOS
        )
        summary['n_discarded_combo_queries'] = summary['n_discarded_windows'] * len(
            SUB_SES_COMBOS
        )
        summary['discard_reason_combo_queries'] = {
            reason: count * len(SUB_SES_COMBOS)
            for reason, count in summary['discard_reasons'].items()
        }
        feature_states[spec.name] = state
        feature_plans[spec.name] = plans
        feature_donor_candidate_caches[spec.name] = donor_candidate_cache
        feature_eligibility[spec.name] = summary
        eligibility_records.extend(records)

    eligibility_table = pd.DataFrame([asdict(record) for record in eligibility_records])
    atomic_write_csv(output_dir / 'window_eligibility.csv', eligibility_table)
    feature_diagnostics: dict[str, dict[str, Any]] = {}
    for spec in feature_specs:
        feature_diagnostics[spec.name] = summarize_feature_diagnostics(
            feature_states[spec.name],
            test_windows=inputs.test_windows,
            donor_sounds=donor_sounds,
            plans=feature_plans[spec.name],
            donor_cache=feature_donor_candidate_caches[spec.name],
            eligibility=feature_eligibility[spec.name],
        )
    test_present_masks = {
        spec.name: concatenate_test_present_mask(
            feature_states[spec.name].present,
            inputs.test_windows,
        )
        for spec in feature_specs
    }
    jaccard_table, containment_table = pairwise_jaccard(test_present_masks)
    atomic_write_csv(
        output_dir / 'feature_mask_jaccard.csv',
        jaccard_table.reset_index().rename(columns={'index': 'feature'}),
    )
    atomic_write_csv(
        output_dir / 'feature_mask_containment.csv',
        containment_table.reset_index().rename(columns={'index': 'feature'}),
    )
    for suffix in ('png', 'pdf'):
        final_heatmap = output_dir / f'feature_mask_jaccard.{suffix}'
        temporary_heatmap = output_dir / (
            f'.feature_mask_jaccard.{os.getpid()}.{suffix}'
        )
        plot_jaccard_heatmap(jaccard_table, temporary_heatmap)
        temporary_heatmap.replace(final_heatmap)

    analyzable_specs = [spec for spec in feature_specs if feature_plans[spec.name]]
    if not analyzable_specs:
        empty = pd.DataFrame()
        atomic_write_csv(output_dir / 'feature_summary.csv', empty)
        atomic_write_csv(output_dir / 'combo_metrics.csv', empty)
        atomic_write_csv(output_dir / 'participant_metrics.csv', empty)
        audit_stats = {
            'analysis': ANALYSIS_NAME,
            'status': 'not_analyzable',
            'configuration_id': configuration_id,
            'run_id': args.run_id,
            'run_group': args.run_group,
            'analysis_configuration_path': str(output_dir / 'analysis_config.json'),
            'run_configuration': {
                'config_path': str(inputs.run_dir / 'config.json'),
                'checkpoint_path': str(
                    inputs.run_dir / f'{hyper_params["checkpoint"]}.pt'
                ),
                'audio_model_checkpoint_path': str(
                    inputs.run_dir / 'audio_model_best.pt'
                ),
                'training_seed': hyper_params.get('seed'),
                'training_batch_size': int(hyper_params['batch_size']),
                'test_batch_size': configured_test_batch_size,
                'baseline_reproduction_batch_size': canonical_baseline_batch_size,
                'meg_inference_batch_size': inference_batch_size,
                'meg_inference_batch_source': inference_batch_source,
                'audio_candidate_bank_batch_size': int(len(inputs.candidate_ids)),
                'audio_candidate_bank_batch_source': (
                    'full_bank_matching_trainer_evaluation'
                ),
                'analysis_seed': analysis_seed,
                'analysis_seed_source': (
                    'run_config.seed' if args.seed is None else 'explicit_cli_override'
                ),
                'model_mode': 'eval',
                'audio_model_mode': 'eval',
                'torch_deterministic_configured': configured_torch_deterministic,
                'torch_deterministic_enabled': bool(
                    torch.are_deterministic_algorithms_enabled()
                ),
                'configured_device': configured_device,
                'device': str(device),
                'device_source': device_source,
                'using_default_runtime_settings': (using_default_runtime_settings),
                'runtime_configuration_class': runtime_configuration_class,
                'device_description': device_description,
                'torch_version': torch.__version__,
                'numpy_version': np.__version__,
                'inference_dtype': 'torch.float32',
            },
            'input_artifacts': {
                'configured_dirprocess': str(inputs.configured_dirprocess),
                'resolved_dirprocess': str(inputs.dirprocess),
                'used_dirprocess_fallback': (
                    inputs.configured_dirprocess != inputs.dirprocess
                ),
                'meg_archive': str(inputs.meg_path),
                'audio_embeddings': str(inputs.audio_embeddings_path),
                'train_dataframe': str(inputs.train_dataframe_path),
                'test_dataframe': str(inputs.test_dataframe_path),
                'raw_audio_dir': str(args.audio_dir),
                'mfa_textgrid_dir': str(args.mfa_dir),
                'word_features': (
                    str(word_features_path) if needs_word_features else None
                ),
            },
            'scientific_estimand': (
                'annotation-present versus annotation-absent raw-MEG '
                'substitution advantage among eligible feature-positive queries'
            ),
            'primary_effect': (
                'strict_rank_f_to_absent_minus_strict_rank_f_to_present'
            ),
            'candidate_bank': {
                'size': int(len(inputs.candidate_ids)),
                'candidate_ids': inputs.candidate_ids.tolist(),
                'filtering': ('never filtered; identical for baseline and both arms'),
                'normalization': (
                    'encoded and normalized once, then reused for every query batch'
                ),
                'similarity_reduction': (
                    'same dimensional normalization and einsum as trainer evaluation'
                ),
            },
            'query_filtering': (
                'feature eligibility would be applied identically to baseline, '
                'f_to_absent, and f_to_present metrics only after full-bank ranks'
            ),
            'mask': {
                'padding_ms': float(args.mask_padding_ms),
                'padding_frames': int(padding_frames),
                'taper_ms': float(args.taper_ms),
                'taper_frames': int(taper_frames),
                'all_occurrences_simultaneously': True,
            },
            'donors': {
                'story_id': int(args.donor_story_id),
                'sound_ids': [int(value) for value in args.donor_sound_ids],
                'sound_fnames': sorted(donor_sounds),
                'same_subject_session': True,
                'same_file_within_pair': True,
                'cross_file_gaps_allowed': False,
                'n_distinct_pairs_per_component': int(args.n_donor_pairs),
                'timestamp_draw_scope': 'participant_specific',
                'shared_across_sessions_within_participant': True,
                'participant_seed_rule': (
                    "stable_seed(analysis_seed, 'participant_donors', subject_id)"
                ),
                'distinctness_scope': (
                    'within each connected target component; different target '
                    'components are sampled independently'
                ),
            },
            'features': {
                spec.name: {
                    'spec': asdict(spec),
                    'state_metadata': feature_states[spec.name].metadata,
                    'eligibility': feature_eligibility[spec.name],
                    'diagnostics': feature_diagnostics[spec.name],
                    'result': {
                        'conclusion': 'not_analyzable',
                        'reason': 'no_eligible_windows',
                    },
                }
                for spec in feature_specs
            },
            'feature_mask_overlap': {
                'jaccard': 'feature_mask_jaccard.csv',
                'containment': 'feature_mask_containment.csv',
                'heatmap': ['feature_mask_jaccard.png', 'feature_mask_jaccard.pdf'],
                'zero_union': 'jaccard=1 when both masks are empty',
            },
        }
        atomic_write_json(output_dir / 'stats.json', audit_stats)
        atomic_write_json(
            output_dir / 'COMPLETED.json',
            {
                'status': 'not_analyzable',
                'configuration_id': configuration_id,
            },
        )
        return {
            'output_dir': output_dir,
            'stats': audit_stats,
            'feature_summary': empty,
            'combo_metrics': empty,
            'participant_metrics': empty,
            'eligibility': eligibility_table,
        }

    n_combos = len(SUB_SES_COMBOS)
    n_windows = len(inputs.candidate_ids)
    checkpoint_dir = output_dir / 'checkpoints'
    baseline_ranks = open_rank_memmap(
        checkpoint_dir / 'baseline_rank_full_bank.npy',
        (n_combos, n_windows),
    )
    rank_absent: dict[str, np.memmap] = {}
    rank_present: dict[str, np.memmap] = {}
    eligible_masks: dict[str, np.ndarray] = {}
    for feature_index, spec in enumerate(analyzable_specs):
        checkpoint_tag = (
            f'{feature_index:02d}_{hashlib.sha256(spec.name.encode()).hexdigest()[:12]}'
        )
        rank_absent[spec.name] = open_rank_memmap(
            checkpoint_dir / f'{checkpoint_tag}_rank_f_to_absent.npy',
            (n_combos, n_windows, args.n_donor_pairs),
        )
        rank_present[spec.name] = open_rank_memmap(
            checkpoint_dir / f'{checkpoint_tag}_rank_f_to_present.npy',
            (n_combos, n_windows, args.n_donor_pairs),
        )
        eligible = np.zeros(n_windows, dtype=bool)
        eligible[list(feature_plans[spec.name])] = True
        eligible_masks[spec.name] = eligible

    expected_combos = set(SUB_SES_COMBOS)
    observed_combos = set(
        zip(
            inputs.df_test['subject_id'].astype(int),
            inputs.df_test['session_id'].astype(int),
        )
    )
    if observed_combos != expected_combos:
        raise ValueError(
            'Test subject-session combinations do not match project constants.'
        )

    baseline_was_complete = [
        baseline_checkpoint_complete(
            baseline_ranks,
            combo_index,
            candidate_bank_size=n_windows,
        )
        for combo_index in range(n_combos)
    ]
    occlusion_was_complete: dict[tuple[int, str], bool] = {}
    for combo_index in range(n_combos):
        for spec in analyzable_specs:
            occlusion_was_complete[(combo_index, spec.name)] = (
                occlusion_checkpoint_complete(
                    rank_absent[spec.name],
                    rank_present[spec.name],
                    combo_index,
                    eligible_masks[spec.name],
                    candidate_bank_size=n_windows,
                )
            )

    any_forward_needed = not all(baseline_was_complete) or not all(
        occlusion_was_complete.values()
    )
    candidate_bank: torch.Tensor | None = None
    candidate_ids_device: torch.Tensor | None = None
    if any_forward_needed:
        candidate_bank = encode_candidate_bank(
            inputs.audio_model,
            inputs.hidden_test,
            device,
        )
        candidate_ids_device = torch.as_tensor(
            inputs.candidate_ids, dtype=torch.long, device=device
        )

    # Compute unoccluded ranks using the same strict full-bank rank definition
    # used by both perturbation arms.
    with np.load(inputs.meg_path) as archive:
        baseline_progress = tqdm(
            list(enumerate(SUB_SES_COMBOS)),
            desc='Unoccluded baseline queries',
        )
        for combo_index, (subject, session) in baseline_progress:
            baseline_progress.set_postfix_str(combo_key(subject, session))
            if baseline_was_complete[combo_index]:
                combo_baseline = np.asarray(baseline_ranks[combo_index])
            else:
                if candidate_bank is None or candidate_ids_device is None:
                    raise RuntimeError('Candidate bank was not prepared.')
                raw_key = meg_subset_key(subject, session, args.donor_story_id)
                if raw_key not in archive.files:
                    raise KeyError(f'MEG archive does not contain {raw_key!r}.')
                raw_story = np.asarray(archive[raw_key], dtype=np.float32)
                combo_test = inputs.df_test[
                    (inputs.df_test['subject_id'].astype(int) == subject)
                    & (inputs.df_test['session_id'].astype(int) == session)
                ]
                targets = load_combo_targets(
                    raw_story,
                    combo_test,
                    meg_sr=meg_sr,
                    meg_offset_frames=meg_offset_frames,
                    expected_candidate_ids=inputs.candidate_ids,
                )
                combo_baseline = rank_query_batches(
                    inputs.model,
                    targets,
                    subject - 1,
                    inputs.candidate_ids,
                    candidate_bank,
                    candidate_ids_device,
                    device,
                    canonical_baseline_batch_size,
                )
                del targets, raw_story
            if not baseline_was_complete[combo_index]:
                baseline_ranks[combo_index] = combo_baseline
                baseline_ranks.flush()

    if np.any(baseline_ranks < 1):
        raise RuntimeError('Some baseline ranks were not evaluated.')

    eligible_window_counts = {
        spec.name: len(feature_plans[spec.name]) for spec in analyzable_specs
    }
    total_query_work, completed_query_work = occlusion_progress_totals(
        occlusion_was_complete,
        eligible_window_counts,
        args.n_donor_pairs,
    )
    pending_forward_batches = pending_occlusion_forward_batches(
        occlusion_was_complete,
        eligible_window_counts,
        args.n_donor_pairs,
        inference_batch_size,
    )
    pending_query_work = total_query_work - completed_query_work
    total_feature_cells = len(occlusion_was_complete)
    saved_feature_cells = int(sum(occlusion_was_complete.values()))
    tqdm.write(
        'Perturbed-query workload: '
        f'{pending_query_work:,}/{total_query_work:,} queries pending; '
        f'{pending_forward_batches:,} MEG forward batches at '
        f'batch_size={inference_batch_size}; '
        f'{saved_feature_cells:,}/{total_feature_cells:,} feature cells reused.'
    )
    subject_pending_feature_names: dict[int, list[str]] = {}
    for subject in sorted({subject for subject, _ in SUB_SES_COMBOS}):
        subject_combo_indices = [
            combo_index
            for combo_index, (combo_subject, _session) in enumerate(SUB_SES_COMBOS)
            if combo_subject == subject
        ]
        subject_pending_feature_names[subject] = [
            spec.name
            for spec in analyzable_specs
            if any(
                not occlusion_was_complete[(combo_index, spec.name)]
                for combo_index in subject_combo_indices
            )
        ]

    planned_subject: int | None = None
    participant_feature_plans: dict[str, dict[int, WindowPlan]] = {}
    with (
        np.load(inputs.meg_path) as archive,
        tqdm(
            total=total_query_work,
            initial=completed_query_work,
            desc='Perturbed queries evaluated',
            unit='query',
            unit_scale=True,
            dynamic_ncols=True,
        ) as occlusion_progress,
    ):
        for combo_index, (subject, session) in enumerate(SUB_SES_COMBOS):
            incomplete_specs = [
                spec
                for spec in analyzable_specs
                if not occlusion_was_complete[(combo_index, spec.name)]
            ]
            if not incomplete_specs:
                continue
            if candidate_bank is None:
                raise RuntimeError('Candidate bank was not prepared.')
            if planned_subject != subject:
                occlusion_progress.set_postfix_str(
                    f'{combo_key(subject, session)} planning donors'
                )
                participant_feature_plans = build_participant_feature_plans(
                    subject=subject,
                    base_feature_plans=feature_plans,
                    feature_donor_candidate_caches=(feature_donor_candidate_caches),
                    feature_names=subject_pending_feature_names[subject],
                    n_donor_pairs=args.n_donor_pairs,
                    analysis_seed=analysis_seed,
                )
                planned_subject = subject
            occlusion_progress.set_postfix_str(
                f'{combo_key(subject, session)} loading MEG'
            )
            raw_key = meg_subset_key(subject, session, args.donor_story_id)
            if raw_key not in archive.files:
                raise KeyError(f'MEG archive does not contain {raw_key!r}.')
            # One story array at a time; never dict(np.load(...)).
            raw_story = np.asarray(archive[raw_key], dtype=np.float32)
            combo_test = inputs.df_test[
                (inputs.df_test['subject_id'].astype(int) == subject)
                & (inputs.df_test['session_id'].astype(int) == session)
            ]
            combo_train = inputs.df_train[
                (inputs.df_train['subject_id'].astype(int) == subject)
                & (inputs.df_train['session_id'].astype(int) == session)
            ]
            targets = load_combo_targets(
                raw_story,
                combo_test,
                meg_sr=meg_sr,
                meg_offset_frames=meg_offset_frames,
                expected_candidate_ids=inputs.candidate_ids,
            )
            origins = derive_sound_origins(
                combo_train,
                story_id=args.donor_story_id,
                donor_sounds=donor_sounds,
                meg_sr=meg_sr,
            )
            donor_reader = ComboDonorReader(
                raw_story,
                origins,
                meg_offset_frames,
            )
            for spec in incomplete_specs:
                cell_query_total = (
                    2 * args.n_donor_pairs * len(participant_feature_plans[spec.name])
                )
                cell_queries_done = 0

                def update_occlusion_progress(completed: int) -> None:
                    nonlocal cell_queries_done
                    cell_queries_done += completed
                    occlusion_progress.update(completed)
                    occlusion_progress.set_postfix_str(
                        f'{combo_key(subject, session)} {spec.name} '
                        f'feature_queries={cell_queries_done:,}/'
                        f'{cell_query_total:,} '
                        f'cells={saved_feature_cells}/{total_feature_cells}'
                    )

                occlusion_progress.set_postfix_str(
                    f'{combo_key(subject, session)} {spec.name} '
                    f'feature_queries=0/{cell_query_total:,} '
                    f'cells={saved_feature_cells}/{total_feature_cells}'
                )
                absent, present = evaluate_occlusion_queries(
                    model=inputs.model,
                    targets=targets,
                    plans=participant_feature_plans[spec.name],
                    donor_reader=donor_reader,
                    n_donor_pairs=args.n_donor_pairs,
                    subject_index=subject - 1,
                    candidate_bank=candidate_bank,
                    candidate_ids=inputs.candidate_ids,
                    device=device,
                    batch_size=inference_batch_size,
                    progress_callback=update_occlusion_progress,
                )
                if cell_queries_done != cell_query_total:
                    raise RuntimeError(
                        f'Progress accounting mismatch for {spec.name}: '
                        f'completed={cell_queries_done}, '
                        f'expected={cell_query_total}.'
                    )
                rank_absent[spec.name][combo_index] = absent
                rank_present[spec.name][combo_index] = present
                rank_absent[spec.name].flush()
                rank_present[spec.name].flush()
                saved_feature_cells += 1
                occlusion_progress.set_postfix_str(
                    f'{combo_key(subject, session)} {spec.name} saved '
                    f'cells={saved_feature_cells}/{total_feature_cells}'
                )
            del targets, raw_story

    for combo_index in range(n_combos):
        for spec in analyzable_specs:
            if not occlusion_checkpoint_complete(
                rank_absent[spec.name],
                rank_present[spec.name],
                combo_index,
                eligible_masks[spec.name],
                candidate_bank_size=n_windows,
            ):
                raise RuntimeError(
                    'Occlusion checkpoint remains incomplete for '
                    f'{combo_key(*SUB_SES_COMBOS[combo_index])}/{spec.name}.'
                )

    combo_rows: list[dict[str, Any]] = []
    participant_frames: list[pd.DataFrame] = []
    for spec in analyzable_specs:
        eligible = np.zeros(n_windows, dtype=bool)
        eligible[list(feature_plans[spec.name])] = True
        for combo_index, (subject, session) in enumerate(SUB_SES_COMBOS):
            metrics = paired_feature_metrics(
                baseline_ranks[combo_index],
                rank_absent[spec.name][combo_index],
                rank_present[spec.name][combo_index],
                eligible,
            )
            combo_rows.append(
                {
                    'feature': spec.name,
                    'subject_id': subject,
                    'session_id': session,
                    'candidate_bank_size': n_windows,
                    **metrics,
                }
            )
        feature_combo = pd.DataFrame(
            [row for row in combo_rows if row['feature'] == spec.name]
        )
        value_columns = [
            column
            for column in feature_combo.columns
            if column
            not in {
                'feature',
                'subject_id',
                'session_id',
                'candidate_bank_size',
                'n_queries',
            }
        ]
        participant = aggregate_sessions_to_participants(
            feature_combo,
            value_columns,
        )
        participant.insert(0, 'feature', spec.name)
        participant_frames.append(participant)

    combo_table = pd.DataFrame(combo_rows)
    participant_table = pd.concat(participant_frames, ignore_index=True)
    tested_names = [spec.name for spec in analyzable_specs]
    subject_order = sorted(participant_table['subject_id'].astype(int).unique())
    effect_matrix = np.column_stack(
        [
            participant_table[participant_table['feature'] == name]
            .set_index('subject_id')
            .loc[subject_order, 'effect_rank_absent_minus_present']
            .to_numpy(dtype=np.float64)
            for name in tested_names
        ]
    )
    permutation = two_sided_signflip_max_abs_t(
        effect_matrix,
        n_permutations=args.n_permutations,
        seed=analysis_seed,
    )
    max_t_null = permutation['max_abs_t_null']
    max_t_null_probabilities = (0.5, 0.9, 0.95, 0.975, 0.99)
    max_t_null_quantile_values = np.quantile(
        max_t_null,
        max_t_null_probabilities,
    )
    max_t_null_quantiles = {
        f'q{int(round(probability * 1000)):03d}': float(value)
        for probability, value in zip(
            max_t_null_probabilities,
            max_t_null_quantile_values,
            strict=True,
        )
    }

    summary_rows: list[dict[str, Any]] = []
    for feature_index, spec in enumerate(analyzable_specs):
        participant = participant_table[
            participant_table['feature'] == spec.name
        ].sort_values('subject_id')
        effects = participant['effect_rank_absent_minus_present'].to_numpy(
            dtype=np.float64
        )
        ci_low, ci_high = participant_t_interval(effects)
        eligibility = feature_eligibility[spec.name]
        control_mean_rank = float(participant['f_to_present_mean_rank'].mean())
        mean_absent_rank = float(participant['f_to_absent_mean_rank'].mean())
        mean_baseline_rank = float(participant['baseline_mean_rank'].mean())
        mean_effect = float(np.mean(effects))
        corrected_p = float(permutation['max_t_fwer_p'][feature_index])
        raw_p = float(permutation['raw_p'][feature_index])
        conclusion, control_saturation_rank, control_saturated = (
            classify_feature_result(
                mean_effect=mean_effect,
                corrected_p=corrected_p,
                feature_present_control_mean_rank=control_mean_rank,
                candidate_bank_size=n_windows,
            )
        )
        diagnostics = feature_diagnostics[spec.name]
        summary_rows.append(
            {
                'feature': spec.name,
                'label': spec.label or spec.name,
                'mean_rank_effect': mean_effect,
                'ci95_low': ci_low,
                'ci95_high': ci_high,
                'observed_t': float(permutation['observed_t'][feature_index]),
                'observed_abs_t': float(permutation['observed_abs_t'][feature_index]),
                'p_raw_two_sided': raw_p,
                'p_max_t_fwer': corrected_p,
                'mean_rank_absent_damage': float(
                    participant['damage_rank_absent_minus_baseline'].mean()
                ),
                'mean_rank_present_damage': float(
                    participant['damage_rank_present_minus_baseline'].mean()
                ),
                'mean_eligible_baseline_rank': mean_baseline_rank,
                'mean_f_to_absent_rank': mean_absent_rank,
                'mean_f_to_present_rank': control_mean_rank,
                'mean_feature_present_control_rank': control_mean_rank,
                'control_saturation_rank': control_saturation_rank,
                'control_saturated': control_saturated,
                'median_support_fraction': float(
                    eligibility['eligible_support_fraction_median']
                ),
                'mean_weighted_fraction': float(
                    eligibility['eligible_weighted_fraction_mean']
                ),
                'mean_meg_replaced_fraction': diagnostics[
                    'mean_meg_replaced_fraction'
                ],
                'n_eligible_windows': int(eligibility['n_eligible_windows']),
                'n_discarded_windows': int(eligibility['n_discarded_windows']),
                'n_test_support_frames': diagnostics['n_test_support_frames'],
                'test_support_fraction': diagnostics['test_support_fraction'],
                'n_test_feature_intervals': diagnostics['n_test_feature_intervals'],
                'mean_test_interval_length': diagnostics['mean_test_interval_length'],
                'mean_eligible_component_length': diagnostics[
                    'mean_eligible_component_length'
                ],
                'n_eligible_components': diagnostics['n_eligible_components'],
                'donor_present_frames': diagnostics['donor_present_frames'],
                'donor_absent_frames': diagnostics['donor_absent_frames'],
                'donor_present_intervals': diagnostics['donor_present_intervals'],
                'donor_absent_intervals': diagnostics['donor_absent_intervals'],
                'donor_present_pool_starts': diagnostics['donor_present_pool_starts'],
                'donor_absent_pool_starts': diagnostics['donor_absent_pool_starts'],
                'discard_reasons': json.dumps(
                    eligibility['discard_reasons'], sort_keys=True
                ),
                'conclusion': conclusion,
            }
        )
    summary_table = pd.DataFrame(summary_rows)

    atomic_write_csv(output_dir / 'combo_metrics.csv', combo_table)
    atomic_write_csv(output_dir / 'participant_metrics.csv', participant_table)
    atomic_write_csv(output_dir / 'feature_summary.csv', summary_table)
    for suffix in ('png', 'pdf'):
        final_plot = output_dir / f'feature_effects.{suffix}'
        temporary_plot = output_dir / f'.feature_effects.{os.getpid()}.{suffix}'
        plot_feature_summary(
            summary_table,
            participant_table,
            temporary_plot,
            null_mean_effects=permutation['null_mean'],
        )
        temporary_plot.replace(final_plot)

    if args.save_ranks:
        arrays: dict[str, np.ndarray] = {
            'candidate_ids': inputs.candidate_ids,
            'combo_subject': np.asarray(
                [subject for subject, _ in SUB_SES_COMBOS], dtype=np.int16
            ),
            'combo_session': np.asarray(
                [session for _, session in SUB_SES_COMBOS], dtype=np.int8
            ),
            'baseline_rank_full_bank': baseline_ranks,
        }
        for spec in analyzable_specs:
            name = spec.name
            eligible = np.zeros(n_windows, dtype=np.uint8)
            eligible[list(feature_plans[name])] = 1
            arrays[f'{name}__eligible_query'] = eligible
            arrays[f'{name}__rank_f_to_absent'] = rank_absent[name]
            arrays[f'{name}__rank_f_to_present'] = rank_present[name]
        final_ranks = output_dir / 'ranks.npz'
        temporary_ranks = output_dir / f'.ranks.{os.getpid()}.npz'
        with temporary_ranks.open('wb') as file:
            np.savez_compressed(file, **arrays)
        temporary_ranks.replace(final_ranks)

    feature_stats: dict[str, Any] = {}
    summary_by_name = summary_table.set_index('feature')
    for spec in feature_specs:
        item: dict[str, Any] = {
            'spec': asdict(spec),
            'state_metadata': feature_states[spec.name].metadata,
            'eligibility': feature_eligibility[spec.name],
            'diagnostics': feature_diagnostics[spec.name],
            'eligible_wav_indices': sorted(feature_plans[spec.name]),
        }
        if spec.name in summary_by_name.index:
            item['result'] = summary_by_name.loc[spec.name].to_dict()
        else:
            item['result'] = {
                'conclusion': 'not_analyzable',
                'reason': 'no_eligible_windows',
            }
        feature_stats[spec.name] = item

    stats: dict[str, Any] = {
        'analysis': ANALYSIS_NAME,
        'status': 'complete',
        'configuration_id': configuration_id,
        'run_id': args.run_id,
        'run_group': args.run_group,
        'run_configuration': {
            'config_path': str(inputs.run_dir / 'config.json'),
            'checkpoint_path': str(inputs.run_dir / f'{hyper_params["checkpoint"]}.pt'),
            'audio_model_checkpoint_path': str(inputs.run_dir / 'audio_model_best.pt'),
            'training_seed': hyper_params.get('seed'),
            'training_batch_size': int(hyper_params['batch_size']),
            'test_batch_size': configured_test_batch_size,
            'baseline_reproduction_batch_size': canonical_baseline_batch_size,
            'meg_inference_batch_size': inference_batch_size,
            'meg_inference_batch_source': inference_batch_source,
            'audio_candidate_bank_batch_size': int(n_windows),
            'audio_candidate_bank_batch_source': (
                'full_bank_matching_trainer_evaluation'
            ),
            'analysis_seed': analysis_seed,
            'analysis_seed_source': (
                'run_config.seed' if args.seed is None else 'explicit_cli_override'
            ),
            'model_mode': 'eval',
            'audio_model_mode': 'eval',
            'torch_deterministic_configured': configured_torch_deterministic,
            'torch_deterministic_enabled': bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            'configured_device': configured_device,
            'device': str(device),
            'device_source': device_source,
            'using_default_runtime_settings': (using_default_runtime_settings),
            'runtime_configuration_class': runtime_configuration_class,
            'device_description': device_description,
            'torch_version': torch.__version__,
            'numpy_version': np.__version__,
            'inference_dtype': 'torch.float32',
        },
        'input_artifacts': {
            'configured_dirprocess': str(inputs.configured_dirprocess),
            'resolved_dirprocess': str(inputs.dirprocess),
            'used_dirprocess_fallback': (
                inputs.configured_dirprocess != inputs.dirprocess
            ),
            'meg_archive': str(inputs.meg_path),
            'audio_embeddings': str(inputs.audio_embeddings_path),
            'train_dataframe': str(inputs.train_dataframe_path),
            'test_dataframe': str(inputs.test_dataframe_path),
            'raw_audio_dir': str(args.audio_dir),
            'mfa_textgrid_dir': str(args.mfa_dir),
            'word_features': (str(word_features_path) if needs_word_features else None),
        },
        'scientific_estimand': (
            'annotation-present versus annotation-absent raw-MEG substitution '
            'advantage among eligible feature-positive queries'
        ),
        'primary_effect': ('strict_rank_f_to_absent_minus_strict_rank_f_to_present'),
        'rank_convention': (
            'one_based; 1 + number of candidates with similarity strictly greater '
            'than the true candidate'
        ),
        'baseline_damage_convention': (
            'occluded_rank_minus_baseline_rank; positive means worse retrieval'
        ),
        'cross_feature_interpretation': (
            'effects are total feature-positive-query contrasts with matched mask '
            'extent within feature; do not rank features as encoding strengths; '
            'interpret alongside reported mask occupancy'
        ),
        'candidate_bank': {
            'size': int(n_windows),
            'candidate_ids': inputs.candidate_ids.tolist(),
            'filtering': 'never filtered; identical for baseline and both arms',
            'normalization': (
                'encoded and normalized once, then reused for every query batch'
            ),
            'similarity_reduction': (
                'same dimensional normalization and einsum as trainer evaluation'
            ),
        },
        'resume': {
            'checkpoint_directory': str(checkpoint_dir),
            'baseline_combos_reused': int(sum(baseline_was_complete)),
            'baseline_combos_computed': int(
                len(baseline_was_complete) - sum(baseline_was_complete)
            ),
            'feature_combo_cells_reused': int(sum(occlusion_was_complete.values())),
            'feature_combo_cells_computed': int(
                len(occlusion_was_complete) - sum(occlusion_was_complete.values())
            ),
            'perturbed_queries_total': int(total_query_work),
            'perturbed_queries_reused': int(completed_query_work),
            'perturbed_queries_computed': int(pending_query_work),
            'meg_forward_batches_computed': int(pending_forward_batches),
            'checkpoint_granularity': 'complete subject-session/feature cell',
        },
        'query_filtering': (
            'feature eligibility is applied identically to baseline, f_to_absent, '
            'and f_to_present metrics only after full-bank ranks are computed'
        ),
        'aggregation': [
            'donor pairs averaged within query',
            'eligible queries averaged within session',
            'sessions averaged within participant',
            'participants are the inferential units',
        ],
        'inference': {
            'alternative': 'mean participant effect != 0',
            'test': 'shared participant sign flips',
            'multiple_comparisons': 'two-sided single-step max-|T| FWER',
            'n_permutations': int(args.n_permutations),
            'n_participants': int(len(subject_order)),
            'tested_features': tested_names,
            'inferential_units': 'participants',
            'max_abs_t_null': {
                'mean': float(np.mean(max_t_null)),
                'standard_deviation': float(np.std(max_t_null, ddof=1)),
                'quantiles': max_t_null_quantiles,
                'fwer_0.05_critical_abs_t': max_t_null_quantiles['q950'],
            },
            'control_validity_gate': {
                'field': 'mean_feature_present_control_rank',
                'saturated_when': 'greater_than_or_equal_to_random_rank_expectation',
                'random_rank_expectation': float((n_windows + 1) / 2.0),
                'overrides_positive_conclusion': True,
            },
        },
        'feature_mask_overlap': {
            'jaccard': 'feature_mask_jaccard.csv',
            'containment': 'feature_mask_containment.csv',
            'heatmap': ['feature_mask_jaccard.png', 'feature_mask_jaccard.pdf'],
            'zero_union': 'jaccard=1 when both masks are empty',
        },
        'mask': {
            'padding_ms': float(args.mask_padding_ms),
            'padding_frames': int(padding_frames),
            'taper_ms': float(args.taper_ms),
            'taper_frames': int(taper_frames),
            'all_occurrences_simultaneously': True,
        },
        'donors': {
            'story_id': int(args.donor_story_id),
            'sound_ids': [int(value) for value in args.donor_sound_ids],
            'sound_fnames': sorted(donor_sounds),
            'same_subject_session': True,
            'same_file_within_pair': True,
            'cross_file_gaps_allowed': False,
            'n_distinct_pairs_per_component': int(args.n_donor_pairs),
            'timestamp_draw_scope': 'participant_specific',
            'shared_across_sessions_within_participant': True,
            'participant_seed_rule': (
                "stable_seed(analysis_seed, 'participant_donors', subject_id)"
            ),
            'distinctness_scope': (
                'within each connected target component; different target '
                'components are sampled independently'
            ),
        },
        'features': feature_stats,
    }
    atomic_write_json(output_dir / 'stats.json', stats)
    atomic_write_json(
        output_dir / 'COMPLETED.json',
        {
            'status': 'complete',
            'configuration_id': configuration_id,
            'stats': 'stats.json',
        },
    )
    return {
        'output_dir': output_dir,
        'stats': stats,
        'feature_summary': summary_table,
        'combo_metrics': combo_table,
        'participant_metrics': participant_table,
        'eligibility': eligibility_table,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
    )
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--run-group', required=True)
    parser.add_argument(
        '--experiments-root',
        type=Path,
        default=Path(EXPERIMENTS_DIR),
    )
    parser.add_argument(
        '--device',
        default=None,
        help=(
            'Runtime device. By default, use the device recorded by the selected '
            'run. An explicit override creates a separately identified runtime '
            'configuration.'
        ),
    )
    parser.add_argument(
        '--audio-dir',
        type=Path,
        default=Path(DATA_ROOT) / 'MASC-MEG' / 'stimuli' / 'audio',
    )
    parser.add_argument(
        '--mfa-dir',
        type=Path,
        default=Path(DATA_ROOT) / 'MASC-MEG' / 'stimuli' / 'combined_mfa',
    )
    parser.add_argument(
        '--word-features',
        type=Path,
        default=None,
        help=(
            'Linguistic feature table. By default, use '
            '<resolved run dirprocess>/linguistic/word_features.csv.'
        ),
    )
    parser.add_argument(
        '--features-json',
        type=Path,
        default=None,
        help='Optional JSON list of FeatureSpec fields; default is the fixed battery.',
    )
    parser.add_argument('--donor-story-id', type=int, default=DEFAULT_TARGET_STORY_ID)
    parser.add_argument(
        '--donor-sound-ids',
        type=int,
        nargs='+',
        default=list(DEFAULT_DONOR_SOUND_IDS),
    )
    parser.add_argument('--mask-padding-ms', type=float, default=20.0)
    parser.add_argument('--taper-ms', type=float, default=20.0)
    parser.add_argument('--n-donor-pairs', type=int, default=5)
    parser.add_argument(
        '--inference-batch-size',
        type=int,
        default=None,
        help=(
            'Runtime-only chunk size for MEG forwards. The default is 1000 on '
            'CUDA and max(training batch_size, test_batch_size) on CPU. Lower '
            'this value if CUDA reports out of memory. The audio candidate bank '
            'is always forwarded once as one full bank.'
        ),
    )
    parser.add_argument('--n-permutations', type=int, default=100_000)
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help=(
            'Donor-selection/permutation seed. By default, reuse seed from the '
            'selected run config.'
        ),
    )
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument(
        '--plot-preview',
        type=Path,
        default=None,
        help=(
            'Render the current feature-effects figure with clearly marked fake '
            'values and exit without loading data or a model.'
        ),
    )
    parser.add_argument(
        '--save-ranks',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main_cli(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.plot_preview is not None:
        write_plot_style_preview(args.plot_preview)
        print(f'Wrote fake-data plot style preview to {args.plot_preview}')
        return
    if args.inference_batch_size is not None and args.inference_batch_size <= 0:
        raise SystemExit('--inference-batch-size must be positive.')
    result = run_analysis(args)
    print(f'Wrote {ANALYSIS_NAME} results to {result["output_dir"]}')


if __name__ == '__main__':
    main_cli()
