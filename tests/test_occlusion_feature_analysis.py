"""Tests for the standalone paired raw-MEG occlusion experiment."""

import numpy as np
import pandas as pd
import pytest
import torch

from lisa.training.criteria import get_clip_similarity

ofa = pytest.importorskip('lisa.plots.occlusion_feature_analysis')


def test_runtime_overrides_are_unset_until_device_aware_defaults_are_selected():
    args = ofa.build_parser().parse_args(
        ['--run-id', 'run-id', '--run-group', 'run-group']
    )

    assert args.device is None
    assert args.inference_batch_size is None
    assert args.seed is None
    assert ofa.DEFAULT_CUDA_INFERENCE_BATCH_SIZE == 1000


def test_default_output_path_matches_other_run_specific_plot_analyses(tmp_path):
    output = ofa.default_analysis_output_dir(
        tmp_path,
        run_group='2conv-seed42-paper',
        run_name='2conv-seed42-paper-25branches',
        run_id='abc123',
    )

    assert output == (
        tmp_path / '2conv-seed42-paper' / '2conv-seed42-paper-25branches-abc123'
    )


def test_plot_style_preview_is_rendered_without_analysis_data(tmp_path):
    output = tmp_path / 'preview.png'

    ofa.write_plot_style_preview(output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_canonical_baseline_keeps_recorded_test_batch_while_cuda_is_optimized():
    canonical, perturbation, source = ofa.select_evaluation_batch_sizes(
        device_type='cuda',
        training_batch_size=100,
        test_batch_size=20,
        inference_override=None,
    )

    assert canonical == 20
    assert perturbation == 1000
    assert source == 'optimized_cuda_default'


def test_explicit_perturbation_batch_does_not_change_canonical_baseline_batch():
    canonical, perturbation, source = ofa.select_evaluation_batch_sizes(
        device_type='cuda',
        training_batch_size=100,
        test_batch_size=20,
        inference_override=500,
    )

    assert canonical == 20
    assert perturbation == 500
    assert source == 'explicit_cli_override'


def test_strict_rank_uses_every_candidate_column():
    # Candidate 40 would disappear if someone incorrectly reduced the bank to
    # eligible query IDs. It must still outrank the true candidate 10.
    similarity = torch.tensor([[0.5, 0.2, 0.6, 0.8]])
    candidate_ids = torch.tensor([10, 20, 30, 40])
    labels = torch.tensor([10])

    rank = ofa.strict_one_based_ranks(similarity, labels, candidate_ids)

    assert rank.tolist() == [3]


def test_strict_rank_ties_do_not_outrank_true_candidate():
    similarity = torch.tensor([[0.5, 0.5, 0.6]])
    rank = ofa.strict_one_based_ranks(
        similarity,
        labels=torch.tensor([7]),
        candidate_ids=torch.tensor([7, 8, 9]),
    )
    assert rank.tolist() == [2]


def test_strict_rank_rejects_nonfinite_similarity():
    with pytest.raises(ValueError, match='non-finite'):
        ofa.strict_one_based_ranks(
            torch.tensor([[float('nan'), 0.2]]),
            labels=torch.tensor([7]),
            candidate_ids=torch.tensor([7, 8]),
        )


def test_candidate_bank_is_forwarded_as_one_full_trainer_style_batch():
    class RecordingAudioModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(self, values):
            self.batch_sizes.append(values.shape[0])
            return values

    model = RecordingAudioModel()
    hidden = np.ones((7, 3, 2), dtype=np.float32)

    bank = ofa.encode_candidate_bank(model, hidden, torch.device('cpu'))

    assert model.batch_sizes == [7]
    assert bank.shape == (7, 2, 3)
    torch.testing.assert_close(
        torch.linalg.vector_norm(bank, dim=(-2, -1)),
        torch.ones(7),
    )


@pytest.mark.parametrize('embedding_ndim', [2, 3])
def test_cached_candidate_normalization_matches_original_clip_ranks_exactly(
    embedding_ndim,
    monkeypatch,
):
    generator = torch.Generator().manual_seed(17 + embedding_ndim)
    if embedding_ndim == 3:
        queries = torch.randn(11, 13, 17, generator=generator)
        candidates = torch.randn(19, 17, 13, generator=generator).transpose(1, 2)
        normalized_candidates = torch.nn.functional.normalize(
            candidates,
            dim=(-2, -1),
        )
    else:
        queries = torch.randn(11, 29, generator=generator)
        candidates = torch.randn(19, 29, generator=generator)
        normalized_candidates = torch.nn.functional.normalize(candidates, dim=-1)
    candidate_ids = torch.arange(100, 119)
    labels = candidate_ids[torch.tensor([0, 4, 8, 12, 16, 1, 5, 9, 13, 17, 18])]
    _, _, expected_similarity = get_clip_similarity(queries, candidates)
    expected_ranks = ofa.strict_one_based_ranks(
        expected_similarity,
        labels,
        candidate_ids,
    )
    observed_similarity = None
    original_ranker = ofa.strict_one_based_ranks

    def capture_ranker(similarity, rank_labels, rank_candidate_ids):
        nonlocal observed_similarity
        observed_similarity = similarity
        return original_ranker(similarity, rank_labels, rank_candidate_ids)

    monkeypatch.setattr(ofa, 'strict_one_based_ranks', capture_ranker)
    observed_ranks = ofa.fixed_bank_ranks(
        queries,
        normalized_candidates,
        labels,
        candidate_ids,
    )

    assert observed_similarity is not None
    assert torch.equal(observed_similarity, expected_similarity)
    assert torch.equal(observed_ranks, expected_ranks)


def test_vowel_labels_match_black_willow_textgrid_inventory():
    vowels = set(ofa.PHONEME_GROUPS['vowels_all'])
    assert {'aw', 'ɒː', 'ɔj'} <= vowels
    assert not {'aʊ', 'ɔː', 'ɒj'} & vowels


def test_output_manifest_refuses_a_different_configuration(tmp_path):
    config = {'seed': 42, 'features': ['a']}
    config_id = ofa.analysis_configuration_id(config)
    ofa.prepare_output_directory(
        tmp_path,
        configuration_id=config_id,
        configuration=config,
    )
    ofa.prepare_output_directory(
        tmp_path,
        configuration_id=config_id,
        configuration=config,
    )

    with pytest.raises(ValueError, match='different analysis configuration'):
        ofa.prepare_output_directory(
            tmp_path,
            configuration_id=ofa.analysis_configuration_id({'seed': 7}),
            configuration={'seed': 7},
        )


def test_completed_result_requires_intact_artifacts(tmp_path):
    config_id = 'abc'
    ofa.atomic_write_json(
        tmp_path / 'stats.json',
        {'configuration_id': config_id, 'status': 'complete'},
    )
    ofa.atomic_write_json(
        tmp_path / 'COMPLETED.json',
        {'configuration_id': config_id, 'status': 'complete'},
    )

    assert (
        ofa.load_completed_result(
            tmp_path,
            configuration_id=config_id,
            save_ranks=False,
        )
        is None
    )
    assert not (tmp_path / 'COMPLETED.json').exists()
    assert list(tmp_path.glob('INVALID_COMPLETED.*.json'))


def test_rank_checkpoint_resumes_complete_and_resets_partial_rows(tmp_path):
    baseline = ofa.open_rank_memmap(tmp_path / 'baseline.npy', (2, 3))
    baseline[0] = [1, 2, 3]
    baseline[1, 0] = 2
    baseline.flush()
    assert ofa.baseline_checkpoint_complete(
        baseline,
        0,
        candidate_bank_size=3,
    )
    assert not ofa.baseline_checkpoint_complete(
        baseline,
        1,
        candidate_bank_size=3,
    )
    np.testing.assert_array_equal(baseline[1], [-1, -1, -1])

    absent = ofa.open_rank_memmap(tmp_path / 'absent.npy', (1, 3, 2))
    present = ofa.open_rank_memmap(tmp_path / 'present.npy', (1, 3, 2))
    eligible = np.array([True, False, True])
    absent[0, 0] = [1, 2]
    present[0, 0] = [2, 3]
    absent.flush()
    present.flush()
    assert not ofa.occlusion_checkpoint_complete(
        absent,
        present,
        0,
        eligible,
        candidate_bank_size=3,
    )
    assert np.all(absent == -1)
    assert np.all(present == -1)


def test_word_feature_coverage_rejects_missing_sound():
    frame = pd.DataFrame({'sound_fname': ['a.wav']})
    with pytest.raises(ValueError, match='b.wav'):
        ofa.validate_word_feature_sound_coverage(frame, ['a.wav', 'b.wav'])


def test_lexical_value_masks_use_only_unmanipulated_narrative_words():
    frame = pd.DataFrame(
        {
            'sound_fname': ['donor.wav'] * 5 + ['test.wav'] * 2,
            'insertion_kind': ['none'] * 4 + ['single', 'none', 'single'],
            'surprisal_bits': [1.0, 2.0, 3.0, 4.0, 100.0, 4.0, 100.0],
            'start_sample': [0, 2, 4, 6, 8, 0, 2],
            'end_sample': [1, 3, 5, 7, 9, 1, 3],
        }
    )
    spec = ofa.FeatureSpec(
        'surprisal',
        'word_value',
        column='surprisal_bits',
        high_is_present=True,
    )

    present, absent, metadata = ofa._word_value_masks(
        spec,
        frame,
        {'donor.wav': 12, 'test.wav': 12},
        {'donor.wav'},
        hop_samples=1,
    )

    assert metadata['lower_quartile'] == pytest.approx(1.75)
    assert metadata['upper_quartile'] == pytest.approx(3.25)
    assert metadata['word_scope'] == 'unmanipulated_narrative_words_only'
    assert present['test.wav'][0]
    assert not present['test.wav'][2]
    assert not absent['test.wav'][2]


def test_paired_metrics_apply_same_query_filter_to_every_arm():
    baseline = np.array([1, 100, 2])
    absent = np.array([[3, 5], [-1, -1], [8, 6]], dtype=float)
    present = np.array([[2, 2], [-1, -1], [4, 4]], dtype=float)
    eligible = np.array([True, False, True])

    metrics = ofa.paired_feature_metrics(
        baseline,
        absent,
        present,
        eligible,
    )

    # The discarded baseline rank 100 must not enter any feature-specific
    # original metric.
    assert metrics['n_queries'] == 2
    assert metrics['baseline_mean_rank'] == pytest.approx(1.5)
    assert metrics['f_to_absent_mean_rank'] == pytest.approx(5.5)
    assert metrics['f_to_present_mean_rank'] == pytest.approx(3.0)
    assert metrics['effect_rank_absent_minus_present'] == pytest.approx(2.5)
    assert metrics['damage_rank_absent_minus_baseline'] == pytest.approx(4.0)
    assert metrics['damage_rank_present_minus_baseline'] == pytest.approx(1.5)


def test_paired_metrics_refuse_missing_rank_in_eligible_query():
    with pytest.raises(ValueError, match='missing/non-finite'):
        ofa.paired_feature_metrics(
            np.array([1, 2]),
            np.array([[3.0], [np.nan]]),
            np.array([[2.0], [4.0]]),
            np.array([True, True]),
        )


def test_point_mask_padding_and_inner_taper_are_five_frames():
    point = np.zeros(9, dtype=bool)
    point[4] = True

    support = ofa.dilate_mask(point, radius=2)
    alpha = ofa.inner_raised_cosine_alpha(support, taper_radius=2)

    np.testing.assert_array_equal(
        support,
        np.array([0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=bool),
    )
    np.testing.assert_allclose(
        alpha,
        np.array([0, 0, 0.25, 0.75, 1.0, 0.75, 0.25, 0, 0]),
    )


def test_pair_donors_are_distinct_nonoverlapping_and_same_file():
    present = {
        'a.wav': np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=bool),
        'b.wav': np.zeros(8, dtype=bool),
    }
    absent = {
        'a.wav': np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=bool),
        'b.wav': np.ones(8, dtype=bool),
    }

    pairs, reason, detail = ofa.pair_donor_candidates(
        present,
        absent,
        length=2,
        n_pairs=2,
        rng=np.random.default_rng(4),
    )

    assert reason == detail == ''
    assert pairs is not None
    assert len(pairs) == 2
    assert all(pair.present.sound_fname == pair.absent.sound_fname for pair in pairs)
    assert len({pair.present for pair in pairs}) == 2
    assert len({pair.absent for pair in pairs}) == 2


def test_pair_donors_reports_specific_shortage_reason():
    pairs, reason, detail = ofa.pair_donor_candidates(
        present={'a.wav': np.array([1, 1, 0, 0], dtype=bool)},
        absent={'a.wav': np.array([0, 0, 1, 1], dtype=bool)},
        length=2,
        n_pairs=2,
        rng=np.random.default_rng(0),
    )

    assert pairs is None
    assert reason == 'insufficient_feature_present_donors'
    assert 'needed=2' in detail


def _toy_feature_state():
    spec = ofa.FeatureSpec('toy', 'phoneme')
    present = {
        # window 0 absent; window 1 full; window 2 has one two-frame component
        'test.wav': np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0], dtype=bool),
        # two present and two absent donor components of length two
        'donor.wav': np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool),
    }
    absent = {sound: ~mask for sound, mask in present.items()}
    return ofa.FeatureState(spec=spec, present=present, absent=absent)


def test_window_audit_records_absent_full_and_eligible_once_each():
    windows = pd.DataFrame(
        {
            'wav_index': [0, 1, 2],
            'sound_fname': ['test.wav'] * 3,
            'wav_start': [0, 4, 8],
            'wav_stop': [4, 8, 12],
        }
    )

    plans, records = ofa.build_window_plans(
        _toy_feature_state(),
        windows,
        {'donor.wav'},
        hop_samples=1,
        taper_frames=0,
        n_donor_pairs=2,
        seed=7,
    )
    summary = ofa.summarize_eligibility(records)

    assert set(plans) == {2}
    assert [record.reason for record in records] == [
        'feature_absent',
        'full_mask',
        '',
    ]
    assert summary['n_total_windows'] == 3
    assert summary['n_eligible_windows'] == 1
    assert summary['n_discarded_windows'] == 2
    assert summary['discard_reasons'] == {
        'feature_absent': 1,
        'full_mask': 1,
    }


def test_donor_timestamps_are_reproducible_and_participant_specific(monkeypatch):
    state = ofa.FeatureState(
        spec=ofa.FeatureSpec('toy', 'phoneme'),
        present={
            'test.wav': np.array([0, 0, 1, 1], dtype=bool),
            'donor.wav': np.tile(
                np.array([1, 1, 0, 0], dtype=bool),
                12,
            ),
        },
        absent={
            'test.wav': np.array([1, 1, 0, 0], dtype=bool),
            'donor.wav': np.tile(
                np.array([0, 0, 1, 1], dtype=bool),
                12,
            ),
        },
    )
    windows = pd.DataFrame(
        {
            'wav_index': [0],
            'sound_fname': ['test.wav'],
            'wav_start': [0],
            'wav_stop': [4],
        }
    )
    candidate_cache = {}
    base, _ = ofa.build_window_plans(
        state,
        windows,
        {'donor.wav'},
        hop_samples=1,
        taper_frames=0,
        n_donor_pairs=3,
        seed=123,
        donor_candidate_cache=candidate_cache,
    )
    assert candidate_cache

    def unexpected_rescan(*_args, **_kwargs):
        raise AssertionError('Participant redraw must reuse cached donor pools.')

    monkeypatch.setattr(
        ofa,
        'build_donor_candidate_pool',
        unexpected_rescan,
    )

    def draw(subject):
        plans = ofa.build_participant_feature_plans(
            subject=subject,
            base_feature_plans={'toy': base},
            feature_donor_candidate_caches={'toy': candidate_cache},
            feature_names=['toy'],
            n_donor_pairs=3,
            analysis_seed=123,
        )
        assert set(plans['toy']) == set(base)
        return plans['toy'][0].components[0].donor_pairs

    assert draw(1) == draw(1)
    assert draw(1) != draw(2)


def test_progress_totals_include_resumed_cells_but_only_pending_batches():
    complete = {
        (0, 'a'): True,
        (1, 'a'): False,
        (0, 'b'): True,
    }
    eligible = {'a': 3, 'b': 2}

    total, reused = ofa.occlusion_progress_totals(
        complete,
        eligible,
        n_donor_pairs=2,
    )
    pending_batches = ofa.pending_occlusion_forward_batches(
        complete,
        eligible,
        n_donor_pairs=2,
        batch_size=5,
    )

    assert (total, reused) == (32, 20)
    assert pending_batches == 3


def test_query_progress_callback_reports_each_completed_batch(monkeypatch):
    donor_pairs = tuple(
        ofa.DonorPair(
            absent=ofa.DonorInterval('donor.wav', replicate, replicate + 1),
            present=ofa.DonorInterval(
                'donor.wav',
                replicate + 2,
                replicate + 3,
            ),
        )
        for replicate in range(2)
    )
    plan = ofa.WindowPlan(
        wav_index=0,
        sound_fname='test.wav',
        sound_start_frame=0,
        support_fraction=1 / 3,
        weighted_fraction=1 / 3,
        components=(
            ofa.ComponentPlan(
                start=1,
                stop=2,
                alpha=np.ones(1),
                donor_pairs=donor_pairs,
            ),
        ),
    )

    def fake_rank_query_batches(
        _model,
        _queries,
        _subject_index,
        labels,
        _candidate_bank,
        _candidate_ids_device,
        _device,
        batch_size,
    ):
        assert len(labels) <= batch_size
        return np.ones(len(labels), dtype=np.int16)

    monkeypatch.setattr(ofa, 'rank_query_batches', fake_rank_query_batches)
    updates = []
    absent, present = ofa.evaluate_occlusion_queries(
        model=torch.nn.Identity(),
        targets=torch.zeros(2, 2, 3),
        plans={0: plan},
        donor_reader=lambda _interval: torch.ones(2, 1),
        n_donor_pairs=2,
        subject_index=0,
        candidate_bank=torch.ones(2, 1),
        candidate_ids=np.array([0, 1]),
        device=torch.device('cpu'),
        batch_size=3,
        progress_callback=updates.append,
    )

    assert updates == [3, 1]
    assert sum(updates) == 4
    np.testing.assert_array_equal(absent[0], [1, 1])
    np.testing.assert_array_equal(present[0], [1, 1])
    np.testing.assert_array_equal(absent[1], [-1, -1])
    np.testing.assert_array_equal(present[1], [-1, -1])


def test_component_plan_changes_all_channels_and_only_component():
    target = torch.arange(18, dtype=torch.float32).reshape(2, 9)
    pair = ofa.DonorPair(
        absent=ofa.DonorInterval('donor.wav', 0, 3),
        present=ofa.DonorInterval('donor.wav', 3, 6),
    )
    component = ofa.ComponentPlan(
        start=2,
        stop=5,
        alpha=np.array([0.25, 1.0, 0.25]),
        donor_pairs=(pair,),
    )
    plan = ofa.WindowPlan(
        wav_index=0,
        sound_fname='target.wav',
        sound_start_frame=0,
        support_fraction=1 / 3,
        weighted_fraction=1.5 / 9,
        components=(component,),
    )
    donor_values = {
        pair.absent: torch.zeros(2, 3),
        pair.present: torch.full((2, 3), 100.0),
    }

    absent, present = ofa.apply_component_plan(
        target,
        plan,
        replicate=0,
        donor_reader=donor_values.__getitem__,
    )

    torch.testing.assert_close(absent[:, :2], target[:, :2])
    torch.testing.assert_close(absent[:, 5:], target[:, 5:])
    torch.testing.assert_close(present[:, :2], target[:, :2])
    torch.testing.assert_close(present[:, 5:], target[:, 5:])
    torch.testing.assert_close(absent[:, 3], torch.zeros(2))
    torch.testing.assert_close(present[:, 3], torch.full((2,), 100.0))


def test_sound_origins_use_integer_dataframe_coordinates():
    frame = pd.DataFrame(
        {
            'story_id': [3, 3, 3, 3],
            'sound_fname': ['a.wav', 'a.wav', 'b.wav', 'b.wav'],
            'wav_start': [0, 16000, 0, 16000],
            'meg100_start': [123, 223, 900, 1000],
        }
    )

    origins = ofa.derive_sound_origins(
        frame,
        story_id=3,
        donor_sounds={'a.wav', 'b.wav'},
        meg_sr=100,
    )

    assert origins == {'a.wav': 123, 'b.wav': 900}


def test_aggregate_sessions_gives_each_participant_equal_weight():
    combo = pd.DataFrame(
        {
            'subject_id': [1, 1, 2],
            'session_id': [0, 1, 0],
            'effect': [0.0, 2.0, 10.0],
        }
    )
    original = ofa.SUB_SES_COMBOS
    try:
        ofa.SUB_SES_COMBOS = [(1, 0), (1, 1), (2, 0)]
        participant = ofa.aggregate_sessions_to_participants(combo, ['effect'])
    finally:
        ofa.SUB_SES_COMBOS = original

    assert participant['effect'].tolist() == [1.0, 10.0]


def test_signflip_uses_one_shared_subject_sign_across_features():
    effects = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )
    result = ofa.two_sided_signflip_max_abs_t(
        effects,
        n_permutations=128,
        seed=11,
    )

    # A perfectly scaled feature pair remains perfectly scaled before
    # studentization and receives the same t statistic in every sign flip.
    np.testing.assert_allclose(result['null_t'][:, 0], result['null_t'][:, 1])
    np.testing.assert_allclose(
        result['max_abs_t_null'],
        np.max(np.abs(result['null_t']), axis=1),
    )
    assert result['signs'].shape == (128, 4)
    assert np.all(result['max_t_fwer_p'] >= result['raw_p'])


def test_two_sided_signflip_is_sign_symmetric():
    effects = np.array(
        [
            [1.5, -0.4],
            [2.0, -1.1],
            [2.5, 0.2],
            [3.0, -0.8],
            [1.0, 0.5],
        ]
    )
    positive = ofa.two_sided_signflip_max_abs_t(
        effects,
        n_permutations=256,
        seed=19,
    )
    negative = ofa.two_sided_signflip_max_abs_t(
        -effects,
        n_permutations=256,
        seed=19,
    )

    np.testing.assert_allclose(positive['observed_abs_t'], negative['observed_abs_t'])
    np.testing.assert_allclose(positive['raw_p'], negative['raw_p'])
    np.testing.assert_allclose(positive['max_t_fwer_p'], negative['max_t_fwer_p'])
    np.testing.assert_allclose(positive['max_abs_t_null'], negative['max_abs_t_null'])
    np.testing.assert_allclose(positive['null_t'], -negative['null_t'])


def test_two_sided_signflip_treats_equal_magnitude_signs_equivalently():
    positive = np.array([[2.0], [2.0], [2.0], [2.0], [2.0], [2.0]])
    mixed = np.column_stack((positive[:, 0], -positive[:, 0]))
    result = ofa.two_sided_signflip_max_abs_t(
        mixed,
        n_permutations=200,
        seed=5,
    )

    np.testing.assert_allclose(result['observed_t'][0], -result['observed_t'][1])
    np.testing.assert_allclose(result['observed_abs_t'][0], result['observed_abs_t'][1])
    np.testing.assert_allclose(result['raw_p'][0], result['raw_p'][1])
    np.testing.assert_allclose(result['max_t_fwer_p'][0], result['max_t_fwer_p'][1])


def test_two_sided_signflip_can_call_a_large_negative_effect_significant():
    effects = np.full((27, 2), 0.05)
    effects[:, 0] = -8.0
    result = ofa.two_sided_signflip_max_abs_t(
        effects,
        n_permutations=2000,
        seed=3,
    )

    assert result['observed_t'][0] < 0
    assert result['raw_p'][0] < 0.01
    assert result['max_t_fwer_p'][0] < 0.05
    assert result['max_t_fwer_p'][0] >= result['raw_p'][0]
    np.testing.assert_array_equal(
        result['max_abs_t_null'],
        np.max(np.abs(result['null_t']), axis=1),
    )


def test_pairwise_jaccard_matches_known_masks_and_empty_union():
    masks = {
        'a': np.array([1, 1, 0, 0, 1], dtype=bool),
        'b': np.array([1, 0, 1, 0, 1], dtype=bool),
        'empty': np.array([0, 0, 0, 0, 0], dtype=bool),
        'also_empty': np.array([0, 0, 0, 0, 0], dtype=bool),
    }
    jaccard, containment = ofa.pairwise_jaccard(masks)

    assert jaccard.loc['a', 'b'] == pytest.approx(2 / 4)
    assert containment.loc['a', 'b'] == pytest.approx(2 / 3)
    assert containment.loc['b', 'a'] == pytest.approx(2 / 3)
    assert jaccard.loc['empty', 'also_empty'] == pytest.approx(1.0)
    assert np.isnan(containment.loc['empty', 'a'])
    assert containment.loc['a', 'empty'] == pytest.approx(0.0)


def test_jaccard_uses_concatenated_test_material_masks():
    present = {
        'test.wav': np.array([1, 0, 1], dtype=bool),
        'other.wav': np.array([1, 1], dtype=bool),
        'ignored.wav': np.array([1, 1, 1, 1], dtype=bool),
    }
    windows = pd.DataFrame(
        {'sound_fname': ['test.wav', 'other.wav', 'test.wav']}
    )
    mask = ofa.concatenate_test_present_mask(present, windows)
    np.testing.assert_array_equal(mask, np.array([1, 0, 1, 1, 1], dtype=bool))


def test_jaccard_heatmap_and_two_sided_metadata_smoke(tmp_path):
    masks = {
        'phoneme_stops_all': np.array([1, 1, 0, 0], dtype=bool),
        'phoneme_vowels_all': np.array([0, 1, 1, 0], dtype=bool),
        'word_onset': np.array([0, 0, 0, 1], dtype=bool),
    }
    jaccard, _containment = ofa.pairwise_jaccard(masks)
    png = tmp_path / 'feature_mask_jaccard.png'
    ofa.plot_jaccard_heatmap(jaccard, png)
    assert png.exists() and png.stat().st_size > 0

    effects = np.array(
        [
            [1.0, -1.0],
            [1.2, -1.1],
            [0.8, -0.9],
            [1.1, -1.2],
        ]
    )
    result = ofa.two_sided_signflip_max_abs_t(
        effects, n_permutations=64, seed=1
    )
    assert np.all(result['max_abs_t_null'] == np.max(np.abs(result['null_t']), axis=1))
    assert np.all(result['max_t_fwer_p'] >= result['raw_p'])


def test_control_saturation_overrides_a_significant_positive_effect():
    conclusion, threshold, saturated = ofa.classify_feature_result(
        mean_effect=5.0,
        corrected_p=0.001,
        feature_present_control_mean_rank=503.0,
        candidate_bank_size=1005,
    )

    assert threshold == 503.0
    assert saturated
    assert conclusion == 'control_saturated'


@pytest.mark.parametrize(
    ('mean_effect', 'corrected_p', 'expected'),
    [
        (5.0, 0.001, 'significant_positive_effect'),
        (5.0, 0.2, 'inconclusive'),
        (-5.0, 0.001, 'significant_negative_effect'),
    ],
)
def test_feature_conclusion_is_limited_and_directional(
    mean_effect,
    corrected_p,
    expected,
):
    conclusion, _, saturated = ofa.classify_feature_result(
        mean_effect=mean_effect,
        corrected_p=corrected_p,
        feature_present_control_mean_rank=10.0,
        candidate_bank_size=1005,
    )

    assert not saturated
    assert conclusion == expected
