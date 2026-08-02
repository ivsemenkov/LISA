import pytest
import torch
import torch.nn.functional as F

from lisa.training.criteria import (
    CLIPLoss,
    get_clip_similarity,
    metrics,
    retrieval_topk,
)


def test_clip_loss_unique_ids_matches_diagonal_cross_entropy():
    brain = torch.randn(4, 5)
    audio = torch.randn(4, 5)
    ids = torch.arange(4)
    criterion = CLIPLoss(clip_temperature=1.0)

    similarity = get_clip_similarity(brain, audio)[2]
    expected = F.cross_entropy(similarity, torch.arange(4))
    actual, actual_temperature = criterion(brain, audio, positive_ids=ids)

    assert torch.allclose(actual, expected)
    assert torch.allclose(actual_temperature, expected)


def test_clip_loss_duplicate_positive_ids_use_unique_audio_candidates():
    brain = torch.randn(3, 5)
    audio_unique = torch.randn(2, 5)
    audio = audio_unique[[0, 0, 1]]
    ids = torch.tensor([7, 7, 8])
    criterion = CLIPLoss(clip_temperature=1.0)

    similarity = get_clip_similarity(brain, audio_unique)[2]
    expected = F.cross_entropy(similarity, torch.tensor([0, 0, 1]))
    actual, _ = criterion(brain, audio, positive_ids=ids)

    assert torch.allclose(actual, expected)


def test_clip_loss_rejects_batches_without_negatives():
    criterion = CLIPLoss(clip_temperature=1.0)

    with pytest.raises(ValueError, match='at least two unique'):
        criterion(
            torch.randn(3, 5),
            torch.randn(3, 5),
            positive_ids=torch.tensor([7, 7, 7]),
        )


def test_metrics_full_candidate_bank_matches_index_labels():
    audio = torch.eye(4)
    brain = audio.clone()
    labels = torch.arange(4)
    candidate_ids = torch.arange(4)

    top10, top1 = metrics(brain, audio, labels, candidate_ids=candidate_ids, topk=3)

    assert top10.tolist() == [True, True, True, True]
    assert top1.tolist() == [True, True, True, True]


def test_metrics_subset_candidate_bank_uses_candidate_ids():
    audio = torch.eye(4)
    brain = audio[[1, 3]]
    labels = torch.tensor([20, 40])
    candidate_ids = torch.tensor([10, 20, 30, 40])

    top10, top1 = metrics(brain, audio, labels, candidate_ids=candidate_ids, topk=3)

    assert top10.tolist() == [True, True]
    assert top1.tolist() == [True, True]


def test_retrieval_topk_matches_metrics_and_sorted_topk():
    angles = torch.arange(12, dtype=torch.float32) * 0.1
    audio = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    brain = audio[[0, 11]]
    candidate_ids = torch.arange(100, 112)
    labels = torch.tensor([102, 110])

    ranking = retrieval_topk(
        brain,
        audio,
        labels,
        candidate_ids=candidate_ids,
        top_k=3,
    )
    top3, top1 = metrics(brain, audio, labels, candidate_ids=candidate_ids, topk=3)
    top2, _ = metrics(brain, audio, labels, candidate_ids=candidate_ids, topk=2)
    ranking_top2 = retrieval_topk(
        brain,
        audio,
        labels,
        candidate_ids=candidate_ids,
        top_k=2,
    )

    assert ranking['top_k_wav_indices'].tolist() == [[100, 101, 102], [111, 110, 109]]
    assert ranking['rank_of_true'].tolist() == [2, 1]
    assert ranking['is_in_topk'].tolist() == [True, True]
    assert ranking_top2['rank_of_true'].tolist() == [-1, 1]
    assert top3.tolist() == [True, True]
    assert top2.tolist() == [False, True]
    assert top1.tolist() == [False, False]


def test_metrics_fail_fast_for_bad_candidate_ids():
    audio = torch.eye(3)
    brain = audio[:2]

    with pytest.raises(ValueError, match='must be unique'):
        metrics(
            brain,
            audio,
            torch.tensor([0, 1]),
            candidate_ids=torch.tensor([0, 1, 1]),
        )

    with pytest.raises(ValueError, match='Every label must be present'):
        metrics(
            brain,
            audio,
            torch.tensor([0, 99]),
            candidate_ids=torch.tensor([0, 1, 2]),
        )


def test_metrics_fail_fast_with_fewer_than_10_candidates():
    audio = torch.eye(3)
    brain = audio[:2]

    with pytest.raises(ValueError, match='larger than candidate count'):
        metrics(
            brain,
            audio,
            torch.tensor([0, 1]),
            candidate_ids=torch.tensor([0, 1, 2]),
        )
