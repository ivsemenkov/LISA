"""Losses and metrics for LISA training."""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_clip_similarity(
    brainwave_embeddings: torch.Tensor,
    audio_embeddings: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize embeddings and compute CLIP-style similarity.

    Args:
        brainwave_embeddings: Tensor shaped (B, F) or (B, F, T).
        audio_embeddings: Tensor shaped (N, F) or (N, F, T).

    Returns:
        brainwave_embeddings: Normalized embeddings.
        audio_embeddings: Normalized embeddings.
        similarity: Similarity matrix (B, N).
    """
    if brainwave_embeddings.shape[1:] != audio_embeddings.shape[1:]:
        raise ValueError(
            'Brainwave and audio embeddings must have matching non-batch shapes. '
            f'Got brainwave={brainwave_embeddings.shape}, audio={audio_embeddings.shape}.'
        )
    if len(audio_embeddings.shape) == 3:
        brainwave_embeddings = F.normalize(brainwave_embeddings, dim=(-2, -1))
        audio_embeddings = F.normalize(audio_embeddings, dim=(-2, -1))
        similarity = torch.einsum(
            'Bef, bef -> Bb', brainwave_embeddings, audio_embeddings
        )
    elif len(audio_embeddings.shape) == 2:
        brainwave_embeddings = F.normalize(brainwave_embeddings, dim=(-1))
        audio_embeddings = F.normalize(audio_embeddings, dim=(-1))
        similarity = torch.einsum(
            'Bf, bf -> Bb', brainwave_embeddings, audio_embeddings
        )
    else:
        raise NotImplementedError(
            f'Expected ndim for brainwave_embeddings and audio_embeddings is 2 or 3. Received: brainwave_embeddings '
            f'{brainwave_embeddings.size()}, audio_embeddings {audio_embeddings.size()}'
        )
    return brainwave_embeddings, audio_embeddings, similarity


class CLIPLoss(nn.Module):
    """CLIP-style contrastive loss with learnable temperature."""

    def __init__(
        self,
        clip_temperature: float,
    ):
        """Initialize CLIP loss.

        Args:
            clip_temperature: Initial temperature value (learnable parameter).
        """
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor(math.log(clip_temperature), dtype=torch.float32)
        )

    def forward(
        self,
        brainwave_embeddings: torch.Tensor,
        audio_embeddings: torch.Tensor,
        positive_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute CLIP loss with and without temperature scaling.

        Returns:
            loss: Loss without temperature scaling.
            loss_temperature: Loss with temperature scaling (used for training).
        """
        if positive_ids is None:
            raise ValueError('positive_ids is required for CLIPLoss.')
        batch_size = brainwave_embeddings.size(0)
        if positive_ids.ndim != 1 or positive_ids.numel() != batch_size:
            raise ValueError(f'positive_ids must be 1D with length {batch_size}.')
        if audio_embeddings.size(0) != batch_size:
            raise ValueError(
                'CLIPLoss expects paired batch embeddings with the same batch size.'
            )

        positive_ids = positive_ids.to(audio_embeddings.device)
        unique_positive_ids, labels = torch.unique(
            positive_ids, sorted=True, return_inverse=True
        )
        if unique_positive_ids.numel() < 2:
            raise ValueError(
                'CLIPLoss requires at least two unique positive_ids in a batch.'
            )
        audio_indices = []
        for positive_id in unique_positive_ids:
            sample_indices = torch.where(positive_ids == positive_id)[0]
            audio_indices.append(sample_indices[0])
        audio_embeddings = audio_embeddings[torch.stack(audio_indices)]

        brainwave_embeddings, audio_embeddings, similarity = get_clip_similarity(
            brainwave_embeddings=brainwave_embeddings, audio_embeddings=audio_embeddings
        )
        similarity_temperature = similarity / torch.exp(self.temperature)
        labels = labels.to(similarity.device)
        loss = F.cross_entropy(similarity, labels)
        loss_temperature = F.cross_entropy(similarity_temperature, labels)
        return loss, loss_temperature


def retrieval_topk(
    brainwave_embeddings: torch.Tensor,
    audio_embeddings: torch.Tensor,
    labels: torch.Tensor,
    candidate_ids: torch.Tensor,
    top_k: int,
) -> dict[str, torch.Tensor]:
    """Compute retrieval top-K flags and optional per-row details."""
    if candidate_ids is None:
        raise ValueError('candidate_ids is required for retrieval metrics.')
    if top_k <= 0:
        raise ValueError(f'top_k must be positive, got {top_k}.')
    if (
        labels.ndim != 1
        or candidate_ids.ndim != 1
        or labels.numel() != brainwave_embeddings.size(0)
        or candidate_ids.numel() != audio_embeddings.size(0)
    ):
        raise ValueError(
            'labels and candidate_ids must be 1D and match the embedding counts.'
        )
    if torch.unique(candidate_ids).numel() != candidate_ids.numel():
        raise ValueError('candidate_ids must be unique.')

    brainwave_embeddings, audio_embeddings, similarity = get_clip_similarity(
        brainwave_embeddings=brainwave_embeddings, audio_embeddings=audio_embeddings
    )
    candidate_ids = candidate_ids.to(similarity.device)
    labels = labels.to(similarity.device)

    label_found = torch.isin(labels, candidate_ids)
    if not label_found.all():
        missing = labels[~label_found].detach().cpu().tolist()
        raise ValueError(
            'Every label must be present in candidate_ids. '
            f'Missing labels: {missing[:10]}.'
        )
    if top_k > similarity.shape[-1]:
        raise ValueError(
            f'top_k={top_k} is larger than candidate count {similarity.shape[-1]}.'
        )

    top_k_result = torch.topk(
        similarity,
        top_k,
        dim=-1,
        largest=True,
        sorted=True,
    )
    top_k_wav_indices = candidate_ids[top_k_result.indices]
    top_k_similarities = top_k_result.values

    true_wav_matches = top_k_wav_indices == labels[:, None]
    match_count = true_wav_matches.sum(dim=1)
    if (match_count > 1).any():
        raise ValueError(
            'Each label must match at most one top-K candidate. '
            'Check candidate_ids uniqueness.'
        )
    is_in_topk = true_wav_matches.any(dim=1)
    is_in_top1 = true_wav_matches[:, 0]

    rank_of_true = torch.full(
        (labels.shape[0],),
        -1,
        dtype=torch.long,
        device=similarity.device,
    )
    similarity_to_true = torch.full(
        (labels.shape[0],),
        torch.nan,
        dtype=similarity.dtype,
        device=similarity.device,
    )
    matched_rows, matched_ranks = true_wav_matches.nonzero(as_tuple=True)
    rank_of_true[matched_rows] = matched_ranks
    similarity_to_true[matched_rows] = top_k_similarities[
        matched_rows, matched_ranks
    ]

    return {
        'rank_of_true': rank_of_true,
        'similarity_to_true': similarity_to_true,
        'top_k_wav_indices': top_k_wav_indices,
        'top_k_similarities': top_k_similarities,
        'is_in_topk': is_in_topk,
        'is_in_top1': is_in_top1,
    }


def metrics(
    brainwave_embeddings: torch.Tensor,
    audio_embeddings: torch.Tensor,
    labels: torch.Tensor,
    candidate_ids: torch.Tensor,
    topk: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute top-K and top-1 retrieval accuracy flags.

    Returns:
        is_in_topk: Bool tensor (B,) for top-K correctness.
        is_in_top1: Bool tensor (B,) for top-1 correctness.
    """
    ranking = retrieval_topk(
        brainwave_embeddings,
        audio_embeddings,
        labels,
        candidate_ids=candidate_ids,
        top_k=topk,
    )
    return ranking['is_in_topk'], ranking['is_in_top1']
