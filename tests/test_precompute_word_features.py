"""Tests for GPT-2 word-feature precomputation."""

from types import SimpleNamespace

import numpy as np
import torch

from lisa.cli.precompute_word_features import score_words


class _ToyTokenizer:
    bos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [1] if text.strip() == 'first' else [2]


class _UniformLanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        logits = torch.zeros((*input_ids.shape, 4), device=input_ids.device)
        return SimpleNamespace(logits=logits)


def test_score_words_scores_first_token_from_document_boundary():
    surprisal, entropy = score_words(
        ['first', 'second'],
        _ToyTokenizer(),
        _UniformLanguageModel(),
        max_len=4,
        stride=2,
    )

    assert np.allclose(surprisal, [2.0, 2.0])
    assert np.allclose(entropy, [2.0, 2.0])
