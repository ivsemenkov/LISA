"""Determinism utilities for reproducible runs."""

import os
import random
import warnings

import numpy as np
import torch


def fix_seed(seed: int | None, torch_deterministic: bool = True) -> None:
    """Set RNG seeds for Python, NumPy, and Torch.

    When torch_deterministic is True, also enables deterministic PyTorch ops and
    CuDNN. References: PyTorch Reproducibility
    (https://pytorch.org/docs/stable/notes/randomness.html). That doc states:
    (1) torch.use_deterministic_algorithms(True) and torch.backends.cudnn.deterministic=True
    are separate (either can make CUDA conv deterministic; we set both when requested).
    (2) cudnn.benchmark=False is in the same doc under "CUDA convolution benchmarking".
    torch.use_deterministic_algorithms() does not set cudnn.deterministic/benchmark itself.

    Performance note: torch_deterministic=True can add substantial overhead, especially
    with many small conv layers (e.g. temporal blocks). Deterministic mode disables
    fast non-deterministic cuDNN algorithms and auto-tuning (cudnn.benchmark=False).
    Use torch_deterministic=False for faster training when exact bit-reproducibility
    is not required.

    Args:
        seed: Seed value or None to skip seeding (deterministic flags still apply if torch_deterministic).
        torch_deterministic: If True, enable deterministic algorithms in Torch and CuDNN.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    elif torch_deterministic:
        warnings.warn(
            'torch_deterministic is True but seed is None; low-level ops will be '
            'deterministic while RNG (Python, NumPy, PyTorch) remains non-reproducible. '
            'Pass a seed for full reproducibility.',
            UserWarning,
            stacklevel=2,
        )

    if torch_deterministic:
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
