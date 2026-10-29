"""Pytest configuration and fixtures for Snake DQN tests."""

import random
import numpy as np
import torch
import pytest


@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset global random state before each test for reproducibility."""
    # Set fixed seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    yield


@pytest.fixture(autouse=True)
def reset_torch_deterministic():
    """Ensure deterministic behavior for PyTorch operations."""
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_benchmark = torch.backends.cudnn.benchmark if torch.cuda.is_available() else None

    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False

    yield

    torch.use_deterministic_algorithms(old_deterministic)
    if torch.cuda.is_available() and old_benchmark is not None:
        torch.backends.cudnn.benchmark = old_benchmark