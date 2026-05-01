"""Reproducibility helpers."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Seed Python, NumPy, and PyTorch for Kaggle-friendly experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    return seed


def seed_worker(worker_id: int, base_seed: Optional[int] = None) -> None:
    """Seed a DataLoader worker."""
    if base_seed is None:
        worker_seed = torch.initial_seed() % 2**32
    else:
        worker_seed = (base_seed + worker_id) % 2**32

    np.random.seed(worker_seed)
    random.seed(worker_seed)
