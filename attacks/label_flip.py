"""Label flipping utilities for malicious clients."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, List

import torch


def select_malicious_clients(
    num_clients: int,
    malicious_ratio: float,
    seed: int,
) -> List[int]:
    """Select a fixed malicious client set for an experiment."""
    malicious_ratio = max(0.0, min(1.0, float(malicious_ratio)))
    num_malicious = int(round(num_clients * malicious_ratio))
    if num_malicious <= 0:
        return []

    rng = random.Random(seed)
    return sorted(rng.sample(range(num_clients), num_malicious))


def save_malicious_client_ids(
    malicious_client_ids: Iterable[int],
    path: str | Path,
    attack_type: str,
    malicious_ratio: float,
    scale_factor: float | int | None = None,
) -> Path:
    """Save malicious client metadata as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "attack_type": attack_type,
        "malicious_ratio": float(malicious_ratio),
        "scale_factor": scale_factor,
        "malicious_client_ids": list(malicious_client_ids),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def flip_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Apply y -> (y + 1) mod K."""
    return (labels + 1) % int(num_classes)
