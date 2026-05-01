"""Client data partitioning utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def get_dataset_targets(dataset: object) -> np.ndarray:
    """Return labels from a torchvision-style dataset or a Subset."""
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        parent_targets = get_dataset_targets(dataset.dataset)
        return parent_targets[np.asarray(dataset.indices)]

    if hasattr(dataset, "targets"):
        return np.asarray(getattr(dataset, "targets"))

    if hasattr(dataset, "labels"):
        return np.asarray(getattr(dataset, "labels"))

    raise AttributeError("Dataset must expose `targets` or `labels`.")


def dirichlet_partition(
    dataset: object,
    num_clients: int,
    alpha: float,
    num_classes: int,
    seed: int,
) -> Dict[int, List[int]]:
    """Partition a labeled dataset into Non-IID client splits.

    The sampler retries until every client receives at least a small number of
    examples. This keeps small Kaggle MVP runs from failing because of an empty
    DataLoader while still supporting concentrated alpha values such as 0.1.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    targets = get_dataset_targets(dataset)
    min_required = min(10, max(1, len(targets) // max(num_clients * 20, 1)))
    max_attempts = 1000

    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt)
        client_indices = {client_id: [] for client_id in range(num_clients)}

        for class_id in range(num_classes):
            class_indices = np.where(targets == class_id)[0]
            rng.shuffle(class_indices)

            proportions = rng.dirichlet(np.full(num_clients, alpha))
            split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
            class_splits = np.split(class_indices, split_points)

            for client_id, split in enumerate(class_splits):
                client_indices[client_id].extend(split.tolist())

        client_sizes = [len(indices) for indices in client_indices.values()]
        if min(client_sizes) >= min_required:
            for indices in client_indices.values():
                rng.shuffle(indices)
            return client_indices

    raise RuntimeError(
        "Failed to create a Dirichlet partition with non-empty clients. "
        f"Try a larger alpha or fewer clients. Last minimum size: {min(client_sizes)}"
    )


def compute_client_label_distribution(
    dataset: object,
    client_indices: Dict[int, Sequence[int]],
    num_classes: int,
) -> Dict[int, List[int]]:
    """Count labels for every client."""
    targets = get_dataset_targets(dataset)
    distribution: Dict[int, List[int]] = {}

    for client_id, indices in client_indices.items():
        labels = targets[np.asarray(indices, dtype=np.int64)]
        counts = np.bincount(labels, minlength=num_classes)
        distribution[client_id] = counts.astype(int).tolist()

    return distribution


def print_client_label_distribution(
    distribution: Dict[int, Sequence[int]],
) -> None:
    """Print client sample counts and class histograms."""
    for client_id in sorted(distribution):
        counts = list(distribution[client_id])
        total = int(sum(counts))
        print(f"Client {client_id:02d}: num_samples={total}, label_distribution={counts}")


def save_client_label_distribution(
    distribution: Dict[int, Sequence[int]],
    csv_path: str | Path,
) -> Path:
    """Save client label histograms as a CSV file."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    num_classes = len(next(iter(distribution.values()))) if distribution else 0
    header = ["client_id", "num_samples"] + [f"class_{i}" for i in range(num_classes)]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for client_id in sorted(distribution):
            counts = list(distribution[client_id])
            writer.writerow([client_id, int(sum(counts)), *counts])

    return csv_path
