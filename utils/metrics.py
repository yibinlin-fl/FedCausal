"""Metric helpers."""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Return batch accuracy as a Python float."""
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device | str,
) -> Dict[str, float]:
    """Evaluate a classifier on a dataloader."""
    model.eval()
    total = 0
    correct = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        total += labels.numel()
        correct += (preds == labels).sum().item()

    acc = correct / total if total > 0 else 0.0
    return {"accuracy": acc, "num_samples": float(total)}
