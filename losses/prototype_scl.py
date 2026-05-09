"""Prototype-level supervised contrastive loss used by FedProto."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def prototype_scl_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    global_prototypes: torch.Tensor | None,
    tau_s: float = 0.2,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy over similarities to global class prototypes.

    Args:
        features: Local features with shape [B, D].
        labels: Ground-truth labels with shape [B].
        global_prototypes: Global prototypes with shape [K, D].
        tau_s: Temperature for similarity logits.
        valid_mask: Optional boolean mask for classes with valid prototypes.
    """
    if global_prototypes is None:
        return features.new_tensor(0.0)

    if global_prototypes.numel() == 0:
        return features.new_tensor(0.0)

    prototypes = global_prototypes.to(device=features.device, dtype=features.dtype)
    if valid_mask is None:
        valid_mask = prototypes.norm(dim=1) > 0
    else:
        valid_mask = valid_mask.to(device=features.device, dtype=torch.bool)

    target_is_valid = valid_mask[labels]
    if not torch.any(target_is_valid):
        return features.new_tensor(0.0)

    norm_features = F.normalize(features[target_is_valid], dim=1)
    norm_prototypes = F.normalize(prototypes, dim=1)
    logits = norm_features @ norm_prototypes.T / tau_s
    logits[:, ~valid_mask] = -1.0e9

    return F.cross_entropy(logits, labels[target_is_valid])


def prototype_mse_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    global_prototypes: torch.Tensor | None,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """FedProto-style MSE alignment to the global prototype of each label."""
    if global_prototypes is None:
        return features.new_tensor(0.0)

    if global_prototypes.numel() == 0:
        return features.new_tensor(0.0)

    prototypes = global_prototypes.to(device=features.device, dtype=features.dtype)
    if valid_mask is None:
        valid_mask = prototypes.norm(dim=1) > 0
    else:
        valid_mask = valid_mask.to(device=features.device, dtype=torch.bool)

    target_is_valid = valid_mask[labels]
    if not torch.any(target_is_valid):
        return features.new_tensor(0.0)

    targets = prototypes[labels[target_is_valid]].detach()
    return F.mse_loss(features[target_is_valid], targets)
