"""Frequency-domain counterfactual interventions for FedCausal."""

from __future__ import annotations

from typing import Tuple

import torch

from methods.frequency_mask import LearnableFrequencyMask


def _sample_donors(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a different-label donor for every item in the batch."""
    batch_size = labels.size(0)
    donor_indices = torch.arange(batch_size, device=labels.device)
    valid = torch.zeros(batch_size, dtype=torch.bool, device=labels.device)

    for idx in range(batch_size):
        candidates = torch.nonzero(labels != labels[idx], as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue

        choice = torch.randint(
            low=0,
            high=candidates.numel(),
            size=(1,),
            device=labels.device,
        )
        donor_indices[idx] = candidates[choice.item()]
        valid[idx] = True

    return donor_indices, valid


def build_counterfactual_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    frequency_mask: LearnableFrequencyMask,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build causal and counterfactual images with spurious frequency swap.

    For sample A, the counterfactual spectrum is
    ``Z_cf,A = Z_c,A + Z_s,B`` where B is a random in-batch sample with a
    different label. Samples without a valid donor are marked invalid.

    Args:
        x: Input images with shape [B, C, H, W].
        y: Labels with shape [B].
        frequency_mask: Learnable FFT mask module.

    Returns:
        x_c: Causal-filtered images.
        x_cf: Counterfactual images.
        valid: Boolean mask indicating samples with a different-label donor.
        mask: Current bounded frequency mask.
    """
    z = torch.fft.fft2(x, dim=(-2, -1))
    mask = frequency_mask.get_mask().to(device=x.device, dtype=x.dtype)
    z_c = mask * z
    z_s = (1.0 - mask) * z

    donor_indices, valid = _sample_donors(y)
    z_cf = z_c + z_s[donor_indices]

    x_c = torch.fft.ifft2(z_c, dim=(-2, -1)).real
    x_cf = torch.fft.ifft2(z_cf, dim=(-2, -1)).real
    return x_c, x_cf, valid, mask
