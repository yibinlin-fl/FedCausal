"""Learnable FFT frequency mask for FedCausal."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class LearnableFrequencyMask(nn.Module):
    """A learnable causal frequency mask shared across CIFAR-sized inputs."""

    def __init__(
        self,
        channels: int = 3,
        height: int = 32,
        width: int = 32,
        mask_init: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 < mask_init < 1.0:
            raise ValueError("mask_init must be in (0, 1).")

        init_logit = torch.logit(torch.tensor(float(mask_init)))
        self.mask_param = nn.Parameter(
            torch.full((1, channels, height, width), init_logit.item())
        )

    def get_mask(self) -> torch.Tensor:
        """Return the bounded mask M in [0, 1]."""
        return torch.sigmoid(self.mask_param)

    @torch.no_grad()
    def set_mask(self, mask: torch.Tensor) -> None:
        """Synchronize this learnable mask from a bounded mask tensor."""
        eps = torch.finfo(self.mask_param.dtype).eps
        mask = mask.to(device=self.mask_param.device, dtype=self.mask_param.dtype)
        mask = mask.clamp(min=eps, max=1.0 - eps)
        self.mask_param.copy_(torch.logit(mask))

    def apply_causal_filter(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply causal frequency filtering.

        Args:
            x: Input tensor with shape [B, C, H, W].

        Returns:
            x_c: Inverse-FFT causal image.
            Z: Full complex spectrum.
            Z_c: Masked causal complex spectrum.
            M: Bounded mask.
        """
        z = torch.fft.fft2(x, dim=(-2, -1))
        mask = self.get_mask().to(dtype=x.dtype)
        z_c = mask * z
        x_c = torch.fft.ifft2(z_c, dim=(-2, -1)).real
        return x_c, z, z_c, mask

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Alias for apply_causal_filter."""
        return self.apply_causal_filter(x)

    @torch.no_grad()
    def stats(self) -> dict[str, float]:
        """Return simple mask statistics for logging."""
        mask = self.get_mask()
        return {
            "mask_mean": mask.mean().item(),
            "mask_std": mask.std(unbiased=False).item(),
        }
