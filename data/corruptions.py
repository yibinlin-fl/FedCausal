"""Lightweight corruption transforms for corrupted-client training."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


SUPPORTED_TRAIN_CORRUPTIONS = {
    "gaussian_noise",
    "shot_noise",
    "motion_blur",
    "defocus_blur",
    "fog",
    "jpeg_compression",
    "pixelate",
}


def _severity_scale(severity: int) -> float:
    severity = int(severity)
    if severity not in {1, 2, 3, 4, 5}:
        raise ValueError("severity must be one of {1, 2, 3, 4, 5}.")
    return severity / 5.0


def _clamp_like_normalized(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-3.0, 3.0)


def gaussian_noise(x: torch.Tensor, severity: int) -> torch.Tensor:
    scale = 0.10 + 0.25 * _severity_scale(severity)
    return _clamp_like_normalized(x + torch.randn_like(x) * scale)


def shot_noise(x: torch.Tensor, severity: int) -> torch.Tensor:
    scale = 0.06 + 0.18 * _severity_scale(severity)
    noise = torch.randn_like(x) * torch.sqrt(x.abs() + 0.05) * scale
    return _clamp_like_normalized(x + noise)


def _depthwise_conv2d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    channels = x.size(0)
    kernel = kernel.to(device=x.device, dtype=x.dtype)
    kernel = kernel.view(1, 1, *kernel.shape).repeat(channels, 1, 1, 1)
    y = F.conv2d(x.unsqueeze(0), kernel, padding=kernel.shape[-1] // 2, groups=channels)
    return y.squeeze(0)


def motion_blur(x: torch.Tensor, severity: int) -> torch.Tensor:
    size = 3 if severity <= 2 else 5 if severity <= 4 else 7
    kernel = torch.zeros(size, size)
    kernel[size // 2, :] = 1.0 / size
    return _depthwise_conv2d(x, kernel)


def defocus_blur(x: torch.Tensor, severity: int) -> torch.Tensor:
    size = 3 if severity <= 2 else 5 if severity <= 4 else 7
    kernel = torch.ones(size, size) / float(size * size)
    return _depthwise_conv2d(x, kernel)


def fog(x: torch.Tensor, severity: int) -> torch.Tensor:
    strength = 0.10 + 0.25 * _severity_scale(severity)
    low_freq = F.avg_pool2d(x.unsqueeze(0), kernel_size=9, stride=1, padding=4).squeeze(0)
    return _clamp_like_normalized(x * (1.0 - strength) + low_freq * strength + strength * 0.5)


def jpeg_compression(x: torch.Tensor, severity: int) -> torch.Tensor:
    levels = {1: 64, 2: 48, 3: 32, 4: 24, 5: 16}
    level = float(levels[int(severity)])
    return torch.round(x * level) / level


def pixelate(x: torch.Tensor, severity: int) -> torch.Tensor:
    size = {1: 28, 2: 24, 3: 20, 4: 16, 5: 12}[int(severity)]
    y = F.interpolate(
        x.unsqueeze(0),
        size=(size, size),
        mode="nearest",
    )
    y = F.interpolate(y, size=x.shape[-2:], mode="nearest")
    return y.squeeze(0)


def apply_corruption(x: torch.Tensor, corruption_type: str, severity: int) -> torch.Tensor:
    """Apply a supported lightweight tensor corruption."""
    corruption_type = corruption_type.lower()
    if corruption_type not in SUPPORTED_TRAIN_CORRUPTIONS:
        available = ", ".join(sorted(SUPPORTED_TRAIN_CORRUPTIONS))
        raise ValueError(f"Unsupported corruption_type={corruption_type!r}. Available: {available}")

    return globals()[corruption_type](x, severity)


class CorruptedDataset(torch.utils.data.Dataset):
    """Dataset wrapper that corrupts samples from selected training clients."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        corruption_type: str,
        severity: int,
    ) -> None:
        self.dataset = dataset
        self.corruption_type = corruption_type
        self.severity = severity

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        image = apply_corruption(image, self.corruption_type, self.severity)
        return image, label


def build_corruption_transform(corruption_type: str, severity: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a callable tensor corruption transform."""
    return lambda x: apply_corruption(x, corruption_type, severity)
