"""MobileNetV2 client model adapted for CIFAR-10."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import mobilenet_v2


class MobileNetV2CIFAR(nn.Module):
    """MobileNetV2 with a CIFAR-friendly stem and FedCausal projector."""

    def __init__(self, num_classes: int = 10, feature_dim: int = 128) -> None:
        super().__init__()
        net = mobilenet_v2(weights=None)
        net.features[0][0] = nn.Conv2d(
            3,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        in_features = net.classifier[1].in_features
        self.backbone = nn.Sequential(
            net.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.projector = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return projected features with shape [B, D]."""
        features = self.backbone(x)
        return self.projector(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return classification logits."""
        features = self.extract_features(x)
        return self.classifier(features)
