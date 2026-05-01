"""ResNet client models adapted for CIFAR-10."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class ResNet18CIFAR(nn.Module):
    """ResNet18 with CIFAR-sized stem and FedCausal projector."""

    def __init__(self, num_classes: int = 10, feature_dim: int = 128) -> None:
        super().__init__()
        net = resnet18(weights=None)
        net.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        net.maxpool = nn.Identity()

        in_features = net.fc.in_features
        net.fc = nn.Identity()

        self.backbone = net
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
