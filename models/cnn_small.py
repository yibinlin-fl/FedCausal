"""Small CNN client model for CIFAR-10."""

from __future__ import annotations

import torch
from torch import nn


class CNNSmall(nn.Module):
    """A lightweight CNN with a shared FedCausal-style interface."""

    def __init__(self, num_classes: int = 10, feature_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.05),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.10),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.BatchNorm1d(feature_dim),
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
