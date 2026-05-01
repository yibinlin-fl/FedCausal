"""Model definitions and factory for FedCausal clients."""

from __future__ import annotations

from torch import nn

from models.cnn_small import CNNSmall
from models.mobilenet_cifar import MobileNetV2CIFAR
from models.resnet_cifar import ResNet18CIFAR


MODEL_REGISTRY = {
    "cnn_small": CNNSmall,
    "cnn-small": CNNSmall,
    "resnet18": ResNet18CIFAR,
    "resnet18_cifar": ResNet18CIFAR,
    "mobilenetv2": MobileNetV2CIFAR,
    "mobilenet_v2": MobileNetV2CIFAR,
    "mobilenetv2_cifar": MobileNetV2CIFAR,
}


def build_model(
    model_name: str,
    num_classes: int = 10,
    feature_dim: int = 128,
) -> nn.Module:
    """Build a heterogeneous client model by name."""
    key = model_name.lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model_name={model_name!r}. Available: {available}")

    return MODEL_REGISTRY[key](num_classes=num_classes, feature_dim=feature_dim)


__all__ = [
    "CNNSmall",
    "MobileNetV2CIFAR",
    "ResNet18CIFAR",
    "MODEL_REGISTRY",
    "build_model",
]
