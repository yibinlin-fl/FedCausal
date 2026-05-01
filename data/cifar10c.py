"""CIFAR-10-C dataset loader for Kaggle evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data.cifar import CIFAR10_MEAN, CIFAR10_STD


CIFAR10C_ROOT = Path("/kaggle/input/cifar10-c")
SUPPORTED_CIFAR10C_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "motion_blur",
    "defocus_blur",
    "fog",
    "jpeg_compression",
    "pixelate",
]
SUPPORTED_CIFAR10C_SEVERITIES = [1, 3, 5]
CIFAR10C_MISSING_MESSAGE = (
    "未检测到 CIFAR-10-C，请在 Kaggle Dataset 中添加 CIFAR-10-C 数据集，"
    "并确保路径为 /kaggle/input/cifar10-c/。"
)


class CIFAR10CDataset(Dataset):
    """CIFAR-10-C severity slice dataset."""

    def __init__(
        self,
        root: str | Path = CIFAR10C_ROOT,
        corruption: str = "gaussian_noise",
        severity: int = 3,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        root = Path(root)
        if severity not in {1, 2, 3, 4, 5}:
            raise ValueError("severity must be one of {1, 2, 3, 4, 5}.")
        if corruption not in SUPPORTED_CIFAR10C_CORRUPTIONS:
            available = ", ".join(SUPPORTED_CIFAR10C_CORRUPTIONS)
            raise ValueError(f"Unsupported CIFAR-10-C corruption: {corruption}. Available: {available}")

        data_path = root / f"{corruption}.npy"
        labels_path = root / "labels.npy"
        if not root.exists() or not data_path.exists() or not labels_path.exists():
            raise FileNotFoundError(CIFAR10C_MISSING_MESSAGE)

        start = (severity - 1) * 10000
        end = severity * 10000
        data = np.load(data_path, mmap_mode="r")
        labels = np.load(labels_path, mmap_mode="r")

        self.data = data[start:end]
        if len(labels) == 10000:
            self.labels = labels.astype(np.int64)
        else:
            self.labels = labels[start:end].astype(np.int64)
        self.corruption = corruption
        self.severity = severity
        self.transform = transform or build_cifar10c_transform()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image_array = np.asarray(self.data[index]).astype(np.uint8)
        if image_array.ndim == 3 and image_array.shape[0] == 3:
            image_array = np.transpose(image_array, (1, 2, 0))
        image = Image.fromarray(image_array)
        label = int(self.labels[index])
        return self.transform(image), label


def build_cifar10c_transform() -> transforms.Compose:
    """Build the CIFAR-10-C evaluation transform."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def build_cifar10c_loader(
    root: str | Path = CIFAR10C_ROOT,
    corruption: str = "gaussian_noise",
    severity: int = 3,
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Optional[DataLoader]:
    """Build a CIFAR-10-C loader, returning None when the Kaggle data is missing."""
    try:
        dataset = CIFAR10CDataset(root=root, corruption=corruption, severity=severity)
    except FileNotFoundError:
        print(CIFAR10C_MISSING_MESSAGE)
        return None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def cifar10c_available(root: str | Path = CIFAR10C_ROOT) -> bool:
    """Return whether the expected CIFAR-10-C files are available."""
    root = Path(root)
    return root.exists() and (root / "labels.npy").exists()
