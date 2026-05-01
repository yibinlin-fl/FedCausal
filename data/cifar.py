"""CIFAR-10 data loading for Kaggle FedCausal experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from data.partition import (
    compute_client_label_distribution,
    dirichlet_partition,
    print_client_label_distribution,
    save_client_label_distribution,
)


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_cifar10_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Build the standard train/test transforms used by the MVP."""
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return train_transform, test_transform


def get_cifar10_datasets(
    data_root: str | Path = "/kaggle/working/data",
    download: bool = True,
) -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Load CIFAR-10 train and clean test datasets."""
    train_transform, test_transform = build_cifar10_transforms()
    data_root = Path(data_root)

    train_dataset = datasets.CIFAR10(
        root=str(data_root),
        train=True,
        transform=train_transform,
        download=download,
    )
    test_dataset = datasets.CIFAR10(
        root=str(data_root),
        train=False,
        transform=test_transform,
        download=download,
    )
    return train_dataset, test_dataset


def _cfg_value(
    cfg: Optional[Mapping[str, Any]],
    section: str,
    key: str,
    default: Any,
) -> Any:
    if cfg is None:
        return default
    return cfg.get(section, {}).get(key, default)


def build_client_loaders(
    cfg: Optional[Mapping[str, Any]] = None,
    data_root: Optional[str | Path] = None,
    batch_size: Optional[int] = None,
    num_clients: Optional[int] = None,
    alpha: Optional[float] = None,
    num_classes: Optional[int] = None,
    seed: Optional[int] = None,
    num_workers: int = 2,
    pin_memory: bool = True,
    download: bool = True,
    result_dir: Optional[str | Path] = None,
) -> Tuple[Dict[int, DataLoader], DataLoader, Dict[int, list[int]]]:
    """Build per-client train loaders and one clean CIFAR-10 test loader.

    Args:
        cfg: Optional loaded YAML config. Explicit keyword arguments override it.

    Returns:
        client_loaders: Mapping from client id to train DataLoader.
        test_loader: Clean CIFAR-10 test DataLoader.
        client_indices: Mapping from client id to dataset indices.
    """
    data_root = data_root or _cfg_value(cfg, "dataset", "data_root", "/kaggle/working/data")
    batch_size = int(batch_size or _cfg_value(cfg, "federated", "batch_size", 64))
    num_clients = int(num_clients or _cfg_value(cfg, "federated", "num_clients", 10))
    alpha = float(alpha or _cfg_value(cfg, "federated", "dirichlet_alpha", 0.3))
    num_classes = int(num_classes or _cfg_value(cfg, "dataset", "num_classes", 10))
    seed = int(seed if seed is not None else (cfg.get("seed", 42) if cfg else 42))
    result_dir = Path(
        result_dir
        or _cfg_value(cfg, "output", "result_dir", "/kaggle/working/FedCausal/results")
    )

    train_dataset, test_dataset = get_cifar10_datasets(data_root=data_root, download=download)

    client_indices = dirichlet_partition(
        dataset=train_dataset,
        num_clients=num_clients,
        alpha=alpha,
        num_classes=num_classes,
        seed=seed,
    )

    distribution = compute_client_label_distribution(train_dataset, client_indices, num_classes)
    print_client_label_distribution(distribution)
    csv_path = result_dir / "client_label_distribution.csv"
    save_client_label_distribution(distribution, csv_path)
    print(f"Saved client label distribution to: {csv_path}")

    client_loaders: Dict[int, DataLoader] = {}
    for client_id in range(num_clients):
        subset = Subset(train_dataset, client_indices[client_id])
        generator = torch.Generator()
        generator.manual_seed(seed + client_id)
        client_loaders[client_id] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=generator,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return client_loaders, test_loader, client_indices
