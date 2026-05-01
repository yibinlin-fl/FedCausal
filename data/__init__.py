"""Dataset and partitioning components.

This package keeps imports lazy so utilities such as ``data.partition`` can be
tested without importing PyTorch or torchvision.
"""

__all__ = [
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "build_cifar10_transforms",
    "build_client_loaders",
    "compute_client_label_distribution",
    "dirichlet_partition",
    "get_cifar10_datasets",
    "get_dataset_targets",
    "print_client_label_distribution",
    "save_client_label_distribution",
]


def __getattr__(name: str):
    if name in {
        "CIFAR10_MEAN",
        "CIFAR10_STD",
        "build_cifar10_transforms",
        "build_client_loaders",
        "get_cifar10_datasets",
    }:
        from data import cifar

        return getattr(cifar, name)

    if name in {
        "compute_client_label_distribution",
        "dirichlet_partition",
        "get_dataset_targets",
        "print_client_label_distribution",
        "save_client_label_distribution",
    }:
        from data import partition

        return getattr(partition, name)

    raise AttributeError(f"module 'data' has no attribute {name!r}")
