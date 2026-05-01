"""Dataset and partitioning components.

This package keeps imports lazy so utilities such as ``data.partition`` can be
tested without importing PyTorch or torchvision.
"""

__all__ = [
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "build_cifar10_transforms",
    "build_client_loaders",
    "build_cifar10c_loader",
    "compute_client_label_distribution",
    "dirichlet_partition",
    "get_cifar10_datasets",
    "get_dataset_targets",
    "print_client_label_distribution",
    "save_client_label_distribution",
    "select_corrupted_clients",
]


def __getattr__(name: str):
    if name in {
        "CIFAR10_MEAN",
        "CIFAR10_STD",
        "build_cifar10_transforms",
        "build_client_loaders",
        "build_cifar10c_loader",
        "get_cifar10_datasets",
    }:
        if name == "build_cifar10c_loader":
            from data import cifar10c

            return getattr(cifar10c, name)

        from data import cifar

        return getattr(cifar, name)

    if name in {
        "compute_client_label_distribution",
        "dirichlet_partition",
        "get_dataset_targets",
        "print_client_label_distribution",
        "save_client_label_distribution",
        "select_corrupted_clients",
    }:
        if name == "select_corrupted_clients":
            from data import cifar

            return getattr(cifar, name)

        from data import partition

        return getattr(partition, name)

    raise AttributeError(f"module 'data' has no attribute {name!r}")
