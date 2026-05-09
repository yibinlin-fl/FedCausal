"""CIFAR-10-C dataset loader for Kaggle evaluation."""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data.cifar import CIFAR10_MEAN, CIFAR10_STD


CIFAR10C_ROOT = Path("/kaggle/input/cifar10-c")
CIFAR10C_URL = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
CIFAR10C_ARCHIVE_NAME = "CIFAR-10-C.tar"
CIFAR10C_TEMP_ROOT = Path("/kaggle/temp/CIFAR-10-C")
CIFAR10C_TEMP_ARCHIVE = Path("/kaggle/temp") / CIFAR10C_ARCHIVE_NAME
CIFAR10C_WORKING_ROOT = Path("/kaggle/working/CIFAR-10-C")
CIFAR10C_WORKING_ARCHIVE = Path("/kaggle/working") / CIFAR10C_ARCHIVE_NAME

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
    "CIFAR-10-C was not found. On Kaggle, either enable Internet so the notebook "
    "can download it from Zenodo, or attach/upload a dataset containing labels.npy "
    "and corruption .npy files such as gaussian_noise.npy."
)


def _candidate_roots(configured_root: str | Path = CIFAR10C_ROOT) -> list[Path]:
    roots = [
        Path(configured_root),
        CIFAR10C_TEMP_ROOT,
        Path("/kaggle/temp/cifar10-c"),
        Path("/kaggle/temp"),
        CIFAR10C_WORKING_ROOT,
        Path("/kaggle/working/cifar10-c"),
        Path("/kaggle/working"),
        Path("/kaggle/input"),
    ]
    return [root for root in roots if root.exists()]


def find_cifar10c_root(configured_root: str | Path = CIFAR10C_ROOT) -> Optional[Path]:
    """Find the directory that directly contains CIFAR-10-C labels.npy."""
    seen = set()
    for candidate in _candidate_roots(configured_root):
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)

        if (candidate / "labels.npy").exists() and (candidate / "gaussian_noise.npy").exists():
            return candidate

        nested = candidate / "CIFAR-10-C"
        if (nested / "labels.npy").exists() and (nested / "gaussian_noise.npy").exists():
            return nested

        if candidate.name in {"input", "temp", "working"}:
            for labels_path in candidate.rglob("labels.npy"):
                parent = labels_path.parent
                if (parent / "gaussian_noise.npy").exists():
                    return parent
    return None


def find_cifar10c_archive() -> Optional[Path]:
    """Find a CIFAR-10-C tar archive in common Kaggle locations."""
    candidates = [
        CIFAR10C_TEMP_ARCHIVE,
        CIFAR10C_WORKING_ARCHIVE,
        Path("/kaggle/input"),
        Path("/kaggle/temp"),
        Path("/kaggle/working"),
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.name == CIFAR10C_ARCHIVE_NAME:
            return candidate
        if candidate.is_dir():
            for archive_path in candidate.rglob(CIFAR10C_ARCHIVE_NAME):
                if archive_path.is_file():
                    return archive_path
    return None


def safe_extract_cifar10c_tar(archive_path: str | Path, extract_root: str | Path) -> None:
    """Extract a tar archive without allowing paths outside extract_root."""
    archive_path = Path(archive_path)
    extract_root = Path(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    extract_root_resolved = extract_root.resolve()

    with tarfile.open(archive_path, "r") as tar:
        for member in tar.getmembers():
            target_path = (extract_root / member.name).resolve()
            try:
                target_path.relative_to(extract_root_resolved)
            except ValueError as exc:
                raise RuntimeError(f"Unsafe path in CIFAR-10-C archive: {member.name}") from exc
        tar.extractall(path=extract_root)


def ensure_cifar10c_root(
    configured_root: str | Path = CIFAR10C_ROOT,
    download: bool = True,
) -> Optional[Path]:
    """Find, extract, or download CIFAR-10-C for Kaggle evaluation."""
    root = find_cifar10c_root(configured_root)
    if root is not None:
        print(f"Using CIFAR-10-C from: {root}")
        return root

    archive_path = find_cifar10c_archive()
    cache_root = Path("/kaggle/temp") if Path("/kaggle").exists() else Path("/kaggle/working")
    archive_target = cache_root / CIFAR10C_ARCHIVE_NAME
    if archive_path is None and download:
        print("CIFAR-10-C not found locally. Trying to download it from Zenodo.")
        print(f"URL: {CIFAR10C_URL}")
        print(f"Target: {archive_target}")
        try:
            cache_root.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(CIFAR10C_URL, archive_target)
        except Exception as exc:
            print(f"CIFAR-10-C download failed: {type(exc).__name__}: {exc}")
            print("Enable Internet in the Kaggle notebook, or attach CIFAR-10-C as a dataset.")
            return None
        archive_path = archive_target

    if archive_path is not None:
        print(f"Extracting CIFAR-10-C archive: {archive_path}")
        safe_extract_cifar10c_tar(archive_path, archive_path.parent)
        root = find_cifar10c_root(configured_root)
        if root is not None:
            print(f"Using CIFAR-10-C from: {root}")
            return root

    print(CIFAR10C_MISSING_MESSAGE)
    return None


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
    """Build a CIFAR-10-C loader, returning None when the data is missing."""
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
    return find_cifar10c_root(root) is not None
