"""Checkpoint helpers for Kaggle runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


PathLike = Union[str, Path]


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: PathLike,
    filename: str,
) -> Path:
    """Save a checkpoint and create the target directory if needed."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    path = checkpoint_dir / filename
    torch.save(state, path)
    return path


def load_checkpoint(
    path: PathLike,
    map_location: Optional[Union[str, torch.device]] = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return torch.load(path, map_location=map_location)


def checkpoint_exists(path: PathLike) -> bool:
    """Return whether a checkpoint path exists."""
    return Path(path).exists()
