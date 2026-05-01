"""Prototype scaling attack utilities."""

from __future__ import annotations

from typing import Dict

import torch


def scale_prototypes(
    prototypes: torch.Tensor,
    scale_factor: float,
) -> torch.Tensor:
    """Return scaled prototypes."""
    return prototypes * float(scale_factor)


def apply_prototype_scaling_to_payload(
    payload: Dict[str, torch.Tensor],
    scale_factor: float,
) -> Dict[str, torch.Tensor]:
    """Scale uploaded prototypes in a client payload."""
    payload = dict(payload)
    payload["prototypes"] = scale_prototypes(payload["prototypes"], scale_factor)
    return payload
