"""Server-side prototype aggregation."""

from __future__ import annotations

from typing import Dict, Iterable

import torch


class Server:
    """FedProto server that aggregates prototypes, not model parameters."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = torch.device(device)
        self.global_prototypes: torch.Tensor | None = None
        self.global_counts = torch.zeros(num_classes)
        self.valid_classes = torch.zeros(num_classes, dtype=torch.bool)

    def aggregate(
        self,
        client_payloads: Iterable[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Sample-count weighted class-wise prototype averaging."""
        sums = torch.zeros(self.num_classes, self.feature_dim)
        counts = torch.zeros(self.num_classes)

        for payload in client_payloads:
            prototypes = payload["prototypes"].detach().cpu()
            client_counts = payload["counts"].detach().cpu().float()
            valid = client_counts > 0
            sums[valid] += prototypes[valid] * client_counts[valid].unsqueeze(1)
            counts[valid] += client_counts[valid]

        if self.global_prototypes is None:
            new_global = torch.zeros(self.num_classes, self.feature_dim)
        else:
            new_global = self.global_prototypes.detach().cpu().clone()

        updated = counts > 0
        new_global[updated] = sums[updated] / counts[updated].unsqueeze(1)

        self.global_prototypes = new_global
        self.global_counts = counts
        self.valid_classes = self.valid_classes | updated
        return self.global_prototypes.to(self.device)

    def get_global_prototypes(self) -> torch.Tensor | None:
        """Return global prototypes on the configured device."""
        if self.global_prototypes is None:
            return None
        return self.global_prototypes.to(self.device)

    def state_dict(self) -> Dict[str, torch.Tensor | None]:
        """Return serializable server state."""
        return {
            "global_prototypes": self.global_prototypes,
            "global_counts": self.global_counts,
            "valid_classes": self.valid_classes,
        }
