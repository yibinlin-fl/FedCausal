"""Federated client for FedProto-style heterogeneous FL."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from losses.prototype_scl import prototype_scl_loss
from utils.metrics import evaluate_model


class Client:
    """A single heterogeneous federated client."""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str,
        num_classes: int,
        feature_dim: int,
        local_epochs: int = 1,
        tau_s: float = 0.2,
        lambda_scl: float = 0.1,
        global_prototypes: torch.Tensor | None = None,
    ) -> None:
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.local_epochs = local_epochs
        self.tau_s = tau_s
        self.lambda_scl = lambda_scl
        self.global_prototypes = global_prototypes
        self.criterion = nn.CrossEntropyLoss()

    def train_one_round(
        self,
        global_prototypes: torch.Tensor | None,
    ) -> Dict[str, float]:
        """Train locally for one communication round."""
        self.model.train()
        self.global_prototypes = (
            global_prototypes.to(self.device) if global_prototypes is not None else None
        )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for _ in range(self.local_epochs):
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                features = self.model.extract_features(images)
                logits = self.model.classifier(features)

                loss_ce = self.criterion(logits, labels)
                loss_scl = prototype_scl_loss(
                    features=features,
                    labels=labels,
                    global_prototypes=self.global_prototypes,
                    tau_s=self.tau_s,
                )
                loss = loss_ce + self.lambda_scl * loss_scl
                loss.backward()
                self.optimizer.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (logits.argmax(dim=1) == labels).sum().item()
                total_samples += batch_size

        return {
            "local_loss": total_loss / total_samples if total_samples else 0.0,
            "local_acc": total_correct / total_samples if total_samples else 0.0,
        }

    @torch.no_grad()
    def compute_local_prototypes(self) -> Dict[str, torch.Tensor]:
        """Compute class-wise mean features on local data."""
        self.model.eval()
        sums = torch.zeros(self.num_classes, self.feature_dim, device=self.device)
        counts = torch.zeros(self.num_classes, device=self.device)

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            features = self.model.extract_features(images)

            sums.index_add_(0, labels, features)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=sums.dtype))

        valid_classes = counts > 0
        prototypes = torch.zeros_like(sums)
        prototypes[valid_classes] = sums[valid_classes] / counts[valid_classes].unsqueeze(1)

        return {
            "client_id": torch.tensor(self.client_id),
            "prototypes": prototypes.detach().cpu(),
            "counts": counts.detach().cpu(),
            "valid_classes": valid_classes.detach().cpu(),
        }

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate this client's model on a clean test loader."""
        return evaluate_model(self.model, test_loader, self.device)

    def state_dict(self) -> Dict[str, Any]:
        """Return serializable client state."""
        return {
            "client_id": self.client_id,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
