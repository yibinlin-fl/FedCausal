"""Prototype and mask aggregation strategies for FedCausal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F


SUPPORTED_AGGREGATION_MODES = {
    "avg",
    "prototype_median",
    "mask_only_energy",
    "mask_proto_energy",
}


@dataclass
class AggregationResult:
    """Result of server aggregation."""

    global_prototypes: torch.Tensor
    global_mask: torch.Tensor
    client_stats: List[Dict[str, float]]


def _payload_client_id(payload: Dict[str, torch.Tensor]) -> int:
    value = payload.get("client_id", -1)
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _safe_num_samples(payload: Dict[str, torch.Tensor]) -> float:
    value = payload.get("num_samples", payload["counts"].sum())
    if isinstance(value, torch.Tensor):
        return max(float(value.item()), 0.0)
    return max(float(value), 0.0)


def _median_mask(payloads: List[Dict[str, torch.Tensor]], current_mask: torch.Tensor) -> torch.Tensor:
    masks = [payload["local_mask"].detach().cpu().float() for payload in payloads if "local_mask" in payload]
    if not masks:
        return current_mask.detach().cpu().float().clone()
    return torch.stack(masks, dim=0).median(dim=0).values


def _median_prototypes(
    payloads: List[Dict[str, torch.Tensor]],
    num_classes: int,
    feature_dim: int,
    current_prototypes: torch.Tensor | None,
) -> torch.Tensor:
    if current_prototypes is None:
        med = torch.zeros(num_classes, feature_dim)
    else:
        med = current_prototypes.detach().cpu().float().clone()

    for class_id in range(num_classes):
        values = []
        for payload in payloads:
            counts = payload["counts"].detach().cpu().float()
            if counts[class_id] > 0:
                values.append(payload["prototypes"].detach().cpu().float()[class_id])
        if values:
            med[class_id] = torch.stack(values, dim=0).median(dim=0).values
    return med


def _client_distances(
    payload: Dict[str, torch.Tensor],
    median_mask: torch.Tensor,
    median_prototypes: torch.Tensor,
) -> tuple[float, float]:
    local_mask = payload["local_mask"].detach().cpu().float()
    d_mask = torch.mean(torch.abs(local_mask - median_mask)).item()

    counts = payload["counts"].detach().cpu().float()
    valid = counts > 0
    if not torch.any(valid):
        return d_mask, 0.0

    local_proto = payload["prototypes"].detach().cpu().float()[valid].flatten()
    med_proto = median_prototypes.detach().cpu().float()[valid].flatten()
    if local_proto.numel() == 0 or local_proto.norm() == 0 or med_proto.norm() == 0:
        return d_mask, 0.0

    cosine = F.cosine_similarity(local_proto.unsqueeze(0), med_proto.unsqueeze(0)).item()
    d_proto = float(1.0 - cosine)
    return d_mask, d_proto


def _energy_weights(
    payloads: List[Dict[str, torch.Tensor]],
    median_mask: torch.Tensor,
    median_prototypes: torch.Tensor,
    mode: str,
    beta: float,
    tau: float,
) -> tuple[torch.Tensor, List[Dict[str, float]]]:
    energies = []
    stats = []

    for payload in payloads:
        d_mask, d_proto = _client_distances(payload, median_mask, median_prototypes)
        if mode == "mask_only_energy":
            energy = d_mask
        else:
            energy = d_mask + float(beta) * d_proto
        energies.append(energy)
        stats.append(
            {
                "client_id": float(_payload_client_id(payload)),
                "d_mask": d_mask,
                "d_proto": d_proto,
                "energy": energy,
                "alpha_i": 0.0,
            }
        )

    energy_tensor = torch.tensor(energies, dtype=torch.float32)
    tau = max(float(tau), 1.0e-8)
    alphas = torch.softmax(-energy_tensor / tau, dim=0)
    for idx, alpha in enumerate(alphas):
        stats[idx]["alpha_i"] = alpha.item()
    return alphas, stats


def _sample_weights(payloads: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    weights = torch.tensor([_safe_num_samples(payload) for payload in payloads], dtype=torch.float32)
    if torch.sum(weights) <= 0:
        weights = torch.ones(len(payloads), dtype=torch.float32)
    return weights / torch.sum(weights)


def _aggregate_mask(
    payloads: List[Dict[str, torch.Tensor]],
    alphas: torch.Tensor,
    current_mask: torch.Tensor,
) -> torch.Tensor:
    if not payloads:
        return current_mask.detach().cpu().float().clone()

    mask_sum = torch.zeros_like(current_mask.detach().cpu().float())
    for alpha, payload in zip(alphas, payloads):
        mask_sum += float(alpha.item()) * payload["local_mask"].detach().cpu().float()
    return mask_sum.clamp(0.0, 1.0)


def _aggregate_prototypes_weighted(
    payloads: List[Dict[str, torch.Tensor]],
    alphas: torch.Tensor,
    num_classes: int,
    feature_dim: int,
    current_prototypes: torch.Tensor | None,
) -> torch.Tensor:
    if current_prototypes is None:
        new_global = torch.zeros(num_classes, feature_dim)
    else:
        new_global = current_prototypes.detach().cpu().float().clone()

    for class_id in range(num_classes):
        proto_sum = torch.zeros(feature_dim)
        weight_sum = 0.0
        for alpha, payload in zip(alphas, payloads):
            counts = payload["counts"].detach().cpu().float()
            if counts[class_id] <= 0:
                continue
            weight = float(alpha.item())
            proto_sum += weight * payload["prototypes"].detach().cpu().float()[class_id]
            weight_sum += weight
        if weight_sum > 0:
            new_global[class_id] = proto_sum / weight_sum
    return new_global


def aggregate_mask_and_prototypes(
    payloads: Iterable[Dict[str, torch.Tensor]],
    num_classes: int,
    feature_dim: int,
    current_prototypes: torch.Tensor | None,
    current_mask: torch.Tensor,
    mode: str = "mask_proto_energy",
    beta: float = 1.0,
    tau: float = 1.0,
) -> AggregationResult:
    """Aggregate uploaded masks and prototypes.

    Missing classes are skipped class-wise. All returned tensors live on CPU.
    """
    payloads = list(payloads)
    mode = str(mode).lower()
    if mode not in SUPPORTED_AGGREGATION_MODES:
        available = ", ".join(sorted(SUPPORTED_AGGREGATION_MODES))
        raise ValueError(f"Unsupported aggregation mode={mode!r}. Available: {available}")

    current_mask = current_mask.detach().cpu().float()
    median_mask = _median_mask(payloads, current_mask)
    median_prototypes = _median_prototypes(payloads, num_classes, feature_dim, current_prototypes)

    if not payloads:
        return AggregationResult(
            global_prototypes=median_prototypes,
            global_mask=current_mask,
            client_stats=[],
        )

    if mode == "prototype_median":
        alphas = _sample_weights(payloads)
        stats = []
        for alpha, payload in zip(alphas, payloads):
            d_mask, d_proto = _client_distances(payload, median_mask, median_prototypes)
            stats.append(
                {
                    "client_id": float(_payload_client_id(payload)),
                    "d_mask": d_mask,
                    "d_proto": d_proto,
                    "energy": 0.0,
                    "alpha_i": alpha.item(),
                }
            )
        return AggregationResult(
            global_prototypes=median_prototypes,
            global_mask=_aggregate_mask(payloads, alphas, current_mask),
            client_stats=stats,
        )

    if mode == "avg":
        alphas = _sample_weights(payloads)
        stats = []
        for alpha, payload in zip(alphas, payloads):
            d_mask, d_proto = _client_distances(payload, median_mask, median_prototypes)
            stats.append(
                {
                    "client_id": float(_payload_client_id(payload)),
                    "d_mask": d_mask,
                    "d_proto": d_proto,
                    "energy": 0.0,
                    "alpha_i": alpha.item(),
                }
            )
    else:
        alphas, stats = _energy_weights(
            payloads,
            median_mask=median_mask,
            median_prototypes=median_prototypes,
            mode=mode,
            beta=beta,
            tau=tau,
        )

    global_mask = _aggregate_mask(payloads, alphas, current_mask)
    global_prototypes = _aggregate_prototypes_weighted(
        payloads,
        alphas,
        num_classes=num_classes,
        feature_dim=feature_dim,
        current_prototypes=current_prototypes,
    )
    return AggregationResult(
        global_prototypes=global_prototypes,
        global_mask=global_mask,
        client_stats=stats,
    )
