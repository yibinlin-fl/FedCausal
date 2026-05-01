"""FedCausal-Mask: FedProto with learnable FFT frequency masks."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from attacks.label_flip import flip_labels, save_malicious_client_ids, select_malicious_clients
from attacks.prototype_scaling import apply_prototype_scaling_to_payload
from attacks.visualization import save_attack_visualizations
from data.cifar import build_client_loaders
from federated.aggregation import aggregate_mask_and_prototypes
from losses.prototype_scl import prototype_scl_loss
from methods.frequency_mask import LearnableFrequencyMask
from models import build_model
from utils.checkpoint import save_checkpoint
from utils.logger import CSVLogger
from utils.seed import seed_everything


def _cfg(cfg: Mapping[str, Any], section: str, key: str, default: Any) -> Any:
    return cfg.get(section, {}).get(key, default)


def _build_optimizer(
    model: nn.Module,
    frequency_mask: LearnableFrequencyMask,
    cfg: Mapping[str, Any],
) -> torch.optim.Optimizer:
    name = str(_cfg(cfg, "optimizer", "name", "adam")).lower()
    lr = float(_cfg(cfg, "optimizer", "lr", 0.001))
    weight_decay = float(_cfg(cfg, "optimizer", "weight_decay", 0.0005))
    params = list(model.parameters()) + list(frequency_mask.parameters())

    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def _select_clients(
    num_clients: int,
    participation_rate: float,
    round_id: int,
    seed: int,
) -> List[int]:
    if participation_rate >= 1.0:
        return list(range(num_clients))

    sample_size = max(1, int(num_clients * participation_rate))
    rng = random.Random(seed + round_id)
    return sorted(rng.sample(range(num_clients), sample_size))


class FedCausalMaskClient:
    """Client that trains a heterogeneous model plus a learnable FFT mask."""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        frequency_mask: LearnableFrequencyMask,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str,
        num_classes: int,
        feature_dim: int,
        local_epochs: int,
        tau_s: float,
        lambda_scl: float,
        lambda_mask: float,
        lambda_sparse: float,
        attack_type: str = "none",
        is_malicious: bool = False,
        scale_factor: float = 10.0,
    ) -> None:
        self.client_id = client_id
        self.model = model.to(device)
        self.frequency_mask = frequency_mask.to(device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.local_epochs = local_epochs
        self.tau_s = tau_s
        self.lambda_scl = lambda_scl
        self.lambda_mask = lambda_mask
        self.lambda_sparse = lambda_sparse
        self.attack_type = attack_type
        self.is_malicious = is_malicious
        self.scale_factor = scale_factor
        self.criterion = nn.CrossEntropyLoss()

    def _training_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if self.is_malicious and self.attack_type == "label_flip":
            return flip_labels(labels, self.num_classes)
        return labels

    def sync_mask(self, global_mask: torch.Tensor) -> None:
        """Initialize or synchronize the local learnable mask from the server."""
        self.frequency_mask.set_mask(global_mask)

    def train_one_round(
        self,
        global_prototypes: torch.Tensor | None,
        global_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """Train model and mask locally for one communication round."""
        self.model.train()
        self.frequency_mask.train()
        self.sync_mask(global_mask)

        global_prototypes = (
            global_prototypes.to(self.device) if global_prototypes is not None else None
        )
        global_mask = global_mask.to(self.device)

        total_loss = 0.0
        total_loss_mask = 0.0
        total_loss_sparse = 0.0
        total_correct = 0
        total_samples = 0

        for _ in range(self.local_epochs):
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                train_labels = self._training_labels(labels)

                self.optimizer.zero_grad(set_to_none=True)
                x_c, _, _, local_mask = self.frequency_mask.apply_causal_filter(images)
                features = self.model.extract_features(x_c)
                logits = self.model.classifier(features)

                loss_ce = self.criterion(logits, train_labels)
                loss_scl = prototype_scl_loss(
                    features=features,
                    labels=train_labels,
                    global_prototypes=global_prototypes,
                    tau_s=self.tau_s,
                )
                loss_mask = F.mse_loss(local_mask, global_mask)
                loss_sparse = local_mask.abs().mean()
                loss = (
                    loss_ce
                    + self.lambda_scl * loss_scl
                    + self.lambda_mask * loss_mask
                    + self.lambda_sparse * loss_sparse
                )
                loss.backward()
                self.optimizer.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_loss_mask += loss_mask.item() * batch_size
                total_loss_sparse += loss_sparse.item() * batch_size
                total_correct += (logits.argmax(dim=1) == train_labels).sum().item()
                total_samples += batch_size

        mask_stats = self.frequency_mask.stats()
        return {
            "local_loss": total_loss / total_samples if total_samples else 0.0,
            "local_acc": total_correct / total_samples if total_samples else 0.0,
            "loss_mask": total_loss_mask / total_samples if total_samples else 0.0,
            "loss_sparse": total_loss_sparse / total_samples if total_samples else 0.0,
            "mask_mean": mask_stats["mask_mean"],
            "mask_std": mask_stats["mask_std"],
        }

    @torch.no_grad()
    def compute_local_payload(self) -> Dict[str, torch.Tensor]:
        """Compute causal-image prototypes and upload the local mask."""
        self.model.eval()
        self.frequency_mask.eval()
        sums = torch.zeros(self.num_classes, self.feature_dim, device=self.device)
        counts = torch.zeros(self.num_classes, device=self.device)

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            proto_labels = self._training_labels(labels)
            x_c, _, _, _ = self.frequency_mask.apply_causal_filter(images)
            features = self.model.extract_features(x_c)
            sums.index_add_(0, proto_labels, features)
            counts.index_add_(0, proto_labels, torch.ones_like(proto_labels, dtype=sums.dtype))

        valid_classes = counts > 0
        prototypes = torch.zeros_like(sums)
        prototypes[valid_classes] = sums[valid_classes] / counts[valid_classes].unsqueeze(1)
        local_mask = self.frequency_mask.get_mask().detach().cpu()

        return {
            "client_id": torch.tensor(self.client_id),
            "prototypes": prototypes.detach().cpu(),
            "counts": counts.detach().cpu(),
            "valid_classes": valid_classes.detach().cpu(),
            "local_mask": local_mask,
            "num_samples": counts.sum().detach().cpu(),
        }
        if self.is_malicious and self.attack_type == "prototype_scaling":
            payload = apply_prototype_scaling_to_payload(payload, self.scale_factor)
        return payload

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the masked model on clean test data."""
        self.model.eval()
        self.frequency_mask.eval()
        total = 0
        correct = 0

        for images, labels in test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            x_c, _, _, _ = self.frequency_mask.apply_causal_filter(images)
            logits = self.model(x_c)
            total += labels.numel()
            correct += (logits.argmax(dim=1) == labels).sum().item()

        return {"accuracy": correct / total if total else 0.0, "num_samples": float(total)}

    def state_dict(self) -> Dict[str, Any]:
        """Return serializable client state."""
        return {
            "client_id": self.client_id,
            "model": self.model.state_dict(),
            "frequency_mask": self.frequency_mask.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }


class FedCausalMaskServer:
    """Server that aggregates class prototypes and FFT masks."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        mask_shape: tuple[int, int, int, int],
        mask_init: float,
        device: torch.device | str,
        aggregation_mode: str = "mask_proto_energy",
        beta: float = 1.0,
        tau: float = 1.0,
    ) -> None:
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = torch.device(device)
        self.global_prototypes: torch.Tensor | None = None
        self.global_counts = torch.zeros(num_classes)
        self.valid_classes = torch.zeros(num_classes, dtype=torch.bool)
        self.global_mask = torch.full(mask_shape, float(mask_init))
        self.aggregation_mode = aggregation_mode
        self.beta = beta
        self.tau = tau
        self.last_client_stats: List[Dict[str, float]] = []

    def get_global_prototypes(self) -> torch.Tensor | None:
        """Return global prototypes on the server device."""
        if self.global_prototypes is None:
            return None
        return self.global_prototypes.to(self.device)

    def get_global_mask(self) -> torch.Tensor:
        """Return global mask on the server device."""
        return self.global_mask.to(self.device)

    def aggregate(self, payloads: Iterable[Dict[str, torch.Tensor]]) -> List[Dict[str, float]]:
        """Aggregate prototypes and masks with the configured strategy."""
        result = aggregate_mask_and_prototypes(
            payloads,
            num_classes=self.num_classes,
            feature_dim=self.feature_dim,
            current_prototypes=self.global_prototypes,
            current_mask=self.global_mask,
            mode=self.aggregation_mode,
            beta=self.beta,
            tau=self.tau,
        )
        self.global_prototypes = result.global_prototypes
        self.global_counts = torch.zeros(self.num_classes)
        self.valid_classes = self.global_prototypes.norm(dim=1) > 0
        self.global_mask = result.global_mask
        self.last_client_stats = result.client_stats
        return self.last_client_stats

    def _aggregate_prototypes(self, payloads: List[Dict[str, torch.Tensor]]) -> None:
        sums = torch.zeros(self.num_classes, self.feature_dim)
        counts = torch.zeros(self.num_classes)

        for payload in payloads:
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

    def _aggregate_masks(self, payloads: List[Dict[str, torch.Tensor]]) -> None:
        if not payloads:
            return

        total_weight = 0.0
        mask_sum = torch.zeros_like(self.global_mask)
        for payload in payloads:
            local_mask = payload["local_mask"].detach().cpu()
            weight = float(payload["num_samples"].item())
            mask_sum += local_mask * weight
            total_weight += weight

        if total_weight > 0:
            self.global_mask = (mask_sum / total_weight).clamp(0.0, 1.0)

    def state_dict(self) -> Dict[str, torch.Tensor | None]:
        """Return serializable server state."""
        return {
            "global_prototypes": self.global_prototypes,
            "global_counts": self.global_counts,
            "valid_classes": self.valid_classes,
            "global_mask": self.global_mask,
        }


def save_mask_heatmap(mask: torch.Tensor, path: str | Path) -> Path:
    """Save a channel-averaged FFT mask heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    heatmap = mask.detach().cpu().squeeze(0).mean(dim=0)
    heatmap = torch.fft.fftshift(heatmap).numpy()

    plt.figure(figsize=(4, 4))
    plt.imshow(heatmap, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Global Frequency Mask")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def run_fedcausal_mask(
    config: Mapping[str, Any],
    debug: bool = False,
) -> Dict[str, Any]:
    """Run FedCausal with learnable FFT masks and prototype aggregation."""
    seed = int(config.get("seed", 42))
    seed_everything(seed)

    requested_device = str(config.get("device", "cuda"))
    device = torch.device(
        "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    num_clients = int(_cfg(config, "federated", "num_clients", 10))
    rounds = int(_cfg(config, "federated", "rounds", 20))
    if debug:
        num_clients = min(num_clients, 5)
        rounds = min(rounds, 3)

    num_classes = int(_cfg(config, "dataset", "num_classes", 10))
    feature_dim = int(_cfg(config, "model", "feature_dim", 128))
    local_epochs = int(_cfg(config, "federated", "local_epochs", 1))
    participation_rate = float(_cfg(config, "federated", "participation_rate", 1.0))
    tau_s = float(_cfg(config, "fedcausal", "tau_s", 0.2))
    lambda_scl = float(_cfg(config, "fedcausal", "lambda_scl", 0.1))
    lambda_mask = float(_cfg(config, "fedcausal", "lambda_mask", 0.01))
    lambda_sparse = float(_cfg(config, "fedcausal", "lambda_sparse", 0.0001))
    mask_init = float(_cfg(config, "fedcausal", "mask_init", 0.5))
    aggregation_mode = str(_cfg(config, "aggregation", "mode", "mask_proto_energy"))
    aggregation_beta = float(_cfg(config, "aggregation", "beta", 1.0))
    aggregation_tau = float(_cfg(config, "aggregation", "tau", 1.0))
    attack_type = str(_cfg(config, "attack", "type", "none")).lower()
    malicious_ratio = float(_cfg(config, "attack", "malicious_ratio", 0.0))
    scale_factor = float(_cfg(config, "attack", "scale_factor", 10.0))
    malicious_client_ids = (
        select_malicious_clients(num_clients, malicious_ratio, seed)
        if attack_type != "none"
        else []
    )

    client_loaders, test_loader, _ = build_client_loaders(
        cfg=config,
        num_clients=num_clients,
    )

    client_model_names = list(_cfg(config, "model", "client_models", ["cnn_small"]))
    if len(client_model_names) < num_clients:
        repeats = (num_clients + len(client_model_names) - 1) // len(client_model_names)
        client_model_names = (client_model_names * repeats)[:num_clients]
    else:
        client_model_names = client_model_names[:num_clients]

    clients: List[FedCausalMaskClient] = []
    for client_id in range(num_clients):
        model = build_model(
            model_name=client_model_names[client_id],
            num_classes=num_classes,
            feature_dim=feature_dim,
        )
        frequency_mask = LearnableFrequencyMask(mask_init=mask_init)
        optimizer = _build_optimizer(model, frequency_mask, config)
        clients.append(
            FedCausalMaskClient(
                client_id=client_id,
                model=model,
                frequency_mask=frequency_mask,
                train_loader=client_loaders[client_id],
                optimizer=optimizer,
                device=device,
                num_classes=num_classes,
                feature_dim=feature_dim,
                local_epochs=local_epochs,
                tau_s=tau_s,
                lambda_scl=lambda_scl,
                lambda_mask=lambda_mask,
                lambda_sparse=lambda_sparse,
                attack_type=attack_type,
                is_malicious=client_id in malicious_client_ids,
                scale_factor=scale_factor,
            )
        )

    server = FedCausalMaskServer(
        num_classes=num_classes,
        feature_dim=feature_dim,
        mask_shape=(1, 3, 32, 32),
        mask_init=mask_init,
        device=device,
        aggregation_mode=aggregation_mode,
        beta=aggregation_beta,
        tau=aggregation_tau,
    )

    result_dir = Path(_cfg(config, "output", "result_dir", "/kaggle/working/FedCausal/results"))
    checkpoint_dir = Path(
        _cfg(config, "output", "checkpoint_dir", "/kaggle/working/FedCausal/checkpoints")
    )
    figure_dir = Path(_cfg(config, "output", "figure_dir", "/kaggle/working/FedCausal/figures"))
    malicious_ids_path = result_dir / "malicious_client_ids.json"
    save_malicious_client_ids(
        malicious_client_ids,
        malicious_ids_path,
        attack_type=attack_type,
        malicious_ratio=malicious_ratio,
        scale_factor=scale_factor,
    )
    print(f"Malicious clients: {malicious_client_ids}")
    print(f"Saved malicious client ids to: {malicious_ids_path}")

    logger = CSVLogger(
        result_dir / "fedcausal_mask_clean_results.csv",
        fieldnames=[
            "round",
            "method",
            "client_id",
            "local_loss",
            "local_acc",
            "clean_acc",
            "mask_mean",
            "mask_std",
            "loss_mask",
            "loss_sparse",
        ],
        reset=True,
    )
    attack_logger = CSVLogger(
        result_dir / "attack_results.csv",
        fieldnames=[
            "round",
            "method",
            "attack_type",
            "malicious_ratio",
            "client_id",
            "is_malicious",
            "d_mask",
            "d_proto",
            "energy",
            "alpha_i",
            "clean_acc",
        ],
        reset=True,
    )

    history: List[Dict[str, Any]] = []
    for round_id in range(rounds):
        print(f"\n[FedCausal-Mask] Round {round_id + 1}/{rounds}")
        selected_client_ids = _select_clients(num_clients, participation_rate, round_id, seed)
        global_prototypes = server.get_global_prototypes()
        global_mask = server.get_global_mask()

        payloads = []
        train_metrics_by_client: Dict[int, Dict[str, float]] = {}
        for client_id in selected_client_ids:
            client = clients[client_id]
            train_metrics = client.train_one_round(global_prototypes, global_mask)
            train_metrics_by_client[client_id] = train_metrics
            payloads.append(client.compute_local_payload())
            print(
                f"  client={client_id:02d} "
                f"loss={train_metrics['local_loss']:.4f} "
                f"local_acc={train_metrics['local_acc']:.4f} "
                f"mask_mean={train_metrics['mask_mean']:.4f}"
            )

        aggregation_stats = server.aggregate(payloads)
        aggregation_stats_by_client = {
            int(stat["client_id"]): stat for stat in aggregation_stats
        }
        current_global_mask = server.get_global_mask().detach().cpu()
        mask_path = checkpoint_dir / "global_mask.pt"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"round": round_id, "global_mask": current_global_mask}, mask_path)

        if (round_id + 1) % 5 == 0 or (round_id + 1) == rounds:
            heatmap_path = save_mask_heatmap(
                current_global_mask,
                figure_dir / f"global_mask_round_{round_id + 1}.png",
            )
            print(f"  saved mask heatmap: {heatmap_path}")

        clean_accs = []
        clean_acc_by_client: Dict[int, float] = {}
        for client in clients:
            eval_metrics = client.evaluate(test_loader)
            clean_acc = eval_metrics["accuracy"]
            clean_acc_by_client[client.client_id] = clean_acc
            clean_accs.append(clean_acc)
            train_metrics = train_metrics_by_client.get(client.client_id, {})
            mask_stats = client.frequency_mask.stats()

            row = {
                "round": round_id,
                "method": "fedcausal_mask",
                "client_id": client.client_id,
                "local_loss": train_metrics.get("local_loss", ""),
                "local_acc": train_metrics.get("local_acc", ""),
                "clean_acc": clean_acc,
                "mask_mean": train_metrics.get("mask_mean", mask_stats["mask_mean"]),
                "mask_std": train_metrics.get("mask_std", mask_stats["mask_std"]),
                "loss_mask": train_metrics.get("loss_mask", ""),
                "loss_sparse": train_metrics.get("loss_sparse", ""),
            }
            logger.log(row)
            history.append(row)

        for client_id in selected_client_ids:
            stat = aggregation_stats_by_client.get(client_id, {})
            attack_logger.log(
                {
                    "round": round_id,
                    "method": "fedcausal_mask",
                    "attack_type": attack_type,
                    "malicious_ratio": malicious_ratio,
                    "client_id": client_id,
                    "is_malicious": client_id in malicious_client_ids,
                    "d_mask": stat.get("d_mask", ""),
                    "d_proto": stat.get("d_proto", ""),
                    "energy": stat.get("energy", ""),
                    "alpha_i": stat.get("alpha_i", ""),
                    "clean_acc": clean_acc_by_client.get(client_id, ""),
                }
            )

        mean_clean_acc = sum(clean_accs) / len(clean_accs) if clean_accs else 0.0
        print(
            f"  mean_clean_acc={mean_clean_acc:.4f} "
            f"global_mask_mean={current_global_mask.mean().item():.4f} "
            f"global_mask_std={current_global_mask.std(unbiased=False).item():.4f}"
        )

        if (round_id + 1) % 5 == 0 or (round_id + 1) == rounds:
            checkpoint_path = save_checkpoint(
                {
                    "round": round_id,
                    "method": "fedcausal_mask",
                    "config": dict(config),
                    "server": server.state_dict(),
                    "clients": [client.state_dict() for client in clients],
                },
                checkpoint_dir=checkpoint_dir,
                filename=f"fedcausal_mask_round_{round_id + 1}.pt",
            )
            print(f"  saved checkpoint: {checkpoint_path}")

    attack_figure_paths = save_attack_visualizations(
        result_dir / "attack_results.csv",
        figure_dir,
        method="fedcausal_mask",
        attack_type=attack_type,
        malicious_ratio=malicious_ratio,
        scale_factor=scale_factor,
    )

    return {
        "history": history,
        "server": server,
        "clients": clients,
        "result_csv": result_dir / "fedcausal_mask_clean_results.csv",
        "global_mask_path": checkpoint_dir / "global_mask.pt",
        "attack_csv": result_dir / "attack_results.csv",
        "attack_figures": attack_figure_paths,
    }
