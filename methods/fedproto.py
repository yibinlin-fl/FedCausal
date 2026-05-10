"""FedProto baseline training loop for Kaggle."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch

from data.cifar import build_client_loaders
from federated.client import Client
from federated.server import Server
from models import build_model
from utils.checkpoint import save_checkpoint
from utils.logger import CSVLogger
from utils.seed import seed_everything


def _cfg(cfg: Mapping[str, Any], section: str, key: str, default: Any) -> Any:
    return cfg.get(section, {}).get(key, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _should_log_round(round_id: int, rounds: int, log_every: int) -> bool:
    return round_id == 0 or (round_id + 1) % log_every == 0 or (round_id + 1) == rounds


def _should_save_interval(round_id: int, rounds: int, interval: int) -> bool:
    return (round_id + 1) == rounds or (interval > 0 and (round_id + 1) % interval == 0)


def _build_optimizer(model: torch.nn.Module, cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    name = str(_cfg(cfg, "optimizer", "name", "adam")).lower()
    lr = float(_cfg(cfg, "optimizer", "lr", 0.001))
    weight_decay = float(_cfg(cfg, "optimizer", "weight_decay", 0.0005))

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
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


def run_fedproto(
    config: Mapping[str, Any],
    debug: bool = False,
) -> Dict[str, Any]:
    """Run FedProto on CIFAR-10 with heterogeneous client models.

    Kaggle usage:
        from methods.fedproto import run_fedproto
        history = run_fedproto(config)
    """
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
    lambda_proto = float(
        config.get("fedproto", {}).get(
            "lambda_proto",
            _cfg(config, "fedcausal", "lambda_scl", 0.1),
        )
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

    clients: List[Client] = []
    for client_id in range(num_clients):
        model = build_model(
            model_name=client_model_names[client_id],
            num_classes=num_classes,
            feature_dim=feature_dim,
        )
        optimizer = _build_optimizer(model, config)
        clients.append(
            Client(
                client_id=client_id,
                model=model,
                train_loader=client_loaders[client_id],
                optimizer=optimizer,
                device=device,
                num_classes=num_classes,
                feature_dim=feature_dim,
                local_epochs=local_epochs,
                tau_s=tau_s,
                lambda_scl=lambda_proto,
                lambda_proto=lambda_proto,
            )
        )

    server = Server(num_classes=num_classes, feature_dim=feature_dim, device=device)

    result_dir = Path(_cfg(config, "output", "result_dir", "/kaggle/working/FedCausal/results"))
    checkpoint_dir = Path(
        _cfg(config, "output", "checkpoint_dir", "/kaggle/working/FedCausal/checkpoints")
    )
    output_cfg = config.get("output", {})
    log_every = max(1, int(output_cfg.get("log_every", 1)))
    log_client_metrics = _as_bool(output_cfg.get("log_client_metrics", True))
    save_checkpoints = _as_bool(output_cfg.get("save_checkpoints", True))
    checkpoint_interval = int(output_cfg.get("checkpoint_interval", 5))
    logger = CSVLogger(
        result_dir / "fedproto_clean_results.csv",
        fieldnames=["round", "method", "client_id", "local_loss", "local_acc", "clean_acc"],
        reset=True,
    )

    history: List[Dict[str, Any]] = []
    for round_id in range(rounds):
        should_log = _should_log_round(round_id, rounds, log_every)
        if should_log:
            print(f"\n[FedProto] Round {round_id + 1}/{rounds}")
        selected_client_ids = _select_clients(num_clients, participation_rate, round_id, seed)
        global_prototypes = server.get_global_prototypes()

        local_payloads = []
        train_metrics_by_client: Dict[int, Dict[str, float]] = {}
        for client_id in selected_client_ids:
            client = clients[client_id]
            train_metrics = client.train_one_round(global_prototypes)
            train_metrics_by_client[client_id] = train_metrics
            local_payloads.append(client.compute_local_prototypes())
            if should_log and log_client_metrics:
                print(
                    f"  client={client_id:02d} "
                    f"loss={train_metrics['local_loss']:.4f} "
                    f"local_acc={train_metrics['local_acc']:.4f}"
                )

        server.aggregate(local_payloads)

        clean_accs = []
        for client in clients:
            eval_metrics = client.evaluate(test_loader)
            clean_acc = eval_metrics["accuracy"]
            clean_accs.append(clean_acc)
            train_metrics = train_metrics_by_client.get(client.client_id, {})

            row = {
                "round": round_id,
                "method": "fedproto",
                "client_id": client.client_id,
                "local_loss": train_metrics.get("local_loss", ""),
                "local_acc": train_metrics.get("local_acc", ""),
                "clean_acc": clean_acc,
            }
            logger.log(row)
            history.append(row)

        mean_clean_acc = sum(clean_accs) / len(clean_accs) if clean_accs else 0.0
        if should_log:
            print(f"  mean_clean_acc={mean_clean_acc:.4f}")

        if save_checkpoints and _should_save_interval(round_id, rounds, checkpoint_interval):
            checkpoint_path = save_checkpoint(
                {
                    "round": round_id,
                    "method": "fedproto",
                    "config": dict(config),
                    "server": server.state_dict(),
                    "clients": [client.state_dict() for client in clients],
                },
                checkpoint_dir=checkpoint_dir,
                filename=f"fedproto_round_{round_id + 1}.pt",
            )
            print(f"  saved checkpoint: {checkpoint_path}")

    return {
        "history": history,
        "server": server,
        "clients": clients,
        "result_csv": result_dir / "fedproto_clean_results.csv",
    }
