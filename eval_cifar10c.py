"""CIFAR-10-C and corrupted-client evaluation entry points."""

from __future__ import annotations

import copy
from pathlib import Path
from statistics import pstdev
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
from torch.utils.data import DataLoader

from data.cifar import get_cifar10_datasets
from data.cifar10c import (
    CIFAR10C_MISSING_MESSAGE,
    SUPPORTED_CIFAR10C_CORRUPTIONS,
    SUPPORTED_CIFAR10C_SEVERITIES,
    build_cifar10c_loader,
    cifar10c_available,
)
from methods.fedcausal_mask import run_fedcausal_mask
from methods.fedcausal_mvp import run_fedcausal_mvp
from methods.fedproto import run_fedproto
from utils.logger import CSVLogger


METHOD_ALIASES = {
    "fedproto": "fedproto",
    "fedcausal_wo_mask": "fedcausal_wo_mask",
    "fedcausal_without_mask": "fedcausal_wo_mask",
    "fedcausal_wo_inv": "fedcausal_wo_inv",
    "fedcausal_without_inv": "fedcausal_wo_inv",
    "fedcausal_mask": "fedcausal_mask",
    "fedcausal_mvp": "fedcausal_mvp",
}


def _cfg(cfg: Mapping[str, Any], section: str, key: str, default: Any) -> Any:
    return cfg.get(section, {}).get(key, default)


def _build_clean_test_loader(config: Mapping[str, Any]) -> DataLoader:
    _, test_dataset = get_cifar10_datasets(
        data_root=_cfg(config, "dataset", "data_root", "/kaggle/working/data"),
        download=True,
    )
    return DataLoader(
        test_dataset,
        batch_size=int(_cfg(config, "federated", "batch_size", 64)),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )


def _run_method(method: str, config: Mapping[str, Any], debug: bool) -> Dict[str, Any]:
    normalized = METHOD_ALIASES.get(method.lower())
    if normalized is None:
        available = ", ".join(sorted(METHOD_ALIASES))
        raise ValueError(f"Unknown method={method!r}. Available: {available}")

    if normalized == "fedproto":
        outputs = run_fedproto(config, debug=debug)
    elif normalized == "fedcausal_wo_mask":
        outputs = run_fedproto(config, debug=debug)
    elif normalized == "fedcausal_wo_inv":
        outputs = run_fedcausal_mvp(config, debug=debug, disable_inv=True)
    elif normalized == "fedcausal_mvp":
        outputs = run_fedcausal_mvp(config, debug=debug, disable_inv=False)
    elif normalized == "fedcausal_mask":
        outputs = run_fedcausal_mask(config, debug=debug)
    else:
        raise ValueError(f"Unhandled method={normalized!r}")

    outputs["method_name"] = normalized
    return outputs


@torch.no_grad()
def _evaluate_clients(
    clients: Iterable[Any],
    loader: DataLoader,
) -> Dict[str, float]:
    accs = []
    for client in clients:
        metrics = client.evaluate(loader)
        accs.append(float(metrics["accuracy"]))

    if not accs:
        return {"mean_acc": 0.0, "client_acc_std": 0.0}

    return {
        "mean_acc": sum(accs) / len(accs),
        "client_acc_std": pstdev(accs) if len(accs) > 1 else 0.0,
    }


def evaluate_method_on_cifar10c(
    method: str,
    outputs: Mapping[str, Any],
    config: Mapping[str, Any],
    round_id: int,
    corruptions: Optional[List[str]] = None,
    severities: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate trained client models on clean CIFAR-10 and CIFAR-10-C."""
    root = _cfg(config, "dataset", "cifar10c_root", "/kaggle/input/cifar10-c")
    if not cifar10c_available(root):
        print(CIFAR10C_MISSING_MESSAGE)
        return []

    corruptions = corruptions or SUPPORTED_CIFAR10C_CORRUPTIONS
    severities = severities or SUPPORTED_CIFAR10C_SEVERITIES
    batch_size = int(_cfg(config, "federated", "batch_size", 64))
    clients = outputs["clients"]

    clean_loader = _build_clean_test_loader(config)
    clean_metrics = _evaluate_clients(clients, clean_loader)
    clean_acc = clean_metrics["mean_acc"]
    client_acc_std = clean_metrics["client_acc_std"]

    train_corruption_cfg = config.get("corruption", {})
    train_enabled = bool(train_corruption_cfg.get("enable_train_corruption", False))
    train_type = str(train_corruption_cfg.get("type", "none")) if train_enabled else "none"
    train_ratio = float(train_corruption_cfg.get("client_ratio", 0.0)) if train_enabled else 0.0

    rows: List[Dict[str, Any]] = []
    for severity in severities:
        severity_rows = []
        for corruption in corruptions:
            loader = build_cifar10c_loader(
                root=root,
                corruption=corruption,
                severity=severity,
                batch_size=batch_size,
                num_workers=2,
                pin_memory=True,
            )
            if loader is None:
                return rows

            corrupt_metrics = _evaluate_clients(clients, loader)
            severity_rows.append(
                {
                    "method": method,
                    "round": round_id,
                    "train_corruption_enabled": train_enabled,
                    "train_corruption_type": train_type,
                    "train_corruption_ratio": train_ratio,
                    "test_corruption_type": corruption,
                    "severity": severity,
                    "clean_acc": clean_acc,
                    "corrupt_acc": corrupt_metrics["mean_acc"],
                    "client_acc_std": client_acc_std,
                }
            )

        mca = sum(row["corrupt_acc"] for row in severity_rows) / len(severity_rows)
        drop = clean_acc - mca
        for row in severity_rows:
            row["mCA"] = mca
            row["drop"] = drop
            rows.append(row)

    return rows


def run_corruption_experiments(
    config: Mapping[str, Any],
    methods: Optional[List[str]] = None,
    corruptions: Optional[List[str]] = None,
    severities: Optional[List[int]] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """Train selected methods and evaluate clean/corrupt robustness.

    This function supports clean-train/corrupt-test and corrupted-client train
    settings through the ``corruption`` section in the config.
    """
    methods = methods or ["fedproto", "fedcausal_wo_mask", "fedcausal_wo_inv", "fedcausal_mvp"]
    result_dir = Path(_cfg(config, "output", "result_dir", "/kaggle/working/FedCausal/results"))
    result_csv = result_dir / "corruption_results.csv"
    logger = CSVLogger(
        result_csv,
        fieldnames=[
            "method",
            "round",
            "train_corruption_enabled",
            "train_corruption_type",
            "train_corruption_ratio",
            "test_corruption_type",
            "severity",
            "clean_acc",
            "corrupt_acc",
            "mCA",
            "drop",
            "client_acc_std",
        ],
        reset=True,
    )

    all_rows: List[Dict[str, Any]] = []
    trained_outputs: Dict[str, Dict[str, Any]] = {}
    rounds = int(_cfg(config, "federated", "rounds", 20))
    if debug:
        rounds = min(rounds, 3)
    final_round = rounds - 1

    for method in methods:
        method_config = copy.deepcopy(dict(config))
        normalized = METHOD_ALIASES.get(method.lower(), method)
        print(f"\n[Corruption Eval] Training method: {normalized}")
        outputs = _run_method(normalized, method_config, debug=debug)
        trained_outputs[normalized] = outputs

        rows = evaluate_method_on_cifar10c(
            method=normalized,
            outputs=outputs,
            config=method_config,
            round_id=final_round,
            corruptions=corruptions,
            severities=severities,
        )
        for row in rows:
            logger.log(row)
        all_rows.extend(rows)

    return {
        "rows": all_rows,
        "outputs": trained_outputs,
        "result_csv": result_csv,
    }


if __name__ == "__main__":
    import yaml

    with open("configs/default_kaggle.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_corruption_experiments(cfg, methods=["fedproto"], debug=True)
