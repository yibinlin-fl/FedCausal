"""IID model-heterogeneity and corruption-robustness experiment launcher."""

from __future__ import annotations

import copy
from pathlib import Path
from statistics import pstdev
from typing import Any, Dict, Iterable, List, Mapping, Optional

from eval_cifar10c import _run_method, evaluate_method_on_cifar10c
from utils.logger import CSVLogger
from utils.seed import seed_everything


DEFAULT_METHODS = ["fedproto", "fedcausal_mask", "fedcausal_mvp"]
DEFAULT_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "motion_blur",
    "fog",
    "jpeg_compression",
]
DEFAULT_SEVERITIES = [1, 3, 5]


def _cfg(cfg: Mapping[str, Any], section: str, key: str, default: Any) -> Any:
    return cfg.get(section, {}).get(key, default)


def _ensure_dirs(config: Mapping[str, Any]) -> None:
    for section, key in [
        ("output", "result_dir"),
        ("output", "checkpoint_dir"),
        ("output", "figure_dir"),
    ]:
        Path(_cfg(config, section, key, ".")).mkdir(parents=True, exist_ok=True)


def _with_output_subdir(config: Mapping[str, Any], experiment_name: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(dict(config))
    output = cfg.setdefault("output", {})
    for key, folder in [
        ("result_dir", "results"),
        ("checkpoint_dir", "checkpoints"),
        ("figure_dir", "figures"),
    ]:
        base = Path(output.get(key, f"/kaggle/working/FedCausal/{folder}"))
        output[key] = str(base / experiment_name)
    _ensure_dirs(cfg)
    return cfg


def make_iid_no_attack_config(
    base_config: Mapping[str, Any],
    experiment_name: str,
    train_corruption_enabled: bool = False,
    train_corruption_type: str = "gaussian_noise",
    train_corruption_ratio: float = 0.0,
    train_corruption_severity: int = 3,
) -> Dict[str, Any]:
    """Return a config that isolates model heterogeneity from data heterogeneity."""
    cfg = _with_output_subdir(base_config, experiment_name)
    cfg.setdefault("federated", {})
    cfg["federated"]["partition_mode"] = "iid"
    cfg["federated"]["participation_rate"] = 1.0

    cfg.setdefault("attack", {})
    cfg["attack"]["type"] = "none"
    cfg["attack"]["malicious_ratio"] = 0.0

    cfg.setdefault("corruption", {})
    cfg["corruption"]["enable_train_corruption"] = bool(train_corruption_enabled)
    cfg["corruption"]["client_ratio"] = float(train_corruption_ratio)
    cfg["corruption"]["type"] = str(train_corruption_type)
    cfg["corruption"]["severity"] = int(train_corruption_severity)
    return cfg


def _round_clean_means(history: Iterable[Mapping[str, Any]]) -> Dict[int, List[float]]:
    rounds: Dict[int, List[float]] = {}
    for row in history:
        value = row.get("clean_acc", "")
        if value in ("", None):
            continue
        round_id = int(row["round"])
        rounds.setdefault(round_id, []).append(float(value))
    return rounds


def summarize_clean_history(
    method: str,
    outputs: Mapping[str, Any],
    experiment_name: str,
) -> Dict[str, Any]:
    """Summarize final and best clean accuracy from a method history."""
    history = outputs.get("history", [])
    round_accs = _round_clean_means(history)
    if not round_accs:
        return {
            "experiment": experiment_name,
            "method": method,
            "final_round": "",
            "final_clean_acc": "",
            "best_round": "",
            "best_clean_acc": "",
            "final_client_acc_std": "",
            "result_csv": outputs.get("result_csv", ""),
        }

    final_round = max(round_accs)
    final_accs = round_accs[final_round]
    best_round, best_accs = max(
        round_accs.items(),
        key=lambda item: sum(item[1]) / len(item[1]) if item[1] else 0.0,
    )
    return {
        "experiment": experiment_name,
        "method": method,
        "final_round": final_round,
        "final_clean_acc": sum(final_accs) / len(final_accs),
        "best_round": best_round,
        "best_clean_acc": sum(best_accs) / len(best_accs),
        "final_client_acc_std": pstdev(final_accs) if len(final_accs) > 1 else 0.0,
        "result_csv": outputs.get("result_csv", ""),
    }


def _write_clean_summary(
    rows: List[Mapping[str, Any]],
    path: str | Path,
    reset: bool = True,
) -> Path:
    path = Path(path)
    logger = CSVLogger(
        path,
        fieldnames=[
            "experiment",
            "method",
            "final_round",
            "final_clean_acc",
            "best_round",
            "best_clean_acc",
            "final_client_acc_std",
            "result_csv",
        ],
        reset=reset,
    )
    logger.log_many(rows)
    return path


def run_clean_iid_model_heterogeneity(
    config: Mapping[str, Any],
    methods: Optional[List[str]] = None,
    debug: bool = False,
    experiment_name: str = "exp1_iid_clean_train_clean_test",
) -> Dict[str, Any]:
    """Experiment 1: IID data, no attack, model-heterogeneous clean test."""
    methods = methods or DEFAULT_METHODS
    cfg = make_iid_no_attack_config(config, experiment_name)
    seed_everything(int(cfg.get("seed", 42)))

    clean_rows: List[Dict[str, Any]] = []
    outputs_by_method: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        print(f"\n[Experiment 1] Training {method}")
        method_outputs = _run_method(method, cfg, debug=debug)
        outputs_by_method[method_outputs["method_name"]] = method_outputs
        clean_rows.append(
            summarize_clean_history(
                method=method_outputs["method_name"],
                outputs=method_outputs,
                experiment_name=experiment_name,
            )
        )

    result_dir = Path(_cfg(cfg, "output", "result_dir", "results"))
    clean_summary_csv = _write_clean_summary(
        clean_rows,
        result_dir / "exp1_iid_clean_summary.csv",
    )
    return {
        "config": cfg,
        "outputs": outputs_by_method,
        "clean_rows": clean_rows,
        "clean_summary_csv": clean_summary_csv,
    }


def run_clean_iid_cifar10c_robustness(
    config: Mapping[str, Any],
    methods: Optional[List[str]] = None,
    corruptions: Optional[List[str]] = None,
    severities: Optional[List[int]] = None,
    debug: bool = False,
    experiment_name: str = "exp2_iid_clean_train_cifar10c_test",
    train_corruption_enabled: bool = False,
    train_corruption_type: str = "gaussian_noise",
    train_corruption_ratio: float = 0.0,
    train_corruption_severity: int = 3,
) -> Dict[str, Any]:
    """Experiment 2: IID clean training, no attack, CIFAR-10-C evaluation."""
    methods = methods or DEFAULT_METHODS
    corruptions = corruptions or DEFAULT_CORRUPTIONS
    severities = severities or DEFAULT_SEVERITIES
    cfg = make_iid_no_attack_config(
        config,
        experiment_name,
        train_corruption_enabled=train_corruption_enabled,
        train_corruption_type=train_corruption_type,
        train_corruption_ratio=train_corruption_ratio,
        train_corruption_severity=train_corruption_severity,
    )
    seed_everything(int(cfg.get("seed", 42)))

    clean_rows: List[Dict[str, Any]] = []
    corruption_rows: List[Dict[str, Any]] = []
    outputs_by_method: Dict[str, Dict[str, Any]] = {}
    rounds = int(_cfg(cfg, "federated", "rounds", 20))
    if debug:
        rounds = min(rounds, 3)
    final_round = rounds - 1

    result_dir = Path(_cfg(cfg, "output", "result_dir", "results"))
    corruption_csv = result_dir / f"{experiment_name}_cifar10c_summary.csv"
    corruption_logger = CSVLogger(
        corruption_csv,
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

    for method in methods:
        print(f"\n[Experiment 2] Training {method}")
        method_outputs = _run_method(method, cfg, debug=debug)
        method_name = method_outputs["method_name"]
        outputs_by_method[method_name] = method_outputs
        clean_rows.append(
            summarize_clean_history(
                method=method_name,
                outputs=method_outputs,
                experiment_name=experiment_name,
            )
        )

        rows = evaluate_method_on_cifar10c(
            method=method_name,
            outputs=method_outputs,
            config=cfg,
            round_id=final_round,
            corruptions=corruptions,
            severities=severities,
        )
        corruption_logger.log_many(rows)
        corruption_rows.extend(rows)

    clean_summary_csv = _write_clean_summary(
        clean_rows,
        result_dir / f"{experiment_name}_clean_summary.csv",
    )
    return {
        "config": cfg,
        "outputs": outputs_by_method,
        "clean_rows": clean_rows,
        "corruption_rows": corruption_rows,
        "clean_summary_csv": clean_summary_csv,
        "corruption_summary_csv": corruption_csv,
    }


def run_clean_iid_experiments_1_and_2(
    config: Mapping[str, Any],
    methods: Optional[List[str]] = None,
    corruptions: Optional[List[str]] = None,
    severities: Optional[List[int]] = None,
    debug: bool = False,
    experiment_name: str = "exp1_2_iid_clean_train",
) -> Dict[str, Any]:
    """Train once and emit both experiment 1 and experiment 2 summaries."""
    methods = methods or DEFAULT_METHODS
    corruptions = corruptions or DEFAULT_CORRUPTIONS
    severities = severities or DEFAULT_SEVERITIES
    cfg = make_iid_no_attack_config(config, experiment_name)
    seed_everything(int(cfg.get("seed", 42)))

    result_dir = Path(_cfg(cfg, "output", "result_dir", "results"))
    corruption_csv = result_dir / "exp2_iid_cifar10c_summary.csv"
    corruption_logger = CSVLogger(
        corruption_csv,
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

    rounds = int(_cfg(cfg, "federated", "rounds", 20))
    if debug:
        rounds = min(rounds, 3)
    final_round = rounds - 1

    clean_rows: List[Dict[str, Any]] = []
    corruption_rows: List[Dict[str, Any]] = []
    outputs_by_method: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        print(f"\n[Experiments 1+2] Training {method}")
        method_outputs = _run_method(method, cfg, debug=debug)
        method_name = method_outputs["method_name"]
        outputs_by_method[method_name] = method_outputs
        clean_rows.append(
            summarize_clean_history(
                method=method_name,
                outputs=method_outputs,
                experiment_name="exp1_iid_clean_train_clean_test",
            )
        )

        rows = evaluate_method_on_cifar10c(
            method=method_name,
            outputs=method_outputs,
            config=cfg,
            round_id=final_round,
            corruptions=corruptions,
            severities=severities,
        )
        corruption_logger.log_many(rows)
        corruption_rows.extend(rows)

    clean_summary_csv = _write_clean_summary(
        clean_rows,
        result_dir / "exp1_iid_clean_summary.csv",
    )
    return {
        "config": cfg,
        "outputs": outputs_by_method,
        "clean_rows": clean_rows,
        "corruption_rows": corruption_rows,
        "clean_summary_csv": clean_summary_csv,
        "corruption_summary_csv": corruption_csv,
    }


def run_four_experiment_suite(
    config: Mapping[str, Any],
    methods: Optional[List[str]] = None,
    corruptions: Optional[List[str]] = None,
    severities: Optional[List[int]] = None,
    debug: bool = False,
    run_corrupted_client_experiments: bool = False,
) -> Dict[str, Any]:
    """Kaggle-friendly launcher for the four planned IID experiments.

    Experiments 1 and 2 are implemented and run by default. Experiments 3 and 4
    use the same IID/no-attack setup but enable train-time corrupted clients;
    leave them disabled until you are ready for the longer run.
    """
    methods = methods or DEFAULT_METHODS
    results: Dict[str, Any] = {}
    results["exp1_2"] = run_clean_iid_experiments_1_and_2(
        config=config,
        methods=methods,
        corruptions=corruptions,
        severities=severities,
        debug=debug,
    )

    if not run_corrupted_client_experiments:
        return results

    corrupted_specs = [
        ("exp3_iid_gaussian_corrupted_clients", "gaussian_noise"),
        ("exp4_iid_jpeg_corrupted_clients", "jpeg_compression"),
    ]
    for experiment_name, corruption_type in corrupted_specs:
        results[experiment_name] = run_clean_iid_cifar10c_robustness(
            config=config,
            methods=methods,
            corruptions=corruptions,
            severities=severities,
            debug=debug,
            experiment_name=experiment_name,
            train_corruption_enabled=True,
            train_corruption_ratio=0.3,
            train_corruption_type=corruption_type,
            train_corruption_severity=3,
        )

    return results
