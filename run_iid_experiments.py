"""IID model-heterogeneity and corruption-robustness experiment launcher."""

from __future__ import annotations

import copy
import csv
import shutil
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


def _tables_dir_for_result_dir(result_dir: str | Path) -> Path:
    result_dir = Path(result_dir)
    project_root = result_dir.parent.parent if result_dir.parent.name == "results" else result_dir.parent
    table_dir = project_root / "tables" / result_dir.name
    table_dir.mkdir(parents=True, exist_ok=True)
    return table_dir


def _format_markdown_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown_table(rows: List[Mapping[str, Any]], path: str | Path) -> Path:
    """Write a small Markdown table for quick Kaggle inspection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("| empty |\n| --- |\n", encoding="utf-8")
        return path

    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_markdown_value(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _copy_method_artifacts(method_name: str, outputs: Mapping[str, Any]) -> Dict[str, Path]:
    """Keep method-specific copies of generic CSV/checkpoint artifacts."""
    copied: Dict[str, Path] = {}
    for key, suffix in [
        ("attack_csv", "aggregation_weights"),
        ("global_mask_path", "global_mask"),
    ]:
        source = outputs.get(key)
        if not source:
            continue
        source_path = Path(source)
        if not source_path.exists():
            continue
        destination = source_path.with_name(f"{method_name}_{suffix}{source_path.suffix}")
        shutil.copy2(source_path, destination)
        copied[key] = destination
    return copied


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


def _load_corrupted_client_ids(result_dir: str | Path) -> List[int]:
    path = Path(result_dir) / "corrupted_client_ids.csv"
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return sorted(int(row["client_id"]) for row in reader if row.get("client_id") not in ("", None))


def _mean(values: List[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def summarize_corrupted_client_trust(
    method: str,
    aggregation_csv: str | Path,
    corrupted_client_ids: List[int],
    experiment_name: str,
) -> Dict[str, Any]:
    """Summarize final-round trust weights for corrupted vs clean clients."""
    aggregation_csv = Path(aggregation_csv)
    if not aggregation_csv.exists():
        return {
            "experiment": experiment_name,
            "method": method,
            "final_round": "",
            "num_corrupted_clients": len(corrupted_client_ids),
            "alpha_corrupted": "",
            "alpha_clean": "",
            "alpha_ratio_corrupted_to_clean": "",
            "energy_corrupted": "",
            "energy_clean": "",
            "aggregation_csv": aggregation_csv,
        }

    with aggregation_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    valid_rows = [row for row in rows if row.get("round") not in ("", None)]
    if not valid_rows:
        return {
            "experiment": experiment_name,
            "method": method,
            "final_round": "",
            "num_corrupted_clients": len(corrupted_client_ids),
            "alpha_corrupted": "",
            "alpha_clean": "",
            "alpha_ratio_corrupted_to_clean": "",
            "energy_corrupted": "",
            "energy_clean": "",
            "aggregation_csv": aggregation_csv,
        }

    final_round = max(int(row["round"]) for row in valid_rows)
    final_rows = [row for row in valid_rows if int(row["round"]) == final_round]
    corrupted_set = set(corrupted_client_ids)
    corrupted_rows = [row for row in final_rows if int(row["client_id"]) in corrupted_set]
    clean_rows = [row for row in final_rows if int(row["client_id"]) not in corrupted_set]

    alpha_corrupted = _mean([float(row["alpha_i"]) for row in corrupted_rows if row.get("alpha_i") not in ("", None)])
    alpha_clean = _mean([float(row["alpha_i"]) for row in clean_rows if row.get("alpha_i") not in ("", None)])
    energy_corrupted = _mean([float(row["energy"]) for row in corrupted_rows if row.get("energy") not in ("", None)])
    energy_clean = _mean([float(row["energy"]) for row in clean_rows if row.get("energy") not in ("", None)])
    alpha_ratio: float | str = ""
    if isinstance(alpha_corrupted, float) and isinstance(alpha_clean, float) and alpha_clean != 0.0:
        alpha_ratio = alpha_corrupted / alpha_clean

    return {
        "experiment": experiment_name,
        "method": method,
        "final_round": final_round,
        "num_corrupted_clients": len(corrupted_client_ids),
        "alpha_corrupted": alpha_corrupted,
        "alpha_clean": alpha_clean,
        "alpha_ratio_corrupted_to_clean": alpha_ratio,
        "energy_corrupted": energy_corrupted,
        "energy_clean": energy_clean,
        "aggregation_csv": aggregation_csv,
    }


def _write_corrupted_client_trust_summary(
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
            "num_corrupted_clients",
            "alpha_corrupted",
            "alpha_clean",
            "alpha_ratio_corrupted_to_clean",
            "energy_corrupted",
            "energy_clean",
            "aggregation_csv",
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
        _copy_method_artifacts(method_outputs["method_name"], method_outputs)
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
    table_dir = _tables_dir_for_result_dir(result_dir)
    clean_summary_md = write_markdown_table(clean_rows, table_dir / "exp1_iid_clean_summary.md")
    return {
        "config": cfg,
        "outputs": outputs_by_method,
        "clean_rows": clean_rows,
        "clean_summary_csv": clean_summary_csv,
        "clean_summary_md": clean_summary_md,
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
    trust_rows: List[Dict[str, Any]] = []
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
        copied_artifacts = _copy_method_artifacts(method_name, method_outputs)
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

        aggregation_csv = copied_artifacts.get("attack_csv")
        if train_corruption_enabled and aggregation_csv is not None:
            corrupted_client_ids = _load_corrupted_client_ids(result_dir)
            trust_rows.append(
                summarize_corrupted_client_trust(
                    method=method_name,
                    aggregation_csv=aggregation_csv,
                    corrupted_client_ids=corrupted_client_ids,
                    experiment_name=experiment_name,
                )
            )

    clean_summary_csv = _write_clean_summary(
        clean_rows,
        result_dir / f"{experiment_name}_clean_summary.csv",
    )
    table_dir = _tables_dir_for_result_dir(result_dir)
    clean_summary_md = write_markdown_table(
        clean_rows,
        table_dir / f"{experiment_name}_clean_summary.md",
    )
    corruption_summary_md = write_markdown_table(
        corruption_rows,
        table_dir / f"{experiment_name}_cifar10c_summary.md",
    )
    trust_summary_csv = None
    trust_summary_md = None
    if train_corruption_enabled:
        trust_summary_csv = _write_corrupted_client_trust_summary(
            trust_rows,
            result_dir / f"{experiment_name}_trust_summary.csv",
        )
        trust_summary_md = write_markdown_table(
            trust_rows,
            table_dir / f"{experiment_name}_trust_summary.md",
        )
    return {
        "config": cfg,
        "outputs": outputs_by_method,
        "clean_rows": clean_rows,
        "corruption_rows": corruption_rows,
        "trust_rows": trust_rows,
        "clean_summary_csv": clean_summary_csv,
        "corruption_summary_csv": corruption_csv,
        "trust_summary_csv": trust_summary_csv,
        "clean_summary_md": clean_summary_md,
        "corruption_summary_md": corruption_summary_md,
        "trust_summary_md": trust_summary_md,
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
        _copy_method_artifacts(method_name, method_outputs)
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
    table_dir = _tables_dir_for_result_dir(result_dir)
    clean_summary_md = write_markdown_table(
        clean_rows,
        table_dir / "exp1_iid_clean_summary.md",
    )
    corruption_summary_md = write_markdown_table(
        corruption_rows,
        table_dir / "exp2_iid_cifar10c_summary.md",
    )
    return {
        "config": cfg,
        "outputs": outputs_by_method,
        "clean_rows": clean_rows,
        "corruption_rows": corruption_rows,
        "clean_summary_csv": clean_summary_csv,
        "corruption_summary_csv": corruption_csv,
        "clean_summary_md": clean_summary_md,
        "corruption_summary_md": corruption_summary_md,
    }


def run_iid_corrupted_client_clean_and_cifar10c_test(
    config: Mapping[str, Any],
    methods: Optional[List[str]] = None,
    corruptions: Optional[List[str]] = None,
    severities: Optional[List[int]] = None,
    debug: bool = False,
    experiment_name: str = "exp3_iid_corrupted_clients_clean_cifar10c_test",
    train_corruption_type: str = "gaussian_noise",
    train_corruption_ratio: float = 0.3,
    train_corruption_severity: int = 3,
) -> Dict[str, Any]:
    """Experiment 3: IID data with partially corrupted clients, no attack."""
    return run_clean_iid_cifar10c_robustness(
        config=config,
        methods=methods,
        corruptions=corruptions,
        severities=severities,
        debug=debug,
        experiment_name=experiment_name,
        train_corruption_enabled=True,
        train_corruption_type=train_corruption_type,
        train_corruption_ratio=train_corruption_ratio,
        train_corruption_severity=train_corruption_severity,
    )


def run_three_experiment_suite(
    config: Mapping[str, Any],
    methods: Optional[List[str]] = None,
    corruptions: Optional[List[str]] = None,
    severities: Optional[List[int]] = None,
    debug: bool = False,
    run_experiment_3: bool = True,
    train_corruption_type: str = "gaussian_noise",
    train_corruption_ratio: float = 0.3,
    train_corruption_severity: int = 3,
) -> Dict[str, Any]:
    """Run Experiments 1, 2, and optionally 3 in Kaggle."""
    methods = methods or DEFAULT_METHODS
    results: Dict[str, Any] = {}
    results["exp1_2"] = run_clean_iid_experiments_1_and_2(
        config=config,
        methods=methods,
        corruptions=corruptions,
        severities=severities,
        debug=debug,
    )

    if run_experiment_3:
        results["exp3"] = run_iid_corrupted_client_clean_and_cifar10c_test(
            config=config,
            methods=methods,
            corruptions=corruptions,
            severities=severities,
            debug=debug,
            train_corruption_type=train_corruption_type,
            train_corruption_ratio=train_corruption_ratio,
            train_corruption_severity=train_corruption_severity,
        )

    return results


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
