"""Generate FedCausal analysis tables, figures, and summary text."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from utils.visualization import (
    plot_bar,
    plot_grouped_bars,
    plot_lines,
    plot_mask_heatmap,
)


PROJECT_ROOT = Path("/kaggle/working/FedCausal")
RESULT_FILES = [
    "fedproto_clean_results.csv",
    "aughfl_lite_results.csv",
    "rahfl_lite_results.csv",
    "fedcausal_results.csv",
    "fedcausal_mask_clean_results.csv",
    "fedcausal_mvp_clean_results.csv",
    "fedcausal_wo_inv_clean_results.csv",
    "corruption_results.csv",
    "attack_results.csv",
    "ablation_results.csv",
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        print(f"[skip] Missing CSV: {path}")
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def _write_markdown_table(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |\n")
    return path


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def _round_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _final_round_rows(rows: Iterable[Mapping[str, str]]) -> List[Mapping[str, str]]:
    rows = list(rows)
    if not rows:
        return []
    max_round = max(_int(row.get("round", 0)) for row in rows)
    return [row for row in rows if _int(row.get("round", 0)) == max_round]


def _method_from_filename(filename: str) -> str:
    mapping = {
        "fedproto_clean_results.csv": "fedproto",
        "aughfl_lite_results.csv": "aughfl_lite",
        "rahfl_lite_results.csv": "rahfl_lite",
        "fedcausal_results.csv": "fedcausal_mvp",
        "fedcausal_mask_clean_results.csv": "fedcausal_mask",
        "fedcausal_mvp_clean_results.csv": "fedcausal_mvp",
        "fedcausal_wo_inv_clean_results.csv": "fedcausal_wo_inv",
    }
    return mapping.get(filename, filename.replace("_results.csv", ""))


def _load_all_results(result_dir: Path) -> Dict[str, List[Dict[str, str]]]:
    return {filename: _read_csv(result_dir / filename) for filename in RESULT_FILES}


def build_table1_clean_accuracy(
    all_results: Mapping[str, List[Dict[str, str]]],
    alpha: float,
) -> List[Dict[str, Any]]:
    """Table 1: heterogeneous Non-IID clean accuracy."""
    clean_files = [
        "fedproto_clean_results.csv",
        "aughfl_lite_results.csv",
        "rahfl_lite_results.csv",
        "fedcausal_results.csv",
        "fedcausal_mask_clean_results.csv",
        "fedcausal_mvp_clean_results.csv",
        "fedcausal_wo_inv_clean_results.csv",
    ]
    table = []
    seen = set()
    for filename in clean_files:
        rows = all_results.get(filename, [])
        if not rows:
            continue
        final_rows = _final_round_rows(rows)
        method = final_rows[0].get("method") or _method_from_filename(filename)
        if method in seen:
            continue
        seen.add(method)
        accs = [_float(row.get("clean_acc")) for row in final_rows if row.get("clean_acc") not in {"", None}]
        if not accs:
            continue
        table.append(
            {
                "method": method,
                "alpha": alpha,
                "clean_acc_mean": _round_float(mean(accs)),
                "clean_acc_std": _round_float(pstdev(accs) if len(accs) > 1 else 0.0),
                "client_acc_std": _round_float(pstdev(accs) if len(accs) > 1 else 0.0),
            }
        )
    return sorted(table, key=lambda row: row["clean_acc_mean"], reverse=True)


def build_table2_corruption(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    """Table 2: common corruption robustness."""
    grouped: Dict[tuple[str, int], List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("method", ""), _int(row.get("severity", 0)))].append(row)

    table = []
    for (method, severity), group in grouped.items():
        if not method:
            continue
        table.append(
            {
                "method": method,
                "clean_acc": _round_float(mean(_float(row.get("clean_acc")) for row in group)),
                "mCA": _round_float(mean(_float(row.get("mCA")) for row in group)),
                "drop": _round_float(mean(_float(row.get("drop")) for row in group)),
                "severity": severity,
            }
        )
    return sorted(table, key=lambda row: (row["severity"], -row["mCA"]))


def build_table3_corrupted_clients(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    """Table 3: corrupted clients training."""
    grouped: Dict[tuple[str, float, str], List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        ratio = _float(row.get("train_corruption_ratio"))
        enabled = _bool(row.get("train_corruption_enabled")) or ratio > 0
        if not enabled:
            continue
        key = (row.get("method", ""), ratio, row.get("train_corruption_type", "none"))
        grouped[key].append(row)

    table = []
    for (method, ratio, corruption_type), group in grouped.items():
        table.append(
            {
                "method": method,
                "corrupted_client_ratio": ratio,
                "corruption_type": corruption_type,
                "clean_acc": _round_float(mean(_float(row.get("clean_acc")) for row in group)),
                "corrupt_acc": _round_float(mean(_float(row.get("corrupt_acc")) for row in group)),
                "drop": _round_float(mean(_float(row.get("drop")) for row in group)),
            }
        )
    return sorted(table, key=lambda row: (row["corrupted_client_ratio"], row["method"]))


def build_table4_attack(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    """Table 4: attack robustness."""
    final_rows = _final_round_rows(rows)
    grouped: Dict[tuple[str, str, float], List[Mapping[str, str]]] = defaultdict(list)
    for row in final_rows:
        key = (
            row.get("method", ""),
            row.get("attack_type", "none"),
            _float(row.get("malicious_ratio")),
        )
        grouped[key].append(row)

    table = []
    for (method, attack_type, ratio), group in grouped.items():
        if not method:
            continue
        mal_alphas = [_float(row.get("alpha_i")) for row in group if _bool(row.get("is_malicious"))]
        benign_alphas = [_float(row.get("alpha_i")) for row in group if not _bool(row.get("is_malicious"))]
        table.append(
            {
                "method": method,
                "attack_type": attack_type,
                "malicious_ratio": ratio,
                "clean_acc_under_attack": _round_float(mean(_float(row.get("clean_acc")) for row in group)),
                "alpha_mal": _round_float(mean(mal_alphas) if mal_alphas else 0.0),
                "alpha_benign": _round_float(mean(benign_alphas) if benign_alphas else 0.0),
            }
        )
    return sorted(table, key=lambda row: (row["attack_type"], row["malicious_ratio"], row["method"]))


def build_table5_ablation(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    """Table 5: ablation results."""
    grouped: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        variant = row.get("variant") or row.get("method") or row.get("ablation") or ""
        if variant:
            grouped[variant].append(row)

    table = []
    for variant, group in grouped.items():
        table.append(
            {
                "variant": variant,
                "clean_acc": _round_float(mean(_float(row.get("clean_acc")) for row in group)),
                "mCA": _round_float(mean(_float(row.get("mCA")) for row in group)),
                "drop": _round_float(mean(_float(row.get("drop")) for row in group)),
                "attack_acc": _round_float(
                    mean(_float(row.get("attack_acc", row.get("clean_acc_under_attack"))) for row in group)
                ),
            }
        )
    return sorted(table, key=lambda row: row["clean_acc"], reverse=True)


def _write_table_bundle(
    tables_dir: Path,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    _write_csv(tables_dir / f"{name}.csv", rows, fieldnames)
    _write_markdown_table(tables_dir / f"{name}.md", rows, fieldnames)


def _accuracy_series(all_results: Mapping[str, List[Dict[str, str]]]) -> Dict[str, List[tuple[float, float]]]:
    series = {}
    for filename, rows in all_results.items():
        if not rows or "clean_results" not in filename and filename not in {
            "fedcausal_results.csv",
            "aughfl_lite_results.csv",
            "rahfl_lite_results.csv",
        }:
            continue
        grouped: Dict[int, List[float]] = defaultdict(list)
        for row in rows:
            if row.get("clean_acc") in {"", None}:
                continue
            grouped[_int(row.get("round"))].append(_float(row.get("clean_acc")))
        if grouped:
            label = rows[0].get("method") or _method_from_filename(filename)
            series[label] = [(round_id, mean(values)) for round_id, values in grouped.items()]
    return series


def _alpha_series(rows: Sequence[Mapping[str, str]]) -> Dict[str, List[tuple[float, float]]]:
    grouped = defaultdict(lambda: {"malicious": [], "benign": []})
    for row in rows:
        bucket = "malicious" if _bool(row.get("is_malicious")) else "benign"
        grouped[_int(row.get("round"))][bucket].append(_float(row.get("alpha_i")))
    return {
        "malicious": [(r, mean(v["malicious"]) if v["malicious"] else 0.0) for r, v in grouped.items()],
        "benign": [(r, mean(v["benign"]) if v["benign"] else 0.0) for r, v in grouped.items()],
    }


def generate_figures(
    project_root: Path,
    figures_dir: Path,
    all_results: Mapping[str, List[Dict[str, str]]],
    table1: Sequence[Mapping[str, Any]],
    table2: Sequence[Mapping[str, Any]],
) -> List[Path]:
    """Generate requested paper figures."""
    generated = []
    path = plot_lines(
        _accuracy_series(all_results),
        figures_dir / "accuracy_vs_round.png",
        xlabel="round",
        ylabel="clean accuracy",
        title="Clean Accuracy vs Round",
    )
    if path:
        generated.append(path)

    if table2:
        mca_by_method = defaultdict(list)
        drop_by_method = defaultdict(list)
        for row in table2:
            mca_by_method[str(row["method"])].append(float(row["mCA"]))
            drop_by_method[str(row["method"])].append(float(row["drop"]))
        labels = sorted(mca_by_method)
        path = plot_bar(
            labels,
            [mean(mca_by_method[label]) for label in labels],
            figures_dir / "mca_bar_chart.png",
            ylabel="mCA",
            title="Common Corruption Robustness",
        )
        if path:
            generated.append(path)
        path = plot_bar(
            labels,
            [mean(drop_by_method[label]) for label in labels],
            figures_dir / "corruption_drop.png",
            ylabel="clean acc - mCA",
            title="Corruption Accuracy Drop",
        )
        if path:
            generated.append(path)

    attack_rows = all_results.get("attack_results.csv", [])
    if attack_rows:
        path = plot_lines(
            _alpha_series(attack_rows),
            figures_dir / "alpha_malicious_vs_round.png",
            xlabel="round",
            ylabel="mean alpha",
            title="Energy Aggregation Trust Weights",
        )
        if path:
            generated.append(path)

    mask_path = project_root / "checkpoints" / "global_mask.pt"
    path = plot_mask_heatmap(mask_path, figures_dir / "mask_heatmap.png")
    if path:
        generated.append(path)
    else:
        print(f"[skip] Missing or unreadable global mask: {mask_path}")

    if table1:
        labels = [str(row["method"]) for row in table1]
        values = [float(row["client_acc_std"]) for row in table1]
        path = plot_bar(
            labels,
            values,
            figures_dir / "client_accuracy_std.png",
            ylabel="client accuracy std",
            title="Client Accuracy Dispersion",
        )
        if path:
            generated.append(path)

    return generated


def _best_by(rows: Sequence[Mapping[str, Any]], key: str, higher_is_better: bool = True) -> Mapping[str, Any] | None:
    if not rows:
        return None
    return sorted(rows, key=lambda row: float(row.get(key, 0.0)), reverse=higher_is_better)[0]


def _method_metric(rows: Sequence[Mapping[str, Any]], method_contains: str, key: str) -> float | None:
    matches = [float(row[key]) for row in rows if method_contains.lower() in str(row.get("method", "")).lower()]
    return max(matches) if matches else None


def write_analysis_summary(
    path: Path,
    table1: Sequence[Mapping[str, Any]],
    table2: Sequence[Mapping[str, Any]],
    table4: Sequence[Mapping[str, Any]],
    table5: Sequence[Mapping[str, Any]],
) -> Path:
    """Write automatic text summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    best_clean = _best_by(table1, "clean_acc_mean", True)
    best_mca = _best_by(table2, "mCA", True)
    best_drop = _best_by(table2, "drop", False)

    fedproto_clean = _method_metric(table1, "fedproto", "clean_acc_mean")
    fedcausal_clean = _method_metric(table1, "fedcausal", "clean_acc_mean")
    aug_clean = _method_metric(table1, "aughfl", "clean_acc_mean")
    rahfl_clean = _method_metric(table1, "rahfl", "clean_acc_mean")

    fedproto_mca = _method_metric(table2, "fedproto", "mCA")
    fedcausal_mca = _method_metric(table2, "fedcausal", "mCA")
    aug_mca = _method_metric(table2, "aughfl", "mCA")
    rahfl_mca = _method_metric(table2, "rahfl", "mCA")

    alpha_success = "证据不足"
    if table4:
        gaps = [float(row["alpha_benign"]) - float(row["alpha_mal"]) for row in table4]
        alpha_success = "是" if mean(gaps) > 0 else "否"

    most_important = "证据不足"
    if table5:
        full_candidates = [
            float(row.get("clean_acc", 0.0))
            for row in table5
            if "full" in str(row.get("variant", "")).lower()
            or "fedcausal" in str(row.get("variant", "")).lower()
        ]
        full = max(full_candidates) if full_candidates else None
        if full is not None:
            drops = []
            for row in table5:
                variant = str(row.get("variant", ""))
                if "full" in variant.lower():
                    continue
                drops.append((full - float(row.get("clean_acc", 0.0)), variant))
            if drops:
                most_important = max(drops)[1]

    def verdict(lhs: float | None, rhs: float | None) -> str:
        if lhs is None or rhs is None:
            return "证据不足"
        return "是" if lhs > rhs else "否"

    with path.open("w", encoding="utf-8") as f:
        f.write("# FedCausal Analysis Summary\n\n")
        f.write(f"- clean acc 最高方法：{best_clean.get('method') if best_clean else '无数据'}\n")
        f.write(f"- mCA 最高方法：{best_mca.get('method') if best_mca else '无数据'}\n")
        f.write(f"- drop 最小方法：{best_drop.get('method') if best_drop else '无数据'}\n")
        f.write(f"- FedCausal clean acc 是否优于 FedProto：{verdict(fedcausal_clean, fedproto_clean)}\n")
        f.write(f"- FedCausal mCA 是否优于 FedProto：{verdict(fedcausal_mca, fedproto_mca)}\n")
        f.write(f"- FedCausal 是否优于 AugHFL-lite：{verdict(fedcausal_mca or fedcausal_clean, aug_mca or aug_clean)}\n")
        f.write(f"- FedCausal 是否优于 RAHFL-lite：{verdict(fedcausal_mca or fedcausal_clean, rahfl_mca or rahfl_clean)}\n")
        f.write(f"- energy aggregation 是否压低 malicious alpha：{alpha_success}\n")
        f.write(f"- 最重要的消融模块：{most_important}\n")
    return path


def analyze_results(
    project_root: str | Path = PROJECT_ROOT,
    alpha: float = 0.3,
) -> Dict[str, Any]:
    """Run the full analysis pipeline."""
    project_root = Path(project_root)
    result_dir = project_root / "results"
    figures_dir = project_root / "figures"
    tables_dir = project_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    all_results = _load_all_results(result_dir)
    table1 = build_table1_clean_accuracy(all_results, alpha=alpha)
    table2 = build_table2_corruption(all_results.get("corruption_results.csv", []))
    table3 = build_table3_corrupted_clients(all_results.get("corruption_results.csv", []))
    table4 = build_table4_attack(all_results.get("attack_results.csv", []))
    table5 = build_table5_ablation(all_results.get("ablation_results.csv", []))

    _write_table_bundle(
        tables_dir,
        "table1_clean_accuracy",
        table1,
        ["method", "alpha", "clean_acc_mean", "clean_acc_std", "client_acc_std"],
    )
    _write_table_bundle(
        tables_dir,
        "table2_corruption_robustness",
        table2,
        ["method", "clean_acc", "mCA", "drop", "severity"],
    )
    _write_table_bundle(
        tables_dir,
        "table3_corrupted_clients",
        table3,
        ["method", "corrupted_client_ratio", "corruption_type", "clean_acc", "corrupt_acc", "drop"],
    )
    _write_table_bundle(
        tables_dir,
        "table4_attack_robustness",
        table4,
        ["method", "attack_type", "malicious_ratio", "clean_acc_under_attack", "alpha_mal", "alpha_benign"],
    )
    _write_table_bundle(
        tables_dir,
        "table5_ablation",
        table5,
        ["variant", "clean_acc", "mCA", "drop", "attack_acc"],
    )

    figures = generate_figures(project_root, figures_dir, all_results, table1, table2)
    summary_path = write_analysis_summary(project_root / "analysis_summary.md", table1, table2, table4, table5)
    print(f"Saved tables to: {tables_dir}")
    print(f"Saved figures to: {figures_dir}")
    print(f"Saved summary to: {summary_path}")

    return {
        "tables": {
            "table1": table1,
            "table2": table2,
            "table3": table3,
            "table4": table4,
            "table5": table5,
        },
        "figures": figures,
        "summary": summary_path,
    }


if __name__ == "__main__":
    analyze_results()
