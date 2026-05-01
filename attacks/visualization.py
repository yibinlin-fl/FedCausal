"""Attack-result visualization helpers."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _read_rows(csv_path: str | Path) -> list[dict[str, str]]:
    path = Path(csv_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def save_attack_visualizations(
    attack_csv_path: str | Path,
    figure_dir: str | Path,
    method: str,
    attack_type: str,
    malicious_ratio: float,
    scale_factor: float,
) -> list[Path]:
    """Save attack diagnostic plots from ``attack_results.csv``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_rows(attack_csv_path)
    if not rows:
        return []

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    alpha_by_round = defaultdict(lambda: {"mal": [], "benign": []})
    acc_by_round = defaultdict(list)
    for row in rows:
        round_id = int(row["round"])
        alpha = float(row["alpha_i"]) if row["alpha_i"] else 0.0
        clean_acc = float(row["clean_acc"]) if row["clean_acc"] else 0.0
        is_malicious = str(row["is_malicious"]).lower() in {"true", "1", "yes"}
        key = "mal" if is_malicious else "benign"
        alpha_by_round[round_id][key].append(alpha)
        acc_by_round[round_id].append(clean_acc)

    rounds = sorted(alpha_by_round)
    alpha_mal = [_mean(alpha_by_round[r]["mal"]) for r in rounds]
    alpha_benign = [_mean(alpha_by_round[r]["benign"]) for r in rounds]

    plt.figure(figsize=(6, 4))
    plt.plot(rounds, alpha_mal, marker="o", label="malicious")
    plt.plot(rounds, alpha_benign, marker="o", label="benign")
    plt.xlabel("round")
    plt.ylabel("mean alpha")
    plt.title(f"{method}: trust weights")
    plt.legend()
    plt.tight_layout()
    path = figure_dir / f"{method}_{attack_type}_alpha_by_round.png"
    plt.savefig(path, dpi=160)
    plt.close()
    saved.append(path)

    final_round = max(acc_by_round)
    final_acc = _mean(acc_by_round[final_round])

    plt.figure(figsize=(5, 4))
    plt.scatter([malicious_ratio], [final_acc])
    plt.xlabel("malicious ratio")
    plt.ylabel("clean accuracy")
    plt.title(f"{method}: attack ratio vs accuracy")
    plt.xlim(0.0, 1.0)
    plt.tight_layout()
    path = figure_dir / f"{method}_{attack_type}_ratio_vs_accuracy.png"
    plt.savefig(path, dpi=160)
    plt.close()
    saved.append(path)

    plt.figure(figsize=(5, 4))
    plt.scatter([scale_factor], [final_acc])
    plt.xlabel("scale factor")
    plt.ylabel("clean accuracy")
    plt.title(f"{method}: scale factor vs accuracy")
    plt.tight_layout()
    path = figure_dir / f"{method}_{attack_type}_scale_vs_accuracy.png"
    plt.savefig(path, dpi=160)
    plt.close()
    saved.append(path)

    return saved
