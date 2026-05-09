# Kaggle one-cell launcher for FedProto long-round sanity check.
# Paste this whole file into one Kaggle Notebook cell after the project is
# available at /kaggle/working/FedCausal.

import csv
import os
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml


PROJECT_ROOT = Path("/kaggle/working/FedCausal")
if not PROJECT_ROOT.exists():
    raise FileNotFoundError(
        "FedCausal project directory not found at /kaggle/working/FedCausal. "
        "Upload or copy the project there before running this cell."
    )

os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Working directory:", Path.cwd())
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

from methods.fedproto import run_fedproto
from utils.seed import seed_everything


EXPERIMENT_NAME = "fedproto_long_sanity"
ROUNDS = 110
BATCH_SIZE = 32

config_path = PROJECT_ROOT / "configs" / "default_kaggle.yaml"
with config_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
config["seed"] = int(config.get("seed", 42))
config["federated"]["num_clients"] = 10
config["federated"]["rounds"] = ROUNDS
config["federated"]["local_epochs"] = 1
config["federated"]["batch_size"] = BATCH_SIZE
config["federated"]["partition_mode"] = "iid"
config["federated"]["participation_rate"] = 1.0

config["attack"]["type"] = "none"
config["attack"]["malicious_ratio"] = 0.0
config["corruption"]["enable_train_corruption"] = False
config["corruption"]["client_ratio"] = 0.0

config["model"]["client_models"] = [
    "cnn_small",
    "cnn_small",
    "cnn_small",
    "cnn_small",
    "resnet18",
    "resnet18",
    "resnet18",
    "mobilenetv2",
    "mobilenetv2",
    "mobilenetv2",
]

config["output"]["result_dir"] = str(PROJECT_ROOT / "results" / EXPERIMENT_NAME)
config["output"]["checkpoint_dir"] = str(PROJECT_ROOT / "checkpoints" / EXPERIMENT_NAME)
config["output"]["figure_dir"] = str(PROJECT_ROOT / "figures" / EXPERIMENT_NAME)

for key in ["result_dir", "checkpoint_dir", "figure_dir"]:
    Path(config["output"][key]).mkdir(parents=True, exist_ok=True)
tables_dir = PROJECT_ROOT / "tables" / EXPERIMENT_NAME
tables_dir.mkdir(parents=True, exist_ok=True)

seed_everything(config["seed"])
print("FedProto long sanity config:")
print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))

outputs = run_fedproto(config, debug=False)
history = outputs["history"]

round_to_accs = {}
for row in history:
    round_to_accs.setdefault(int(row["round"]), []).append(float(row["clean_acc"]))

curve_rows = []
for round_id in sorted(round_to_accs):
    accs = round_to_accs[round_id]
    curve_rows.append(
        {
            "round": round_id,
            "mean_clean_acc": sum(accs) / len(accs),
            "min_clean_acc": min(accs),
            "max_clean_acc": max(accs),
        }
    )

final_round = curve_rows[-1]["round"]
final_clean_acc = curve_rows[-1]["mean_clean_acc"]
best_row = max(curve_rows, key=lambda row: row["mean_clean_acc"])

summary = {
    "experiment": EXPERIMENT_NAME,
    "method": "fedproto",
    "rounds": ROUNDS,
    "batch_size": BATCH_SIZE,
    "partition_mode": "iid",
    "final_round": final_round,
    "final_clean_acc": final_clean_acc,
    "best_round": best_row["round"],
    "best_clean_acc": best_row["mean_clean_acc"],
    "result_csv": str(outputs["result_csv"]),
}

summary_csv = Path(config["output"]["result_dir"]) / "fedproto_long_clean_summary.csv"
with summary_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
    writer.writeheader()
    writer.writerow(summary)

curve_csv = Path(config["output"]["result_dir"]) / "fedproto_long_clean_curve.csv"
with curve_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["round", "mean_clean_acc", "min_clean_acc", "max_clean_acc"])
    writer.writeheader()
    writer.writerows(curve_rows)

summary_md = tables_dir / "fedproto_long_clean_summary.md"
summary_md.write_text(
    "\n".join(
        [
            "| experiment | method | rounds | batch_size | final_round | final_clean_acc | best_round | best_clean_acc |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
            (
                f"| {summary['experiment']} | {summary['method']} | {summary['rounds']} | "
                f"{summary['batch_size']} | {summary['final_round']} | {summary['final_clean_acc']:.6f} | "
                f"{summary['best_round']} | {summary['best_clean_acc']:.6f} |"
            ),
        ]
    )
    + "\n",
    encoding="utf-8",
)

figure_path = Path(config["output"]["figure_dir"]) / "fedproto_clean_accuracy_vs_round.png"
plt.figure(figsize=(8, 5))
plt.plot(
    [row["round"] + 1 for row in curve_rows],
    [row["mean_clean_acc"] for row in curve_rows],
    marker="o",
    linewidth=1.8,
    markersize=3,
)
plt.xlabel("Round")
plt.ylabel("Mean clean accuracy")
plt.title("FedProto Long Sanity: Clean Accuracy vs Round")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figure_path, dpi=160)
plt.close()

zip_path = Path("/kaggle/working/fedproto_long_sanity_outputs.zip")
include_folders = [
    PROJECT_ROOT / "results" / EXPERIMENT_NAME,
    PROJECT_ROOT / "tables" / EXPERIMENT_NAME,
    PROJECT_ROOT / "figures" / EXPERIMENT_NAME,
]

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for folder in include_folders:
        if not folder.exists():
            print("Skip missing folder:", folder)
            continue
        for path in folder.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(PROJECT_ROOT))

print("\nFinished FedProto long sanity.")
print("Final clean accuracy:", round(final_clean_acc, 6))
print("Best clean accuracy:", round(best_row["mean_clean_acc"], 6), "at round", best_row["round"] + 1)
print("Summary CSV:", summary_csv)
print("Curve CSV:", curve_csv)
print("Summary table:", summary_md)
print("Figure:", figure_path)
print("Download ZIP:", zip_path)
print("ZIP size MB:", round(zip_path.stat().st_size / (1024 * 1024), 2))
