# Kaggle one-cell launcher for Experiment 3.
# Paste this whole file into one Kaggle Notebook cell after the project is
# available at /kaggle/working/FedCausal.

import os
import sys
import zipfile
from pathlib import Path

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
print(
    "CIFAR-10-C: if it is not attached under /kaggle/input, the evaluation code "
    "will try to download it to /kaggle/temp. Enable Internet in Kaggle for that path."
)

config_path = PROJECT_ROOT / "configs" / "default_kaggle.yaml"
with config_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
config["federated"]["num_clients"] = 10
config["federated"]["rounds"] = 130
config["federated"]["local_epochs"] = 1
config["federated"]["batch_size"] = 64
config["federated"]["partition_mode"] = "iid"
config["federated"]["participation_rate"] = 1.0
config["attack"]["type"] = "none"
config["attack"]["malicious_ratio"] = 0.0

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

config.setdefault("output", {})
config["output"]["checkpoint_dir"] = "/kaggle/temp/FedCausal/checkpoints"
config["output"]["log_every"] = 10
config["output"]["log_client_metrics"] = False
config["output"]["save_checkpoints"] = False
config["output"]["checkpoint_interval"] = 0
config["output"]["save_mask_heatmaps"] = True
config["output"]["heatmap_final_only"] = True
config["output"]["heatmap_interval"] = 0

METHODS = ["fedproto", "fedcausal_mask", "fedcausal_mvp"]
CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "motion_blur",
    "fog",
    "jpeg_compression",
]
SEVERITIES = [3, 5]
TRAIN_CORRUPTION_TYPE = "gaussian_noise"
TRAIN_CORRUPTION_RATIO = 0.3
TRAIN_CORRUPTION_SEVERITY = 3
DEBUG = False

from run_iid_experiments import run_iid_corrupted_client_clean_and_cifar10c_test

results = {
    "exp3": run_iid_corrupted_client_clean_and_cifar10c_test(
        config=config,
        methods=METHODS,
        corruptions=CORRUPTIONS,
        severities=SEVERITIES,
        debug=DEBUG,
        train_corruption_type=TRAIN_CORRUPTION_TYPE,
        train_corruption_ratio=TRAIN_CORRUPTION_RATIO,
        train_corruption_severity=TRAIN_CORRUPTION_SEVERITY,
    )
}

print("\nFinished Experiment 3.")
for name, output in results.items():
    print(f"\n{name}")
    for key in [
        "clean_summary_csv",
        "corruption_summary_csv",
        "trust_summary_csv",
        "clean_summary_md",
        "corruption_summary_md",
        "trust_summary_md",
    ]:
        value = output.get(key)
        if value:
            print(f"  {key}: {value}")

zip_path = Path("/kaggle/working/fedcausal_exp3_130rounds_outputs.zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for folder_name in ("results", "tables", "figures"):
        folder = PROJECT_ROOT / folder_name
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(PROJECT_ROOT))

print(f"\nPackaged results/tables/figures to: {zip_path}")
