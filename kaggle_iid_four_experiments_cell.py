# Kaggle one-cell launcher for the IID FedCausal experiment suite.
# Paste this whole file into one Kaggle Notebook cell after the project is
# available at /kaggle/working/FedCausal.

import os
import sys
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

config_path = PROJECT_ROOT / "configs" / "default_kaggle.yaml"
with config_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Core IID model-heterogeneity setup.
config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
config["federated"]["num_clients"] = 10
config["federated"]["rounds"] = 20
config["federated"]["local_epochs"] = 1
config["federated"]["batch_size"] = 64
config["federated"]["partition_mode"] = "iid"
config["federated"]["participation_rate"] = 1.0
config["attack"]["type"] = "none"
config["attack"]["malicious_ratio"] = 0.0
config["corruption"]["enable_train_corruption"] = False

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

# Keep these lists small for the first paper-debug pass. Add more corruptions
# later only after the clean IID run looks sane.
METHODS = ["fedproto", "fedcausal_mask", "fedcausal_mvp"]
CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "motion_blur",
    "fog",
    "jpeg_compression",
]
SEVERITIES = [1, 3, 5]

# Default: run only Experiment 1 and Experiment 2.
# Set this to True when you are ready to also run:
#   Experiment 3: 30% gaussian-noise corrupted clients, no attack
#   Experiment 4: 30% jpeg-compression corrupted clients, no attack
RUN_CORRUPTED_CLIENT_EXPERIMENTS = False

# Debug mode shrinks to at most 5 clients and 3 rounds inside each method.
# Keep False for the real Kaggle run.
DEBUG = False

from run_iid_experiments import run_four_experiment_suite

results = run_four_experiment_suite(
    config=config,
    methods=METHODS,
    corruptions=CORRUPTIONS,
    severities=SEVERITIES,
    debug=DEBUG,
    run_corrupted_client_experiments=RUN_CORRUPTED_CLIENT_EXPERIMENTS,
)

print("\nFinished IID experiment suite.")
for name, output in results.items():
    print(f"\n{name}")
    if isinstance(output, dict):
        for key in ["clean_summary_csv", "corruption_summary_csv"]:
            value = output.get(key)
            if value:
                print(f"  {key}: {value}")

