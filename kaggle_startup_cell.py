# Kaggle Notebook startup cell for FedCausal.
# Paste this cell near the top of the notebook after uploading/copying the project.

import os
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path("/kaggle/working/FedCausal")
if not PROJECT_ROOT.exists():
    raise FileNotFoundError(
        "FedCausal project directory not found at /kaggle/working/FedCausal. "
        "Please upload or copy the project there before running this cell."
    )

os.chdir(PROJECT_ROOT)

print(f"Current working directory: {Path.cwd()}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU name: CPU only")

config_path = PROJECT_ROOT / "configs" / "default_kaggle.yaml"
with config_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

for output_key in ["result_dir", "checkpoint_dir", "figure_dir"]:
    Path(cfg["output"][output_key]).mkdir(parents=True, exist_ok=True)

print("Current config:")
print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
