# %% [markdown]
# # FedCausal Kaggle 一键运行 Cell
#
# 这个文件按 Kaggle Notebook 的执行顺序组织为 10 个 cell。
# 可以整段复制到 Notebook，也可以按 `# %%` 分隔逐段复制运行。

# %%
# Cell 1：环境检查
import os
import platform
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path("/kaggle/working/FedCausal")
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU: not available, experiments will run on CPU")
print("Project root:", PROJECT_ROOT)
print("Current working directory:", Path.cwd())

# %%
# Cell 2：创建项目目录并导入模块
from pathlib import Path

PROJECT_DIRS = [
    "configs",
    "data",
    "models",
    "losses",
    "methods",
    "federated",
    "attacks",
    "utils",
    "results",
    "checkpoints",
    "figures",
    "tables",
]

for dirname in PROJECT_DIRS:
    (PROJECT_ROOT / dirname).mkdir(parents=True, exist_ok=True)

config_path = PROJECT_ROOT / "configs" / "default_kaggle.yaml"
if not config_path.exists():
    raise FileNotFoundError(
        "未找到 configs/default_kaggle.yaml。请先把 FedCausal 项目代码上传或复制到 "
        "/kaggle/working/FedCausal/。"
    )

from methods.fedproto import run_fedproto
from methods.fedcausal_mask import run_fedcausal_mask
from methods.fedcausal_mvp import run_fedcausal_mvp
from eval_cifar10c import evaluate_method_on_cifar10c
from analyze_results import analyze_results
from utils.logger import CSVLogger
from utils.seed import seed_everything

print("Project directories are ready.")
print("Core modules imported successfully.")

# %%
# Cell 3：加载配置 default_kaggle.yaml
import copy
import pprint

import yaml

with config_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

seed_everything(int(config.get("seed", 42)))

print("Loaded config:")
pprint.pp(config)


def final_clean_accuracy(outputs):
    """Return mean clean accuracy of the final round from a method output dict."""
    history = outputs.get("history", []) if isinstance(outputs, dict) else []
    if not history:
        return None
    final_round = max(int(row["round"]) for row in history)
    accs = [
        float(row["clean_acc"])
        for row in history
        if int(row["round"]) == final_round and row.get("clean_acc") not in ("", None)
    ]
    return sum(accs) / len(accs) if accs else None


def set_debug_federated(cfg, num_clients=5, rounds=3, local_epochs=1):
    """Make a small Kaggle-safe debug config copy."""
    cfg = copy.deepcopy(cfg)
    cfg["federated"]["num_clients"] = num_clients
    cfg["federated"]["rounds"] = rounds
    cfg["federated"]["local_epochs"] = local_epochs
    cfg["federated"]["participation_rate"] = 1.0
    cfg["model"]["client_models"] = cfg["model"]["client_models"][:num_clients]
    return cfg

# %%
# Cell 4：快速 debug FedProto
fedproto_debug_config = set_debug_federated(config, num_clients=5, rounds=3, local_epochs=1)
fedproto_debug_config["attack"]["type"] = "none"
fedproto_debug_config["attack"]["malicious_ratio"] = 0.0
fedproto_debug_config["corruption"]["enable_train_corruption"] = False

fedproto_debug_outputs = run_fedproto(fedproto_debug_config, debug=True)
fedproto_debug_acc = final_clean_accuracy(fedproto_debug_outputs)
print("FedProto debug final clean accuracy:", fedproto_debug_acc)
print("FedProto CSV:", fedproto_debug_outputs.get("result_csv"))

# %%
# Cell 5：运行 FedCausal-Mask debug
mask_debug_config = set_debug_federated(config, num_clients=5, rounds=3, local_epochs=1)
mask_debug_config["fedcausal"]["use_fft_mask"] = True
mask_debug_config["fedcausal"]["use_counterfactual"] = False
mask_debug_config["fedcausal"]["disable_inv"] = True
mask_debug_config["attack"]["type"] = "none"
mask_debug_config["attack"]["malicious_ratio"] = 0.0

mask_debug_outputs = run_fedcausal_mask(mask_debug_config, debug=True)
mask_debug_acc = final_clean_accuracy(mask_debug_outputs)
print("FedCausal-Mask debug final clean accuracy:", mask_debug_acc)

mask_ckpt = torch.load(mask_debug_outputs["global_mask_path"], map_location="cpu")
global_mask = mask_ckpt["global_mask"]
mask_init = float(mask_debug_config["fedcausal"]["mask_init"])
print("Global mask shape:", tuple(global_mask.shape))
print("Global mask mean:", float(global_mask.mean()))
print("Global mask std:", float(global_mask.std(unbiased=False)))
print("Abs(mean - mask_init):", abs(float(global_mask.mean()) - mask_init))

saved_heatmaps = sorted((PROJECT_ROOT / "figures").glob("global_mask_round_*.png"))
print("Mask heatmaps:", [str(path) for path in saved_heatmaps[-3:]])

# %%
# Cell 6：运行 FedCausal-MVP
mvp_config = copy.deepcopy(config)
mvp_config["federated"]["num_clients"] = 10
mvp_config["federated"]["rounds"] = 20
mvp_config["federated"]["local_epochs"] = 1
mvp_config["federated"]["participation_rate"] = 1.0
mvp_config["model"]["client_models"] = config["model"]["client_models"][:10]
mvp_config["fedcausal"]["use_fft_mask"] = True
mvp_config["fedcausal"]["use_counterfactual"] = True
mvp_config["fedcausal"]["disable_inv"] = False
mvp_config["fedcausal"]["lambda_inv"] = 0.1
mvp_config["fedcausal"]["lambda_scl"] = 0.1
mvp_config["attack"]["type"] = "none"
mvp_config["attack"]["malicious_ratio"] = 0.0
mvp_config["aggregation"]["mode"] = "mask_proto_energy"

mvp_outputs = run_fedcausal_mvp(mvp_config, debug=False, disable_inv=False)
mvp_acc = final_clean_accuracy(mvp_outputs)
print("FedCausal-MVP final clean accuracy:", mvp_acc)
print("FedCausal-MVP CSV:", mvp_outputs.get("result_csv"))
print("Global mask checkpoint:", mvp_outputs.get("global_mask_path"))

# %%
# Cell 7：运行 CIFAR-10-C 评估
import csv

cifar10c_root = Path(config["dataset"].get("cifar10c_root", "/kaggle/input/cifar10-c"))
if cifar10c_root.exists():
    if "mvp_outputs" not in globals():
        print("未检测到 mvp_outputs，先运行一个小规模 FedCausal-MVP 供 CIFAR-10-C 评估。")
        eval_config = set_debug_federated(config, num_clients=5, rounds=3, local_epochs=1)
        eval_config["fedcausal"]["use_counterfactual"] = True
        eval_config["fedcausal"]["disable_inv"] = False
        mvp_outputs = run_fedcausal_mvp(eval_config, debug=True, disable_inv=False)
        mvp_config = eval_config

    corruption_rows = evaluate_method_on_cifar10c(
        method="fedcausal_mvp",
        outputs=mvp_outputs,
        config=mvp_config,
        round_id=int(mvp_config["federated"]["rounds"]) - 1,
        corruptions=[
            "gaussian_noise",
            "shot_noise",
            "motion_blur",
            "defocus_blur",
            "fog",
            "jpeg_compression",
            "pixelate",
        ],
        severities=[1, 3, 5],
    )

    corruption_csv = PROJECT_ROOT / "results" / "corruption_results.csv"
    corruption_fields = [
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
    ]
    logger = CSVLogger(corruption_csv, fieldnames=corruption_fields, reset=True)
    logger.log_many(corruption_rows)
    print("CIFAR-10-C rows:", len(corruption_rows))
    print("Saved:", corruption_csv)
else:
    print(
        "未检测到 CIFAR-10-C，请在 Kaggle Dataset 中添加 CIFAR-10-C 数据集，"
        "并确保路径为 /kaggle/input/cifar10-c/。本 cell 已跳过。"
    )

# %%
# Cell 8：运行攻击实验
import shutil


def copy_current_attack_csv(outputs, suffix):
    src = Path(outputs["attack_csv"])
    dst = PROJECT_ROOT / "results" / f"attack_results_{suffix}.csv"
    if src.exists():
        shutil.copy2(src, dst)
        print("Saved attack copy:", dst)
        return dst
    print("Attack CSV missing:", src)
    return None


def merge_attack_csvs(paths, output_path):
    paths = [Path(path) for path in paths if path is not None and Path(path).exists()]
    if not paths:
        print("No attack CSVs to merge.")
        return
    fieldnames = None
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            rows.extend(reader)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("Merged attack CSV:", output_path)


label_attack_config = copy.deepcopy(config)
label_attack_config["federated"]["num_clients"] = 10
label_attack_config["federated"]["rounds"] = 20
label_attack_config["federated"]["local_epochs"] = 1
label_attack_config["model"]["client_models"] = config["model"]["client_models"][:10]
label_attack_config["attack"]["type"] = "label_flip"
label_attack_config["attack"]["malicious_ratio"] = 0.2
label_attack_config["attack"]["scale_factor"] = 10
label_attack_config["aggregation"]["mode"] = "mask_proto_energy"
label_attack_config["fedcausal"]["use_counterfactual"] = True
label_attack_config["fedcausal"]["disable_inv"] = False

label_attack_outputs = run_fedcausal_mvp(label_attack_config, debug=False, disable_inv=False)
label_attack_csv = copy_current_attack_csv(label_attack_outputs, "label_flip")
print("Label flip final clean accuracy:", final_clean_accuracy(label_attack_outputs))

scale_attack_config = copy.deepcopy(config)
scale_attack_config["federated"]["num_clients"] = 10
scale_attack_config["federated"]["rounds"] = 20
scale_attack_config["federated"]["local_epochs"] = 1
scale_attack_config["model"]["client_models"] = config["model"]["client_models"][:10]
scale_attack_config["attack"]["type"] = "prototype_scaling"
scale_attack_config["attack"]["malicious_ratio"] = 0.2
scale_attack_config["attack"]["scale_factor"] = 10
scale_attack_config["aggregation"]["mode"] = "mask_proto_energy"
scale_attack_config["fedcausal"]["use_counterfactual"] = True
scale_attack_config["fedcausal"]["disable_inv"] = False

scale_attack_outputs = run_fedcausal_mvp(scale_attack_config, debug=False, disable_inv=False)
scale_attack_csv = copy_current_attack_csv(scale_attack_outputs, "prototype_scaling")
print("Prototype scaling final clean accuracy:", final_clean_accuracy(scale_attack_outputs))

merge_attack_csvs(
    [label_attack_csv, scale_attack_csv],
    PROJECT_ROOT / "results" / "attack_results.csv",
)

# %%
# Cell 9：运行分析脚本
analysis_outputs = analyze_results(
    project_root=PROJECT_ROOT,
    alpha=float(config["federated"].get("dirichlet_alpha", 0.3)),
)

summary_path = PROJECT_ROOT / "analysis_summary.md"
if summary_path.exists():
    print(summary_path.read_text(encoding="utf-8"))
else:
    print("analysis_summary.md not found.")

# %%
# Cell 10：打包输出
import zipfile

zip_path = Path("/kaggle/working/fedcausal_outputs.zip")
include_dirs = ["results", "figures", "tables", "checkpoints"]

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for dirname in include_dirs:
        folder = PROJECT_ROOT / dirname
        if not folder.exists():
            print("Skip missing folder:", folder)
            continue
        for path in folder.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(PROJECT_ROOT))

print("Saved output zip:", zip_path)
print("Zip size MB:", round(zip_path.stat().st_size / (1024 * 1024), 2))
