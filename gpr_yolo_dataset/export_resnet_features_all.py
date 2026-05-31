import os
import sys
import json
from pathlib import Path

import numpy as np
import torch

#批量调用ResNet18提特征来生成train_feature.npz\val_features.npz\test_features.npz

# =========================================================
# 批量生成:
#   train_features.npz
#   val_features.npz
#   test_features.npz
# =========================================================

# ===== 旧项目根目录（用于导入 models.resnet_model）=====
PROJECT_ROOT = r"C:\temporary internet files\GPR_ModelSpace_New"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import resnet_model  # noqa: E402

# ===== 输入 / 输出路径 =====
DATA_DIR = Path(
    r"C:\temporary internet files\GPR_ModelSpace_New\gpr_yolo_dataset\window_dataset_overlap40_w128_s32"
)
OUT_DIR = DATA_DIR / "resnet_features"

SPLITS = ["train", "val", "test"]

# ===== 运行参数 =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

# 可选：只在调试时限制样本数；正式跑请保持 None
MAX_SAMPLES_PER_SPLIT = None


def load_split_npz(split: str):
    """读取某个 split 的窗口数据"""
    npz_path = DATA_DIR / f"{split}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"找不到文件: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    return data, npz_path


def export_one_split(extractor, split: str):
    """对单个 split 提取 ResNet 特征并保存"""
    data, npz_path = load_split_npz(split)

    windows = data["windows"]        # [N, 224, 128], uint8
    labels = data["labels"]          # [N]
    cls_ids = data["cls_ids"]        # [N]
    x_starts = data["x_starts"]      # [N]
    x_ends = data["x_ends"]          # [N]
    img_names = data["img_names"]    # [N]
    window_ids = data["window_ids"]  # [N]

    print("\n" + "=" * 72)
    print(f"[SPLIT ] {split}")
    print(f"[INPUT ] {npz_path}")
    print("=" * 72)
    print(f"[INFO] windows.shape   = {windows.shape}, dtype={windows.dtype}")
    print(f"[INFO] labels.shape    = {labels.shape}, dtype={labels.dtype}")
    print(f"[INFO] cls_ids.shape   = {cls_ids.shape}, dtype={cls_ids.dtype}")

    if MAX_SAMPLES_PER_SPLIT is not None:
        n = min(MAX_SAMPLES_PER_SPLIT, len(windows))
        windows = windows[:n]
        labels = labels[:n]
        cls_ids = cls_ids[:n]
        x_starts = x_starts[:n]
        x_ends = x_ends[:n]
        img_names = img_names[:n]
        window_ids = window_ids[:n]
        print(f"[INFO] debug mode: only using first {n} samples")

    if len(windows) == 0:
        raise ValueError(f"{split} 没有可用窗口，无法提特征。")

    print(f"[STEP] extracting ResNet features on {DEVICE} ...")
    features = extractor.extract_features(windows, batch_size=BATCH_SIZE)

    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()

    features = features.astype(np.float32)

    out_path = OUT_DIR / f"{split}_features.npz"
    np.savez_compressed(
        out_path,
        features=features,      # [N, T, 512]
        labels=labels,
        cls_ids=cls_ids,
        x_starts=x_starts,
        x_ends=x_ends,
        img_names=img_names,
        window_ids=window_ids,
    )

    meta = {
        "split": split,
        "input_npz": str(npz_path),
        "num_samples": int(len(labels)),
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "batch_size": BATCH_SIZE,
        "device": str(DEVICE),
        "max_samples_per_split": MAX_SAMPLES_PER_SPLIT,
    }

    meta_path = OUT_DIR / f"{split}_features_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] feature file saved -> {out_path}")
    print(f"[OK] meta file saved    -> {meta_path}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    return meta


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"[DATA_DIR   ] {DATA_DIR}")
    print(f"[OUT_DIR    ] {OUT_DIR}")
    print(f"[DEVICE     ] {DEVICE}")
    print(f"[BATCH_SIZE ] {BATCH_SIZE}")
    print("=" * 72)

    print("\n[STEP] loading ResNetFeatureExtractor ...")
    extractor = resnet_model.ResNetFeatureExtractor(device=DEVICE)

    summary = {}
    for split in SPLITS:
        summary[split] = export_one_split(extractor, split)

    summary_path = OUT_DIR / "features_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print(f"[DONE] all feature files exported -> {OUT_DIR}")
    print(f"[DONE] summary saved -> {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()