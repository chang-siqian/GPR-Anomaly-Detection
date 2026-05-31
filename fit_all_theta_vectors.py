import os
import sys
import json
import numpy as np

#局部动态模型拟合 ——> 得到模型向量theta

# ===== 项目根目录 =====
PROJECT_ROOT = r"C:\temporary internet files\GPR_ModelSpace_New"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis import detection

PCA_DIR = r"C:\temporary internet files\GPR_ModelSpace_New\gpr_yolo_dataset\window_dataset_overlap40_w128_s32\resnet_features\pca32"
OUT_DIR = os.path.join(PCA_DIR, "theta_vectors")

SPLITS = ["train", "val", "test"]

#把某一个数据划分（train/val/test）的PCA特征文件读进来，拟合成theta向量，再把结果保存进去
def fit_one_split(split):
    in_path = os.path.join(PCA_DIR, f"{split}_features_pca32.npz")
    out_path = os.path.join(OUT_DIR, f"{split}_theta.npz")
    meta_path = os.path.join(OUT_DIR, f"{split}_theta_meta.json")

    print("\n" + "=" * 72)
    print(f"[SPLIT] {split}")
    print(f"[IN   ] {in_path}")
    print(f"[OUT  ] {out_path}")
    print("=" * 72)

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"找不到文件: {in_path}")

    data = np.load(in_path, allow_pickle=True)

    features = data["features"]   # [N, 15, 32]
    labels = data["labels"]
    cls_ids = data["cls_ids"]
    x_starts = data["x_starts"]
    x_ends = data["x_ends"]
    img_names = data["img_names"]
    window_ids = data["window_ids"]

    print(f"[INFO] features.shape = {features.shape}, dtype={features.dtype}")
    print(f"[INFO] labels.shape   = {labels.shape}, dtype={labels.dtype}")

    print("\n[STEP] fitting dynamic model for all windows ...")
    thetas = detection.fit_dynamic_model(features)

    if not isinstance(thetas, np.ndarray):
        thetas = np.array(thetas)

    thetas = thetas.astype(np.float32)

    os.makedirs(OUT_DIR, exist_ok=True)

    np.savez_compressed(
        out_path,
        thetas=thetas,          # [N, 1056]
        labels=labels,
        cls_ids=cls_ids,
        x_starts=x_starts,
        x_ends=x_ends,
        img_names=img_names,
        window_ids=window_ids,
    )

    abnormal = int(labels.sum())
    total = len(labels)
    normal = total - abnormal

    meta = {
        "split": split,
        "num_samples": int(total),
        "num_normal": int(normal),
        "num_abnormal": int(abnormal),
        "theta_shape": list(thetas.shape),
        "theta_dim": int(thetas.shape[1]) if thetas.ndim == 2 else None,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n[OK] theta export done.")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def main():
    print("=" * 72)
    print(f"[PCA_DIR] {PCA_DIR}")
    print(f"[OUTDIR ] {OUT_DIR}")
    print("=" * 72)

    for split in SPLITS:
        fit_one_split(split)

    print("\n" + "=" * 72)
    print("[DONE] all theta vectors exported.")
    print("=" * 72)


if __name__ == "__main__":
    main()