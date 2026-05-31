import os
import json
import numpy as np
import joblib
from sklearn.decomposition import PCA

FEATURE_DIR = r"C:\temporary internet files\GPR_ModelSpace_New\gpr_yolo_dataset\window_dataset_overlap40_w128_s32\resnet_features"
OUT_DIR = os.path.join(FEATURE_DIR, "pca32")

SPLITS = ["train", "val", "test"]
N_COMPONENTS = 32

#降维(PCA)

def load_split(split):
    path = os.path.join(FEATURE_DIR, f"{split}_features.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")
    data = np.load(path, allow_pickle=True)
    return data


def transform_one_split(pca, split):
    data = load_split(split)

    features = data["features"]   # [N, 15, 512]
    labels = data["labels"]
    cls_ids = data["cls_ids"]
    x_starts = data["x_starts"]
    x_ends = data["x_ends"]
    img_names = data["img_names"]
    window_ids = data["window_ids"]

    n, t, d = features.shape
    flat = features.reshape(-1, d)           # [N*15, 512]
    flat_reduced = pca.transform(flat)       # [N*15, 32]
    reduced = flat_reduced.reshape(n, t, -1) # [N, 15, 32]

    out_path = os.path.join(OUT_DIR, f"{split}_features_pca32.npz")
    np.savez_compressed(
        out_path,
        features=reduced.astype(np.float32),
        labels=labels,
        cls_ids=cls_ids,
        x_starts=x_starts,
        x_ends=x_ends,
        img_names=img_names,
        window_ids=window_ids,
    )

    meta = {
        "split": split,
        "input_shape": list(features.shape),
        "output_shape": list(reduced.shape),
        "n_components": N_COMPONENTS,
    }
    meta_path = os.path.join(OUT_DIR, f"{split}_features_pca32_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[{split}] transform done -> {out_path}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 72)
    print(f"[FEATURE_DIR ] {FEATURE_DIR}")
    print(f"[OUT_DIR     ] {OUT_DIR}")
    print(f"[N_COMPONENTS] {N_COMPONENTS}")
    print("=" * 72)

    # ===== 1. 读取 train 特征 =====
    train = load_split("train")
    train_features = train["features"]   # [N, 15, 512]
    train_labels = train["labels"]       # [N]

    print(f"[INFO] train_features.shape = {train_features.shape}")
    print(f"[INFO] train_labels.shape   = {train_labels.shape}")

    # ===== 2. 只取正常窗口 =====
    normal_mask = (train_labels == 0)
    normal_features = train_features[normal_mask]   # [N_normal, 15, 512]

    print(f"[INFO] normal train windows = {normal_features.shape[0]}")

    # ===== 3. 展平成 patch 特征 =====
    n, t, d = normal_features.shape
    flat_normal = normal_features.reshape(-1, d)    # [N_normal*15, 512]

    print(f"[INFO] flat_normal.shape = {flat_normal.shape}")

    # ===== 4. 拟合 PCA =====
    print("\n[STEP] fitting PCA ...")
    pca = PCA(n_components=N_COMPONENTS, svd_solver="randomized", random_state=42)
    pca.fit(flat_normal)

    explained = float(np.sum(pca.explained_variance_ratio_))
    print(f"[INFO] total explained variance ratio = {explained:.6f}")

    # 保存 PCA 模型
    pca_model_path = os.path.join(OUT_DIR, "pca_32_model.joblib")
    joblib.dump(pca, pca_model_path)
    print(f"[OK] PCA model saved -> {pca_model_path}")

    pca_info = {
        "n_components": N_COMPONENTS,
        "input_dim": int(d),
        "explained_variance_ratio_sum": explained,
        "normal_train_windows": int(normal_features.shape[0]),
        "flat_normal_samples": int(flat_normal.shape[0]),
    }

    with open(os.path.join(OUT_DIR, "pca_32_info.json"), "w", encoding="utf-8") as f:
        json.dump(pca_info, f, ensure_ascii=False, indent=2)

    print(json.dumps(pca_info, ensure_ascii=False, indent=2))

    # ===== 5. 转换 train / val / test =====
    for split in SPLITS:
        transform_one_split(pca, split)

    print("\n" + "=" * 72)
    print("[DONE] PCA fit + all split transform finished.")
    print("=" * 72)


if __name__ == "__main__":
    main()