from pathlib import Path
import json

import numpy as np
import matplotlib

# 只保存图片，不弹窗，避免 PyCharm 后端问题
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib


# ============================================================
# 1. 路径配置
# ============================================================

ROOT = Path(r"C:\temporary internet files\GPR_ModelSpace_New\gpr_yolo_dataset")

# 你的 HOG+SVM baseline 输出目录
BASELINE_DIR = ROOT / "hog_svm_baseline_binary"

# 画哪个 split
# 推荐先画 test，因为测试集最适合展示最终效果
# 也可以改成 ["train", "val", "test"]
SPLITS_TO_USE = ["test"]

OUT_DIR = BASELINE_DIR / "hog_feature_space_tsne3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. HOG 参数：必须和 run_hog_svm_baseline.py 保持一致
# ============================================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

IMAGE_W = 256
IMAGE_H = 256

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (32, 32)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"

RANDOM_SEED = 42


# ============================================================
# 3. t-SNE 参数
# ============================================================

# 如果画 train+val+test，点可能比较多，可以采样
MAX_POINTS_TOTAL = 3000

# 二分类均衡采样，每类最多多少
MAX_PER_CLASS = 1500

PCA_BEFORE_TSNE_DIM = 50
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1500


# ============================================================
# 4. 模型路径
# ============================================================

SCALER_PATH = BASELINE_DIR / "best_scaler.joblib"
MODEL_PATH = BASELINE_DIR / "best_svm_model.joblib"


# ============================================================
# 5. 类别名称和颜色
# ============================================================

GT_CLASS_NAMES = {
    0: "normal",
    1: "abnormal",
}

GT_CLASS_COLORS = {
    0: "blue",
    1: "red",
}

PRED_CLASS_NAMES = {
    0: "pred_normal",
    1: "pred_abnormal",
}

PRED_CLASS_COLORS = {
    0: "blue",
    1: "red",
}


# ============================================================
# 6. 读取图片、标签、HOG 特征
# ============================================================

def load_gray_image(img_path: Path) -> np.ndarray:
    """
    读取灰度图并 resize 到固定大小。
    """
    with Image.open(img_path) as im:
        im = im.convert("L")
        im = im.resize((IMAGE_W, IMAGE_H), Image.BILINEAR)
        arr = np.array(im, dtype=np.uint8)
    return arr


def load_binary_label(label_path: Path) -> int:
    """
    二分类标签定义：
    - 空 txt: normal = 0
    - 非空 txt: abnormal = 1
    """
    if not label_path.exists():
        raise FileNotFoundError(f"找不到标注文件: {label_path}")

    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    return 0 if text == "" else 1


def extract_hog_feature(img_gray: np.ndarray) -> np.ndarray:
    """
    提取 HOG 特征。
    """
    feat = hog(
        img_gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        visualize=False,
        feature_vector=True,
    )
    return feat.astype(np.float32)


def load_split(split: str):
    """
    读取某个 split 的全部图像，并提取 HOG 特征。
    """
    img_dir = ROOT / "images" / split
    lab_dir = ROOT / "labels" / split

    if not img_dir.exists():
        raise FileNotFoundError(f"找不到图像目录: {img_dir}")
    if not lab_dir.exists():
        raise FileNotFoundError(f"找不到标签目录: {lab_dir}")

    image_paths = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    )

    if len(image_paths) == 0:
        raise ValueError(f"{split} 没有找到图像文件: {img_dir}")

    X_list = []
    y_list = []
    name_list = []
    split_list = []

    print("\n" + "=" * 80)
    print(f"[LOAD SPLIT] {split}")
    print(f"[IMG DIR   ] {img_dir}")
    print(f"[LAB DIR   ] {lab_dir}")
    print("=" * 80)

    for i, img_path in enumerate(image_paths, start=1):
        label_path = lab_dir / f"{img_path.stem}.txt"

        img = load_gray_image(img_path)
        y = load_binary_label(label_path)
        feat = extract_hog_feature(img)

        X_list.append(feat)
        y_list.append(y)
        name_list.append(img_path.name)
        split_list.append(split)

        if i <= 5:
            print(
                f"[sample {i}] {img_path.name} | "
                f"img_shape={img.shape} | label={y} | feat_dim={feat.shape[0]}"
            )

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    names = np.array(name_list)
    splits = np.array(split_list)

    print(f"[INFO] X.shape = {X.shape}")
    print(f"[INFO] y.shape = {y.shape}")
    print(f"[INFO] normal   = {int((y == 0).sum())}")
    print(f"[INFO] abnormal = {int((y == 1).sum())}")

    return X, y, names, splits


def load_all_splits():
    all_X = []
    all_y = []
    all_names = []
    all_splits = []

    for split in SPLITS_TO_USE:
        X, y, names, splits = load_split(split)
        all_X.append(X)
        all_y.append(y)
        all_names.append(names)
        all_splits.append(splits)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    names = np.concatenate(all_names, axis=0)
    splits = np.concatenate(all_splits, axis=0)

    print("\n" + "=" * 80)
    print("[MERGED DATA]")
    print(f"X.shape      = {X.shape}")
    print(f"y.shape      = {y.shape}")
    print(f"names.shape  = {names.shape}")
    print(f"splits.shape = {splits.shape}")
    print(f"normal       = {int((y == 0).sum())}")
    print(f"abnormal     = {int((y == 1).sum())}")
    print("=" * 80)

    return X, y, names, splits


# ============================================================
# 7. 采样、降维、画图
# ============================================================

def balanced_sample_indices(y):
    """
    二分类均衡采样，防止某一类太多导致图看起来乱。
    """
    n = len(y)

    if n <= MAX_POINTS_TOTAL:
        return np.arange(n)

    rng = np.random.default_rng(RANDOM_SEED)

    idx_normal = np.where(y == 0)[0]
    idx_abnormal = np.where(y == 1)[0]

    rng.shuffle(idx_normal)
    rng.shuffle(idx_abnormal)

    take_normal = min(MAX_PER_CLASS, len(idx_normal))
    take_abnormal = min(MAX_PER_CLASS, len(idx_abnormal))

    selected = np.concatenate([
        idx_normal[:take_normal],
        idx_abnormal[:take_abnormal],
    ])

    rng.shuffle(selected)

    if len(selected) > MAX_POINTS_TOTAL:
        selected = selected[:MAX_POINTS_TOTAL]

    return selected


def compute_tsne3d(X):
    """
    StandardScaler + PCA + 3D t-SNE。
    """
    print("\n" + "=" * 80)
    print("[STEP] scale HOG features")
    print("=" * 80)

    # 优先使用 baseline 训练好的 scaler，保证和 SVM 输入空间一致
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X)
        print(f"[OK] loaded baseline scaler -> {SCALER_PATH}")
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("[WARN] 没找到 baseline scaler，临时 fit 一个 StandardScaler")

    n_samples, n_features = X_scaled.shape

    pca_dim = min(PCA_BEFORE_TSNE_DIM, n_features, n_samples - 1)

    if pca_dim >= 3:
        print("\n" + "=" * 80)
        print(f"[STEP] PCA before t-SNE: {n_features} -> {pca_dim}")
        print("=" * 80)

        pca = PCA(n_components=pca_dim, random_state=RANDOM_SEED)
        X_reduced = pca.fit_transform(X_scaled)

        print(
            f"[INFO] PCA explained variance ratio sum = "
            f"{pca.explained_variance_ratio_.sum():.6f}"
        )
    else:
        X_reduced = X_scaled

    perplexity = min(TSNE_PERPLEXITY, max(5, (n_samples - 1) // 3))

    print("\n" + "=" * 80)
    print("[STEP] t-SNE 3D")
    print(f"[INFO] n_samples  = {n_samples}")
    print(f"[INFO] input_dim  = {X_reduced.shape[1]}")
    print(f"[INFO] perplexity = {perplexity}")
    print(f"[INFO] n_iter     = {TSNE_N_ITER}")
    print("=" * 80)

    # sklearn 新版本用 max_iter，旧版本用 n_iter
    try:
        tsne = TSNE(
            n_components=3,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            max_iter=TSNE_N_ITER,
            random_state=RANDOM_SEED,
            verbose=1,
        )
    except TypeError:
        tsne = TSNE(
            n_components=3,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            n_iter=TSNE_N_ITER,
            random_state=RANDOM_SEED,
            verbose=1,
        )

    emb = tsne.fit_transform(X_reduced)
    return emb.astype(np.float32)


def set_3d_style(ax):
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")

    # 不显示数值刻度，强调这是可视化坐标，不是物理坐标
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.grid(False)

    try:
        ax.set_box_aspect((1.5, 1.0, 0.75))
    except Exception:
        pass


def plot_3d_two_views(
    emb,
    y,
    class_names,
    class_colors,
    title,
    out_path,
    marker_size=22,
    alpha=0.75,
):
    """
    一张图放两个视角，类似论文里的 (a)(b)。
    """
    fig = plt.figure(figsize=(9, 12))

    views = [
        (18, -65, "(a)"),
        (18, 35, "(b)"),
    ]

    for i, (elev, azim, label) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 1, i, projection="3d")

        for c in sorted(np.unique(y)):
            mask = y == c

            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                emb[mask, 2],
                s=marker_size,
                c=class_colors.get(int(c), "gray"),
                label=class_names.get(int(c), str(c)),
                alpha=alpha,
                depthshade=False,
            )

        ax.view_init(elev=elev, azim=azim)
        set_3d_style(ax)
        ax.legend(loc="upper right", fontsize=9, frameon=True)
        ax.set_title(title, fontsize=12)
        ax.text2D(0.48, -0.08, label, transform=ax.transAxes, fontsize=18)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] saved figure -> {out_path}")


def robust_normalize(x):
    x = x.astype(np.float32)

    lo, hi = np.percentile(x, [2, 98])
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)

    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0, 1)
    return y


def plot_3d_score_two_views(
    emb,
    score,
    title,
    out_path,
    marker_size=22,
    alpha=0.80,
):
    """
    用连续颜色画 SVM decision score。
    """
    fig = plt.figure(figsize=(9, 12))

    views = [
        (18, -65, "(a)"),
        (18, 35, "(b)"),
    ]

    for i, (elev, azim, label) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 1, i, projection="3d")

        p = ax.scatter(
            emb[:, 0],
            emb[:, 1],
            emb[:, 2],
            s=marker_size,
            c=score,
            cmap="jet",
            alpha=alpha,
            depthshade=False,
        )

        ax.view_init(elev=elev, azim=azim)
        set_3d_style(ax)
        ax.set_title(title, fontsize=12)
        ax.text2D(0.48, -0.08, label, transform=ax.transAxes, fontsize=18)

        cbar = fig.colorbar(p, ax=ax, shrink=0.65, pad=0.08)
        cbar.set_label("SVM abnormal score")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] saved figure -> {out_path}")


# ============================================================
# 8. SVM 预测
# ============================================================

def load_model_and_predict(X):
    """
    加载 baseline 训练好的 scaler 和 SVM，对当前 X 做预测。
    如果模型不存在，就返回 None。
    """
    if not SCALER_PATH.exists() or not MODEL_PATH.exists():
        print("[WARN] 没找到 best_scaler.joblib 或 best_svm_model.joblib")
        print("[WARN] 将只画 GT 图，不画 SVM 预测图。")
        print(f"[SCALER_PATH] {SCALER_PATH}")
        print(f"[MODEL_PATH ] {MODEL_PATH}")
        return None, None

    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)

    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled).astype(int)

    score = None
    if hasattr(model, "decision_function"):
        score = model.decision_function(X_scaled).astype(np.float32)

        # 对二分类 SVC 来说，classes_ 通常是 [0, 1]；
        # decision_function 越大越倾向 class 1，也就是 abnormal。
        if hasattr(model, "classes_"):
            print("[INFO] SVM classes_ =", model.classes_)

    print("[INFO] pred_normal   =", int((y_pred == 0).sum()))
    print("[INFO] pred_abnormal =", int((y_pred == 1).sum()))

    return y_pred, score


# ============================================================
# 9. 主程序
# ============================================================

def main():
    np.random.seed(RANDOM_SEED)

    print("=" * 80)
    print("HOG + SVM baseline: 3D t-SNE visualization")
    print("=" * 80)
    print(f"[ROOT        ] {ROOT}")
    print(f"[BASELINE_DIR] {BASELINE_DIR}")
    print(f"[SPLITS      ] {SPLITS_TO_USE}")
    print(f"[OUT_DIR     ] {OUT_DIR}")
    print("=" * 80)

    # 1. 读取 HOG 特征
    X, y_gt, names, splits = load_all_splits()

    # 2. 均衡采样，减少图太乱的问题
    idx = balanced_sample_indices(y_gt)

    X_s = X[idx]
    y_gt_s = y_gt[idx]
    names_s = names[idx]
    splits_s = splits[idx]

    print("\n" + "=" * 80)
    print("[SAMPLED DATA]")
    print(f"X_s.shape = {X_s.shape}")
    print(f"normal    = {int((y_gt_s == 0).sum())}")
    print(f"abnormal  = {int((y_gt_s == 1).sum())}")
    print("=" * 80)

    # 3. 计算 3D t-SNE
    emb = compute_tsne3d(X_s)

    split_tag = "_".join(SPLITS_TO_USE)

    # 4. 图 1：按真实标签画
    out_gt = OUT_DIR / f"hog_tsne3d_GT_{split_tag}.png"
    plot_3d_two_views(
        emb=emb,
        y=y_gt_s,
        class_names=GT_CLASS_NAMES,
        class_colors=GT_CLASS_COLORS,
        title="3D t-SNE visualization of HOG feature space (GT labels)",
        out_path=out_gt,
    )

    # 5. 图 2：按 SVM 预测标签画
    y_pred_s, score_s = load_model_and_predict(X_s)

    if y_pred_s is not None:
        out_pred = OUT_DIR / f"hog_tsne3d_SVM_pred_{split_tag}.png"
        plot_3d_two_views(
            emb=emb,
            y=y_pred_s,
            class_names=PRED_CLASS_NAMES,
            class_colors=PRED_CLASS_COLORS,
            title="3D t-SNE visualization of HOG feature space (SVM prediction)",
            out_path=out_pred,
        )

    # 6. 图 3：按 SVM decision score 连续着色
    if score_s is not None:
        score_norm = robust_normalize(score_s)

        out_score = OUT_DIR / f"hog_tsne3d_SVM_score_{split_tag}.png"
        plot_3d_score_two_views(
            emb=emb,
            score=score_norm,
            title="3D t-SNE visualization of HOG feature space (SVM abnormal score)",
            out_path=out_score,
        )

    # 7. 保存 embedding，方便以后复用
    emb_path = OUT_DIR / f"hog_tsne3d_embedding_{split_tag}.npz"
    np.savez_compressed(
        emb_path,
        embedding=emb,
        X_sampled=X_s,
        y_gt=y_gt_s,
        names=names_s,
        splits=splits_s,
        sample_idx=idx,
        y_pred=y_pred_s if y_pred_s is not None else np.array([]),
        svm_score=score_s if score_s is not None else np.array([]),
    )

    config = {
        "root": str(ROOT),
        "baseline_dir": str(BASELINE_DIR),
        "splits_to_use": SPLITS_TO_USE,
        "image_size": [IMAGE_H, IMAGE_W],
        "hog_config": {
            "orientations": HOG_ORIENTATIONS,
            "pixels_per_cell": list(HOG_PIXELS_PER_CELL),
            "cells_per_block": list(HOG_CELLS_PER_BLOCK),
            "block_norm": HOG_BLOCK_NORM,
        },
        "max_points_total": MAX_POINTS_TOTAL,
        "max_per_class": MAX_PER_CLASS,
        "pca_before_tsne_dim": PCA_BEFORE_TSNE_DIM,
        "tsne_perplexity": TSNE_PERPLEXITY,
        "tsne_n_iter": TSNE_N_ITER,
        "num_points_all": int(len(X)),
        "num_points_sampled": int(len(X_s)),
        "num_normal_sampled": int((y_gt_s == 0).sum()),
        "num_abnormal_sampled": int((y_gt_s == 1).sum()),
    }

    config_path = OUT_DIR / f"hog_tsne3d_config_{split_tag}.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("[DONE] HOG + SVM 3D t-SNE 可视化完成")
    print(f"[OUT_DIR] {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()