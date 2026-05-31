from pathlib import Path
import json

import numpy as np
import matplotlib

# 只保存图片，不弹窗，避免 PyCharm 后端问题
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib


# ============================================================
# 1. 路径配置：按你当前 2D-ESN 运行结果设置
# ============================================================

PROJECT_ROOT = Path(r"C:\temporary internet files\GPR_ModelSpace_New")
ROOT = PROJECT_ROOT / "gpr_yolo_dataset"

DATA_DIR = ROOT / "window_dataset_overlap40_w128_s32"

THETA_DIR = DATA_DIR / "theta_vectors_2desn_n30_h32_w32"
OCSVM_DIR = THETA_DIR / "ocsvm_2desn_baseline"

# 推荐先只画 test，点数 1320，速度快，也适合展示测试集结果
# 如果想画全部，可以改成 ["train", "val", "test"]
SPLITS_TO_USE = ["test"]

# 使用 default 还是 best_thr 的预测结果
# 你的 test segment-level default 更好，所以我建议先用 default
# 可改成 "best_thr"
PRED_TAG = "default"

# 你的日志里 best threshold = -8.540830
# 但如果读取预测文件，一般不需要手动用这个阈值
BEST_THRESHOLD = -8.540830

OUT_DIR = THETA_DIR / "tsne3d_2desn_vis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. t-SNE 参数
# ============================================================

RANDOM_STATE = 42

# 只画 test 不会采样；画 train+val+test 时会采样
MAX_POINTS_TOTAL = 5000
MAX_PER_CLASS = 2500

PCA_BEFORE_TSNE_DIM = 50
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1500


# ============================================================
# 3. 类别颜色
# ============================================================

GT_BINARY_NAMES = {
    0: "normal",
    1: "abnormal",
}

GT_BINARY_COLORS = {
    0: "blue",
    1: "red",
}

GT_THREE_NAMES = {
    0: "normal",
    1: "cavity",
    2: "utility",
    3: "abnormal_unknown",
}

GT_THREE_COLORS = {
    0: "blue",
    1: "red",
    2: "black",
    3: "magenta",
}

PRED_NAMES = {
    0: "pred_normal",
    1: "pred_abnormal",
}

PRED_COLORS = {
    0: "blue",
    1: "red",
}


# ============================================================
# 4. 读取数据
# ============================================================

def decode_str_array(arr):
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return np.array(out)


def load_theta_split(split: str):
    path = THETA_DIR / f"{split}_theta.npz"

    if not path.exists():
        raise FileNotFoundError(f"找不到 theta 文件: {path}")

    d = np.load(path, allow_pickle=True)

    print("\n" + "=" * 80)
    print(f"[LOAD THETA] split = {split}")
    print(f"[FILE      ] {path}")
    print(f"[KEYS      ] {list(d.files)}")
    print("=" * 80)

    theta = d["theta"].astype(np.float32)
    labels = d["labels"].astype(np.int8)
    cls_ids = d["cls_ids"].astype(np.int16)

    if "img_names" in d.files:
        img_names = decode_str_array(d["img_names"])
    else:
        img_names = np.array([f"{split}_unknown"] * len(theta))

    if "window_ids" in d.files:
        window_ids = d["window_ids"].astype(np.int16)
    else:
        window_ids = np.arange(len(theta), dtype=np.int16)

    print(f"[theta] shape = {theta.shape}, dtype = {theta.dtype}")
    print(f"[labels] normal = {int((labels == 0).sum())}, abnormal = {int((labels == 1).sum())}")
    print(f"[cls_ids unique] {np.unique(cls_ids)}")

    return {
        "theta": theta,
        "labels": labels,
        "cls_ids": cls_ids,
        "img_names": img_names,
        "window_ids": window_ids,
        "split": np.array([split] * len(theta)),
    }


def load_pred_split(split: str, n_expected: int):
    pred_path = OCSVM_DIR / f"{split}_{PRED_TAG}_predictions.npz"

    if not pred_path.exists():
        print(f"[WARN] 找不到预测文件: {pred_path}")
        return None, None, None

    d = np.load(pred_path, allow_pickle=True)

    print("\n" + "-" * 80)
    print(f"[LOAD PRED] split = {split}")
    print(f"[FILE     ] {pred_path}")
    print(f"[KEYS     ] {list(d.files)}")
    print("-" * 80)

    if "pred_abnormal" in d.files:
        pred_binary = d["pred_abnormal"].astype(np.int8)
    elif "pred_labels" in d.files:
        pred_binary = d["pred_labels"].astype(np.int8)
    else:
        pred_binary = None

    scores = d["scores"].astype(np.float32) if "scores" in d.files else None

    if "threshold" in d.files:
        threshold = float(np.asarray(d["threshold"]).reshape(-1)[0])
    else:
        threshold = None

    if pred_binary is None and scores is not None and threshold is not None:
        # 你的 2D-ESN OCSVM 是 scores >= threshold 判为异常
        pred_binary = (scores >= threshold).astype(np.int8)

    if pred_binary is not None and len(pred_binary) != n_expected:
        raise ValueError(
            f"{split} 预测数量和 theta 数量不一致: "
            f"pred={len(pred_binary)}, theta={n_expected}"
        )

    if scores is not None and len(scores) != n_expected:
        raise ValueError(
            f"{split} score 数量和 theta 数量不一致: "
            f"scores={len(scores)}, theta={n_expected}"
        )

    if pred_binary is not None:
        print(f"[PRED] normal = {int((pred_binary == 0).sum())}, abnormal = {int((pred_binary == 1).sum())}")
    if scores is not None:
        print(f"[SCORE] min={scores.min():.6f}, max={scores.max():.6f}, mean={scores.mean():.6f}")
    print(f"[THRESHOLD] {threshold}")

    return pred_binary, scores, threshold


def build_gt_three_class(labels, cls_ids):
    """
    输出：
        0 normal
        1 cavity
        2 utility
        3 abnormal_unknown
    """
    y = np.full_like(labels, fill_value=3, dtype=np.int16)

    y[labels == 0] = 0
    y[(labels == 1) & (cls_ids == 0)] = 1
    y[(labels == 1) & (cls_ids == 1)] = 2

    return y


def load_all_data():
    all_theta = []
    all_labels = []
    all_cls_ids = []
    all_img_names = []
    all_window_ids = []
    all_split = []
    all_pred = []
    all_scores = []

    pred_available = True
    score_available = True

    for split in SPLITS_TO_USE:
        item = load_theta_split(split)
        n = len(item["theta"])

        pred_binary, scores, threshold = load_pred_split(split, n)

        if pred_binary is None:
            pred_available = False
            pred_binary = np.full(n, fill_value=-1, dtype=np.int8)

        if scores is None:
            score_available = False
            scores = np.full(n, fill_value=np.nan, dtype=np.float32)

        all_theta.append(item["theta"])
        all_labels.append(item["labels"])
        all_cls_ids.append(item["cls_ids"])
        all_img_names.append(item["img_names"])
        all_window_ids.append(item["window_ids"])
        all_split.append(item["split"])
        all_pred.append(pred_binary)
        all_scores.append(scores)

    theta = np.concatenate(all_theta, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    cls_ids = np.concatenate(all_cls_ids, axis=0)
    img_names = np.concatenate(all_img_names, axis=0)
    window_ids = np.concatenate(all_window_ids, axis=0)
    split_arr = np.concatenate(all_split, axis=0)
    pred_binary = np.concatenate(all_pred, axis=0)
    scores = np.concatenate(all_scores, axis=0)

    gt_binary = labels.astype(np.int8)
    gt_three = build_gt_three_class(labels, cls_ids)

    print("\n" + "=" * 80)
    print("[MERGED DATA]")
    print(f"theta.shape      = {theta.shape}")
    print(f"labels.shape     = {labels.shape}")
    print(f"cls_ids.shape    = {cls_ids.shape}")
    print(f"img_names.shape  = {img_names.shape}")
    print(f"window_ids.shape = {window_ids.shape}")
    print("=" * 80)

    return {
        "theta": theta,
        "labels": labels,
        "cls_ids": cls_ids,
        "gt_binary": gt_binary,
        "gt_three": gt_three,
        "img_names": img_names,
        "window_ids": window_ids,
        "split": split_arr,
        "pred_binary": pred_binary,
        "scores": scores,
        "pred_available": pred_available,
        "score_available": score_available,
    }


# ============================================================
# 5. 采样和降维
# ============================================================

def print_stat(name, y, names):
    unique, counts = np.unique(y, return_counts=True)
    stat = {}
    for k, v in zip(unique, counts):
        stat[names.get(int(k), str(k))] = int(v)
    print(f"[{name}] {stat}")


def balanced_sample_indices(y):
    """
    二分类均衡采样。
    如果点数不多，直接全用。
    """
    n = len(y)

    if n <= MAX_POINTS_TOTAL:
        return np.arange(n)

    rng = np.random.default_rng(RANDOM_STATE)

    selected = []

    for c in sorted(np.unique(y)):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        selected.extend(idx[:min(MAX_PER_CLASS, len(idx))].tolist())

    selected = np.array(selected, dtype=int)
    rng.shuffle(selected)

    if len(selected) > MAX_POINTS_TOTAL:
        selected = selected[:MAX_POINTS_TOTAL]

    return selected


def scale_theta(theta):
    """
    优先使用 2D-ESN OCSVM 保存的 scaler，这样和 OCSVM 输入空间一致。
    如果没有 scaler，就临时 fit 一个 StandardScaler。
    """
    scaler_path = OCSVM_DIR / "theta_scaler.joblib"

    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        x = scaler.transform(theta)
        print(f"[OK] loaded OCSVM scaler -> {scaler_path}")
    else:
        scaler = StandardScaler()
        x = scaler.fit_transform(theta)
        print("[WARN] 没找到 OCSVM scaler，临时 fit StandardScaler")

    return x.astype(np.float32)


def compute_tsne3d(theta):
    print("\n" + "=" * 80)
    print("[STEP] StandardScaler + PCA + 3D t-SNE")
    print("=" * 80)

    x = scale_theta(theta)

    n_samples, n_features = x.shape
    pca_dim = min(PCA_BEFORE_TSNE_DIM, n_features, n_samples - 1)

    if pca_dim >= 3:
        print(f"[STEP] PCA: {n_features} -> {pca_dim}")
        pca = PCA(n_components=pca_dim, random_state=RANDOM_STATE)
        x = pca.fit_transform(x)
        print(f"[PCA] explained variance ratio sum = {pca.explained_variance_ratio_.sum():.6f}")

    perplexity = min(TSNE_PERPLEXITY, max(5, (n_samples - 1) // 3))

    print(f"[TSNE] n_samples  = {n_samples}")
    print(f"[TSNE] input_dim  = {x.shape[1]}")
    print(f"[TSNE] perplexity = {perplexity}")
    print(f"[TSNE] n_iter     = {TSNE_N_ITER}")

    try:
        tsne = TSNE(
            n_components=3,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            max_iter=TSNE_N_ITER,
            random_state=RANDOM_STATE,
            verbose=1,
        )
    except TypeError:
        tsne = TSNE(
            n_components=3,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            n_iter=TSNE_N_ITER,
            random_state=RANDOM_STATE,
            verbose=1,
        )

    emb = tsne.fit_transform(x)
    return emb.astype(np.float32)


# ============================================================
# 6. 画图
# ============================================================

def set_3d_style(ax):
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")

    # t-SNE 坐标只表示可视化嵌入，不是物理坐标，所以隐藏数值刻度
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
    names,
    colors,
    title,
    out_path,
    marker_size=18,
    alpha=0.72,
):
    fig = plt.figure(figsize=(9, 12))

    views = [
        (18, -65, "(a)"),
        (18, 35, "(b)"),
    ]

    for i, (elev, azim, label) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 1, i, projection="3d")

        for c in sorted(np.unique(y)):
            if int(c) < 0:
                continue

            mask = y == c
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                emb[mask, 2],
                s=marker_size,
                c=colors.get(int(c), "gray"),
                label=names.get(int(c), str(c)),
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

    print(f"[OK] saved -> {out_path}")


def robust_normalize(x):
    x = x.astype(np.float32)
    finite = np.isfinite(x)

    if not np.any(finite):
        return np.zeros_like(x, dtype=np.float32)

    lo, hi = np.percentile(x[finite], [2, 98])

    if hi - lo < 1e-8:
        y = np.zeros_like(x, dtype=np.float32)
    else:
        y = (x - lo) / (hi - lo)
        y = np.clip(y, 0, 1)

    y[~finite] = 0
    return y.astype(np.float32)


def plot_3d_score_two_views(
    emb,
    score,
    title,
    out_path,
    marker_size=18,
    alpha=0.78,
):
    score = robust_normalize(score)

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
        cbar.set_label("2D-ESN OCSVM anomaly score")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] saved -> {out_path}")


# ============================================================
# 7. 主程序
# ============================================================

def main():
    print("=" * 80)
    print("2D-ESN model space 3D t-SNE visualization")
    print("=" * 80)
    print(f"[PROJECT_ROOT] {PROJECT_ROOT}")
    print(f"[DATA_DIR    ] {DATA_DIR}")
    print(f"[THETA_DIR   ] {THETA_DIR}")
    print(f"[OCSVM_DIR   ] {OCSVM_DIR}")
    print(f"[SPLITS      ] {SPLITS_TO_USE}")
    print(f"[PRED_TAG    ] {PRED_TAG}")
    print(f"[OUT_DIR     ] {OUT_DIR}")
    print("=" * 80)

    data = load_all_data()

    theta = data["theta"]
    gt_binary = data["gt_binary"]
    gt_three = data["gt_three"]
    pred_binary = data["pred_binary"]
    scores = data["scores"]

    print_stat("GT binary all", gt_binary, GT_BINARY_NAMES)
    print_stat("GT three-class all", gt_three, GT_THREE_NAMES)

    if data["pred_available"]:
        print_stat("OCSVM pred all", pred_binary, PRED_NAMES)

    # 以 normal / abnormal 二分类做均衡采样
    idx = balanced_sample_indices(gt_binary)

    theta_s = theta[idx]
    gt_binary_s = gt_binary[idx]
    gt_three_s = gt_three[idx]
    pred_binary_s = pred_binary[idx]
    scores_s = scores[idx]

    print("\n" + "=" * 80)
    print("[SAMPLED DATA]")
    print(f"theta_s.shape = {theta_s.shape}")
    print("=" * 80)

    print_stat("GT binary sampled", gt_binary_s, GT_BINARY_NAMES)
    print_stat("GT three-class sampled", gt_three_s, GT_THREE_NAMES)

    if data["pred_available"]:
        print_stat("OCSVM pred sampled", pred_binary_s, PRED_NAMES)

    emb = compute_tsne3d(theta_s)

    split_tag = "_".join(SPLITS_TO_USE)

    # ------------------------------------------------------------
    # 图 1：GT normal / abnormal
    # 最推荐给老师看
    # ------------------------------------------------------------
    out_gt_binary = OUT_DIR / f"2desn_tsne3d_GT_binary_{split_tag}.png"
    plot_3d_two_views(
        emb=emb,
        y=gt_binary_s,
        names=GT_BINARY_NAMES,
        colors=GT_BINARY_COLORS,
        title="3D t-SNE visualization of 2D-ESN model space (GT normal/abnormal)",
        out_path=out_gt_binary,
        marker_size=18,
        alpha=0.72,
    )

    # ------------------------------------------------------------
    # 图 2：GT normal / cavity / utility
    # 作为补充图
    # ------------------------------------------------------------
    out_gt_three = OUT_DIR / f"2desn_tsne3d_GT_threeclass_{split_tag}.png"
    plot_3d_two_views(
        emb=emb,
        y=gt_three_s,
        names=GT_THREE_NAMES,
        colors=GT_THREE_COLORS,
        title="3D t-SNE visualization of 2D-ESN model space (GT classes)",
        out_path=out_gt_three,
        marker_size=18,
        alpha=0.68,
    )

    # ------------------------------------------------------------
    # 图 3：OCSVM 预测 normal / abnormal
    # ------------------------------------------------------------
    if data["pred_available"]:
        out_pred = OUT_DIR / f"2desn_tsne3d_OCSVM_{PRED_TAG}_pred_{split_tag}.png"
        plot_3d_two_views(
            emb=emb,
            y=pred_binary_s,
            names=PRED_NAMES,
            colors=PRED_COLORS,
            title=f"3D t-SNE visualization of 2D-ESN model space (OCSVM {PRED_TAG})",
            out_path=out_pred,
            marker_size=18,
            alpha=0.72,
        )

    # ------------------------------------------------------------
    # 图 4：OCSVM anomaly score 连续颜色图
    # 这个通常比硬分类图更好解释
    # ------------------------------------------------------------
    if data["score_available"]:
        out_score = OUT_DIR / f"2desn_tsne3d_OCSVM_{PRED_TAG}_score_{split_tag}.png"
        plot_3d_score_two_views(
            emb=emb,
            score=scores_s,
            title=f"3D t-SNE visualization of 2D-ESN model space (OCSVM score, {PRED_TAG})",
            out_path=out_score,
            marker_size=18,
            alpha=0.78,
        )

    # 保存 embedding，方便后面复用
    emb_path = OUT_DIR / f"2desn_tsne3d_embedding_{split_tag}_{PRED_TAG}.npz"
    np.savez_compressed(
        emb_path,
        embedding=emb,
        sample_idx=idx,
        theta_sampled=theta_s,
        gt_binary=gt_binary_s,
        gt_three=gt_three_s,
        pred_binary=pred_binary_s,
        scores=scores_s,
    )
    print(f"[OK] saved embedding -> {emb_path}")

    config = {
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR),
        "theta_dir": str(THETA_DIR),
        "ocsvm_dir": str(OCSVM_DIR),
        "splits_to_use": SPLITS_TO_USE,
        "pred_tag": PRED_TAG,
        "max_points_total": MAX_POINTS_TOTAL,
        "max_per_class": MAX_PER_CLASS,
        "pca_before_tsne_dim": PCA_BEFORE_TSNE_DIM,
        "tsne_perplexity": TSNE_PERPLEXITY,
        "tsne_n_iter": TSNE_N_ITER,
        "num_points_all": int(len(theta)),
        "num_points_sampled": int(len(theta_s)),
    }

    config_path = OUT_DIR / f"2desn_tsne3d_config_{split_tag}_{PRED_TAG}.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved config -> {config_path}")

    print("\n" + "=" * 80)
    print("[DONE] 2D-ESN 3D t-SNE 可视化完成")
    print(f"[OUT_DIR] {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()