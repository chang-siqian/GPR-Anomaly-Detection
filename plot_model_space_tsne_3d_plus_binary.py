import json
from pathlib import Path

import numpy as np
import matplotlib

# 只保存图片，不弹窗，避免 PyCharm 后端问题
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ============================================================
# 1. 路径配置：已经按你这次运行结果改好了
# ============================================================

PROJECT_ROOT = Path(r"C:\temporary internet files\GPR_ModelSpace_New")
DATASET_ROOT = PROJECT_ROOT / "gpr_yolo_dataset"

# 你这次实际运行出来的是 overlap40，不是 overlap20
WINDOW_DIR = DATASET_ROOT / "window_dataset_overlap40_w128_s32"

THETA_DIR = (
    WINDOW_DIR
    / "resnet_features"
    / "pca32"
    / "theta_vectors"
)

OCSVM_DIR = THETA_DIR / "ocsvm_baseline"

# 你的 OCSVM 结果文件前缀
NU_TAG = "nu_0p15"
SAVE_TAG = "best_thr"

# 你日志里的 best threshold
BEST_THRESHOLD = 21.161540088865763

# 画哪些 split
# 推荐先画 test，最快，也最适合展示测试结果
# 想画全部就改成 ["train", "val", "test"]
SPLITS_TO_USE = ["test"]

# t-SNE 最多使用多少个点
# test 只有 1320 个点，不会采样
# 如果你改成 train+val+test，点太多时会自动均衡采样
MAX_POINTS = 5000

# t-SNE 参数
RANDOM_STATE = 42
PCA_BEFORE_TSNE_DIM = 50
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1500

# 输出目录
OUT_DIR = WINDOW_DIR / "model_space_vis_tsne3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 类别设置
# ============================================================

# 你的数据集类别：
# labels: 0 normal, 1 abnormal
# cls_ids: -1 normal, 0 cavity, 1 utility
GT_CLASS_NAMES = {
    0: "normal",
    1: "cavity",
    2: "utility",
    3: "abnormal_unknown",
}

GT_CLASS_COLORS = {
    0: "blue",
    1: "red",
    2: "black",
    3: "magenta",
}

# 二分类 GT：
# labels: 0 normal, 1 abnormal
# 这张图不区分 cavity 和 utility，只看 normal / abnormal
GT_BINARY_CLASS_NAMES = {
    0: "normal",
    1: "abnormal",
}

GT_BINARY_CLASS_COLORS = {
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
# 3. 通用工具函数
# ============================================================

def decode_str_array(arr):
    result = []
    for x in arr:
        if isinstance(x, bytes):
            result.append(x.decode("utf-8"))
        else:
            result.append(str(x))
    return np.array(result)


def find_2d_numeric_key(npz, preferred_keys):
    """
    从 npz 里找 theta 矩阵。
    优先用 preferred_keys。
    如果找不到，就自动找二维数值数组。
    """
    keys = list(npz.keys())

    for k in preferred_keys:
        if k in keys:
            arr = npz[k]
            if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                return k

    candidates = []
    for k in keys:
        arr = npz[k]
        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
            candidates.append((k, arr.shape))

    if not candidates:
        raise KeyError(
            f"没有找到二维数值矩阵。当前 npz keys = {keys}"
        )

    # 选列数最大的二维数组，通常就是 theta
    candidates = sorted(candidates, key=lambda x: x[1][1], reverse=True)
    return candidates[0][0]


def find_1d_key(npz, preferred_keys, n=None, dtype_kind=None, avoid_words=None):
    """
    从 npz 中找一维数组。
    dtype_kind:
        "int"   找整数
        "float" 找浮点数
        None    不限制
    avoid_words:
        避开某些名字，比如 label / true / gt
    """
    keys = list(npz.keys())
    avoid_words = avoid_words or []

    for k in preferred_keys:
        if k not in keys:
            continue

        arr = npz[k]
        if arr.ndim != 1:
            continue
        if n is not None and arr.shape[0] != n:
            continue
        if dtype_kind == "int" and not np.issubdtype(arr.dtype, np.integer):
            continue
        if dtype_kind == "float" and not np.issubdtype(arr.dtype, np.floating):
            continue

        return k

    for k in keys:
        name = k.lower()
        if any(w in name for w in avoid_words):
            continue

        arr = npz[k]
        if arr.ndim != 1:
            continue
        if n is not None and arr.shape[0] != n:
            continue
        if dtype_kind == "int" and not np.issubdtype(arr.dtype, np.integer):
            continue
        if dtype_kind == "float" and not np.issubdtype(arr.dtype, np.floating):
            continue

        return k

    return None


def load_window_meta(split, n_expected):
    """
    从 window_dataset 的 train/val/test.npz 中读取标签、类别、图片名等信息。
    这个最稳，因为你的 inspect 日志已经确认这些 key 存在。
    """
    window_npz_path = WINDOW_DIR / f"{split}.npz"
    if not window_npz_path.exists():
        raise FileNotFoundError(f"找不到窗口数据文件: {window_npz_path}")

    data = np.load(window_npz_path, allow_pickle=True)

    labels = data["labels"].astype(int)
    cls_ids = data["cls_ids"].astype(int)
    img_names = decode_str_array(data["img_names"])

    if "window_ids" in data.files:
        window_ids = data["window_ids"].astype(int)
    else:
        window_ids = np.arange(len(labels), dtype=int)

    if len(labels) != n_expected:
        raise ValueError(
            f"{split} 的 theta 数量和 window npz 标签数量不一致："
            f"theta={n_expected}, window_labels={len(labels)}"
        )

    return labels, cls_ids, img_names, window_ids


def build_gt_class(labels, cls_ids):
    """
    生成用于画图的 GT 类别：
        0 normal
        1 cavity
        2 utility
        3 abnormal_unknown
    """
    labels = labels.astype(int)
    cls_ids = cls_ids.astype(int)

    gt_class = np.full_like(labels, fill_value=3, dtype=int)

    # normal
    gt_class[labels == 0] = 0

    # abnormal: cavity / utility
    gt_class[(labels == 1) & (cls_ids == 0)] = 1
    gt_class[(labels == 1) & (cls_ids == 1)] = 2

    return gt_class


def load_theta_split(split):
    """
    读取某个 split 的 theta 向量和对应标签。
    """
    theta_path = THETA_DIR / f"{split}_theta.npz"
    if not theta_path.exists():
        raise FileNotFoundError(f"找不到 theta 文件: {theta_path}")

    data = np.load(theta_path, allow_pickle=True)

    print("\n" + "=" * 80)
    print(f"[LOAD THETA] split = {split}")
    print(f"[FILE      ] {theta_path}")
    print(f"[KEYS      ] {list(data.keys())}")

    theta_key = find_2d_numeric_key(
        data,
        preferred_keys=[
            "theta",
            "thetas",
            "theta_vectors",
            "X",
            "X_theta",
            "vectors",
        ],
    )

    theta = data[theta_key].astype(np.float32)
    n = theta.shape[0]

    print(f"[THETA KEY ] {theta_key}")
    print(f"[THETA     ] shape = {theta.shape}, dtype = {theta.dtype}")

    # 优先从 theta 文件里读标签；如果没有，就从 window npz 里读
    label_key = find_1d_key(
        data,
        preferred_keys=["labels", "y", "y_true", "gt_labels"],
        n=n,
        dtype_kind="int",
    )

    cls_key = find_1d_key(
        data,
        preferred_keys=["cls_ids", "class_ids", "classes"],
        n=n,
        dtype_kind="int",
    )

    img_key = find_1d_key(
        data,
        preferred_keys=["img_names", "image_names", "names"],
        n=n,
        dtype_kind=None,
    )

    win_key = find_1d_key(
        data,
        preferred_keys=["window_ids", "win_ids"],
        n=n,
        dtype_kind="int",
    )

    if label_key is not None and cls_key is not None:
        labels = data[label_key].astype(int)
        cls_ids = data[cls_key].astype(int)

        if img_key is not None:
            img_names = decode_str_array(data[img_key])
        else:
            img_names = np.array([f"{split}_unknown"] * n)

        if win_key is not None:
            window_ids = data[win_key].astype(int)
        else:
            window_ids = np.arange(n, dtype=int)

        print(f"[META      ] loaded from theta npz")
    else:
        labels, cls_ids, img_names, window_ids = load_window_meta(split, n)
        print(f"[META      ] loaded from window npz")

    gt_class = build_gt_class(labels, cls_ids)

    unique, counts = np.unique(gt_class, return_counts=True)
    stat = {GT_CLASS_NAMES.get(int(k), str(k)): int(v) for k, v in zip(unique, counts)}
    print(f"[GT CLASS  ] {stat}")

    split_arr = np.array([split] * n)

    return {
        "theta": theta,
        "labels": labels,
        "cls_ids": cls_ids,
        "gt_class": gt_class,
        "img_names": img_names,
        "window_ids": window_ids,
        "split": split_arr,
    }


def pred_to_binary_abnormal(pred):
    """
    将预测结果转成：
        0 predicted normal
        1 predicted abnormal

    兼容两种格式：
        OCSVM 原生：-1 abnormal, 1 normal
        二分类格式：1 abnormal, 0 normal
    """
    pred = pred.astype(int)
    unique = set(np.unique(pred).tolist())

    if -1 in unique:
        return (pred == -1).astype(int)
    else:
        return (pred == 1).astype(int)


def load_prediction_split(split, n_expected):
    """
    读取 OCSVM 预测结果。
    如果找不到预测文件，返回 None。
    """
    pred_path = OCSVM_DIR / f"{NU_TAG}_{SAVE_TAG}_{split}_predictions.npz"

    if not pred_path.exists():
        print(f"[WARN] 找不到预测文件，跳过预测图: {pred_path}")
        return None

    data = np.load(pred_path, allow_pickle=True)

    print("\n" + "-" * 80)
    print(f"[LOAD PRED] split = {split}")
    print(f"[FILE     ] {pred_path}")
    print(f"[KEYS     ] {list(data.keys())}")

    pred_key = find_1d_key(
        data,
        preferred_keys=[
            "y_pred",
            "pred",
            "preds",
            "pred_labels",
            "predictions",
            "test_pred",
            "val_pred",
            "train_pred",
        ],
        n=n_expected,
        dtype_kind="int",
        avoid_words=["label", "true", "gt", "target"],
    )

    if pred_key is not None:
        pred = data[pred_key].astype(int)
        pred_binary = pred_to_binary_abnormal(pred)
        print(f"[PRED KEY] {pred_key}, unique = {np.unique(pred)}")
        return pred_binary

    # 如果没有 pred，就尝试用 score + threshold 算
    score_key = find_1d_key(
        data,
        preferred_keys=[
            "scores",
            "score",
            "decision_scores",
            "ocsvm_scores",
            "test_scores",
            "val_scores",
            "train_scores",
            "raw_scores",
        ],
        n=n_expected,
        dtype_kind="float",
    )

    if score_key is not None:
        scores = data[score_key].astype(float)

        # 你当前 run_ocsvm 的逻辑是 score <= threshold 判为异常
        pred_binary = (scores <= BEST_THRESHOLD).astype(int)

        print(f"[SCORE KEY] {score_key}")
        print(f"[INFO     ] no pred key, use scores <= {BEST_THRESHOLD:.6f}")
        return pred_binary

    print("[WARN] 预测文件里没有找到 pred 或 score，跳过预测图")
    return None


def balanced_sample_indices(class_ids, max_points, random_state=42):
    """
    类别均衡采样，避免 normal 太多、异常类太少导致图不好看。
    """
    n = len(class_ids)
    if n <= max_points:
        return np.arange(n)

    rng = np.random.default_rng(random_state)
    unique_classes = np.unique(class_ids)

    per_class = max(1, max_points // len(unique_classes))

    selected = []
    remaining_pool = []

    for c in unique_classes:
        idx = np.where(class_ids == c)[0]
        rng.shuffle(idx)

        take = min(per_class, len(idx))
        selected.extend(idx[:take].tolist())

        if take < len(idx):
            remaining_pool.extend(idx[take:].tolist())

    selected = np.array(selected, dtype=int)

    if len(selected) < max_points and len(remaining_pool) > 0:
        remaining_pool = np.array(remaining_pool, dtype=int)
        rng.shuffle(remaining_pool)

        need = max_points - len(selected)
        extra = remaining_pool[:need]
        selected = np.concatenate([selected, extra])

    rng.shuffle(selected)
    return selected[:max_points]


def compute_tsne_3d(theta):
    """
    标准化 + PCA 预降维 + 3D t-SNE。
    """
    print("\n" + "=" * 80)
    print("[STEP] StandardScaler")
    print("=" * 80)

    X = StandardScaler().fit_transform(theta)

    n_samples, n_features = X.shape

    pca_dim = min(PCA_BEFORE_TSNE_DIM, n_features, n_samples - 1)

    if pca_dim >= 3:
        print("\n" + "=" * 80)
        print(f"[STEP] PCA before t-SNE: {n_features} -> {pca_dim}")
        print("=" * 80)

        pca = PCA(n_components=pca_dim, random_state=RANDOM_STATE)
        X_reduced = pca.fit_transform(X)

        print(f"[INFO] PCA explained variance ratio sum = {pca.explained_variance_ratio_.sum():.6f}")
    else:
        X_reduced = X

    perplexity = min(TSNE_PERPLEXITY, max(5, (n_samples - 1) // 3))

    print("\n" + "=" * 80)
    print("[STEP] t-SNE 3D")
    print(f"[INFO] n_samples   = {n_samples}")
    print(f"[INFO] input_dim   = {X_reduced.shape[1]}")
    print(f"[INFO] perplexity  = {perplexity}")
    print(f"[INFO] n_iter      = {TSNE_N_ITER}")
    print("=" * 80)

    # sklearn 新版本用 max_iter，旧版本用 n_iter，这里做兼容
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

    emb = tsne.fit_transform(X_reduced)
    return emb.astype(np.float32)


def set_3d_axis_style(ax):
    """
    调整 3D 图样式，让它更像论文里的模型空间图。
    """
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")

    # 不显示具体数值，强调这只是可视化坐标
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
    class_ids,
    class_names,
    class_colors,
    out_path,
    title,
    marker_size=18,
    alpha=0.88,
):
    """
    画两种视角的 3D t-SNE 图，类似论文里面的 (a)(b)。
    """
    fig = plt.figure(figsize=(9, 12))

    views = [
        (18, -65, "(a)"),
        (18, 35, "(b)"),
    ]

    for i, (elev, azim, sub_label) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 1, i, projection="3d")

        unique_classes = sorted(np.unique(class_ids).tolist())

        for c in unique_classes:
            mask = class_ids == c
            if not np.any(mask):
                continue

            name = class_names.get(int(c), str(c))
            color = class_colors.get(int(c), "gray")

            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                emb[mask, 2],
                s=marker_size,
                c=color,
                label=name,
                alpha=alpha,
                depthshade=False,
            )

        ax.view_init(elev=elev, azim=azim)
        set_3d_axis_style(ax)
        ax.legend(loc="upper right", fontsize=8, frameon=True)
        ax.set_title(title, fontsize=11)
        ax.text2D(0.48, -0.08, sub_label, transform=ax.transAxes, fontsize=16)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] saved figure -> {out_path}")


# ============================================================
# 4. 主程序
# ============================================================

def main():
    print("=" * 80)
    print("GPR Model Space 3D t-SNE Visualization")
    print("=" * 80)
    print(f"[PROJECT_ROOT] {PROJECT_ROOT}")
    print(f"[WINDOW_DIR  ] {WINDOW_DIR}")
    print(f"[THETA_DIR   ] {THETA_DIR}")
    print(f"[OCSVM_DIR   ] {OCSVM_DIR}")
    print(f"[SPLITS      ] {SPLITS_TO_USE}")
    print(f"[OUT_DIR     ] {OUT_DIR}")
    print("=" * 80)

    all_theta = []
    all_labels = []
    all_cls_ids = []
    all_gt_class = []
    all_img_names = []
    all_window_ids = []
    all_splits = []
    all_pred_binary = []

    pred_available_for_all = True

    for split in SPLITS_TO_USE:
        item = load_theta_split(split)

        theta = item["theta"]
        n = theta.shape[0]

        pred_binary = load_prediction_split(split, n)
        if pred_binary is None:
            pred_available_for_all = False
            pred_binary = np.full(n, fill_value=-1, dtype=int)

        all_theta.append(theta)
        all_labels.append(item["labels"])
        all_cls_ids.append(item["cls_ids"])
        all_gt_class.append(item["gt_class"])
        all_img_names.append(item["img_names"])
        all_window_ids.append(item["window_ids"])
        all_splits.append(item["split"])
        all_pred_binary.append(pred_binary)

    theta = np.concatenate(all_theta, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    cls_ids = np.concatenate(all_cls_ids, axis=0)
    gt_class = np.concatenate(all_gt_class, axis=0)
    img_names = np.concatenate(all_img_names, axis=0)
    window_ids = np.concatenate(all_window_ids, axis=0)
    split_arr = np.concatenate(all_splits, axis=0)
    pred_binary = np.concatenate(all_pred_binary, axis=0)

    print("\n" + "=" * 80)
    print("[MERGED DATA]")
    print(f"theta.shape      = {theta.shape}")
    print(f"labels.shape     = {labels.shape}")
    print(f"cls_ids.shape    = {cls_ids.shape}")
    print(f"gt_class.shape   = {gt_class.shape}")
    print(f"img_names.shape  = {img_names.shape}")
    print(f"window_ids.shape = {window_ids.shape}")
    print("=" * 80)

    gt_unique, gt_counts = np.unique(gt_class, return_counts=True)
    gt_stat = {
        GT_CLASS_NAMES.get(int(k), str(k)): int(v)
        for k, v in zip(gt_unique, gt_counts)
    }
    print("[GT CLASS STAT]", gt_stat)

    # ------------------------------------------------------------
    # 类别均衡采样
    # ------------------------------------------------------------
    sample_idx = balanced_sample_indices(
        gt_class,
        max_points=MAX_POINTS,
        random_state=RANDOM_STATE,
    )

    theta_s = theta[sample_idx]
    labels_s = labels[sample_idx]
    cls_ids_s = cls_ids[sample_idx]
    gt_class_s = gt_class[sample_idx]
    img_names_s = img_names[sample_idx]
    window_ids_s = window_ids[sample_idx]
    split_s = split_arr[sample_idx]
    pred_binary_s = pred_binary[sample_idx]

    print("\n" + "=" * 80)
    print("[SAMPLED DATA]")
    print(f"sampled theta.shape = {theta_s.shape}")
    print(f"sampled points      = {len(sample_idx)} / {len(theta)}")
    print("=" * 80)

    gt_unique_s, gt_counts_s = np.unique(gt_class_s, return_counts=True)
    gt_stat_s = {
        GT_CLASS_NAMES.get(int(k), str(k)): int(v)
        for k, v in zip(gt_unique_s, gt_counts_s)
    }
    print("[SAMPLED GT CLASS STAT]", gt_stat_s)

    # ------------------------------------------------------------
    # 计算 3D t-SNE
    # ------------------------------------------------------------
    emb = compute_tsne_3d(theta_s)

    split_tag = "_".join(SPLITS_TO_USE)

    # 保存 embedding 数据，方便以后复用
    emb_path = OUT_DIR / f"tsne3d_embedding_{split_tag}.npz"
    np.savez_compressed(
        emb_path,
        embedding=emb,
        theta_sampled=theta_s,
        labels=labels_s,
        cls_ids=cls_ids_s,
        gt_class=gt_class_s,
        pred_binary=pred_binary_s,
        img_names=img_names_s,
        window_ids=window_ids_s,
        split=split_s,
        sample_idx=sample_idx,
    )
    print(f"[OK] saved embedding -> {emb_path}")

    # 保存配置
    config = {
        "project_root": str(PROJECT_ROOT),
        "window_dir": str(WINDOW_DIR),
        "theta_dir": str(THETA_DIR),
        "ocsvm_dir": str(OCSVM_DIR),
        "splits_to_use": SPLITS_TO_USE,
        "max_points": MAX_POINTS,
        "random_state": RANDOM_STATE,
        "pca_before_tsne_dim": PCA_BEFORE_TSNE_DIM,
        "tsne_perplexity": TSNE_PERPLEXITY,
        "tsne_n_iter": TSNE_N_ITER,
        "nu_tag": NU_TAG,
        "save_tag": SAVE_TAG,
        "best_threshold": BEST_THRESHOLD,
        "gt_class_names": GT_CLASS_NAMES,
        "gt_class_stat_all": gt_stat,
        "gt_class_stat_sampled": gt_stat_s,
    }

    config_path = OUT_DIR / f"tsne3d_config_{split_tag}.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved config -> {config_path}")

    # ------------------------------------------------------------
    # 图 1：按 GT 二分类画
    # normal / abnormal，不区分 cavity 和 utility
    # ------------------------------------------------------------
    out_gt_binary = OUT_DIR / f"model_space_tsne3d_GT_binary_{split_tag}.png"
    plot_3d_two_views(
        emb=emb,
        class_ids=labels_s,
        class_names=GT_BINARY_CLASS_NAMES,
        class_colors=GT_BINARY_CLASS_COLORS,
        out_path=out_gt_binary,
        title="3D t-SNE visualization of model space (GT normal/abnormal)",
    )

    # ------------------------------------------------------------
    # 图 2：按 GT 三分类画
    # normal / cavity / utility
    # ------------------------------------------------------------
    out_gt = OUT_DIR / f"model_space_tsne3d_GT_{split_tag}.png"
    plot_3d_two_views(
        emb=emb,
        class_ids=gt_class_s,
        class_names=GT_CLASS_NAMES,
        class_colors=GT_CLASS_COLORS,
        out_path=out_gt,
        title="3D t-SNE visualization of model space (GT labels)",
    )

    # ------------------------------------------------------------
    # 图 3：按 OCSVM 预测结果画
    # ------------------------------------------------------------
    if pred_available_for_all:
        out_pred = OUT_DIR / f"model_space_tsne3d_OCSVM_pred_{split_tag}.png"
        plot_3d_two_views(
            emb=emb,
            class_ids=pred_binary_s,
            class_names=PRED_CLASS_NAMES,
            class_colors=PRED_CLASS_COLORS,
            out_path=out_pred,
            title=f"3D t-SNE visualization of model space (OCSVM {NU_TAG}, {SAVE_TAG})",
        )
    else:
        print("[WARN] 因为部分 split 没有预测文件，所以不画 OCSVM 预测图。")

    print("\n" + "=" * 80)
    print("[DONE] 模型空间 3D t-SNE 可视化完成")
    print(f"[OUT_DIR] {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()