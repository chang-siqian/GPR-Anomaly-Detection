import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ============================================================
# 1. 路径配置
# ============================================================

PROJECT_ROOT = Path(r"C:\temporary internet files\GPR_ModelSpace_New")
DATASET_ROOT = PROJECT_ROOT / "gpr_yolo_dataset"

WINDOW_DIR = DATASET_ROOT / "window_dataset_overlap40_w128_s32"

THETA_DIR = (
    WINDOW_DIR
    / "resnet_features"
    / "pca32"
    / "theta_vectors"
)

OCSVM_DIR = THETA_DIR / "ocsvm_baseline"

SPLIT = "test"

NU_TAG = "nu_0p15"
SAVE_TAG = "best_thr"
BEST_THRESHOLD = 21.161540088865763

THETA_NPZ = THETA_DIR / f"{SPLIT}_theta.npz"
WINDOW_NPZ = WINDOW_DIR / f"{SPLIT}.npz"
PRED_NPZ = OCSVM_DIR / f"{NU_TAG}_{SAVE_TAG}_{SPLIT}_predictions.npz"

OUT_DIR = WINDOW_DIR / "model_space_vis_tsne2d_clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 可视化参数
# ============================================================

RANDOM_STATE = 42

# 为了图不要太乱，二分类时最多采样这些点
MAX_NORMAL = 500
MAX_ABNORMAL = 500

# 三分类图里每类最多采样这些点
MAX_PER_CLASS = 350

PCA_DIM = 50
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1500


# ============================================================
# 3. 工具函数
# ============================================================

def decode_name(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def find_theta_key(npz):
    preferred = ["theta", "thetas", "theta_vectors", "X", "X_theta", "vectors"]

    for k in preferred:
        if k in npz.files:
            arr = npz[k]
            if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                return k

    candidates = []
    for k in npz.files:
        arr = npz[k]
        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
            candidates.append((k, arr.shape))

    if not candidates:
        raise KeyError(f"没有找到 theta 二维矩阵，当前 keys = {list(npz.files)}")

    candidates = sorted(candidates, key=lambda x: x[1][1], reverse=True)
    return candidates[0][0]


def find_1d_key(npz, preferred, n=None, dtype_kind=None, avoid_words=None):
    avoid_words = avoid_words or []

    for k in preferred:
        if k not in npz.files:
            continue

        arr = npz[k]
        if arr.ndim != 1:
            continue
        if n is not None and len(arr) != n:
            continue
        if dtype_kind == "int" and not np.issubdtype(arr.dtype, np.integer):
            continue
        if dtype_kind == "float" and not np.issubdtype(arr.dtype, np.floating):
            continue

        return k

    for k in npz.files:
        name = k.lower()
        if any(w in name for w in avoid_words):
            continue

        arr = npz[k]
        if arr.ndim != 1:
            continue
        if n is not None and len(arr) != n:
            continue
        if dtype_kind == "int" and not np.issubdtype(arr.dtype, np.integer):
            continue
        if dtype_kind == "float" and not np.issubdtype(arr.dtype, np.floating):
            continue

        return k

    return None


def load_theta_and_meta():
    print("=" * 80)
    print("[THETA_NPZ ]", THETA_NPZ)
    print("[WINDOW_NPZ]", WINDOW_NPZ)
    print("=" * 80)

    if not THETA_NPZ.exists():
        raise FileNotFoundError(f"找不到 theta 文件: {THETA_NPZ}")

    if not WINDOW_NPZ.exists():
        raise FileNotFoundError(f"找不到 window 文件: {WINDOW_NPZ}")

    theta_data = np.load(THETA_NPZ, allow_pickle=True)
    win_data = np.load(WINDOW_NPZ, allow_pickle=True)

    print("[THETA KEYS ]", list(theta_data.files))
    print("[WINDOW KEYS]", list(win_data.files))

    theta_key = find_theta_key(theta_data)
    theta = theta_data[theta_key].astype(np.float32)

    labels = win_data["labels"].astype(int)
    cls_ids = win_data["cls_ids"].astype(int)
    img_names = np.array([decode_name(x) for x in win_data["img_names"]])

    print("[THETA KEY]", theta_key)
    print("[THETA    ]", theta.shape)
    print("[LABELS   ]", labels.shape)
    print("[CLS_IDS  ]", cls_ids.shape)

    if len(theta) != len(labels):
        raise ValueError(f"theta 数量和 labels 数量不一致: {len(theta)} vs {len(labels)}")

    return theta, labels, cls_ids, img_names


def build_gt_binary(labels):
    """
    0 = normal
    1 = abnormal
    """
    return labels.astype(int)


def build_gt_threeclass(labels, cls_ids):
    """
    0 = normal
    1 = cavity
    2 = utility
    3 = unknown abnormal
    """
    y = np.full_like(labels, 3, dtype=int)

    y[labels == 0] = 0
    y[(labels == 1) & (cls_ids == 0)] = 1
    y[(labels == 1) & (cls_ids == 1)] = 2

    return y


def load_ocsvm_pred_and_score(n):
    if not PRED_NPZ.exists():
        print("[WARN] 找不到预测文件:", PRED_NPZ)
        return None, None

    data = np.load(PRED_NPZ, allow_pickle=True)

    print("=" * 80)
    print("[PRED_NPZ ]", PRED_NPZ)
    print("[PRED KEYS]", list(data.files))
    print("=" * 80)

    pred_key = find_1d_key(
        data,
        preferred=[
            "y_pred",
            "pred",
            "preds",
            "pred_labels",
            "predictions",
            "test_pred",
        ],
        n=n,
        dtype_kind="int",
        avoid_words=["label", "true", "gt", "target"],
    )

    score_key = find_1d_key(
        data,
        preferred=[
            "scores",
            "score",
            "decision_scores",
            "ocsvm_scores",
            "test_scores",
            "raw_scores",
        ],
        n=n,
        dtype_kind="float",
    )

    pred_binary = None
    scores = None

    if pred_key is not None:
        pred = data[pred_key].astype(int)
        unique = set(np.unique(pred).tolist())

        if -1 in unique:
            pred_binary = (pred == -1).astype(int)
        else:
            pred_binary = (pred == 1).astype(int)

        print("[PRED KEY ]", pred_key, "unique =", np.unique(pred))
    else:
        print("[PRED KEY ] not found")

    if score_key is not None:
        scores = data[score_key].astype(np.float32)
        print("[SCORE KEY]", score_key)
    else:
        print("[SCORE KEY] not found")

    if pred_binary is None and scores is not None:
        pred_binary = (scores <= BEST_THRESHOLD).astype(int)
        print("[INFO] pred not found, use scores <= BEST_THRESHOLD")

    return pred_binary, scores


def balanced_indices_binary(y_binary):
    rng = np.random.default_rng(RANDOM_STATE)

    idx_normal = np.where(y_binary == 0)[0]
    idx_abnormal = np.where(y_binary == 1)[0]

    rng.shuffle(idx_normal)
    rng.shuffle(idx_abnormal)

    idx_normal = idx_normal[:min(MAX_NORMAL, len(idx_normal))]
    idx_abnormal = idx_abnormal[:min(MAX_ABNORMAL, len(idx_abnormal))]

    idx = np.concatenate([idx_normal, idx_abnormal])
    rng.shuffle(idx)

    return idx


def balanced_indices_multiclass(y_class):
    rng = np.random.default_rng(RANDOM_STATE)
    selected = []

    for c in sorted(np.unique(y_class)):
        idx = np.where(y_class == c)[0]
        rng.shuffle(idx)
        selected.extend(idx[:min(MAX_PER_CLASS, len(idx))].tolist())

    selected = np.array(selected, dtype=int)
    rng.shuffle(selected)

    return selected


def compute_tsne2d(theta):
    print("=" * 80)
    print("[STEP] StandardScaler + PCA + t-SNE 2D")
    print("=" * 80)

    X = StandardScaler().fit_transform(theta)

    pca_dim = min(PCA_DIM, X.shape[1], X.shape[0] - 1)

    if pca_dim >= 3:
        pca = PCA(n_components=pca_dim, random_state=RANDOM_STATE)
        X = pca.fit_transform(X)
        print(f"[PCA] dim = {pca_dim}, explained ratio = {pca.explained_variance_ratio_.sum():.6f}")

    perplexity = min(TSNE_PERPLEXITY, max(5, (len(X) - 1) // 3))

    print("[TSNE] samples =", len(X))
    print("[TSNE] perplexity =", perplexity)

    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            max_iter=TSNE_N_ITER,
            random_state=RANDOM_STATE,
            verbose=1,
        )
    except TypeError:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            n_iter=TSNE_N_ITER,
            random_state=RANDOM_STATE,
            verbose=1,
        )

    emb = tsne.fit_transform(X)
    return emb.astype(np.float32)


def plot_scatter_2d(emb, y, names, colors, title, out_path, s=18, alpha=0.55):
    plt.figure(figsize=(8, 6))

    for c in sorted(np.unique(y)):
        mask = y == c
        plt.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=s,
            alpha=alpha,
            c=colors.get(int(c), "gray"),
            label=names.get(int(c), str(c)),
            edgecolors="none",
        )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(frameon=True)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("[OK] saved ->", out_path)


def robust_normalize(x):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [2, 98])

    if hi - lo < 1e-8:
        return np.zeros_like(x)

    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1)


def plot_score_2d(emb, score, title, out_path):
    plt.figure(figsize=(8, 6))

    plt.scatter(
        emb[:, 0],
        emb[:, 1],
        c=score,
        s=18,
        alpha=0.65,
        cmap="jet",
        edgecolors="none",
    )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(label="anomaly strength")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("[OK] saved ->", out_path)


def print_class_stat(name, y, names):
    unique, counts = np.unique(y, return_counts=True)
    stat = {names.get(int(k), str(k)): int(v) for k, v in zip(unique, counts)}
    print(f"[{name}]", stat)


# ============================================================
# 4. 主程序
# ============================================================

def main():
    theta, labels, cls_ids, img_names = load_theta_and_meta()

    y_binary = build_gt_binary(labels)
    y_three = build_gt_threeclass(labels, cls_ids)

    pred_binary, scores = load_ocsvm_pred_and_score(len(theta))

    binary_names = {
        0: "normal",
        1: "abnormal",
    }

    binary_colors = {
        0: "blue",
        1: "red",
    }

    three_names = {
        0: "normal",
        1: "cavity",
        2: "utility",
        3: "abnormal_unknown",
    }

    three_colors = {
        0: "blue",
        1: "red",
        2: "black",
        3: "magenta",
    }

    pred_names = {
        0: "pred_normal",
        1: "pred_abnormal",
    }

    pred_colors = {
        0: "blue",
        1: "red",
    }

    print_class_stat("GT binary all", y_binary, binary_names)
    print_class_stat("GT three-class all", y_three, three_names)

    # ========================================================
    # 图 1：最推荐，GT normal vs abnormal
    # ========================================================

    idx_binary = balanced_indices_binary(y_binary)

    theta_b = theta[idx_binary]
    y_binary_b = y_binary[idx_binary]

    print_class_stat("GT binary sampled", y_binary_b, binary_names)

    emb_b = compute_tsne2d(theta_b)

    out1 = OUT_DIR / f"clean_tsne2d_GT_binary_{SPLIT}.png"
    plot_scatter_2d(
        emb=emb_b,
        y=y_binary_b,
        names=binary_names,
        colors=binary_colors,
        title="2D t-SNE of model space: normal vs abnormal (GT)",
        out_path=out1,
        s=20,
        alpha=0.55,
    )

    # ========================================================
    # 图 2：三分类，只作为补充
    # ========================================================

    idx_three = balanced_indices_multiclass(y_three)

    theta_t = theta[idx_three]
    y_three_t = y_three[idx_three]

    print_class_stat("GT three-class sampled", y_three_t, three_names)

    emb_t = compute_tsne2d(theta_t)

    out2 = OUT_DIR / f"clean_tsne2d_GT_threeclass_{SPLIT}.png"
    plot_scatter_2d(
        emb=emb_t,
        y=y_three_t,
        names=three_names,
        colors=three_colors,
        title="2D t-SNE of model space: normal / cavity / utility (GT)",
        out_path=out2,
        s=18,
        alpha=0.50,
    )

    # ========================================================
    # 图 3：OCSVM 预测 normal vs abnormal
    # 使用和图 1 同一批点，方便对比
    # ========================================================

    if pred_binary is not None:
        pred_binary_b = pred_binary[idx_binary]

        print_class_stat("OCSVM pred sampled", pred_binary_b, pred_names)

        out3 = OUT_DIR / f"clean_tsne2d_OCSVM_pred_{SPLIT}.png"
        plot_scatter_2d(
            emb=emb_b,
            y=pred_binary_b,
            names=pred_names,
            colors=pred_colors,
            title=f"2D t-SNE of model space: OCSVM prediction ({NU_TAG}, {SAVE_TAG})",
            out_path=out3,
            s=20,
            alpha=0.55,
        )

    # ========================================================
    # 图 4：异常分数连续色彩图
    # 这个通常比 pred 图更好看，也更容易解释
    # ========================================================

    if scores is not None:
        # 你的当前逻辑是 score <= threshold 越异常
        anomaly_strength = BEST_THRESHOLD - scores
        anomaly_strength = robust_normalize(anomaly_strength)

        score_b = anomaly_strength[idx_binary]

        out4 = OUT_DIR / f"clean_tsne2d_OCSVM_score_{SPLIT}.png"
        plot_score_2d(
            emb=emb_b,
            score=score_b,
            title="2D t-SNE of model space: OCSVM anomaly strength",
            out_path=out4,
        )

    # 保存采样后的 embedding，方便之后复用
    save_path = OUT_DIR / f"clean_tsne2d_embedding_{SPLIT}.npz"
    np.savez_compressed(
        save_path,
        emb_binary=emb_b,
        idx_binary=idx_binary,
        y_binary=y_binary_b,
        emb_three=emb_t,
        idx_three=idx_three,
        y_three=y_three_t,
    )

    print("=" * 80)
    print("[DONE] clean 2D t-SNE 可视化完成")
    print("[OUT_DIR]", OUT_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()