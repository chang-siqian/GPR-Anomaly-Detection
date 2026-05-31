# -*- coding: utf-8 -*-
"""
plot_modelspace_loss_like_curves.py

作用：
为 ResNet18 模型空间流程生成“loss-like”曲线：

1. PCA reconstruction error curve
2. Local dynamic model fitting loss curve
3. OCSVM validation objective curve

注意：
这些不是 YOLO 那种 epoch-wise loss，
而是模型空间流程中各模块对应的训练/拟合误差曲线。
"""

from pathlib import Path
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


# ============================================================
# 1. 路径配置
# ============================================================

PROJECT_ROOT = Path(r"C:\temporary internet files\GPR_ModelSpace_New")

# 你之前常用的几个结果目录，脚本会自动找存在的那个
CANDIDATE_EXP_DIRS = [
    PROJECT_ROOT / "gpr_yolo_dataset" / "window_dataset_overlap40_w128_s32",
    PROJECT_ROOT / "gpr_yolo_dataset" / "window_dataset_overlap20_w128_s32",
    PROJECT_ROOT / "gpr_yolo_dataset" / "window_dataset_center2_w128_s32",
]

RIDGE_LAMBDA = 1e-3
ROLLING_WINDOW = 100


# ============================================================
# 2. 通用工具函数
# ============================================================

def find_existing_exp_dir():
    for d in CANDIDATE_EXP_DIRS:
        if d.exists():
            print(f"[OK] 使用实验目录: {d}")
            return d
    raise FileNotFoundError(
        "没有找到实验目录，请检查下面这些路径是否存在：\n"
        + "\n".join(str(p) for p in CANDIDATE_EXP_DIRS)
    )


def find_npz_file(root: Path, patterns):
    for pattern in patterns:
        files = list(root.rglob(pattern))
        if files:
            files = sorted(files)
            print(f"[OK] 找到文件: {files[0]}")
            return files[0]
    raise FileNotFoundError(f"在 {root} 下找不到这些文件: {patterns}")


def load_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    print(f"\n[LOAD] {path}")
    print("keys =", list(data.keys()))
    return data


def get_array(data, candidates, required=True):
    keys = list(data.keys())
    for k in candidates:
        if k in keys:
            return data[k], k
    if required:
        raise KeyError(
            f"没有找到字段 {candidates}\n"
            f"当前 npz 实际字段为: {keys}\n"
            f"把这行 keys 发给我，我可以帮你改。"
        )
    return None, None


def rolling_mean(x, window=100):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def save_fig(out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved -> {out_path}")


# ============================================================
# 3. PCA reconstruction error curve
# ============================================================

def plot_pca_reconstruction_error(exp_dir: Path, fig_dir: Path):
    """
    PCA 没有 epoch loss。
    所以这里画：随着 PCA 维度增加，剩余未解释方差 1 - cumulative explained variance。
    """

    pca_files = list(exp_dir.rglob("*pca*.joblib")) + list(exp_dir.rglob("*pca*.pkl"))
    if not pca_files:
        print("[WARN] 没找到 PCA 模型文件，跳过 PCA reconstruction error 图。")
        return

    pca_path = sorted(pca_files)[0]
    print(f"\n[PCA MODEL] {pca_path}")

    pca = joblib.load(pca_path)

    if not hasattr(pca, "explained_variance_ratio_"):
        print("[WARN] 这个 PCA 文件没有 explained_variance_ratio_，跳过。")
        return

    evr = np.asarray(pca.explained_variance_ratio_)
    cum_evr = np.cumsum(evr)
    reconstruction_error = 1.0 - cum_evr
    dims = np.arange(1, len(evr) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(dims, reconstruction_error, marker="o")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Reconstruction Error Proxy: 1 - Cumulative Explained Variance")
    plt.title("PCA Reconstruction Error Curve")
    plt.grid(alpha=0.3)

    # 标出最终维度
    final_dim = len(evr)
    final_error = reconstruction_error[-1]
    plt.scatter([final_dim], [final_error], s=80)
    plt.text(final_dim, final_error, f"  dim={final_dim}, error={final_error:.4f}")

    save_fig(fig_dir, "01_pca_reconstruction_error_curve.png")


# ============================================================
# 4. Local dynamic model fitting loss curve
# ============================================================

def fit_ridge_and_mse_one_window(seq, ridge_lambda=1e-3):
    """
    输入 seq: [T, D]
    拟合 next-step prediction:
        z_{t+1} = A z_t + b
    返回该窗口的 MSE。
    """
    seq = np.asarray(seq, dtype=np.float64)

    if seq.ndim != 2:
        raise ValueError(f"单个窗口特征应该是 [T, D]，但拿到的是 {seq.shape}")

    X = seq[:-1]      # [T-1, D]
    Y = seq[1:]       # [T-1, D]

    # 加 bias
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    Xb = np.concatenate([X, ones], axis=1)  # [T-1, D+1]

    D_in = Xb.shape[1]
    I = np.eye(D_in, dtype=np.float64)
    I[-1, -1] = 0.0  # bias 不正则化

    # theta = (X^T X + lambda I)^-1 X^T Y
    theta = np.linalg.solve(Xb.T @ Xb + ridge_lambda * I, Xb.T @ Y)

    Y_pred = Xb @ theta
    mse = np.mean((Y_pred - Y) ** 2)

    return float(mse)


def plot_dynamic_model_fitting_loss(exp_dir: Path, fig_dir: Path):
    """
    读取 PCA 后的序列特征，重新计算每个窗口的 ridge fitting MSE。
    这就是“局部动态模型拟合 loss”。
    """

    # 自动找 PCA 后的 train/val/test feature 文件
    pca_dir_candidates = [
        exp_dir / "resnet_features" / "pca32",
        exp_dir / "resnet_features",
        exp_dir,
    ]

    pca_dir = None
    for d in pca_dir_candidates:
        if d.exists():
            pca_dir = d
            break

    if pca_dir is None:
        print("[WARN] 没找到 resnet_features/pca32 目录，跳过 dynamic model loss 图。")
        return

    split_losses = {}

    for split in ["train", "val", "test"]:
        try:
            feature_file = find_npz_file(
                pca_dir,
                [
                    f"{split}*pca*.npz",
                    f"{split}_features.npz",
                    f"{split}*.npz",
                ],
            )
        except FileNotFoundError:
            print(f"[WARN] 没找到 {split} 的 PCA feature 文件，跳过 {split}。")
            continue

        data = load_npz(feature_file)

        features, feat_key = get_array(
            data,
            [
                "features",
                "pca_features",
                "X",
                "feat",
                "resnet_features",
            ],
        )

        features = np.asarray(features)
        print(f"[{split}] feature key = {feat_key}, shape = {features.shape}")

        if features.ndim != 3:
            print(
                f"[WARN] {split} features 不是 [N,T,D]，而是 {features.shape}，"
                f"无法计算局部动态模型 fitting MSE。"
            )
            continue

        losses = []
        for i in range(features.shape[0]):
            mse = fit_ridge_and_mse_one_window(features[i], ridge_lambda=RIDGE_LAMBDA)
            losses.append(mse)

        losses = np.asarray(losses, dtype=float)
        split_losses[split] = losses

        print(
            f"[{split}] dynamic fitting MSE: "
            f"mean={losses.mean():.6f}, min={losses.min():.6f}, max={losses.max():.6f}"
        )

        # 单独保存每个 split 的 loss
        np.savez(
            fig_dir / f"{split}_dynamic_model_fitting_losses.npz",
            losses=losses,
        )

    if not split_losses:
        print("[WARN] 没有成功计算任何 split 的 dynamic fitting loss。")
        return

    # 图 1：rolling loss curve
    plt.figure(figsize=(9, 5))
    for split, losses in split_losses.items():
        y = rolling_mean(losses, window=ROLLING_WINDOW)
        x = np.arange(len(y))
        plt.plot(x, y, label=f"{split} rolling MSE")

    plt.xlabel(f"Window Index / Rolling Mean Window={ROLLING_WINDOW}")
    plt.ylabel("Next-step Prediction MSE")
    plt.title("Local Dynamic Model Fitting Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    save_fig(fig_dir, "02_local_dynamic_model_fitting_loss_curve.png")

    # 图 2：loss 分布
    plt.figure(figsize=(9, 5))
    for split, losses in split_losses.items():
        plt.hist(losses, bins=50, alpha=0.5, label=split)

    plt.xlabel("Next-step Prediction MSE")
    plt.ylabel("Number of Windows")
    plt.title("Local Dynamic Model Fitting Loss Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    save_fig(fig_dir, "03_local_dynamic_model_fitting_loss_distribution.png")


# ============================================================
# 5. OCSVM validation objective curve
# ============================================================

def infer_abnormal_direction(scores, y_pred_saved=None):
    """
    判断 score 是越大越异常，还是越小越异常。
    你的 2D-ESN 日志里 best threshold 从 0 变成 -8.54 后，预测异常数变多，
    说明大概率是 score >= threshold 判为异常。
    """
    if y_pred_saved is None:
        return "greater"

    y_pred_saved = np.asarray(y_pred_saved).reshape(-1)

    pred_greater = (scores >= 0).astype(int)
    pred_less = (scores <= 0).astype(int)

    match_greater = np.mean(pred_greater == y_pred_saved)
    match_less = np.mean(pred_less == y_pred_saved)

    if match_greater >= match_less:
        return "greater"
    else:
        return "less"


def plot_ocsvm_validation_objective(exp_dir: Path, fig_dir: Path):
    """
    OCSVM 没有 sklearn 可直接导出的 epoch loss。
    这里画验证集 threshold 搜索过程中的 objective:
        loss = 1 - F1
    """

    ocsvm_dirs = list(exp_dir.rglob("*ocsvm*"))
    ocsvm_dirs = [d for d in ocsvm_dirs if d.is_dir()]

    if not ocsvm_dirs:
        print("[WARN] 没找到 ocsvm 结果目录，跳过 OCSVM objective 图。")
        return

    # 优先使用 baseline 目录
    ocsvm_dirs = sorted(ocsvm_dirs, key=lambda p: len(str(p)))
    ocsvm_dir = ocsvm_dirs[0]
    print(f"\n[OCSVM DIR] {ocsvm_dir}")

    try:
        val_pred_file = find_npz_file(
            ocsvm_dir,
            [
                "val_default_predictions.npz",
                "val_best_thr_predictions.npz",
                "val*pred*.npz",
            ],
        )
    except FileNotFoundError:
        print("[WARN] 没找到 val predictions，跳过 OCSVM objective 图。")
        return

    data = load_npz(val_pred_file)

    scores, score_key = get_array(
        data,
        ["scores", "score", "decision_scores", "decision_score", "ocsvm_scores"],
    )

    y_true, label_key = get_array(
        data,
        ["y_true", "labels", "gt_labels", "label"],
    )

    y_pred_saved, pred_key = get_array(
        data,
        ["y_pred", "preds", "pred_labels", "predictions"],
        required=False,
    )

    scores = np.asarray(scores).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1).astype(int)

    # 如果 y_true 里面是 -1/1，转成 0/1
    if set(np.unique(y_true)).issubset({-1, 1}):
        # 假设 -1 是异常，1 是正常
        y_true = (y_true == -1).astype(int)

    direction = infer_abnormal_direction(scores, y_pred_saved)
    print(f"[INFO] score_key={score_key}, label_key={label_key}, direction={direction}")

    thresholds = np.linspace(np.percentile(scores, 1), np.percentile(scores, 99), 300)

    f1_list = []
    acc_list = []
    precision_list = []
    recall_list = []

    for thr in thresholds:
        if direction == "greater":
            y_pred = (scores >= thr).astype(int)
        else:
            y_pred = (scores <= thr).astype(int)

        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        f1_list.append(f1)
        acc_list.append(acc)
        precision_list.append(pre)
        recall_list.append(rec)

    f1_arr = np.asarray(f1_list)
    acc_arr = np.asarray(acc_list)
    precision_arr = np.asarray(precision_list)
    recall_arr = np.asarray(recall_list)

    loss_arr = 1.0 - f1_arr
    best_idx = int(np.argmax(f1_arr))
    best_thr = thresholds[best_idx]
    best_f1 = f1_arr[best_idx]

    print(f"[BEST] threshold={best_thr:.6f}, val_f1={best_f1:.6f}, loss={1-best_f1:.6f}")

    # 图 1：OCSVM validation loss-like curve
    plt.figure(figsize=(9, 5))
    plt.plot(thresholds, loss_arr, label="Validation Loss Proxy = 1 - F1")
    plt.axvline(best_thr, linestyle="--", linewidth=2, label=f"Best threshold={best_thr:.3f}")
    plt.xlabel("OCSVM Decision Threshold")
    plt.ylabel("Loss Proxy: 1 - F1")
    plt.title("OCSVM Validation Objective Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    save_fig(fig_dir, "04_ocsvm_validation_objective_loss_curve.png")

    # 图 2：precision / recall / f1 随 threshold 变化
    plt.figure(figsize=(9, 5))
    plt.plot(thresholds, precision_arr, label="Precision")
    plt.plot(thresholds, recall_arr, label="Recall")
    plt.plot(thresholds, f1_arr, label="F1")
    plt.axvline(best_thr, linestyle="--", linewidth=2, label=f"Best threshold={best_thr:.3f}")
    plt.xlabel("OCSVM Decision Threshold")
    plt.ylabel("Metric")
    plt.title("OCSVM Threshold Search Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    save_fig(fig_dir, "05_ocsvm_threshold_search_metrics_curve.png")


# ============================================================
# 6. 主函数
# ============================================================

def main():
    print("=" * 80)
    print("Plot Model-Space Loss-like Curves")
    print("=" * 80)

    exp_dir = find_existing_exp_dir()
    fig_dir = exp_dir / "modelspace_loss_like_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EXP_DIR] {exp_dir}")
    print(f"[FIG_DIR] {fig_dir}")

    plot_pca_reconstruction_error(exp_dir, fig_dir)
    plot_dynamic_model_fitting_loss(exp_dir, fig_dir)
    plot_ocsvm_validation_objective(exp_dir, fig_dir)

    print("\n[DONE] loss-like curves 已生成")
    print(f"输出目录: {fig_dir}")


if __name__ == "__main__":
    main()