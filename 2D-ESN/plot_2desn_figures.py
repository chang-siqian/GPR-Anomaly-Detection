# -*- coding: utf-8 -*-
"""
plot_2desn_loss_like_curves.py

用于给 2D-ESN 生成三张“类似 loss 的曲线”：

1. 2D-ESN model-space deviation curve
   - 用 theta 到正常训练样本中心的距离表示
   - 可以作为 2D-ESN 的 loss-like / fitting-deviation 曲线

2. OCSVM anomaly score curve
   - 展示每个窗口的异常分数
   - 同时画出 default threshold 和 best threshold

3. Validation threshold-F1 curve
   - 展示不同 threshold 下 val 集 F1 / Precision / Recall 的变化
   - 说明 best threshold 是怎么选出来的

运行方式：
直接在 PyCharm 中右键运行本脚本即可。
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 路径配置：根据你这次 2D-ESN 运行结果写好的
# ============================================================

PROJECT_ROOT = Path(r"C:\temporary internet files\GPR_ModelSpace_New")

DATA_DIR = PROJECT_ROOT / "gpr_yolo_dataset" / "window_dataset_overlap40_w128_s32"

THETA_DIR = DATA_DIR / "theta_vectors_2desn_n30_h32_w32"

PRED_DIR = THETA_DIR / "ocsvm_2desn_baseline"

OUT_DIR = PRED_DIR / "loss_like_figures_2desn"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 工具函数
# ============================================================

def load_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    return np.load(path, allow_pickle=True)


def print_npz_keys(path: Path):
    data = load_npz(path)
    print(f"\n[NPZ KEYS] {path}")
    for k in data.files:
        arr = data[k]
        print(f"  - {k}: shape={arr.shape}, dtype={arr.dtype}")
    data.close()


def find_key(npz_data, candidates):
    """
    从 npz 里自动找字段名。
    因为不同脚本保存 npz 时字段名可能略有不同，所以这里做兼容。
    """
    keys = list(npz_data.files)
    for c in candidates:
        if c in keys:
            return c
    raise KeyError(f"没有找到字段。候选字段={candidates}, 实际字段={keys}")


def load_theta_split(split: str):
    """
    加载 train_theta.npz / val_theta.npz / test_theta.npz
    """
    path = THETA_DIR / f"{split}_theta.npz"
    data = load_npz(path)

    theta_key = find_key(
        data,
        candidates=[
            "theta",
            "thetas",
            "theta_vectors",
            "X",
            "features",
            "X_theta",
        ],
    )

    label_key = find_key(
        data,
        candidates=[
            "labels",
            "y",
            "y_true",
            "gt_labels",
        ],
    )

    theta = np.asarray(data[theta_key], dtype=np.float32)
    labels = np.asarray(data[label_key]).astype(int)

    data.close()
    return theta, labels


def load_pred(split: str, tag: str = "best_thr"):
    """
    加载 train/val/test 的 OCSVM 预测结果。
    tag 可以是:
      - default
      - best_thr
    """
    path = PRED_DIR / f"{split}_{tag}_predictions.npz"
    data = load_npz(path)

    # 分数字段兼容
    score_key = find_key(
        data,
        candidates=[
            "scores",
            "score",
            "decision_scores",
            "ocsvm_scores",
            "anomaly_scores",
        ],
    )

    label_key = find_key(
        data,
        candidates=[
            "labels",
            "y_true",
            "gt_labels",
            "y",
        ],
    )

    pred_key = find_key(
        data,
        candidates=[
            "preds",
            "pred_labels",
            "y_pred",
            "predictions",
        ],
    )

    scores = np.asarray(data[score_key], dtype=np.float32)
    labels = np.asarray(data[label_key]).astype(int)
    preds = np.asarray(data[pred_key]).astype(int)

    data.close()
    return scores, labels, preds


def load_thresholds():
    """
    从 ocsvm_2desn_results.json 里读取 default threshold 和 best threshold。
    如果读取失败，就使用日志中的默认值。
    """
    default_thr = 0.0
    best_thr = -8.540830

    path = PRED_DIR / "ocsvm_2desn_results.json"
    if not path.exists():
        print(f"[WARN] 没找到 {path}，使用默认阈值 default=0.0, best=-8.540830")
        return default_thr, best_thr

    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        # 兼容不同 json 字段名
        for key in ["default_threshold", "thr_default", "default_thr"]:
            if key in obj:
                default_thr = float(obj[key])

        for key in ["best_threshold", "best_thr", "val_best_threshold", "threshold"]:
            if key in obj:
                best_thr = float(obj[key])

        # 如果 json 是嵌套结构，也尽量找一下
        text = json.dumps(obj)
        if "best on val" in text:
            pass

    except Exception as e:
        print(f"[WARN] 读取 threshold json 失败: {e}")
        print("[WARN] 使用默认阈值 default=0.0, best=-8.540830")

    return default_thr, best_thr


def calc_metrics_by_threshold(scores, labels, thr):
    """
    注意：
    你当前代码里 score 越低越异常。
    从日志看 default threshold = 0，best threshold = -8.540830。
    所以这里使用:
        pred_abnormal = score < threshold
    """
    y_true = labels.astype(int)
    y_pred = (scores < thr).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)

    return precision, recall, f1, acc


def smooth_curve(y, window=21):
    """
    简单滑动平均，让曲线更适合放 PPT。
    """
    y = np.asarray(y, dtype=np.float32)
    if len(y) < window:
        return y
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(y, kernel, mode="same")


# ============================================================
# 3. 图 1：2D-ESN model-space deviation curve
# ============================================================

def plot_theta_deviation_curve(split="test"):
    """
    用 theta 到正常训练 theta 中心的欧氏距离，作为 2D-ESN 的 loss-like curve。
    这不是 YOLO 那种 epoch loss，而是每个窗口在模型空间里的偏离程度。
    """

    train_theta, train_labels = load_theta_split("train")
    theta, labels = load_theta_split(split)

    normal_train_theta = train_theta[train_labels == 0]
    if len(normal_train_theta) == 0:
        raise ValueError("训练集中没有 normal label=0 的 theta，无法计算 normal center。")

    normal_center = normal_train_theta.mean(axis=0)

    deviation = np.linalg.norm(theta - normal_center[None, :], axis=1)
    deviation_smooth = smooth_curve(deviation, window=21)

    x = np.arange(len(deviation))

    plt.figure(figsize=(13, 5))
    plt.plot(x, deviation, linewidth=0.8, alpha=0.35, label="Raw model-space deviation")
    plt.plot(x, deviation_smooth, linewidth=2.0, label="Smoothed deviation")

    abnormal_idx = np.where(labels == 1)[0]
    if len(abnormal_idx) > 0:
        plt.scatter(
            abnormal_idx,
            deviation[abnormal_idx],
            s=8,
            alpha=0.35,
            label="GT abnormal windows",
        )

    plt.title(f"2D-ESN Loss-like Curve: Model-space Deviation ({split})")
    plt.xlabel("Window index")
    plt.ylabel("Distance to normal theta center")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    out_path = OUT_DIR / f"01_2desn_model_space_deviation_{split}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] 图1保存完成: {out_path}")


# ============================================================
# 4. 图 2：OCSVM anomaly score curve
# ============================================================

def plot_ocsvm_score_curve(split="test", tag="best_thr"):
    """
    画 OCSVM score 曲线。
    当前流程里 score 越低越异常，所以低于 threshold 的部分是异常。
    """

    scores, labels, preds = load_pred(split, tag=tag)
    default_thr, best_thr = load_thresholds()

    x = np.arange(len(scores))
    scores_smooth = smooth_curve(scores, window=21)

    plt.figure(figsize=(13, 5))
    plt.plot(x, scores, linewidth=0.8, alpha=0.35, label="Raw OCSVM score")
    plt.plot(x, scores_smooth, linewidth=2.0, label="Smoothed OCSVM score")

    plt.axhline(default_thr, linestyle="--", linewidth=1.5, label=f"Default threshold = {default_thr:.3f}")
    plt.axhline(best_thr, linestyle="--", linewidth=1.5, label=f"Best val threshold = {best_thr:.3f}")

    abnormal_idx = np.where(labels == 1)[0]
    if len(abnormal_idx) > 0:
        plt.scatter(
            abnormal_idx,
            scores[abnormal_idx],
            s=8,
            alpha=0.35,
            label="GT abnormal windows",
        )

    plt.title(f"OCSVM Anomaly Score Curve on 2D-ESN Theta ({split}, {tag})")
    plt.xlabel("Window index")
    plt.ylabel("OCSVM decision score")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    out_path = OUT_DIR / f"02_ocsvm_anomaly_score_{split}_{tag}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] 图2保存完成: {out_path}")


# ============================================================
# 5. 图 3：Validation threshold-F1 curve
# ============================================================

def plot_val_threshold_curve():
    """
    在 val 集上遍历 threshold，画 Precision / Recall / F1。
    用来说明 best threshold 的选择过程。
    """

    scores, labels, _ = load_pred("val", tag="best_thr")
    default_thr, best_thr = load_thresholds()

    # 在 val score 的范围内取一批 threshold
    thresholds = np.linspace(scores.min(), scores.max(), 300)

    precisions = []
    recalls = []
    f1s = []
    accs = []

    for thr in thresholds:
        p, r, f1, acc = calc_metrics_by_threshold(scores, labels, thr)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        accs.append(acc)

    precisions = np.asarray(precisions)
    recalls = np.asarray(recalls)
    f1s = np.asarray(f1s)
    accs = np.asarray(accs)

    best_idx = int(np.argmax(f1s))
    best_thr_by_curve = float(thresholds[best_idx])
    best_f1_by_curve = float(f1s[best_idx])

    plt.figure(figsize=(13, 5))
    plt.plot(thresholds, precisions, linewidth=2.0, label="Precision")
    plt.plot(thresholds, recalls, linewidth=2.0, label="Recall")
    plt.plot(thresholds, f1s, linewidth=2.3, label="F1")
    plt.plot(thresholds, accs, linewidth=1.5, alpha=0.7, label="Accuracy")

    plt.axvline(default_thr, linestyle="--", linewidth=1.5, label=f"Default threshold = {default_thr:.3f}")
    plt.axvline(best_thr, linestyle="--", linewidth=1.5, label=f"Saved best threshold = {best_thr:.3f}")
    plt.scatter([best_thr_by_curve], [best_f1_by_curve], s=50, label=f"Curve best F1={best_f1_by_curve:.3f}")

    plt.title("Validation Threshold Selection Curve on 2D-ESN + OCSVM")
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    out_path = OUT_DIR / "03_val_threshold_precision_recall_f1_curve.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] 图3保存完成: {out_path}")
    print(f"[INFO] curve best threshold = {best_thr_by_curve:.6f}, best F1 = {best_f1_by_curve:.6f}")


# ============================================================
# 6. 主函数
# ============================================================

def main():
    print("=" * 80)
    print("Plot 2D-ESN loss-like curves")
    print("=" * 80)
    print(f"[DATA_DIR ] {DATA_DIR}")
    print(f"[THETA_DIR] {THETA_DIR}")
    print(f"[PRED_DIR ] {PRED_DIR}")
    print(f"[OUT_DIR  ] {OUT_DIR}")

    # 先打印关键 npz 的字段，方便你确认脚本是否读对
    print_npz_keys(THETA_DIR / "train_theta.npz")
    print_npz_keys(PRED_DIR / "test_best_thr_predictions.npz")

    # 生成三张图
    plot_theta_deviation_curve(split="test")
    plot_ocsvm_score_curve(split="test", tag="best_thr")
    plot_val_threshold_curve()

    print("\n" + "=" * 80)
    print("[DONE] 三张图全部生成完成")
    print(f"[OUT_DIR] {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()