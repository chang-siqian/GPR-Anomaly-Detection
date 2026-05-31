# -*- coding: utf-8 -*-
"""
plot_hog_svm_loss_like_curves.py

给 HOG + SVM baseline 生成三张“类似 loss / 训练过程”的曲线：

1. SVM Hyperparameter Search Curve
   - 横轴：候选参数编号
   - 纵轴：val F1 / val accuracy
   - 说明 SVM 是如何选择最优参数的

2. SVM Decision Score Curve
   - 横轴：test 图像序号
   - 纵轴：SVM decision score
   - score 越大越倾向 abnormal
   - threshold=0 是 SVM 默认分类边界

3. Validation Threshold-F1 Curve
   - 横轴：decision threshold
   - 纵轴：Precision / Recall / F1 / Accuracy
   - 说明阈值变化对性能的影响

注意：
HOG+SVM 没有 epoch-loss。
SVM 不是像 YOLO 那样一轮一轮反向传播训练的模型。
所以这里画的是 loss-like / training-diagnostic curves。
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import load
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from skimage.feature import hog

try:
    import cv2
except ImportError:
    cv2 = None
    from PIL import Image


# ============================================================
# 1. 路径配置
# ============================================================

PROJECT_ROOT = Path(r"C:\temporary internet files\GPR_ModelSpace_New")

DATA_ROOT = PROJECT_ROOT / "gpr_yolo_dataset"

IMAGE_ROOT = DATA_ROOT / "images"
LABEL_ROOT = DATA_ROOT / "labels"

RESULT_DIR = DATA_ROOT / "hog_svm_baseline_binary"

MODEL_PATH = RESULT_DIR / "best_svm_model.joblib"
SCALER_PATH = RESULT_DIR / "best_scaler.joblib"

OUT_DIR = RESULT_DIR / "loss_like_figures_hog_svm"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. HOG 配置：与你 run_hog_svm_baseline.py 日志一致
# ============================================================

IMAGE_SIZE = (256, 256)

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (32, 32)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"


# ============================================================
# 3. SVM 参数搜索空间：与你日志中的 24 组一致
# ============================================================

def build_svm_candidates():
    candidates = []

    # linear: 4 个 C × 2 个 class_weight = 8 组
    for class_weight in [None, "balanced"]:
        for C in [0.1, 1, 10, 100]:
            candidates.append({
                "kernel": "linear",
                "C": C,
                "gamma": None,
                "class_weight": class_weight,
            })

    # rbf: 4 个 C × 2 个 gamma × 2 个 class_weight = 16 组
    for class_weight in [None, "balanced"]:
        for C in [0.1, 1, 10, 100]:
            for gamma in ["scale", "auto"]:
                candidates.append({
                    "kernel": "rbf",
                    "C": C,
                    "gamma": gamma,
                    "class_weight": class_weight,
                })

    return candidates


# ============================================================
# 4. 数据读取与 HOG 特征提取
# ============================================================

def read_gray_image(path: Path):
    if cv2 is not None:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"读取图片失败: {path}")
        img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        return img

    img = Image.open(path).convert("L")
    img = img.resize(IMAGE_SIZE)
    return np.asarray(img)


def label_from_yolo_txt(label_path: Path):
    """
    二分类：
    - txt 为空：normal = 0
    - txt 非空：abnormal = 1
    """
    if not label_path.exists():
        return 0

    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if len(text) == 0:
        return 0
    return 1


def extract_hog_feature(gray_img):
    feat = hog(
        gray_img,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        feature_vector=True,
    )
    return feat.astype(np.float32)


def load_split_features(split: str):
    img_dir = IMAGE_ROOT / split
    lab_dir = LABEL_ROOT / split

    if not img_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {img_dir}")
    if not lab_dir.exists():
        raise FileNotFoundError(f"标签目录不存在: {lab_dir}")

    img_paths = sorted(
        list(img_dir.glob("*.jpg")) +
        list(img_dir.glob("*.png")) +
        list(img_dir.glob("*.jpeg"))
    )

    if len(img_paths) == 0:
        raise RuntimeError(f"没有找到图片: {img_dir}")

    X = []
    y = []
    names = []

    print("\n" + "=" * 80)
    print(f"[LOAD SPLIT] {split}")
    print(f"[IMG DIR] {img_dir}")
    print(f"[LAB DIR] {lab_dir}")
    print("=" * 80)

    for idx, img_path in enumerate(img_paths):
        label_path = lab_dir / f"{img_path.stem}.txt"

        img = read_gray_image(img_path)
        feat = extract_hog_feature(img)
        label = label_from_yolo_txt(label_path)

        X.append(feat)
        y.append(label)
        names.append(img_path.name)

        if idx < 5:
            print(
                f"[sample {idx + 1}] {img_path.name} | "
                f"img_shape={img.shape} | label={label} | feat_dim={feat.shape[0]}"
            )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    names = np.asarray(names)

    print(f"[INFO] X.shape = {X.shape}")
    print(f"[INFO] y.shape = {y.shape}")
    print(f"[INFO] normal = {int(np.sum(y == 0))}")
    print(f"[INFO] abnormal = {int(np.sum(y == 1))}")

    return X, y, names


# ============================================================
# 5. 指标计算
# ============================================================

def binary_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_abnormal": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_abnormal": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_abnormal": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def metrics_by_threshold(scores, labels, thr):
    """
    对 SVM decision score 来说：
    score >= threshold 判为 abnormal
    默认 threshold = 0
    """
    preds = (scores >= thr).astype(int)

    acc = accuracy_score(labels, preds)
    pre = precision_score(labels, preds, pos_label=1, zero_division=0)
    rec = recall_score(labels, preds, pos_label=1, zero_division=0)
    f1 = f1_score(labels, preds, pos_label=1, zero_division=0)

    return pre, rec, f1, acc


def get_decision_scores(model, X_scaled):
    scores = model.decision_function(X_scaled)

    # binary SVC 一般返回 shape=(N,)
    if scores.ndim > 1:
        # 如果是二维，就取 abnormal 类对应列
        if hasattr(model, "classes_") and 1 in list(model.classes_):
            abnormal_col = list(model.classes_).index(1)
            scores = scores[:, abnormal_col]
        else:
            scores = scores[:, -1]

    scores = np.asarray(scores, dtype=np.float32)

    # 正常情况下 classes_ = [0, 1]，score 越大越像 1。
    # 如果异常类 1 不是 classes_[1]，做一次方向修正。
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if len(classes) == 2 and classes[1] != 1:
            scores = -scores

    return scores


# ============================================================
# 6. 图 1：SVM 参数搜索曲线
# ============================================================

def run_grid_search_and_plot(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 80)
    print("[STEP] Re-run SVM hyperparameter search on VAL")
    print("=" * 80)

    candidates = build_svm_candidates()
    records = []

    for idx, cfg in enumerate(candidates, start=1):
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        if cfg["kernel"] == "linear":
            model = SVC(
                kernel="linear",
                C=cfg["C"],
                class_weight=cfg["class_weight"],
            )
        else:
            model = SVC(
                kernel="rbf",
                C=cfg["C"],
                gamma=cfg["gamma"],
                class_weight=cfg["class_weight"],
            )

        model.fit(X_train_s, y_train)
        pred_val = model.predict(X_val_s)

        m = binary_metrics(y_val, pred_val)

        record = {
            "candidate_id": idx,
            "kernel": cfg["kernel"],
            "C": cfg["C"],
            "gamma": cfg["gamma"],
            "class_weight": cfg["class_weight"],
            "val_accuracy": m["accuracy"],
            "val_precision_abnormal": m["precision_abnormal"],
            "val_recall_abnormal": m["recall_abnormal"],
            "val_f1_abnormal": m["f1_abnormal"],
        }
        records.append(record)

        print(
            f"[{idx:02d}/{len(candidates)}] "
            f"kernel={cfg['kernel']}, C={cfg['C']}, gamma={cfg['gamma']}, "
            f"class_weight={cfg['class_weight']} -> "
            f"val_f1={m['f1_abnormal']:.4f}, val_acc={m['accuracy']:.4f}"
        )

    df = pd.DataFrame(records)
    csv_path = OUT_DIR / "hog_svm_grid_search_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    best_idx = int(df["val_f1_abnormal"].idxmax())
    best_row = df.loc[best_idx]

    plt.figure(figsize=(13, 5))
    plt.plot(df["candidate_id"], df["val_f1_abnormal"], marker="o", linewidth=2.0, label="Val F1")
    plt.plot(df["candidate_id"], df["val_accuracy"], marker="s", linewidth=2.0, label="Val Accuracy")

    plt.scatter(
        [best_row["candidate_id"]],
        [best_row["val_f1_abnormal"]],
        s=80,
        label=f"Best F1={best_row['val_f1_abnormal']:.4f}",
        zorder=5,
    )

    plt.title("HOG + SVM Hyperparameter Search Curve")
    plt.xlabel("Candidate index")
    plt.ylabel("Metric value")
    plt.ylim(0.80, 1.01)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    out_path = OUT_DIR / "01_hog_svm_hyperparameter_search_curve.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] grid search csv saved -> {csv_path}")
    print(f"[OK] 图1保存完成 -> {out_path}")

    best_info_path = OUT_DIR / "hog_svm_best_grid_config.json"
    with open(best_info_path, "w", encoding="utf-8") as f:
        json.dump(best_row.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"[OK] best config saved -> {best_info_path}")

    return df


# ============================================================
# 7. 图 2：SVM decision score 曲线
# ============================================================

def plot_decision_score_curve(split, names, y_true, scores, threshold=0.0):
    x = np.arange(len(scores))

    plt.figure(figsize=(13, 5))
    plt.plot(x, scores, linewidth=1.4, label="SVM decision score")
    plt.axhline(threshold, linestyle="--", linewidth=1.8, label=f"Decision threshold = {threshold:.3f}")

    abnormal_idx = np.where(y_true == 1)[0]
    normal_idx = np.where(y_true == 0)[0]

    plt.scatter(
        abnormal_idx,
        scores[abnormal_idx],
        s=16,
        alpha=0.65,
        label="GT abnormal",
    )
    plt.scatter(
        normal_idx,
        scores[normal_idx],
        s=12,
        alpha=0.35,
        label="GT normal",
    )

    plt.title(f"HOG + SVM Decision Score Curve ({split})")
    plt.xlabel("Image index")
    plt.ylabel("SVM decision score")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    out_path = OUT_DIR / f"02_hog_svm_decision_score_curve_{split}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] 图2保存完成 -> {out_path}")

    df = pd.DataFrame({
        "image_name": names,
        "label": y_true,
        "decision_score": scores,
        "pred_by_zero_threshold": (scores >= threshold).astype(int),
    })
    csv_path = OUT_DIR / f"hog_svm_decision_scores_{split}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] score csv saved -> {csv_path}")


# ============================================================
# 8. 图 3：Validation threshold-F1 曲线
# ============================================================

def plot_threshold_f1_curve(y_val, val_scores):
    thresholds = np.linspace(float(val_scores.min()), float(val_scores.max()), 400)

    precisions = []
    recalls = []
    f1s = []
    accs = []

    for thr in thresholds:
        p, r, f1, acc = metrics_by_threshold(val_scores, y_val, thr)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        accs.append(acc)

    precisions = np.asarray(precisions)
    recalls = np.asarray(recalls)
    f1s = np.asarray(f1s)
    accs = np.asarray(accs)

    best_idx = int(np.argmax(f1s))
    best_thr = float(thresholds[best_idx])
    best_f1 = float(f1s[best_idx])

    plt.figure(figsize=(13, 5))
    plt.plot(thresholds, precisions, linewidth=2.0, label="Precision")
    plt.plot(thresholds, recalls, linewidth=2.0, label="Recall")
    plt.plot(thresholds, f1s, linewidth=2.3, label="F1")
    plt.plot(thresholds, accs, linewidth=1.6, alpha=0.75, label="Accuracy")

    plt.axvline(0.0, linestyle="--", linewidth=1.8, label="Default SVM threshold = 0")
    plt.axvline(best_thr, linestyle="--", linewidth=1.8, label=f"Best threshold = {best_thr:.4f}")
    plt.scatter([best_thr], [best_f1], s=80, label=f"Best F1 = {best_f1:.4f}", zorder=5)

    plt.title("Validation Threshold-F1 Curve for HOG + SVM")
    plt.xlabel("Decision threshold")
    plt.ylabel("Metric value")
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    out_path = OUT_DIR / "03_hog_svm_val_threshold_f1_curve.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] 图3保存完成 -> {out_path}")

    info = {
        "best_threshold_on_val": best_thr,
        "best_f1_on_val": best_f1,
        "default_threshold": 0.0,
    }
    json_path = OUT_DIR / "hog_svm_threshold_curve_info.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"[OK] threshold info saved -> {json_path}")


# ============================================================
# 9. 主函数
# ============================================================

def main():
    warnings.filterwarnings("ignore")

    print("=" * 80)
    print("Plot HOG + SVM loss-like curves")
    print("=" * 80)
    print(f"[DATA_ROOT ] {DATA_ROOT}")
    print(f"[RESULT_DIR] {RESULT_DIR}")
    print(f"[MODEL    ] {MODEL_PATH}")
    print(f"[SCALER   ] {SCALER_PATH}")
    print(f"[OUT_DIR  ] {OUT_DIR}")
    print("=" * 80)

    # 读取数据并提取 HOG 特征
    X_train, y_train, train_names = load_split_features("train")
    X_val, y_val, val_names = load_split_features("val")
    X_test, y_test, test_names = load_split_features("test")

    # 图 1：重新跑 24 组参数搜索，画 val F1/acc 曲线
    run_grid_search_and_plot(X_train, y_train, X_val, y_val)

    # 加载你 baseline 已经保存的最优 scaler 和 SVM
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"没有找到 SVM 模型: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"没有找到 scaler: {SCALER_PATH}")

    scaler = load(SCALER_PATH)
    model = load(MODEL_PATH)

    print("\n" + "=" * 80)
    print("[INFO] Loaded saved best model")
    print("=" * 80)
    print(model)

    # 计算 val/test decision score
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    val_scores = get_decision_scores(model, X_val_s)
    test_scores = get_decision_scores(model, X_test_s)

    # 图 2：test decision score 曲线
    plot_decision_score_curve(
        split="test",
        names=test_names,
        y_true=y_test,
        scores=test_scores,
        threshold=0.0,
    )

    # 图 3：val threshold-F1 曲线
    plot_threshold_f1_curve(y_val, val_scores)

    print("\n" + "=" * 80)
    print("[DONE] HOG + SVM 三张曲线全部生成完成")
    print(f"[OUT_DIR] {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()