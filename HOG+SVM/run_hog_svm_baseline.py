from pathlib import Path
import json
import csv

import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import joblib


# =========================================================
# 标准 HOG + SVM baseline
# 任务：二分类
#   空 txt  -> normal (0)
#   非空 txt -> abnormal (1)
# =========================================================

ROOT = Path(r"C:\temporary internet files\GPR_ModelSpace_New\gpr_yolo_dataset")
OUT_DIR = ROOT / "hog_svm_baseline_binary"

SPLITS = ["train", "val", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# 图像统一尺寸
IMAGE_W = 256
IMAGE_H = 256

# HOG 参数（按论文里的 32x32 cell 来做）
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (32, 32)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"

# SVM 搜索范围
LINEAR_C_LIST = [0.1, 1, 10, 100]
RBF_C_LIST = [0.1, 1, 10, 100]
RBF_GAMMA_LIST = ["sca"
                  ""
                  "le", "auto"]
CLASS_WEIGHT_LIST = [None, "balanced"]

RANDOM_SEED = 42


def load_gray_image(img_path: Path) -> np.ndarray:
    """
    读取灰度图并 resize 到固定大小
    返回 uint8, shape = [H, W]
    """
    with Image.open(img_path) as im:
        im = im.convert("L")
        im = im.resize((IMAGE_W, IMAGE_H), Image.BILINEAR)
        arr = np.array(im, dtype=np.uint8)
    return arr


def load_binary_label(label_path: Path) -> int:
    """
    二分类标签定义：
    - 空 txt: 0 (normal)
    - 非空 txt: 1 (abnormal)
    """
    if not label_path.exists():
        raise FileNotFoundError(f"找不到标注文件: {label_path}")

    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    return 0 if text == "" else 1


def extract_hog_feature(img_gray: np.ndarray) -> np.ndarray:
    """
    对单张灰度图提取 HOG 特征
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
    读取某个 split 的全部样本
    返回：
        X: [N, D]
        y: [N]
        names: list[str]
    """
    img_dir = ROOT / "images" / split
    lab_dir = ROOT / "labels" / split

    if not img_dir.exists():
        raise FileNotFoundError(f"找不到图像目录: {img_dir}")
    if not lab_dir.exists():
        raise FileNotFoundError(f"找不到标注目录: {lab_dir}")

    image_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

    if len(image_paths) == 0:
        raise ValueError(f"{split} 没有找到图像文件: {img_dir}")

    features = []
    labels = []
    names = []

    print("\n" + "=" * 72)
    print(f"[LOAD SPLIT] {split}")
    print(f"[IMG DIR] {img_dir}")
    print(f"[LAB DIR] {lab_dir}")
    print("=" * 72)

    for i, img_path in enumerate(image_paths, start=1):
        label_path = lab_dir / f"{img_path.stem}.txt"

        img = load_gray_image(img_path)
        y = load_binary_label(label_path)
        feat = extract_hog_feature(img)

        features.append(feat)
        labels.append(y)
        names.append(img_path.name)

        if i <= 5:
            print(
                f"[sample {i}] {img_path.name} | "
                f"img_shape={img.shape} | label={y} | feat_dim={feat.shape[0]}"
            )

    X = np.stack(features, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int64)

    num_abnormal = int((y == 1).sum())
    num_normal = int((y == 0).sum())

    print(f"[INFO] X.shape = {X.shape}")
    print(f"[INFO] y.shape = {y.shape}")
    print(f"[INFO] normal = {num_normal}")
    print(f"[INFO] abnormal = {num_abnormal}")

    return X, y, names


def calc_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "num_samples": int(len(y_true)),
        "num_normal_gt": int((np.array(y_true) == 0).sum()),
        "num_abnormal_gt": int((np.array(y_true) == 1).sum()),
        "num_pred_normal": int((np.array(y_pred) == 0).sum()),
        "num_pred_abnormal": int((np.array(y_pred) == 1).sum()),
        "accuracy": float(acc),
        "precision_abnormal": float(p),
        "recall_abnormal": float(r),
        "f1_abnormal": float(f1),
        "confusion_matrix_[[tn,fp],[fn,tp]]": cm.tolist(),
    }


def save_predictions_csv(
    out_path: Path,
    names,
    y_true,
    y_pred,
):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["img_name", "y_true", "y_pred"])
        for n, t, p in zip(names, y_true, y_pred):
            writer.writerow([n, int(t), int(p)])


def evaluate_and_save(split, X, y, names, scaler, model, out_dir: Path):
    Xs = scaler.transform(X)
    y_pred = model.predict(Xs)

    metrics = calc_metrics(y, y_pred)

    print("\n" + "=" * 72)
    print(f"[{split.upper()} RESULT]")
    print("=" * 72)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    pred_csv = out_dir / f"{split}_predictions.csv"
    save_predictions_csv(pred_csv, names, y, y_pred)

    summary_json = out_dir / f"{split}_metrics.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved predictions -> {pred_csv}")
    print(f"[OK] saved metrics     -> {summary_json}")

    return metrics


def build_candidate_configs():
    configs = []

    for cw in CLASS_WEIGHT_LIST:
        for c in LINEAR_C_LIST:
            configs.append({
                "kernel": "linear",
                "C": c,
                "class_weight": cw,
            })

    for cw in CLASS_WEIGHT_LIST:
        for c in RBF_C_LIST:
            for gamma in RBF_GAMMA_LIST:
                configs.append({
                    "kernel": "rbf",
                    "C": c,
                    "gamma": gamma,
                    "class_weight": cw,
                })

    return configs


def search_best_model_on_val(X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    candidates = build_candidate_configs()
    all_records = []
    best = None
    best_model = None

    print("\n" + "=" * 72)
    print("[STEP] searching best SVM config on VAL")
    print("=" * 72)
    print(f"[INFO] num_candidates = {len(candidates)}")

    for i, cfg in enumerate(candidates, start=1):
        model = SVC(
            kernel=cfg["kernel"],
            C=cfg["C"],
            gamma=cfg.get("gamma", "scale"),
            class_weight=cfg["class_weight"],
        )
        model.fit(X_train_s, y_train)

        y_val_pred = model.predict(X_val_s)
        m = calc_metrics(y_val, y_val_pred)

        record = {
            "kernel": cfg["kernel"],
            "C": float(cfg["C"]),
            "gamma": cfg.get("gamma", None),
            "class_weight": cfg["class_weight"],
            "val_accuracy": float(m["accuracy"]),
            "val_precision_abnormal": float(m["precision_abnormal"]),
            "val_recall_abnormal": float(m["recall_abnormal"]),
            "val_f1_abnormal": float(m["f1_abnormal"]),
        }
        all_records.append(record)

        print(
            f"[{i:02d}/{len(candidates)}] "
            f"kernel={cfg['kernel']}, C={cfg['C']}, gamma={cfg.get('gamma', None)}, "
            f"class_weight={cfg['class_weight']} "
            f"-> val_f1={record['val_f1_abnormal']:.4f}, val_acc={record['val_accuracy']:.4f}"
        )

        # 主排序：val_f1_abnormal
        # 次排序：val_recall_abnormal
        # 再次排序：val_accuracy
        score_tuple = (
            record["val_f1_abnormal"],
            record["val_recall_abnormal"],
            record["val_accuracy"],
        )

        if best is None:
            best = record
            best_model = model
            best["_score_tuple"] = score_tuple
        else:
            if score_tuple > best["_score_tuple"]:
                best = record
                best_model = model
                best["_score_tuple"] = score_tuple

    best = dict(best)
    best.pop("_score_tuple", None)

    print("\n[BEST CONFIG ON VAL]")
    print(json.dumps(best, ensure_ascii=False, indent=2))

    return scaler, best_model, best, all_records


def main():
    np.random.seed(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"[ROOT      ] {ROOT}")
    print(f"[OUT_DIR   ] {OUT_DIR}")
    print(f"[IMAGE_SIZE] ({IMAGE_H}, {IMAGE_W})")
    print(f"[HOG] orientations={HOG_ORIENTATIONS}, "
          f"pixels_per_cell={HOG_PIXELS_PER_CELL}, "
          f"cells_per_block={HOG_CELLS_PER_BLOCK}, "
          f"block_norm={HOG_BLOCK_NORM}")
    print("=" * 72)

    # 1) 读取数据
    X_train, y_train, names_train = load_split("train")
    X_val, y_val, names_val = load_split("val")
    X_test, y_test, names_test = load_split("test")

    # 2) 在 val 上搜最优 SVM
    scaler, best_model, best_cfg, all_records = search_best_model_on_val(
        X_train, y_train, X_val, y_val
    )

    # 3) 保存 scaler / model / 搜索记录
    joblib.dump(scaler, OUT_DIR / "best_scaler.joblib")
    joblib.dump(best_model, OUT_DIR / "best_svm_model.joblib")

    with open(OUT_DIR / "val_search_records.json", "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    with open(OUT_DIR / "best_config_on_val.json", "w", encoding="utf-8") as f:
        json.dump(best_cfg, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved scaler -> {OUT_DIR / 'best_scaler.joblib'}")
    print(f"[OK] saved model  -> {OUT_DIR / 'best_svm_model.joblib'}")

    # 4) 评估 train / val / test
    train_metrics = evaluate_and_save(
        "train", X_train, y_train, names_train, scaler, best_model, OUT_DIR
    )
    val_metrics = evaluate_and_save(
        "val", X_val, y_val, names_val, scaler, best_model, OUT_DIR
    )
    test_metrics = evaluate_and_save(
        "test", X_test, y_test, names_test, scaler, best_model, OUT_DIR
    )

    all_summary = {
        "task": "binary_classification_empty_txt_is_normal_nonempty_txt_is_abnormal",
        "root": str(ROOT),
        "image_size": [IMAGE_H, IMAGE_W],
        "hog_config": {
            "orientations": HOG_ORIENTATIONS,
            "pixels_per_cell": list(HOG_PIXELS_PER_CELL),
            "cells_per_block": list(HOG_CELLS_PER_BLOCK),
            "block_norm": HOG_BLOCK_NORM,
        },
        "best_config_on_val": best_cfg,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(OUT_DIR / "all_results_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("[DONE] HOG + SVM baseline finished.")
    print(f"[RESULT DIR] {OUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()