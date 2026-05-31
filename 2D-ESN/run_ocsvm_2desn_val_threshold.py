from pathlib import Path
import json
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# ============================================================
# Train One-Class SVM on 2D-ESN theta vectors.
# labels in exported dataset: 0 = normal, 1 = abnormal.
# anomaly_score = - decision_function, so larger score means more abnormal.
# ============================================================

ROOT = Path(r"C:\temporary internet files\GPR_ModelSpace_New\gpr_yolo_dataset")
DATA_DIR = ROOT / "window_dataset_overlap40_w128_s32"
THETA_DIR = DATA_DIR / "theta_vectors_2desn_n30_h32_w32"
OUT_DIR = THETA_DIR / "ocsvm_2desn_baseline"

SPLITS = ["train", "val", "test"]

NU = 0.15
GAMMA = "scale"
DEFAULT_THRESHOLD = 0.0
MAX_THRESHOLD_CANDIDATES = 3000


def load_theta_split(split):
    path = THETA_DIR / f"{split}_theta.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing theta file: {path}")
    d = np.load(path, allow_pickle=True)
    return {
        "theta": d["theta"].astype(np.float32),
        "labels": d["labels"].astype(np.int8),
        "cls_ids": d["cls_ids"].astype(np.int16),
        "x_starts": d["x_starts"].astype(np.int32),
        "x_ends": d["x_ends"].astype(np.int32),
        "img_names": d["img_names"],
        "window_ids": d["window_ids"].astype(np.int16),
    }


def binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int32)
    y_pred = np.asarray(y_pred).astype(np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    acc = (tp + tn) / max(1, len(y_true))

    return {
        "num_samples": int(len(y_true)),
        "num_normal_gt": int(np.sum(y_true == 0)),
        "num_abnormal_gt": int(np.sum(y_true == 1)),
        "num_pred_normal": int(np.sum(y_pred == 0)),
        "num_pred_abnormal": int(np.sum(y_pred == 1)),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix_[[tn,fp],[fn,tp]]": [[tn, fp], [fn, tp]],
    }


def choose_threshold_by_val_f1(y_true, scores):
    y_true = np.asarray(y_true).astype(np.int32)
    scores = np.asarray(scores).astype(np.float64)

    unique_scores = np.unique(scores)
    if len(unique_scores) <= MAX_THRESHOLD_CANDIDATES:
        candidates = unique_scores
    else:
        qs = np.linspace(0.0, 1.0, MAX_THRESHOLD_CANDIDATES)
        candidates = np.unique(np.quantile(scores, qs))

    # include two edge thresholds
    candidates = np.concatenate([
        [float(scores.min()) - 1e-8],
        candidates,
        [float(scores.max()) + 1e-8],
    ])

    best_thr = float(DEFAULT_THRESHOLD)
    best_metrics = None
    best_key = (-1.0, -1.0, -1.0)

    for thr in candidates:
        pred = (scores >= thr).astype(np.int8)
        m = binary_metrics(y_true, pred)
        # Primary: F1. Tie-breaker: recall, then precision.
        key = (m["f1"], m["recall"], m["precision"])
        if key > best_key:
            best_key = key
            best_thr = float(thr)
            best_metrics = m

    return best_thr, best_metrics


def save_predictions(split, tag, data, scores, threshold):
    pred_abnormal = (scores >= threshold).astype(np.int8)
    out_path = OUT_DIR / f"{split}_{tag}_predictions.npz"
    np.savez_compressed(
        out_path,
        scores=scores.astype(np.float32),
        threshold=np.array([threshold], dtype=np.float32),
        pred_abnormal=pred_abnormal,
        pred_labels=pred_abnormal,
        labels=data["labels"].astype(np.int8),
        cls_ids=data["cls_ids"].astype(np.int16),
        x_starts=data["x_starts"].astype(np.int32),
        x_ends=data["x_ends"].astype(np.int32),
        img_names=data["img_names"],
        window_ids=data["window_ids"].astype(np.int16),
    )
    return out_path, binary_metrics(data["labels"], pred_abnormal)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("OCSVM on 2D-ESN theta vectors")
    print(f"[THETA_DIR] {THETA_DIR}")
    print(f"[OUT_DIR  ] {OUT_DIR}")
    print(f"[NU       ] {NU}")
    print(f"[GAMMA    ] {GAMMA}")
    print("=" * 72)

    train = load_theta_split("train")
    val = load_theta_split("val")
    test = load_theta_split("test")
    all_data = {"train": train, "val": val, "test": test}

    x_train = train["theta"]
    y_train = train["labels"]
    normal_mask = (y_train == 0)
    x_train_normal = x_train[normal_mask]

    print(f"[INFO] X_train.shape = {x_train.shape}")
    print(f"[INFO] y_train.shape = {y_train.shape}")
    print(f"[INFO] normal train theta = {x_train_normal.shape}")

    scaler = StandardScaler()
    x_train_normal_s = scaler.fit_transform(x_train_normal)

    ocsvm = OneClassSVM(nu=NU, kernel="rbf", gamma=GAMMA)
    print("[STEP] training OneClassSVM ...")
    ocsvm.fit(x_train_normal_s)

    joblib.dump(scaler, OUT_DIR / "theta_scaler.joblib")
    joblib.dump(ocsvm, OUT_DIR / "ocsvm_model.joblib")
    print(f"[OK] scaler saved -> {OUT_DIR / 'theta_scaler.joblib'}")
    print(f"[OK] model  saved -> {OUT_DIR / 'ocsvm_model.joblib'}")

    scores = {}
    for split, data in all_data.items():
        x_s = scaler.transform(data["theta"])
        decision = ocsvm.decision_function(x_s).reshape(-1)
        # Larger score means more abnormal.
        scores[split] = (-decision).astype(np.float32)
        print(
            f"[SCORE] {split}: min={scores[split].min():.6f}, "
            f"max={scores[split].max():.6f}, mean={scores[split].mean():.6f}"
        )

    best_thr, best_val_metrics = choose_threshold_by_val_f1(val["labels"], scores["val"])
    print(f"[THR] default = {DEFAULT_THRESHOLD:.6f}")
    print(f"[THR] best on val = {best_thr:.6f}")
    print(f"[VAL BEST METRICS] {json.dumps(best_val_metrics, ensure_ascii=False, indent=2)}")

    final_results = {
        "config": {
            "theta_dir": str(THETA_DIR),
            "out_dir": str(OUT_DIR),
            "nu": NU,
            "gamma": GAMMA,
            "default_threshold": DEFAULT_THRESHOLD,
            "best_threshold_from_val": best_thr,
        },
        "results": {},
    }

    for split, data in all_data.items():
        print("\n" + "-" * 72)
        print(f"[EVAL] {split}")
        print("-" * 72)

        out_default, m_default = save_predictions(split, "default", data, scores[split], DEFAULT_THRESHOLD)
        out_best, m_best = save_predictions(split, "best_thr", data, scores[split], best_thr)

        final_results["results"][split] = {
            "default": {
                "threshold": DEFAULT_THRESHOLD,
                "prediction_file": str(out_default),
                "window_metrics": m_default,
            },
            "best_thr": {
                "threshold": best_thr,
                "prediction_file": str(out_best),
                "window_metrics": m_best,
            },
        }

        print(f"[DEFAULT] {json.dumps(m_default, ensure_ascii=False, indent=2)}")
        print(f"[BEST   ] {json.dumps(m_best, ensure_ascii=False, indent=2)}")
        print(f"[OK] predictions saved -> {out_default}")
        print(f"[OK] predictions saved -> {out_best}")

    result_path = OUT_DIR / "ocsvm_2desn_results.json"
    result_path.write_text(json.dumps(final_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 72)
    print(f"[DONE] results saved -> {result_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
