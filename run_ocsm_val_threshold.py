import os
import json
from typing import List, Tuple, Dict, Any

import numpy as np
import joblib

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


THETA_DIR = r"C:\temporary internet files\GPR_ModelSpace_New\gpr_yolo_dataset\window_dataset_overlap40_w128_s32\resnet_features\pca32\theta_vectors"
OUT_DIR = os.path.join(THETA_DIR, "ocsvm_baseline")

# 每次手动改这里
NU = 0.15
GAMMA = "scale"

IOU_THR = 0.30
MERGE_GAP = 0

# 区段级后处理：过滤过短的预测异常段
# 由 test 分析可知，短段误报较多；保留宽度 >= 161 的预测段效果最好
MIN_SEGMENT_WIDTH = 193

NUM_THR_CANDIDATES = 200


def load_split(split):
    path = os.path.join(THETA_DIR, f"{split}_theta.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")

    data = np.load(path, allow_pickle=True)
    return {
        "thetas": data["thetas"],
        "labels": data["labels"].astype(np.int8),   # 0=normal, 1=abnormal
        "cls_ids": data["cls_ids"] if "cls_ids" in data else None,
        "x_starts": data["x_starts"].astype(np.int32),
        "x_ends": data["x_ends"].astype(np.int32),
        "img_names": np.array(
            [
                x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else str(x)
                for x in data["img_names"]
            ],
            dtype=object,
        ),
        "window_ids": data["window_ids"] if "window_ids" in data else None,
    }


def merge_positive_windows_to_segments(
    starts: np.ndarray,
    ends: np.ndarray,
    labels: np.ndarray,
    positive_label: int = 1,
    merge_gap: int = 0,
) -> List[Tuple[int, int]]:
    segments = []
    cur_start = None
    cur_end = None

    for s, e, lab in zip(starts, ends, labels):
        s = int(s)
        e = int(e)
        lab = int(lab)

        if lab != positive_label:
            continue

        if cur_start is None:
            cur_start = s
            cur_end = e
            continue

        if s <= cur_end + merge_gap:
            cur_end = max(cur_end, e)
        else:
            segments.append((cur_start, cur_end))
            cur_start = s
            cur_end = e

    if cur_start is not None:
        segments.append((cur_start, cur_end))

    return segments


def segment_iou(seg_a: Tuple[int, int], seg_b: Tuple[int, int]) -> float:
    a0, a1 = seg_a
    b0, b1 = seg_b

    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)

    if union <= 0:
        return 0.0
    return float(inter / union)


def greedy_match_segments(
    pred_segments: List[Tuple[int, int]],
    gt_segments: List[Tuple[int, int]],
    iou_thr: float = 0.30,
):
    candidates = []
    for pi, pseg in enumerate(pred_segments):
        for gi, gseg in enumerate(gt_segments):
            iou = segment_iou(pseg, gseg)
            if iou >= iou_thr:
                candidates.append((iou, pi, gi))

    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_pred = set()
    matched_gt = set()

    for iou, pi, gi in candidates:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)

    tp = len(matched_pred)
    fp = len(pred_segments) - tp
    fn = len(gt_segments) - tp

    return tp, fp, fn


def calc_window_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "num_samples": int(len(y_true)),
        "num_normal_gt": int((y_true == 0).sum()),
        "num_abnormal_gt": int((y_true == 1).sum()),
        "num_pred_normal": int((y_pred == 0).sum()),
        "num_pred_abnormal": int((y_pred == 1).sum()),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix_[[tn,fp],[fn,tp]]": cm.tolist(),
    }


def calc_segment_metrics(
    img_names: np.ndarray,
    x_starts: np.ndarray,
    x_ends: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    iou_thr: float = 0.30,
    merge_gap: int = 0,
) -> Dict[str, Any]:
    unique_imgs = sorted(set(img_names.tolist()))

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for img_name in unique_imgs:
        idx = np.where(img_names == img_name)[0]
        order = np.argsort(x_starts[idx])
        img_idx = idx[order]

        starts_i = x_starts[img_idx]
        ends_i = x_ends[img_idx]
        y_true_i = y_true[img_idx]
        y_pred_i = y_pred[img_idx]

        gt_segments = merge_positive_windows_to_segments(
            starts_i, ends_i, y_true_i, positive_label=1, merge_gap=merge_gap
        )
        pred_segments = merge_positive_windows_to_segments(
            starts_i, ends_i, y_pred_i, positive_label=1, merge_gap=merge_gap
        )
        if MIN_SEGMENT_WIDTH is not None and MIN_SEGMENT_WIDTH > 0:
            pred_segments = [
                seg for seg in pred_segments
                if (seg[1] - seg[0]) >= MIN_SEGMENT_WIDTH
            ]

        tp, fp, fn = greedy_match_segments(pred_segments, gt_segments, iou_thr=iou_thr)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "num_images": int(len(unique_imgs)),
        "segment_tp": int(total_tp),
        "segment_fp": int(total_fp),
        "segment_fn": int(total_fn),
        "segment_precision": float(precision),
        "segment_recall": float(recall),
        "segment_f1": float(f1),
    }


def search_best_threshold_on_val(val_data, scaler, model):
    X_val = val_data["thetas"]
    y_val = val_data["labels"]
    x_starts = val_data["x_starts"]
    x_ends = val_data["x_ends"]
    img_names = val_data["img_names"]

    Xv = scaler.transform(X_val)
    val_scores = model.decision_function(Xv).ravel()

    print("\n" + "=" * 72)
    print("[STEP] searching best threshold on VAL (target = segment_f1)")
    print("=" * 72)
    print(f"[INFO] val_scores.shape = {val_scores.shape}")
    print(f"[INFO] val score range = [{val_scores.min():.6f}, {val_scores.max():.6f}]")

    candidate_thrs = np.percentile(
        val_scores, np.linspace(1, 99, NUM_THR_CANDIDATES)
    )
    candidate_thrs = np.unique(
        np.concatenate([candidate_thrs, np.array([0.0], dtype=np.float64)])
    )
    candidate_thrs.sort()

    best = None
    all_records = []

    for thr in candidate_thrs:
        y_pred = (val_scores < thr).astype(np.int8)

        window_metrics = calc_window_metrics(y_val, y_pred)
        segment_metrics = calc_segment_metrics(
            img_names=img_names,
            x_starts=x_starts,
            x_ends=x_ends,
            y_true=y_val,
            y_pred=y_pred,
            iou_thr=IOU_THR,
            merge_gap=MERGE_GAP,
        )

        rec = {
            "thr": float(thr),
            "window_precision": float(window_metrics["precision"]),
            "window_recall": float(window_metrics["recall"]),
            "window_f1": float(window_metrics["f1"]),
            "segment_precision": float(segment_metrics["segment_precision"]),
            "segment_recall": float(segment_metrics["segment_recall"]),
            "segment_f1": float(segment_metrics["segment_f1"]),
        }
        all_records.append(rec)

        if best is None or rec["segment_f1"] > best["segment_f1"]:
            best = rec

    record_path = os.path.join(
        OUT_DIR, f"nu_{str(NU).replace('.', 'p')}_val_threshold_search_records.json"
    )
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    best_path = os.path.join(
        OUT_DIR, f"nu_{str(NU).replace('.', 'p')}_best_threshold_on_val.json"
    )
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print("[BEST THRESHOLD ON VAL]")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"[OK] threshold search records saved -> {record_path}")
    print(f"[OK] best threshold saved           -> {best_path}")

    return best


def evaluate_split(split_name, data, scaler, model, thr, save_tag):
    X = data["thetas"]
    y_true = data["labels"]
    x_starts = data["x_starts"]
    x_ends = data["x_ends"]
    img_names = data["img_names"]

    Xs = scaler.transform(X)
    scores = model.decision_function(Xs).ravel()
    y_pred = (scores < thr).astype(np.int8)

    window_metrics = calc_window_metrics(y_true, y_pred)
    segment_metrics = calc_segment_metrics(
        img_names=img_names,
        x_starts=x_starts,
        x_ends=x_ends,
        y_true=y_true,
        y_pred=y_pred,
        iou_thr=IOU_THR,
        merge_gap=MERGE_GAP,
    )

    result = {
        "split": split_name,
        "nu": float(NU),
        "threshold": float(thr),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "window_metrics": window_metrics,
        "segment_metrics": segment_metrics,
    }

    prefix = f"nu_{str(NU).replace('.', 'p')}_{save_tag}_{split_name}"

    npz_path = os.path.join(OUT_DIR, f"{prefix}_predictions.npz")
    np.savez_compressed(
        npz_path,
        y_true=y_true,
        y_pred=y_pred,
        scores=scores,
        thr=np.array([thr], dtype=np.float32),
        thetas=X,
        cls_ids=data["cls_ids"],
        x_starts=x_starts,
        x_ends=x_ends,
        img_names=img_names,
        window_ids=data["window_ids"],
    )

    json_path = os.path.join(OUT_DIR, f"{prefix}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print(f"[{split_name.upper()} RESULT] nu={NU}, save_tag={save_tag}, thr={thr:.6f}")
    print("=" * 72)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[OK] saved predictions -> {npz_path}")
    print(f"[OK] saved summary     -> {json_path}")

    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 72)
    print(f"[THETA_DIR] {THETA_DIR}")
    print(f"[OUT_DIR  ] {OUT_DIR}")
    print(f"[NU       ] {NU}")
    print(f"[GAMMA    ] {GAMMA}")
    print(f"[IOU_THR  ] {IOU_THR}")
    print(f"[MERGE_GAP] {MERGE_GAP}")
    print(f"[MIN_SEGMENT_WIDTH] {MIN_SEGMENT_WIDTH}")
    print("=" * 72)

    train_data = load_split("train")
    val_data = load_split("val")
    test_data = load_split("test")

    X_train = train_data["thetas"]
    y_train = train_data["labels"]

    print(f"[INFO] X_train.shape = {X_train.shape}")
    print(f"[INFO] y_train.shape = {y_train.shape}")
    print(f"[INFO] train normal count   = {(y_train == 0).sum()}")
    print(f"[INFO] train abnormal count = {(y_train == 1).sum()}")

    normal_mask = (y_train == 0)
    X_train_normal = X_train[normal_mask]

    print(f"[INFO] normal train theta = {X_train_normal.shape}")

    scaler = StandardScaler()
    Xn = scaler.fit_transform(X_train_normal)

    scaler_path = os.path.join(
        OUT_DIR, f"nu_{str(NU).replace('.', 'p')}_theta_scaler.joblib"
    )
    joblib.dump(scaler, scaler_path)
    print(f"[OK] scaler saved -> {scaler_path}")

    print("\n[STEP] training OneClassSVM ...")
    model = OneClassSVM(kernel="rbf", gamma=GAMMA, nu=NU)
    model.fit(Xn)

    model_path = os.path.join(
        OUT_DIR, f"nu_{str(NU).replace('.', 'p')}_ocsvm_model.joblib"
    )
    joblib.dump(model, model_path)
    print(f"[OK] model saved -> {model_path}")

    best_thr = search_best_threshold_on_val(val_data, scaler, model)

    # default / best 全部输出
    train_default = evaluate_split("train", train_data, scaler, model, thr=0.0, save_tag="default_thr0")
    train_best = evaluate_split("train", train_data, scaler, model, thr=best_thr["thr"], save_tag="best_thr")

    val_default = evaluate_split("val", val_data, scaler, model, thr=0.0, save_tag="default_thr0")
    val_best = evaluate_split("val", val_data, scaler, model, thr=best_thr["thr"], save_tag="best_thr")

    test_default = evaluate_split("test", test_data, scaler, model, thr=0.0, save_tag="default_thr0")
    test_best = evaluate_split("test", test_data, scaler, model, thr=best_thr["thr"], save_tag="best_thr")

    all_results = {
        "config": {
            "nu": float(NU),
            "gamma": GAMMA,
            "iou_thr": float(IOU_THR),
            "merge_gap": int(MERGE_GAP),
            "train_total": int(len(X_train)),
            "train_normal_used": int(len(X_train_normal)),
        },
        "best_threshold_on_val": best_thr,
        "train_default_thr0": train_default,
        "train_best_thr": train_best,
        "val_default_thr0": val_default,
        "val_best_thr": val_best,
        "test_default_thr0": test_default,
        "test_best_thr": test_best,
    }

    summary_path = os.path.join(
        OUT_DIR, f"nu_{str(NU).replace('.', 'p')}_all_results_summary.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print(f"[DONE] saved summary -> {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()