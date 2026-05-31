from pathlib import Path
import json
from collections import defaultdict
import numpy as np

# ============================================================
# Segment-level evaluation from OCSVM prediction npz files.
# It merges continuous abnormal windows in each image and compares
# predicted segments with GT segments derived from window labels.
# ============================================================

ROOT = Path(r"C:\temporary internet files\GPR_ModelSpace_New\gpr_yolo_dataset")
DATA_DIR = ROOT / "window_dataset_overlap40_w128_s32"
THETA_DIR = DATA_DIR / "theta_vectors_2desn_n30_h32_w32"
PRED_DIR = THETA_DIR / "ocsvm_2desn_baseline"
OUT_DIR = PRED_DIR / "segment_eval"

SPLITS = ["train", "val", "test"]
TAGS = ["default", "best_thr"]

# Your exported windows use STRIDE=32 and WIN_W=128.
# merge_gap=32 merges adjacent sliding windows into one continuous region.
MERGE_GAP = 32
IOU_THRESHOLD = 0.30


def box_iou_1d(a, b):
    a0, a1 = a
    b0, b1 = b
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 0.0
    return inter / union


def labels_to_segments(x_starts, x_ends, labels, merge_gap=32):
    """
    Convert abnormal window labels into merged 1D segments.
    labels: 1 means abnormal, 0 means normal.
    """
    segments = []
    cur = None

    for s, e, lab in zip(x_starts, x_ends, labels):
        if int(lab) != 1:
            continue
        s = int(s)
        e = int(e)
        if cur is None:
            cur = [s, e]
        else:
            if s <= cur[1] + merge_gap:
                cur[1] = max(cur[1], e)
            else:
                segments.append(tuple(cur))
                cur = [s, e]

    if cur is not None:
        segments.append(tuple(cur))
    return segments


def group_by_image(data):
    groups = defaultdict(list)
    names = data["img_names"]
    for i, name in enumerate(names):
        groups[str(name)].append(i)
    return groups


def match_segments(pred_segments, gt_segments, iou_thr=0.3):
    matched_gt = set()
    tp = 0
    matches = []

    for pi, p in enumerate(pred_segments):
        best_gi = -1
        best_iou = 0.0
        for gi, g in enumerate(gt_segments):
            if gi in matched_gt:
                continue
            iou = box_iou_1d(p, g)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi

        if best_gi >= 0 and best_iou >= iou_thr:
            tp += 1
            matched_gt.add(best_gi)
            matches.append({
                "pred_index": int(pi),
                "gt_index": int(best_gi),
                "iou": float(best_iou),
                "pred_segment": [int(p[0]), int(p[1])],
                "gt_segment": [int(gt_segments[best_gi][0]), int(gt_segments[best_gi][1])],
            })

    fp = len(pred_segments) - tp
    fn = len(gt_segments) - tp
    return tp, fp, fn, matches


def segment_metrics_from_prediction_file(pred_path):
    d = np.load(pred_path, allow_pickle=True)

    y_true = d["labels"].astype(np.int8)
    y_pred = d["pred_abnormal"].astype(np.int8)
    x_starts = d["x_starts"].astype(np.int32)
    x_ends = d["x_ends"].astype(np.int32)
    img_names = d["img_names"]
    window_ids = d["window_ids"].astype(np.int32)

    packed = {
        "labels": y_true,
        "pred_abnormal": y_pred,
        "x_starts": x_starts,
        "x_ends": x_ends,
        "img_names": img_names,
        "window_ids": window_ids,
    }

    groups = group_by_image(packed)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_pred_segments = 0
    total_gt_segments = 0
    per_image = []

    for img_name, idxs in groups.items():
        idxs = np.array(idxs, dtype=np.int64)
        order = np.lexsort((window_ids[idxs], x_starts[idxs]))
        idxs = idxs[order]

        gt_segments = labels_to_segments(
            x_starts[idxs], x_ends[idxs], y_true[idxs], merge_gap=MERGE_GAP
        )
        pred_segments = labels_to_segments(
            x_starts[idxs], x_ends[idxs], y_pred[idxs], merge_gap=MERGE_GAP
        )

        tp, fp, fn, matches = match_segments(pred_segments, gt_segments, iou_thr=IOU_THRESHOLD)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_pred_segments += len(pred_segments)
        total_gt_segments += len(gt_segments)

        if len(gt_segments) > 0 or len(pred_segments) > 0:
            per_image.append({
                "img_name": img_name,
                "gt_segments": [[int(a), int(b)] for a, b in gt_segments],
                "pred_segments": [[int(a), int(b)] for a, b in pred_segments],
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "matches": matches,
            })

    precision = 0.0 if (total_tp + total_fp) == 0 else total_tp / (total_tp + total_fp)
    recall = 0.0 if (total_tp + total_fn) == 0 else total_tp / (total_tp + total_fn)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return {
        "prediction_file": str(pred_path),
        "merge_gap": MERGE_GAP,
        "iou_threshold": IOU_THRESHOLD,
        "num_images": int(len(groups)),
        "num_pred_segments": int(total_pred_segments),
        "num_gt_segments": int(total_gt_segments),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "per_image_with_segments": per_image,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Segment evaluation for 2D-ESN OCSVM predictions")
    print(f"[PRED_DIR] {PRED_DIR}")
    print(f"[OUT_DIR ] {OUT_DIR}")
    print(f"[CFG     ] MERGE_GAP={MERGE_GAP}, IOU_THRESHOLD={IOU_THRESHOLD}")
    print("=" * 72)

    all_results = {}
    for split in SPLITS:
        all_results[split] = {}
        for tag in TAGS:
            pred_path = PRED_DIR / f"{split}_{tag}_predictions.npz"
            print("\n" + "-" * 72)
            print(f"[EVAL] split={split}, tag={tag}")
            print(f"[FILE] {pred_path}")
            print("-" * 72)

            if not pred_path.exists():
                print(f"[ERROR] missing prediction file: {pred_path}")
                continue

            result = segment_metrics_from_prediction_file(pred_path)
            all_results[split][tag] = result

            light_result = {k: v for k, v in result.items() if k != "per_image_with_segments"}
            print(json.dumps(light_result, ensure_ascii=False, indent=2))

            out_path = OUT_DIR / f"{split}_{tag}_segment_metrics.json"
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[OK] saved -> {out_path}")

    summary_path = OUT_DIR / "all_segment_metrics.json"
    summary_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 72)
    print(f"[DONE] summary -> {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
