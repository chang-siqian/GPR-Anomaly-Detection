from pathlib import Path
import csv
import numpy as np


PROJECT_ROOT = Path(r"C:\temporary internet files\GPR_ModelSpace_New")

PRED_NPZ = (
    PROJECT_ROOT
    / "gpr_yolo_dataset"
    / "window_dataset_overlap40_w128_s32"
    / "resnet_features"
    / "pca32"
    / "theta_vectors"
    / "ocsvm_baseline"
    / "nu_0p15_best_thr_test_predictions.npz"
)

OUT_CSV = (
    PROJECT_ROOT
    / "gpr_yolo_dataset"
    / "window_dataset_overlap40_w128_s32"
    / "resnet_features"
    / "pca32"
    / "theta_vectors"
    / "ocsvm_baseline"
    / "min_segment_width_sweep_test.csv"
)

MERGE_GAP = 64
IOU_THR = 0.30

# keep 规则：pred_segment_width >= min_segment_width 才保留
# 0 表示不过滤
MIN_WIDTH_LIST = [0, 129, 161, 193, 225]


def merge_positive_windows_to_segments(starts, ends, labels, positive_label=1, merge_gap=0):
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


def segment_iou(a, b):
    a0, a1 = a
    b0, b1 = b

    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)

    if union <= 0:
        return 0.0

    return inter / union


def greedy_match_segments(pred_segments, gt_segments, iou_thr=0.30):
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


def evaluate_with_min_width(min_segment_width):
    data = np.load(PRED_NPZ, allow_pickle=True)

    y_true = data["y_true"].astype(np.int8)
    y_pred = data["y_pred"].astype(np.int8)
    x_starts = data["x_starts"].astype(np.int32)
    x_ends = data["x_ends"].astype(np.int32)

    img_names = np.array(
        [
            x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else str(x)
            for x in data["img_names"]
        ],
        dtype=object,
    )

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_pred_before_filter = 0
    total_pred_after_filter = 0
    removed_pred_segments = 0

    unique_imgs = sorted(set(img_names.tolist()))

    for img_name in unique_imgs:
        idx = np.where(img_names == img_name)[0]
        order = np.argsort(x_starts[idx])
        img_idx = idx[order]

        starts_i = x_starts[img_idx]
        ends_i = x_ends[img_idx]
        true_i = y_true[img_idx]
        pred_i = y_pred[img_idx]

        gt_segments = merge_positive_windows_to_segments(
            starts_i,
            ends_i,
            true_i,
            positive_label=1,
            merge_gap=MERGE_GAP,
        )

        pred_segments = merge_positive_windows_to_segments(
            starts_i,
            ends_i,
            pred_i,
            positive_label=1,
            merge_gap=MERGE_GAP,
        )

        total_pred_before_filter += len(pred_segments)

        if min_segment_width > 0:
            pred_segments_filtered = [
                seg for seg in pred_segments
                if (seg[1] - seg[0]) >= min_segment_width
            ]
        else:
            pred_segments_filtered = pred_segments

        total_pred_after_filter += len(pred_segments_filtered)
        removed_pred_segments += len(pred_segments) - len(pred_segments_filtered)

        tp, fp, fn = greedy_match_segments(
            pred_segments_filtered,
            gt_segments,
            iou_thr=IOU_THR,
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "min_segment_width": min_segment_width,
        "segment_tp": total_tp,
        "segment_fp": total_fp,
        "segment_fn": total_fn,
        "segment_precision": precision,
        "segment_recall": recall,
        "segment_f1": f1,
        "pred_segments_before_filter": total_pred_before_filter,
        "pred_segments_after_filter": total_pred_after_filter,
        "removed_pred_segments": removed_pred_segments,
    }


def main():
    print("=" * 80)
    print("[PRED_NPZ]", PRED_NPZ)
    print("[OUT_CSV ]", OUT_CSV)
    print("[MERGE_GAP]", MERGE_GAP)
    print("[IOU_THR  ]", IOU_THR)
    print("=" * 80)

    records = []

    for min_w in MIN_WIDTH_LIST:
        result = evaluate_with_min_width(min_w)
        records.append(result)

        print("\n" + "-" * 80)
        print(f"min_segment_width = {min_w}")
        print(f"TP = {result['segment_tp']}, FP = {result['segment_fp']}, FN = {result['segment_fn']}")
        print(f"Precision = {result['segment_precision']:.6f}")
        print(f"Recall    = {result['segment_recall']:.6f}")
        print(f"F1        = {result['segment_f1']:.6f}")
        print(f"Removed pred segments = {result['removed_pred_segments']}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print("\n" + "=" * 80)
    print("[DONE] sweep finished.")
    print("[CSV ]", OUT_CSV)
    print("=" * 80)


if __name__ == "__main__":
    main()