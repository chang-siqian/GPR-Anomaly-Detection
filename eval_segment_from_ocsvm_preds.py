import os
import json
import argparse
from typing import List, Tuple, Dict, Any

import numpy as np


PRED_DIR = r"C:\temporary internet files\GPR_ModelSpace_New\gpr_yolo_dataset\window_dataset_overlap20_w128_s32\resnet_features\pca32\theta_vectors\ocsvm_baseline"

DEFAULT_FILES = [
    "val_best_thr_predictions.npz",
    "test_best_thr_predictions.npz",
]

IOU_THR = 0.30
MERGE_GAP = 32


def normalize_name(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def load_predictions(npz_path: str) -> Dict[str, Any]:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"找不到文件: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    return {
        "y_true": data["y_true"].astype(np.int8),
        "y_pred": data["y_pred"].astype(np.int8),
        "scores": data["scores"].astype(np.float64) if "scores" in data else None,
        "thr": float(data["thr"][0]) if "thr" in data else None,
        "x_starts": data["x_starts"].astype(np.int32),
        "x_ends": data["x_ends"].astype(np.int32),
        "img_names": np.array([normalize_name(x) for x in data["img_names"]], dtype=object),
        "window_ids": data["window_ids"].astype(np.int32) if "window_ids" in data else None,
        "cls_ids": data["cls_ids"].astype(np.int32) if "cls_ids" in data else None,
    }


def merge_positive_windows_to_segments(
    starts: np.ndarray,
    ends: np.ndarray,
    labels: np.ndarray,
    positive_label: int = 1,
    merge_gap: int = 0,
) -> List[Tuple[int, int]]:
    """
    把同一张图里 labels==positive_label 的窗口合并成连续区段。
    假设 starts / ends / labels 已经按 x_start 升序排序。
    """
    segments: List[Tuple[int, int]] = []

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

        # 如果下一个 abnormal 窗口和当前段有重叠，或者间隔 <= merge_gap，就合并
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
) -> Tuple[int, int, int, List[Dict[str, Any]]]:
    """
    一对一 greedy 匹配：
    - 枚举所有 pred-gt 对
    - 保留 iou >= iou_thr 的候选
    - 按 iou 从大到小排序
    - 贪心一对一匹配

    返回:
    TP, FP, FN, matches
    """
    candidates = []

    for pi, pseg in enumerate(pred_segments):
        for gi, gseg in enumerate(gt_segments):
            iou = segment_iou(pseg, gseg)
            if iou >= iou_thr:
                candidates.append((iou, pi, gi))

    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_pred = set()
    matched_gt = set()
    matches = []

    for iou, pi, gi in candidates:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        matches.append(
            {
                "pred_idx": int(pi),
                "gt_idx": int(gi),
                "pred_segment": [int(pred_segments[pi][0]), int(pred_segments[pi][1])],
                "gt_segment": [int(gt_segments[gi][0]), int(gt_segments[gi][1])],
                "iou": float(iou),
            }
        )

    tp = len(matches)
    fp = len(pred_segments) - tp
    fn = len(gt_segments) - tp

    return tp, fp, fn, matches


def evaluate_one_npz(
    npz_path: str,
    iou_thr: float = 0.30,
    merge_gap: int = 0,
    save_dir: str = None,
) -> Dict[str, Any]:
    data = load_predictions(npz_path)

    y_true = data["y_true"]
    y_pred = data["y_pred"]
    x_starts = data["x_starts"]
    x_ends = data["x_ends"]
    img_names = data["img_names"]
    thr = data["thr"]

    if save_dir is None:
        save_dir = os.path.dirname(npz_path)
    os.makedirs(save_dir, exist_ok=True)

    unique_imgs = sorted(set(img_names.tolist()))

    total_tp = 0
    total_fp = 0
    total_fn = 0

    per_image_records = []

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

        tp, fp, fn, matches = greedy_match_segments(
            pred_segments, gt_segments, iou_thr=iou_thr
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        record = {
            "img_name": img_name,
            "num_windows": int(len(img_idx)),
            "num_gt_abnormal_windows": int((y_true_i == 1).sum()),
            "num_pred_abnormal_windows": int((y_pred_i == 1).sum()),
            "gt_segments": [[int(a), int(b)] for a, b in gt_segments],
            "pred_segments": [[int(a), int(b)] for a, b in pred_segments],
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "matches": matches,
        }
        per_image_records.append(record)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    summary = {
        "input_file": npz_path,
        "threshold_from_npz": thr,
        "iou_thr": float(iou_thr),
        "merge_gap": int(merge_gap),
        "num_images": int(len(unique_imgs)),
        "segment_tp": int(total_tp),
        "segment_fp": int(total_fp),
        "segment_fn": int(total_fn),
        "segment_precision": float(precision),
        "segment_recall": float(recall),
        "segment_f1": float(f1),
    }

    stem = os.path.splitext(os.path.basename(npz_path))[0]
    summary_path = os.path.join(save_dir, f"{stem}_segment_eval_summary.json")
    detail_path = os.path.join(save_dir, f"{stem}_segment_eval_details.json")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(per_image_records, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"[SEGMENT EVAL] {os.path.basename(npz_path)}")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] saved summary -> {summary_path}")
    print(f"[OK] saved details -> {detail_path}")

    return {
        "summary": summary,
        "details": per_image_records,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate segment-level metrics from OCSVM window predictions.")
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=PRED_DIR,
        help="目录，里面放 val_best_thr_predictions.npz / test_best_thr_predictions.npz",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=DEFAULT_FILES,
        help="要评估的 npz 文件名列表",
    )
    parser.add_argument(
        "--iou_thr",
        type=float,
        default=IOU_THR,
        help="1D IoU 阈值，默认 0.30",
    )
    parser.add_argument(
        "--merge_gap",
        type=int,
        default=MERGE_GAP,
        help="合并相邻 abnormal 窗口时允许的最大间隔，默认 0",
    )
    args = parser.parse_args()

    print("=" * 80)
    print(f"[PRED_DIR ] {args.pred_dir}")
    print(f"[FILES    ] {args.files}")
    print(f"[IOU_THR  ] {args.iou_thr}")
    print(f"[MERGE_GAP] {args.merge_gap}")
    print("=" * 80)

    all_summaries = {}

    for fname in args.files:
        npz_path = os.path.join(args.pred_dir, fname)
        if not os.path.exists(npz_path):
            print(f"[WARN] file not found, skip -> {npz_path}")
            continue

        result = evaluate_one_npz(
            npz_path=npz_path,
            iou_thr=args.iou_thr,
            merge_gap=args.merge_gap,
            save_dir=args.pred_dir,
        )
        all_summaries[fname] = result["summary"]

    all_summary_path = os.path.join(args.pred_dir, "all_segment_eval_summary.json")
    with open(all_summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"[DONE] saved all summaries -> {all_summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()