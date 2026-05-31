from pathlib import Path
import json
import csv
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# =========================================================
# 配置区：按你当前 overlap40 + nu=0.15 + best_thr 的结果来
# =========================================================
PROJECT_ROOT = Path(r"C:\temporary internet files\GPR_ModelSpace_New")

DATASET_DIR = PROJECT_ROOT / "gpr_yolo_dataset"
IMG_DIR = DATASET_DIR / "images" / "test"
LABEL_DIR = DATASET_DIR / "labels" / "test"

PRED_NPZ = (
    DATASET_DIR
    / "window_dataset_overlap40_w128_s32"
    / "resnet_features"
    / "pca32"
    / "theta_vectors"
    / "ocsvm_baseline"
    / "nu_0p15_best_thr_test_predictions.npz"
)

OUT_DIR = (
    DATASET_DIR
    / "window_dataset_overlap40_w128_s32"
    / "resnet_features"
    / "pca32"
    / "theta_vectors"
    / "ocsvm_baseline"
    / "fp_analysis_test_best_thr"
)

# 为了和你这次日志完全对应，这里先用 64
# 如果你最后决定统一用 0，再改成 0 也行；你前面测试过结果一样
MERGE_GAP = 64
IOU_THR = 0.30

CLASS_NAMES = ["cavity", "utility"]
IMG_W = 224
IMG_H = 224


def parse_yolo_boxes(label_path: Path, img_w=224, img_h=224):
    boxes = []
    if not label_path.exists():
        return boxes

    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return boxes

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls_id = int(float(parts[0]))
        xc = float(parts[1]) * img_w
        yc = float(parts[2]) * img_h
        bw = float(parts[3]) * img_w
        bh = float(parts[4]) * img_h

        xmin = max(0, int(round(xc - bw / 2)))
        ymin = max(0, int(round(yc - bh / 2)))
        xmax = min(img_w, int(round(xc + bw / 2)))
        ymax = min(img_h, int(round(yc + bh / 2)))

        boxes.append(
            {
                "cls_id": cls_id,
                "cls_name": CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"unknown_{cls_id}",
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

    return boxes


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


def segment_distance(a, b):
    """两个一维区段的距离；有重叠则为 0"""
    a0, a1 = a
    b0, b1 = b
    if min(a1, b1) > max(a0, b0):
        return 0
    if a1 <= b0:
        return b0 - a1
    return a0 - b1


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

    return matched_pred, matched_gt


def classify_fp(pred_seg, gt_segments):
    length = pred_seg[1] - pred_seg[0]

    if len(gt_segments) == 0:
        return "image_has_no_gt_abnormal"

    best_iou = max(segment_iou(pred_seg, g) for g in gt_segments)
    nearest_dist = min(segment_distance(pred_seg, g) for g in gt_segments)

    if best_iou > 0:
        return "overlap_gt_but_iou_low"

    if nearest_dist <= 32:
        return "near_gt_edge_within_32px"

    if nearest_dist <= 64:
        return "near_gt_edge_within_64px"

    if length <= 128:
        return "short_isolated_fp"

    return "far_from_gt_fp"


def draw_visualization(img_name, pred_segments, gt_segments, fp_segment, boxes, out_path):
    img_path = IMG_DIR / img_name

    with Image.open(img_path) as im:
        im = im.convert("RGB")
        im = im.resize((IMG_W, IMG_H))

    draw = ImageDraw.Draw(im, "RGBA")

    # GT segment：绿色半透明竖条
    for gs, ge in gt_segments:
        draw.rectangle([gs, 0, ge, IMG_H], fill=(0, 255, 0, 45), outline=(0, 180, 0, 200), width=2)

    # 所有预测 segment：浅红色
    for ps, pe in pred_segments:
        draw.rectangle([ps, 0, pe, IMG_H], fill=(255, 0, 0, 30), outline=(255, 100, 100, 160), width=2)

    # 当前 FP segment：深红色加粗
    fs, fe = fp_segment
    draw.rectangle([fs, 0, fe, IMG_H], fill=(255, 0, 0, 80), outline=(255, 0, 0, 255), width=4)

    # YOLO 原始 GT 框：绿色框
    for b in boxes:
        draw.rectangle([b["xmin"], b["ymin"], b["xmax"], b["ymax"]], outline=(0, 255, 0, 255), width=3)
        draw.text((b["xmin"] + 2, max(0, b["ymin"] - 12)), b["cls_name"], fill=(0, 255, 0, 255))

    im.save(out_path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR = OUT_DIR / "visualizations"
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    if not PRED_NPZ.exists():
        raise FileNotFoundError(f"找不到预测文件: {PRED_NPZ}")

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

    print("=" * 80)
    print("[PRED_NPZ]", PRED_NPZ)
    print("[OUT_DIR ]", OUT_DIR)
    print("[MERGE_GAP]", MERGE_GAP)
    print("[IOU_THR  ]", IOU_THR)
    print("=" * 80)

    records = []
    unique_imgs = sorted(set(img_names.tolist()))

    total_tp = 0
    total_fp = 0
    total_fn = 0

    fp_global_id = 0

    for img_name in unique_imgs:
        idx = np.where(img_names == img_name)[0]
        order = np.argsort(x_starts[idx])
        img_idx = idx[order]

        starts_i = x_starts[img_idx]
        ends_i = x_ends[img_idx]
        true_i = y_true[img_idx]
        pred_i = y_pred[img_idx]

        gt_segments = merge_positive_windows_to_segments(
            starts_i, ends_i, true_i, positive_label=1, merge_gap=MERGE_GAP
        )
        pred_segments = merge_positive_windows_to_segments(
            starts_i, ends_i, pred_i, positive_label=1, merge_gap=MERGE_GAP
        )

        matched_pred, matched_gt = greedy_match_segments(pred_segments, gt_segments, IOU_THR)

        tp = len(matched_pred)
        fp = len(pred_segments) - tp
        fn = len(gt_segments) - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

        boxes = parse_yolo_boxes(LABEL_DIR / f"{Path(img_name).stem}.txt", IMG_W, IMG_H)

        for pi, pseg in enumerate(pred_segments):
            if pi in matched_pred:
                continue

            fp_global_id += 1

            best_iou = max([segment_iou(pseg, g) for g in gt_segments], default=0.0)
            nearest_dist = min([segment_distance(pseg, g) for g in gt_segments], default=-1)
            category = classify_fp(pseg, gt_segments)

            vis_name = f"fp_{fp_global_id:03d}_{Path(img_name).stem}_x{pseg[0]}_{pseg[1]}.png"
            vis_path = VIS_DIR / vis_name

            draw_visualization(
                img_name=img_name,
                pred_segments=pred_segments,
                gt_segments=gt_segments,
                fp_segment=pseg,
                boxes=boxes,
                out_path=vis_path,
            )

            records.append(
                {
                    "fp_id": fp_global_id,
                    "img_name": img_name,
                    "pred_start": int(pseg[0]),
                    "pred_end": int(pseg[1]),
                    "pred_width": int(pseg[1] - pseg[0]),
                    "num_gt_segments_in_image": int(len(gt_segments)),
                    "gt_segments": str(gt_segments),
                    "best_iou_with_gt": float(best_iou),
                    "nearest_gt_distance": int(nearest_dist),
                    "category_guess": category,
                    "visualization": str(vis_path),
                }
            )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    csv_path = OUT_DIR / "fp_segments_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(records)

    # 统计 FP 类型
    category_count = {}
    width_count = {}
    for r in records:
        category_count[r["category_guess"]] = category_count.get(r["category_guess"], 0) + 1
        width_count[str(r["pred_width"])] = width_count.get(str(r["pred_width"]), 0) + 1

    summary = {
        "merge_gap": MERGE_GAP,
        "iou_thr": IOU_THR,
        "num_images": len(unique_imgs),
        "segment_tp": total_tp,
        "segment_fp": total_fp,
        "segment_fn": total_fn,
        "segment_precision": precision,
        "segment_recall": recall,
        "segment_f1": f1,
        "fp_count": len(records),
        "fp_category_count": category_count,
        "fp_width_count": width_count,
        "csv_path": str(csv_path),
        "visualization_dir": str(VIS_DIR),
    }

    summary_path = OUT_DIR / "fp_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("[DONE] FP segment analysis finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 80)

    print("\n你重点看这两个输出：")
    print(f"1) FP 表格: {csv_path}")
    print(f"2) FP 可视化图片文件夹: {VIS_DIR}")


if __name__ == "__main__":
    main()