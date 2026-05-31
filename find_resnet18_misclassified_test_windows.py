# -*- coding: utf-8 -*-
"""
找出 ResNet18 模型空间流程在 test 集中所有配错的窗口，
并反查这些窗口来自 test 集哪一张原图。

输出：
1. misclassified_windows.csv
   每一个配错窗口一行：原图名、test图序号、窗口位置、GT、Pred、错误类型、score

2. misclassified_images_summary.csv
   每一张出错原图一行：这张图有几个窗口配错、FP/FN数量、错的窗口范围

3. annotated_full_images/
   对每张出错原图画出配错窗口位置

4. wrong_window_crops/
   单独保存每一个配错窗口的小图
"""

from pathlib import Path
import csv
import json
import numpy as np
import cv2


# =========================
# 1. 按你的项目路径修改这里
# =========================

PROJECT_ROOT = Path(r"C:\temporary internet files\GPR_ModelSpace_New")

DATASET_ROOT = PROJECT_ROOT / "gpr_yolo_dataset"

# 你当前 ResNet18 流程用的是这个窗口数据目录
# 如果你用的是 center2，就改成：
# WINDOW_DATASET_DIR = DATASET_ROOT / "window_dataset_center2_w128_s32"
WINDOW_DATASET_DIR = DATASET_ROOT / "window_dataset_overlap40_w128_s32"

TEST_NPZ = WINDOW_DATASET_DIR / "test.npz"

OCSVM_DIR = (
    WINDOW_DATASET_DIR
    / "resnet_features"
    / "pca32"
    / "theta_vectors"
    / "ocsvm_baseline"
)

# 你的实际预测文件名带 nu_0p15 前缀
PRED_NPZ = OCSVM_DIR / "nu_0p15_best_thr_test_predictions.npz"

OUT_DIR = OCSVM_DIR / "nu_0p15_misclassified_test_best_thr"

# 关键：你这个 OCSVM 脚本里是 score <= threshold 判为 abnormal
SCORE_LARGER_IS_ABNORMAL = False


# =========================
# 2. 工具函数
# =========================

def decode_arr(x):
    """把 bytes 类型的 img_name 转成普通字符串。"""
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def pick_key(npz, candidates, required=True):
    """从 npz 里自动寻找可能的 key。"""
    keys = list(npz.keys())
    for c in candidates:
        if c in npz:
            return c
    if required:
        raise KeyError(
            f"找不到需要的 key，候选={candidates}\n"
            f"当前文件里的 keys={keys}"
        )
    return None


def normalize_binary_label(arr):
    """
    统一成 0/1：
    0 = normal
    1 = abnormal

    支持：
    - 原本就是 0/1
    - OCSVM 常见输出：1 表示 normal，-1 表示 abnormal
    """
    arr = np.asarray(arr).astype(int)

    vals = set(np.unique(arr).tolist())

    if vals.issubset({0, 1}):
        return arr

    if vals.issubset({-1, 1}):
        # sklearn OneClassSVM: 1=normal, -1=abnormal
        return np.where(arr == -1, 1, 0).astype(int)

    raise ValueError(f"无法识别标签取值：{sorted(vals)}，请检查 predictions.npz")


def find_image_path(img_name):
    """
    根据 test.npz 里的 img_name，在 images/test 下面找原图。
    """
    img_name = Path(img_name).name
    image_dir = DATASET_ROOT / "images" / "test"

    p = image_dir / img_name
    if p.exists():
        return p

    stem = Path(img_name).stem
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p

    return None


def read_yolo_boxes_for_image(img_name, img_w, img_h):
    """
    读取 YOLO 标签，返回像素坐标 box。
    这里只用于可视化 GT 框。
    """
    stem = Path(img_name).stem
    label_path = DATASET_ROOT / "labels" / "test" / f"{stem}.txt"

    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue

        cls_id = int(float(parts[0]))
        xc = float(parts[1]) * img_w
        yc = float(parts[2]) * img_h
        bw = float(parts[3]) * img_w
        bh = float(parts[4]) * img_h

        x1 = int(round(xc - bw / 2))
        y1 = int(round(yc - bh / 2))
        x2 = int(round(xc + bw / 2))
        y2 = int(round(yc + bh / 2))

        x1 = max(0, min(img_w - 1, x1))
        x2 = max(0, min(img_w - 1, x2))
        y1 = max(0, min(img_h - 1, y1))
        y2 = max(0, min(img_h - 1, y2))

        boxes.append((cls_id, x1, y1, x2, y2))

    return boxes


def draw_text(img, text, x, y, color):
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


# =========================
# 3. 主程序
# =========================

def main():
    print("=" * 80)
    print("Find ResNet18 misclassified test windows")
    print("=" * 80)
    print(f"[TEST_NPZ] {TEST_NPZ}")
    print(f"[PRED_NPZ] {PRED_NPZ}")
    print(f"[OUT_DIR ] {OUT_DIR}")

    if not TEST_NPZ.exists():
        raise FileNotFoundError(f"找不到 test.npz: {TEST_NPZ}")

    if not PRED_NPZ.exists():
        raise FileNotFoundError(f"找不到预测文件: {PRED_NPZ}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    crop_dir = OUT_DIR / "wrong_window_crops"
    anno_dir = OUT_DIR / "annotated_full_images"
    crop_dir.mkdir(parents=True, exist_ok=True)
    anno_dir.mkdir(parents=True, exist_ok=True)

    test_npz = np.load(TEST_NPZ, allow_pickle=True)
    pred_npz = np.load(PRED_NPZ, allow_pickle=True)

    print("\n[INFO] test.npz keys:")
    print(list(test_npz.keys()))

    print("\n[INFO] pred npz keys:")
    print(list(pred_npz.keys()))

    # test.npz 中的基本信息
    labels_key = pick_key(test_npz, ["labels", "y", "y_true"])
    img_names_key = pick_key(test_npz, ["img_names", "image_names", "filenames", "names"])
    windows_key = pick_key(test_npz, ["windows", "X"], required=False)
    x_starts_key = pick_key(test_npz, ["x_starts", "starts", "window_starts"], required=False)
    x_ends_key = pick_key(test_npz, ["x_ends", "ends", "window_ends"], required=False)
    window_ids_key = pick_key(test_npz, ["window_ids", "win_ids"], required=False)
    cls_ids_key = pick_key(test_npz, ["cls_ids", "class_ids"], required=False)

    y_true_from_test = normalize_binary_label(test_npz[labels_key])
    img_names = np.array([decode_arr(x) for x in test_npz[img_names_key]])

    windows = test_npz[windows_key] if windows_key is not None else None

    if x_starts_key is not None:
        x_starts = test_npz[x_starts_key].astype(int)
    else:
        x_starts = np.full(len(y_true_from_test), -1, dtype=int)

    if x_ends_key is not None:
        x_ends = test_npz[x_ends_key].astype(int)
    else:
        x_ends = np.full(len(y_true_from_test), -1, dtype=int)

    if window_ids_key is not None:
        window_ids = test_npz[window_ids_key]
    else:
        window_ids = np.arange(len(y_true_from_test))

    if cls_ids_key is not None:
        cls_ids = test_npz[cls_ids_key]
    else:
        cls_ids = np.full(len(y_true_from_test), -1, dtype=int)

    # predictions.npz 中的 y_true/y_pred/score
    pred_key = pick_key(
        pred_npz,
        ["y_pred", "pred", "preds", "pred_labels", "predictions", "test_pred"],
        required=False,
    )

    score_key = pick_key(
        pred_npz,
        ["scores", "score", "abnormal_scores", "decision_scores", "test_scores"],
        required=False,
    )

    pred_ytrue_key = pick_key(
        pred_npz,
        ["y_true", "labels", "test_labels"],
        required=False,
    )

    if pred_ytrue_key is not None:
        y_true = normalize_binary_label(pred_npz[pred_ytrue_key])
    else:
        y_true = y_true_from_test

    if pred_key is not None:
        y_pred = normalize_binary_label(pred_npz[pred_key])
    else:
        if score_key is None:
            raise KeyError(
                "predictions.npz 里面既没有 y_pred，也没有 score，无法判断配错窗口。"
            )

        score = np.asarray(pred_npz[score_key]).reshape(-1)

        thr_key = pick_key(pred_npz, ["threshold", "thr", "best_threshold"], required=False)
        if thr_key is None:
            raise KeyError(
                "predictions.npz 里面没有 y_pred，只找到了 score，"
                "但没有 threshold/thr/best_threshold。"
            )

        thr = float(np.asarray(pred_npz[thr_key]).reshape(-1)[0])

        if SCORE_LARGER_IS_ABNORMAL:
            y_pred = (score >= thr).astype(int)
        else:
            y_pred = (score <= thr).astype(int)

    if score_key is not None:
        scores = np.asarray(pred_npz[score_key]).reshape(-1)
    else:
        scores = np.full(len(y_true), np.nan)

    # 长度检查
    n = len(y_true)
    assert len(y_pred) == n, f"y_pred 长度不一致: {len(y_pred)} vs {n}"
    assert len(img_names) == n, f"img_names 长度不一致: {len(img_names)} vs {n}"
    assert len(x_starts) == n, f"x_starts 长度不一致: {len(x_starts)} vs {n}"

    wrong_idx = np.where(y_true != y_pred)[0]

    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]

    print("\n" + "=" * 80)
    print("[RESULT]")
    print(f"test windows total = {n}")
    print(f"wrong windows      = {len(wrong_idx)}")
    print(f"FP 误检            = {len(fp_idx)}  normal -> abnormal")
    print(f"FN 漏检            = {len(fn_idx)}  abnormal -> normal")
    print("=" * 80)

    # test 图片序号：按 images/test 文件名排序得到第几张图
    test_img_dir = DATASET_ROOT / "images" / "test"
    test_img_list = []
    if test_img_dir.exists():
        test_img_list = sorted(
            [
                p.name
                for p in test_img_dir.iterdir()
                if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
            ]
        )
    img_to_test_index = {name: i + 1 for i, name in enumerate(test_img_list)}

    wrong_rows = []
    grouped = {}

    for idx in wrong_idx:
        img_name = Path(img_names[idx]).name
        err_type = "FP_false_alarm" if y_true[idx] == 0 and y_pred[idx] == 1 else "FN_missed"

        image_path = find_image_path(img_name)
        test_image_index = img_to_test_index.get(img_name, "")

        row = {
            "global_window_index": int(idx),
            "test_image_index_1based": test_image_index,
            "img_name": img_name,
            "image_path": str(image_path) if image_path is not None else "",
            "window_id": decode_arr(window_ids[idx]),
            "x_start": int(x_starts[idx]),
            "x_end": int(x_ends[idx]),
            "gt_label": int(y_true[idx]),
            "pred_label": int(y_pred[idx]),
            "gt_name": "abnormal" if y_true[idx] == 1 else "normal",
            "pred_name": "abnormal" if y_pred[idx] == 1 else "normal",
            "error_type": err_type,
            "cls_id": int(cls_ids[idx]) if str(cls_ids[idx]).lstrip("-").isdigit() else decode_arr(cls_ids[idx]),
            "score": float(scores[idx]) if not np.isnan(scores[idx]) else "",
        }
        wrong_rows.append(row)

        grouped.setdefault(img_name, []).append(row)

        # 保存窗口 crop
        if windows is not None:
            win = windows[idx]
            if win.ndim == 2:
                crop = win
            elif win.ndim == 3:
                # 可能是 H,W,C 或 C,H,W
                if win.shape[-1] in [1, 3]:
                    crop = win
                elif win.shape[0] in [1, 3]:
                    crop = np.transpose(win, (1, 2, 0))
                else:
                    crop = win[:, :, 0]
            else:
                crop = None

            if crop is not None:
                crop = np.asarray(crop)
                if crop.dtype != np.uint8:
                    crop_min, crop_max = crop.min(), crop.max()
                    if crop_max > crop_min:
                        crop = ((crop - crop_min) / (crop_max - crop_min) * 255).astype(np.uint8)
                    else:
                        crop = np.zeros_like(crop, dtype=np.uint8)

                crop_name = (
                    f"{idx:06d}_{err_type}_"
                    f"GT{int(y_true[idx])}_PRED{int(y_pred[idx])}_"
                    f"{Path(img_name).stem}_x{int(x_starts[idx])}_{int(x_ends[idx])}.png"
                )
                cv2.imwrite(str(crop_dir / crop_name), crop)

    # 保存 wrong windows csv
    wrong_csv = OUT_DIR / "misclassified_windows.csv"
    fieldnames = [
        "global_window_index",
        "test_image_index_1based",
        "img_name",
        "image_path",
        "window_id",
        "x_start",
        "x_end",
        "gt_label",
        "pred_label",
        "gt_name",
        "pred_name",
        "error_type",
        "cls_id",
        "score",
    ]

    with open(wrong_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(wrong_rows)

    # 保存图片级 summary
    summary_rows = []

    for img_name, rows in grouped.items():
        n_fp = sum(1 for r in rows if r["error_type"] == "FP_false_alarm")
        n_fn = sum(1 for r in rows if r["error_type"] == "FN_missed")
        ranges = "; ".join([f'{r["error_type"]}:x{r["x_start"]}-{r["x_end"]}' for r in rows])

        image_path = find_image_path(img_name)
        test_image_index = img_to_test_index.get(img_name, "")

        summary_rows.append({
            "test_image_index_1based": test_image_index,
            "img_name": img_name,
            "image_path": str(image_path) if image_path is not None else "",
            "num_wrong_windows": len(rows),
            "num_FP_false_alarm": n_fp,
            "num_FN_missed": n_fn,
            "wrong_window_ranges": ranges,
        })

    summary_rows = sorted(
        summary_rows,
        key=lambda r: (
            int(r["test_image_index_1based"]) if str(r["test_image_index_1based"]).isdigit() else 999999,
            r["img_name"],
        )
    )

    summary_csv = OUT_DIR / "misclassified_images_summary.csv"
    summary_fields = [
        "test_image_index_1based",
        "img_name",
        "image_path",
        "num_wrong_windows",
        "num_FP_false_alarm",
        "num_FN_missed",
        "wrong_window_ranges",
    ]

    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    # 保存标注后的整图
    for img_name, rows in grouped.items():
        image_path = find_image_path(img_name)
        if image_path is None:
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # 先画 GT 框，绿色
        gt_boxes = read_yolo_boxes_for_image(img_name, w, h)
        for cls_id, x1, y1, x2, y2 in gt_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 0), 2)
            draw_text(img, f"GT cls={cls_id}", x1, max(12, y1 - 4), (0, 180, 0))

        # 再画配错窗口
        for r in rows:
            x1 = int(r["x_start"])
            x2 = int(r["x_end"])

            if x1 < 0 or x2 < 0:
                continue

            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))

            if r["error_type"] == "FP_false_alarm":
                # 误检：红色
                color = (0, 0, 255)
                label = f'FP GT0 Pred1 idx={r["global_window_index"]}'
            else:
                # 漏检：蓝色
                color = (255, 0, 0)
                label = f'FN GT1 Pred0 idx={r["global_window_index"]}'

            cv2.rectangle(img, (x1, 0), (x2, h - 1), color, 2)
            draw_text(img, label, x1 + 2, 18, color)

        out_name = f"{Path(img_name).stem}_wrong_annotated.png"
        cv2.imwrite(str(anno_dir / out_name), img)

    # 保存一个 json，方便以后复查配置
    meta = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "DATASET_ROOT": str(DATASET_ROOT),
        "WINDOW_DATASET_DIR": str(WINDOW_DATASET_DIR),
        "TEST_NPZ": str(TEST_NPZ),
        "PRED_NPZ": str(PRED_NPZ),
        "OUT_DIR": str(OUT_DIR),
        "num_test_windows": int(n),
        "num_wrong_windows": int(len(wrong_idx)),
        "num_FP_false_alarm": int(len(fp_idx)),
        "num_FN_missed": int(len(fn_idx)),
        "num_wrong_images": int(len(grouped)),
        "score_larger_is_abnormal": SCORE_LARGER_IS_ABNORMAL,
    }

    with open(OUT_DIR / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n[OK] 已输出：")
    print(f"1) {wrong_csv}")
    print(f"2) {summary_csv}")
    print(f"3) {anno_dir}")
    print(f"4) {crop_dir}")
    print(f"5) {OUT_DIR / 'run_meta.json'}")

    print("\n你交给老师主要看这两个表：")
    print(f"- 每个配错窗口：{wrong_csv}")
    print(f"- 每张配错图片汇总：{summary_csv}")


if __name__ == "__main__":
    main()