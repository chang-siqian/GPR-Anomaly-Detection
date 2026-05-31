from pathlib import Path
import json
import numpy as np
from PIL import Image

#正式批量生成窗口数据集

# ===== 路径配置 =====
ROOT = Path(r"C:\temporary internet files\GPR_ModelSpace_New\gpr_yolo_dataset")
#OUT_DIR = ROOT / "window_dataset_center2_w128_s32"
#OUT_DIR = ROOT / "window_dataset_overlap20_w128_s32"
OUT_DIR = ROOT / "window_dataset_overlap40_w128_s32"

SPLITS = ["train", "val", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ===== 数据配置 =====
WIN_W = 128
STRIDE = 32
MIN_BOX_COVER = 0.40     #这里要修改的话，也要修改对应的路径
#OUT_DIR = ROOT / "window_dataset_overlap30_w128_s32"
CLASS_NAMES = ["cavity", "utility"]

TARGET_H = 224
TARGET_W = 224


def load_gray_image(img_path):
    """
    读取灰度图，并统一 resize 到固定大小
    返回 uint8 的 HxW numpy 数组
    """
    with Image.open(img_path) as im:
        im = im.convert("L")
        im = im.resize((TARGET_W, TARGET_H), Image.BILINEAR)
        arr = np.array(im, dtype=np.uint8)
    return arr


def parse_yolo_boxes(label_path, img_w, img_h):
    """
    YOLO 格式:
    class_id xc yc bw bh  (归一化)
    转成像素框
    """
    boxes = []

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
        xmax = min(img_w - 1, int(round(xc + bw / 2)))
        ymax = min(img_h - 1, int(round(yc + bh / 2)))

        boxes.append({
            "cls_id": cls_id,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        })

    return boxes


def get_window_starts(img_w, win_w, stride):
    """返回所有窗口起点，保证最后一个窗口能贴到最右边"""
    if img_w <= win_w:
        return [0]

    starts = list(range(0, img_w - win_w + 1, stride))
    if starts[-1] != img_w - win_w:
        starts.append(img_w - win_w)
    return starts


def assign_windows_overlap_based(starts, win_w, boxes, img_h, min_box_cover=0.2):
    """
    按窗口与 GT 框的重叠程度来打标签
    输出:
        labels: 0/1
        cls_ids: 正常=-1, 异常=对应类别id

    规则:
    - 对每个窗口，计算它和每个 GT 框的 cover = 交集面积 / GT框面积
    - 只要最大 cover >= min_box_cover，就判为 abnormal
    - cls_id 取 cover 最大的那个框的类别
    """
    n = len(starts)
    labels = np.zeros(n, dtype=np.int8)
    cls_ids = np.full(n, -1, dtype=np.int16)

    if n == 0 or len(boxes) == 0:
        return labels, cls_ids

    for i, x_start in enumerate(starts):
        x_end = x_start + win_w

        best_cover = 0.0
        best_cls_id = -1

        for b in boxes:
            ix0 = max(x_start, b["xmin"])
            iy0 = max(0, b["ymin"])
            ix1 = min(x_end, b["xmax"])
            iy1 = min(img_h, b["ymax"])

            inter_w = max(0, ix1 - ix0)
            inter_h = max(0, iy1 - iy0)
            inter = inter_w * inter_h

            box_area = max(1, (b["xmax"] - b["xmin"]) * (b["ymax"] - b["ymin"]))
            cover = inter / box_area

            if cover > best_cover:
                best_cover = cover
                best_cls_id = b["cls_id"]

        if best_cover >= min_box_cover:
            labels[i] = 1
            cls_ids[i] = best_cls_id

    return labels, cls_ids


def crop_window(img, x_start, win_w):
    """
    从灰度图裁一个窗口，输出固定宽度 [H, WIN_W]
    如果极端情况下图宽小于 WIN_W，就右侧补0
    """
    h, w = img.shape
    x_end = min(w, x_start + win_w)
    crop = img[:, x_start:x_end]

    if crop.shape[1] < win_w:
        pad_w = win_w - crop.shape[1]
        crop = np.pad(crop, ((0, 0), (0, pad_w)), mode="constant", constant_values=0)

    return crop


def export_split(split):
    img_dir = ROOT / "images" / split
    lab_dir = ROOT / "labels" / split

    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

    windows_all = []
    labels_all = []
    cls_ids_all = []
    x_starts_all = []
    x_ends_all = []
    img_names_all = []
    window_ids_all = []

    image_count = 0
    skipped = 0

    for idx, img_path in enumerate(images, start=1):
        label_path = lab_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            skipped += 1
            continue

        img = load_gray_image(img_path)
        h, w = img.shape
        boxes = parse_yolo_boxes(label_path, w, h)
        starts = get_window_starts(w, WIN_W, STRIDE)

        labels, cls_ids = assign_windows_overlap_based(
            starts,
            WIN_W,
            boxes,
            h,
            min_box_cover=MIN_BOX_COVER
        )

        for win_idx, x_start in enumerate(starts):
            x_end = x_start + WIN_W
            crop = crop_window(img, x_start, WIN_W)

            windows_all.append(crop)
            labels_all.append(labels[win_idx])
            cls_ids_all.append(cls_ids[win_idx])
            x_starts_all.append(x_start)
            x_ends_all.append(x_end)
            img_names_all.append(img_path.name)
            window_ids_all.append(win_idx)

        image_count += 1

        if idx <= 3:
            print(
                f"[{split} sample {idx}] {img_path.name} | "
                f"shape={img.shape} | boxes={len(boxes)} | windows={len(starts)} | "
                f"abnormal={int(labels.sum())}"
            )

    windows_all = np.stack(windows_all, axis=0).astype(np.uint8)         # [N, H, W]
    labels_all = np.array(labels_all, dtype=np.int8)                     # [N]
    cls_ids_all = np.array(cls_ids_all, dtype=np.int16)                  # [N]
    x_starts_all = np.array(x_starts_all, dtype=np.int32)                # [N]
    x_ends_all = np.array(x_ends_all, dtype=np.int32)                    # [N]
    img_names_all = np.array(img_names_all)                              # [N]
    window_ids_all = np.array(window_ids_all, dtype=np.int16)            # [N]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{split}.npz"

    np.savez_compressed(
        out_path,
        windows=windows_all,
        labels=labels_all,
        cls_ids=cls_ids_all,
        x_starts=x_starts_all,
        x_ends=x_ends_all,
        img_names=img_names_all,
        window_ids=window_ids_all,
    )

    abnormal = int(labels_all.sum())
    total = len(labels_all)
    normal = total - abnormal

    info = {
        "split": split,
        "num_images": image_count,
        "num_windows": total,
        "num_normal": normal,
        "num_abnormal": abnormal,
        "abnormal_ratio": 0.0 if total == 0 else abnormal / total,
        "skipped_images": skipped,
        "window_width": WIN_W,
        "stride": STRIDE,
        "min_box_cover": MIN_BOX_COVER,
        "class_names": CLASS_NAMES,
    }

    info_path = OUT_DIR / f"{split}_meta.json"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[{split}] 导出完成 -> {out_path}")
    print(json.dumps(info, ensure_ascii=False, indent=2))

    return info


def main():
    print("=" * 72)
    print(f"[ROOT] {ROOT}")
    print(f"[OUT ] {OUT_DIR}")
    print(f"[CFG ] WIN_W={WIN_W}, STRIDE={STRIDE}, MIN_BOX_COVER={MIN_BOX_COVER}, TARGET=({TARGET_H},{TARGET_W})")
    print("=" * 72)

    summary = {}
    for split in SPLITS:
        print("\n" + "-" * 72)
        print(f"[EXPORT SPLIT] {split}")
        print("-" * 72)
        info = export_split(split)
        summary[split] = info

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"全部导出完成，摘要文件: {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()