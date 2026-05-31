from pathlib import Path
from collections import Counter
from PIL import Image

#统计滑窗之后的数据分布，正常窗口多少，异常窗口多少等

# ===== 你自己的数据集根目录 =====
ROOT = Path(r"C:\temporary internet files\GPR_ModelSpace_New\gpr_yolo_dataset")

SPLITS = ["train", "val", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ===== 先固定成和你调试时一样 =====
WIN_W = 128
STRIDE = 32
CLASS_NAMES = ["cavity", "utility"]


def load_image_size(img_path):
    with Image.open(img_path) as im:
        w, h = im.size
    return w, h


def parse_yolo_boxes(label_path, img_w, img_h):
    """
    把 YOLO 格式框转成像素坐标框:
    每行: class_id xc yc bw bh   (都是归一化坐标)
    返回:
        [
            {
                "cls_id": 0,
                "cls_name": "cavity",
                "xmin": ...,
                "ymin": ...,
                "xmax": ...,
                "ymax": ...
            },
            ...
        ]
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

        if 0 <= cls_id < len(CLASS_NAMES):
            cls_name = CLASS_NAMES[cls_id]
        else:
            cls_name = f"unknown_{cls_id}"

        boxes.append({
            "cls_id": cls_id,
            "cls_name": cls_name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        })

    return boxes


def get_window_starts(img_w, win_w, stride):
    """
    给定图像宽度，返回所有窗口起点
    保证最后一个窗口能贴到最右边
    """
    if img_w <= win_w:
        return [0]

    starts = list(range(0, img_w - win_w + 1, stride))
    if starts[-1] != img_w - win_w:
        starts.append(img_w - win_w)
    return starts

def label_windows_by_box_center(starts, win_w, boxes, top_k=2):
    """
    对每个 GT 框：
    用框中心 x 与各窗口中心 x 的距离来排序，
    只把最近的 top_k 个窗口标成 abnormal。

    这样可以避免“只要沾边就整片变红”的问题。
    """
    labels = [0] * len(starts)

    if not boxes or not starts:
        return labels

    window_centers = [s + win_w / 2.0 for s in starts]

    for b in boxes:
        box_center = (b["xmin"] + b["xmax"]) / 2.0

        order = sorted(
            range(len(starts)),
            key=lambda i: abs(window_centers[i] - box_center)
        )

        for i in order[:top_k]:
            labels[i] = 1

    return labels

def process_one_image(img_path, label_path, win_w, stride):
    img_w, img_h = load_image_size(img_path)
    boxes = parse_yolo_boxes(label_path, img_w, img_h)

    starts = get_window_starts(img_w, win_w, stride)

    labels = label_windows_by_box_center(starts, win_w, boxes, top_k=2)

    abnormal_count = sum(labels)
    normal_count = len(labels) - abnormal_count

    return {
        "img_w": img_w,
        "img_h": img_h,
        "num_boxes": len(boxes),
        "num_windows": len(starts),
        "normal_windows": normal_count,
        "abnormal_windows": abnormal_count,
    }


def main():
    print("=" * 72)
    print(f"[ROOT] {ROOT}")
    print(f"[CONFIG] WIN_W={WIN_W}, STRIDE={STRIDE}")
    print("=" * 72)

    grand_images = 0
    grand_windows = 0
    grand_normal = 0
    grand_abnormal = 0

    for split in SPLITS:
        img_dir = ROOT / "images" / split
        lab_dir = ROOT / "labels" / split

        images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

        split_images = 0
        split_windows = 0
        split_normal = 0
        split_abnormal = 0
        split_box_counter = Counter()

        print("\n" + "-" * 72)
        print(f"[SPLIT] {split}")
        print("-" * 72)

        for idx, img_path in enumerate(images, start=1):
            label_path = lab_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            info = process_one_image(img_path, label_path, WIN_W, STRIDE)

            split_images += 1
            split_windows += info["num_windows"]
            split_normal += info["normal_windows"]
            split_abnormal += info["abnormal_windows"]

            # 顺便累计类别框数
            img_w, img_h = info["img_w"], info["img_h"]
            boxes = parse_yolo_boxes(label_path, img_w, img_h)
            for b in boxes:
                split_box_counter[b["cls_name"]] += 1

            if idx <= 3:
                print(
                    f"[sample {idx}] {img_path.name} | "
                    f"size=({info['img_h']},{info['img_w']}) | "
                    f"boxes={info['num_boxes']} | "
                    f"windows={info['num_windows']} | "
                    f"normal={info['normal_windows']} | "
                    f"abnormal={info['abnormal_windows']}"
                )

        grand_images += split_images
        grand_windows += split_windows
        grand_normal += split_normal
        grand_abnormal += split_abnormal

        abnormal_ratio = 0.0 if split_windows == 0 else split_abnormal / split_windows

        print(f"\n图片数                   : {split_images}")
        print(f"总窗口数                 : {split_windows}")
        print(f"正常窗口数               : {split_normal}")
        print(f"异常窗口数               : {split_abnormal}")
        print(f"异常窗口占比             : {abnormal_ratio:.4f}")
        print(f"类别框统计               : {dict(split_box_counter)}")

    grand_ratio = 0.0 if grand_windows == 0 else grand_abnormal / grand_windows

    print("\n" + "=" * 72)
    print("[SUMMARY]")
    print("=" * 72)
    print(f"总图片数                 : {grand_images}")
    print(f"总窗口数                 : {grand_windows}")
    print(f"总正常窗口数             : {grand_normal}")
    print(f"总异常窗口数             : {grand_abnormal}")
    print(f"总异常窗口占比           : {grand_ratio:.4f}")


if __name__ == "__main__":
    main()