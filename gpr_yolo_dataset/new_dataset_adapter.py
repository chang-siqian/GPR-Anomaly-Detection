import os
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#验证经过滑窗后，窗口标签到底打得对不对

TARGET_H = 224
TARGET_W = 224

def read_image_gray(img_path, target_size=None):
    """
    读取灰度图，返回 numpy 数组，shape = (H, W)
    target_size: (target_w, target_h) 或 None
    """
    img = Image.open(img_path).convert("L")
    if target_size is not None:
        target_w, target_h = target_size
        img = img.resize((target_w, target_h), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)
    return arr


def read_xml_boxes(xml_path):
    """
    读取 Pascal VOC 格式的 xml 标注
    返回:
        boxes = [
            {"cls": "cavity", "xmin": 10, "ymin": 20, "xmax": 50, "ymax": 80},
            ...
        ]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        bnd = obj.find("bndbox")
        if name_node is None or bnd is None:
            continue

        name = name_node.text.strip().lower()
        xmin = int(float(bnd.find("xmin").text))
        ymin = int(float(bnd.find("ymin").text))
        xmax = int(float(bnd.find("xmax").text))
        ymax = int(float(bnd.find("ymax").text))

        boxes.append({
            "cls": name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax
        })
    return boxes


def read_yolo_txt(txt_path, img_w, img_h, class_names):
    """
    读取 YOLO txt 标注（每行5列）:
        class_id x_center y_center width height
    其中坐标是相对比例，范围通常在 [0,1]

    返回格式与 read_xml_boxes 一致
    """
    boxes = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id, xc, yc, w, h = map(float, parts)
            cls_id = int(cls_id)

            if cls_id < 0 or cls_id >= len(class_names):
                cls_name = f"class_{cls_id}"
            else:
                cls_name = class_names[cls_id]

            xc *= img_w
            yc *= img_h
            w *= img_w
            h *= img_h

            xmin = int(xc - w / 2)
            ymin = int(yc - h / 2)
            xmax = int(xc + w / 2)
            ymax = int(yc + h / 2)

            boxes.append({
                "cls": cls_name,
                "xmin": max(0, xmin),
                "ymin": max(0, ymin),
                "xmax": min(img_w - 1, xmax),
                "ymax": min(img_h - 1, ymax),
            })
    return boxes


def load_boxes(ann_path, img_w, img_h, class_names=None):
    """
    根据后缀自动读取 xml 或 txt
    """
    ann_path = str(ann_path)
    ext = os.path.splitext(ann_path)[1].lower()

    if ext == ".xml":
        return read_xml_boxes(ann_path)
    elif ext == ".txt":
        if class_names is None or len(class_names) == 0:
            raise ValueError("读取 YOLO txt 时，必须提供 class_names")
        return read_yolo_txt(ann_path, img_w, img_h, class_names)
    else:
        raise ValueError(f"暂不支持的标注格式: {ext}")

def get_window_starts(img_w, win_w, stride):
    """
    返回所有窗口起点，保证最后一个窗口能贴到最右边
    """
    if img_w <= win_w:
        return [0]

    starts = list(range(0, img_w - win_w + 1, stride))
    if starts[-1] != img_w - win_w:
        starts.append(img_w - win_w)
    return starts

def make_windows(img, win_w=128, stride=32):
    """
    沿横向滑窗，保持整幅图的高度不变
    返回:
        windows: list of np.ndarray
        positions: list of dict
    """
    H, W = img.shape
    windows = []
    positions = []

    if W < win_w:
        raise ValueError(f"图像宽度 W={W} 小于窗口宽度 win_w={win_w}")

    starts = get_window_starts(W, win_w, stride)

    for x0 in starts:
        x1 = x0 + win_w
        win = img[:, x0:x1]
        windows.append(win)
        positions.append({
            "start": x0,
            "end": x1,
            "center": (x0 + x1) // 2
        })

    return windows, positions


def label_window_binary(pos, boxes, img_h, min_box_cover=0.2):
    """
    给窗口打二分类标签:
        0 = normal
        1 = abnormal

    判定规则:
    - 只要窗口与任一 GT 框有足够重叠，就认为这个窗口 abnormal
    - 这里用 “交集面积 / GT框面积” 作为覆盖率
    """
    x0, x1 = pos["start"], pos["end"]
    y0, y1 = 0, img_h

    for b in boxes:
        ix0 = max(x0, b["xmin"])
        iy0 = max(y0, b["ymin"])
        ix1 = min(x1, b["xmax"])
        iy1 = min(y1, b["ymax"])

        inter_w = max(0, ix1 - ix0)
        inter_h = max(0, iy1 - iy0)
        inter = inter_w * inter_h

        box_area = max(1, (b["xmax"] - b["xmin"]) * (b["ymax"] - b["ymin"]))
        cover = inter / box_area

        if cover >= min_box_cover:
            return 1

    return 0


def summarize_boxes(boxes):
    """
    统计每类框数量
    """
    stat = {}
    for b in boxes:
        stat[b["cls"]] = stat.get(b["cls"], 0) + 1
    return stat


def visualize_adapter(img, boxes, positions, labels, out_path):
    """
    生成调试图:
    - 绿框: GT
    - 红色半透明竖条: 判为 abnormal 的窗口
    """
    H, W = img.shape
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(img, cmap="gray", aspect="auto")

    # 画 GT 框（绿色）
    for b in boxes:
        rect = Rectangle(
            (b["xmin"], b["ymin"]),
            b["xmax"] - b["xmin"],
            b["ymax"] - b["ymin"],
            fill=False,
            edgecolor="lime",
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(
            b["xmin"],
            max(0, b["ymin"] - 4),
            b["cls"],
            color="lime",
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.5, pad=1)
        )

    # 画 abnormal 窗口（红色半透明）
    for pos, lab in zip(positions, labels):
        if lab == 1:
            rect = Rectangle(
                (pos["start"], 0),
                pos["end"] - pos["start"],
                H,
                facecolor="red",
                edgecolor="red",
                alpha=0.15
            )
            ax.add_patch(rect)

    ax.set_title("Green = GT boxes, Red transparent = abnormal windows")
    ax.set_xlabel("Horizontal position")
    ax.set_ylabel("Depth / row index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] 调试图已保存到: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="图像路径，如 .png/.jpg")
    parser.add_argument("--ann", required=True, help="标注路径，如 .xml 或 .txt")
    parser.add_argument("--win_w", type=int, default=128, help="窗口宽度")
    parser.add_argument("--stride", type=int, default=32, help="滑窗步长")
    parser.add_argument("--min_box_cover", type=float, default=0.2,
                        help="窗口与GT框重叠阈值")
    parser.add_argument("--out", default="debug_adapter.png", help="输出调试图路径")
    parser.add_argument("--classes", nargs="*", default=None,
                        help="如果是 YOLO txt，这里填类别名列表，例如 cavity water crack")
    parser.add_argument("--target_h", type=int, default=224, help="resize 后高度")
    parser.add_argument("--target_w", type=int, default=224, help="resize 后宽度")
    args = parser.parse_args()

    # 1) 读图
    img = read_image_gray(args.img, target_size=(args.target_w, args.target_h))
    H, W = img.shape
    print(f"[INFO] image shape = {img.shape}")

    # 2) 读标注
    boxes = load_boxes(args.ann, W, H, class_names=args.classes)
    print(f"[INFO] number of GT boxes = {len(boxes)}")
    print(f"[INFO] box statistics = {summarize_boxes(boxes)}")

    if len(boxes) > 0:
        print("[INFO] first 3 boxes:")
        for b in boxes[:3]:
            print("   ", b)

    # 3) 滑窗
    windows, positions = make_windows(img, win_w=args.win_w, stride=args.stride)
    print(f"[INFO] total windows = {len(windows)}")

    # 4) 给每个窗口打二分类标签
    labels = [
        label_window_binary(pos, boxes, H, min_box_cover=args.min_box_cover)
        for pos in positions
    ]

    abnormal_count = int(np.sum(labels))
    normal_count = len(labels) - abnormal_count
    print(f"[INFO] normal windows   = {normal_count}")
    print(f"[INFO] abnormal windows = {abnormal_count}")
    print(f"[INFO] first 20 labels  = {labels[:20]}")

    # 打印前几个 abnormal 窗口位置，便于人工核查
    abnormal_positions = [p for p, y in zip(positions, labels) if y == 1]
    print("[INFO] first 10 abnormal window positions:")
    for p in abnormal_positions[:10]:
        print("   ", p)

    # 5) 保存调试图
    visualize_adapter(img, boxes, positions, labels, args.out)


if __name__ == "__main__":
    main()