from pathlib import Path
from collections import Counter

#检查YOLO数据集本身有没有整理好

# ===== 这里改成你自己的数据集根目录 =====
ROOT = Path(r"C:\temporary internet files\GPR_ModelSpace_New\gpr_yolo_dataset")

SPLITS = ["train", "val", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_NAMES = ["cavity", "utility"]   # 先按你现在这两个类来统计


def parse_label_file(label_path):
    """
    读取一个 YOLO txt 标注文件
    每行格式通常是: class x_center y_center width height
    返回:
        boxes_count: 这个文件里有多少个框
        class_ids:   这个文件里出现过哪些类别（按框统计）
        bad_lines:   格式异常的行数
        is_empty:    文件是否为空/无有效框
    """
    class_ids = []
    bad_lines = 0

    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return 0, class_ids, bad_lines, True

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            bad_lines += 1
            continue

        try:
            cls_id = int(float(parts[0]))
            # 后面四个一般是 x y w h，这里只检查能不能转成 float
            _ = [float(x) for x in parts[1:5]]
            class_ids.append(cls_id)
        except Exception:
            bad_lines += 1

    return len(class_ids), class_ids, bad_lines, (len(class_ids) == 0)


def main():
    print("=" * 70)
    print(f"[ROOT] {ROOT}")
    print("=" * 70)

    if not ROOT.exists():
        print(f"[ERROR] 数据集根目录不存在: {ROOT}")
        return

    grand_total_images = 0
    grand_total_labels = 0
    grand_total_boxes = 0
    grand_bad_lines = 0
    grand_class_counter = Counter()

    for split in SPLITS:
        img_dir = ROOT / "images" / split
        lab_dir = ROOT / "labels" / split

        print("\n" + "-" * 70)
        print(f"[SPLIT] {split}")
        print("-" * 70)

        if not img_dir.exists():
            print(f"[ERROR] 图片目录不存在: {img_dir}")
            continue
        if not lab_dir.exists():
            print(f"[ERROR] 标注目录不存在: {lab_dir}")
            continue

        images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
        labels = sorted(lab_dir.glob("*.txt"))

        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}

        missing_labels = sorted(image_stems - label_stems)   # 有图没标注
        orphan_labels = sorted(label_stems - image_stems)    # 有标注没图

        matched = sorted(image_stems & label_stems)

        split_boxes = 0
        split_bad_lines = 0
        split_class_counter = Counter()
        empty_label_files = 0

        for stem in matched:
            label_path = lab_dir / f"{stem}.txt"
            boxes_count, class_ids, bad_lines, is_empty = parse_label_file(label_path)

            split_boxes += boxes_count
            split_bad_lines += bad_lines

            if is_empty:
                empty_label_files += 1

            for cid in class_ids:
                if 0 <= cid < len(CLASS_NAMES):
                    split_class_counter[CLASS_NAMES[cid]] += 1
                else:
                    split_class_counter[f"unknown_class_{cid}"] += 1

        grand_total_images += len(images)
        grand_total_labels += len(labels)
        grand_total_boxes += split_boxes
        grand_bad_lines += split_bad_lines
        grand_class_counter.update(split_class_counter)

        print(f"图片数                 : {len(images)}")
        print(f"标注txt数              : {len(labels)}")
        print(f"成功配对数             : {len(matched)}")
        print(f"有图没标注             : {len(missing_labels)}")
        print(f"有标注没图             : {len(orphan_labels)}")
        print(f"空标注文件数           : {empty_label_files}")
        print(f"总框数                 : {split_boxes}")
        print(f"异常格式行数           : {split_bad_lines}")
        print(f"类别框统计             : {dict(split_class_counter)}")

        if missing_labels:
            print("\n[示例] 有图没标注（最多显示10个）:")
            for x in missing_labels[:10]:
                print("   ", x)

        if orphan_labels:
            print("\n[示例] 有标注没图（最多显示10个）:")
            for x in orphan_labels[:10]:
                print("   ", x)

    print("\n" + "=" * 70)
    print("[SUMMARY]")
    print("=" * 70)
    print(f"总图片数               : {grand_total_images}")
    print(f"总标注txt数            : {grand_total_labels}")
    print(f"总框数                 : {grand_total_boxes}")
    print(f"总异常格式行数         : {grand_bad_lines}")
    print(f"总类别框统计           : {dict(grand_class_counter)}")


if __name__ == "__main__":
    main()