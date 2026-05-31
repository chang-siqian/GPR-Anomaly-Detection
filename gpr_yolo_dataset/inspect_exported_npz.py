from pathlib import Path
import numpy as np
from collections import Counter

#检查导出的窗口数据集有没有问题

ROOT = Path(r"C:\temporary internet files\GPR_ModelSpace_New\gpr_yolo_dataset")
DATA_DIR = ROOT / "window_dataset_overlap40_w128_s32"

SPLITS = ["train", "val", "test"]
CLASS_NAMES = ["cavity", "utility"]


def inspect_split(split):
    npz_path = DATA_DIR / f"{split}.npz"
    print("\n" + "=" * 72)
    print(f"[SPLIT] {split}")
    print(f"[FILE ] {npz_path}")
    print("=" * 72)

    if not npz_path.exists():
        print(f"[ERROR] 文件不存在: {npz_path}")
        return

    data = np.load(npz_path, allow_pickle=True)

    print("keys:", list(data.keys()))

    windows = data["windows"]        # [N, H, W]
    labels = data["labels"]          # [N]
    cls_ids = data["cls_ids"]        # [N]
    x_starts = data["x_starts"]      # [N]
    x_ends = data["x_ends"]          # [N]
    img_names = data["img_names"]    # [N]
    window_ids = data["window_ids"]  # [N]

    print(f"windows.shape   = {windows.shape}, dtype={windows.dtype}")
    print(f"labels.shape    = {labels.shape}, dtype={labels.dtype}")
    print(f"cls_ids.shape   = {cls_ids.shape}, dtype={cls_ids.dtype}")
    print(f"x_starts.shape  = {x_starts.shape}, dtype={x_starts.dtype}")
    print(f"x_ends.shape    = {x_ends.shape}, dtype={x_ends.dtype}")
    print(f"img_names.shape = {img_names.shape}, dtype={img_names.dtype}")
    print(f"window_ids.shape= {window_ids.shape}, dtype={window_ids.dtype}")

    total = len(labels)
    abnormal = int(labels.sum())
    normal = total - abnormal
    print(f"\n总窗口数       : {total}")
    print(f"正常窗口数     : {normal}")
    print(f"异常窗口数     : {abnormal}")
    print(f"异常占比       : {abnormal / total:.4f}")

    cls_counter = Counter()
    for cid in cls_ids:
        if cid == -1:
            continue
        if 0 <= cid < len(CLASS_NAMES):
            cls_counter[CLASS_NAMES[cid]] += 1
        else:
            cls_counter[f"unknown_{cid}"] += 1

    print(f"异常类别统计   : {dict(cls_counter)}")

    print("\n前 8 个样本预览：")
    for i in range(min(8, total)):
        lab = int(labels[i])
        cid = int(cls_ids[i])
        cname = "normal" if cid == -1 else CLASS_NAMES[cid]
        print(
            f"[{i}] img={img_names[i]} | win_id={int(window_ids[i])} | "
            f"x=({int(x_starts[i])},{int(x_ends[i])}) | "
            f"label={lab} | cls={cname} | "
            f"window_shape={windows[i].shape}"
        )


def main():
    print(f"[DATA_DIR] {DATA_DIR}")
    for split in SPLITS:
        inspect_split(split)


if __name__ == "__main__":
    main()