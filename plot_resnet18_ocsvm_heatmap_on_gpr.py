import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle


# ============================================================
# 1. 路径配置：这里已经按你这次运行结果改好了
# ============================================================

PROJECT_ROOT = Path(r"C:\temporary internet files\GPR_ModelSpace_New")
DATASET_ROOT = PROJECT_ROOT / "gpr_yolo_dataset"

# 你这次实际生成的是 overlap40，不是 overlap20
WINDOW_DIR = DATASET_ROOT / "window_dataset_overlap40_w128_s32"

# 先画 test；后面也可以改成 train 或 val
SPLIT = "test"

# 你的 OCSVM 设置
NU_TAG = "nu_0p15"
SAVE_TAG = "best_thr"
BEST_THRESHOLD = 21.161540088865763

# 和你评估脚本保持一致
MERGE_GAP = 0
MIN_SEGMENT_WIDTH = 193

WINDOW_NPZ = WINDOW_DIR / f"{SPLIT}.npz"

PRED_NPZ = (
    WINDOW_DIR
    / "resnet_features"
    / "pca32"
    / "theta_vectors"
    / "ocsvm_baseline"
    / f"{NU_TAG}_{SAVE_TAG}_{SPLIT}_predictions.npz"
)

IMAGE_DIR = DATASET_ROOT / "images" / SPLIT
LABEL_DIR = DATASET_ROOT / "labels" / SPLIT

OUT_DIR = WINDOW_DIR / "heatmap_vis" / f"{NU_TAG}_{SAVE_TAG}_{SPLIT}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "cavity",
    1: "utility",
}


# ============================================================
# 2. 工具函数
# ============================================================

def decode_name(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def pick_key(npz, candidates):
    keys = list(npz.keys())
    for c in candidates:
        if c in keys:
            return c
    return None


def auto_find_score_key(npz, n):
    """
    自动找长度为 n 的浮点数组，作为 score。
    """
    for k in npz.keys():
        arr = npz[k]
        if arr.ndim == 1 and arr.shape[0] == n and np.issubdtype(arr.dtype, np.floating):
            return k
    return None


def auto_find_pred_key(npz, n):
    """
    自动找长度为 n 的整数预测数组。
    避开 true/gt/label 这类真值字段。
    """
    bad_words = ["true", "gt", "label", "target", "y"]
    for k in npz.keys():
        name = k.lower()
        if any(w in name for w in bad_words):
            continue

        arr = npz[k]
        if arr.ndim == 1 and arr.shape[0] == n and np.issubdtype(arr.dtype, np.integer):
            return k

    return None


def robust_normalize(x):
    """
    鲁棒归一化到 0~1。
    """
    x = x.astype(np.float32)

    lo, hi = np.percentile(x, [2, 98])
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)

    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0, 1)
    return y


def pred_to_abnormal_mask(pred):
    """
    兼容两种预测格式：
    1. OCSVM 原生：-1 异常，1 正常
    2. 二分类格式：1 异常，0 正常
    """
    unique = set(np.unique(pred).tolist())

    if -1 in unique:
        return pred == -1
    else:
        return pred == 1


def read_yolo_boxes(label_path, img_w, img_h):
    """
    读取 YOLO 标签：
    cls cx cy w h
    坐标为归一化坐标，需要转成像素坐标。
    """
    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls_id = int(float(parts[0]))
        cx = float(parts[1]) * img_w
        cy = float(parts[2]) * img_h
        bw = float(parts[3]) * img_w
        bh = float(parts[4]) * img_h

        xmin = cx - bw / 2
        ymin = cy - bh / 2
        xmax = cx + bw / 2
        ymax = cy + bh / 2

        boxes.append((cls_id, xmin, ymin, xmax, ymax))

    return boxes


def merge_abnormal_windows(x_starts, x_ends, abnormal_mask, merge_gap=0, min_width=193):
    """
    把异常窗口合并成连续异常区段。
    参数和你评估脚本保持一致：
    merge_gap = 0
    min_width = 193
    """
    segments = []
    cur = None

    for s, e, is_ab in zip(x_starts, x_ends, abnormal_mask):
        s = int(s)
        e = int(e)

        if not is_ab:
            continue

        if cur is None:
            cur = [s, e]
        else:
            # 注意：虽然 merge_gap=0，但是窗口本身有重叠，所以相邻异常窗口仍然会合并
            if s <= cur[1] + merge_gap:
                cur[1] = max(cur[1], e)
            else:
                if cur[1] - cur[0] >= min_width:
                    segments.append(tuple(cur))
                cur = [s, e]

    if cur is not None:
        if cur[1] - cur[0] >= min_width:
            segments.append(tuple(cur))

    return segments


# ============================================================
# 3. 主程序
# ============================================================

def main():
    print("=" * 80)
    print("[WINDOW_NPZ]", WINDOW_NPZ)
    print("[PRED_NPZ  ]", PRED_NPZ)
    print("[IMAGE_DIR ]", IMAGE_DIR)
    print("[LABEL_DIR ]", LABEL_DIR)
    print("[OUT_DIR   ]", OUT_DIR)
    print("=" * 80)

    if not WINDOW_NPZ.exists():
        raise FileNotFoundError(f"找不到窗口文件: {WINDOW_NPZ}")

    if not PRED_NPZ.exists():
        raise FileNotFoundError(f"找不到预测文件: {PRED_NPZ}")

    win_data = np.load(WINDOW_NPZ, allow_pickle=True)
    pred_data = np.load(PRED_NPZ, allow_pickle=True)

    print("[WINDOW KEYS]", list(win_data.keys()))
    print("[PRED KEYS  ]", list(pred_data.keys()))

    img_names = np.array([decode_name(x) for x in win_data["img_names"]])
    x_starts = win_data["x_starts"].astype(int)
    x_ends = win_data["x_ends"].astype(int)
    gt_window_labels = win_data["labels"].astype(int)

    n = len(img_names)

    # ------------------------------------------------------------
    # 读取 score
    # ------------------------------------------------------------

    score_key = pick_key(pred_data, [
        "scores",
        "score",
        "decision_scores",
        "ocsvm_scores",
        "test_scores",
        "val_scores",
        "train_scores",
        "raw_scores",
    ])

    if score_key is None:
        score_key = auto_find_score_key(pred_data, n)

    if score_key is not None:
        raw_scores = pred_data[score_key].astype(np.float32)
        print("[SCORE KEY]", score_key)
    else:
        raw_scores = None
        print("[SCORE KEY] 未找到，将只用预测标签画热力图")

    # ------------------------------------------------------------
    # 读取 pred
    # ------------------------------------------------------------

    pred_key = pick_key(pred_data, [
        "y_pred",
        "pred",
        "preds",
        "pred_labels",
        "predictions",
        "test_pred",
        "val_pred",
        "train_pred",
    ])

    if pred_key is None:
        pred_key = auto_find_pred_key(pred_data, n)

    if pred_key is not None:
        pred = pred_data[pred_key]
        abnormal_mask_all = pred_to_abnormal_mask(pred)
        print("[PRED KEY ]", pred_key, "unique =", np.unique(pred))
    else:
        print("[PRED KEY ] 未找到，将根据 score <= BEST_THRESHOLD 计算预测异常")
        if raw_scores is None:
            raise KeyError("预测文件里既没有 pred，也没有 score，无法画热力图。")
        abnormal_mask_all = raw_scores <= BEST_THRESHOLD

    if len(abnormal_mask_all) != n:
        raise ValueError(
            f"预测数量和窗口数量不一致: pred={len(abnormal_mask_all)}, windows={n}"
        )

    # ------------------------------------------------------------
    # 热力图强度
    # 你的 OCSVM 逻辑是 score <= threshold 判为异常，
    # 所以热力图要用 threshold - score。
    # ------------------------------------------------------------

    if raw_scores is not None:
        anomaly_strength = BEST_THRESHOLD - raw_scores
        heat_scores_all = robust_normalize(anomaly_strength)
    else:
        heat_scores_all = abnormal_mask_all.astype(np.float32)

    unique_imgs = sorted(set(img_names))
    print(f"[INFO] total images in {SPLIT}: {len(unique_imgs)}")

    for idx, img_name in enumerate(unique_imgs):
        mask = img_names == img_name

        img_path = IMAGE_DIR / img_name
        if not img_path.exists():
            print("[WARN] image not found:", img_path)
            continue

        image = Image.open(img_path).convert("L")
        img = np.array(image)
        img_h, img_w = img.shape

        xs = x_starts[mask]
        xe = x_ends[mask]
        heat_scores = heat_scores_all[mask]
        ab = abnormal_mask_all[mask]
        gt_win = gt_window_labels[mask]

        # 按 x 坐标排序，保证合并顺序正确
        order = np.argsort(xs)
        xs = xs[order]
        xe = xe[order]
        heat_scores = heat_scores[order]
        ab = ab[order]
        gt_win = gt_win[order]

        # ------------------------------------------------------------
        # 1D 窗口分数铺回整张图的 x 方向
        # ------------------------------------------------------------

        heat_1d = np.zeros(img_w, dtype=np.float32)
        count_1d = np.zeros(img_w, dtype=np.float32)

        for s, e, v in zip(xs, xe, heat_scores):
            s = max(0, int(s))
            e = min(img_w, int(e))

            heat_1d[s:e] += float(v)
            count_1d[s:e] += 1.0

        count_1d[count_1d == 0] = 1.0
        heat_1d = heat_1d / count_1d
        heat_2d = np.tile(heat_1d[None, :], (img_h, 1))

        # ------------------------------------------------------------
        # 合并预测异常区段
        # ------------------------------------------------------------

        pred_segments = merge_abnormal_windows(
            xs,
            xe,
            ab,
            merge_gap=MERGE_GAP,
            min_width=MIN_SEGMENT_WIDTH,
        )

        # ------------------------------------------------------------
        # 读取 GT 框
        # ------------------------------------------------------------

        label_path = LABEL_DIR / (Path(img_name).stem + ".txt")
        gt_boxes = read_yolo_boxes(label_path, img_w, img_h)

        # ------------------------------------------------------------
        # 画图
        # ------------------------------------------------------------

        plt.figure(figsize=(10, 5))

        # 背景：GPR 图像
        plt.imshow(img, cmap="gray", aspect="auto")

        # 热力图：模型空间异常强度
        plt.imshow(
            heat_2d,
            cmap="jet",
            alpha=0.38,
            vmin=0,
            vmax=1,
            aspect="auto",
        )

        # 绿色框：GT 标注框
        for cls_id, xmin, ymin, xmax, ymax in gt_boxes:
            cls_name = CLASS_NAMES.get(cls_id, str(cls_id))

            rect = Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            plt.gca().add_patch(rect)

            plt.text(
                xmin,
                max(0, ymin - 3),
                f"GT:{cls_name}",
                color="lime",
                fontsize=8,
                bbox=dict(facecolor="black", alpha=0.45, pad=1),
            )

        # 红色虚线框：最终合并后的预测异常区段
        for s, e in pred_segments:
            rect = Rectangle(
                (s, 0),
                e - s,
                img_h,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                linestyle="--",
            )
            plt.gca().add_patch(rect)

            plt.text(
                s,
                12,
                "Pred",
                color="red",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.65, pad=1),
            )

        plt.colorbar(label="Model-space anomaly strength")
        plt.title(
            f"{img_name} | pred abnormal windows={int(ab.sum())} | "
            f"GT abnormal windows={int(gt_win.sum())} | GT boxes={len(gt_boxes)}"
        )
        plt.xlabel("Horizontal position / trace")
        plt.ylabel("Time / depth pixel")
        plt.tight_layout()

        out_path = OUT_DIR / f"{Path(img_name).stem}_heatmap.png"
        plt.savefig(out_path, dpi=200)
        plt.close()

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"[{idx + 1}/{len(unique_imgs)}] saved -> {out_path}")

    print("\n[DONE] 热力图已保存到：")
    print(OUT_DIR)


if __name__ == "__main__":
    main()