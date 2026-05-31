# -*- coding: utf-8 -*-
"""
plot_yolo_tsne3d.py

功能：
1. 加载训练好的 YOLO best.pt
2. 读取 gpr_yolo_dataset/images/test 下的图片
3. 用 YOLO 的 Detect 层输入特征作为图像特征
4. 对特征做 3D t-SNE
5. 保存 3D 可视化图

每个点 = 一张 test 图片
颜色 = 该图片根据 YOLO 标签得到的类别：
    normal  : 无目标框
    cavity  : 只有 cavity
    utility : 只有 utility
    mixed   : 同时包含 cavity 和 utility
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO


# =========================================================
# 1. 路径配置
# =========================================================

PROJECT_ROOT = Path(r"C:\temporary internet files\GPR_ModelSpace_New")

DATA_ROOT = PROJECT_ROOT / "gpr_yolo_dataset"
IMG_DIR = DATA_ROOT / "images" / "test"
LABEL_DIR = DATA_ROOT / "labels" / "test"

WEIGHT_PATH = PROJECT_ROOT / "yolo_baseline" / "yolo_gpr" / "weights" / "best.pt"

OUT_DIR = PROJECT_ROOT / "yolo_baseline" / "yolo_tsne3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FIG = OUT_DIR / "yolo_feature_tsne3d_test.png"
OUT_NPZ = OUT_DIR / "yolo_feature_tsne3d_test.npz"


# =========================================================
# 2. 基本参数
# =========================================================

IMGSZ = 224
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# t-SNE 参数
RANDOM_STATE = 42
PERPLEXITY = 30


CLASS_NAMES = {
    0: "cavity",
    1: "utility",
}


# =========================================================
# 3. 读取标签，转成 image-level 类别
# =========================================================

def get_image_level_label(label_path: Path):
    """
    根据 YOLO txt 标签，给整张图片一个类别：
    0 normal
    1 cavity
    2 utility
    3 mixed
    """
    if not label_path.exists():
        return 0, "normal"

    text = label_path.read_text(encoding="utf-8").strip()
    if text == "":
        return 0, "normal"

    cls_set = set()
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cls_set.add(cls_id)

    if len(cls_set) == 0:
        return 0, "normal"

    if cls_set == {0}:
        return 1, "cavity"

    if cls_set == {1}:
        return 2, "utility"

    return 3, "mixed"


def collect_images_and_labels():
    img_paths = []
    label_ids = []
    label_names = []

    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    all_imgs = []
    for ext in exts:
        all_imgs.extend(sorted(IMG_DIR.glob(ext)))

    if len(all_imgs) == 0:
        raise FileNotFoundError(f"没有找到图片: {IMG_DIR}")

    for img_path in all_imgs:
        label_path = LABEL_DIR / f"{img_path.stem}.txt"
        lab_id, lab_name = get_image_level_label(label_path)

        img_paths.append(img_path)
        label_ids.append(lab_id)
        label_names.append(lab_name)

    return img_paths, np.array(label_ids), np.array(label_names)


# =========================================================
# 4. 图像读取与预处理
# =========================================================

def load_image_as_tensor(img_path: Path):
    """
    读取图片，并转成 YOLO 可输入的 tensor:
    [3, H, W], float32, 0~1
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"读取图片失败: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)

    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

    return torch.from_numpy(img)


# =========================================================
# 5. 提取 YOLO 中间层特征
# =========================================================

class YOLOFeatureExtractor:
    """
    从 YOLO 的 Detect 层输入处提取多尺度特征。

    YOLO 最后一层一般是 Detect 层。
    Detect 层的输入通常是多个尺度的 feature map。
    我们对每个尺度做全局平均池化，然后拼接成一个向量。
    """

    def __init__(self, weight_path: Path, device: str):
        self.device = device
        self.yolo = YOLO(str(weight_path))
        self.model = self.yolo.model.to(device).eval()

        self.features = None

        # YOLO 检测头一般是最后一层
        detect_layer = self.model.model[-1]

        def pre_hook(module, inputs):
            x = inputs[0]

            # x 通常是 list/tuple，包含多个尺度特征图
            if isinstance(x, (list, tuple)):
                pooled_list = []
                for fmap in x:
                    # fmap: [B, C, H, W]
                    pooled = F.adaptive_avg_pool2d(fmap, output_size=1).flatten(1)
                    pooled_list.append(pooled)
                feat = torch.cat(pooled_list, dim=1)
            else:
                feat = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)

            self.features = feat.detach().cpu()

        self.hook_handle = detect_layer.register_forward_pre_hook(pre_hook)

    @torch.no_grad()
    def extract_batch(self, batch_tensor: torch.Tensor):
        """
        batch_tensor: [B, 3, H, W]
        """
        self.features = None

        batch_tensor = batch_tensor.to(self.device)

        _ = self.model(batch_tensor)

        if self.features is None:
            raise RuntimeError("没有成功提取到 YOLO 中间层特征。")

        return self.features.numpy()

    def close(self):
        self.hook_handle.remove()


def extract_all_features(img_paths):
    extractor = YOLOFeatureExtractor(WEIGHT_PATH, DEVICE)

    all_features = []

    try:
        for i in range(0, len(img_paths), BATCH_SIZE):
            batch_paths = img_paths[i:i + BATCH_SIZE]

            batch_imgs = []
            for p in batch_paths:
                batch_imgs.append(load_image_as_tensor(p))

            batch_tensor = torch.stack(batch_imgs, dim=0)

            feats = extractor.extract_batch(batch_tensor)
            all_features.append(feats)

            print(f"[FEATURE] {min(i + BATCH_SIZE, len(img_paths))}/{len(img_paths)} done")

    finally:
        extractor.close()

    features = np.concatenate(all_features, axis=0)
    return features


# =========================================================
# 6. 画 3D t-SNE
# =========================================================

def plot_tsne_3d(z3, label_ids, label_names, img_paths):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    plot_classes = [
        (0, "normal", "o"),
        (1, "cavity", "^"),
        (2, "utility", "s"),
        (3, "mixed", "x"),
    ]

    for cls_id, cls_name, marker in plot_classes:
        mask = label_ids == cls_id
        if mask.sum() == 0:
            continue

        ax.scatter(
            z3[mask, 0],
            z3[mask, 1],
            z3[mask, 2],
            marker=marker,
            s=35,
            alpha=0.75,
            label=f"{cls_name} (n={mask.sum()})"
        )

    ax.set_title("3D t-SNE of YOLO Features on GPR Test Set")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_zlabel("t-SNE dim 3")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300)
    plt.show()

    print(f"[OK] 3D t-SNE 图已保存到: {OUT_FIG}")


# =========================================================
# 7. 主函数
# =========================================================

def main():
    print("=" * 80)
    print("YOLO Feature 3D t-SNE Visualization")
    print("=" * 80)
    print(f"[PROJECT_ROOT] {PROJECT_ROOT}")
    print(f"[IMG_DIR     ] {IMG_DIR}")
    print(f"[LABEL_DIR   ] {LABEL_DIR}")
    print(f"[WEIGHT_PATH ] {WEIGHT_PATH}")
    print(f"[OUT_DIR     ] {OUT_DIR}")
    print(f"[DEVICE      ] {DEVICE}")
    print("=" * 80)

    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(f"找不到 YOLO 权重: {WEIGHT_PATH}")

    img_paths, label_ids, label_names = collect_images_and_labels()

    print(f"[INFO] test images = {len(img_paths)}")
    unique, counts = np.unique(label_names, return_counts=True)
    print("[INFO] image-level labels:")
    for name, cnt in zip(unique, counts):
        print(f"    {name}: {cnt}")

    print("\n[STEP 1] 提取 YOLO 中间层特征...")
    features = extract_all_features(img_paths)
    print(f"[INFO] features.shape = {features.shape}")

    print("\n[STEP 2] 标准化特征...")
    features_std = StandardScaler().fit_transform(features)

    print("\n[STEP 3] 运行 3D t-SNE...")
    n_samples = features_std.shape[0]
    perplexity = min(PERPLEXITY, max(5, (n_samples - 1) // 3))

    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=RANDOM_STATE,
    )

    z3 = tsne.fit_transform(features_std)
    print(f"[INFO] tsne.shape = {z3.shape}")

    print("\n[STEP 4] 保存 npz...")
    np.savez_compressed(
        OUT_NPZ,
        features=features,
        tsne3d=z3,
        label_ids=label_ids,
        label_names=label_names,
        img_paths=np.array([str(p) for p in img_paths]),
    )
    print(f"[OK] npz 已保存到: {OUT_NPZ}")

    print("\n[STEP 5] 绘制 3D t-SNE 图...")
    plot_tsne_3d(z3, label_ids, label_names, img_paths)

    print("\n[DONE] YOLO 3D t-SNE 可视化完成")


if __name__ == "__main__":
    main()