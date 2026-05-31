# -*- coding: utf-8 -*-
"""
plot_yolo_tsne_3d_multi.py

功能：
1. 加载训练好的 YOLO best.pt
2. 读取 gpr_yolo_dataset/images/test 下的图片
3. 用 YOLO Detect 层输入特征作为图像级特征
4. 对特征做 3D t-SNE
5. 按“前面模型空间图”的格式保存多张两视角图：

输出 3 张主要图片：
    1) yolo_feature_tsne3d_GT_binary_test.png
       GT 二分类：normal / abnormal
       其中 cavity、utility、mixed 都合并为 abnormal

    2) yolo_feature_tsne3d_GT_classes_test.png
       GT 多分类：normal / cavity / utility / mixed

    3) yolo_feature_tsne3d_YOLO_pred_binary_test.png
       YOLO 预测二分类：pred_normal / pred_abnormal
       根据 YOLO 预测框是否存在来判断整张图是否异常

说明：
    每个点 = 一张 test 图片
    t-SNE 1/2/3 只是可视化坐标，没有实际物理含义
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

# 只保存图片，不弹窗，避免 PyCharm 后端问题
matplotlib.use("Agg")

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

# 缓存 npz：如果已经算过特征和 t-SNE，下次可以直接读，避免重复提特征
OUT_NPZ = OUT_DIR / "yolo_feature_tsne3d_test_multi.npz"
CONFIG_JSON = OUT_DIR / "yolo_feature_tsne3d_test_multi_config.json"

# 三张主图
OUT_GT_BINARY_FIG = OUT_DIR / "yolo_feature_tsne3d_GT_binary_test.png"
OUT_GT_CLASSES_FIG = OUT_DIR / "yolo_feature_tsne3d_GT_classes_test.png"
OUT_YOLO_PRED_BINARY_FIG = OUT_DIR / "yolo_feature_tsne3d_YOLO_pred_binary_test.png"


# =========================================================
# 2. 基本参数
# =========================================================

IMGSZ = 224
BATCH_SIZE = 32

TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_PRED_DEVICE = 0 if torch.cuda.is_available() else "cpu"

# 是否优先使用已经保存的 npz 缓存
# 第一次运行时没有缓存，会自动提特征、跑 t-SNE、跑 YOLO 预测。
# 之后再运行时，如果缓存存在，会直接读取缓存并画图。
USE_CACHE = True

# 如果你想强制重新提取 YOLO 特征和重新跑 t-SNE，就改成 True
FORCE_RECOMPUTE_FEATURES_TSNE = False

# 如果你想强制重新跑 YOLO 预测，就改成 True
FORCE_RECOMPUTE_YOLO_PRED = False

# YOLO 预测框置信度阈值
YOLO_CONF_THRES = 0.25

# t-SNE 参数
RANDOM_STATE = 42
PERPLEXITY = 30

# 画图参数
MARKER_SIZE = 24
ALPHA = 0.85
SHOW_COUNTS_IN_LEGEND = False


# =========================================================
# 3. 类别定义
# =========================================================

# YOLO txt 中的类别：
# 0 cavity
# 1 utility
YOLO_TXT_CLASS_NAMES = {
    0: "cavity",
    1: "utility",
}

# image-level GT 多分类：
# 0 normal
# 1 cavity
# 2 utility
# 3 mixed
GT_CLASS_NAMES = {
    0: "normal",
    1: "cavity",
    2: "utility",
    3: "mixed",
}

GT_CLASS_COLORS = {
    0: "blue",
    1: "red",
    2: "black",
    3: "magenta",
}

# image-level GT 二分类：
# 0 normal
# 1 abnormal
GT_BINARY_CLASS_NAMES = {
    0: "normal",
    1: "abnormal",
}

GT_BINARY_CLASS_COLORS = {
    0: "blue",
    1: "red",
}

# YOLO 预测二分类：
# 0 pred_normal
# 1 pred_abnormal
PRED_BINARY_CLASS_NAMES = {
    0: "pred_normal",
    1: "pred_abnormal",
}

PRED_BINARY_CLASS_COLORS = {
    0: "blue",
    1: "red",
}


# =========================================================
# 4. 读取标签，转成 image-level GT 类别
# =========================================================

def get_image_level_label(label_path: Path):
    """
    根据 YOLO txt 标签，给整张图片一个 image-level GT 类别：
        0 normal  : 没有任何目标框
        1 cavity  : 只有 cavity
        2 utility : 只有 utility
        3 mixed   : 同时有 cavity 和 utility
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


def collect_images_and_gt_labels():
    """
    收集 test 图片路径与 image-level GT 标签。
    """
    img_paths = []

    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    for ext in exts:
        img_paths.extend(sorted(IMG_DIR.glob(ext)))

    if len(img_paths) == 0:
        raise FileNotFoundError(f"没有找到图片: {IMG_DIR}")

    label_ids = []
    label_names = []

    for img_path in img_paths:
        label_path = LABEL_DIR / f"{img_path.stem}.txt"
        lab_id, lab_name = get_image_level_label(label_path)

        label_ids.append(lab_id)
        label_names.append(lab_name)

    label_ids = np.array(label_ids, dtype=int)
    label_names = np.array(label_names)

    # 二分类：非 normal 都算 abnormal
    gt_binary = (label_ids != 0).astype(int)

    return img_paths, label_ids, label_names, gt_binary


# =========================================================
# 5. 图像读取与预处理
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
# 6. 提取 YOLO 中间层特征
# =========================================================

class YOLOFeatureExtractor:
    """
    从 YOLO Detect 层输入处提取多尺度特征。

    YOLO 最后一层一般是 Detect 层。
    Detect 层的输入通常是多个尺度的 feature map。
    这里对每个尺度做全局平均池化，然后拼接成一个向量。
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
    extractor = YOLOFeatureExtractor(WEIGHT_PATH, TORCH_DEVICE)
    all_features = []

    try:
        for i in range(0, len(img_paths), BATCH_SIZE):
            batch_paths = img_paths[i:i + BATCH_SIZE]

            batch_imgs = [load_image_as_tensor(p) for p in batch_paths]
            batch_tensor = torch.stack(batch_imgs, dim=0)

            feats = extractor.extract_batch(batch_tensor)
            all_features.append(feats)

            print(f"[FEATURE] {min(i + BATCH_SIZE, len(img_paths))}/{len(img_paths)} done")

    finally:
        extractor.close()

    features = np.concatenate(all_features, axis=0)
    return features


# =========================================================
# 7. YOLO image-level 预测标签
# =========================================================

def yolo_prediction_to_image_level(result):
    """
    将单张图片的 YOLO 检测结果转成 image-level 预测类别：
        0 normal
        1 cavity
        2 utility
        3 mixed

    规则：
        没有预测框 -> normal
        只预测到 cavity -> cavity
        只预测到 utility -> utility
        同时预测到 cavity 和 utility -> mixed
    """
    if result.boxes is None or len(result.boxes) == 0:
        return 0

    cls_arr = result.boxes.cls.detach().cpu().numpy().astype(int)
    cls_set = set(cls_arr.tolist())

    if len(cls_set) == 0:
        return 0

    if cls_set == {0}:
        return 1

    if cls_set == {1}:
        return 2

    return 3


def predict_yolo_image_level(img_paths):
    """
    用训练好的 YOLO 对 test 图片做预测，并转成 image-level 预测标签。
    """
    model = YOLO(str(WEIGHT_PATH))

    pred_class_ids = []

    for i in range(0, len(img_paths), BATCH_SIZE):
        batch_paths = img_paths[i:i + BATCH_SIZE]

        results = model.predict(
            source=[str(p) for p in batch_paths],
            imgsz=IMGSZ,
            conf=YOLO_CONF_THRES,
            device=YOLO_PRED_DEVICE,
            verbose=False,
            save=False,
        )

        for r in results:
            pred_class_ids.append(yolo_prediction_to_image_level(r))

        print(f"[YOLO PRED] {min(i + BATCH_SIZE, len(img_paths))}/{len(img_paths)} done")

    pred_class_ids = np.array(pred_class_ids, dtype=int)

    # 二分类：只要有任何预测框，就算 pred_abnormal
    pred_binary = (pred_class_ids != 0).astype(int)

    return pred_class_ids, pred_binary


# =========================================================
# 8. t-SNE
# =========================================================

def compute_tsne_3d(features):
    print("\n[STEP] 标准化 YOLO 特征...")
    features_std = StandardScaler().fit_transform(features)

    n_samples = features_std.shape[0]
    perplexity = min(PERPLEXITY, max(5, (n_samples - 1) // 3))

    print("\n[STEP] 运行 3D t-SNE...")
    print(f"[INFO] n_samples  = {n_samples}")
    print(f"[INFO] input_dim  = {features_std.shape[1]}")
    print(f"[INFO] perplexity = {perplexity}")

    # sklearn 新旧版本兼容
    try:
        tsne = TSNE(
            n_components=3,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=1500,
            random_state=RANDOM_STATE,
        )
    except TypeError:
        tsne = TSNE(
            n_components=3,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            n_iter=1500,
            random_state=RANDOM_STATE,
        )

    z3 = tsne.fit_transform(features_std)
    return z3.astype(np.float32)


# =========================================================
# 9. 画两视角 3D 图
# =========================================================

def set_3d_axis_style(ax):
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")

    # 不显示具体数值，强调 t-SNE 坐标只是可视化坐标
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.grid(False)

    try:
        ax.set_box_aspect((1.5, 1.0, 0.75))
    except Exception:
        pass


def plot_3d_two_views(
    z3,
    class_ids,
    class_names,
    class_colors,
    out_path,
    title,
):
    """
    一张图片里放两个视角，标注为 (a) 和 (b)，和前面的模型空间图保持一致。
    """
    fig = plt.figure(figsize=(9, 12))

    views = [
        (18, -65, "(a)"),
        (18, 35, "(b)"),
    ]

    for i, (elev, azim, sub_label) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 1, i, projection="3d")

        unique_classes = sorted(np.unique(class_ids).tolist())

        for c in unique_classes:
            mask = class_ids == c
            if mask.sum() == 0:
                continue

            name = class_names.get(int(c), str(c))
            color = class_colors.get(int(c), "gray")

            if SHOW_COUNTS_IN_LEGEND:
                legend_label = f"{name} (n={mask.sum()})"
            else:
                legend_label = name

            ax.scatter(
                z3[mask, 0],
                z3[mask, 1],
                z3[mask, 2],
                s=MARKER_SIZE,
                c=color,
                label=legend_label,
                alpha=ALPHA,
                depthshade=False,
            )

        ax.view_init(elev=elev, azim=azim)
        set_3d_axis_style(ax)
        ax.legend(loc="upper right", fontsize=8, frameon=True)
        ax.set_title(title, fontsize=11)
        ax.text2D(0.48, -0.08, sub_label, transform=ax.transAxes, fontsize=16)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] saved figure -> {out_path}")


def print_class_stat(title, ids, names):
    unique, counts = np.unique(ids, return_counts=True)
    print(title)
    for k, v in zip(unique, counts):
        print(f"    {names.get(int(k), str(k))}: {int(v)}")


# =========================================================
# 10. 主函数
# =========================================================

def main():
    print("=" * 80)
    print("YOLO Feature 3D t-SNE Multi-Visualization")
    print("=" * 80)
    print(f"[PROJECT_ROOT] {PROJECT_ROOT}")
    print(f"[IMG_DIR     ] {IMG_DIR}")
    print(f"[LABEL_DIR   ] {LABEL_DIR}")
    print(f"[WEIGHT_PATH ] {WEIGHT_PATH}")
    print(f"[OUT_DIR     ] {OUT_DIR}")
    print(f"[TORCH_DEVICE] {TORCH_DEVICE}")
    print(f"[PRED_DEVICE ] {YOLO_PRED_DEVICE}")
    print("=" * 80)

    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(f"找不到 YOLO 权重: {WEIGHT_PATH}")

    img_paths, gt_class_ids, gt_class_names_arr, gt_binary = collect_images_and_gt_labels()
    img_paths_str = np.array([str(p) for p in img_paths])

    print(f"[INFO] test images = {len(img_paths)}")
    print_class_stat("[GT MULTI CLASS STAT]", gt_class_ids, GT_CLASS_NAMES)
    print_class_stat("[GT BINARY STAT]", gt_binary, GT_BINARY_CLASS_NAMES)

    cache_ok = (
        USE_CACHE
        and OUT_NPZ.exists()
        and not FORCE_RECOMPUTE_FEATURES_TSNE
    )

    features = None
    z3 = None
    pred_class_ids = None
    pred_binary = None

    if cache_ok:
        print("\n[STEP] 读取缓存 npz...")
        cache = np.load(OUT_NPZ, allow_pickle=True)

        if "features" in cache.files and "tsne3d" in cache.files:
            features = cache["features"]
            z3 = cache["tsne3d"]

            print(f"[OK] loaded features: {features.shape}")
            print(f"[OK] loaded tsne3d : {z3.shape}")
        else:
            print("[WARN] 缓存缺少 features 或 tsne3d，将重新计算。")
            cache_ok = False

        if (
            cache_ok
            and "pred_class_ids" in cache.files
            and "pred_binary" in cache.files
            and not FORCE_RECOMPUTE_YOLO_PRED
        ):
            pred_class_ids = cache["pred_class_ids"].astype(int)
            pred_binary = cache["pred_binary"].astype(int)
            print("[OK] loaded YOLO prediction from cache")

    if not cache_ok:
        print("\n[STEP 1] 提取 YOLO 中间层特征...")
        features = extract_all_features(img_paths)
        print(f"[INFO] features.shape = {features.shape}")

        print("\n[STEP 2] 计算 3D t-SNE...")
        z3 = compute_tsne_3d(features)
        print(f"[INFO] tsne3d.shape = {z3.shape}")

    if pred_class_ids is None or pred_binary is None or FORCE_RECOMPUTE_YOLO_PRED:
        print("\n[STEP 3] 运行 YOLO 预测，并转成 image-level 标签...")
        pred_class_ids, pred_binary = predict_yolo_image_level(img_paths)

    print_class_stat("[YOLO PRED MULTI CLASS STAT]", pred_class_ids, GT_CLASS_NAMES)
    print_class_stat("[YOLO PRED BINARY STAT]", pred_binary, PRED_BINARY_CLASS_NAMES)

    print("\n[STEP 4] 保存 npz 缓存...")
    np.savez_compressed(
        OUT_NPZ,
        features=features,
        tsne3d=z3,
        gt_class_ids=gt_class_ids,
        gt_class_names=gt_class_names_arr,
        gt_binary=gt_binary,
        pred_class_ids=pred_class_ids,
        pred_binary=pred_binary,
        img_paths=img_paths_str,
    )
    print(f"[OK] npz saved -> {OUT_NPZ}")

    config = {
        "project_root": str(PROJECT_ROOT),
        "img_dir": str(IMG_DIR),
        "label_dir": str(LABEL_DIR),
        "weight_path": str(WEIGHT_PATH),
        "out_dir": str(OUT_DIR),
        "imgs_size": IMGSZ,
        "batch_size": BATCH_SIZE,
        "random_state": RANDOM_STATE,
        "perplexity": PERPLEXITY,
        "yolo_conf_thres": YOLO_CONF_THRES,
        "figures": {
            "gt_binary": str(OUT_GT_BINARY_FIG),
            "gt_classes": str(OUT_GT_CLASSES_FIG),
            "yolo_pred_binary": str(OUT_YOLO_PRED_BINARY_FIG),
        },
    }

    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"[OK] config saved -> {CONFIG_JSON}")

    print("\n[STEP 5] 绘制 3D t-SNE 图...")

    # 图 1：GT 二分类 normal / abnormal
    plot_3d_two_views(
        z3=z3,
        class_ids=gt_binary,
        class_names=GT_BINARY_CLASS_NAMES,
        class_colors=GT_BINARY_CLASS_COLORS,
        out_path=OUT_GT_BINARY_FIG,
        title="3D t-SNE visualization of YOLO feature space (GT normal/abnormal)",
    )

    # 图 2：GT 多分类 normal / cavity / utility / mixed
    plot_3d_two_views(
        z3=z3,
        class_ids=gt_class_ids,
        class_names=GT_CLASS_NAMES,
        class_colors=GT_CLASS_COLORS,
        out_path=OUT_GT_CLASSES_FIG,
        title="3D t-SNE visualization of YOLO feature space (GT classes)",
    )

    # 图 3：YOLO 预测二分类 pred_normal / pred_abnormal
    plot_3d_two_views(
        z3=z3,
        class_ids=pred_binary,
        class_names=PRED_BINARY_CLASS_NAMES,
        class_colors=PRED_BINARY_CLASS_COLORS,
        out_path=OUT_YOLO_PRED_BINARY_FIG,
        title=f"3D t-SNE visualization of YOLO feature space (YOLO pred, conf={YOLO_CONF_THRES})",
    )

    print("\n" + "=" * 80)
    print("[DONE] YOLO 3D t-SNE 多图可视化完成")
    print(f"[OUT_DIR] {OUT_DIR}")
    print("[FIG 1  ]", OUT_GT_BINARY_FIG)
    print("[FIG 2  ]", OUT_GT_CLASSES_FIG)
    print("[FIG 3  ]", OUT_YOLO_PRED_BINARY_FIG)
    print("=" * 80)


if __name__ == "__main__":
    main()
