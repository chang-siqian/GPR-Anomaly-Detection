import os
import sys
import json
import numpy as np
import torch

# ===== 项目根目录 =====
PROJECT_ROOT = r"C:\temporary internet files\GPR_ModelSpace_New"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import resnet_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NPZ_PATH = r"C:\temporary internet files\GPR_ModelSpace_New\gpr_yolo_dataset\window_dataset_center2_w128_s32\train.npz"
MAX_SAMPLES = 32   # 先只测前32个窗口


def main():
    print("=" * 72)
    print(f"[NPZ   ] {NPZ_PATH}")
    print(f"[DEVICE] {DEVICE}")
    print("=" * 72)

    if not os.path.exists(NPZ_PATH):
        print(f"[ERROR] 文件不存在: {NPZ_PATH}")
        return

    data = np.load(NPZ_PATH, allow_pickle=True)

    windows = data["windows"]      # [N, 224, 128], uint8
    labels = data["labels"]        # [N]
    cls_ids = data["cls_ids"]      # [N]

    print(f"[INFO] windows.shape = {windows.shape}, dtype = {windows.dtype}")
    print(f"[INFO] labels.shape  = {labels.shape}, dtype = {labels.dtype}")
    print(f"[INFO] cls_ids.shape = {cls_ids.shape}, dtype = {cls_ids.dtype}")

    n = min(MAX_SAMPLES, len(windows))
    windows_small = windows[:n]

    print(f"[INFO] using first {n} windows for feature extraction")
    print(f"[INFO] sample window shape = {windows_small[0].shape}")

    # ===== 实例化旧项目的 ResNet18 特征提取器 =====
    print("\n[STEP] loading ResNetFeatureExtractor ...")
    extractor = resnet_model.ResNetFeatureExtractor(device=DEVICE)

    # ===== 提特征 =====
    print("[STEP] extracting features ...")
    feats = extractor.extract_features(windows_small)

    # 转成 numpy 方便看
    if isinstance(feats, torch.Tensor):
        feats = feats.detach().cpu().numpy()

    print("\n" + "=" * 72)
    print("[RESULT]")
    print("=" * 72)
    print(f"features.shape = {feats.shape}")
    print(f"features.dtype = {feats.dtype}")
    print(f"first feature vector, first 10 dims =\n{feats[0][:10]}")

    # 顺便看看前几个标签
    print("\n[PREVIEW]")
    for i in range(min(8, n)):
        print(
            f"[{i}] label={int(labels[i])}, cls_id={int(cls_ids[i])}, "
            f"window_shape={windows_small[i].shape}"
        )


if __name__ == "__main__":
    main()