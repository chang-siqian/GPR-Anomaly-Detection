import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
import numpy as np
try:
    from config import PATCH_WIDTH, PATCH_STRIDE
except Exception:
    # 默认值：让宽度 128 的窗口切成 15 个 patch
    PATCH_WIDTH = 16
    PATCH_STRIDE = 8


#ResNet18特征提取方法

# ==========================================
# 1. Dataset: 负责把窗口切成序列 (Patch Sequence)
# ==========================================
class GPRPatchSequenceDataset(Dataset):
    def __init__(self, windows, positions=None, patch_width=50, stride=25):
        """
        :param windows: (N, H, W) 如 (N, 321, 300)
        :param patch_width: 每个小块的宽度 (文档建议 32 或 50)
        :param stride: 滑动步长 (文档建议重叠)
        """
        self.windows = windows
        self.positions = positions
        self.patch_width = patch_width
        self.stride = stride

        # 预计算切片位置
        H, W = windows.shape[1], windows.shape[2]
        self.starts = range(0, W - patch_width + 1, stride)
        self.T = len(self.starts)  # 序列长度 T

        # ResNet 预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # 取出一个完整窗口 (H, W)
        window = self.windows[idx]

        # 归一化到 [0, 255] uint8，方便 ToPILImage 处理
        # 假设输入已经在 [-1, 1] 之间
        win_norm = (window - window.min()) / (window.max() - window.min() + 1e-6)
        win_uint8 = (win_norm * 255).astype(np.uint8)

        patches = []
        for s in self.starts:
            # 切片: [所有行, 列片段]
            patch = win_uint8[:, s: s + self.patch_width]

            # 转为 3 通道 (H, p, 3)
            patch_rgb = np.stack([patch] * 3, axis=-1)

            # 预处理 -> (3, 224, 224)
            patch_tensor = self.transform(patch_rgb)
            patches.append(patch_tensor)

        # 堆叠成 (T, 3, 224, 224)
        seq = torch.stack(patches)

        # ✅ positions 为空也能跑
        if self.positions is None:
            return seq

        return seq, self.positions[idx]


# ==========================================
# 2. 特征提取器: 输出 (N, T, 512)
# ==========================================
class ResNetFeatureExtractor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"   [模型] 正在加载 ResNet18 (Device: {self.device})...")

        # 加载预训练 ResNet18
        try:
            weights = models.ResNet18_Weights.DEFAULT
            resnet = models.resnet18(weights=weights)
        except:
            resnet = models.resnet18(pretrained=True)

        # 去掉全连接层
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

        self.pca = None

    def extract_features(self, windows, batch_size=32):
        """
        输入: (N, H, W)
        输出: (N, T, 512) - T 是切片数量
        """
        if len(windows) == 0:
            return np.array([])

        # 使用 Dataset 自动切片
        dataset = GPRPatchSequenceDataset(
            windows,
            patch_width=PATCH_WIDTH,
            stride=PATCH_STRIDE
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_features = []

        with torch.no_grad():
            for batch in dataloader:
                # 兼容 (seq) 或 (seq, pos)
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    batch_seqs, batch_pos = batch
                else:
                    batch_seqs = batch
                    batch_pos = None

                # batch_seqs shape: (Batch, T, 3, 224, 224)
                B, T, C, H, W = batch_seqs.shape

                # 展平 batch 和 T，一次性喂给 ResNet 以加速
                # input shape: (B*T, 3, 224, 224)
                batch_flat = batch_seqs.view(B * T, C, H, W).to(self.device)

                # ResNet output: (B*T, 512, 1, 1)
                feats = self.model(batch_flat)

                # Reshape 回 (B, T, 512)
                feats = feats.view(B, T, -1)
                all_features.append(feats.cpu().numpy())

        # 拼接所有 batch
        return np.concatenate(all_features, axis=0)

    def fit_pca(self, features, n_components=32):
        """
        输入: (N, T, 512)
        操作: 展平 -> 拟合 -> 记录 PCA 参数
        """
        if features.size == 0: return

        # 展平为 2D 矩阵 (N*T, 512)
        N, T, D = features.shape
        flat_feats = features.reshape(-1, D)

        # 确保组件数不超过样本数
        n_components = min(n_components, flat_feats.shape[0], flat_feats.shape[1])

        self.pca = PCA(n_components=n_components)
        self.pca.fit(flat_feats)

    def reduce_dimension(self, features):
        """
        输入: (N, T, 512)
        输出: (N, T, 32)
        """
        if features.size == 0: return np.array([])
        if self.pca is None: self.fit_pca(features)

        N, T, D = features.shape
        flat_feats = features.reshape(-1, D)

        # 降维 -> (N*T, 32)
        reduced = self.pca.transform(flat_feats)

        # 变回 3D -> (N, T, 32)
        return reduced.reshape(N, T, -1)