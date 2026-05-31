# -*- coding: utf-8 -*-
"""
plot_2desn_real_fitting_loss.py

作用：
重新计算 2D-ESN 的真实 fitting loss-like curve。

核心思想：
每个窗口图像 -> 2D-ESN hidden states -> ridge regression 拟合输出层
-> 计算 next-item prediction MSE

生成的图：
1. 2D-ESN running average fitting loss curve
2. 2D-ESN rolling fitting loss curve
3. 2D-ESN fitting loss distribution
4. normal / abnormal fitting loss distribution

注意：
这仍然不是 YOLO 那种 epoch-loss。
因为 2D-ESN 是每个窗口闭式拟合一次，不是按 epoch 反复训练。
但这个是 2D-ESN 对应的真正拟合误差曲线。
"""

from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 路径配置
# ============================================================

PROJECT_ROOT = Path(r"C:\temporary internet files\GPR_ModelSpace_New")

DATA_DIR = (
    PROJECT_ROOT
    / "gpr_yolo_dataset"
    / "window_dataset_overlap40_w128_s32"
)

OUT_DIR = (
    DATA_DIR
    / "theta_vectors_2desn_n30_h32_w32"
    / "real_fitting_loss"
)

FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 2D-ESN 参数
# ============================================================

N_RESERVOIR = 30
SPECTRAL_RADIUS = 0.1
TARGET_H = 32
TARGET_W = 32
RIDGE_LAMBDA = 1e-6
RANDOM_SEED = 42

ROLLING_WINDOW = 100

# 如果你想先快速测试，可以改成 300。
# 如果要完整结果，就保持 None。
MAX_WINDOWS_PER_SPLIT = None


# ============================================================
# 3. 可选 tqdm
# ============================================================

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ============================================================
# 4. 工具函数
# ============================================================

def save_fig(name):
    out_path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved -> {out_path}")


def rolling_mean(x, window=100):
    x = np.asarray(x, dtype=float).reshape(-1)
    if len(x) < window:
        return x
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="valid")


def running_average(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.cumsum(x) / (np.arange(len(x)) + 1)


def resize_image_to_32x32(img):
    """
    把窗口图像 resize 到 32x32。
    优先用 cv2；如果没有 cv2，就用 PIL。
    """
    img = np.asarray(img)

    if img.ndim == 3:
        img = img[..., 0]

    try:
        import cv2
        resized = cv2.resize(
            img,
            (TARGET_W, TARGET_H),
            interpolation=cv2.INTER_AREA,
        )
    except Exception:
        from PIL import Image
        pil_img = Image.fromarray(img.astype(np.uint8))
        pil_img = pil_img.resize((TARGET_W, TARGET_H))
        resized = np.asarray(pil_img)

    resized = resized.astype(np.float32)

    # 归一化到 [0, 1]
    if resized.max() > 1.5:
        resized = resized / 255.0

    return resized


def make_fixed_2desn_weights():
    """
    生成固定的 2D-ESN reservoir 权重。
    这里采用 diag mode，对应你日志里的 mode=diag。
    theta 维度 = 2*N + 1 = 61。
    """
    rng = np.random.default_rng(RANDOM_SEED)

    w_in = rng.uniform(-1.0, 1.0, size=(N_RESERVOIR,)).astype(np.float32)
    w_up = rng.uniform(-1.0, 1.0, size=(N_RESERVOIR,)).astype(np.float32)
    w_left = rng.uniform(-1.0, 1.0, size=(N_RESERVOIR,)).astype(np.float32)

    def scale_to_radius(w):
        m = np.max(np.abs(w))
        if m < 1e-12:
            return w
        return w / m * SPECTRAL_RADIUS

    w_up = scale_to_radius(w_up)
    w_left = scale_to_radius(w_left)

    return w_in, w_up, w_left


def compute_hidden_states_2desn(img32, w_in, w_up, w_left):
    """
    2D-ESN hidden state:
    h(i,j) = tanh(w_up * h(i-1,j) + w_left * h(i,j-1) + w_in * x(i,j))
    """
    H, W = img32.shape
    N = len(w_in)

    hidden = np.zeros((H, W, N), dtype=np.float32)

    zero = np.zeros((N,), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            h_up = hidden[i - 1, j] if i > 0 else zero
            h_left = hidden[i, j - 1] if j > 0 else zero

            hidden[i, j] = np.tanh(
                w_up * h_up
                + w_left * h_left
                + w_in * img32[i, j]
            )

    return hidden


def fit_output_and_compute_mse(img32, hidden):
    """
    对当前窗口拟合 2D-ESN 输出层：

    y(i,j) = Wyh1 * h(i-1,j) + Wyh2 * h(i,j-1) + bias

    返回：
    theta: [2N + 1]
    mse  : next-item prediction MSE
    """
    H, W = img32.shape
    N = hidden.shape[-1]

    # 预测区域：排除第一行和第一列
    # target 是当前点 x(i,j)
    y_true = img32[1:, 1:].reshape(-1, 1)

    # 上方 hidden: h(i-1,j)
    h_up = hidden[:-1, 1:, :].reshape(-1, N)

    # 左侧 hidden: h(i,j-1)
    h_left = hidden[1:, :-1, :].reshape(-1, N)

    bias = np.ones((y_true.shape[0], 1), dtype=np.float32)

    # Phi: [num_points, 2N + 1]
    Phi = np.concatenate([h_up, h_left, bias], axis=1).astype(np.float64)
    y_true = y_true.astype(np.float64)

    dim = Phi.shape[1]
    reg = np.eye(dim, dtype=np.float64) * RIDGE_LAMBDA
    reg[-1, -1] = 0.0  # bias 不正则化

    # ridge closed-form
    theta = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ y_true)

    y_pred = Phi @ theta
    mse = np.mean((y_pred - y_true) ** 2)

    theta = theta.reshape(-1).astype(np.float32)

    return theta, float(mse)


def compute_one_window_loss(window, weights):
    img32 = resize_image_to_32x32(window)
    w_in, w_up, w_left = weights

    hidden = compute_hidden_states_2desn(img32, w_in, w_up, w_left)
    theta, mse = fit_output_and_compute_mse(img32, hidden)

    return theta, mse


# ============================================================
# 5. 处理 train / val / test
# ============================================================

def process_split(split):
    npz_path = DATA_DIR / f"{split}.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {npz_path}")

    print("\n" + "=" * 80)
    print(f"[PROCESS SPLIT] {split}")
    print(f"[INPUT] {npz_path}")
    print("=" * 80)

    data = np.load(npz_path, allow_pickle=True)

    print("keys =", list(data.keys()))

    windows = data["windows"]
    labels = data["labels"]

    if MAX_WINDOWS_PER_SPLIT is not None:
        windows = windows[:MAX_WINDOWS_PER_SPLIT]
        labels = labels[:MAX_WINDOWS_PER_SPLIT]

    num_windows = len(windows)

    print(f"[WINDOWS] {windows.shape}")
    print(f"[LABELS ] {labels.shape}")
    print(f"[USE    ] {num_windows}")
    print(f"[CFG    ] N={N_RESERVOIR}, radius={SPECTRAL_RADIUS}, size=({TARGET_H},{TARGET_W})")
    print(f"[THETA DIM] {2 * N_RESERVOIR + 1}")

    weights = make_fixed_2desn_weights()

    thetas = np.zeros((num_windows, 2 * N_RESERVOIR + 1), dtype=np.float32)
    losses = np.zeros((num_windows,), dtype=np.float32)

    if tqdm is not None:
        iterator = tqdm(range(num_windows), desc=f"2D-ESN loss {split}")
    else:
        iterator = range(num_windows)

    start_time = time.time()

    for idx in iterator:
        theta, mse = compute_one_window_loss(windows[idx], weights)
        thetas[idx] = theta
        losses[idx] = mse

    cost = time.time() - start_time

    print(
        f"[LOSS] {split}: "
        f"mean={losses.mean():.8f}, "
        f"min={losses.min():.8f}, "
        f"max={losses.max():.8f}, "
        f"std={losses.std():.8f}"
    )
    print(f"[TIME] {cost:.2f} sec")

    out_path = OUT_DIR / f"{split}_real_2desn_fitting_loss.npz"

    np.savez(
        out_path,
        thetas=thetas,
        fitting_losses=losses,
        labels=labels,
        split=split,
        n_reservoir=N_RESERVOIR,
        spectral_radius=SPECTRAL_RADIUS,
        target_h=TARGET_H,
        target_w=TARGET_W,
        ridge_lambda=RIDGE_LAMBDA,
    )

    print(f"[OK] saved -> {out_path}")

    return losses, labels


# ============================================================
# 6. 画图
# ============================================================

def plot_running_average_loss(all_losses):
    plt.figure(figsize=(9, 5))

    for split, losses in all_losses.items():
        y = running_average(losses)
        x = np.arange(len(y))
        plt.plot(x, y, label=f"{split} running average MSE")

    plt.xlabel("Processed window index")
    plt.ylabel("Running average MSE")
    plt.title("2D-ESN Fitting Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    save_fig("01_2desn_running_average_fitting_loss_curve.png")


def plot_rolling_loss(all_losses):
    plt.figure(figsize=(9, 5))

    for split, losses in all_losses.items():
        y = rolling_mean(losses, window=ROLLING_WINDOW)
        x = np.arange(len(y))
        plt.plot(x, y, label=f"{split} rolling MSE")

    plt.xlabel(f"Window index, rolling window = {ROLLING_WINDOW}")
    plt.ylabel("Next-item prediction MSE")
    plt.title("2D-ESN Rolling Fitting Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    save_fig("02_2desn_rolling_fitting_loss_curve.png")


def plot_loss_distribution(all_losses):
    plt.figure(figsize=(9, 5))

    for split, losses in all_losses.items():
        plt.hist(losses, bins=60, alpha=0.5, label=split)

    plt.xlabel("Next-item prediction MSE")
    plt.ylabel("Number of windows")
    plt.title("2D-ESN Fitting Loss Distribution")
    plt.legend()
    plt.grid(alpha=0.3)

    save_fig("03_2desn_fitting_loss_distribution.png")


def plot_normal_abnormal_distribution(all_losses, all_labels):
    plt.figure(figsize=(9, 5))

    for split in all_losses.keys():
        losses = all_losses[split]
        labels = all_labels[split].astype(int)

        normal_losses = losses[labels == 0]
        abnormal_losses = losses[labels == 1]

        if split == "train":
            plt.hist(normal_losses, bins=60, alpha=0.5, label="normal windows")
            plt.hist(abnormal_losses, bins=60, alpha=0.5, label="abnormal windows")

    plt.xlabel("Next-item prediction MSE")
    plt.ylabel("Number of windows")
    plt.title("2D-ESN Fitting Loss Distribution: Normal vs Abnormal")
    plt.legend()
    plt.grid(alpha=0.3)

    save_fig("04_2desn_normal_vs_abnormal_fitting_loss_distribution.png")


def plot_mean_loss_bar(all_losses):
    splits = list(all_losses.keys())
    means = [float(np.mean(all_losses[s])) for s in splits]
    stds = [float(np.std(all_losses[s])) for s in splits]

    x = np.arange(len(splits))

    plt.figure(figsize=(7, 5))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, splits)
    plt.ylabel("Mean next-item prediction MSE")
    plt.title("2D-ESN Mean Fitting Loss by Split")
    plt.grid(axis="y", alpha=0.3)

    save_fig("05_2desn_mean_fitting_loss_by_split.png")


# ============================================================
# 7. 主函数
# ============================================================

def main():
    print("=" * 80)
    print("Compute and Plot Real 2D-ESN Fitting Loss")
    print("=" * 80)
    print(f"[DATA_DIR] {DATA_DIR}")
    print(f"[OUT_DIR ] {OUT_DIR}")
    print(f"[FIG_DIR ] {FIG_DIR}")

    all_losses = {}
    all_labels = {}

    for split in ["train", "val", "test"]:
        losses, labels = process_split(split)
        all_losses[split] = losses
        all_labels[split] = labels

    plot_running_average_loss(all_losses)
    plot_rolling_loss(all_losses)
    plot_loss_distribution(all_losses)
    plot_normal_abnormal_distribution(all_losses, all_labels)
    plot_mean_loss_bar(all_losses)

    print("\n[DONE] real 2D-ESN fitting loss figures generated.")
    print(f"图片输出目录: {FIG_DIR}")


if __name__ == "__main__":
    main()