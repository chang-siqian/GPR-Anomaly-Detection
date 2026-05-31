from pathlib import Path
import json
import time
import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ============================================================
# Fit 2D-ESN theta vectors from exported window npz files
# Input npz is produced by export_window_dataset.py
# Output is compatible with the later OCSVM script below.
# ============================================================

# ===== Path config =====
ROOT = Path(r"C:\temporary internet files\GPR_ModelSpace_New\gpr_yolo_dataset")
DATA_DIR = ROOT / "window_dataset_overlap40_w128_s32"
OUT_DIR = DATA_DIR / "theta_vectors_2desn_n30_h32_w32"

SPLITS = ["train", "val", "test"]

# ===== 2D-ESN config =====
# For the first runnable baseline, use a small resized window and diagonal reservoirs.
# This is much faster on CPU and is enough to verify the whole 2D-ESN model-space pipeline.
PROCESS_H = 32
PROCESS_W = 32
RESERVOIR_SIZE = 30
SPECTRAL_RADIUS = 0.10
INPUT_SCALE = 0.50
RIDGE_ALPHA = 1e-3
RANDOM_SEED = 42

# "diag" is fast. "full" is closer to the paper but much slower in pure Python.
RESERVOIR_MODE = "diag"   # choose from: "diag", "full"

# Set to a small number such as 100 for debugging. Use None for full dataset.
DEBUG_LIMIT = None


class TwoDESNThetaFitter:
    """
    2-Direction Echo State Network fitter for one image window.

    For every pixel x(i,j), hidden state is updated using current pixel,
    upper hidden state and left hidden state:
        h(i,j) = tanh(W_up h(i-1,j) + W_left h(i,j-1) + W_in x(i,j))

    Then a next-item prediction model is fitted by ridge regression:
        x(i,j) ~= Wyh1 h(i-1,j) + Wyh2 h(i,j-1) + bias

    The fitted theta is:
        theta = [Wyh1, Wyh2, bias]
    with dimension 2 * reservoir_size + 1.
    """

    def __init__(
        self,
        reservoir_size=30,
        spectral_radius=0.1,
        input_scale=0.5,
        ridge_alpha=1e-3,
        seed=42,
        mode="diag",
    ):
        self.n = int(reservoir_size)
        self.spectral_radius = float(spectral_radius)
        self.input_scale = float(input_scale)
        self.ridge_alpha = float(ridge_alpha)
        self.seed = int(seed)
        self.mode = str(mode).lower()

        rng = np.random.default_rng(self.seed)
        self.w_in = rng.uniform(-self.input_scale, self.input_scale, size=self.n).astype(np.float32)

        if self.mode == "diag":
            self.w_up = rng.uniform(-1.0, 1.0, size=self.n).astype(np.float32)
            self.w_left = rng.uniform(-1.0, 1.0, size=self.n).astype(np.float32)
            self.w_up = self._scale_diag(self.w_up)
            self.w_left = self._scale_diag(self.w_left)
        elif self.mode == "full":
            self.w_up = rng.normal(0.0, 1.0, size=(self.n, self.n)).astype(np.float32)
            self.w_left = rng.normal(0.0, 1.0, size=(self.n, self.n)).astype(np.float32)
            self.w_up = self._scale_full(self.w_up)
            self.w_left = self._scale_full(self.w_left)
        else:
            raise ValueError("RESERVOIR_MODE must be 'diag' or 'full'.")

    def _scale_diag(self, w):
        max_abs = float(np.max(np.abs(w)))
        if max_abs < 1e-12:
            return w
        return (w / max_abs * self.spectral_radius).astype(np.float32)

    def _scale_full(self, w):
        eigvals = np.linalg.eigvals(w.astype(np.float64))
        radius = float(np.max(np.abs(eigvals)))
        if radius < 1e-12:
            return w.astype(np.float32)
        return (w / radius * self.spectral_radius).astype(np.float32)

    def compute_hidden_states(self, img_float):
        """
        img_float: [H, W], float32, usually normalized to [-1, 1]
        return: hidden states [H, W, N]
        """
        h_img, w_img = img_float.shape
        states = np.zeros((h_img, w_img, self.n), dtype=np.float32)
        zero = np.zeros(self.n, dtype=np.float32)

        for i in range(h_img):
            for j in range(w_img):
                h_up = states[i - 1, j] if i > 0 else zero
                h_left = states[i, j - 1] if j > 0 else zero
                x = float(img_float[i, j])

                if self.mode == "diag":
                    pre = self.w_up * h_up + self.w_left * h_left + self.w_in * x
                else:
                    pre = self.w_up @ h_up + self.w_left @ h_left + self.w_in * x

                states[i, j] = np.tanh(pre).astype(np.float32)

        return states

    def fit_theta(self, img_float):
        """
        Fit one theta vector for one window image.
        return theta: [2N + 1], float32
        """
        h_img, w_img = img_float.shape
        states = self.compute_hidden_states(img_float)
        num_points = h_img * w_img

        # Z = [h_upper, h_left, bias]
        z = np.zeros((num_points, 2 * self.n + 1), dtype=np.float32)
        y = img_float.reshape(-1).astype(np.float32)

        k = 0
        for i in range(h_img):
            for j in range(w_img):
                if i > 0:
                    z[k, 0:self.n] = states[i - 1, j]
                if j > 0:
                    z[k, self.n:2 * self.n] = states[i, j - 1]
                z[k, -1] = 1.0
                k += 1

        z64 = z.astype(np.float64)
        y64 = y.astype(np.float64)
        reg = np.eye(z64.shape[1], dtype=np.float64) * self.ridge_alpha
        reg[-1, -1] = 0.0  # do not regularize bias

        a = z64.T @ z64 + reg
        b = z64.T @ y64

        try:
            theta = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(a) @ b

        return theta.astype(np.float32)


def resize_and_normalize_window(win_uint8):
    """
    Input window from exported npz: [H, W], uint8.
    Resize to PROCESS_H x PROCESS_W and normalize to [-1, 1].
    """
    im = Image.fromarray(win_uint8.astype(np.uint8), mode="L")
    im = im.resize((PROCESS_W, PROCESS_H), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32)
    arr = arr / 127.5 - 1.0
    return arr


def iter_with_progress(items, desc):
    if tqdm is not None:
        return tqdm(items, desc=desc)
    return items


def fit_split(split, fitter):
    in_path = DATA_DIR / f"{split}.npz"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {in_path}")

    data = np.load(in_path, allow_pickle=True)
    windows = data["windows"]
    labels = data["labels"]
    cls_ids = data["cls_ids"]
    x_starts = data["x_starts"]
    x_ends = data["x_ends"]
    img_names = data["img_names"]
    window_ids = data["window_ids"]

    n_total = len(windows) if DEBUG_LIMIT is None else min(len(windows), int(DEBUG_LIMIT))
    theta_dim = 2 * RESERVOIR_SIZE + 1
    theta_all = np.zeros((n_total, theta_dim), dtype=np.float32)

    print("=" * 72)
    print(f"[FIT SPLIT] {split}")
    print(f"[INPUT    ] {in_path}")
    print(f"[WINDOWS  ] original={windows.shape}, used={n_total}")
    print(f"[THETA DIM] {theta_dim}")
    print("=" * 72)

    t0 = time.time()
    for i in iter_with_progress(range(n_total), desc=f"fit {split}"):
        img = resize_and_normalize_window(windows[i])
        theta_all[i] = fitter.fit_theta(img)

        if tqdm is None and (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{n_total}")

    used_sec = time.time() - t0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{split}_theta.npz"
    np.savez_compressed(
        out_path,
        theta=theta_all,
        labels=labels[:n_total].astype(np.int8),
        cls_ids=cls_ids[:n_total].astype(np.int16),
        x_starts=x_starts[:n_total].astype(np.int32),
        x_ends=x_ends[:n_total].astype(np.int32),
        img_names=img_names[:n_total],
        window_ids=window_ids[:n_total].astype(np.int16),
    )

    info = {
        "split": split,
        "input_file": str(in_path),
        "output_file": str(out_path),
        "num_windows": int(n_total),
        "theta_shape": list(theta_all.shape),
        "process_h": PROCESS_H,
        "process_w": PROCESS_W,
        "reservoir_size": RESERVOIR_SIZE,
        "theta_dim": theta_dim,
        "spectral_radius": SPECTRAL_RADIUS,
        "input_scale": INPUT_SCALE,
        "ridge_alpha": RIDGE_ALPHA,
        "random_seed": RANDOM_SEED,
        "reservoir_mode": RESERVOIR_MODE,
        "elapsed_seconds": used_sec,
    }
    info_path = OUT_DIR / f"{split}_theta_meta.json"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] saved theta -> {out_path}")
    print(f"[TIME] {used_sec:.2f} sec")
    return info


def main():
    print("=" * 72)
    print("2D-ESN theta fitting")
    print(f"[DATA_DIR] {DATA_DIR}")
    print(f"[OUT_DIR ] {OUT_DIR}")
    print(f"[CFG     ] mode={RESERVOIR_MODE}, N={RESERVOIR_SIZE}, radius={SPECTRAL_RADIUS}, size=({PROCESS_H},{PROCESS_W})")
    print("=" * 72)

    fitter = TwoDESNThetaFitter(
        reservoir_size=RESERVOIR_SIZE,
        spectral_radius=SPECTRAL_RADIUS,
        input_scale=INPUT_SCALE,
        ridge_alpha=RIDGE_ALPHA,
        seed=RANDOM_SEED,
        mode=RESERVOIR_MODE,
    )

    summary = {}
    for split in SPLITS:
        summary[split] = fit_split(split, fitter)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "summary_2desn_theta.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 72)
    print(f"[DONE] summary -> {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
