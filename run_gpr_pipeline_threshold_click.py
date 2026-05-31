import sys
import subprocess
from pathlib import Path
from datetime import datetime
import io

#数据集+ResNet18+PCA+局部动态模拟组合theta+模拟空间异常监测或分类+合并成连续的异常区段+对比评估

# Reconfigure stdout to handle encoding errors gracefully on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def run_step(python_exe: str, script_path: Path, log_dir: Path):
    step_name = script_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{timestamp}_{step_name}.log"

    print("\n" + "=" * 80)
    print(f"[RUN ] {script_path}")
    print(f"[LOG ] {log_path}")
    print("=" * 80)

    with open(log_path, "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            [python_exe, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(script_path.parent),  # 很关键：每个脚本在它自己的目录里运行
        )

        for line in process.stdout:
            print(line, end="")
            f.write(line)

        process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"步骤失败: {script_path.name}，返回码 = {process.returncode}")

    print(f"\n[OK] {script_path.name} 运行完成。")


def main():
    # 当前这个一键脚本放在项目根目录
    project_root = Path(__file__).resolve().parent
    gpr_dir = project_root / "gpr_yolo_dataset"
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable

    print("=" * 80)
    print("[PROJECT ROOT]", project_root)
    print("[PYTHON      ]", python_exe)
    print("[GPR DIR     ]", gpr_dir)
    print("[LOG DIR     ]", log_dir)
    print("=" * 80)

    # 按你现在项目的真实位置来
    steps = [
        gpr_dir / "export_window_dataset.py",
        gpr_dir / "inspect_exported_npz.py",
        gpr_dir / "export_resnet_features_all.py",
        project_root / "fit_pca_and_transform_features.py",
        project_root / "fit_all_theta_vectors.py",
        project_root / "run_ocsm_val_threshold.py",   # 这里改成你现在在用的脚本
    ]

    # 先检查脚本是否都存在
    missing = [p for p in steps if not p.exists()]
    if missing:
        print("\n[ERROR] 以下脚本不存在：")
        for p in missing:
            print("  -", p)
        return

    for script in steps:
        run_step(python_exe, script, log_dir)

    print("\n" + "=" * 80)
    print("[DONE] 全部步骤运行完成。")
    print("=" * 80)


if __name__ == "__main__":
    main()