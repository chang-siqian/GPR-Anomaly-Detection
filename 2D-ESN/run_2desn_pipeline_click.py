from pathlib import Path
import subprocess
import sys
import os
import time


# ============================================================
# 一键运行 2D-ESN 模型空间流程
#
# 当前脚本可以放在：
# 1. 项目根目录 GPR_ModelSpace_New 下
# 2. 2D-ESN 文件夹下
#
# 运行顺序：
# 1. export_window_dataset.py
# 2. inspect_exported_npz.py
# 3. fit_2desn_theta_vectors.py
# 4. run_ocsvm_2desn_val_threshold.py
# 5. eval_segment_from_2desn_preds.py
# ============================================================


# ===== 是否运行每一步 =====
# 第一次建议前两个保持 True
# 后面窗口数据已经导出过，就可以把前两个改成 False，节省时间
RUN_EXPORT_WINDOW_DATASET = True
RUN_INSPECT_EXPORTED_NPZ = True

RUN_FIT_2DESN_THETA = True
RUN_OCSVM_2DESN = True
RUN_SEGMENT_EVAL = True


def find_project_root():
    """
    自动寻找项目根目录。

    正确的项目根目录应该同时包含：
    - gpr_yolo_dataset 文件夹
    - 2D-ESN 文件夹
    """
    cur = Path(__file__).resolve().parent

    candidates = [
        cur,
        cur.parent,
        cur.parent.parent,
    ]

    for c in candidates:
        if (c / "gpr_yolo_dataset").exists() and (c / "2D-ESN").exists():
            return c

    raise FileNotFoundError(
        "无法自动找到项目根目录。\n"
        "请确认项目结构中同时存在 gpr_yolo_dataset 和 2D-ESN 文件夹。\n"
        f"当前脚本位置: {Path(__file__).resolve()}"
    )


def run_python_script(step_name, script_path, project_root):
    """
    用当前 PyCharm 解释器运行指定 Python 脚本
    """
    script_path = Path(script_path)

    print("\n" + "=" * 80)
    print(f"[STEP] {step_name}")
    print(f"[SCRIPT] {script_path}")
    print("=" * 80)

    if not script_path.exists():
        raise FileNotFoundError(f"脚本不存在: {script_path}")

    env = os.environ.copy()

    # 尽量减少 Windows / PyCharm 输出乱码问题
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    start_time = time.time()

    cmd = [
        sys.executable,
        str(script_path)
    ]

    print(f"[CMD] {' '.join(cmd)}")
    print(f"[CWD] {project_root}")
    print("-" * 80)

    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        check=False
    )

    cost = time.time() - start_time

    print("-" * 80)
    print(f"[STEP DONE] {step_name}")
    print(f"[TIME] {cost:.2f} 秒")
    print(f"[RETURN CODE] {result.returncode}")

    if result.returncode != 0:
        raise RuntimeError(
            f"\n步骤失败: {step_name}\n"
            f"脚本路径: {script_path}\n"
            f"返回码: {result.returncode}\n"
            f"请先看上面这个步骤的报错信息。"
        )


def main():
    project_root = find_project_root()

    print("\n" + "#" * 80)
    print("2D-ESN GPR Model Space Pipeline 一键运行脚本")
    print("#" * 80)
    print(f"[THIS SCRIPT ] {Path(__file__).resolve()}")
    print(f"[PROJECT_ROOT] {project_root}")
    print(f"[PYTHON      ] {sys.executable}")

    steps = []

    if RUN_EXPORT_WINDOW_DATASET:
        steps.append((
            "1/5 导出窗口数据集",
            project_root / "gpr_yolo_dataset" / "export_window_dataset.py"
        ))

    if RUN_INSPECT_EXPORTED_NPZ:
        steps.append((
            "2/5 检查导出的窗口数据集",
            project_root / "gpr_yolo_dataset" / "inspect_exported_npz.py"
        ))

    if RUN_FIT_2DESN_THETA:
        steps.append((
            "3/5 2D-ESN 拟合窗口并生成 theta 向量",
            project_root / "2D-ESN" / "fit_2desn_theta_vectors.py"
        ))

    if RUN_OCSVM_2DESN:
        steps.append((
            "4/5 使用 2D-ESN theta 训练并评估 OCSVM",
            project_root / "2D-ESN" / "run_ocsvm_2desn_val_threshold.py"
        ))

    if RUN_SEGMENT_EVAL:
        steps.append((
            "5/5 合并连续异常窗口并做 segment-level 评价",
            project_root / "2D-ESN" / "eval_segment_from_2desn_preds.py"
        ))

    if not steps:
        print("[WARN] 没有任何步骤被设置为运行，请检查 RUN_xxx 开关。")
        return

    print("\n本次将运行以下步骤：")
    for name, path in steps:
        print(f"  - {name}: {path}")

    total_start = time.time()

    for step_name, script_path in steps:
        run_python_script(step_name, script_path, project_root)

    total_cost = time.time() - total_start

    print("\n" + "#" * 80)
    print("[DONE] 2D-ESN 模型空间流程全部运行完成")
    print(f"[TOTAL TIME] {total_cost:.2f} 秒")
    print("#" * 80)

    print("\n输出结果一般在这里：")
    print(
        project_root
        / "gpr_yolo_dataset"
        / "window_dataset_overlap40_w128_s32"
        / "theta_vectors_2desn_n30_h32_w32"
    )


if __name__ == "__main__":
    main()