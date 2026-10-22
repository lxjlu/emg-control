# -*- coding: utf-8 -*-
"""
对比实验: 多任务学习 vs 域泛化
"""

import os
import sys
import subprocess
import json
import time

EXPERIMENTS = [
    {
        "name": "multi_task",
        "desc": "多任务学习 (baseline)",
        "args": "--model multi_task --use_imu --norm_strategy per_user"
    },
    {
        "name": "domain_gen_mmd01",
        "desc": "域泛化 (MMD=0.1)",
        "args": "--model domain_gen --use_imu --norm_strategy per_user --mmd_weight 0.1"
    },
    {
        "name": "domain_gen_mmd1",
        "desc": "域泛化 (MMD=1.0)",
        "args": "--model domain_gen --use_imu --norm_strategy per_user --mmd_weight 1.0"
    },
    {
        "name": "domain_gen_mmd10",
        "desc": "域泛化 (MMD=10.0)",
        "args": "--model domain_gen --use_imu --norm_strategy per_user --mmd_weight 10.0"
    },
]

PYTHON = sys.executable

BASE_CMD = (
    f"{PYTHON} train_v2.py {{args}} "
    "--epochs 100 --batch 256 --lr 1e-3 --test_ratio 0.2 --seed 42 "
    "--num_workers 4 --patience 15 --user_weight 0.3 "
    "--exp_name {name}"
)


def is_done(name):
    """检查实验是否已完成"""
    result_file = f"./results/results_{name}.json"
    if not os.path.exists(result_file):
        return False
    try:
        with open(result_file) as f:
            data = json.load(f)
        return "zeroshot_acc" in data
    except:
        return False


def run_exp(exp):
    """运行单个实验"""
    name = exp["name"]
    desc = exp["desc"]
    args = exp["args"]

    if is_done(name):
        print(f"[SKIP] {name}: {desc} (已完成)")
        return

    cmd = BASE_CMD.format(args=args, name=name)
    print(f"\n{'='*60}")
    print(f"[RUN] {name}: {desc}")
    print(f"CMD: {cmd}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, shell=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"[ERROR] {name} failed!")
    else:
        print(f"[DONE] {name} completed!")


def main():
    print("=" * 60)
    print("域泛化 vs 多任务学习 对比实验")
    print("=" * 60)

    for exp in EXPERIMENTS:
        run_exp(exp)

    print("\n" + "=" * 60)
    print("所有实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()