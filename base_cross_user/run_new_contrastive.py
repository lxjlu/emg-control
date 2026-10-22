#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行四种新型对比学习实验
1. 原型对比学习
2. 记忆库对比学习
3. 用户归一化 + 对比学习
4. 双分支对比学习
"""

import os
import subprocess
import sys

EXPERIMENTS = [
    {
        "name": "proto_contra",
        "desc": "原型对比学习",
        "args": ["--model", "proto_contra", "--use_imu", "--contrastive_weight", "0.5"]
    },
    {
        "name": "memory_contra",
        "desc": "记忆库对比学习",
        "args": ["--model", "memory_contra", "--use_imu", "--contrastive_weight", "0.5"]
    },
    {
        "name": "user_norm_contra",
        "desc": "用户归一化 + 对比学习",
        "args": ["--model", "user_norm_contra", "--use_imu", "--contrastive_weight", "0.5"]
    },
    {
        "name": "dual_branch_contra",
        "desc": "双分支对比学习",
        "args": ["--model", "dual_branch_contra", "--use_imu", "--contrastive_weight", "0.5"]
    },
]

BASE_ARGS = [
    "--epochs", "100",
    "--batch", "256",
    "--lr", "1e-3",
    "--test_ratio", "0.2",
    "--seed", "42",
    "--num_workers", "4",
    "--patience", "15",
]


def is_experiment_done(exp):
    """检查实验是否已完成"""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    fpath = os.path.join(results_dir, f"results_{exp['name']}_imu.json")
    return os.path.exists(fpath)


def run_experiment(exp):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"Experiment: {exp['name']}")
    print(f"Description: {exp['desc']}")
    print(f"Command: python train_v2.py {' '.join(exp['args'])} {' '.join(map(str, BASE_ARGS))}")
    print(f"{'='*60}")

    cmd = [sys.executable, "train_v2.py"] + exp["args"] + [str(a) for a in BASE_ARGS]
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0


def main():
    print("四种新型对比学习实验")
    print("="*60)

    # 检查已完成的实验
    done = [exp for exp in EXPERIMENTS if is_experiment_done(exp)]
    todo = [exp for exp in EXPERIMENTS if not is_experiment_done(exp)]

    if done:
        print(f"\n已完成的实验: {', '.join([e['name'] for e in done])}")
    if todo:
        print(f"待运行的实验: {', '.join([e['name'] for e in todo])}")
    else:
        print("\n所有实验已完成！")
        return

    # 运行待完成的实验
    for exp in todo:
        success = run_experiment(exp)
        if not success:
            print(f"\n⚠️  实验 {exp['name']} 运行失败，跳过")

    print("\n" + "="*60)
    print("所有实验运行完成！")
    print("运行 python check_all.py 查看结果")


if __name__ == "__main__":
    main()