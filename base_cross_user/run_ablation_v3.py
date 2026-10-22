# -*- coding: utf-8 -*-
"""
run_ablation_v3.py - 5 组件消融实验
测试每个组件对 zero-shot 准确率的贡献
"""
import os
import sys
import subprocess
import json

PYTHON = sys.executable

EXPERIMENTS = [
    # baseline: 无额外组件 (multi_task + per_user)
    {"name": "v3_baseline", "desc": "baseline (无组件)", "args": ""},
    # 单组件
    {"name": "v3_whitening", "desc": "+ Feature Whitening", "args": "--use_whitening"},
    {"name": "v3_arcface", "desc": "+ ArcFace", "args": "--use_arcface"},
    {"name": "v3_slim", "desc": "+ Slim (瘦身+正则)", "args": "--use_slim"},
    {"name": "v3_contra", "desc": "+ 修复对比学习", "args": "--use_contra"},
    {"name": "v3_v02", "desc": "+ v02 组件", "args": "--use_v02"},
    # 全部组合
    {"name": "v3_all", "desc": "全部组件", "args": "--use_whitening --use_arcface --use_slim --use_contra --use_v02"},
]

BASE_CMD = (
    f"{PYTHON} train_v3.py --model unified --use_imu --norm_strategy per_user "
    "--epochs 100 --batch 256 --lr 1e-3 --test_ratio 0.2 --seed 42 "
    "--num_workers 4 --patience 15 --user_weight 0.3 "
    "--exp_name {name} {args}"
)


def is_done(name):
    result_file = f"./results/results_{name}.json"
    if not os.path.exists(result_file):
        return False
    try:
        with open(result_file) as f:
            data = json.load(f)
        return "zeroshot_acc" in data
    except Exception:
        return False


def run_exp(exp):
    name = exp["name"]
    desc = exp["desc"]
    args = exp["args"]

    if is_done(name):
        print(f"[SKIP] {name}: {desc} (已完成)")
        return

    cmd = BASE_CMD.format(name=name, args=args)
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
    print("跨用户手势识别 - 5 组件消融实验")
    print("=" * 60)

    for exp in EXPERIMENTS:
        run_exp(exp)

    print("\n" + "=" * 60)
    print("消融实验完成！结果汇总:")
    print("=" * 60)
    print(f"{'实验':20s} | {'zero-shot':>10s} | {'描述'}")
    print("-" * 60)
    for exp in EXPERIMENTS:
        name = exp["name"]
        result_file = f"./results/results_{name}.json"
        if os.path.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
            acc = data.get('zeroshot_acc', 0)
            print(f"{name:20s} | {acc:10.4f} | {exp['desc']}")
        else:
            print(f"{name:20s} | {'N/A':>10s} | {exp['desc']}")


if __name__ == "__main__":
    main()
