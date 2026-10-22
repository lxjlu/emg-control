#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""归一化策略对比实验"""
import os, json, subprocess, sys

EXPERIMENTS = [
    {"name": "none", "desc": "无归一化 + 多任务学习", "args": "--norm_strategy none"},
    {"name": "per_user", "desc": "逐用户Z-Score + 多任务学习", "args": "--norm_strategy per_user"},
    {"name": "global", "desc": "全局Z-Score + 多任务学习", "args": "--norm_strategy global"},
]

BASE_CMD = (
    "python train_v2.py --model multi_task --use_imu {args} "
    "--epochs 100 --batch 256 --lr 1e-3 --test_ratio 0.2 --seed 42 "
    "--num_workers 4 --patience 15 --user_weight 0.3"
)

def check_done(name):
    results_dir = "results"
    if not os.path.exists(results_dir):
        return False
    for f in os.listdir(results_dir):
        if f.startswith("results_multi_task_") and f.endswith(".json"):
            with open(os.path.join(results_dir, f), "r") as fp:
                r = json.load(fp)
            if r.get("norm_strategy") == name:
                return True
    return False

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    done = [e["name"] for e in EXPERIMENTS if check_done(e["name"])]
    todo = [e for e in EXPERIMENTS if not check_done(e["name"])]
    
    print("=" * 60)
    print("归一化策略对比实验")
    print("=" * 60)
    print(f"\n已完成的实验: {', '.join(done) if done else '无'}")
    print(f"待运行的实验: {', '.join(e['name'] for e in todo) if todo else '无'}\n")
    
    for exp in todo:
        cmd = BASE_CMD.format(args=exp["args"])
        print("=" * 60)
        print(f"Experiment: {exp['name']}")
        print(f"Description: {exp['desc']}")
        print(f"Command: {cmd}")
        print("=" * 60)
        
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"\n⚠️  实验 {exp['name']} 运行失败，跳过\n")
        else:
            print(f"\n✅ 实验 {exp['name']} 运行完成\n")
    
    print("=" * 60)
    print("所有实验运行完成！")
    print("运行 python check_all.py 查看结果")

if __name__ == "__main__":
    main()