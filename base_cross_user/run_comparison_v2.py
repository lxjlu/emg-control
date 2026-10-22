# -*- coding: utf-8 -*-
"""
对比实验脚本 v2: 运行所有改进策略的对比
"""

import os
import json
import subprocess
import time


EXPERIMENTS = [
    {"name": "cnn1d",         "desc": "CNN-1D Baseline (8通道)",           "args": ["--model", "cnn1d"]},
    {"name": "transformer",   "desc": "Transformer Baseline (8通道)",      "args": ["--model", "transformer"]},
    {"name": "dann",          "desc": "DANN 域对抗 (8通道)",               "args": ["--model", "dann"]},
    {"name": "dann_imu",      "desc": "DANN + EMG+IMU多模态",              "args": ["--model", "dann", "--use_imu"]},
    {"name": "dann_contra",   "desc": "DANN + 跨用户对比学习",             "args": ["--model", "dann", "--use_contrastive"]},
    {"name": "dann_full",     "desc": "DANN + IMU + 对比学习 (完整)",      "args": ["--model", "dann", "--use_imu", "--use_contrastive"]},
    {"name": "disentangled",  "desc": "特征解耦 (8通道)",                  "args": ["--model", "disentangled"]},
    {"name": "disentangled_imu", "desc": "特征解耦 + IMU多模态",           "args": ["--model", "disentangled", "--use_imu"]},
    {"name": "disentangled_contra", "desc": "特征解耦 + 对比学习",         "args": ["--model", "disentangled", "--use_contrastive"]},
    {"name": "disentangled_full", "desc": "特征解耦 + IMU + 对比 (完整)",  "args": ["--model", "disentangled", "--use_imu", "--use_contrastive"]},
    # IMU锚定方案
    {"name": "imu_attn",      "desc": "IMU引导注意力",                     "args": ["--model", "imu_attn", "--use_imu"]},
    {"name": "imu_attn_dann", "desc": "IMU引导注意力 + DANN",              "args": ["--model", "imu_attn", "--use_imu", "--use_contrastive"]},
    {"name": "imu_decompose", "desc": "IMU锚定特征分解",                   "args": ["--model", "imu_decompose", "--use_imu"]},
    {"name": "imu_decompose_dann", "desc": "IMU锚定特征分解 + DANN",       "args": ["--model", "imu_decompose", "--use_imu", "--use_contrastive"]},
    {"name": "imu_crossmodal","desc": "跨模态对比学习",                    "args": ["--model", "imu_crossmodal", "--use_imu"]},
    {"name": "imu_crossmodal_dann", "desc": "跨模态对比 + DANN",           "args": ["--model", "imu_crossmodal", "--use_imu", "--use_contrastive"]},
]

COMMON_ARGS = [
    "--epochs", "100",
    "--batch", "256",
    "--lr", "1e-3",
    "--test_ratio", "0.2",
    "--seed", "42",
    "--num_workers", "4",
    "--patience", "15",
]


def run_experiment(exp, dry_run=False):
    name = exp["name"]
    cmd = ["python", "train_v2.py"] + COMMON_ARGS + exp["args"]

    print("\n" + "=" * 70)
    print(f"Experiment: {name}")
    print(f"Description: {exp['desc']}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70)

    if dry_run:
        print("[DRY RUN] Skipping")
        return None

    start = time.time()
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)),
                                capture_output=False, text=True)
        elapsed = time.time() - start
        print(f"\nDone in {elapsed:.1f}s (exit={result.returncode})")
        return {"name": name, "status": "success", "time": elapsed}
    except Exception as e:
        elapsed = time.time() - start
        print(f"\nFailed after {elapsed:.1f}s: {e}")
        return {"name": name, "status": "failed", "time": elapsed, "error": str(e)}


def collect_results():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    summary = []
    for exp in EXPERIMENTS:
        name = exp["name"]
        fpath = os.path.join(results_dir, f"results_{name}.json")
        if not os.path.exists(fpath):
            fpath = os.path.join(results_dir, f"results_{name}_imu.json")
        if not os.path.exists(fpath):
            fpath = os.path.join(results_dir, f"results_{name}_contra.json")
        if not os.path.exists(fpath):
            fpath = os.path.join(results_dir, f"results_{name}_imu_contra.json")

        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            history = data.get("history", [])
            best_epoch = max(history, key=lambda h: h["zeroshot_acc"]) if history else {}
            summary.append({
                "name": name,
                "desc": exp["desc"],
                "zeroshot_acc": data.get("zeroshot_acc", 0),
                "train_acc": best_epoch.get("train_acc", 0),
                "best_epoch": best_epoch.get("epoch", 0),
                "early_stopped": data.get("early_stop_epoch", 0) < 100,
            })
    return summary


def print_table(summary):
    print("\n" + "=" * 90)
    print("COMPARISON RESULTS")
    print("=" * 90)
    print(f"{'Model':<20} {'Zero-Shot':>12} {'Train Acc':>12} {'Best Ep':>10} {'EarlyStop':>10}")
    print("-" * 90)

    baseline = None
    for item in summary:
        if item["name"] == "cnn1d":
            baseline = item["zeroshot_acc"]
            break

    for item in summary:
        delta = ""
        if baseline is not None and item["name"] != "cnn1d":
            d = item["zeroshot_acc"] - baseline
            delta = f" ({d:+.4f})"
        print(f"{item['name']:<20} {item['zeroshot_acc']:>12.4f}{delta:<12} "
              f"{item['train_acc']:>12.4f} {item['best_epoch']:>10d} "
              f"{'Yes' if item['early_stopped'] else 'No':>10}")

    print("=" * 90)


def is_experiment_done(exp):
    """检查实验是否已完成"""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    name = exp["name"]
    args = exp["args"]
    
    # 根据参数确定文件名后缀
    suffix = name
    if "--use_imu" in args and "--use_contrastive" in args:
        suffix += "_imu_contra"
    elif "--use_imu" in args:
        suffix += "_imu"
    elif "--use_contrastive" in args:
        suffix += "_contra"
    
    fpath = os.path.join(results_dir, f"results_{suffix}.json")
    return os.path.exists(fpath)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--exp", type=str, default=None)
    ap.add_argument("--force", action="store_true", help="强制重新运行所有实验")
    args = ap.parse_args()

    if args.collect_only:
        summary = collect_results()
        print_table(summary)
        return

    results = []
    skipped = 0
    for exp in EXPERIMENTS:
        if args.exp and exp["name"] != args.exp:
            continue
        
        if not args.force and is_experiment_done(exp):
            print(f"\n[SKIP] {exp['name']}: {exp['desc']} (already done)")
            skipped += 1
            continue
            
        r = run_experiment(exp, dry_run=args.dry_run)
        if r:
            results.append(r)

    if skipped > 0:
        print(f"\n[Info] Skipped {skipped} completed experiments")

    if not args.dry_run:
        print("\nCollecting results...")
        summary = collect_results()
        print_table(summary)


if __name__ == "__main__":
    main()