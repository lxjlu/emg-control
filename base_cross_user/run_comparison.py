# -*- coding: utf-8 -*-
"""
对比实验脚本: 运行 CNN-1D 和 Transformer 的 zero-shot 对比
"""

import os
import sys
import json
import subprocess
import time


EXPERIMENTS = [
    {"name": "cnn1d",       "desc": "CNN-1D Baseline",       "model": "cnn1d"},
    {"name": "transformer", "desc": "Transformer Baseline",  "model": "transformer"},
]

COMMON_ARGS = [
    "--epochs", "50",
    "--batch", "256",
    "--lr", "1e-3",
    "--test_ratio", "0.2",
    "--seed", "42",
    "--num_workers", "4",
]


def run_experiment(exp, dry_run=False):
    name = exp["name"]
    model = exp["model"]
    cmd = ["python", "train.py", "--model", model] + COMMON_ARGS

    print("\n" + "=" * 70)
    print(f"Experiment: {name} ({exp['desc']})")
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
        if os.path.exists(fpath):
            with open(fpath) as f:
                data = json.load(f)
            summary.append({
                "name": name,
                "desc": exp["desc"],
                "zeroshot_acc": data.get("zeroshot_acc", 0),
                "n_params": sum(1 for _ in data.get("history", [])),
            })
    return summary


def print_table(summary):
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Model':<20} {'Zero-Shot Acc':>15}")
    print("-" * 60)
    for item in summary:
        print(f"{item['name']:<20} {item['zeroshot_acc']:>15.4f}")
    print("=" * 60)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--exp", type=str, default=None)
    args = ap.parse_args()

    if args.collect_only:
        summary = collect_results()
        print_table(summary)
        return

    results = []
    for exp in EXPERIMENTS:
        if args.exp and exp["name"] != args.exp:
            continue
        r = run_experiment(exp, dry_run=args.dry_run)
        if r:
            results.append(r)

    if not args.dry_run:
        print("\nCollecting results...")
        summary = collect_results()
        print_table(summary)


if __name__ == "__main__":
    main()