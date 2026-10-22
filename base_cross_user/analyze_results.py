# -*- coding: utf-8 -*-
"""
结果分析脚本: 读取训练结果，生成对比图表
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(results_dir="./results"):
    results = {}
    for fname in os.listdir(results_dir):
        if fname.startswith("results_") and fname.endswith(".json"):
            model_name = fname.replace("results_", "").replace(".json", "")
            with open(os.path.join(results_dir, fname)) as f:
                results[model_name] = json.load(f)
    return results


def plot_training_curves(results, save_path="./results/training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"cnn1d": "#1f77b4", "transformer": "#ff7f0e"}

    for name, data in results.items():
        history = data.get("history", [])
        if not history:
            continue
        epochs = [h["epoch"] for h in history]
        losses = [h["loss"] for h in history]
        train_accs = [h["train_acc"] for h in history]
        test_accs = [h["zeroshot_acc"] for h in history]

        color = colors.get(name, None)

        axes[0].plot(epochs, losses, "o-", label=name, color=color)
        axes[1].plot(epochs, train_accs, "o-", label=f"{name} (train)", color=color)
        axes[1].plot(epochs, test_accs, "s--", label=f"{name} (zeroshot)", color=color)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_comparison_bar(results, save_path="./results/comparison_bar.png"):
    models = []
    accs = []
    for name, data in results.items():
        models.append(name)
        accs.append(data.get("zeroshot_acc", 0))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bars = ax.bar(models, accs, color=colors[:len(models)], edgecolor="black", alpha=0.8)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_ylabel("Zero-Shot Accuracy")
    ax.set_title("Cross-User Zero-Shot Comparison")
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.25, color="red", linestyle="--", alpha=0.5, label="Random (25%)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison bar chart saved to {save_path}")


def print_summary(results):
    print("\n" + "=" * 70)
    print("RESULT SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Zero-Shot':>12} {'Train Acc':>12} {'Best Epoch':>12}")
    print("-" * 70)

    for name, data in results.items():
        history = data.get("history", [])
        if history:
            best = max(history, key=lambda h: h["zeroshot_acc"])
            print(f"{name:<20} {best['zeroshot_acc']:>12.4f} "
                  f"{best['train_acc']:>12.4f} {best['epoch']:>12d}")
        else:
            print(f"{name:<20} {'N/A':>12}")

    print("=" * 70)


def main():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    results = load_results(results_dir)

    if not results:
        print("No results found. Run experiments first.")
        return

    print_summary(results)
    plot_training_curves(results)
    plot_comparison_bar(results)


if __name__ == "__main__":
    main()