# -*- coding: utf-8 -*-
"""
完整结果分析脚本 v2
读取所有实验结果，生成对比图表和详细分析报告
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "./results"

# 实验配置
EXPERIMENTS = {
    "cnn1d": {"name": "CNN-1D", "desc": "CNN-1D Baseline (8通道)", "color": "#1f77b4"},
    "transformer": {"name": "Transformer", "desc": "Transformer Baseline (8通道)", "color": "#ff7f0e"},
    "dann": {"name": "DANN", "desc": "DANN 域对抗 (8通道)", "color": "#2ca02c"},
    "dann_imu": {"name": "DANN+IMU", "desc": "DANN + EMG+IMU多模态", "color": "#d62728"},
    "dann_contra": {"name": "DANN+Contra", "desc": "DANN + 跨用户对比学习", "color": "#9467bd"},
    "dann_imu_contra": {"name": "DANN+IMU+Contra", "desc": "DANN + IMU + 对比学习 (完整)", "color": "#8c564b"},
    "disentangled": {"name": "Disentangled", "desc": "特征解耦 (8通道)", "color": "#e377c2"},
    "disentangled_imu": {"name": "Disentangled+IMU", "desc": "特征解耦 + IMU多模态", "color": "#7f7f7f"},
    "disentangled_contra": {"name": "Disentangled+Contra", "desc": "特征解耦 + 对比学习", "color": "#bcbd22"},
    "disentangled_imu_contra": {"name": "Disentangled+Full", "desc": "特征解耦 + IMU + 对比 (完整)", "color": "#17becf"},
    "imu_attn": {"name": "IMU-Attn", "desc": "IMU引导注意力", "color": "#ff9896"},
    "imu_attn_dann": {"name": "IMU-Attn+DANN", "desc": "IMU引导注意力 + DANN", "color": "#9edae5"},
    "imu_decompose": {"name": "IMU-Decomp", "desc": "IMU锚定特征分解", "color": "#c5b0d5"},
    "imu_decompose_dann": {"name": "IMU-Decomp+DANN", "desc": "IMU锚定特征分解 + DANN", "color": "#c49c94"},
    "imu_crossmodal": {"name": "IMU-CrossModal", "desc": "跨模态对比学习", "color": "#f7b6d2"},
    "imu_crossmodal_dann": {"name": "IMU-CrossModal+DANN", "desc": "跨模态对比 + DANN", "color": "#dbdb8d"},
}


def load_all_results():
    """加载所有实验结果"""
    results = {}
    for key, cfg in EXPERIMENTS.items():
        fname = f"results_{key}.json"
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                results[key] = json.load(f)
            print(f"[OK] {cfg['name']}: zeroshot={results[key].get('zeroshot_acc', 0):.4f}")
        else:
            print(f"[MISSING] {cfg['name']}")
    return results


def plot_training_curves(results):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for key, data in results.items():
        history = data.get("history", [])
        if not history:
            continue
        cfg = EXPERIMENTS.get(key, {})
        color = cfg.get("color", "gray")
        label = cfg.get("name", key)

        epochs = [h["epoch"] for h in history]
        losses = [h["loss"] for h in history]
        train_accs = [h["train_acc"] for h in history]
        test_accs = [h["zeroshot_acc"] for h in history]

        axes[0].plot(epochs, losses, "o-", label=label, color=color, linewidth=2, markersize=4)
        axes[1].plot(epochs, train_accs, "o-", label=label, color=color, linewidth=2, markersize=4)
        axes[2].plot(epochs, test_accs, "s--", label=label, color=color, linewidth=2, markersize=4)

    # Loss曲线
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training Loss", fontsize=14)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Train Acc曲线
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Training Accuracy", fontsize=14)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Zero-Shot Acc曲线
    axes[2].set_xlabel("Epoch", fontsize=12)
    axes[2].set_ylabel("Zero-Shot Accuracy", fontsize=12)
    axes[2].set_title("Zero-Shot Accuracy (Test)", fontsize=14)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "training_curves_v2.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[Saved] {save_path}")
    plt.close()


def plot_comparison_bar(results):
    """绘制对比柱状图"""
    names = []
    zeroshot_accs = []
    train_accs = []
    gaps = []
    colors = []

    baseline = None
    for key, data in results.items():
        cfg = EXPERIMENTS.get(key, {})
        names.append(cfg.get("name", key))
        zs = data.get("zeroshot_acc", 0)
        ta = data["history"][-1]["train_acc"] if data.get("history") else 0
        zeroshot_accs.append(zs)
        train_accs.append(ta)
        gaps.append(ta - zs)
        colors.append(cfg.get("color", "gray"))

        if key == "cnn1d":
            baseline = zs

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Zero-Shot准确率对比
    bars = axes[0].bar(range(len(names)), zeroshot_accs, color=colors, width=0.6)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    axes[0].set_ylabel("Zero-Shot Accuracy", fontsize=12)
    axes[0].set_title("Zero-Shot Accuracy Comparison", fontsize=14)
    axes[0].grid(True, alpha=0.3, axis="y")

    # 添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, zeroshot_accs)):
        diff = acc - baseline if baseline else 0
        sign = "+" if diff >= 0 else ""
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{acc:.3f}\n({sign}{diff:.3f})",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 2. 训练准确率 vs Zero-Shot
    x = np.arange(len(names))
    width = 0.35
    axes[1].bar(x - width/2, train_accs, width, label="Train Acc", color="#3498db", alpha=0.8)
    axes[1].bar(x + width/2, zeroshot_accs, width, label="Zero-Shot", color="#e74c3c", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Train vs Zero-Shot Accuracy", fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis="y")

    # 3. 过拟合程度 (Train - Zero-Shot)
    gap_colors = ["#2ecc71" if g < 0.2 else "#f39c12" if g < 0.35 else "#e74c3c" for g in gaps]
    axes[2].bar(range(len(names)), gaps, color=gap_colors, width=0.6)
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    axes[2].set_ylabel("Gap (Train - Zero-Shot)", fontsize=12)
    axes[2].set_title("Overfitting Gap (lower is better)", fontsize=14)
    axes[2].grid(True, alpha=0.3, axis="y")
    axes[2].axhline(y=0.2, color="green", linestyle="--", alpha=0.5, label="Good (<0.2)")
    axes[2].axhline(y=0.35, color="orange", linestyle="--", alpha=0.5, label="Moderate (<0.35)")
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "comparison_bar_v2.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Saved] {save_path}")
    plt.close()


def print_analysis(results):
    """打印详细分析报告"""
    print("\n" + "=" * 80)
    print("详 细 分 析 报 告")
    print("=" * 80)

    # 1. 结果汇总表
    print("\n【1. 结果汇总表】")
    print(f"{'模型':<20} {'Zero-Shot':<12} {'Train Acc':<12} {'Gap':<10} {'Epochs':<8}")
    print("-" * 80)

    baseline_zs = results.get("cnn1d", {}).get("zeroshot_acc", 0)
    sorted_results = sorted(results.items(), key=lambda x: x[1].get("zeroshot_acc", 0), reverse=True)

    for key, data in sorted_results:
        cfg = EXPERIMENTS.get(key, {})
        zs = data.get("zeroshot_acc", 0)
        ta = data["history"][-1]["train_acc"] if data.get("history") else 0
        gap = ta - zs
        ep = data.get("early_stop_epoch", 0)
        diff = zs - baseline_zs
        sign = "+" if diff >= 0 else ""
        print(f"{cfg.get('name', key):<20} {zs:.4f}       {ta:.4f}      {gap:.4f}    {ep:<8} ({sign}{diff:.4f})")

    # 2. 各策略效果分析
    print("\n【2. 各策略效果分析】")

    # DANN vs CNN
    dann_zs = results.get("dann", {}).get("zeroshot_acc", 0)
    cnn_zs = results.get("cnn1d", {}).get("zeroshot_acc", 0)
    print(f"  DANN域对抗 vs CNN-1D: {dann_zs - cnn_zs:+.4f}")
    if dann_zs > cnn_zs:
        print("    -> DANN有效提升了zero-shot准确率，说明域对抗训练帮助学习了用户无关特征")
    else:
        print("    -> DANN在8通道EMG上效果不明显，可能需要更多通道或IMU配合")

    # IMU效果
    dann_imu_zs = results.get("dann_imu", {}).get("zeroshot_acc", 0)
    print(f"  DANN+IMU vs DANN: {dann_imu_zs - dann_zs:+.4f}")
    if dann_imu_zs > dann_zs:
        print("    -> IMU多模态融合有效，运动学信息确实更用户无关")
    else:
        print("    -> IMU多模态未带来提升，可能原因:")
        print("       - IMU信号质量不佳或噪声较大")
        print("       - 融合模块权重初始化不当")
        print("       - 训练轮数不足，融合模块未充分学习")

    # 对比学习效果
    contra_zs = results.get("dann_contra", {}).get("zeroshot_acc", 0)
    print(f"  DANN+对比 vs DANN: {contra_zs - dann_zs:+.4f}")
    if contra_zs > dann_zs:
        print("    -> 对比学习有效，同手势不同用户的特征确实被拉近了")
    else:
        print("    -> 对比学习未带来明显提升")

    # 完整方案
    full_zs = results.get("dann_imu_contra", {}).get("zeroshot_acc", 0)
    print(f"  完整方案 vs CNN-1D: {full_zs - cnn_zs:+.4f}")
    if full_zs > cnn_zs:
        print("    -> 完整方案优于基线，多策略协同有效")
    else:
        print("    -> 完整方案反而低于基线，存在严重问题:")
        print("       - 训练准确率92%，zero-shot仅41%，严重过拟合")
        print("       - 多种策略叠加可能导致优化目标冲突")
        print("       - 建议: 降低对比学习权重，增加正则化")

    # 3. 过拟合分析
    print("\n【3. 过拟合分析】")
    print(f"  {'模型':<20} {'Train-ZS Gap':<15} {'过拟合程度':<10}")
    print("  " + "-" * 50)
    for key, data in sorted_results:
        cfg = EXPERIMENTS.get(key, {})
        zs = data.get("zeroshot_acc", 0)
        ta = data["history"][-1]["train_acc"] if data.get("history") else 0
        gap = ta - zs
        level = "轻微" if gap < 0.2 else "中等" if gap < 0.35 else "严重"
        print(f"  {cfg.get('name', key):<20} {gap:.4f}          {level}")

    # 4. 结论与建议
    print("\n【4. 结论与建议】")
    best_key = sorted_results[0][0]
    best_zs = sorted_results[0][1].get("zeroshot_acc", 0)
    best_cfg = EXPERIMENTS.get(best_key, {})
    print(f"  最佳模型: {best_cfg.get('name', best_key)} (Zero-Shot: {best_zs:.4f})")
    print(f"  相比CNN-1D基线: {best_zs - cnn_zs:+.4f}")

    print("\n  改进建议:")
    if full_zs < cnn_zs:
        print("  1. 完整方案过拟合严重，建议:")
        print("     - 降低对比学习权重 (0.5 -> 0.1-0.2)")
        print("     - 增加Dropout (0.3 -> 0.5)")
        print("     - 使用更强的数据增广")
        print("     - 减小模型参数量")
    if dann_imu_zs < cnn_zs:
        print("  2. DANN+IMU效果不佳，建议:")
        print("     - 检查IMU数据质量")
        print("     - 调整融合策略 (concat -> attention)")
        print("     - 增加IMU编码器容量")
    print("  3. 整体建议:")
    print("     - 尝试更大的batch size (512)")
    print("     - 使用学习率warmup")
    print("     - 增加用户无关的归一化策略")
    print("     - 考虑使用预训练模型")

    print("\n" + "=" * 80)


def main():
    print("加载实验结果...")
    results = load_all_results()

    if not results:
        print("没有找到任何实验结果!")
        return

    print(f"\n共加载 {len(results)} 个实验结果")

    print("\n生成训练曲线图...")
    plot_training_curves(results)

    print("生成对比柱状图...")
    plot_comparison_bar(results)

    print_analysis(results)


if __name__ == "__main__":
    main()