# -*- coding: utf-8 -*-
"""
tta_analysis.py - Test-Time Adaptation (无标签) 特征校准实验

动机 (来自 few-shot 实验):
  新用户相对"平均用户"的偏差 Δ_u 是低秩/可校准的 — 20-shot 线性适配即到 76%。
  TTA 目标: 不用任何标签, 仅用测试用户的【无标签】数据估计该用户的特征统计,
  把该用户特征分布"拉回"训练源分布, 让源分类器直接可用。

方法对比 (均在固定编码器特征空间, 逐测试用户独立校准):
  [none]    源 LR 直接预测                                      = 0-shot 基线
  [mean]    z' = z - μ_u + μ_s       (用户均值对齐 → 处理 mean shift)
  [meanstd] z' = (z-μ_u)/σ_u * σ_s + μ_s  (对齐 mean + per-dim scale)
  [gauss]   更完整的 whitening/recoloring (无)  — 本轮只做对角近似

合法性: per_user 归一化已在用测试用户无标签数据, TTA 同样只用无标签数据 → 协议内。
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fewshot_analysis as fs
from models_v4 import build_model as build_model_v4
from data_loader_v2 import load_h5_data
from diagnose_internal_louo import replicate_external_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default="./results/best_v4_allusers_nouser.pt")
    ap.add_argument("--methods", type=str, default="none,mean,meanstd")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp_name", type=str, default="tta")
    ap.add_argument("--save_dir", type=str, default="./results")
    args = ap.parse_args()

    methods = args.methods.split(",")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 70)
    print(f"TTA Calibration Analysis | Device: {device} | methods={methods}")
    print("=" * 70)

    # 数据 + split + 特征 (复用 fewshot_analysis)
    X_emg, X_imu, Y, U, num_users, _ = load_h5_data()
    train_users, test_users = replicate_external_split(U, seed=args.seed, test_ratio=0.2)

    model = build_model_v4(
        "unified", num_classes=4, num_users=num_users,
        emg_channels=80, imu_channels=6, use_imu=True,
        use_whitening=False, use_contra=True,
        use_grl=False, use_mmd=False, feat_dim=128, dropout=0.0).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False))
    model.eval()

    print("[Extract] train users...")
    F_tr, Y_tr, _ = fs.extract_users(model, X_emg, X_imu, Y, U, train_users, device)
    print(f"  train: {F_tr.shape}")
    print("[Extract] test users...")
    F_te, Y_te, U_te = fs.extract_users(model, X_emg, X_imu, Y, U, test_users, device)
    print(f"  test:  {F_te.shape}")

    # 源分类器 (raw 特征)
    clf_src = LogisticRegression(C=1.0, max_iter=5000, n_jobs=-1)
    clf_src.fit(F_tr, Y_tr)

    # 源统计
    mu_s = F_tr.mean(axis=0)
    sigma_s = F_tr.std(axis=0) + 1e-6

    results = {}
    user_ids = np.unique(U_te)
    for method in methods:
        accs = []
        for u in user_ids:
            mask = U_te == u
            F_u = F_te[mask]
            Y_u = Y_te[mask]
            if method == "none":
                F_adj = F_u
            else:
                mu_u = F_u.mean(axis=0)
                if method == "mean":
                    F_adj = F_u - mu_u + mu_s
                elif method == "meanstd":
                    sigma_u = F_u.std(axis=0) + 1e-6
                    F_adj = (F_u - mu_u) / sigma_u * sigma_s + mu_s
                else:
                    raise ValueError(f"unknown method {method}")
            accs.append(accuracy_score(Y_u, clf_src.predict(F_adj)))
        results[method] = float(np.mean(accs))
        print(f"[TTA {method:>8}] avg acc = {results[method]:.4f}")

    # 与 few-shot 曲线对照
    print("\n=== 对照: few-shot (来自 results_fewshot.json) ===")
    fewshot_path = os.path.join(args.save_dir, "results_fewshot.json")
    if os.path.exists(fewshot_path):
        with open(fewshot_path) as f:
            fk = json.load(f)
        for k in ["0shot_lr", "1shot", "5shot", "10shot", "20shot"]:
            if k in fk:
                print(f"  {k:>10} : {fk[k]:.4f}")

    # 汇总表
    print("\n=== TTA vs few-shot 汇总 ===")
    print(f"  {'TTA-none  (0-shot 基线)':>24} : {results.get('none', float('nan')):.4f}")
    for m in methods:
        if m != "none":
            print(f"  {'TTA-'+m:>24} : {results[m]:.4f}")

    with open(os.path.join(args.save_dir, f"results_{args.exp_name}.json"), "w") as f:
        json.dump({'config': vars(args), 'methods': results}, f, indent=2)
    print(f"\nSaved to {args.save_dir}/results_{args.exp_name}.json")


if __name__ == "__main__":
    main()
