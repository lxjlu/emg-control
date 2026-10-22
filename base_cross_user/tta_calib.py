# -*- coding: utf-8 -*-
"""
tta_calib.py - 统一的无标签测试时适配 (TTA) 实验
支持特征空间来自不同窗口缓存 (150pt / 300pt) 与对应 best 模型。

方法:
  none       源 LR 直接预测                           = 0-shot 基线
  mean       用户全局均值对齐                         (TTA-mean)
  pseudo     伪标签 per-class 中心对齐 (1 轮, 高置信)  ← 新增: 逼近 few-shot 机制
  pseudo2    伪标签对齐迭代 2 轮

伪标签逻辑 (per 测试用户 u):
  1. 源 LR 给 u 的无标签特征打伪标签, 保留预测置信度高的样本 (top_conf_ratio)
  2. 估计每类伪标签中心 mu_u,c (缺类回退全局均值)
  3. 特征按各自伪标签类平移: z' = z - mu_u, yhat(z) + mu_s, yhat(z)   (mu_s,c 源类中心)
  4. (pseudo2) 再预测 → 再对齐一次
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


def per_class_centers(F, Y, n_class=4, fallback=None):
    centers = []
    for c in range(n_class):
        m = Y == c
        if m.sum() > 0:
            centers.append(F[m].mean(axis=0))
        else:
            centers.append(fallback if fallback is not None
                           else F.mean(axis=0))
    return np.stack(centers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default="./results/best_v4_allusers_nouser.pt")
    ap.add_argument("--cache_dir", type=str, default="./data_cache_v2")
    ap.add_argument("--methods", type=str, default="none,mean,pseudo,pseudo2")
    ap.add_argument("--conf_ratio", type=float, default=0.5,
                    help="伪标签保留的高置信样本比例 (0.5 = 概率最高的50%)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp_name", type=str, default="tta_calib")
    ap.add_argument("--save_dir", type=str, default="./results")
    args = ap.parse_args()

    methods = args.methods.split(",")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 70)
    print(f"TTA+Pseudo Calibration | cache={args.cache_dir}")
    print(f"methods={methods} conf_ratio={args.conf_ratio}")
    print("=" * 70)

    X_emg, X_imu, Y, U, num_users, _ = load_h5_data(cache_dir=args.cache_dir)
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

    clf_src = LogisticRegression(C=1.0, max_iter=5000, n_jobs=-1)
    clf_src.fit(F_tr, Y_tr)
    mu_s = F_tr.mean(axis=0)
    mu_s_c = per_class_centers(F_tr, Y_tr)   # 源每类中心

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
            elif method == "mean":
                F_adj = F_u - F_u.mean(axis=0) + mu_s
            elif method in ("pseudo", "pseudo2"):
                # 伪标签 (用源分类器)
                proba = clf_src.predict_proba(F_u)
                yhat = proba.argmax(axis=1)
                conf = proba.max(axis=1)
                keep_n = max(1, int(len(F_u) * args.conf_ratio))
                keep = np.argsort(conf)[-keep_n:]
                # 每类伪标签中心 (仅用高置信子集, 缺类回退全局)
                mu_u_c = per_class_centers(F_u[keep], yhat[keep],
                                           fallback=F_u.mean(axis=0))
                # 按其伪标签类平移
                F_adj = F_u - mu_u_c[yhat] + mu_s_c[yhat]
                if method == "pseudo2":
                    yhat2 = clf_src.predict(F_adj)
                    conf2 = clf_src.predict_proba(F_adj).max(axis=1)
                    keep2 = np.argsort(conf2)[-max(1, int(len(F_u) * args.conf_ratio)):]
                    mu_u_c2 = per_class_centers(F_adj[keep2], yhat2[keep2],
                                                fallback=F_adj.mean(axis=0))
                    F_adj = F_adj - mu_u_c2[yhat2] + mu_s_c[yhat2]
            else:
                raise ValueError(f"unknown method {method}")
            accs.append(accuracy_score(Y_u, clf_src.predict(F_adj)))
        results[method] = float(np.mean(accs))
        print(f"[{method:>8}] avg acc = {results[method]:.4f}")

    print("\n=== 对照 few-shot (results_fewshot.json) ===")
    fsp = os.path.join(args.save_dir, "results_fewshot.json")
    if os.path.exists(fsp):
        with open(fsp) as f:
            fk = json.load(f)
        for k in ["0shot_lr", "1shot", "5shot", "10shot", "20shot"]:
            if k in fk:
                print(f"  {k:>10}: {fk[k]:.4f}")

    with open(os.path.join(args.save_dir, f"results_{args.exp_name}.json"), "w") as f:
        json.dump({'config': vars(args), 'results': results}, f, indent=2)
    print(f"\nSaved to {args.save_dir}/results_{args.exp_name}.json")


if __name__ == "__main__":
    main()
