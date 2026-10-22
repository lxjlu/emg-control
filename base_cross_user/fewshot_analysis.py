# -*- coding: utf-8 -*-
"""
fewshot_analysis.py - 测试用户 Few-Shot 适配分析

目的: 判断当前 59% 的 zero-shot 瓶颈中, "用户特异性"有多大比例是
     可快速适配的 (低维偏移 / 佩戴配置 / 风格), 多大比例是真正无法学习的。

方法:
  1. 加载 best 模型 (v4 all-users pretrain + nouser, 59.31%), 冻结编码器
  2. 提取特征: 140 训练用户 + 35 外部测试用户 (per_user 归一化, 同主协议)
  3. 分类器: 特征空间 Logistic Regression
     - 0-shot : 只用 140 训练用户特征拟合 → 预测 35 个测试用户
     - k-shot : 每个测试用户 u, 给 k 条/类 标注样本, 连同训练特征一起拟合
                → 评测 u 的剩余样本  (k = 1, 5, 10, 20)
  4. 曲线解读:
     - 若 k=1 就跳升 ~15pp+ (如 59 → 75+) → 用户特异性主要是"可快速对齐的低维偏移"
     - 若 k 增大仍缓慢爬升 → 说明新用户真正落入特征空间之外, 适配也难
  另外打印主分类头 0-shot 作为 sanity check (应复现 ~59.31%)。
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_v4 import build_model as build_model_v4
from data_loader_v2 import load_h5_data
from diagnose_internal_louo import replicate_external_split


def featurize(model, Xe, Xi, device, bs=1024):
    """批量前向, 提取 projection 特征"""
    model.eval()
    feats = []
    for i in range(0, len(Xe), bs):
        xe = torch.from_numpy(Xe[i:i + bs]).to(device)
        xi = torch.from_numpy(Xi[i:i + bs]).to(device)
        with torch.no_grad():
            ef = model.emg_encoder(xe)
            imf = model.imu_encoder(xi)
            fused = model.fusion(ef, imf)
            f = model.projection(fused)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def per_user_norm_block(Xe, Xi, U, user):
    """对单个用户的窗口做 per_user z-score (channel 维度), 与主协议一致"""
    mask = U == user
    Xe_b, Xi_b = Xe[mask].copy(), Xi[mask].copy()
    mu_e = Xe_b.mean(axis=(0, 2), keepdims=True)
    sd_e = Xe_b.std(axis=(0, 2), keepdims=True) + 1e-8
    mu_i = Xi_b.mean(axis=(0, 2), keepdims=True)
    sd_i = Xi_b.std(axis=(0, 2), keepdims=True) + 1e-8
    Xe_b = (Xe_b - mu_e) / sd_e
    Xi_b = (Xi_b - mu_i) / sd_i
    return Xe_b, Xi_b


def extract_users(model, X_emg, X_imu, Y, U, user_list, device):
    """逐用户提取特征 (避免一次性整库归一化的内存峰值)"""
    F, Ys, Us = [], [], []
    for u in user_list:
        Xe_b, Xi_b = per_user_norm_block(X_emg, X_imu, U, u)
        f = featurize(model, Xe_b, Xi_b, device)
        mask = U == u
        F.append(f)
        Ys.append(Y[mask])
        Us.append(np.full(mask.sum(), u, dtype=np.int64))
    return (np.concatenate(F), np.concatenate(Ys), np.concatenate(Us))


def split_shots(F_u, Y_u, k, seed, num_classes=4):
    """每类抽 k 条作为 few-shot, 返回 (shot_idx, rest_idx)"""
    rng = np.random.RandomState(seed)
    shot_idx, rest_idx = [], []
    for c in range(num_classes):
        idx = np.where(Y_u == c)[0]
        rng.shuffle(idx)
        n = min(k, len(idx))
        shot_idx.extend(idx[:n].tolist())
        rest_idx.extend(idx[n:].tolist())
    return np.array(shot_idx), np.array(rest_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default="./results/best_v4_allusers_nouser.pt")
    ap.add_argument("--ks", type=str, default="1,5,10,20")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp_name", type=str, default="fewshot")
    ap.add_argument("--save_dir", type=str, default="./results")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 70)
    print(f"Few-Shot Analysis | Device: {device} | k in {ks}")
    print("=" * 70)

    # 1. 数据 + split
    X_emg, X_imu, Y, U, num_users, _ = load_h5_data()
    train_users, test_users = replicate_external_split(U, seed=args.seed, test_ratio=0.2)
    print(f"[Split] {len(train_users)} train / {len(test_users)} test users")

    # 2. 模型 (与 best 配置一致)
    model = build_model_v4(
        "unified", num_classes=4, num_users=num_users,
        emg_channels=80, imu_channels=6, use_imu=True,
        use_whitening=False, use_contra=True,
        use_grl=False, use_mmd=False,
        feat_dim=128, dropout=0.0).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"[Model] Loaded {args.ckpt}")

    # 3. 提取特征
    print("[Extract] train users...")
    F_tr, Y_tr, _ = extract_users(model, X_emg, X_imu, Y, U, train_users, device)
    print(f"  train features: {F_tr.shape}")
    print("[Extract] test users...")
    F_te, Y_te, U_te = extract_users(model, X_emg, X_imu, Y, U, test_users, device)
    print(f"  test features: {F_te.shape}")

    # sanity: 主分类头 0-shot
    logits = model.gesture_classifier(torch.from_numpy(F_te).to(device))
    main0 = (logits.argmax(dim=1).cpu().numpy() == Y_te).mean()
    print(f"[Sanity] main-head 0-shot acc: {main0:.4f} (expect ~0.59)")

    # 4. 特征标准化 (只 fit 训练特征)
    scaler = StandardScaler().fit(F_tr)
    F_tr_s = scaler.transform(F_tr)
    F_te_s = scaler.transform(F_te)

    # 5. 0-shot: 只用训练特征 fit LR
    clf0 = LogisticRegression(C=1.0, max_iter=5000, n_jobs=-1)
    clf0.fit(F_tr_s, Y_tr)
    acc0 = accuracy_score(Y_te, clf0.predict(F_te_s))
    print(f"\n[0-shot] LR on train features -> test users: {acc0:.4f}")

    # 6. k-shot: 逐测试用户适配
    results = {'0shot_lr': acc0, '0shot_mainhead': float(main0),
               'per_user_per_k': {}}
    per_user_accs = {k: [] for k in ks}
    test_user_ids = np.unique(U_te)

    for u in test_user_ids:
        um = U_te == u
        F_u = F_te_s[um]
        Y_u = Y_te[um]
        for k in ks:
            shot_idx, rest_idx = split_shots(F_u, Y_u, k, seed=args.seed + int(u))
            if len(shot_idx) == 0 or len(rest_idx) == 0:
                per_user_accs[k].append(float('nan'))
                continue
            # 适配: 训练特征 + 该用户的 k-shot
            clf = LogisticRegression(C=1.0, max_iter=5000, n_jobs=-1)
            clf.fit(np.vstack([F_tr_s, F_u[shot_idx]]),
                    np.concatenate([Y_tr, Y_u[shot_idx]]))
            acc = accuracy_score(Y_u[rest_idx], clf.predict(F_u[rest_idx]))
            per_user_accs[k].append(acc)

    results['per_user_per_k'] = {k: per_user_accs[k] for k in ks}
    print("\n=== Few-Shot 结果 (固定特征 + LR) ===")
    print(f"{'k-shot':>8} | {'平均 acc':>10} | {'用户数':>6}")
    print(f"{'0 (LR)':>8} | {acc0:>10.4f} | {len(test_user_ids):>6}")
    for k in ks:
        arr = np.array([a for a in per_user_accs[k] if not np.isnan(a)])
        print(f"{k:>6}    | {arr.mean():>10.4f} | {len(arr):>6}")
        results[f'{k}shot'] = float(arr.mean())

    with open(os.path.join(args.save_dir, f"results_{args.exp_name}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.save_dir}/results_{args.exp_name}.json")


if __name__ == "__main__":
    main()
