# -*- coding: utf-8 -*-
"""
diagnose_internal_louo.py - 内部 LOUO 诊断实验

目的: 回答 "59% 是模型设计瓶颈还是数据上界?"

方法: 在训练协议内部构造三级评测, 从"见过用户"到"未见用户"逐级放大域转移:

  [A] same_user_eval   : 训练用户(112人)中留出 15% 样本做评测
                         (见过用户, 未见样本) → 衡量样本级/时间窗级泛化
  [B] internal_louo    : 从 140 训练用户中再留出 28 人做 held-out
                         (同分布, 完全未见用户) → 衡量同人群内的用户级泛化
  [C] external_louo    : 原协议 35 个 held-out 用户 (seed 42 划分)
                         → 与主实验 59.31% 直接对齐

诊断逻辑:
  若 A >> B ≈ C  (~99% vs ~59%): 模型能泛化到"训练用户池内分布",
     但无法泛化到"新用户个体" → 瓶颈在用户域转移本身 (数据/生物多样性上界)
  若 A ≈ B ≈ C: 模型连同人群内未见用户都泛化不了 → 问题在模型容量/训练技巧
  若 B 明显 > C: 说明外部 split 的用户与训练用户存在系统分布差异 (划分偏差)
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_v4 import build_model as build_model_v4
from data_loader_v2 import (load_h5_data, EMGIMUDataset, emg_imu_augment)
from train_v4 import train_one_epoch, evaluate


def replicate_external_split(U, seed=42, test_ratio=0.2):
    """复现 build_louo_loaders_v2 的 seed 划分 (与主实验外部测试用户一致)"""
    unique_users = np.sort(np.unique(U))
    rng = np.random.RandomState(seed)
    shuffled = unique_users.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_ratio))
    train_users = shuffled[:-n_test]
    test_users = shuffled[-n_test:]
    return train_users, test_users


def per_user_normalize(X_emg, X_imu, users):
    """per_user z-score: 每个用户用自己的统计量归一化 (与主实验一致)"""
    X_emg_n = np.zeros_like(X_emg)
    X_imu_n = np.zeros_like(X_imu)
    for u in np.unique(users):
        mask = users == u
        mu_e = X_emg[mask].mean(axis=(0, 2), keepdims=True)
        sd_e = X_emg[mask].std(axis=(0, 2), keepdims=True) + 1e-8
        mu_i = X_imu[mask].mean(axis=(0, 2), keepdims=True)
        sd_i = X_imu[mask].std(axis=(0, 2), keepdims=True) + 1e-8
        X_emg_n[mask] = (X_emg[mask] - mu_e) / sd_e
        X_imu_n[mask] = (X_imu[mask] - mu_i) / sd_i
    return X_emg_n, X_imu_n


def build_loader(X_emg, X_imu, Y, U, sample_mask, batch_size=256,
                 num_workers=4, shuffle=False, augment=False, persistent=False):
    """用给定样本 mask 构造 (已归一化) loader"""
    users_sub = U[sample_mask]
    # 只对该子集的用户做 per_user 归一化
    Xe_n, Xi_n = per_user_normalize(X_emg[sample_mask], X_imu[sample_mask], users_sub)
    ds = EMGIMUDataset(Xe_n, Xi_n, Y[sample_mask], U[sample_mask],
                       augment=emg_imu_augment if augment else None)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=(num_workers > 0 and persistent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42, help="外部 split seed")
    ap.add_argument("--inner_seed", type=int, default=7, help="内部 LOUO split seed")
    ap.add_argument("--sameuser_ratio", type=float, default=0.15,
                    help="训练用户中留出做 same-user 评测的样本比例")
    ap.add_argument("--pretrained", type=str,
                    default="./pretrained/v3_pretrain_spec_allusers_encoder.pt")
    ap.add_argument("--exp_name", type=str, default="internal_louo")
    ap.add_argument("--save_dir", type=str, default="./results")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 70)
    print("Internal LOUO Diagnostic | Device:", device)
    print("=" * 70)

    # 1. 加载数据
    X_emg, X_imu, Y, U, num_users, _ = load_h5_data()

    # 2. 复现外部 split (与主实验一致)
    all_train_users, external_test_users = replicate_external_split(
        U, seed=args.seed, test_ratio=0.2)
    print(f"[Split] external: {len(all_train_users)} train / "
          f"{len(external_test_users)} test users (seed={args.seed})")

    # 3. 内部 LOUO: 在 all_train 内再留 20% (28人) 做 held-out
    rng_inner = np.random.RandomState(args.inner_seed)
    shuffled = all_train_users.copy()
    rng_inner.shuffle(shuffled)
    n_inner_test = int(round(len(all_train_users) * 0.2))
    internal_train_users = shuffled[:-n_inner_test]
    internal_test_users = shuffled[-n_inner_test:]
    print(f"[Split] internal: {len(internal_train_users)} train / "
          f"{len(internal_test_users)} held-out users (seed={args.inner_seed})")

    # 4. same-user: 从 internal_train 每个用户留出 15% 样本做评测
    rng_su = np.random.RandomState(12345)
    is_internal_train = np.isin(U, internal_train_users)
    is_internal_test = np.isin(U, internal_test_users)
    is_external_test = np.isin(U, external_test_users)

    sameuser_eval_mask = np.zeros(len(U), dtype=bool)
    tr_users_ids = np.unique(U[is_internal_train])
    for u in tr_users_ids:
        um = (U == u) & is_internal_train
        idx = np.where(um)[0]
        n_keep = max(1, int(len(idx) * args.sameuser_ratio))
        keep = rng_su.choice(idx, size=n_keep, replace=False)
        sameuser_eval_mask[keep] = True
    internal_train_mask = is_internal_train & ~sameuser_eval_mask
    print(f"[Split] same-user: {internal_train_mask.sum()} train samples, "
          f"{sameuser_eval_mask.sum()} eval samples (ratio={args.sameuser_ratio})")

    # 5. 构造 loader
    print("[Loader] Building loaders...")
    train_loader = build_loader(X_emg, X_imu, Y, U, internal_train_mask,
                                batch_size=args.batch, num_workers=args.num_workers,
                                shuffle=True, augment=True, persistent=True)
    sameuser_loader = build_loader(X_emg, X_imu, Y, U, sameuser_eval_mask,
                                   batch_size=args.batch, num_workers=args.num_workers)
    internal_loader = build_loader(X_emg, X_imu, Y, U, is_internal_test,
                                   batch_size=args.batch, num_workers=args.num_workers)
    external_loader = build_loader(X_emg, X_imu, Y, U, is_external_test,
                                   batch_size=args.batch, num_workers=args.num_workers)
    print(f"[Loader] train={internal_train_mask.sum()} sameuser={sameuser_eval_mask.sum()} "
          f"internal_te={is_internal_test.sum()} external_te={is_external_test.sum()}")

    # 6. 构建模型 (与 best 配置一致: nouser, contra, 无 GRL/MMD/白化)
    model = build_model_v4(
        "unified", num_classes=4, num_users=num_users,
        emg_channels=80, imu_channels=6,
        use_imu=True,
        use_whitening=False, use_contra=True,
        use_grl=False, use_mmd=False,
        feat_dim=128, dropout=0.0).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # 7. 加载全用户预训练编码器
    if args.pretrained and os.path.exists(args.pretrained):
        state = torch.load(args.pretrained, map_location=device, weights_only=False)
        model_state = model.state_dict()
        loaded = skipped = 0
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded += 1
            else:
                skipped += 1
        model.load_state_dict(model_state)
        print(f"[Pretrained] Loaded {loaded} tensors, skipped {skipped}")

    # 8. 优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # 9. 训练 (early stopping 以 external_louo 为准, 与主实验对齐)
    best_acc = 0.0
    best_epoch = 0
    history = []
    suffix = args.exp_name
    print("\n" + "=" * 70)
    print("Training (early-stop on external_louo for comparability)...")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            use_contra=True, contra_weight=0.5,
            use_grl=False, grl_lambda=0.0, domain_weight=0.0,
            use_mmd=False, mmd_weight=0.0)
        scheduler.step()

        ext_acc = evaluate(model, external_loader, device)
        # 每 5 个 epoch 额外评估一次, 观察曲线
        if epoch % 5 == 0 or epoch == 1:
            su_acc = evaluate(model, sameuser_loader, device)
            int_acc = evaluate(model, internal_loader, device)
            print(f"  [check] sameuser={su_acc:.4f} internal={int_acc:.4f} "
                  f"external={ext_acc:.4f}")

        history.append({'epoch': epoch, 'loss': train_loss,
                        'train_acc': train_acc, 'external_acc': ext_acc})

        is_best = ext_acc > best_acc
        if is_best:
            best_acc = ext_acc
            best_epoch = epoch
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f"best_{suffix}.pt"))

        dt = time.time() - t0
        tag = " NEW BEST" if is_best else ""
        print(f"ep{epoch:03d} | loss={train_loss:.4f} | train={train_acc:.4f} "
              f"external={ext_acc:.4f} best={best_acc:.4f} | time={dt:.1f}s{tag}")

        if epoch - best_epoch >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # 10. 用 best checkpoint 统一评测三个 loader
    print("\n" + "=" * 70)
    print("Final evaluation with best checkpoint...")
    print("=" * 70)
    ckpt = torch.load(os.path.join(args.save_dir, f"best_{suffix}.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()

    final = {
        'same_user_acc': evaluate(model, sameuser_loader, device),
        'internal_louo_acc': evaluate(model, internal_loader, device),
        'external_louo_acc': evaluate(model, external_loader, device),
        'train_acc_best_epoch': history[best_epoch - 1]['train_acc'],
        'best_epoch': best_epoch,
    }
    print(f"\n=== 内部 LOUO 诊断结果 ===")
    print(f"[A] same_user_eval   (见过用户, 未见样本): {final['same_user_acc']:.4f}")
    print(f"[B] internal_louo    (同分布未见用户)     : {final['internal_louo_acc']:.4f}")
    print(f"[C] external_louo    (主协议未见用户)     : {final['external_louo_acc']:.4f}")
    gap_user = final['same_user_acc'] - final['internal_louo_acc']
    gap_split = final['internal_louo_acc'] - final['external_louo_acc']
    print(f"\n样本级 gap (A-B): {gap_user:+.4f}")
    print(f"划分偏差 gap (B-C): {gap_split:+.4f}")
    if final['internal_louo_acc'] > final['external_louo_acc'] + 0.05:
        print(">> 结论倾向: 外部 35 用户与训练用户存在系统分布差异 (split 偏差)")
    elif gap_user > 0.1:
        print(">> 结论倾向: 模型在训练用户池内泛化良好, 瓶颈在新用户域转移 (数据上界)")
    else:
        print(">> 结论倾向: 模型连同人群内未见用户都泛化不佳 (模型/训练问题)")

    result = {
        'config': vars(args),
        'split': {
            'n_internal_train_users': len(internal_train_users),
            'n_internal_test_users': len(internal_test_users),
            'n_external_test_users': len(external_test_users),
            'n_train_samples': int(internal_train_mask.sum()),
            'n_sameuser_samples': int(sameuser_eval_mask.sum()),
        },
        'final': final,
        'history': history,
        'num_params': n_params,
    }
    with open(os.path.join(args.save_dir, f"results_{suffix}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {args.save_dir}/results_{suffix}.json")


if __name__ == "__main__":
    main()
