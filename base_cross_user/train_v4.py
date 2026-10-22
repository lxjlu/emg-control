# -*- coding: utf-8 -*-
"""
train_v4.py - v4 模型训练

核心策略:
  1. 无用户分类头 (不保留用户信息)
  2. GRL 对抗域对齐 (主动去除用户信息), lambda 退火
  3. MMD 跨用户分布对齐 (温和去除用户信息)
  4. 特征白化 (去除用户特定协方差)
  5. 跨用户对比学习 (同手势不同用户拉近)
  6. 频谱预训练初始化

训练调度:
  - Warmup (前 N 个 epoch): 只训练手势分类 + 对比学习, GRL lambda=0
  - Annealing: GRL lambda 从 0 线性增加到 max_grl_lambda
  - MMD weight 固定较小值, 避免破坏手势特征
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_v4 import build_model
from data_loader_v2 import load_h5_data, build_louo_loaders_v2
from train_v3 import GestureGroupSampler


def train_one_epoch(model, loader, optimizer, criterion, device, scaler,
                    use_contra=True, contra_weight=0.5,
                    use_grl=True, grl_lambda=0.0, domain_weight=0.1,
                    use_mmd=True, mmd_weight=0.01):
    model.train()
    total_loss, nb = 0.0, 0
    correct, total = 0, 0
    for batch in loader:
        x_emg, x_imu, y, u = batch
        x_emg = x_emg.to(device, non_blocking=True)
        x_imu = x_imu.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        u = u.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            out = model(x_emg, x_imu, label=y, user=u, stage='train',
                        grl_lambda=grl_lambda)

            # 主损失: 手势分类 (无用户分类损失!)
            loss = criterion(out['gesture_logits'], y)

            # 对比学习
            if use_contra and 'contra_loss' in out:
                loss = loss + contra_weight * out['contra_loss']

            # GRL 对抗域对齐: 域判别器损失
            # 注意: 因为用了 GRL, 这个 loss 反传时会反转梯度到编码器
            # 域判别器学会区分用户, 编码器被训练为产生用户不可区分的特征
            if use_grl and 'domain_logits' in out and grl_lambda > 0:
                loss_domain = F.cross_entropy(out['domain_logits'], u)
                loss = loss + domain_weight * loss_domain

            # MMD 分布对齐
            if use_mmd and 'mmd_loss' in out:
                loss = loss + mmd_weight * out['mmd_loss']

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        pred = out['gesture_logits'].argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        total_loss += loss.item()
        nb += 1

    return total_loss / max(nb, 1), correct / max(total, 1)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            x_emg, x_imu, y, _ = batch
            x_emg = x_emg.to(device, non_blocking=True)
            x_imu = x_imu.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x_emg, x_imu, stage='eval')
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="unified")
    ap.add_argument("--use_imu", action="store_true")
    ap.add_argument("--norm_strategy", type=str, default="per_user")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--exp_name", type=str, default=None)
    ap.add_argument("--label_smoothing", type=float, default=0.0,
                    help="手势分类 CE 的 label smoothing")
    ap.add_argument("--save_dir", type=str, default="./results")
    ap.add_argument("--cache_dir", type=str, default="./data_cache_v2",
                    help="数据缓存根目录 (可指向长窗口缓存, 如 data_cache_v2/long_300_30)")

    # v4 组件
    ap.add_argument("--no_whitening", action="store_true", help="禁用特征白化")
    ap.add_argument("--no_contra", action="store_true", help="禁用对比学习")
    ap.add_argument("--no_grl", action="store_true", help="禁用 GRL 对抗域对齐")
    ap.add_argument("--no_mmd", action="store_true", help="禁用 MMD 分布对齐")
    ap.add_argument("--contra_weight", type=float, default=0.5)
    ap.add_argument("--domain_weight", type=float, default=0.1,
                    help="GRL 域判别器损失权重")
    ap.add_argument("--max_grl_lambda", type=float, default=0.5,
                    help="GRL lambda 最大值")
    ap.add_argument("--grl_warmup_epochs", type=int, default=5,
                    help="GRL warmup epoch 数 (前 N 个 epoch lambda=0)")
    ap.add_argument("--grl_anneal_epochs", type=int, default=15,
                    help="GRL lambda 从 0 退火到 max 的 epoch 数")
    ap.add_argument("--mmd_weight", type=float, default=0.01,
                    help="MMD 损失权重")

    # 预训练
    ap.add_argument("--pretrained", type=str, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    suffix = args.exp_name or "v4"
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print(f"Model: v4 | Device: {device}")
    print(f"Components: whitening={not args.no_whitening} "
          f"contra={not args.no_contra} grl={not args.no_grl} mmd={not args.no_mmd}")
    print(f"GRL: max_lambda={args.max_grl_lambda} "
          f"warmup={args.grl_warmup_epochs} anneal={args.grl_anneal_epochs}")
    print(f"MMD weight: {args.mmd_weight}")
    print(f"Norm: {args.norm_strategy} | LR: {args.lr} | WD: {args.weight_decay}")
    print(f"Patience: {args.patience} | Dropout: {args.dropout}")
    print(f"Pretrained: {args.pretrained}")
    print("=" * 60)

    # 加载数据
    X_emg, X_imu, Y, U, num_users, _ = load_h5_data(cache_dir=args.cache_dir)
    train_loader, test_loader, _, _ = build_louo_loaders_v2(
        X_emg, X_imu, Y, U, test_ratio=args.test_ratio, batch_size=args.batch,
        num_workers=args.num_workers, norm_strategy=args.norm_strategy)

    # 构建模型
    model = build_model(
        args.model,
        num_classes=4, num_users=num_users,
        emg_channels=80, imu_channels=6,
        use_imu=args.use_imu,
        use_whitening=not args.no_whitening,
        use_contra=not args.no_contra,
        use_grl=not args.no_grl,
        use_mmd=not args.no_mmd,
        feat_dim=128, dropout=args.dropout).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # 加载预训练
    if args.pretrained and os.path.exists(args.pretrained):
        state = torch.load(args.pretrained, map_location=device, weights_only=False)
        model_state = model.state_dict()
        loaded, skipped = 0, 0
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded += 1
            else:
                skipped += 1
        model.load_state_dict(model_state)
        print(f"[Pretrained] Loaded {loaded} tensors, skipped {skipped}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    print(f"Label smoothing: {args.label_smoothing}")
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    best_acc = 0.0
    best_epoch = 0
    history = []

    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # GRL lambda 退火
        if not args.no_grl:
            if epoch <= args.grl_warmup_epochs:
                grl_lambda = 0.0
            else:
                anneal_progress = min(1.0, (epoch - args.grl_warmup_epochs) /
                                      max(args.grl_anneal_epochs, 1))
                grl_lambda = args.max_grl_lambda * anneal_progress
        else:
            grl_lambda = 0.0

        # 对比学习使用 GroupSampler
        if not args.no_contra:
            train_labels = train_loader.dataset.Y.numpy()
            train_users = train_loader.dataset.U.numpy()
            sampler = GestureGroupSampler(train_labels, train_users, args.batch)
            train_loader = DataLoader(train_loader.dataset, batch_sampler=sampler,
                                      num_workers=args.num_workers, pin_memory=True)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            use_contra=not args.no_contra, contra_weight=args.contra_weight,
            use_grl=not args.no_grl, grl_lambda=grl_lambda,
            domain_weight=args.domain_weight,
            use_mmd=not args.no_mmd, mmd_weight=args.mmd_weight)
        scheduler.step()

        test_acc = evaluate(model, test_loader, device)

        history.append({
            'epoch': epoch, 'loss': train_loss,
            'train_acc': train_acc, 'zeroshot_acc': test_acc,
            'grl_lambda': grl_lambda})

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f"best_{suffix}.pt"))

        dt = time.time() - t0
        tag = " NEW BEST" if is_best else ""
        print(f"ep{epoch:03d} | loss={train_loss:.4f} | "
              f"train={train_acc:.4f} zeroshot={test_acc:.4f} best={best_acc:.4f} "
              f"| grl_lambda={grl_lambda:.3f} | time={dt:.1f}s{tag}")

        if epoch - best_epoch >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nFinal Zero-Shot Accuracy: {best_acc:.4f}")

    result = {
        'model': 'v4', 'use_imu': args.use_imu,
        'norm_strategy': args.norm_strategy,
        'use_whitening': not args.no_whitening,
        'use_contra': not args.no_contra,
        'use_grl': not args.no_grl,
        'use_mmd': not args.no_mmd,
        'pretrained': args.pretrained,
        'config': vars(args), 'zeroshot_acc': best_acc,
        'history': history, 'early_stop_epoch': epoch,
        'num_params': n_params,
    }
    with open(os.path.join(args.save_dir, f"results_{suffix}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {args.save_dir}/results_{suffix}.json")


if __name__ == "__main__":
    main()
