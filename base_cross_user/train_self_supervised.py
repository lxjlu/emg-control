# -*- coding: utf-8 -*-
"""
自监督预训练 + 微调训练脚本
阶段1: 自监督对比学习预训练
阶段2: 冻结编码器，分类头微调
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader_v2 import load_h5_data, build_louo_loaders_v2
from models_v2 import SelfSupervisedModel


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            x_emg, x_imu, y, u = batch
            x_emg = x_emg.to(device, non_blocking=True)
            x_imu = x_imu.to(device, non_blocking=True)
            y = y.to(device)

            logits = model(x_emg, x_imu, stage='eval', mode='classify')
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def train_one_epoch_contrastive(model, loader, optimizer, device, scaler):
    """阶段1: 自监督对比学习"""
    model.train()
    total_loss, nb = 0.0, 0

    for batch in loader:
        x_emg, x_imu, y, u = batch
        x_emg = x_emg.to(device, non_blocking=True)
        x_imu = x_imu.to(device, non_blocking=True)

        # 生成增广版本 (使用dataset的augment)
        x_emg_aug = x_emg.clone()
        x_imu_aug = x_imu.clone()

        # 手动增广
        for i in range(x_emg.size(0)):
            aug_emg, aug_imu = x_emg[i].clone(), x_imu[i].clone()
            # 时间平移
            shift = np.random.randint(-10, 11)
            if shift != 0:
                aug_emg = torch.roll(aug_emg, shifts=shift, dims=-1)
                aug_imu = torch.roll(aug_imu, shifts=shift, dims=-1)
            # 加噪声
            aug_emg = aug_emg + torch.randn_like(aug_emg) * 0.05
            aug_imu = aug_imu + torch.randn_like(aug_imu) * 0.03
            # 幅度缩放
            scale = np.random.uniform(0.8, 1.2)
            aug_emg = aug_emg * scale
            x_emg_aug[i] = aug_emg
            x_imu_aug[i] = aug_imu

        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            out = model(x_emg, x_imu, x_emg_aug=x_emg_aug, x_imu_aug=x_imu_aug,
                        mode='contrastive')
            loss = out['contrastive_loss']

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()
        nb += 1

    return total_loss / nb


def train_one_epoch_finetune(model, loader, optimizer, criterion, device, scaler, user_weight=0.3):
    """阶段2: 微调分类头"""
    model.train()
    total_loss, nb = 0.0, 0

    for batch in loader:
        x_emg, x_imu, y, u = batch
        x_emg = x_emg.to(device, non_blocking=True)
        x_imu = x_imu.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        u = u.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            out = model(x_emg, x_imu, label=y, user=u, stage='train', mode='classify')
            loss_gesture = criterion(out['gesture_logits'], y)
            loss_user = F.cross_entropy(out['user_logits'], u)
            loss = loss_gesture + user_weight * loss_user

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()
        nb += 1

    return total_loss / nb


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = 0.0
        self.early_stop = False

    def __call__(self, val_acc):
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="self_supervised")
    ap.add_argument("--use_imu", action="store_true")
    ap.add_argument("--pretrain_epochs", type=int, default=50)
    ap.add_argument("--finetune_epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr_pretrain", type=float, default=1e-3)
    ap.add_argument("--lr_finetune", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--user_weight", type=float, default=0.3)
    ap.add_argument("--norm_strategy", type=str, default="per_user",
                    choices=["global", "none", "per_user"])
    ap.add_argument("--save_dir", type=str, default="./results")
    args = ap.parse_args()

    device = get_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.save_dir, exist_ok=True)

    # 加载数据
    X_emg, X_imu, Y, U, num_users, user2id = load_h5_data()
    if X_emg is None:
        print("ERROR: No data loaded!")
        return
    print(f"Total: {len(X_emg)} samples, {num_users} users")
    train_loader, test_loader, info, norm_stats = build_louo_loaders_v2(
        X_emg, X_imu, Y, U, test_ratio=args.test_ratio,
        batch_size=args.batch, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), norm_strategy=args.norm_strategy)

    print(f"Train: {info['n_train_users']} users, {info['train_samples']} samples")
    print(f"Test:  {info['n_test_users']} users, {info['test_samples']} samples")

    # 创建模型
    model = SelfSupervisedModel(
        num_classes=4, num_users=num_users,
        emg_channels=80, imu_channels=6,
        use_imu=args.use_imu,
        user_weight=args.user_weight,
        temperature=0.1,
        dropout=0.0).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    suffix = f"self_supervised_imu" if args.use_imu else "self_supervised"

    # ==================== 阶段1: 自监督预训练 ====================
    print("\n" + "=" * 60)
    print("Phase 1: Self-Supervised Pretraining")
    print("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_pretrain, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.pretrain_epochs)

    for ep in range(1, args.pretrain_epochs + 1):
        t0 = time.time()
        loss = train_one_epoch_contrastive(model, train_loader, optimizer, device, scaler)
        scheduler.step()
        t1 = time.time()

        if ep % 5 == 0 or ep == 1:
            print(f"Pretrain ep{ep:03d} | contrastive_loss={loss:.4f} | time={t1-t0:.1f}s")

    # 保存预训练权重
    pretrain_path = os.path.join(args.save_dir, f"pretrain_{suffix}.pt")
    torch.save(model.state_dict(), pretrain_path)
    print(f"Pretrained model saved to {pretrain_path}")

    # ==================== 阶段2: 微调分类 ====================
    print("\n" + "=" * 60)
    print("Phase 2: Fine-tuning with Labels")
    print("=" * 60)

    # 重置优化器 (只优化分类头，用更低学习率保护预训练特征)
    optimizer = torch.optim.AdamW([
        {'params': model.gesture_classifier.parameters(), 'lr': args.lr_finetune},
        {'params': model.user_classifier.parameters(), 'lr': args.lr_finetune},
        {'params': model.projection.parameters(), 'lr': args.lr_finetune * 0.1},
        {'params': model.emg_encoder.parameters(), 'lr': args.lr_finetune * 0.1},
    ], weight_decay=args.weight_decay)
    if args.use_imu:
        optimizer.add_param_group({'params': model.imu_encoder.parameters(), 'lr': args.lr_finetune * 0.1})
        optimizer.add_param_group({'params': model.fusion.parameters(), 'lr': args.lr_finetune * 0.1})

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.002)

    best_acc = 0.0
    history = []

    for ep in range(1, args.finetune_epochs + 1):
        t0 = time.time()
        loss = train_one_epoch_finetune(model, train_loader, optimizer, criterion, device, scaler,
                                        user_weight=args.user_weight)
        scheduler.step()
        t1 = time.time()

        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"best_{suffix}.pt"))
            new_best = " NEW BEST"
        else:
            new_best = ""

        history.append({
            "epoch": ep,
            "loss": loss,
            "train_acc": train_acc,
            "zeroshot_acc": test_acc,
        })

        early_stopping(test_acc)

        print(f"ep{ep:03d} | loss={loss:.4f} | train={train_acc:.4f} zeroshot={test_acc:.4f} best={best_acc:.4f} | time={t1-t0:.1f}s{new_best}")

        if early_stopping.early_stop:
            print(f"\nEarly stopping at epoch {ep}")
            break

    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(args.save_dir, f"best_{suffix}.pt")))
    final_acc = evaluate(model, test_loader, device)
    print(f"\nFinal Zero-Shot Accuracy: {final_acc:.4f}")

    with open(os.path.join(args.save_dir, f"results_{suffix}.json"), "w") as f:
        json.dump({
            "model": "self_supervised",
            "use_imu": args.use_imu,
            "norm_strategy": args.norm_strategy,
            "pretrain_epochs": args.pretrain_epochs,
            "finetune_epochs": args.finetune_epochs,
            "config": vars(args),
            "info": info,
            "zeroshot_acc": final_acc,
            "history": history,
            "early_stop_epoch": len(history),
        }, f, indent=2)

    print(f"Results saved to {args.save_dir}/results_{suffix}.json")


if __name__ == "__main__":
    main()