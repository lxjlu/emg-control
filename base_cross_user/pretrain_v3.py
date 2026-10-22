# -*- coding: utf-8 -*-
"""
pretrain_v3.py - 自监督预训练 v2 (统计特征预测 + 跨用户对比学习)

4 个改进:
1. 统计特征预测替代原始信号重建 (学高层手势语义)
2. 跨用户对比预训练 (同手势不同用户=正, 与下游目标对齐)
3. mask_ratio=0.75 (更难的任务 → 更鲁棒特征)
4. 100 epochs (充分收敛)

在全部用户数据（含测试用户）上预训练编码器，学到跨用户通用表征
保存编码器权重供下游微调使用
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, BatchSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_v3 import build_model
from data_loader_v2 import load_h5_data


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class GestureGroupSampler:
    """按手势分组采样，保证每个 batch 内每个手势有多个用户的样本
    用于对比学习: 确保 batch 内有同手势不同用户的正样本对
    """
    def __init__(self, labels, users, batch_size, samples_per_gesture=None):
        self.labels = labels
        self.users = users
        self.batch_size = batch_size
        num_classes = len(np.unique(labels))
        self.samples_per_gesture = samples_per_gesture or (batch_size // num_classes)
        self.gesture_indices = {}
        for g in range(num_classes):
            idx = np.where(labels == g)[0]
            if len(idx) > 0:
                self.gesture_indices[g] = idx
        self.num_classes = len(self.gesture_indices)

    def __iter__(self):
        for _ in range(len(self)):
            batch = []
            for g in self.gesture_indices:
                idx = self.gesture_indices[g]
                chosen = np.random.choice(idx, size=self.samples_per_gesture, replace=True)
                batch.extend(chosen.tolist())
            np.random.shuffle(batch)
            yield batch[:self.batch_size]

    def __len__(self):
        return len(self.labels) // self.batch_size


def pretrain_one_epoch(model, loader, optimizer, device, scaler,
                       mask_ratio=0.75, contra_weight=0.5, use_imu=True,
                       Y_tensor=None, U_tensor=None):
    model.train()
    total_loss, total_stats, total_contra = 0.0, 0.0, 0.0
    nb = 0
    for batch_indices in loader:
        idx = torch.tensor(batch_indices) if isinstance(batch_indices, list) else batch_indices
        x_emg = X_EMG[idx]
        y = Y_TENSOR[idx]
        u = U_TENSOR[idx]
        if use_imu:
            x_imu = X_IMU[idx]
            x_imu = x_imu.to(device, non_blocking=True)
        else:
            x_imu = None
        x_emg = x_emg.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        u = u.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            loss, stats_loss, contra_loss = model.forward_pretrain(
                x_emg, x_imu, labels=y, users=u,
                mask_ratio=mask_ratio, contra_weight=contra_weight)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_stats += stats_loss.item()
        total_contra += contra_loss.item() if torch.is_tensor(contra_loss) else contra_loss
        nb += 1
    return (total_loss / max(nb, 1),
            total_stats / max(nb, 1),
            total_contra / max(nb, 1))


# 全局变量 (供 pretrain_one_epoch 引用)
X_EMG = None
X_IMU = None
Y_TENSOR = None
U_TENSOR = None


def main():
    global X_EMG, X_IMU, Y_TENSOR, U_TENSOR

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="unified")
    ap.add_argument("--use_imu", action="store_true")
    ap.add_argument("--norm_strategy", type=str, default="per_user",
                    choices=["global", "none", "per_user"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--contra_weight", type=float, default=0.5)
    ap.add_argument("--pretrain_target", type=str, default="stats",
                    choices=["stats", "spectrogram"],
                    help="预训练目标: stats=统计特征预测, spectrogram=STFT频谱图重建")
    ap.add_argument("--train_users_only", action="store_true",
                    help="只用训练用户预训练 (不碰测试用户, 严格纯零样本)")
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--save_dir", type=str, default="./pretrained")
    ap.add_argument("--exp_name", type=str, default="v3_pretrain_v2")
    args = ap.parse_args()

    device = get_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print(f"Self-Supervised Pretraining v2 | Device: {device}")
    print(f"Model: {args.model} | IMU: {args.use_imu}")
    print(f"Pretrain target: {args.pretrain_target}")
    print(f"Mask ratio: {args.mask_ratio} | LR: {args.lr}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch}")
    print(f"Contra weight: {args.contra_weight}")
    if args.pretrain_target == "stats":
        print("  [1] Stats Prediction (MAV/RMS/WL/ZC/SSC + freq)")
    else:
        print("  [1] Spectrogram Reconstruction (STFT log-magnitude)")
    print("  [2] Cross-User Contrastive (temp=0.1, hard_neg=0.5)")
    print("  [3] GroupSampler for batch diversity")
    print("  [4] mask_ratio=0.75 for robust features")
    print("=" * 60)

    # 加载全部数据
    X_emg, X_imu, Y, U, num_users, _ = load_h5_data()

    # 方案1: 只用训练用户预训练 (与微调阶段相同的用户划分, 不碰测试用户)
    if args.train_users_only:
        unique_users = np.unique(U)
        n_test = max(1, int(len(unique_users) * args.test_ratio))
        np.random.shuffle(unique_users)  # 已通过 np.random.seed(args.seed) 固定
        train_users = unique_users[:-n_test]
        test_users = unique_users[-n_test:]
        train_mask = np.isin(U, train_users)
        print(f"[Split] Pretrain on {len(train_users)} train users only "
              f"(excluding {len(test_users)} test users, same as fine-tuning split)")
        X_emg = X_emg[train_mask]
        X_imu = X_imu[train_mask]
        Y = Y[train_mask]
        U = U[train_mask]

    # 应用 per_user 归一化（与微调阶段一致）
    if args.norm_strategy == "per_user":
        print("[Norm] Applying per-user Z-Score normalization...")
        X_emg_norm = np.zeros_like(X_emg)
        X_imu_norm = np.zeros_like(X_imu)
        for u_id in np.unique(U):
            mask = (U == u_id)
            mean_e = X_emg[mask].mean(axis=(0, 2), keepdims=True)
            std_e = X_emg[mask].std(axis=(0, 2), keepdims=True) + 1e-8
            mean_i = X_imu[mask].mean(axis=(0, 2), keepdims=True)
            std_i = X_imu[mask].std(axis=(0, 2), keepdims=True) + 1e-8
            X_emg_norm[mask] = (X_emg[mask] - mean_e) / std_e
            X_imu_norm[mask] = (X_imu[mask] - mean_i) / std_i
        X_emg, X_imu = X_emg_norm, X_imu_norm

    # 设置全局变量 (供 pretrain_one_epoch 引用)
    X_EMG = torch.from_numpy(X_emg).float()
    X_IMU = torch.from_numpy(X_imu).float()
    Y_TENSOR = torch.from_numpy(Y).long()
    U_TENSOR = torch.from_numpy(U).long()

    # 构建 GroupSampler (保证 batch 内同手势多用户 → 对比学习正样本对充足)
    sampler = GestureGroupSampler(Y, U, args.batch)
    loader = [(indices,) for indices in sampler]  # 只用于控制 batch 数量

    # 构建模型（不启用消融组件，保持 baseline 架构）
    model = build_model(
        args.model, num_classes=4, num_users=num_users,
        emg_channels=80, imu_channels=6, use_imu=args.use_imu,
        use_whitening=False, use_arcface=False,
        use_slim=False, use_contra=False, use_v02=False).to(device)

    # 初始化预训练头
    model.init_pretrain_heads(emg_channels=80, emg_len=X_emg.shape[2],
                              pretrain_target=args.pretrain_target)
    if args.pretrain_target == 'stats':
        model.stats_predictor = model.stats_predictor.to(device)
        head_params = sum(p.numel() for p in model.stats_predictor.parameters())
    else:
        model.spec_reconstructor = model.spec_reconstructor.to(device)
        head_params = sum(p.numel() for p in model.spec_reconstructor.parameters())
    model.pretrain_contra = model.pretrain_contra.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} (encoder: {n_params - head_params:,} + pretrain_head: {head_params:,})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    print("\n" + "=" * 60)
    print("Pretraining v2 (Stats Prediction + Cross-User Contrastive)...")
    print("=" * 60)

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        sampler = GestureGroupSampler(Y, U, args.batch)  # 每 epoch 重新采样
        # 使用 BatchSampler 方式
        batch_iter = iter(sampler)
        total_loss, total_stats, total_contra = 0.0, 0.0, 0.0
        nb = 0
        for batch_indices in batch_iter:
            idx_t = torch.tensor(batch_indices)
            x_emg = X_EMG[idx_t]
            y = Y_TENSOR[idx_t]
            u = U_TENSOR[idx_t]
            if args.use_imu:
                x_imu = X_IMU[idx_t]
                x_imu = x_imu.to(device, non_blocking=True)
            else:
                x_imu = None
            x_emg = x_emg.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            u = u.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                loss, stats_loss, contra_loss = model.forward_pretrain(
                    x_emg, x_imu, labels=y, users=u,
                    mask_ratio=args.mask_ratio, contra_weight=args.contra_weight)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_stats += stats_loss.item()
            total_contra += contra_loss.item() if torch.is_tensor(contra_loss) else contra_loss
            nb += 1

        avg_loss = total_loss / max(nb, 1)
        avg_stats = total_stats / max(nb, 1)
        avg_contra = total_contra / max(nb, 1)
        scheduler.step()

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            # 只保存编码器 + 投影头 + 融合层权重（不保存预训练头和分类器）
            save_dict = {}
            for name, param in model.named_parameters():
                if ('stats_predictor' not in name and 'spec_reconstructor' not in name
                        and 'classifier' not in name and 'pretrain_contra' not in name):
                    save_dict[name] = param.data
            torch.save(save_dict, os.path.join(args.save_dir, f"{args.exp_name}_encoder.pt"))

        dt = time.time() - t0
        tag = " NEW BEST" if is_best else ""
        main_name = "stats" if args.pretrain_target == "stats" else "spec"
        print(f"ep{epoch:03d} | loss={avg_loss:.4f} {main_name}={avg_stats:.4f} contra={avg_contra:.4f} best={best_loss:.4f} | {dt:.1f}s{tag}")

    print(f"\nBest total loss: {best_loss:.4f}")
    print(f"Encoder weights saved to {args.save_dir}/{args.exp_name}_encoder.pt")


if __name__ == "__main__":
    main()
