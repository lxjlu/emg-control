# -*- coding: utf-8 -*-
"""
train_v3.py - 消融实验训练脚本
支持 5 个可选组件的组合训练
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
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_v3 import build_model, cross_user_mixup, signal_augmentation
from data_loader_v2 import load_h5_data, build_louo_loaders_v2


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ==================== GroupSampler: 保证 batch 内同手势多用户 ====================
class GestureGroupSampler:
    """按手势分组采样，保证每个 batch 内每个手势有多个用户的样本"""
    def __init__(self, labels, users, batch_size, samples_per_gesture=None):
        self.labels = labels
        self.users = users
        self.batch_size = batch_size
        num_classes = len(np.unique(labels))
        self.samples_per_gesture = samples_per_gesture or (batch_size // num_classes)
        # 按手势分组索引
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


def train_one_epoch(model, loader, optimizer, criterion, device, scaler,
                    use_contra=False, contra_weight=0.5, user_weight=0.3,
                    use_domain_align=False, domain_weight=0.5,
                    use_kl_uniform=False, kl_weight=0.05,
                    use_mixup=False, use_aug=False,
                    update_whitening=False, whitening_feat_list=None):
    model.train()
    total_loss, nb = 0.0, 0
    correct, total = 0, 0
    for batch in loader:
        x_emg, x_imu, y, u = batch
        x_emg = x_emg.to(device, non_blocking=True)
        x_imu = x_imu.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        u = u.to(device, non_blocking=True)

        # 方向C: 信号级增强
        if use_aug:
            x_emg = signal_augmentation(x_emg, aug_prob=0.5)

        # 方向C: 跨用户 Mixup (同手势不同用户混合)
        if use_mixup:
            x_emg, x_imu, y, perm, lam = cross_user_mixup(x_emg, x_imu, y, u, alpha=0.4)

        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            out = model(x_emg, x_imu, label=y, user=u, stage='train')
            loss_gesture = criterion(out['gesture_logits'], y)
            loss_user = F.cross_entropy(out['user_logits'], u)
            loss = loss_gesture + user_weight * loss_user
            if use_contra and 'contra_loss' in out:
                loss = loss + contra_weight * out['contra_loss']
            if use_domain_align and 'domain_logits' in out:
                loss_domain = F.cross_entropy(out['domain_logits'], u)
                loss = loss + domain_weight * loss_domain
            # 方向B: KL 用户均匀约束
            if use_kl_uniform and 'kl_uniform' in out:
                loss = loss + kl_weight * out['kl_uniform']

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        pred = out['gesture_logits'].argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        if update_whitening and whitening_feat_list is not None:
            with torch.no_grad():
                feat = out['gesture_feat'].detach().cpu()
                whitening_feat_list.append(feat)

        total_loss += loss.item()
        nb += 1
    return total_loss / max(nb, 1), correct / max(total, 1)


def evaluate(model, loader, device, use_imu=True):
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
    ap.add_argument("--norm_strategy", type=str, default="per_user",
                    choices=["global", "none", "per_user"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--user_weight", type=float, default=0.3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--exp_name", type=str, default=None)
    ap.add_argument("--save_dir", type=str, default="./results")

    # 可选组件
    ap.add_argument("--use_whitening", action="store_true")
    ap.add_argument("--use_arcface", action="store_true")
    ap.add_argument("--use_slim", action="store_true")
    ap.add_argument("--use_contra", action="store_true")
    ap.add_argument("--use_v02", action="store_true")
    ap.add_argument("--use_domain_align", action="store_true",
                    help="对抗域对齐 (GRL + 用户域判别器)")
    ap.add_argument("--contra_weight", type=float, default=0.5)
    ap.add_argument("--domain_weight", type=float, default=0.5)
    ap.add_argument("--arcface_margin", type=float, default=0.3)
    ap.add_argument("--arcface_scale", type=float, default=30.0)
    # 方向A: 更强对比学习
    ap.add_argument("--use_supcon", action="store_true",
                    help="Supervised Contrastive Loss (所有同手势为正样本)")
    ap.add_argument("--hard_neg_ratio", type=float, default=0.5,
                    help="hard negative 保留比例")
    ap.add_argument("--temp_anneal", action="store_true",
                    help="temperature 退火 (0.1→0.01)")
    # 方向B: 显式用户信息去除
    ap.add_argument("--use_kl_uniform", action="store_true",
                    help="KL 用户均匀约束 (温和去除用户信息)")
    ap.add_argument("--kl_weight", type=float, default=0.05)
    ap.add_argument("--feat_dim_override", type=int, default=None,
                    help="信息瓶颈: 覆盖特征维度")
    # 方向C: 数据增强
    ap.add_argument("--use_mixup", action="store_true",
                    help="跨用户 Mixup (同手势不同用户混合)")
    ap.add_argument("--use_aug", action="store_true",
                    help="信号级增强 (幅值缩放 + 通道 dropout)")
    # 自监督预训练
    ap.add_argument("--pretrained", type=str, default=None,
                    help="预训练编码器权重路径 (.pt)")
    ap.add_argument("--freeze_encoder", action="store_true",
                    help="冻结预训练编码器，只训练分类头")
    args = ap.parse_args()

    device = get_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.save_dir, exist_ok=True)

    # 自动生成后缀
    if args.exp_name:
        suffix = args.exp_name
    else:
        parts = [args.model]
        if args.use_imu: parts.append("imu")
        if args.use_whitening: parts.append("white")
        if args.use_arcface: parts.append("arc")
        if args.use_slim: parts.append("slim")
        if args.use_contra: parts.append("contra")
        if args.use_v02: parts.append("v02")
        suffix = "_".join(parts)

    print("=" * 60)
    print(f"Model: {args.model} | Device: {device}")
    print(f"Components: whitening={args.use_whitening} arcface={args.use_arcface} "
          f"slim={args.use_slim} contra={args.use_contra} v02={args.use_v02} "
          f"domain_align={args.use_domain_align}")
    print(f"Norm: {args.norm_strategy} | LR: {args.lr} | WD: {args.weight_decay}")
    print(f"Patience: {args.patience} | Dropout: {args.dropout}")
    if args.pretrained:
        print(f"Pretrained: {args.pretrained} | Freeze: {args.freeze_encoder}")
    print("=" * 60)

    # 加载数据
    X_emg, X_imu, Y, U, num_users, _ = load_h5_data()
    train_loader, test_loader, _, _ = build_louo_loaders_v2(
        X_emg, X_imu, Y, U, test_ratio=args.test_ratio, batch_size=args.batch,
        num_workers=args.num_workers, norm_strategy=args.norm_strategy)

    # slim 模式: 更强的 weight_decay
    wd = args.weight_decay if not args.use_slim else 1e-3

    # 无冲突域对齐: 启用 GRL 时自动关闭用户分类辅助任务, 避免梯度冲突
    effective_user_weight = 0.0 if args.use_domain_align else args.user_weight
    if args.use_domain_align and args.user_weight > 0:
        print(f"[Domain Align] Auto-set user_weight=0 (was {args.user_weight}) to avoid conflict with GRL")

    model = build_model(
        args.model, num_classes=4, num_users=num_users,
        emg_channels=80, imu_channels=6, use_imu=args.use_imu,
        user_weight=effective_user_weight,
        use_whitening=args.use_whitening, use_arcface=args.use_arcface,
        use_slim=args.use_slim, use_contra=args.use_contra, use_v02=args.use_v02,
        use_domain_align=args.use_domain_align,
        use_kl_uniform=args.use_kl_uniform, use_supcon=args.use_supcon,
        feat_dim_override=args.feat_dim_override,
        arcface_margin=args.arcface_margin, arcface_scale=args.arcface_scale,
        dropout=args.dropout).to(device)

    # 设置 hard_neg_ratio
    if args.use_contra and hasattr(model, 'contra_criterion'):
        model.contra_criterion.hard_neg_ratio = args.hard_neg_ratio

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # 加载预训练编码器权重
    if args.pretrained:
        if os.path.exists(args.pretrained):
            pretrained_dict = torch.load(args.pretrained, map_location=device)
            model_dict = model.state_dict()
            loaded, skipped = 0, 0
            for k, v in pretrained_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    model_dict[k] = v
                    loaded += 1
                else:
                    skipped += 1
            model.load_state_dict(model_dict)
            print(f"[Pretrained] Loaded {loaded} tensors, skipped {skipped}")
        else:
            print(f"[Pretrained] WARNING: {args.pretrained} not found, training from scratch")

    # 冻结编码器
    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if 'encoder' in name or 'fusion' in name or 'projection' in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Freeze] Encoder frozen, trainable params: {trainable:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    best_acc = 0.0
    best_epoch = 0
    history = []
    whitening_feat_list = []

    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # 方向A: temperature 退火 (0.1 → 0.01)
        if args.temp_anneal and args.use_contra and hasattr(model, 'contra_criterion'):
            temp = 0.1 - (0.1 - 0.01) * (epoch - 1) / max(args.epochs - 1, 1)
            model.contra_criterion.set_temperature(temp)

        # 对比学习使用 GroupSampler
        if args.use_contra:
            train_labels = train_loader.dataset.Y.numpy()
            train_users = train_loader.dataset.U.numpy()
            sampler = GestureGroupSampler(train_labels, train_users, args.batch)
            train_loader = DataLoader(train_loader.dataset, batch_sampler=sampler,
                                      num_workers=args.num_workers, pin_memory=True)

        whitening_feat_list.clear()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            use_contra=args.use_contra, contra_weight=args.contra_weight,
            user_weight=effective_user_weight,
            use_domain_align=args.use_domain_align, domain_weight=args.domain_weight,
            use_kl_uniform=args.use_kl_uniform, kl_weight=args.kl_weight,
            use_mixup=args.use_mixup, use_aug=args.use_aug,
            update_whitening=args.use_whitening,
            whitening_feat_list=whitening_feat_list)
        scheduler.step()

        # 更新白化全局统计量
        if args.use_whitening and len(whitening_feat_list) > 0:
            model.update_whitening_stats(whitening_feat_list)

        test_acc = evaluate(model, test_loader, device)

        history.append({
            'epoch': epoch, 'loss': train_loss,
            'train_acc': train_acc, 'zeroshot_acc': test_acc})

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
              f"| time={dt:.1f}s{tag}")

        if epoch - best_epoch >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nFinal Zero-Shot Accuracy: {best_acc:.4f}")

    result = {
        'model': args.model, 'use_imu': args.use_imu,
        'norm_strategy': args.norm_strategy,
        'use_whitening': args.use_whitening, 'use_arcface': args.use_arcface,
        'use_slim': args.use_slim, 'use_contra': args.use_contra,
        'use_v02': args.use_v02, 'use_domain_align': args.use_domain_align,
        'pretrained': args.pretrained, 'freeze_encoder': args.freeze_encoder,
        'config': vars(args), 'zeroshot_acc': best_acc,
        'history': history, 'early_stop_epoch': epoch,
        'num_params': n_params,
    }
    with open(os.path.join(args.save_dir, f"results_{suffix}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {args.save_dir}/results_{suffix}.json")


if __name__ == "__main__":
    main()
