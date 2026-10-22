# -*- coding: utf-8 -*-
"""
训练脚本 v2: 支持早停、DANN、对比学习、多模态
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import load_all_users, build_louo_loaders
from data_loader_v2 import load_h5_data, build_louo_loaders_v2
from models_v2 import (build_model, UserInvariantModel, DisentangledModel,
                       IMUAnchoredAttention, IMUAnchoredDecomposition,
                       IMUCrossModalContrastive, mutual_information_loss,
                       PrototypeContrastive, MemoryBankContrastive,
                       UserNormContrastive, DualBranchContrastive,
                       MultiTaskModel, SelfSupervisedModel,
                       DomainGeneralizationModel)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, use_dann=False, use_imu=False, use_disentangled=False, use_imu_anchor=False, use_new_contra=False, use_multi_task=False, use_domain_gen=False):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        if use_multi_task or use_new_contra or use_imu_anchor or use_domain_gen:
            x_emg, x_imu, y, _ = batch
            x_emg = x_emg.to(device, non_blocking=True)
            x_imu = x_imu.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x_emg, x_imu, stage='eval')
        elif use_disentangled:
            if use_imu:
                x_emg, x_imu, y, _ = batch
                x_emg = x_emg.to(device, non_blocking=True)
                x_imu = x_imu.to(device, non_blocking=True)
            else:
                if len(batch) == 3:
                    x, y, _ = batch
                else:
                    x, y = batch
                x_emg = x.to(device, non_blocking=True)
                x_imu = None
            y = y.to(device, non_blocking=True)
            logits = model(x_emg, x_imu, stage='eval')
        elif use_dann and use_imu:
            x_emg, x_imu, y, _ = batch
            x_emg = x_emg.to(device, non_blocking=True)
            x_imu = x_imu.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x_emg, x_imu, stage='eval')
        elif use_dann and not use_imu:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x, stage='eval')
        else:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += len(y)
    return correct / total if total > 0 else 0.0


def cross_user_contrastive_loss(features, labels, users, temperature=0.1):
    """
    跨用户对比损失: 同手势不同用户 = 正样本
    """
    features = F.normalize(features, dim=1)
    sim = torch.matmul(features, features.T) / temperature

    labels = labels.view(-1, 1)
    users = users.view(-1, 1)
    pos_mask = ((labels == labels.T) & (users != users.T)).float()
    pos_mask.fill_diagonal_(0)
    neg_mask = (labels != labels.T).float()

    sim_max = sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim - sim_max)

    pos_sum = (exp_sim * pos_mask).sum(dim=1)
    neg_sum = (exp_sim * neg_mask).sum(dim=1)

    has_pos = pos_mask.sum(dim=1) > 0
    loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8) + 1e-8)
    loss = (loss * has_pos.float()).sum() / (has_pos.float().sum() + 1e-8)
    return loss


def train_one_epoch(model, loader, optimizer, criterion, device, scaler,
                    use_dann=False, use_imu=False, num_users=230, grl_lambda=1.0,
                    use_contrastive=False, contrastive_weight=0.5,
                    use_disentangled=False, mi_weight=1.0,
                    use_imu_anchor=False, align_weight=1.0, indep_weight=0.1, recon_weight=0.5,
                    use_proto_contra=False, use_memory_contra=False,
                    use_user_norm_contra=False, use_dual_branch_contra=False,
                    use_multi_task=False, user_weight=0.3, use_domain_gen=False, mmd_weight=1.0):
    model.train()
    total_loss, nb = 0.0, 0

    for batch in loader:
        # 新方案: 域泛化
        if use_domain_gen:
            if use_imu:
                x_emg, x_imu, y, u = batch
                x_emg = x_emg.to(device, non_blocking=True)
                x_imu = x_imu.to(device, non_blocking=True)
            else:
                if len(batch) == 3:
                    x, y, u = batch
                else:
                    x, y = batch
                    u = torch.zeros(x.size(0), dtype=torch.long, device=device)
                x_emg = x.to(device, non_blocking=True)
                x_imu = None
            y = y.to(device, non_blocking=True)
            u = u.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                out = model(x_emg, x_imu, label=y, user=u, stage='train')
                loss_gesture = criterion(out['gesture_logits'], y)
                loss_user = F.cross_entropy(out['user_logits'], u)
                loss_mmd = out['mmd_loss']
                loss = loss_gesture + user_weight * loss_user + mmd_weight * loss_mmd
        # 新方案: 多任务学习
        elif use_multi_task:
            if use_imu:
                x_emg, x_imu, y, u = batch
                x_emg = x_emg.to(device, non_blocking=True)
                x_imu = x_imu.to(device, non_blocking=True)
            else:
                if len(batch) == 3:
                    x, y, u = batch
                else:
                    x, y = batch
                    u = torch.zeros(x.size(0), dtype=torch.long, device=device)
                x_emg = x.to(device, non_blocking=True)
                x_imu = None
            y = y.to(device, non_blocking=True)
            u = u.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                out = model(x_emg, x_imu, label=y, user=u, stage='train')
                loss_gesture = criterion(out['gesture_logits'], y)
                loss_user = F.cross_entropy(out['user_logits'], u)
                loss = loss_gesture + user_weight * loss_user
        # 新方案: 原型对比、记忆库对比、用户归一化对比、双分支对比
        elif use_proto_contra or use_memory_contra or use_user_norm_contra or use_dual_branch_contra:
            if use_imu:
                x_emg, x_imu, y, u = batch
                x_emg = x_emg.to(device, non_blocking=True)
                x_imu = x_imu.to(device, non_blocking=True)
            else:
                if len(batch) == 3:
                    x, y, u = batch
                else:
                    x, y = batch
                x_emg = x.to(device, non_blocking=True)
                x_imu = None
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                out = model(x_emg, x_imu, label=y, stage='train')
                loss_gesture = criterion(out['gesture_logits'], y)
                loss = loss_gesture

                # 原型对比损失
                if 'proto_loss' in out:
                    loss = loss + contrastive_weight * out['proto_loss']

                # 记忆库对比损失
                if 'memory_loss' in out:
                    loss = loss + contrastive_weight * out['memory_loss']

                # 用户归一化对比损失
                if use_user_norm_contra and 'contrastive_loss' in out:
                    loss = loss + contrastive_weight * out['contrastive_loss']

                # 双分支对比损失
                if use_dual_branch_contra and 'contrastive_loss' in out:
                    loss = loss + contrastive_weight * out['contrastive_loss']
        elif use_imu_anchor:
            x_emg, x_imu, y, u = batch
            x_emg = x_emg.to(device, non_blocking=True)
            x_imu = x_imu.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            u = u.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                out = model(x_emg, x_imu, label=y, stage='train')
                loss_gesture = criterion(out['gesture_logits'], y)
                loss = loss_gesture

                # 对齐损失
                if 'align_loss' in out:
                    loss = loss + align_weight * out['align_loss']

                # 独立性损失
                if 'indep_loss' in out:
                    loss = loss + indep_weight * out['indep_loss']

                # 重构损失
                if 'recon_loss' in out:
                    loss = loss + recon_weight * out['recon_loss']

                # 跨模态对比损失
                if 'contrastive_loss' in out:
                    loss = loss + contrastive_weight * out['contrastive_loss']

                # DANN用户分类
                if use_dann and 'user_logits' in out:
                    loss_user = F.cross_entropy(out['user_logits'], u)
                    loss = loss + grl_lambda * loss_user

                # 跨用户对比学习
                if use_contrastive and out['gesture_feat'].size(0) > 16:
                    loss_contra = cross_user_contrastive_loss(
                        out['gesture_feat'], y, u, temperature=0.1)
                    loss = loss + contrastive_weight * loss_contra
        elif use_disentangled:
            if use_imu:
                x_emg, x_imu, y, u = batch
                x_emg = x_emg.to(device, non_blocking=True)
                x_imu = x_imu.to(device, non_blocking=True)
            else:
                if len(batch) == 3:
                    x, y, u = batch
                else:
                    x, y = batch
                    u = torch.zeros(x.size(0), dtype=torch.long, device=device)
                x_emg = x.to(device, non_blocking=True)
                x_imu = None
            y = y.to(device, non_blocking=True)
            u = u.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                out = model(x_emg, x_imu, label=y, stage='train')
                loss_gesture = criterion(out['gesture_logits'], y)
                loss_user = F.cross_entropy(out['user_logits'], u)
                loss_mi = mutual_information_loss(out['gesture_feat'], out['user_feat'])
                loss = loss_gesture + loss_user + mi_weight * loss_mi

                if use_contrastive and out['gesture_feat'].size(0) > 16:
                    loss_contra = cross_user_contrastive_loss(
                        out['gesture_feat'], y, u, temperature=0.1)
                    loss = loss + contrastive_weight * loss_contra
        elif use_dann and use_imu:
            x_emg, x_imu, y, u = batch
            x_emg = x_emg.to(device, non_blocking=True)
            x_imu = x_imu.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            u = u.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                out = model(x_emg, x_imu, label=y, stage='train')
                loss_gesture = criterion(out['gesture_logits'], y)
                loss_user = F.cross_entropy(out['user_logits'], u)
                loss = loss_gesture + grl_lambda * loss_user

                if use_contrastive and out['feat'].size(0) > 16:
                    loss_contra = cross_user_contrastive_loss(
                        out['feat'], y, u, temperature=0.1)
                    loss = loss + contrastive_weight * loss_contra
        elif use_dann and not use_imu:
            if len(batch) == 3:
                x, y, u = batch
            else:
                x, y = batch
                u = None
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if u is not None:
                u = u.to(device, non_blocking=True)
            else:
                u = torch.zeros(x.size(0), dtype=torch.long, device=device)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                out = model(x, x, label=y, stage='train')
                loss_gesture = criterion(out['gesture_logits'], y)
                loss_user = F.cross_entropy(out['user_logits'], u)
                loss = loss_gesture + grl_lambda * loss_user

                if use_contrastive and out['feat'].size(0) > 16:
                    loss_contra = cross_user_contrastive_loss(
                        out['feat'], y, u, temperature=0.1)
                    loss = loss + contrastive_weight * loss_contra
        else:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

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
        return self.early_stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="cnn1d",
                    choices=["cnn1d", "transformer", "dann", "disentangled",
                             "imu_attn", "imu_decompose", "imu_crossmodal",
                             "proto_contra", "memory_contra", "user_norm_contra", "dual_branch_contra",
                             "multi_task", "domain_gen"])
    ap.add_argument("--use_imu", action="store_true",
                    help="使用EMG+IMU多模态 (仅dann/disentangled/imu_*模型)")
    ap.add_argument("--use_contrastive", action="store_true",
                    help="使用跨用户对比学习")
    ap.add_argument("--contrastive_weight", type=float, default=0.5)
    ap.add_argument("--mi_weight", type=float, default=1.0,
                    help="互信息损失权重 (仅disentangled模型)")
    ap.add_argument("--align_weight", type=float, default=1.0,
                    help="IMU对齐损失权重 (仅imu_*模型)")
    ap.add_argument("--indep_weight", type=float, default=0.1,
                    help="IMU独立性损失权重 (仅imu_*模型)")
    ap.add_argument("--recon_weight", type=float, default=0.5,
                    help="重构损失权重 (仅imu_decompose模型)")
    ap.add_argument("--user_weight", type=float, default=0.3,
                    help="用户分类损失权重 (仅multi_task/domain_gen模型)")
    ap.add_argument("--mmd_weight", type=float, default=1.0,
                    help="MMD域对齐损失权重 (仅domain_gen模型)")
    ap.add_argument("--norm_strategy", type=str, default="global",
                    choices=["global", "none", "per_user"],
                    help="归一化策略: global=全局Z-Score, none=无归一化, per_user=逐用户Z-Score")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=15,
                    help="早停patience")
    ap.add_argument("--grl_lambda", type=float, default=1.0,
                    help="域对抗梯度反转强度")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="Dropout率 (0.0-0.5)")
    ap.add_argument("--save_dir", type=str, default="./results")
    ap.add_argument("--exp_name", type=str, default=None,
                    help="自定义实验名(用于结果文件命名)，不指定则自动生成")
    args = ap.parse_args()

    device = get_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.save_dir, exist_ok=True)

    use_dann = (args.model == "dann")
    use_disentangled = (args.model == "disentangled")
    use_imu_anchor = args.model in ["imu_attn", "imu_decompose", "imu_crossmodal"]
    use_proto_contra = (args.model == "proto_contra")
    use_memory_contra = (args.model == "memory_contra")
    use_user_norm_contra = (args.model == "user_norm_contra")
    use_dual_branch_contra = (args.model == "dual_branch_contra")
    use_multi_task = (args.model == "multi_task")
    use_domain_gen = (args.model == "domain_gen")
    use_new_contra = use_proto_contra or use_memory_contra or use_user_norm_contra or use_dual_branch_contra
    if args.exp_name:
        suffix = args.exp_name
    else:
        suffix = f"{args.model}"
        if args.use_imu:
            suffix += "_imu"
        if args.use_contrastive:
            suffix += "_contra"

    print("=" * 60)
    print(f"Model: {args.model} | Device: {device}")
    print(f"Multi-modal: {args.use_imu} | Contrastive: {args.use_contrastive}")
    if use_disentangled:
        print(f"MI Weight: {args.mi_weight}")
    if use_imu_anchor:
        print(f"Align Weight: {args.align_weight} | Indep Weight: {args.indep_weight} | Recon Weight: {args.recon_weight}")
    if use_multi_task:
        print(f"User Weight: {args.user_weight}")
    if use_domain_gen:
        print(f"MMD Weight: {args.mmd_weight} | User Weight: {args.user_weight}")
    print(f"Early stopping patience: {args.patience}")
    print("=" * 60)

    # 加载数据
    if args.use_imu and (use_dann or use_disentangled or use_imu_anchor or use_new_contra or use_multi_task or use_domain_gen):
        X_emg, X_imu, Y, U, num_users, user2id = load_h5_data()
        if X_emg is None:
            print("ERROR: No data loaded!")
            return
        print(f"Total: {len(X_emg)} samples, {num_users} users")
        train_loader, test_loader, info, norm_stats = build_louo_loaders_v2(
            X_emg, X_imu, Y, U, test_ratio=args.test_ratio,
            batch_size=args.batch, num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"), norm_strategy=args.norm_strategy)
    else:
        X, Y, U, num_users, user2id = load_all_users()
        if X is None:
            print("ERROR: No data loaded!")
            return
        print(f"Total: {len(X)} samples, {num_users} users")
        train_loader, test_loader, info, mean, std = build_louo_loaders(
            X, Y, U, test_ratio=args.test_ratio, batch_size=args.batch,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    print(f"Train: {info['n_train_users']} users, {info['train_samples']} samples")
    print(f"Test:  {info['n_test_users']} users, {info['test_samples']} samples")

    # 构建模型
    if use_dann:
        if args.use_imu:
            model = UserInvariantModel(
                num_classes=4, num_users=num_users,
                emg_channels=80, imu_channels=6,
                grl_lambda=args.grl_lambda, use_imu=True,
                dropout=args.dropout).to(device)
        else:
            model = UserInvariantModel(
                num_classes=4, num_users=num_users,
                emg_channels=8, imu_channels=6,
                grl_lambda=args.grl_lambda, use_imu=False,
                dropout=args.dropout).to(device)
    elif use_disentangled:
        if args.use_imu:
            model = DisentangledModel(
                num_classes=4, num_users=num_users,
                emg_channels=80, imu_channels=6,
                use_imu=True, dropout=args.dropout).to(device)
        else:
            model = DisentangledModel(
                num_classes=4, num_users=num_users,
                emg_channels=8, imu_channels=6,
                use_imu=False, dropout=args.dropout).to(device)
    elif use_imu_anchor:
        if args.model == "imu_attn":
            model = IMUAnchoredAttention(
                num_classes=4, num_users=num_users,
                emg_channels=80, imu_channels=6,
                use_dann=args.use_contrastive,
                align_weight=args.align_weight,
                indep_weight=args.indep_weight,
                dropout=args.dropout).to(device)
        elif args.model == "imu_decompose":
            model = IMUAnchoredDecomposition(
                num_classes=4, num_users=num_users,
                emg_channels=80, imu_channels=6,
                use_dann=args.use_contrastive,
                align_weight=args.align_weight,
                indep_weight=args.indep_weight,
                recon_weight=args.recon_weight,
                dropout=args.dropout).to(device)
        elif args.model == "imu_crossmodal":
            model = IMUCrossModalContrastive(
                num_classes=4, num_users=num_users,
                emg_channels=80, imu_channels=6,
                use_dann=args.use_contrastive,
                contrastive_weight=args.contrastive_weight,
                dropout=args.dropout).to(device)
    elif use_proto_contra:
        model = PrototypeContrastive(
            num_classes=4, num_users=num_users,
            emg_channels=80, imu_channels=6,
            use_imu=args.use_imu,
            temperature=0.1,
            contrastive_weight=args.contrastive_weight,
            dropout=args.dropout).to(device)
    elif use_memory_contra:
        model = MemoryBankContrastive(
            num_classes=4, num_users=num_users,
            emg_channels=80, imu_channels=6,
            use_imu=args.use_imu,
            memory_size=10000,
            temperature=0.1,
            contrastive_weight=args.contrastive_weight,
            momentum=0.5,
            dropout=args.dropout).to(device)
    elif use_user_norm_contra:
        model = UserNormContrastive(
            num_classes=4, num_users=num_users,
            emg_channels=80, imu_channels=6,
            use_imu=args.use_imu,
            temperature=0.1,
            contrastive_weight=args.contrastive_weight,
            dropout=args.dropout).to(device)
    elif use_dual_branch_contra:
        model = DualBranchContrastive(
            num_classes=4, num_users=num_users,
            emg_channels=80, imu_channels=6,
            use_imu=args.use_imu,
            shared_encoder=True,
            temperature=0.1,
            contrastive_weight=args.contrastive_weight,
            dropout=args.dropout).to(device)
    elif use_multi_task:
        model = MultiTaskModel(
            num_classes=4, num_users=num_users,
            emg_channels=80, imu_channels=6,
            use_imu=args.use_imu,
            user_weight=args.user_weight,
            dropout=args.dropout).to(device)
    elif use_domain_gen:
        model = DomainGeneralizationModel(
            num_classes=4, num_users=num_users,
            emg_channels=80, imu_channels=6,
            use_imu=args.use_imu,
            user_weight=args.user_weight,
            mmd_weight=args.mmd_weight,
            dropout=args.dropout).to(device)
    else:
        model = build_model(args.model, in_channels=8, num_classes=4, dropout=args.dropout).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.002)

    best_acc = 0.0
    history = []

    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()

        # GRL lambda 随epoch增大 (warmup)
        if use_dann:
            current_grl_lambda = args.grl_lambda * min(1.0, ep / 10)
            model.grl.lambda_ = current_grl_lambda

        loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            use_dann=use_dann, use_imu=args.use_imu, num_users=num_users,
            grl_lambda=current_grl_lambda if use_dann else 1.0,
            use_contrastive=args.use_contrastive,
            contrastive_weight=args.contrastive_weight,
            use_disentangled=use_disentangled,
            mi_weight=args.mi_weight,
            use_imu_anchor=use_imu_anchor,
            align_weight=args.align_weight,
            indep_weight=args.indep_weight,
            recon_weight=args.recon_weight,
            use_proto_contra=use_proto_contra,
            use_memory_contra=use_memory_contra,
            use_user_norm_contra=use_user_norm_contra,
            use_dual_branch_contra=use_dual_branch_contra,
            use_multi_task=use_multi_task,
            use_domain_gen=use_domain_gen,
            mmd_weight=args.mmd_weight,
            user_weight=args.user_weight)
        scheduler.step()
        epoch_time = time.time() - t0

        train_acc = evaluate(model, train_loader, device, use_dann=use_dann, use_imu=args.use_imu, use_disentangled=use_disentangled, use_imu_anchor=use_imu_anchor, use_new_contra=use_new_contra, use_multi_task=use_multi_task, use_domain_gen=use_domain_gen)
        test_acc = evaluate(model, test_loader, device, use_dann=use_dann, use_imu=args.use_imu, use_disentangled=use_disentangled, use_imu_anchor=use_imu_anchor, use_new_contra=use_new_contra, use_multi_task=use_multi_task, use_domain_gen=use_domain_gen)

        history.append({
            "epoch": ep, "loss": loss,
            "train_acc": train_acc, "zeroshot_acc": test_acc,
        })

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f"best_{suffix}.pt"))

        status = "NEW BEST" if test_acc >= best_acc - 0.002 else ""
        print(f"ep{ep:03d} | loss={loss:.4f} | "
              f"train={train_acc:.4f} zeroshot={test_acc:.4f} "
              f"best={best_acc:.4f} | time={epoch_time:.1f}s {status}")

        if early_stopping(test_acc):
            print(f"\nEarly stopping at epoch {ep}")
            break

    # 加载最佳模型
    model.load_state_dict(torch.load(
        os.path.join(args.save_dir, f"best_{suffix}.pt")))
    final_acc = evaluate(model, test_loader, device, use_dann=use_dann, use_imu=args.use_imu, use_disentangled=use_disentangled, use_imu_anchor=use_imu_anchor, use_new_contra=use_new_contra, use_multi_task=use_multi_task, use_domain_gen=use_domain_gen)
    print(f"\nFinal Zero-Shot Accuracy: {final_acc:.4f}")

    with open(os.path.join(args.save_dir, f"results_{suffix}.json"), "w") as f:
        json.dump({
            "model": args.model,
            "use_imu": args.use_imu,
            "use_contrastive": args.use_contrastive,
            "norm_strategy": args.norm_strategy,
            "config": vars(args),
            "info": info,
            "zeroshot_acc": final_acc,
            "history": history,
            "early_stop_epoch": len(history),
        }, f, indent=2)

    print(f"Results saved to {args.save_dir}/results_{suffix}.json")


if __name__ == "__main__":
    main()