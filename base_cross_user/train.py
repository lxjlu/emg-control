# -*- coding: utf-8 -*-
"""
训练脚本: 支持 CNN-1D 和 Transformer baseline
关键优化:
1. DataLoader 使用 num_workers>0 + pin_memory 预取数据到GPU
2. non_blocking=True 让CPU-GPU传输异步
3. 使用 torch.cuda.amp 混合精度加速
4. 梯度裁剪防止训练不稳定
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

from data_loader import load_all_users, build_louo_loaders
from models import build_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
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


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, nb = 0.0, 0
    for batch in loader:
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="cnn1d", choices=["cnn1d", "transformer"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--save_dir", type=str, default="./results")
    args = ap.parse_args()

    device = get_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print(f"Model: {args.model} | Device: {device}")
    print("=" * 60)

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

    model = build_model(args.model, in_channels=8, num_classes=4).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_acc = 0.0
    history = []

    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        scheduler.step()
        epoch_time = time.time() - t0

        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            train_acc = evaluate(model, train_loader, device)
            test_acc = evaluate(model, test_loader, device)

            history.append({
                "epoch": ep, "loss": loss,
                "train_acc": train_acc, "zeroshot_acc": test_acc,
            })

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(),
                           os.path.join(args.save_dir, f"best_{args.model}.pt"))

            print(f"ep{ep:03d} | loss={loss:.4f} | "
                  f"train={train_acc:.4f} zeroshot={test_acc:.4f} "
                  f"best={best_acc:.4f} | time={epoch_time:.1f}s")
        else:
            print(f"ep{ep:03d} | loss={loss:.4f} | time={epoch_time:.1f}s")

    model.load_state_dict(torch.load(
        os.path.join(args.save_dir, f"best_{args.model}.pt")))
    final_acc = evaluate(model, test_loader, device)
    print(f"\nFinal Zero-Shot Accuracy: {final_acc:.4f}")

    with open(os.path.join(args.save_dir, f"results_{args.model}.json"), "w") as f:
        json.dump({
            "model": args.model,
            "config": vars(args),
            "info": info,
            "zeroshot_acc": final_acc,
            "history": history,
        }, f, indent=2)

    print(f"Results saved to {args.save_dir}/results_{args.model}.json")


if __name__ == "__main__":
    main()