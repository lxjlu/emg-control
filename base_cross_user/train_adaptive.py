# -*- coding: utf-8 -*-
"""
train_adaptive.py - 三种"用户自适应"训练路径的对照实验 (600 点窗口)

统一在 600pt 窗口 + 同一编码器初始化 (best_v4_p600_allusers_nouser.pt) 下,
只替换/叠加"分类目标", 回答: 哪种训练策略对 zero-shot 有用?

--path baseline : 现有 v4 (全局 CE + 跨用户 contra)          ← 对照
--path A        : + 用户自适应原型 NCM (batch 内按用户建类中心, transductive)
--path B        : 纯 episodic ProtoNet (伪新用户 support→query, meta 目标)
--path C        : 可适配残差 FiLM (per-user 特征均值 EMA → γβ 调制特征)

zero-shot 评测 (各自协议):
  baseline / A : 全局头 argmax (test loader)
  B            : 训练集构建全局类原型 → 最近类中心 (NCM)
  C            : 先用该用户无标签样本估特征均值 → FiLM 调制 → 全局头
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
from models_v4 import build_model as build_model_v4
from data_loader_v2 import load_h5_data, build_louo_loaders_v2
from train_v3 import GestureGroupSampler


def norm_feat(f):
    return F.normalize(f, dim=1)


# ============ 用户/类索引 ============
class UserIndex:
    def __init__(self, U, Y):
        self.U = U
        self.Y = Y
        self.users = np.unique(U)
        # per user per class indices
        self.idx_uc = {}
        for u in self.users:
            mu = U == u
            self.idx_uc[u] = {}
            for c in range(int(Y.max()) + 1):
                self.idx_uc[u][c] = np.where(mu & (Y == c))[0]

    def block(self, rng, n_users, per_class):
        """抽 n_users 个用户, 每用户每类 per_class 条, 返回索引数组"""
        sel_users = rng.choice(self.users, size=min(n_users, len(self.users)),
                               replace=False)
        idx = []
        for u in sel_users:
            for c in range(len(self.idx_uc[u])):
                pool = self.idx_uc[u][c]
                if len(pool) == 0:
                    continue
                n = min(per_class, len(pool))
                idx.append(rng.choice(pool, size=n, replace=False))
        if len(idx) == 0:
            return np.array([], dtype=np.int64), sel_users
        return np.concatenate(idx), sel_users


# ============ C: FiLM 条件调制 ============
class FiLM(nn.Module):
    """per-dim scale/shift, 输入用户特征均值向量"""
    def __init__(self, feat_dim):
        super().__init__()
        self.gamma = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.Tanh())
        self.beta = nn.Sequential(nn.Linear(feat_dim, feat_dim))
        # 初始化 beta≈0
        nn.init.zeros_(self.beta[0].bias)
        self.gamma[0].bias.data.fill_(0.0)

    def forward(self, z, m_u):
        g = 1.0 + 0.2 * self.gamma(m_u)
        b = self.beta(m_u)
        return z * g + b


def enc_feat(model, xe, xi):
    ef = model.emg_encoder(xe)
    imf = model.imu_encoder(xi) if model.imu_encoder is not None else None
    fused = model.fusion(ef, imf) if imf is not None else ef
    return model.projection(fused)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, default="baseline",
                    choices=["baseline", "A", "B", "C"])
    ap.add_argument("--cache_dir", type=str, default="./data_cache_v2/long_p600_s60")
    ap.add_argument("--init_ckpt", type=str,
                    default="./results/best_v4_p600_allusers_nouser.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--nu", type=int, default=8, help="A/B 每步用户数")
    ap.add_argument("--exp_name", type=str, default=None)
    ap.add_argument("--save_dir", type=str, default="./results")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suffix = args.exp_name or f"adapt_{args.path}"
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 70)
    print(f"Adaptive Training Path: {args.path} | 600pt | seed {args.seed}")
    print("=" * 70)

    X_emg, X_imu, Y, U, num_users, _ = load_h5_data(cache_dir=args.cache_dir)
    train_loader, test_loader, info, _ = build_louo_loaders_v2(
        X_emg, X_imu, Y, U, test_ratio=0.2, batch_size=args.batch,
        num_workers=args.num_workers, norm_strategy="per_user")
    train_ds = train_loader.dataset
    test_ds = test_loader.dataset

    # 模型
    model = build_model_v4(
        "unified", num_classes=4, num_users=num_users,
        emg_channels=80, imu_channels=6, use_imu=True,
        use_whitening=False, use_contra=True,
        use_grl=False, use_mmd=False, feat_dim=128, dropout=0.0).to(device)
    if args.init_ckpt and os.path.exists(args.init_ckpt):
        st = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        ms = model.state_dict()
        lo = sk = 0
        for k, v in st.items():
            if k in ms and ms[k].shape == v.shape:
                ms[k] = v
                lo += 1
            else:
                sk += 1
        model.load_state_dict(ms)
        print(f"[Init] loaded {lo}, skipped {sk} from {args.init_ckpt}")

    film = FiLM(128).to(device) if args.path == "C" else None
    params = list(model.parameters())
    if film is not None:
        params += list(film.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # train data as tensors on device? 保持 CPU 索引后搬 GPU.
    if args.path in ("A", "B"):
        Ut = train_ds.U.numpy()
        Yt = train_ds.Y.numpy()
        uidx = UserIndex(Ut, Yt)
    # C: 维护每个(训练)用户的 EMA 特征均值
    ema_mu = {}  # user -> running mean (128,)

    def batch_to_dev(batch):
        x_emg, x_imu, y, u = batch
        return (x_emg.to(device), x_imu.to(device), y.to(device), u.to(device))

    # ---------- 训练 ----------
    best_acc, best_ep = 0.0, 0
    history = []
    print("Training...")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        if film is not None:
            film.train()
        tot_loss, nb = 0.0, 0
        correct = total = 0

        if args.path == "B":
            # ---- episodic ProtoNet: 每 episode 抽 nu 个"伪新用户" ----
            # 每个用户内每类: 前 ns 条作 support, 其后 nq 条作 query
            # 类原型 = support 中同类样本跨用户求均值 (类共享)
            ds = train_ds
            ns, nq = 2, 4
            rng = np.random.RandomState(epoch * 1000 + args.seed)
            n_episodes = 120
            tot_loss, nb = 0.0, 0
            for _ in range(n_episodes):
                users_ep = rng.choice(uidx.users,
                                      size=min(args.nu, len(uidx.users)),
                                      replace=False)
                sup_idx, qry_idx = [], []
                for uu in users_ep:
                    for c in range(4):
                        pool = uidx.idx_uc[int(uu)][c]
                        if len(pool) < ns + 1:
                            continue
                        rng.shuffle(pool)
                        sup_idx.extend(pool[:ns].tolist())
                        qry_idx.extend(pool[ns:ns + nq].tolist())
                if len(sup_idx) == 0 or len(qry_idx) == 0:
                    continue
                xe_s = ds.X_emg[sup_idx].to(device)
                xi_s = ds.X_imu[sup_idx].to(device)
                ye_s = ds.Y[sup_idx].to(device)
                xe_q = ds.X_emg[qry_idx].to(device)
                xi_q = ds.X_imu[qry_idx].to(device)
                yq = ds.Y[qry_idx].to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    zs = norm_feat(enc_feat(model, xe_s, xi_s))
                    protos = torch.stack([zs[ye_s == c].mean(0)
                                          for c in range(4)])
                    zq = norm_feat(enc_feat(model, xe_q, xi_q))
                    logits = zq @ protos.t() * 15.0
                    loss_b = F.cross_entropy(logits, yq)
                optimizer.zero_grad()
                scaler.scale(loss_b).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                tot_loss += loss_b.item()
                nb += 1
            train_acc = 0.0  # episodic 无单步 train acc

        else:
            # ---- 常规 loader (GroupSampler 保障 batch 多样) ----
            labels = train_ds.Y.numpy()
            users = train_ds.U.numpy()
            sampler = GestureGroupSampler(labels, users, args.batch)
            loader = DataLoader(train_ds, batch_sampler=sampler,
                                num_workers=args.num_workers, pin_memory=True)
            rng = np.random.RandomState(epoch * 1000 + args.seed)
            for batch in loader:
                x_emg, x_imu, y, u = batch_to_dev(batch)
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    z = enc_feat(model, x_emg, x_imu)
                    loss = F.cross_entropy(model.gesture_classifier(z), y)
                    loss = loss + 0.5 * model.contra_criterion(norm_feat(z), y, u)

                    if args.path == "A":
                        # 用户自适应 NCM (transductive, 权重 0.5)
                        loss_a = torch.tensor(0.0, device=device)
                        cnt = 0
                        for uu in torch.unique(u):
                            um = u == uu
                            zu = z[um]
                            yu = y[um]
                            valid = all((yu == c).sum() > 0 for c in range(4))
                            if not valid:
                                continue
                            centers = []
                            for c in range(4):
                                centers.append(zu[yu == c].mean(0))
                            centers = torch.stack(centers)
                            d = torch.cdist(zu, centers)
                            loss_a = loss_a + F.cross_entropy(-d * 2.0, yu)
                            cnt += 1
                        if cnt > 0:
                            loss = loss + 0.5 * loss_a / cnt

                    elif args.path == "C":
                        # per-user 均值 EMA
                        with torch.no_grad():
                            un = u.cpu().numpy()
                            for uu in np.unique(un):
                                m = z[un == uu].mean(0).detach()
                                if uu in ema_mu:
                                    ema_mu[uu] = 0.9 * ema_mu[uu] + 0.1 * m
                                else:
                                    ema_mu[uu] = m
                        mu_batch = torch.stack(
                            [ema_mu[int(uv)] for uv in un]).to(device)
                        z = film(z, mu_batch)
                        loss = F.cross_entropy(model.gesture_classifier(z), y)
                        loss = loss + 0.5 * model.contra_criterion(norm_feat(z), y, u)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                pred = model.gesture_classifier(z.float()).argmax(1)
                correct += (pred == y).sum().item()
                total += len(y)
                tot_loss += loss.item()
                nb += 1
            train_acc = correct / max(total, 1)
        scheduler.step()

        # ---------- zero-shot 评测 ----------
        model.eval()
        if film is not None:
            film.eval()
        if args.path == "C":
            # 先估 test 用户特征均值, 再带 FiLM 分类
            with torch.no_grad():
                est = {}
                for batch in test_loader:
                    x_emg, x_imu, _, u = batch
                    z = enc_feat(model, x_emg.to(device), x_imu.to(device)).cpu()
                    un = u.numpy()
                    for uu in np.unique(un):
                        mm = z[un == uu].mean(0)
                        est[int(uu)] = mm
            acc = 0
            tot = 0
            with torch.no_grad():
                for batch in test_loader:
                    x_emg, x_imu, y, u = batch
                    z = enc_feat(model, x_emg.to(device), x_imu.to(device)).cpu()
                    un = u.numpy()
                    mu_batch = torch.stack([est[int(uv)] for uv in un])
                    zf = film(z.to(device), mu_batch.to(device))
                    pred = model.gesture_classifier(zf).argmax(1).cpu()
                    acc += (pred == y).sum().item()
                    tot += len(y)
            test_acc = acc / max(tot, 1)
        elif args.path == "B":
            # 训练集全局类原型 → NCM
            feats, labs = [], []
            with torch.no_grad():
                for batch in train_loader:
                    x_emg, x_imu, y, _ = batch
                    z = enc_feat(model, x_emg.to(device), x_imu.to(device))
                    feats.append(norm_feat(z).cpu())
                    labs.append(y)
            Z = torch.cat(feats)
            Ly = torch.cat(labs)
            protos = torch.stack([Z[Ly == c].mean(0) for c in range(4)])
            acc = 0
            tot = 0
            with torch.no_grad():
                for batch in test_loader:
                    x_emg, x_imu, y, _ = batch
                    z = enc_feat(model, x_emg.to(device), x_imu.to(device))
                    z = norm_feat(z).cpu()
                    sim = z @ protos.t()
                    pred = sim.argmax(1)
                    acc += (pred == y).sum().item()
                    tot += len(y)
            test_acc = acc / max(tot, 1)
        else:
            test_acc = 0
            tot = 0
            with torch.no_grad():
                for batch in test_loader:
                    x_emg, x_imu, y, _ = batch
                    z = enc_feat(model, x_emg.to(device), x_imu.to(device))
                    pred = model.gesture_classifier(z).argmax(1)
                    test_acc += (pred.cpu() == y).sum().item()
                    tot += len(y)
            test_acc = test_acc / max(tot, 1)

        history.append({'epoch': epoch, 'loss': tot_loss / max(nb, 1),
                        'train_acc': train_acc, 'zeroshot': test_acc})
        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            best_ep = epoch
            save = {'model': model.state_dict()}
            if film is not None:
                save['film'] = film.state_dict()
            torch.save(save, os.path.join(args.save_dir, f"best_{suffix}.pt"))

        dt = time.time() - t0
        tag = " NEW BEST" if is_best else ""
        print(f"ep{epoch:03d} | loss={tot_loss/max(nb,1):.4f} | "
              f"train={train_acc if not isinstance(train_acc, float) else train_acc:.4f} "
              f"zeroshot={test_acc:.4f} best={best_acc:.4f} | time={dt:.0f}s{tag}")
        if epoch - best_ep >= args.patience:
            print(f"Early stop at {epoch}")
            break

    print(f"\nFinal [{args.path}] zero-shot: {best_acc:.4f}")
    with open(os.path.join(args.save_dir, f"results_{suffix}.json"), "w") as f:
        json.dump({'path': args.path, 'config': vars(args),
                   'zeroshot': best_acc, 'best_epoch': best_ep,
                   'history': history}, f, indent=2)


if __name__ == "__main__":
    main()
