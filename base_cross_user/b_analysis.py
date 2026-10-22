# -*- coding: utf-8 -*-
"""
b_analysis.py - B 路径 (episodic ProtoNet) 的上限探针
在 B 的编码器特征空间上评测: zero-shot NCM / TTA / few-shot 原型

分类协议统一为 NCM (cosine 最近类中心), 与 B 的训练/zero-shot 协议一致:
  [0shot]      全局类原型 (train)                    → 复现 67.21 sanity
  [tta_mean]   每用户特征 - μ_u (全局原型 - μ_s)      → 无标签均值对齐
  [pseudo]     全局 NCM 伪标签 → 每用户 per-class 中心  → 无标签结构化对齐
  [k-shot]     k∈{1,5,10,20} 该用户自己的类原型        → 有标签原型 (meta 协议)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fewshot_analysis as fs
from models_v4 import build_model as build_model_v4
from data_loader_v2 import load_h5_data
from diagnose_internal_louo import replicate_external_split


def norm(z):
    return F.normalize(z, dim=1)


def ncm_predict(z, protos):
    """z: (n,d) 已 L2 归一化; protos (4,d) 未归一化亦可"""
    p = F.normalize(torch.from_numpy(protos).float(), dim=1)
    return (z @ p.t()).argmax(1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="./results/best_adapt_B.pt")
    ap.add_argument("--cache_dir", type=str, default="./data_cache_v2/long_p600_s60")
    ap.add_argument("--ks", type=str, default="1,5,10,20")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp_name", type=str, default="B_upper")
    ap.add_argument("--save_dir", type=str, default="./results")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_emg, X_imu, Y, U, num_users, _ = load_h5_data(cache_dir=args.cache_dir)
    train_users, test_users = replicate_external_split(U, seed=args.seed, test_ratio=0.2)

    model = build_model_v4(
        "unified", num_classes=4, num_users=num_users,
        emg_channels=80, imu_channels=6, use_imu=True,
        use_whitening=False, use_contra=True,
        use_grl=False, use_mmd=False, feat_dim=128, dropout=0.0).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    st = ck.get('model', ck)   # train_adaptive 保存 {'model': state}
    model.load_state_dict(st)
    model.eval()

    print("[Extract] train features (for global prototypes)...")
    F_tr, Y_tr, _ = fs.extract_users(model, X_emg, X_imu, Y, U, train_users, device)
    print(f"  train {F_tr.shape}")
    F_te, Y_te, U_te = fs.extract_users(model, X_emg, X_imu, Y, U, test_users, device)
    print(f"  test  {F_te.shape}")

    # 全局类原型 (源)
    protos_g = np.stack([norm(torch.from_numpy(F_tr[Y_tr == c])).mean(0).numpy()
                         for c in range(4)])
    mu_s = F_tr.mean(axis=0)

    zte = norm(torch.from_numpy(F_te).float())
    results = {}

    # ---- 0-shot sanity ----
    acc0 = (ncm_predict(zte, protos_g) == Y_te).mean()
    print(f"[0-shot NCM] {acc0:.4f} (expect ~0.672)")
    results['0shot'] = float(acc0)

    # ---- TTA-mean: 每用户居中 ----
    z_np = zte.numpy()
    acc_t = 0
    for u in np.unique(U_te):
        m = U_te == u
        zu = z_np[m]
        zu_c = norm(torch.from_numpy(zu - zu.mean(0)))
        protos_c = norm(torch.from_numpy(protos_g - mu_s))  # 源中心也居中
        acc_t += (ncm_predict(zu_c, protos_g - mu_s) == Y_te[m]).sum()
    acc_t /= len(Y_te)
    print(f"[TTA-mean]  {acc_t:.4f}")
    results['tta_mean'] = float(acc_t)

    # ---- pseudo: 全局NCM伪标签 → per-user per-class 中心 ----
    conf = (zte @ norm(torch.from_numpy(protos_g)).t()).max(1).values
    keep_r = 0.5
    acc_p = 0
    for u in np.unique(U_te):
        m = U_te == u
        yu = Y_te[m]
        zu = z_np[m]
        # 伪标签
        yh = ncm_predict(norm(torch.from_numpy(zu)), protos_g)
        cf = conf[m].numpy()
        keep = np.argsort(cf)[-int(max(1, len(zu) * keep_r)):]
        protos_u = np.stack([norm(torch.from_numpy(zu[keep][yh[keep] == c])).mean(0).numpy()
                             if (yh[keep] == c).any() else protos_g[c]
                             for c in range(4)])
        acc_p += (ncm_predict(norm(torch.from_numpy(zu)), protos_u) == yu).sum()
    acc_p /= len(Y_te)
    print(f"[pseudo]    {acc_p:.4f}")
    results['pseudo'] = float(acc_p)

    # ---- few-shot: 每用户 k 样本/类 建自己的原型 ----
    for k in ks:
        accs = []
        for u in np.unique(U_te):
            m = U_te == u
            yu = Y_te[m]
            zu = z_np[m]
            rng = np.random.RandomState(args.seed + int(u) * 7 + k)
            shot_i, rest_i = [], []
            for c in range(4):
                ci = np.where(yu == c)[0]
                if len(ci) == 0:
                    continue
                rng.shuffle(ci)
                n = min(k, len(ci))
                shot_i.extend(ci[:n].tolist())
                rest_i.extend(ci[n:].tolist())
            if not rest_i:
                continue
            protos_u = np.stack([norm(torch.from_numpy(zu[np.array(shot_i)][yu[shot_i] == c])).mean(0).numpy()
                                 if (yu[shot_i] == c).any() else protos_g[c]
                                 for c in range(4)])
            accs.append((ncm_predict(norm(torch.from_numpy(zu[rest_i])), protos_u)
                         == yu[rest_i]).mean())
        v = float(np.mean(accs))
        print(f"[{k}-shot]    {v:.4f}")
        results[f'{k}shot'] = v

    with open(os.path.join(args.save_dir, f"results_{args.exp_name}.json"), "w") as f:
        json.dump({'config': vars(args), **results}, f, indent=2)
    print(f"\nSaved to {args.save_dir}/results_{args.exp_name}.json")


if __name__ == "__main__":
    main()
