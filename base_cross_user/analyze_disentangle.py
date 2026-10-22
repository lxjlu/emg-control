# -*- coding: utf-8 -*-
"""
analyze_disentangle.py - 特征解耦分析
判断模型是否成功分离了 手势信息 / 用户信息 / 噪音

分析方法:
1. Linear Probing: 用线性分类器从特征预测手势和用户
   - 手势准确率高 + 用户准确率低 → 解耦好
2. 距离分析:
   - 类内距离(同手势) vs 类间距离(不同手势)
   - 用户内距离 vs 用户间距离
   - 手势类内距离 / 用户间距离 比值 → 解耦指标
3. t-SNE 可视化: 分别按手势和用户着色
4. 方差分解: 特征方差中手势/用户各解释多少
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_v3 import build_model
from data_loader_v2 import load_h5_data, build_louo_loaders_v2


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_from_result(result_path, device):
    """从 results json 加载模型配置, 然后加载权重"""
    with open(result_path, 'r') as f:
        result = json.load(f)
    cfg = result['config']

    model = build_model(
        cfg.get('model', 'unified'),
        num_classes=4, num_users=175,
        emg_channels=80, imu_channels=6,
        use_imu=cfg.get('use_imu', True),
        user_weight=cfg.get('user_weight', 0.3),
        use_whitening=cfg.get('use_whitening', False),
        use_arcface=cfg.get('use_arcface', False),
        use_slim=cfg.get('use_slim', False),
        use_contra=cfg.get('use_contra', False),
        use_v02=cfg.get('use_v02', False),
        use_domain_align=cfg.get('use_domain_align', False),
        arcface_margin=cfg.get('arcface_margin', 0.3),
        arcface_scale=cfg.get('arcface_scale', 30.0),
        dropout=cfg.get('dropout', 0.0)).to(device)

    # 尝试加载权重 (results json 不存权重, 需要从 checkpoint 加载)
    # train_v3.py 保存为 best_{exp_name}.pt
    ckpt_path = result_path.replace('results_', 'best_').replace('.json', '.pt')
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        print(f"[Load] Checkpoint: {ckpt_path}")
    else:
        print(f"[Warn] No checkpoint found at {ckpt_path}, using random init")
    return model


def extract_features(model, loader, device, max_samples=10000):
    """提取特征, 返回 features, labels, users
    注意: 不传给模型 label/user, 避免触发 contra loss 的 device 问题
    """
    model.eval()
    all_feats, all_labels, all_users = [], [], []
    count = 0
    with torch.no_grad():
        for batch in loader:
            x_emg, x_imu, y, u = batch
            x_emg = x_emg.to(device)
            x_imu = x_imu.to(device) if x_imu is not None else None
            # 不传 label/user, 只提取特征 (避免 contra/domain loss 的 device 问题)
            out = model(x_emg, x_imu, label=None, user=None, stage='train')
            feat = out['gesture_feat'].cpu().numpy()
            all_feats.append(feat)
            all_labels.append(y.numpy())
            all_users.append(u.numpy())
            count += len(y)
            if count >= max_samples:
                break
    return (np.concatenate(all_feats),
            np.concatenate(all_labels),
            np.concatenate(all_users))


def linear_probing(features, labels, users, test_size=0.3):
    """线性探测: 分别用特征预测手势和用户
    返回: gesture_acc, user_acc
    手势准确率高 + 用户准确率低 → 解耦好
    """
    n = len(features)
    idx = np.random.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(features[train_idx])
    X_test = scaler.transform(features[test_idx])

    # 手势分类
    clf_gesture = LogisticRegression(max_iter=1000, C=1.0, multi_class='multinomial')
    clf_gesture.fit(X_train, labels[train_idx])
    gesture_acc = clf_gesture.score(X_test, labels[test_idx])

    # 用户分类
    clf_user = LogisticRegression(max_iter=1000, C=1.0, multi_class='multinomial')
    clf_user.fit(X_train, users[train_idx])
    user_acc = clf_user.score(X_test, users[test_idx])

    return gesture_acc, user_acc


def distance_analysis(features, labels, users):
    """距离分析:
    - 同手势距离 (intra-gesture): 同一手势不同样本间的平均距离
    - 不同手势距离 (inter-gesture): 不同手势样本间的平均距离
    - 同用户距离 (intra-user): 同一用户不同样本间的平均距离
    - 不同用户距离 (inter-user): 不同用户样本间的平均距离

    解耦指标:
    - 手势可分性 = inter_gesture / intra_gesture (越大越好)
    - 用户可分性 = inter_user / intra_user (越小越好, 说明用户信息被去除)
    - 解耦比 = 手势可分性 / 用户可分性 (越大越好)
    """
    n = min(len(features), 3000)  # 采样加速
    idx = np.random.choice(len(features), n, replace=False)
    feats = features[idx]
    labs = labels[idx]
    usrs = users[idx]

    # 计算距离矩阵
    from scipy.spatial.distance import pdist, squareform
    dist = squareform(pdist(feats, metric='cosine'))

    # 同手势 vs 不同手势
    gesture_same = labs[:, None] == labs[None, :]
    gesture_diff = ~gesture_same
    np.fill_diagonal(gesture_same, False)

    intra_gesture = dist[gesture_same].mean()
    inter_gesture = dist[gesture_diff].mean()

    # 同用户 vs 不同用户
    user_same = usrs[:, None] == usrs[None, :]
    user_diff = ~user_same
    np.fill_diagonal(user_same, False)

    intra_user = dist[user_same].mean()
    inter_user = dist[user_diff].mean()

    # 指标
    gesture_separability = inter_gesture / (intra_gesture + 1e-8)
    user_separability = inter_user / (intra_user + 1e-8)
    disentangle_ratio = gesture_separability / (user_separability + 1e-8)

    return {
        'intra_gesture': float(intra_gesture),
        'inter_gesture': float(inter_gesture),
        'intra_user': float(intra_user),
        'inter_user': float(inter_user),
        'gesture_separability': float(gesture_separability),
        'user_separability': float(user_separability),
        'disentangle_ratio': float(disentangle_ratio),
    }


def variance_decomposition(features, labels, users):
    """方差分解: 特征方差中手势/用户各解释多少
    使用单因素方差分析 (one-way ANOVA) 的 eta-squared
    """
    n_features = features.shape[1]
    gesture_eta = []
    user_eta = []

    for d in range(n_features):
        x = features[:, d]
        # 手势 eta-squared
        gesture_groups = [x[labels == g] for g in np.unique(labels) if (labels == g).sum() > 1]
        if len(gesture_groups) > 1:
            grand_mean = x.mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in gesture_groups)
            ss_total = np.sum((x - grand_mean) ** 2)
            gesture_eta.append(ss_between / (ss_total + 1e-8))
        else:
            gesture_eta.append(0.0)

        # 用户 eta-squared (采样部分用户)
        unique_users = np.unique(users)
        sampled_users = unique_users[:min(20, len(unique_users))]
        user_groups = [x[users == u] for u in sampled_users if (users == u).sum() > 1]
        if len(user_groups) > 1:
            grand_mean = x.mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in user_groups)
            ss_total = np.sum((x - grand_mean) ** 2)
            user_eta.append(ss_between / (ss_total + 1e-8))
        else:
            user_eta.append(0.0)

    return {
        'gesture_eta_mean': float(np.mean(gesture_eta)),
        'user_eta_mean': float(np.mean(user_eta)),
        'gesture_eta_max': float(np.max(gesture_eta)),
        'user_eta_max': float(np.max(user_eta)),
        'ratio': float(np.mean(gesture_eta) / (np.mean(user_eta) + 1e-8)),
    }


def plot_tsne(features, labels, users, save_dir, name):
    """t-SNE 可视化, 分别按手势和用户着色"""
    n = min(len(features), 2000)
    idx = np.random.choice(len(features), n, replace=False)
    feats = features[idx]
    labs = labels[idx]
    usrs = users[idx]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(feats)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 按手势着色
    scatter1 = axes[0].scatter(embedded[:, 0], embedded[:, 1],
                                c=labs, cmap='tab10', s=3, alpha=0.7)
    axes[0].set_title(f'{name} - Colored by Gesture')
    axes[0].legend(*scatter1.legend_elements(), title='Gesture')

    # 按用户着色 (只显示部分用户)
    unique_u = np.unique(usrs)
    user_colors = {u: i for i, u in enumerate(unique_u[:20])}
    color_vals = np.array([user_colors.get(u, -1) for u in usrs])
    mask = color_vals >= 0
    scatter2 = axes[1].scatter(embedded[mask, 0], embedded[mask, 1],
                                c=color_vals[mask], cmap='tab20', s=3, alpha=0.7)
    axes[1].set_title(f'{name} - Colored by User (first 20)')

    plt.tight_layout()
    path = os.path.join(save_dir, f'tsne_{name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] t-SNE saved to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=str, required=True,
                    help="results json 路径")
    ap.add_argument("--max_samples", type=int, default=10000)
    ap.add_argument("--save_dir", type=str, default="./analysis")
    args = ap.parse_args()

    device = get_device()
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print("Feature Disentanglement Analysis")
    print(f"Result: {args.result}")
    print("=" * 60)

    # 加载模型
    model = load_model_from_result(args.result, device)

    # 加载数据 (用全部数据做分析, 不划分 train/test)
    X_emg, X_imu, Y, U, num_users, _ = load_h5_data()
    train_loader, test_loader, _, _ = build_louo_loaders_v2(
        X_emg, X_imu, Y, U, test_ratio=0.2, batch_size=512,
        num_workers=4, norm_strategy='per_user')

    # 提取训练集特征
    print("[Extract] Extracting features from training set...")
    features, labels, users = extract_features(
        model, train_loader, device, max_samples=args.max_samples)
    print(f"[Extract] Features shape: {features.shape}")

    # 1. Linear Probing
    print("\n" + "=" * 40)
    print("[1] Linear Probing (gesture vs user)")
    print("=" * 40)
    gesture_acc, user_acc = linear_probing(features, labels, users)
    print(f"  Gesture accuracy: {gesture_acc:.4f} (高=好, 手势信息保留)")
    print(f"  User accuracy:    {user_acc:.4f} (低=好, 用户信息被去除)")
    print(f"  Disentanglement gap: {gesture_acc - user_acc:+.4f} (越大越好)")

    # 2. 距离分析
    print("\n" + "=" * 40)
    print("[2] Distance Analysis")
    print("=" * 40)
    dist_metrics = distance_analysis(features, labels, users)
    for k, v in dist_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  手势可分性: {dist_metrics['gesture_separability']:.4f} (越大越好)")
    print(f"  用户可分性: {dist_metrics['user_separability']:.4f} (越小越好)")
    print(f"  解耦比: {dist_metrics['disentangle_ratio']:.4f} (越大越好)")

    # 3. 方差分解
    print("\n" + "=" * 40)
    print("[3] Variance Decomposition (eta-squared)")
    print("=" * 40)
    var_metrics = variance_decomposition(features, labels, users)
    for k, v in var_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  手势解释方差比例: {var_metrics['gesture_eta_mean']:.4f}")
    print(f"  用户解释方差比例: {var_metrics['user_eta_mean']:.4f}")
    print(f"  比值 (手势/用户): {var_metrics['ratio']:.4f} (越大越好)")

    # 4. t-SNE 可视化
    print("\n" + "=" * 40)
    print("[4] t-SNE Visualization")
    print("=" * 40)
    name = os.path.basename(args.result).replace('results_', '').replace('.json', '')
    plot_tsne(features, labels, users, args.save_dir, name)

    # 保存结果
    results = {
        'result_file': args.result,
        'linear_probing': {
            'gesture_acc': gesture_acc,
            'user_acc': user_acc,
            'disentangle_gap': gesture_acc - user_acc,
        },
        'distance': dist_metrics,
        'variance': var_metrics,
    }
    out_path = os.path.join(args.save_dir, f'disentangle_{name}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Save] Results saved to {out_path}")


if __name__ == "__main__":
    main()
