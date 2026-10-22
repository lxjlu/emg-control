#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速分析IMU和EMG信号的用户差异性
"""

import numpy as np
from data_loader_v2 import load_h5_data

# 加载数据
X_emg, X_imu, Y, U, num_users, user2id = load_h5_data()

print(f"数据形状: EMG={X_emg.shape}, IMU={X_imu.shape}")
print(f"用户数: {num_users}, 手势数: {len(np.unique(Y))}")

# 随机选10个用户计算
np.random.seed(42)
sample_users = np.random.choice(np.unique(U), size=min(10, num_users), replace=False)

print(f"\n采样用户: {sample_users}")

# 计算每个用户每个手势的平均IMU和EMG模板
templates_emg = {}
templates_imu = {}

for u in sample_users:
    mask_u = U == u
    for g in range(4):
        mask_g = Y == g
        mask = mask_u & mask_g
        if mask.sum() > 0:
            templates_emg[(u, g)] = X_emg[mask].mean(axis=0)  # [80, 150]
            templates_imu[(u, g)] = X_imu[mask].mean(axis=0)  # [6, 150]

# 计算用户间差异 (不同用户同一手势)
emg_between = []
imu_between = []

for g in range(4):
    for i, u1 in enumerate(sample_users):
        for u2 in sample_users[i+1:]:
            if (u1, g) in templates_emg and (u2, g) in templates_emg:
                emg_diff = np.mean((templates_emg[(u1, g)] - templates_emg[(u2, g)]) ** 2) ** 0.5
                imu_diff = np.mean((templates_imu[(u1, g)] - templates_imu[(u2, g)]) ** 2) ** 0.5
                emg_between.append(emg_diff)
                imu_between.append(imu_diff)

print(f"\n用户间差异 (不同用户同一手势的模板差异):")
print(f"  EMG: {np.mean(emg_between):.4f} ± {np.std(emg_between):.4f}")
print(f"  IMU: {np.mean(imu_between):.4f} ± {np.std(imu_between):.4f}")
print(f"  IMU/EMG 比值: {np.mean(imu_between) / np.mean(emg_between):.2f}")

# 计算用户内差异 (同一用户同一手势的不同样本)
emg_within = []
imu_within = []

for u in sample_users:
    for g in range(4):
        mask = (U == u) & (Y == g)
        if mask.sum() > 10:
            samples_emg = X_emg[mask][:100]  # 最多100个样本
            samples_imu = X_imu[mask][:100]
            emg_within.append(np.mean(np.std(samples_emg, axis=0)))
            imu_within.append(np.mean(np.std(samples_imu, axis=0)))

print(f"\n用户内差异 (同一用户同一手势的样本差异):")
print(f"  EMG: {np.mean(emg_within):.4f} ± {np.std(emg_within):.4f}")
print(f"  IMU: {np.mean(imu_within):.4f} ± {np.std(imu_within):.4f}")

print(f"\n" + "="*60)
print("关键指标: 用户间差异 / 用户内差异")
print(f"  (越大说明用户区分度越高，越难泛化)")
print(f"  EMG: {np.mean(emg_between) / np.mean(emg_within):.2f}")
print(f"  IMU: {np.mean(imu_between) / np.mean(imu_within):.2f}")

ratio = (np.mean(imu_between) / np.mean(imu_within)) / (np.mean(emg_between) / np.mean(emg_within))
print(f"\nIMU用户区分度 / EMG用户区分度: {ratio:.2f}")
if ratio < 1:
    print("  ✓ IMU用户差异相对更小 (理论上更适合作锚定)")
else:
    print("  ✗ IMU用户差异相对更大 (不适合作锚定)")