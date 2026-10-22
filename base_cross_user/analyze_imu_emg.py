#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析IMU和EMG信号的用户差异性
验证假设：同一手势的IMU是否真的用户差异小？
"""

import numpy as np
from data_loader_v2 import load_h5_data

# 加载数据
X_emg, X_imu, Y, U, num_users, user2id = load_h5_data()

print(f"数据形状: EMG={X_emg.shape}, IMU={X_imu.shape}")
print(f"用户数: {num_users}, 手势数: {len(np.unique(Y))}")

# 按用户和手势分组
user_gesture_emg = {}
user_gesture_imu = {}

for u in range(num_users):
    user_mask = U == u
    for g in range(4):
        gesture_mask = Y == g
        mask = user_mask & gesture_mask
        if mask.sum() > 0:
            if u not in user_gesture_emg:
                user_gesture_emg[u] = {}
                user_gesture_imu[u] = {}
            user_gesture_emg[u][g] = X_emg[mask].mean(axis=0)  # 平均模板
            user_gesture_imu[u][g] = X_imu[mask].mean(axis=0)

# 计算用户内和用户间差异
emg_within_user = []
emg_between_user = []
imu_within_user = []
imu_between_user = []

users = list(user_gesture_emg.keys())

for u1 in users:
    for g in range(4):
        if g not in user_gesture_emg[u1]:
            continue
        # 用户内差异: 同一用户同一手势的不同样本
        emg_samples = X_emg[(U == u1) & (Y == g)]
        imu_samples = X_imu[(U == u1) & (Y == g)]
        if len(emg_samples) > 1:
            emg_within_user.append(np.std(emg_samples, axis=0).mean())
            imu_within_user.append(np.std(imu_samples, axis=0).mean())
        
        # 用户间差异: 不同用户同一手势
        for u2 in users:
            if u1 >= u2 or g not in user_gesture_emg[u2]:
                continue
            emg_diff = np.abs(user_gesture_emg[u1][g] - user_gesture_emg[u2][g]).mean()
            imu_diff = np.abs(user_gesture_imu[u1][g] - user_gesture_imu[u2][g]).mean()
            emg_between_user.append(emg_diff)
            imu_between_user.append(imu_diff)

print("\n" + "="*60)
print("用户内差异 (同一用户同一手势的不同样本)")
print(f"  EMG: {np.mean(emg_within_user):.4f} ± {np.std(emg_within_user):.4f}")
print(f"  IMU: {np.mean(imu_within_user):.4f} ± {np.std(imu_within_user):.4f}")

print("\n用户间差异 (不同用户同一手势的平均模板)")
print(f"  EMG: {np.mean(emg_between_user):.4f} ± {np.std(emg_between_user):.4f}")
print(f"  IMU: {np.mean(imu_between_user):.4f} ± {np.std(imu_between_user):.4f}")

print("\n" + "="*60)
print("关键指标: 用户间差异 / 用户内差异 (越大说明用户区分度越高)")
print(f"  EMG: {np.mean(emg_between_user) / np.mean(emg_within_user):.2f}")
print(f"  IMU: {np.mean(imu_between_user) / np.mean(imu_within_user):.2f}")

# 计算IMU和EMG的用户差异比值
ratio = (np.mean(imu_between_user) / np.mean(imu_within_user)) / \
        (np.mean(emg_between_user) / np.mean(emg_within_user))
print(f"\nIMU用户区分度 / EMG用户区分度: {ratio:.2f}")
print(f"  < 1 说明IMU用户差异更小 (好!)")
print(f"  > 1 说明IMU用户差异更大 (坏!)")

# 按手势分析
print("\n" + "="*60)
print("按手势分析用户间差异:")
gesture_names = ["握拳", "张开", "捏合", "放松"]
for g in range(4):
    emg_g = [np.abs(user_gesture_emg[u1][g] - user_gesture_emg[u2][g]).mean()
             for u1 in users for u2 in users
             if u1 < u2 and g in user_gesture_emg.get(u2, {})]
    imu_g = [np.abs(user_gesture_imu[u1][g] - user_gesture_imu[u2][g]).mean()
             for u1 in users for u2 in users
             if u1 < u2 and g in user_gesture_imu.get(u2, {})]
    if emg_g and imu_g:
        print(f"  {gesture_names[g]}: EMG={np.mean(emg_g):.4f}, IMU={np.mean(imu_g):.4f}, "
              f"IMU/EMG={np.mean(imu_g)/np.mean(emg_g):.2f}")