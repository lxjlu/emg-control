# -*- coding: utf-8 -*-
"""
跨用户EMG+IMU数据加载 v2
从h5文件读取EMG(80通道) + IMU(6通道)
支持缓存、多模态融合、用户无关归一化
"""

import os
import re
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt

FS = 500
LOWCUT = 20
HIGHCUT = 245
BAND_ORDER = 4
WINDOW_SIZE = 150
STEP_SIZE = 15
NUM_CLASSES = 4
NUM_EMG_CH = 80
NUM_IMU_CH = 6


def bandpass_filter(data, fs=FS, lowcut=LOWCUT, highcut=HIGHCUT, order=BAND_ORDER):
    b, a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype="band")
    padlen = 3 * (max(len(b), len(a)) - 1)
    if data.shape[0] <= padlen:
        return data.astype(np.float32)
    return filtfilt(b, a, data, axis=0).astype(np.float32)


def load_h5_data(base_dir=None, cache_dir="./data_cache_v2"):
    """
    从h5文件加载EMG+IMU数据
    h5文件结构: emg_data (N, 93)
      - col 0: timestamp
      - col 1-2: IMU (acc/gyro部分)
      - col 3-10: 其他传感器
      - col 11-90: 80通道EMG (10组x8通道)
      - col 91-92: 标签/状态
    """
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "表征学习")
    os.makedirs(cache_dir, exist_ok=True)
    npy_dir = os.path.join(cache_dir, "npy")
    cache_file = os.path.join(cache_dir, "emg_imu_all.npz")

    # 优先使用未压缩的 npy 缓存（加载速度快 10x）
    if os.path.isdir(npy_dir) and os.path.exists(os.path.join(npy_dir, "X_emg.npy")):
        print("[Cache] Loading EMG+IMU from npy cache (fast)...")
        X_emg = np.load(os.path.join(npy_dir, "X_emg.npy"))
        X_imu = np.load(os.path.join(npy_dir, "X_imu.npy"))
        Y = np.load(os.path.join(npy_dir, "Y.npy"))
        U = np.load(os.path.join(npy_dir, "U.npy"))
        num_users = int(np.load(os.path.join(npy_dir, "num_users.npy")))
        user2id = np.load(os.path.join(npy_dir, "user2id.npy"), allow_pickle=True).item()
        return X_emg, X_imu, Y, U, num_users, user2id

    if os.path.exists(cache_file):
        print("[Cache] Loading EMG+IMU from cache (compressed, slow)...")
        d = np.load(cache_file, allow_pickle=True)
        return (d["X_emg"], d["X_imu"], d["Y"], d["U"],
                int(d["num_users"]), d["user2id"].item())

    DATES = [f"06{i:02d}" for i in range(16, 27)]
    gm = {"ges01": 0, "ges02": 1, "ges03": 2, "ges04": 3}

    emg_windows, imu_windows, labels, users = [], [], [], []
    user2id = {}
    next_uid = 0

    print("[Data] Scanning h5 files (first time, ~2-3 min)...")
    import h5py

    for date in DATES:
        data_dir = os.path.join(base_dir, "Data", date, "Data")
        if not os.path.isdir(data_dir):
            continue
        for user in sorted(os.listdir(data_dir)):
            upath = os.path.join(data_dir, user)
            if not os.path.isdir(upath) or not user.endswith("_More"):
                continue
            # 提取用户名 (去掉_More后缀)
            base_user = user.replace("_More", "")
            if base_user not in user2id:
                user2id[base_user] = next_uid
                next_uid += 1
            uid = user2id[base_user]

            for gname, glabel in gm.items():
                gpath = os.path.join(upath, gname)
                if not os.path.isdir(gpath):
                    continue
                # 直接遍历h5文件
                for h5_file in sorted(os.listdir(gpath)):
                    if not h5_file.endswith(".h5"):
                        continue
                    h5_path = os.path.join(gpath, h5_file)

                    try:
                        with h5py.File(h5_path, 'r') as hf:
                            raw = hf['emg_data'][...]
                    except Exception:
                        continue

                    if raw.ndim != 2 or raw.shape[1] < 90:
                        continue

                    # 提取EMG: col 11-90 (80通道)
                    emg_raw = raw[:, 11:91].astype(np.float32)
                    # 提取IMU: col 3-8 (6通道: 3轴acc + 3轴gyro)
                    imu_raw = raw[:, 3:9].astype(np.float32)

                    if emg_raw.shape[0] < 550:
                        continue

                    # 带通滤波 (EMG)
                    emg_raw = bandpass_filter(emg_raw)
                    # IMU不需要带通滤波，但需要去趋势
                    imu_raw = imu_raw - imu_raw.mean(axis=0, keepdims=True)

                    # 去掉逐窗口归一化，只保留全局归一化
                    # emg_raw = (emg_raw - emg_raw.mean(0)) / (emg_raw.std(0) + 1e-6)
                    # imu_raw = (imu_raw - imu_raw.mean(0)) / (imu_raw.std(0) + 1e-6)

                    ws = WINDOW_SIZE
                    for j in range(0, len(emg_raw) - ws + 1, STEP_SIZE):
                        emg_windows.append(emg_raw[j:j + ws].T)
                        imu_windows.append(imu_raw[j:j + ws].T)
                        labels.append(glabel)
                        users.append(uid)

    if len(emg_windows) == 0:
        return None, None, None, None, 0, {}

    X_emg = np.stack(emg_windows).astype(np.float32)
    X_imu = np.stack(imu_windows).astype(np.float32)
    Y = np.array(labels, dtype=np.int64)
    U = np.array(users, dtype=np.int64)

    print(f"[Cache] Saving cache ({len(X_emg)} samples, {next_uid} users)...")
    np.savez_compressed(cache_file, X_emg=X_emg, X_imu=X_imu, Y=Y, U=U,
                        num_users=np.array(next_uid),
                        user2id=np.array(json.dumps(user2id)))
    return X_emg, X_imu, Y, U, next_uid, user2id


class EMGIMUDataset(Dataset):
    def __init__(self, X_emg, X_imu, Y, U=None, augment=None):
        self.X_emg = torch.from_numpy(X_emg)
        self.X_imu = torch.from_numpy(X_imu)
        self.Y = torch.from_numpy(Y).long()
        self.U = torch.from_numpy(U).long() if U is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        x_emg = self.X_emg[i].clone()
        x_imu = self.X_imu[i].clone()
        if self.augment is not None:
            x_emg, x_imu = self.augment(x_emg, x_imu)
        if self.U is not None:
            return x_emg, x_imu, self.Y[i], self.U[i]
        return x_emg, x_imu, self.Y[i]


def build_louo_loaders_v2(X_emg, X_imu, Y, U, test_ratio=0.2, batch_size=256,
                          num_workers=4, pin_memory=True, norm_strategy="global"):
    """
    构建LOUO数据加载器
    norm_strategy: 
      - "global": 全局Z-Score（用训练集统计量）
      - "none": 无归一化
      - "per_user": 逐用户Z-Score
    """
    unique_users = np.unique(U)
    n_test = max(1, int(len(unique_users) * test_ratio))
    np.random.shuffle(unique_users)
    train_users = unique_users[:-n_test]
    test_users = unique_users[-n_test:]

    train_mask = np.isin(U, train_users)
    test_mask = np.isin(U, test_users)

    # 根据归一化策略处理数据
    if norm_strategy == "global":
        # 全局Z-Score（用训练集统计量）
        mean_emg = X_emg[train_mask].mean(axis=(0, 2), keepdims=True)
        std_emg = X_emg[train_mask].std(axis=(0, 2), keepdims=True) + 1e-8
        mean_imu = X_imu[train_mask].mean(axis=(0, 2), keepdims=True)
        std_imu = X_imu[train_mask].std(axis=(0, 2), keepdims=True) + 1e-8

        X_emg_tr = (X_emg[train_mask] - mean_emg) / std_emg
        X_emg_te = (X_emg[test_mask] - mean_emg) / std_emg
        X_imu_tr = (X_imu[train_mask] - mean_imu) / std_imu
        X_imu_te = (X_imu[test_mask] - mean_imu) / std_imu
        norm_params = (mean_emg, std_emg, mean_imu, std_imu)

    elif norm_strategy == "none":
        # 无归一化
        X_emg_tr = X_emg[train_mask]
        X_emg_te = X_emg[test_mask]
        X_imu_tr = X_imu[train_mask]
        X_imu_te = X_imu[test_mask]
        norm_params = None

    elif norm_strategy == "per_user":
        # 逐用户Z-Score
        X_emg_tr = np.zeros_like(X_emg[train_mask])
        X_imu_tr = np.zeros_like(X_imu[train_mask])
        X_emg_te = np.zeros_like(X_emg[test_mask])
        X_imu_te = np.zeros_like(X_imu[test_mask])

        # 训练集：每个用户独立归一化
        for u in train_users:
            mask = (U[train_mask] == u)
            mean_e = X_emg[train_mask][mask].mean(axis=(0, 2), keepdims=True)
            std_e = X_emg[train_mask][mask].std(axis=(0, 2), keepdims=True) + 1e-8
            mean_i = X_imu[train_mask][mask].mean(axis=(0, 2), keepdims=True)
            std_i = X_imu[train_mask][mask].std(axis=(0, 2), keepdims=True) + 1e-8
            X_emg_tr[mask] = (X_emg[train_mask][mask] - mean_e) / std_e
            X_imu_tr[mask] = (X_imu[train_mask][mask] - mean_i) / std_i

        # 测试集：用各自用户的统计量归一化
        for u in test_users:
            mask = (U[test_mask] == u)
            mean_e = X_emg[test_mask][mask].mean(axis=(0, 2), keepdims=True)
            std_e = X_emg[test_mask][mask].std(axis=(0, 2), keepdims=True) + 1e-8
            mean_i = X_imu[test_mask][mask].mean(axis=(0, 2), keepdims=True)
            std_i = X_imu[test_mask][mask].std(axis=(0, 2), keepdims=True) + 1e-8
            X_emg_te[mask] = (X_emg[test_mask][mask] - mean_e) / std_e
            X_imu_te[mask] = (X_imu[test_mask][mask] - mean_i) / std_i

        norm_params = None

    train_ds = EMGIMUDataset(X_emg_tr, X_imu_tr, Y[train_mask], U[train_mask],
                             augment=emg_imu_augment)
    test_ds = EMGIMUDataset(X_emg_te, X_imu_te, Y[test_mask], U[test_mask], augment=None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers,
                              pin_memory=pin_memory, persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory,
                             persistent_workers=(num_workers > 0))

    info = {
        "n_total_users": len(unique_users),
        "n_train_users": len(train_users),
        "n_test_users": len(test_users),
        "train_samples": int(train_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "test_users": [int(u) for u in test_users],
    }
    return train_loader, test_loader, info, norm_params


# ==================== 强力数据增广 ====================
def emg_imu_augment(x_emg, x_imu):
    """
    强力增广，模拟用户间差异
    x_emg: (80, T), x_imu: (6, T)
    """
    # 1. 时间平移 (始终应用)
    x_emg, x_imu = time_shift_pair(x_emg, x_imu, max_shift=10)

    # 2. 加噪声
    x_emg = add_noise(x_emg, snr_db=25)
    x_imu = add_noise(x_imu, snr_db=30)

    # 3. 通道dropout (模拟电极接触不良)
    if torch.rand(1).item() < 0.4:
        x_emg = channel_dropout(x_emg, drop_ratio=0.2)

    # 4. 幅度缩放 (模拟肌肉力量差异)
    if torch.rand(1).item() < 0.5:
        x_emg = amplitude_scale(x_emg)

    # 5. IMU扰动 (模拟佩戴位置差异)
    if torch.rand(1).item() < 0.3:
        x_imu = imu_perturb(x_imu)

    return x_emg, x_imu


def time_shift_pair(x_emg, x_imu, max_shift=10):
    shift = int(torch.randint(-max_shift, max_shift + 1, (1,)))
    return torch.roll(x_emg, shifts=shift, dims=1), torch.roll(x_imu, shifts=shift, dims=1)


def add_noise(x, snr_db=25):
    signal_power = (x ** 2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10))
    return x + torch.randn_like(x) * torch.sqrt(noise_power)


def channel_dropout(x, drop_ratio=0.2):
    n_ch = x.shape[0]
    n_drop = max(1, int(n_ch * drop_ratio))
    drop_idx = torch.randperm(n_ch)[:n_drop]
    x_d = x.clone()
    x_d[drop_idx] = 0
    return x_d


def amplitude_scale(x):
    scale = torch.rand(1).item() * 1.4 + 0.3
    return x * scale


def imu_perturb(x_imu):
    """IMU扰动: 随机旋转+缩放"""
    scale = torch.rand(x_imu.shape[0]) * 0.4 + 0.8
    return x_imu * scale.unsqueeze(1)