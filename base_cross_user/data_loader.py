# -*- coding: utf-8 -*-
"""
跨用户EMG数据加载模块
从表征学习/Data目录读取txt文件，按用户划分
支持缓存加速，避免重复预处理
"""

import os
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
NUM_CHANNELS = 8


def bandpass_filter(data, fs=FS, lowcut=LOWCUT, highcut=HIGHCUT, order=BAND_ORDER):
    b, a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype="band")
    padlen = 3 * (max(len(b), len(a)) - 1)
    if data.shape[0] <= padlen:
        return data.astype(np.float32)
    return filtfilt(b, a, data, axis=0).astype(np.float32)


def load_all_users(base_dir=None, cache_dir="./data_cache"):
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "表征学习")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "emg_all.npz")

    if os.path.exists(cache_file):
        print("[Cache] Loading from cache...")
        d = np.load(cache_file, allow_pickle=True)
        return (d["X"], d["Y"], d["U"],
                int(d["num_users"]), d["user2id"].item())

    DATES = [f"06{i:02d}" for i in range(16, 27)]
    gm = {"gesture_1": 0, "gesture_2": 1, "gesture_3": 2, "gesture_4": 3}

    windows, labels, users = [], [], []
    user2id = {}
    next_uid = 0

    print("[Data] Scanning raw files (first time, ~1-2 min)...")
    for date in DATES:
        data_dir = os.path.join(base_dir, "Data", date, "Data")
        if not os.path.isdir(data_dir):
            continue
        for user in sorted(os.listdir(data_dir)):
            upath = os.path.join(data_dir, user)
            if not os.path.isdir(upath) or user.endswith("_More"):
                continue
            if user not in user2id:
                user2id[user] = next_uid
                next_uid += 1
            uid = user2id[user]
            for gname, glabel in gm.items():
                gpath = os.path.join(upath, gname)
                if not os.path.isdir(gpath):
                    continue
                for i in range(1, 6):
                    fpath = os.path.join(gpath, f"{i}.txt")
                    if not os.path.exists(fpath):
                        continue
                    try:
                        arr = np.loadtxt(fpath, delimiter=",", dtype=np.float32)
                    except Exception:
                        continue
                    if arr.shape[0] < 550:
                        continue
                    arr = arr[250:-250]
                    arr = bandpass_filter(arr)
                    if arr.shape[1] != NUM_CHANNELS:
                        continue
                    arr = (arr - arr.mean(0)) / (arr.std(0) + 1e-6)

                    ws = WINDOW_SIZE
                    for j in range(0, len(arr) - ws + 1, STEP_SIZE):
                        windows.append(arr[j:j + ws].T)
                        labels.append(glabel)
                        users.append(uid)

    if len(windows) == 0:
        return None, None, None, 0, {}

    X = np.stack(windows).astype(np.float32)
    Y = np.array(labels, dtype=np.int64)
    U = np.array(users, dtype=np.int64)

    print(f"[Cache] Saving cache ({len(X)} samples, {next_uid} users)...")
    np.savez_compressed(cache_file, X=X, Y=Y, U=U,
                        num_users=np.array(next_uid),
                        user2id=np.array(json.dumps(user2id)))
    return X, Y, U, next_uid, user2id


class EMGDataset(Dataset):
    def __init__(self, X, Y, U=None, augment=None):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y).long()
        self.U = torch.from_numpy(U).long() if U is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        x = self.X[i].clone()
        if self.augment is not None:
            x = self.augment(x)
        if self.U is not None:
            return x, self.Y[i], self.U[i]
        return x, self.Y[i]


def build_louo_loaders(X, Y, U, test_ratio=0.2, batch_size=256,
                       num_workers=4, pin_memory=True):
    unique_users = np.unique(U)
    n_test = max(1, int(len(unique_users) * test_ratio))
    np.random.shuffle(unique_users)
    train_users = unique_users[:-n_test]
    test_users = unique_users[-n_test:]

    train_mask = np.isin(U, train_users)
    test_mask = np.isin(U, test_users)

    X_train = X[train_mask]
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8

    X_tr = (X[train_mask] - mean) / std
    X_te = (X[test_mask] - mean) / std

    train_ds = EMGDataset(X_tr, Y[train_mask], augment=emg_augment)
    test_ds = EMGDataset(X_te, Y[test_mask], augment=None)

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
    return train_loader, test_loader, info, mean, std


# ==================== 轻量数据增广 ====================
def emg_augment(x):
    """x: (8, T) tensor, 轻量增广不卡GPU"""
    x = time_shift(x, max_shift=10)
    x = add_noise(x, snr_db=25)
    if torch.rand(1).item() < 0.5:
        x = channel_roll(x)
    if torch.rand(1).item() < 0.4:
        x = amplitude_scale(x)
    return x


def time_shift(x, max_shift=10):
    shift = int(torch.randint(-max_shift, max_shift + 1, (1,)))
    return torch.roll(x, shifts=shift, dims=1)


def add_noise(x, snr_db=25):
    signal_power = (x ** 2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(x) * torch.sqrt(noise_power)
    return x + noise


def channel_roll(x):
    shift = int(torch.randint(0, 8, (1,)))
    return torch.roll(x, shifts=shift, dims=0)


def amplitude_scale(x):
    scale = torch.rand(1).item() * 0.8 + 0.6
    return x * scale