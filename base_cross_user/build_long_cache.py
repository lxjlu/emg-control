# -*- coding: utf-8 -*-
"""
build_long_cache.py - 从原始 h5 重建更长窗口缓存 (300ms / 600ms)

与 load_h5_data 完全一致的处理流水线 (同源遍历顺序保证 user2id 映射不变,
因此 train/test 用户划分可复现), 唯一差异是窗口长度与步进:
  150ms/15ms (原有)  ->  300ms/30ms 或 600ms/60ms  (窗/步比保持 10)

输出到 ./data_cache_v2/npy_<win>_<step>/   (train_v4 --cache_dir 指定)
"""
import os
import json
import argparse
import numpy as np
import h5py
from scipy.signal import butter, filtfilt

FS = 500
LOWCUT = 20
HIGHCUT = 245
BAND_ORDER = 4
NUM_EMG_CH = 80
NUM_IMU_CH = 6


def bandpass_filter(data, fs=FS, lowcut=LOWCUT, highcut=HIGHCUT, order=BAND_ORDER):
    b, a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype="band")
    padlen = 3 * (max(len(b), len(a)) - 1)
    if data.shape[0] <= padlen:
        return data.astype(np.float32)
    return filtfilt(b, a, data, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--win_pts", type=int, default=300,
                    help="窗口采样点数 (基线=150)")
    ap.add_argument("--step_pts", type=int, default=30,
                    help="窗口步进采样点数 (基线=15, 窗/步比保持10)")
    ap.add_argument("--base_dir", type=str,
                    default=r"d:\罗老师相关\emg_per\表征学习")
    ap.add_argument("--cache_root", type=str, default="./data_cache_v2")
    args = ap.parse_args()

    win, step = args.win_pts, args.step_pts
    # 与 data_loader 的 npy 读取结构兼容: <root>/<name>/npy/X_emg.npy
    out_dir = os.path.join(args.cache_root,
                           f"long_p{win}_s{step}", "npy")
    os.makedirs(out_dir, exist_ok=True)

    # 与 load_h5_data 相同的遍历顺序: DATES 递增, 用户/手势/h5 sorted
    DATES = [f"06{i:02d}" for i in range(16, 27)]
    gm = {"ges01": 0, "ges02": 1, "ges03": 2, "ges04": 3}

    emg_windows, imu_windows, labels, users = [], [], [], []
    user2id = {}
    next_uid = 0
    n_h5 = 0

    print(f"Building {win}pt window cache (step={step})...")
    for date in DATES:
        data_dir = os.path.join(args.base_dir, "Data", date, "Data")
        if not os.path.isdir(data_dir):
            continue
        for user in sorted(os.listdir(data_dir)):
            upath = os.path.join(data_dir, user)
            if not os.path.isdir(upath) or not user.endswith("_More"):
                continue
            base_user = user.replace("_More", "")
            if base_user not in user2id:
                user2id[base_user] = next_uid
                next_uid += 1
            uid = user2id[base_user]

            for gname, glabel in gm.items():
                gpath = os.path.join(upath, gname)
                if not os.path.isdir(gpath):
                    continue
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

                    emg_raw = raw[:, 11:91].astype(np.float32)
                    imu_raw = raw[:, 3:9].astype(np.float32)
                    if emg_raw.shape[0] < win + 50:
                        continue

                    emg_raw = bandpass_filter(emg_raw)
                    imu_raw = imu_raw - imu_raw.mean(axis=0, keepdims=True)

                    n_h5 += 1
                    for j in range(0, len(emg_raw) - win + 1, step):
                        emg_windows.append(emg_raw[j:j + win].T)
                        imu_windows.append(imu_raw[j:j + win].T)
                        labels.append(glabel)
                        users.append(uid)

    if len(emg_windows) == 0:
        print("No windows! Exiting.")
        return

    X_emg = np.stack(emg_windows).astype(np.float32)
    X_imu = np.stack(imu_windows).astype(np.float32)
    Y = np.array(labels, dtype=np.int64)
    U = np.array(users, dtype=np.int64)
    print(f"[OK] {len(X_emg)} samples, {next_uid} users, {n_h5} h5 files")
    print(f"  X_emg {X_emg.shape}, X_imu {X_imu.shape}")

    np.save(os.path.join(out_dir, "X_emg.npy"), X_emg)
    np.save(os.path.join(out_dir, "X_imu.npy"), X_imu)
    np.save(os.path.join(out_dir, "Y.npy"), Y)
    np.save(os.path.join(out_dir, "U.npy"), U)
    np.save(os.path.join(out_dir, "num_users.npy"), np.array(next_uid))
    with open(os.path.join(out_dir, "user2id.npy"), "wb") as f:
        np.save(f, user2id)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
