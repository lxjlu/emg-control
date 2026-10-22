# -*- coding: utf-8 -*-
"""将压缩的 npz 缓存转换为未压缩的 npy 文件，加速加载"""
import os
import numpy as np

cache_dir = "./data_cache_v2"
npz_file = os.path.join(cache_dir, "emg_imu_all.npz")
npy_dir = os.path.join(cache_dir, "npy")

if not os.path.exists(npy_dir):
    os.makedirs(npy_dir, exist_ok=True)
    print(f"[Convert] Loading {npz_file} (4.5GB, may take 3-5 min)...")
    d = np.load(npz_file, allow_pickle=True)
    for key in d.files:
        arr = d[key]
        np.save(os.path.join(npy_dir, f"{key}.npy"), arr)
        print(f"  saved {key}: {arr.shape} {arr.dtype}")
    print("[Convert] Done!")
else:
    print("[Convert] npy cache already exists, skipping.")
    for f in os.listdir(npy_dir):
        print(f"  {f}")
