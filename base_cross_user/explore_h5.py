#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
探索h5文件的完整结构
找出所有可用的信息
"""

import os
import h5py
import numpy as np

# 找一个h5文件
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "表征学习")
DATES = [f"06{i:02d}" for i in range(16, 27)]

found = False
for date in DATES:
    data_dir = os.path.join(base_dir, "Data", date, "Data")
    if not os.path.isdir(data_dir):
        continue
    for user in sorted(os.listdir(data_dir)):
        if not user.endswith("_More"):
            continue
        upath = os.path.join(data_dir, user, "ges01")
        if not os.path.isdir(upath):
            continue
        for h5_file in sorted(os.listdir(upath)):
            if h5_file.endswith(".h5"):
                h5_path = os.path.join(upath, h5_file)
                print(f"找到文件: {h5_path}")
                print("="*60)
                
                with h5py.File(h5_path, 'r') as hf:
                    print(f"Keys: {list(hf.keys())}")
                    print()
                    
                    for key in hf.keys():
                        data = hf[key]
                        print(f"Key: {key}")
                        print(f"  Shape: {data.shape}")
                        print(f"  Dtype: {data.dtype}")
                        if data.shape[0] > 0:
                            print(f"  First row: {data[0][:10]}...")
                            print(f"  Min: {data[...].min():.4f}, Max: {data[...].max():.4f}")
                            print(f"  Mean: {data[...].mean():.4f}, Std: {data[...].std():.4f}")
                        print()
                
                found = True
                break
        if found:
            break
    if found:
        break

if not found:
    print("未找到h5文件")