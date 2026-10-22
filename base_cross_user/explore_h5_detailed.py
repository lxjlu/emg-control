#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
详细分析h5文件的93列分别是什么
"""

import os
import h5py
import numpy as np

# 找一个h5文件
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "表征学习")
h5_path = os.path.join(base_dir, "Data", "0618", "Data", "仁毕觉尼_More", "ges01", "data_20260618_093707.h5")

print(f"分析文件: {h5_path}")
print("="*60)

with h5py.File(h5_path, 'r') as hf:
    data = hf['emg_data'][...]
    print(f"Shape: {data.shape}")
    print(f"  行数(时间帧): {data.shape[0]}")
    print(f"  列数(特征): {data.shape[1]}")
    print()
    
    print("逐列分析:")
    print("-"*60)
    for col in range(data.shape[1]):
        col_data = data[:, col]
        print(f"Col {col:2d}: min={col_data.min():10.2f}, max={col_data.max():10.2f}, "
              f"mean={col_data.mean():10.2f}, std={col_data.std():10.2f}")
    
    print()
    print("="*60)
    print("推测列含义:")
    print("-"*60)
    print("Col 0:  时间戳 (timestamp, 大数值)")
    print("Col 1-2: 可能是IMU (2通道)")
    print("Col 3-8: 可能是IMU (6通道: 3轴acc + 3轴gyro)")
    print("Col 9-10: 可能是其他传感器")
    print("Col 11-90: 80通道EMG (10组x8通道)")
    print("Col 91-92: 可能是标签/状态")
    
    # 验证EMG通道
    print()
    print("EMG通道 (Col 11-90) 示例:")
    for col in [11, 20, 50, 80, 90]:
        if col < data.shape[1]:
            print(f"  Col {col}: min={data[:, col].min():.2f}, max={data[:, col].max():.2f}")
    
    # 验证IMU通道
    print()
    print("IMU通道 (Col 3-8) 示例:")
    for col in range(3, 9):
        print(f"  Col {col}: min={data[:, col].min():.2f}, max={data[:, col].max():.2f}")