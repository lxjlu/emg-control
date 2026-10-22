#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""读取所有实验结果并分析"""
import json, os

results_dir = 'results'
experiments = {
    'cnn1d': 'CNN-1D Baseline',
    'transformer': 'Transformer Baseline',
    'dann': 'DANN 域对抗',
    'dann_imu': 'DANN + IMU',
    'dann_contra': 'DANN + 对比学习',
    'dann_imu_contra': 'DANN + IMU + 对比学习',
    'disentangled': '特征解耦 (8通道)',
    'disentangled_imu': '特征解耦 + IMU',
    'disentangled_contra': '特征解耦 + 对比学习',
    'disentangled_imu_contra': '特征解耦 + IMU + 对比',
    'imu_attn_imu': 'IMU引导注意力',
    'imu_attn_imu_contra': 'IMU引导注意力 + DANN',
    'imu_decompose_imu': 'IMU锚定特征分解',
    'imu_decompose_imu_contra': 'IMU锚定特征分解 + DANN',
    'imu_crossmodal_imu': '跨模态对比学习',
    'imu_crossmodal_imu_contra': '跨模态对比 + DANN',
}

print('='*90)
print('ALL EXPERIMENT RESULTS')
print('='*90)
header = f'{"Experiment":<35} {"Zero-Shot":<12} {"Train Acc":<12} {"Gap":<10} {"Epochs":<8} {"Status"}'
print(header)
print('-'*90)

results = []
for key, name in experiments.items():
    fpath = os.path.join(results_dir, f'results_{key}.json')
    if os.path.exists(fpath):
        with open(fpath, 'r') as f:
            r = json.load(f)
        zs = r.get('final_zeroshot_acc', r.get('best_zeroshot', 0))
        tr = r.get('final_train_acc', r.get('best_train', 0))
        ep = r.get('epochs', 0)
        gap = tr - zs
        results.append((name, zs, tr, gap, ep, 'DONE'))
    else:
        results.append((name, 0, 0, 0, 0, 'TODO'))

results.sort(key=lambda x: x[1], reverse=True)

for i, (name, zs, tr, gap, ep, status) in enumerate(results):
    if status == 'DONE':
        line = f'{i+1:>2}. {name:<33} {zs:<12.4f} {tr:<12.4f} {gap:<10.4f} {ep:<8} {status}'
    else:
        line = f'{i+1:>2}. {name:<33} {"MISSING":<12} {"":<12} {"":<10} {"":<8} {status}'
    print(line)

print('='*90)
print()
print('TOP 5:')
done = [r for r in results if r[5] == 'DONE']
for i, (name, zs, tr, gap, ep, _) in enumerate(done[:5]):
    print(f'  {i+1}. {name}: {zs:.4f} (Gap: {gap:.4f}, Epochs: {ep})')

print()
print('EARLY STOPPING ANALYSIS:')
print('-'*60)
for name, zs, tr, gap, ep, status in done:
    print(f'{name:<35} Epochs: {ep:>3} | Gap: {gap:.4f} | Zero-Shot: {zs:.4f}')