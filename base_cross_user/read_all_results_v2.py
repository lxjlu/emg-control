#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""读取所有实验结果并分析"""
import json, os, glob

results_dir = 'results'
json_files = sorted(glob.glob(os.path.join(results_dir, 'results_*.json')))

print('='*100)
print('ALL EXPERIMENT RESULTS')
print('='*100)

results = []
for fpath in json_files:
    fname = os.path.basename(fpath).replace('results_', '').replace('.json', '')
    with open(fpath, 'r') as f:
        r = json.load(f)
    
    # 尝试不同的key
    zs = r.get('final_zeroshot_acc', r.get('best_zeroshot', r.get('zeroshot_acc', 0)))
    tr = r.get('final_train_acc', r.get('best_train', r.get('train_acc', 0)))
    ep = r.get('epochs', r.get('final_epoch', r.get('best_epoch', 0)))
    
    # 如果有history，从history获取
    if 'history' in r and len(r['history']) > 0:
        history = r['history']
        if isinstance(history, list) and len(history) > 0:
            ep = len(history)
            best_entry = max(history, key=lambda x: x.get('zeroshot_acc', 0))
            zs = best_entry.get('zeroshot_acc', 0)
            tr = best_entry.get('train_acc', 0)
    
    gap = tr - zs
    results.append((fname, zs, tr, gap, ep, fpath))

results.sort(key=lambda x: x[1], reverse=True)

header = f'{"Experiment":<35} {"Zero-Shot":<12} {"Train Acc":<12} {"Gap":<10} {"Epochs":<8}'
print(header)
print('-'*100)

for i, (fname, zs, tr, gap, ep, fpath) in enumerate(results):
    print(f'{i+1:>2}. {fname:<33} {zs:<12.4f} {tr:<12.4f} {gap:<10.4f} {ep:<8}')

print('='*100)
print()
print('TOP 5:')
for i, (fname, zs, tr, gap, ep, fpath) in enumerate(results[:5]):
    print(f'  {i+1}. {fname}: {zs:.4f} (Gap: {gap:.4f}, Epochs: {ep})')

print()
print('EARLY STOPPING ANALYSIS:')
print('-'*80)
for fname, zs, tr, gap, ep, fpath in results:
    status = '早停' if ep < 100 else '完整'
    print(f'{fname:<35} Epochs: {ep:>3}/{100} ({status}) | Gap: {gap:.4f} | Zero-Shot: {zs:.4f}')