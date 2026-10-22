import json, os

results_dir = 'results'
experiments = {
    'cnn1d': 'CNN-1D Baseline',
    'transformer': 'Transformer Baseline',
    'dann': 'DANN 域对抗',
    'dann_imu': 'DANN + IMU',
    'dann_contra': 'DANN + 对比学习',
    'dann_imu_contra': 'DANN + IMU + 对比学习 (完整)',
    'disentangled': '特征解耦 (8通道)',
    'disentangled_imu': '特征解耦 + IMU',
    'disentangled_contra': '特征解耦 + 对比学习',
    'disentangled_imu_contra': '特征解耦 + IMU + 对比 (完整)',
}

print('=' * 80)
print('EXPERIMENT RESULTS')
print('=' * 80)
print(f'{"Experiment":<30} {"Zero-Shot":<12} {"Train Acc":<12} {"Epochs":<8}')
print('-' * 80)

for key, desc in experiments.items():
    fname = f'results_{key}.json'
    fpath = os.path.join(results_dir, fname)
    if os.path.exists(fpath):
        with open(fpath) as f:
            d = json.load(f)
        zs = d.get('zeroshot_acc', 0)
        ta = d['history'][-1]['train_acc'] if d.get('history') else 0
        ep = d.get('early_stop_epoch', 0)
        print(f'{desc:<30} {zs:.4f}       {ta:.4f}      {ep:<8}')
    else:
        print(f'{desc:<30} {"MISSING":<12}')

print('=' * 80)