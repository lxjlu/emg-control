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
    'imu_attn': 'IMU引导注意力',
    'imu_attn_dann': 'IMU引导注意力 + DANN',
    'imu_decompose': 'IMU锚定特征分解',
    'imu_decompose_dann': 'IMU锚定特征分解 + DANN',
    'imu_crossmodal': '跨模态对比学习',
    'imu_crossmodal_dann': '跨模态对比 + DANN',
}

print('=' * 80)
print('EXPERIMENT RESULTS')
print('=' * 80)
print(f'{"Experiment":<35} {"Zero-Shot":<12} {"Train Acc":<12} {"Epochs":<8} {"Status":<10}')
print('-' * 80)

missing = []
done_list = []

for key, desc in experiments.items():
    fname = f'results_{key}.json'
    fpath = os.path.join(results_dir, fname)
    if os.path.exists(fpath):
        with open(fpath) as f:
            d = json.load(f)
        zs = d.get('zeroshot_acc', 0)
        ta = d['history'][-1]['train_acc'] if d.get('history') else 0
        ep = d.get('early_stop_epoch', 0)
        print(f'{desc:<35} {zs:.4f}       {ta:.4f}      {ep:<8} {"DONE":<10}')
        done_list.append((key, desc, zs))
    else:
        print(f'{desc:<35} {"MISSING":<12}                              {"TODO":<10}')
        missing.append((key, desc))

print('=' * 80)

if missing:
    print('\n需要运行的实验:')
    for key, desc in missing:
        print(f'  {key}: {desc}')

if done_list:
    done_list.sort(key=lambda x: x[2], reverse=True)
    print('\n已完成实验排名:')
    for i, (key, desc, zs) in enumerate(done_list):
        print(f'  {i+1}. {desc}: {zs:.4f}')