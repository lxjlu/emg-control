"""
从 Android App 采集的 recordings CSV 文件训练 EMG 手势识别模型

功能：
1. 读取 recordings/ 目录下的 CSV 文件
2. 按动作标签分组并转换为训练格式
3. 使用 CNN1D 模型训练
4. 输出训练/测试准确率
5. 导出 TorchScript 模型供 Android 端使用
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt, lfilter_zi
import numpy as np
import json
import os
import csv
import argparse
from datetime import datetime

# ----------------------
# 配置参数
# ----------------------
# 使用脚本所在目录的绝对路径，避免运行目录不同导致路径错误
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "recordings")
DATA_DIR = "Sep_EMG_Data"
SAVE_MODEL_PATH = "lx_model.pth"
FILTER_COEFFS_PATH = "filter_coeffs.json"
NORM_STATS_PATH = "norm_stats.json"

config = {
    "batch_size": 64,
    "learning_rate": 0.001,
    "num_epochs": 100,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "fs": 500,
    "lowcut": 20,
    "highcut": 249.999,
    "order": 4,
    "window_size": 100,
    "step_size": 20,
    "test_split": 0.2,
    "patience": 15,
}

# 动作标签映射（5个类别：4个动作 + 1个休息状态）
LABEL_TO_CLASS = {
    "休息": 0,
    "握拳": 1,
    "手掌": 2,
    "向右": 3,
    "向左": 4,
}

# ----------------------
# 滤波函数
# ----------------------
def design_filter(lowcut, highcut, fs, order):
    """设计带通滤波器，返回 (b, a) 系数。"""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a

def bandpass_filter(data, b, a):
    """零相位滤波，与 Android 端 ButterworthFilter.filtfilt 对齐。
    
    data: (window, channels)，沿 axis=0 对每个通道独立滤波。
    """
    return filtfilt(b, a, data, axis=0)

# ----------------------
# CSV 数据读取
# ----------------------
def read_csv_file(csv_path):
    """读取 Android App 采集的 CSV 文件，返回 EMG 数据数组。
    
    CSV 格式：
    # sEMG recording v1
    # label=握拳
    # user=lx
    # protocol=STM32
    sample_index,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8
    35800,2069,2107,1999,2036,2038,2054,2028,2072
    ...
    """
    emg_data = []
    label = None
    user = None
    protocol = None
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            line = ','.join(row)
            
            # 解析元数据
            if line.startswith('# label='):
                label = line.split('=')[1].strip()
            elif line.startswith('# user='):
                user = line.split('=')[1].strip()
            elif line.startswith('# protocol='):
                protocol = line.split('=')[1].strip()
            elif line.startswith('#'):
                continue
            elif line.startswith('sample_index'):
                continue  # 跳过表头
            else:
                # 解析数据行
                parts = line.split(',')
                if len(parts) >= 9:
                    try:
                        ch_values = [float(parts[i]) for i in range(1, 9)]
                        emg_data.append(ch_values)
                    except ValueError:
                        continue
    
    if len(emg_data) == 0:
        print(f"  ⚠️ 警告：{csv_path} 中没有有效数据")
        return None, None, None, None
    
    emg_array = np.array(emg_data, dtype=np.float32)
    return emg_array, label, user, protocol

# ----------------------
# 数据加载与预处理
# ----------------------
def load_recordings(recordings_dir, config, b, a):
    """从 recordings 目录加载所有 CSV 文件并切分为窗口。"""
    all_data = []
    all_labels = []
    all_file_ids = []
    file_id = 0
    
    csv_files = sorted([f for f in os.listdir(recordings_dir) if f.endswith('.csv')])
    
    if len(csv_files) == 0:
        raise ValueError(f"在 {recordings_dir} 中未找到 CSV 文件！")
    
    print(f"\n找到 {len(csv_files)} 个 CSV 文件")
    print("="*60)
    
    for csv_file in csv_files:
        csv_path = os.path.join(recordings_dir, csv_file)
        emg_data, label, user, protocol = read_csv_file(csv_path)
        
        if emg_data is None:
            continue
        
        if label not in LABEL_TO_CLASS:
            print(f"  ⚠️ 跳过未知标签：{label} ({csv_file})")
            continue
        
        label_class = LABEL_TO_CLASS[label]
        print(f"  📄 {csv_file} | 标签: {label} (类别{label_class}) | 数据量: {len(emg_data)}")
        
        # 滑动窗口切分
        ws = config["window_size"]
        step = config["step_size"]
        j = 0
        window_count = 0
        while j + ws <= len(emg_data):
            window = emg_data[j:j + ws]  # (100, 8)
            filtered = bandpass_filter(window, b, a)  # 逐窗独立滤波
            all_data.append(filtered)
            all_labels.append(label_class)
            all_file_ids.append(file_id)
            j += step
            window_count += 1
        
        print(f"     → 切分为 {window_count} 个窗口")
        file_id += 1
    
    if len(all_data) == 0:
        raise ValueError("未找到任何有效训练数据！")
    
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    all_file_ids = np.array(all_file_ids)
    
    print(f"\n{'='*60}")
    print(f"📊 总数据量: {all_data.shape}")
    print(f"📊 总窗口数: {len(all_data)}")
    print(f"📊 文件数: {file_id}")
    
    # 统计每个类别的窗口数
    for label_name, label_class in LABEL_TO_CLASS.items():
        count = np.sum(all_labels == label_class)
        print(f"   {label_name}: {count} 个窗口")
    
    return all_data, all_labels, all_file_ids

# ----------------------
# 按文件级别划分训练/测试集
# ----------------------
def split_by_files(data, labels, file_ids, test_ratio=0.2):
    """按文件级别划分，避免相邻重叠窗口造成数据泄露。"""
    unique_files = np.unique(file_ids)
    train_fids, test_fids = [], []
    
    for label_val in np.unique(labels):
        class_files = unique_files[np.isin(unique_files, np.unique(file_ids[labels == label_val]))]
        np.random.shuffle(class_files)
        n_train = max(1, int(len(class_files) * (1 - test_ratio)))
        train_fids.extend(class_files[:n_train].tolist())
        test_fids.extend(class_files[n_train:].tolist())
    
    train_mask = np.isin(file_ids, train_fids)
    test_mask = np.isin(file_ids, test_fids)
    
    print(f"\n📊 数据集划分:")
    print(f"   训练集: {len(train_fids)} 个文件 ({train_mask.sum()} 个窗口)")
    print(f"   测试集: {len(test_fids)} 个文件 ({test_mask.sum()} 个窗口)")
    
    return train_mask, test_mask

# ----------------------
# 归一化
# ----------------------
def compute_norm_stats(train_data):
    """计算训练集 per-channel 归一化统计。"""
    n_channels = train_data.shape[2]
    means = []
    stds = []
    for ch in range(n_channels):
        ch_data = train_data[:, :, ch].reshape(-1)
        means.append(float(np.mean(ch_data)))
        stds.append(float(np.std(ch_data)))
    return means, stds

def normalize_data(data, means, stds):
    """Per-channel Z-score 归一化。"""
    result = data.copy()
    for ch in range(len(means)):
        std = stds[ch] if stds[ch] > 1e-8 else 1.0
        result[:, :, ch] = (data[:, :, ch] - means[ch]) / std
    return result

# ----------------------
# 数据集类
# ----------------------
class EMGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ----------------------
# CNN1D 模型（支持动态类别数）
# ----------------------
class CNN1D(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv1d(8, 64, 10, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.AvgPool1d(5)
        self.drop1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(64, 96, 5, padding=1)
        self.bn2 = nn.BatchNorm1d(96)
        self.drop2 = nn.Dropout(0.3)
        
        self.conv3 = nn.Conv1d(96, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.3)
        
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        x = self.drop3(F.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        return self.fc(x)

# ----------------------
# 训练 & 评估
# ----------------------
def train(model, loader, criterion, optimizer, device):
    model.train()
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return 100 * correct / total

# ----------------------
# 主程序
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 Android App recordings 训练 EMG 模型")
    parser.add_argument("--recordings", type=str, default=RECORDINGS_DIR, help="recordings 目录路径")
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 EMG 手势识别模型训练")
    print("="*60)
    print(f"📂 数据目录: {args.recordings}")
    print(f"🖥️  设备: {config['device']}")
    print(f"📐 窗口大小: {config['window_size']}, 步长: {config['step_size']}")
    print(f"🔢 批次大小: {config['batch_size']}, 学习率: {config['learning_rate']}")
    print(f"⏱️  最大 Epoch: {config['num_epochs']}, 早停 Patience: {config['patience']}")
    print("="*60)
    
    # 1. 设计滤波器
    print("\n🔧 步骤 1: 设计滤波器...")
    b, a = design_filter(config["lowcut"], config["highcut"], config["fs"], config["order"])
    zi = lfilter_zi(b, a)
    
    with open(FILTER_COEFFS_PATH, "w") as f:
        json.dump({
            "b": [float(v) for v in b],
            "a": [float(v) for v in a],
            "zi": [float(v) for v in zi],
            "lowcut": config["lowcut"],
            "highcut": config["highcut"],
            "fs": config["fs"],
            "order": config["order"],
        }, f)
    print(f"   ✅ 滤波器系数已导出: {FILTER_COEFFS_PATH}")
    
    # 2. 加载数据
    print("\n📥 步骤 2: 加载数据...")
    data, labels, file_ids = load_recordings(args.recordings, config, b, a)
    
    # 3. 划分训练/测试集
    print("\n✂️  步骤 3: 划分数据集...")
    train_mask, test_mask = split_by_files(data, labels, file_ids, config["test_split"])
    
    train_data_raw, train_labels = data[train_mask], labels[train_mask]
    test_data_raw, test_labels = data[test_mask], labels[test_mask]
    
    # 4. 归一化
    print("\n📏 步骤 4: 归一化...")
    norm_means, norm_stds = compute_norm_stats(train_data_raw)
    train_data = normalize_data(train_data_raw, norm_means, norm_stds)
    test_data = normalize_data(test_data_raw, norm_means, norm_stds)
    
    with open(NORM_STATS_PATH, "w") as f:
        json.dump({"means": norm_means, "stds": norm_stds}, f)
    print(f"   ✅ 归一化统计已保存: {NORM_STATS_PATH}")
    print(f"   均值: {[round(m, 2) for m in norm_means]}")
    print(f"   标准差: {[round(s, 2) for s in norm_stds]}")
    
    # 5. 创建 DataLoader
    print("\n🔄 步骤 5: 创建 DataLoader...")
    train_data = np.swapaxes(train_data, 1, 2)  # (N, 100, 8) → (N, 8, 100)
    test_data = np.swapaxes(test_data, 1, 2)
    
    print(f"   训练集: {train_data.shape}")
    print(f"   测试集: {test_data.shape}")
    
    train_dataset = EMGDataset(train_data, train_labels)
    test_dataset = EMGDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # 6. 初始化模型
    print("\n🧠 步骤 6: 初始化模型...")
    num_classes = len(LABEL_TO_CLASS)
    model = CNN1D(num_classes=num_classes).to(config["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型参数量: {total_params:,}")
    
    # 7. 训练
    print("\n🏋️  步骤 7: 开始训练...")
    print("="*60)
    
    best_acc = 0
    best_epoch = 0
    patience_counter = 0
    history = []
    
    start_time = datetime.now()
    
    for epoch in range(config["num_epochs"]):
        loss = train(model, train_loader, criterion, optimizer, config["device"])
        acc = test(model, test_loader, config["device"])
        
        history.append({"epoch": epoch+1, "loss": loss, "acc": acc})
        
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), f"best_{SAVE_MODEL_PATH}")
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config['num_epochs']} | Loss: {loss:.4f} | Acc: {acc:.2f}% | Best: {best_acc:.2f}% (epoch {best_epoch})")
        
        if patience_counter >= config["patience"]:
            print(f"\n⏱️  早停触发：测试准确率 {config['patience']} 个 epoch 未提升")
            break
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # 保存最终模型（即使不是最佳）
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    
    print("="*60)
    print(f"\n✅ 训练完成！")
    print(f"   ⏱️  耗时: {elapsed:.1f} 秒")
    print(f"   🏆 最佳准确率: {best_acc:.2f}% (epoch {best_epoch})")
    print(f"   📊 最终 Loss: {history[-1]['loss']:.4f}")
    
    # 8. 导出 TorchScript 模型
    print("\n📦 步骤 8: 导出模型...")
    
    example_input = torch.randn(1, 8, config["window_size"], device=config["device"])
    
    # 导出最佳模型
    best_path = f"best_{SAVE_MODEL_PATH}"
    if os.path.exists(best_path):
        best_model = CNN1D(num_classes=num_classes).to(config["device"])
        best_model.load_state_dict(torch.load(best_path, map_location=config["device"]))
        best_model.eval()
        with torch.no_grad():
            scripted_best = torch.jit.trace(best_model, example_input)
        scripted_best.save("best_lx_model_scripted.pt")
        print(f"   ✅ 已导出: best_lx_model_scripted.pt")
    
    # 导出最终模型
    final_model = CNN1D(num_classes=num_classes).to(config["device"])
    final_model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=config["device"]))
    final_model.eval()
    with torch.no_grad():
        scripted = torch.jit.trace(final_model, example_input)
    scripted.save("lx_model_scripted.pt")
    print(f"   ✅ 已导出: lx_model_scripted.pt")
    
    # 9. 导出标签映射
    print("\n🏷️  步骤 9: 导出标签映射...")
    class_to_label = {v: k for k, v in LABEL_TO_CLASS.items()}
    model_info = {
        "labels": [class_to_label[i] for i in range(len(LABEL_TO_CLASS))],
        "num_classes": len(LABEL_TO_CLASS),
        "window_size": config["window_size"],
        "fs": config["fs"],
        "best_accuracy": best_acc,
        "best_epoch": best_epoch,
        "training_time": elapsed,
    }
    
    with open("model_info.json", "w") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    print(f"   ✅ 已导出: model_info.json")
    print(f"   标签顺序: {model_info['labels']}")
    
    # 10. 训练总结
    print("\n" + "="*60)
    print("📋 训练总结")
    print("="*60)
    print(f"   数据文件: {len(file_ids)} 个")
    print(f"   训练窗口: {train_mask.sum()} 个")
    print(f"   测试窗口: {test_mask.sum()} 个")
    print(f"   最佳准确率: {best_acc:.2f}%")
    print(f"   模型文件: lx_model_scripted.pt, best_lx_model_scripted.pt")
    print(f"   元数据: model_info.json, norm_stats.json, filter_coeffs.json")
    print("="*60)
    
    # 11. 诊断建议
    print("\n💡 诊断建议:")
    if best_acc >= 95:
        print("   ✅ 准确率优秀！模型可以直接使用。")
    elif best_acc >= 85:
        print("   ⚠️  准确率良好，但建议增加更多训练数据以提升泛化能力。")
    elif best_acc >= 70:
        print("   ⚠️  准确率一般，建议：")
        print("      - 每个动作采集更多次（建议8-10次）")
        print("      - 包含不同的力度、速度、角度")
        print("      - 检查数据质量（是否有噪声或丢帧）")
    else:
        print("   ❌ 准确率较低，建议：")
        print("      - 大幅增加训练数据量")
        print("      - 检查标签是否正确")
        print("      - 检查数据质量")
    
    if best_acc == 100.0:
        print("\n   ⚠️  注意：100% 准确率可能表示过拟合！")
        print("      建议：")
        print("      - 增加数据多样性")
        print("      - 使用更多不同的采集时间段")
        print("      - 测试未见过的数据验证泛化能力")