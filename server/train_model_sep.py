import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt, lfilter_zi
import numpy as np
import json
import os
import argparse

# ----------------------
# 配置参数
# ----------------------
DATA_DIR = "Sep_EMG_Data"
SAVE_MODEL_PATH = "lx_model.pth"
FILTER_COEFFS_PATH = "filter_coeffs.json"
config = {
    "batch_size": 128,
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

# ----------------------
# 滤波函数
#
# 重要：训练与推理必须使用【完全相同的 b/a 系数】。
# 这里用 scipy 设计，并把系数导出给 Android 端直接使用，
# 避免 Android 端自行重算时漏掉 scipy 的增益归一化（会导致幅度差约 1.9 倍）。
# ----------------------
def design_filter(lowcut, highcut, fs, order):
    """用 scipy 设计带通滤波器，返回 (b, a)。"""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a

def bandpass_filter(data, b, a):
    """零相位滤波，与 Android 端 ButterworthFilter.filtfilt 对齐。

    data: (window, channels)，沿 axis=0 对每个通道独立滤波。
    """
    return filtfilt(b, a, data, axis=0)

# ----------------------
# 数据加载：先切窗，再对每个窗口独立滤波
#
# 与 Android 推理流程严格对齐：Android 端每次只拿到 100 点窗口，
# 对该窗口单独做 filtfilt。若训练端改成"整段滤波后切窗"，边缘处
# 结果会不一致，因此这里同样按窗口独立滤波。
# ----------------------
def load_all_data(data_dir, config, b, a, subjects=None):
    all_data = []
    all_labels = []
    all_file_ids = []
    file_id = 0

    gesture_map = {
        "gesture_0": 0,  # 休息状态
        "gesture_1": 1,
        "gesture_2": 2,
        "gesture_3": 3,
        "gesture_4": 4
    }

    if subjects:
        user_dirs = [os.path.join(data_dir, s) for s in subjects]
    else:
        if not os.path.exists(data_dir):
            user_dirs = []
        else:
            user_dirs = [
                os.path.join(data_dir, d)
                for d in sorted(os.listdir(data_dir))
                if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith("gesture_")
            ]

    for user_dir in user_dirs:
        if not os.path.isdir(user_dir):
            continue
        user_name = os.path.basename(user_dir)
        for folder, label in gesture_map.items():
            path = os.path.join(user_dir, folder)
            if not os.path.exists(path):
                continue

            txt_files = sorted(
                [f for f in os.listdir(path) if f.endswith(".txt") and f[:-4].isdigit()],
                key=lambda f: int(f[:-4]),
            )
            for filename in txt_files:
                file = os.path.join(path, filename)
                print(f"加载：{user_name}/{folder}/{filename}")

                arr = np.loadtxt(file, delimiter=',', dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.ndim != 2 or arr.shape[1] == 0:
                    print(f"  跳过（格式异常）：shape={arr.shape}")
                    continue
                # 注释掉裁剪逻辑，保留全部数据用于训练
                # if len(arr) > 500:
                #     arr = arr[250:-250]

                ws = config["window_size"]
                step = config["step_size"]
                j = 0
                while j + ws <= len(arr):
                    window = arr[j:j + ws]                      # (100, 8)
                    filtered = bandpass_filter(window, b, a)     # 逐窗独立滤波
                    all_data.append(filtered)
                    all_labels.append(label)
                    all_file_ids.append(file_id)
                    j += step
                file_id += 1

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    all_file_ids = np.array(all_file_ids)

    if len(all_data) == 0:
        raise ValueError("未找到任何训练数据！请检查数据目录是否正确，以及是否已采集数据。")

    print(f"\n总数据量: {all_data.shape}, 来自 {file_id} 个文件")
    return all_data, all_labels, all_file_ids

# ----------------------
# 按文件级别划分训练/测试集（避免相邻重叠窗口造成数据泄露）
# ----------------------
def split_by_files(data, labels, file_ids, test_ratio=0.2):
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

    print(f"文件级划分: 训练 {len(train_fids)} 个文件 ({train_mask.sum()} 窗口), "
          f"测试 {len(test_fids)} 个文件 ({test_mask.sum()} 窗口)")
    return train_mask, test_mask

# ----------------------
# 计算训练集 per-channel 归一化统计
# ----------------------
def compute_norm_stats(train_data):
    # train_data: (N, 100, 8)
    n_channels = train_data.shape[2]
    means = []
    stds = []
    for ch in range(n_channels):
        ch_data = train_data[:, :, ch].reshape(-1)
        means.append(float(np.mean(ch_data)))
        stds.append(float(np.std(ch_data)))
    return means, stds

def normalize_data(data, means, stds):
    # data: (N, 100, 8), means/stds: list of length 8
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
# 模型（含 Dropout 防止过拟合）
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
    parser = argparse.ArgumentParser(description="训练 v01 EMG 动作识别模型")
    parser.add_argument(
        "--subject",
        action="append",
        default=None,
        help="指定训练用户，可多次指定；未指定则使用全部用户",
    )
    args = parser.parse_args()

    subjects = None
    env_users = os.environ.get("TRAIN_USERS")
    if env_users:
        subjects = [s.strip() for s in env_users.split(",") if s.strip()]
    elif args.subject:
        subjects = args.subject

    if subjects:
        print(f"训练用户: {', '.join(subjects)}")
    else:
        print("训练全部用户")

    # 1. 设计滤波器（训练与推理共用同一组系数，并导出给 Android）
    b, a = design_filter(config["lowcut"], config["highcut"], config["fs"], config["order"])
    # zi 是 scipy filtfilt 使用的稳态初始条件（lfilter_zi），
    # 一并导出，手机端据此复现与 scipy 完全一致的滤波行为。
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
    print(f"滤波器系数已导出 {FILTER_COEFFS_PATH} (b:{len(b)}, a:{len(a)}, zi:{len(zi)})")

    # 2. 加载数据（逐窗独立滤波）
    data, labels, file_ids = load_all_data(DATA_DIR, config, b, a, subjects=subjects)

    # 3. 按文件级别划分（避免相邻重叠窗口同时出现在训练和测试集）
    train_mask, test_mask = split_by_files(data, labels, file_ids, config["test_split"])

    train_data_raw, train_labels = data[train_mask], labels[train_mask]
    test_data_raw, test_labels = data[test_mask], labels[test_mask]

    # 4. 基于训练集计算 per-channel Z-score 统计
    norm_means, norm_stds = compute_norm_stats(train_data_raw)
    train_data = normalize_data(train_data_raw, norm_means, norm_stds)
    test_data = normalize_data(test_data_raw, norm_means, norm_stds)

    with open("norm_stats.json", "w") as f:
        json.dump({"means": norm_means, "stds": norm_stds}, f)
    print(f"归一化统计已保存: means={[round(m, 2) for m in norm_means]}, stds={[round(s, 2) for s in norm_stds]}")

    # 5. 转为 (N, 8, 100) 并创建 DataLoader
    train_data = np.swapaxes(train_data, 1, 2)
    test_data = np.swapaxes(test_data, 1, 2)

    print(f"训练集: {train_data.shape}, 测试集: {test_data.shape}")

    train_dataset = EMGDataset(train_data, train_labels)
    test_dataset = EMGDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # 6. 模型
    num_classes = len(gesture_map)
    model = CNN1D(num_classes=num_classes).to(config["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)

    # 7. 训练（带早停）
    best_acc = 0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(config["num_epochs"]):
        loss = train(model, train_loader, criterion, optimizer, config["device"])
        acc = test(model, test_loader, config["device"])

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), f"best_{SAVE_MODEL_PATH}")
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1:2d} | Loss: {loss:.4f} | Acc: {acc:.2f}%")

        if patience_counter >= config["patience"]:
            print(f"早停：测试准确率 {config['patience']} 个 epoch 未提升")
            break

    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"\n训练完成！最佳准确率: {best_acc:.2f}% (epoch {best_epoch})")

    # 8. 导出 TorchScript 模型
    #    统一在 CPU 上导出：Android (PyTorch Mobile CPU) 只能加载 CPU 模型，
    #    example_input 与模型保持同一 device。
    example_input = torch.randn(1, 8, config["window_size"])

    best_path = f"best_{SAVE_MODEL_PATH}"
    if os.path.exists(best_path):
        best_model = CNN1D(num_classes=num_classes)
        best_model.load_state_dict(torch.load(best_path, map_location="cpu"))
        best_model.eval()
        with torch.no_grad():
            scripted_best = torch.jit.trace(best_model, example_input)
        scripted_best.save("best_lx_model_scripted.pt")
        print("已导出 best_lx_model_scripted.pt")

    final_model = CNN1D(num_classes=num_classes)
    final_model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location="cpu"))
    final_model.eval()
    with torch.no_grad():
        scripted = torch.jit.trace(final_model, example_input)
    scripted.save("lx_model_scripted.pt")

    print("已导出 TorchScript 模型：lx_model_scripted.pt, best_lx_model_scripted.pt")