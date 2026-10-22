import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.signal import butter, filtfilt
import numpy as np
import time
import os

# ----------------------
# 配置参数
# ----------------------
DATA_DIR = "Sep_EMG_Data"
SAVE_MODEL_PATH = "lx_model.pth"
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
    "test_split": 0.2  # 20% 自动随机测试集
}

# ----------------------
# 滤波函数
# ----------------------
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data, axis=0)

# ----------------------
# ✅ 新版数据加载：全部读取、不手动分训练测试
# ----------------------
def load_all_data(data_dir, config):
    all_data = []
    all_labels = []

    gesture_map = {
        "gesture_1": 0,
        "gesture_2": 1,
        "gesture_3": 2,
        "gesture_4": 3
    }

    for folder, label in gesture_map.items():
        path = os.path.join(data_dir, folder)
        if not os.path.exists(path):
            continue

        for i in range(1, 6):
            file = os.path.join(path, f"{i}.txt")
            if not os.path.exists(file):
                continue
            print(f"加载：{folder}/{i}.txt")

            arr = np.loadtxt(file, delimiter=',', dtype=np.float32)
            if len(arr) > 500:
                arr = arr[250:-250]

            filtered = bandpass_filter(arr, config["lowcut"], config["highcut"], config["fs"], config["order"])

            ws = config["window_size"]
            step = config["step_size"]
            j = 0
            while j + ws <= len(filtered):
                all_data.append(filtered[j:j+ws])
                all_labels.append(label)
                j += step

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    all_data = np.swapaxes(all_data, 1, 2)  # (N, 100, 8) → (N,8,100)

    print(f"\n总数据量: {all_data.shape}")
    print(f"总标签数量: {all_labels.shape}")
    return all_data, all_labels

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
# 模型
# ----------------------
class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(8, 64, 10, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.AvgPool1d(5)

        self.conv2 = nn.Conv1d(64, 96, 5, padding=1)
        self.bn2 = nn.BatchNorm1d(96)

        self.conv3 = nn.Conv1d(96, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc = nn.Linear(2048, 4)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
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
# 主程序（PyTorch 随机划分）
# ----------------------
if __name__ == "__main__":
    # 1. 加载所有数据
    data, labels = load_all_data(DATA_DIR, config)
    
    # 2. 创建完整数据集
    dataset = EMGDataset(data, labels)
    
    # 3. ✅ PyTorch 自动随机划分 80%/20%
    test_size = int(len(dataset) * config["test_split"])
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 4. 加载器（训练集自动 shuffle！）
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # 5. 模型
    model = CNN1D().to(config["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # 6. 训练
    best_acc = 0
    for epoch in range(config["num_epochs"]):
        loss = train(model, train_loader, criterion, optimizer, config["device"])
        acc = test(model, test_loader, config["device"])
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"best_{SAVE_MODEL_PATH}")
        
        print(f"Epoch {epoch+1:2d} | Loss: {loss:.4f} | Acc: {acc:.2f}%")

    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"\n训练完成！最佳准确率: {best_acc:.2f}%")