"""
CNN-Transformer + Meta-Learning 训练脚本（两阶段）

阶段 1 - 基础模型训练（用所有用户数据）：
  - GRL 对抗域对齐：编码器产生用户不可区分的特征
  - 跨用户对比学习：同手势不同用户拉近
  - 产出：base_model.pt（通用基础权重）

阶段 2 - 用户微调（用当前用户数据 + 基础权重）：
  - 加载基础模型权重作为起点
  - 用当前用户数据微调
  - 产出：user_model.pt（个性化模型）

部署：python3.8 train_cnn_transformer.py --mode base   # 训练基础模型
      python3.8 train_cnn_transformer.py --mode finetune --pretrained base_model.pt  # 微调
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt, lfilter_zi
import numpy as np
import json
import os
import math
import argparse

# ----------------------
# 配置参数
# ----------------------
DATA_DIR = "Sep_EMG_Data"
FILTER_COEFFS_PATH = "filter_coeffs.json"
config = {
    "batch_size": 64,
    "learning_rate": 0.001,
    "finetune_lr": 0.0001,  # 微调学习率（更小）
    "num_epochs": 100,
    "finetune_epochs": 50,  # 微调 epoch 数
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "fs": 500,
    "lowcut": 20,
    "highcut": 249.999,
    "order": 4,
    "window_size": 100,
    "step_size": 20,
    "test_split": 0.2,
    "patience": 15,
    "finetune_patience": 10,
    # Meta-learning 参数
    "grl_lambda": 0.3,
    "grl_warmup_epochs": 5,
    "contra_weight": 0.5,
    "session_weight": 2.0,
    "temperature": 0.1,
    "meta_weight": 0.15,
}

# ----------------------
# 滤波函数
# ----------------------
def design_filter(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a

def bandpass_filter(data, b, a):
    return filtfilt(b, a, data, axis=0)

# ----------------------
# 数据加载
# ----------------------
def load_all_data(data_dir, config, b, a, subjects=None, meta_path=None):
    all_data = []
    all_labels = []
    all_file_ids = []
    all_user_ids = []
    all_session_ids = []
    all_gender = []
    all_weight_bucket = []
    file_id = 0
    user_id_map = {}
    user_counter = 0

    gesture_map = {
        "gesture_0": 0,
        "gesture_1": 1,
        "gesture_2": 2,
        "gesture_3": 3,
        "gesture_4": 4
    }

    # 左手用户标签交换：向右(3) ↔ 向左(4)
    SWAP_MAP = {3: 4, 4: 3}

    # 读取用户元数据：users 为性别/体重（用户级），records 为手别（记录级）
    user_meta = {}
    record_hand = {}
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_all = json.load(f)
            user_meta = meta_all.get("users", {})
            record_hand = meta_all.get("records", {})
        except Exception:
            pass

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
        if user_name not in user_id_map:
            user_id_map[user_name] = user_counter
            user_counter += 1
        uid = user_id_map[user_name]

        # 从 meta.json 获取用户元数据（性别/体重为用户级，手别为记录级）
        meta = user_meta.get(user_name, {})
        default_hand = (meta.get("hand") or "right").lower()
        gender_label = 0 if (meta.get("gender") or "male") == "male" else 1
        w = meta.get("weight")
        if w is None:
            wb = 0
        elif w < 55:
            wb = 0
        elif w < 70:
            wb = 1
        elif w < 85:
            wb = 2
        else:
            wb = 3

        # 检查是否有 session 子目录（新格式），还是直接 gesture_N（旧格式）
        entries = sorted(os.listdir(user_dir))
        session_dirs = [
            os.path.join(user_dir, d) for d in entries
            if os.path.isdir(os.path.join(user_dir, d)) and d.startswith("session_")
        ]

        if session_dirs:
            # 新格式：user/session_N/gesture_M/*.txt
            session_id_map = {}
            for sess_path in session_dirs:
                sess_name = os.path.basename(sess_path)
                if sess_name not in session_id_map:
                    session_id_map[sess_name] = len(session_id_map)
                sid = session_id_map[sess_name]
                for folder, label in gesture_map.items():
                    path = os.path.join(sess_path, folder)
                    if not os.path.exists(path):
                        continue
                    txt_files = sorted(
                        [f for f in os.listdir(path) if f.endswith(".txt") and f[:-4].isdigit()],
                        key=lambda f: int(f[:-4]),
                    )
                    for filename in txt_files:
                        file = os.path.join(path, filename)
                        # 记录级手别：优先按相对路径查记录，缺失时回退用户级默认
                        rel_key = f"{user_name}/{sess_name}/{folder}/{filename}"
                        is_left = record_hand.get(rel_key, default_hand) == "left"
                        arr = np.loadtxt(file, delimiter=',', dtype=np.float32)
                        if arr.ndim == 1:
                            arr = arr.reshape(1, -1)
                        if arr.ndim != 2 or arr.shape[1] == 0:
                            continue
                        ws = config["window_size"]
                        step = config["step_size"]
                        j = 0
                        while j + ws <= len(arr):
                            window = arr[j:j + ws]
                            filtered = bandpass_filter(window, b, a)
                            all_data.append(filtered)
                            cur_label = SWAP_MAP[label] if is_left and label in SWAP_MAP else label
                            all_labels.append(cur_label)
                            all_file_ids.append(file_id)
                            all_user_ids.append(uid)
                            all_session_ids.append(sid)
                            all_gender.append(gender_label)
                            all_weight_bucket.append(wb)
                            j += step
                        file_id += 1
        else:
            # 旧格式向后兼容：user/gesture_N/*.txt，session_id 统一为 0
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
                    # 记录级手别：优先按相对路径查记录，缺失时回退用户级默认
                    rel_key = f"{user_name}/{folder}/{filename}"
                    is_left = record_hand.get(rel_key, default_hand) == "left"
                    arr = np.loadtxt(file, delimiter=',', dtype=np.float32)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    if arr.ndim != 2 or arr.shape[1] == 0:
                        continue
                    ws = config["window_size"]
                    step = config["step_size"]
                    j = 0
                    while j + ws <= len(arr):
                        window = arr[j:j + ws]
                        filtered = bandpass_filter(window, b, a)
                        all_data.append(filtered)
                        cur_label = SWAP_MAP[label] if is_left and label in SWAP_MAP else label
                        all_labels.append(cur_label)
                        all_file_ids.append(file_id)
                        all_user_ids.append(uid)
                        all_session_ids.append(0)
                        all_gender.append(gender_label)
                        all_weight_bucket.append(wb)
                        j += step
                    file_id += 1

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    all_file_ids = np.array(all_file_ids)
    all_user_ids = np.array(all_user_ids)
    all_session_ids = np.array(all_session_ids)
    all_gender = np.array(all_gender, dtype=np.int64)
    all_weight_bucket = np.array(all_weight_bucket, dtype=np.int64)

    if len(all_data) == 0:
        raise ValueError("未找到任何训练数据！")

    num_users = len(user_id_map)
    num_sessions = len(np.unique(all_session_ids))
    print(f"\n总数据量: {all_data.shape}, 来自 {file_id} 个文件, "
          f"{num_users} 个用户, {num_sessions} 个 session")
    return (all_data, all_labels, all_file_ids, all_user_ids, all_session_ids,
            num_users, all_gender, all_weight_bucket)

# ----------------------
# 按文件级别划分训练/测试集
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
# 归一化
# ----------------------
def compute_norm_stats(train_data):
    n_channels = train_data.shape[2]
    means = []
    stds = []
    for ch in range(n_channels):
        ch_data = train_data[:, :, ch].reshape(-1)
        means.append(float(np.mean(ch_data)))
        stds.append(float(np.std(ch_data)))
    return means, stds

def normalize_data(data, means, stds):
    result = data.copy()
    for ch in range(len(means)):
        std = stds[ch] if stds[ch] > 1e-8 else 1.0
        result[:, :, ch] = (data[:, :, ch] - means[ch]) / std
    return result

# ----------------------
# 数据集类
# ----------------------
class EMGDataset(Dataset):
    def __init__(self, data, labels, user_ids=None, session_ids=None,
                 gender_ids=None, weight_bucket_ids=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.user_ids = torch.tensor(user_ids, dtype=torch.long) if user_ids is not None else None
        self.session_ids = torch.tensor(session_ids, dtype=torch.long) if session_ids is not None else None
        self.gender_ids = torch.tensor(gender_ids, dtype=torch.long) if gender_ids is not None else None
        self.weight_bucket_ids = torch.tensor(weight_bucket_ids, dtype=torch.long) if weight_bucket_ids is not None else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        g = self.gender_ids[idx] if self.gender_ids is not None else torch.tensor(-1, dtype=torch.long)
        w = self.weight_bucket_ids[idx] if self.weight_bucket_ids is not None else torch.tensor(-1, dtype=torch.long)
        if self.session_ids is not None:
            u = self.user_ids[idx] if self.user_ids is not None else torch.tensor(-1, dtype=torch.long)
            return self.data[idx], self.labels[idx], u, self.session_ids[idx], g, w
        if self.user_ids is not None:
            return self.data[idx], self.labels[idx], self.user_ids[idx], g, w
        return self.data[idx], self.labels[idx], g, w

# ----------------------
# CNN-Transformer 模型
# ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class GRLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GRLFunction.apply(x, lambda_)


class CNNTransformer(nn.Module):
    def __init__(self, num_classes=5, num_users=1, use_grl=False, use_contra=False,
                 use_meta=False, num_gender=2, num_weight_bucket=4):
        super().__init__()
        self.use_grl = use_grl
        self.use_contra = use_contra
        self.use_meta = use_meta
        self.num_users = num_users
        self.num_gender = num_gender
        self.num_weight_bucket = num_weight_bucket

        # CNN 特征提取
        self.conv1 = nn.Conv1d(8, 64, 10, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.AvgPool1d(5)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(64, 96, 5, padding=1)
        self.bn2 = nn.BatchNorm1d(96)
        self.drop2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv1d(96, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # Transformer 编码器
        d_model = 128
        nhead = 4
        num_layers = 2
        dim_feedforward = 256
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = PositionalEncoding(d_model, max_len=100)

        # 分类头
        self.fc = nn.Linear(d_model, num_classes)

        # GRL 域判别器
        if use_grl and num_users > 1:
            self.domain_classifier = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, num_users)
            )

        # 对比学习投影头
        if use_contra:
            self.contra_head = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )

        # 元数据域判别器（性别+体重桶，GRL 反向梯度让特征对元数据无关）
        if use_meta:
            self.meta_classifier = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, num_gender + num_weight_bucket)
            )

    def forward(self, x, user_ids=None, grl_lambda=0.0, return_feat=False):
        # CNN
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        # Transformer
        x = x.transpose(1, 2)
        x = self.pos_encoding(x)
        x = self.transformer(x)

        feat = x.mean(dim=1)
        logits = self.fc(feat)

        if return_feat:
            return logits, feat

        out = {'logits': logits}

        if self.use_grl and user_ids is not None and grl_lambda > 0 and hasattr(self, 'domain_classifier'):
            reversed_feat = grad_reverse(feat, grl_lambda)
            out['domain_logits'] = self.domain_classifier(reversed_feat)

        if self.use_contra and hasattr(self, 'contra_head'):
            out['contra_feat'] = self.contra_head(feat)

        if self.use_meta and hasattr(self, 'meta_classifier') and grl_lambda > 0:
            reversed_feat = grad_reverse(feat, grl_lambda)
            out['meta_logits'] = self.meta_classifier(reversed_feat)

        return out


# ----------------------
# 对比学习损失
# ----------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, session_weight=2.0):
        super().__init__()
        self.temperature = temperature
        self.session_weight = session_weight

    def forward(self, features, labels, session_ids=None, user_ids=None):
        B = features.size(0)
        features = F.normalize(features, dim=1)
        sim_matrix = torch.mm(features, features.t()) / self.temperature

        base_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        base_mask.fill_diagonal_(0)

        if (session_ids is not None and user_ids is not None
                and self.session_weight > 1.0):
            same_user = (user_ids.unsqueeze(0) == user_ids.unsqueeze(1)).float()
            diff_session = (session_ids.unsqueeze(0) != session_ids.unsqueeze(1)).float()
            session_bonus = base_mask * same_user * diff_session
            pos_mask = base_mask + session_bonus * (self.session_weight - 1)
        else:
            pos_mask = base_mask

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        exp_sim = torch.exp(sim_matrix)
        exp_sim = exp_sim * (1 - torch.eye(B, device=features.device))
        loss = 0
        for i in range(B):
            pos_sum = (exp_sim[i] * pos_mask[i]).sum()
            all_sum = exp_sim[i].sum()
            if pos_sum > 0:
                loss -= torch.log(pos_sum / (all_sum + 1e-8) + 1e-8)

        return loss / max(pos_mask.sum().item(), 1)


# ----------------------
# 模型保存辅助
# ----------------------
def build_inference_state(model):
    """生成可用于推理的 state_dict：剔除对比学习头/域分类器等辅助头（base 模式训练时才有），
    使 checkpoint 与 use_grl=False, use_contra=False 的推理模型 key 完全对齐。"""
    drop_prefixes = ("contra_head.", "domain_classifier.", "meta_classifier.")
    return {k: v for k, v in model.state_dict().items() if not k.startswith(drop_prefixes)}


class _InferenceModule(nn.Module):
    """导出 TorchScript 用包装：CNNTransformer.forward 返回 dict，手机端无法直接取张量。
    这里只返回 logits，保证 trace 出的 .pt 前向输出是一个纯 Tensor。"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)['logits']


# ----------------------
# 训练 & 评估
# ----------------------
def train(model, loader, criterion, optimizer, device,
          use_grl=False, grl_lambda=0.0, domain_weight=0.1,
          use_contra=False, contra_weight=0.5, contra_criterion=None,
          use_meta=False, meta_weight=0.15):
    model.train()
    total_loss, total = 0, 0
    for batch in loader:
        if len(batch) == 6:
            x, y, u, s, g, w = batch
            x, y, u, s = x.to(device), y.to(device), u.to(device), s.to(device)
            g, w = g.to(device), w.to(device)
        elif len(batch) == 4:
            x, y, u, s = batch
            x, y, u, s = x.to(device), y.to(device), u.to(device), s.to(device)
            g = w = None
        elif len(batch) == 3:
            x, y, u = batch
            x, y, u = x.to(device), y.to(device), u.to(device)
            s = None
            g = w = None
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            u = s = None
            g = w = None

        optimizer.zero_grad()
        out = model(x, user_ids=u, grl_lambda=grl_lambda if use_grl else 0.0)

        loss = criterion(out['logits'], y)

        if use_grl and 'domain_logits' in out and u is not None and grl_lambda > 0:
            loss_domain = F.cross_entropy(out['domain_logits'], u)
            loss = loss + domain_weight * loss_domain

        if use_contra and 'contra_feat' in out:
            loss_contra = contra_criterion(out['contra_feat'], y,
                                           session_ids=s, user_ids=u)
            loss = loss + contra_weight * loss_contra

        if use_meta and 'meta_logits' in out and g is not None and meta_weight > 0:
            gender_logits = out['meta_logits'][:, :model.num_gender]
            weight_logits = out['meta_logits'][:, model.num_gender:]
            valid = g >= 0
            if valid.any():
                loss_meta = (F.cross_entropy(gender_logits[valid], g[valid])
                             + F.cross_entropy(weight_logits[valid], w[valid]))
                loss = loss + meta_weight * loss_meta

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total += x.size(0)

    return total_loss / max(total, 1)


def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out['logits'].argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return 100 * correct / max(total, 1)


# ----------------------
# 主程序
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 CNN-Transformer + Meta-Learning 模型（两阶段）")
    parser.add_argument("--mode", type=str, default="base", choices=["base", "finetune"],
                        help="训练模式：base=基础模型（所有用户），finetune=用户微调")
    parser.add_argument("--subject", action="append", default=None,
                        help="指定训练用户（finetune 模式用）")
    parser.add_argument("--no_grl", action="store_true", help="禁用 GRL 对抗")
    parser.add_argument("--no_contra", action="store_true", help="禁用对比学习")
    parser.add_argument("--no_meta", action="store_true", help="禁用元数据 GRL（性别/体重不变性）")
    parser.add_argument("--grl_lambda", type=float, default=0.3, help="GRL 强度")
    parser.add_argument("--contra_weight", type=float, default=0.5, help="对比学习权重")
    parser.add_argument("--meta_weight", type=float, default=0.15, help="元数据 GRL 权重")
    parser.add_argument("--pretrained", type=str, default=None, help="加载预训练权重（finetune 模式用）")
    args = parser.parse_args()

    mode = args.mode
    subjects = None
    env_users = os.environ.get("TRAIN_USERS")
    if env_users:
        subjects = [s.strip() for s in env_users.split(",") if s.strip()]
    elif args.subject:
        subjects = args.subject

    use_grl = not args.no_grl
    use_contra = not args.no_contra
    use_meta = not args.no_meta
    config["grl_lambda"] = args.grl_lambda
    config["contra_weight"] = args.contra_weight
    config["meta_weight"] = args.meta_weight

    print("=" * 60)
    print(f"训练模式: {mode}")
    print(f"Meta-Learning: GRL={use_grl}(λ={config['grl_lambda']}) "
          f"Contra={use_contra}(w={config['contra_weight']}) "
          f"Meta={use_meta}(w={config['meta_weight']})")
    if mode == "finetune":
        print(f"微调学习率: {config['finetune_lr']}")
        print(f"预训练权重: {args.pretrained}")
    print("=" * 60)

    # 1. 设计滤波器
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

    # 2. 加载数据
    meta_path = os.path.join(DATA_DIR, "meta.json")
    (data, labels, file_ids, user_ids, session_ids,
     num_users, all_gender, all_weight_bucket) = load_all_data(
        DATA_DIR, config, b, a, subjects=subjects, meta_path=meta_path)

    # 3. 划分数据集
    train_mask, test_mask = split_by_files(data, labels, file_ids, config["test_split"])

    train_data_raw, train_labels = data[train_mask], labels[train_mask]
    test_data_raw, test_labels = data[test_mask], labels[test_mask]
    train_user_ids = user_ids[train_mask] if len(user_ids) > 0 else None
    test_user_ids = user_ids[test_mask] if len(user_ids) > 0 else None
    train_session_ids = session_ids[train_mask] if len(session_ids) > 0 else None
    test_session_ids = session_ids[test_mask] if len(session_ids) > 0 else None
    train_gender = all_gender[train_mask] if len(all_gender) > 0 else None
    test_gender = all_gender[test_mask] if len(all_gender) > 0 else None
    train_weight_bucket = all_weight_bucket[train_mask] if len(all_weight_bucket) > 0 else None
    test_weight_bucket = all_weight_bucket[test_mask] if len(all_weight_bucket) > 0 else None

    # 4. 归一化
    norm_means, norm_stds = compute_norm_stats(train_data_raw)
    train_data = normalize_data(train_data_raw, norm_means, norm_stds)
    test_data = normalize_data(test_data_raw, norm_means, norm_stds)

    with open("norm_stats.json", "w") as f:
        json.dump({"means": norm_means, "stds": norm_stds}, f)
    print(f"归一化统计已保存: means={[round(m, 2) for m in norm_means]}, stds={[round(s, 2) for s in norm_stds]}")

    # 5. DataLoader
    train_data = np.swapaxes(train_data, 1, 2)
    test_data = np.swapaxes(test_data, 1, 2)

    print(f"训练集: {train_data.shape}, 测试集: {test_data.shape}")

    train_dataset = EMGDataset(train_data, train_labels, train_user_ids, train_session_ids,
                               train_gender, train_weight_bucket)
    test_dataset = EMGDataset(test_data, test_labels, test_user_ids, test_session_ids,
                              test_gender, test_weight_bucket)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # 6. 模型
    num_classes = 5
    
    if mode == "base":
        # 基础模型训练：启用 GRL 和对比学习和元数据 GRL
        model = CNNTransformer(
            num_classes=num_classes,
            num_users=num_users,
            use_grl=use_grl and num_users > 1,
            use_contra=use_contra,
            use_meta=use_meta and num_users > 1
        ).to(config["device"])
        SAVE_MODEL_PATH = "lx_model_transformer.pth"
    else:
        # 微调模式：不启用 GRL 和对比学习（单用户数据不够）
        model = CNNTransformer(
            num_classes=num_classes,
            num_users=num_users,
            use_grl=False,
            use_contra=False
        ).to(config["device"])
        SAVE_MODEL_PATH = "lx_model_transformer_finetuned.pth"
        
        # 加载预训练权重
        if args.pretrained and os.path.exists(args.pretrained):
            state = torch.load(args.pretrained, map_location=config["device"], weights_only=False)
            model_state = model.state_dict()
            loaded, skipped = 0, 0
            for k, v in state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    model_state[k] = v
                    loaded += 1
                else:
                    skipped += 1
            model.load_state_dict(model_state)
            print(f"[预训练] 加载 {loaded} 个参数，跳过 {skipped} 个")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    
    if mode == "base":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
        contra_criterion = ContrastiveLoss(
            temperature=config["temperature"],
            session_weight=config["session_weight"]
        ) if use_contra else None
        num_epochs = config["num_epochs"]
        patience = config["patience"]
    else:
        # 微调模式：更小的学习率，只微调部分层
        optimizer = torch.optim.Adam(model.parameters(), lr=config["finetune_lr"], weight_decay=1e-4)
        contra_criterion = None
        num_epochs = config["finetune_epochs"]
        patience = config["finetune_patience"]

    # 7. 训练
    best_acc = 0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        # GRL warmup
        current_grl_lambda = 0.0
        if mode == "base" and use_grl and epoch >= config["grl_warmup_epochs"]:
            current_grl_lambda = config["grl_lambda"]

        loss = train(
            model, train_loader, criterion, optimizer, config["device"],
            use_grl=(mode == "base" and use_grl), grl_lambda=current_grl_lambda,
            use_contra=(mode == "base" and use_contra), contra_weight=config["contra_weight"],
            contra_criterion=contra_criterion,
            use_meta=(mode == "base" and use_meta), meta_weight=config["meta_weight"]
        )
        acc = test(model, test_loader, config["device"])

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(build_inference_state(model), f"best_{SAVE_MODEL_PATH}")
        else:
            patience_counter += 1

        grl_info = f" GRL_λ={current_grl_lambda:.2f}" if (mode == "base" and use_grl) else ""
        print(f"Epoch {epoch+1:2d} | Loss: {loss:.4f} | Acc: {acc:.2f}%{grl_info}")

        if patience_counter >= patience:
            print(f"早停：测试准确率 {patience} 个 epoch 未提升")
            break

    torch.save(build_inference_state(model), SAVE_MODEL_PATH)
    print(f"\n训练完成！最佳准确率: {best_acc:.2f}% (epoch {best_epoch})")

    # 8. 导出 TorchScript
    example_input = torch.randn(1, 8, config["window_size"])

    best_path = f"best_{SAVE_MODEL_PATH}"
    if os.path.exists(best_path):
        best_model = CNNTransformer(
            num_classes=num_classes, num_users=num_users,
            use_grl=False, use_contra=False, use_meta=False
        )
        best_model.load_state_dict(torch.load(best_path, map_location="cpu"), strict=False)
        best_model.eval()
        with torch.no_grad():
            scripted_best = torch.jit.trace(_InferenceModule(best_model), example_input, strict=False)
        scripted_best.save("best_lx_model_scripted.pt")
        print(f"已导出 best_lx_model_scripted.pt")

    final_model = CNNTransformer(
        num_classes=num_classes, num_users=num_users,
        use_grl=False, use_contra=False, use_meta=False
    )
    final_model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location="cpu"), strict=False)
    final_model.eval()
    with torch.no_grad():
        scripted = torch.jit.trace(_InferenceModule(final_model), example_input, strict=False)
    scripted.save("lx_model_scripted.pt")

    print("已导出 TorchScript 模型：lx_model_scripted.pt, best_lx_model_scripted.pt")