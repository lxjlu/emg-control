# -*- coding: utf-8 -*-
"""
Baseline 模型: CNN-1D 和 Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================== CNN-1D ====================
class CNN1D(nn.Module):
    """
    标准1D CNN: (B, 8, 150) -> (B, 4)
    3层Conv-BN-ReLU-MaxPool + 全部分类头
    """
    def __init__(self, in_channels=8, num_classes=4, feat_dim=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feat_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.fc(x)


# ==================== Transformer ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Transformer1D(nn.Module):
    """
    Transformer baseline: (B, 8, 150) -> (B, 4)
    将每个通道+时间步视为序列元素
    """
    def __init__(self, in_channels=8, seq_len=150, num_classes=4,
                 d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len

        self.input_proj = nn.Conv1d(in_channels, d_model, kernel_size=1)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes))

    def forward(self, x):
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


def build_model(name="cnn1d", **kwargs):
    if name == "cnn1d":
        return CNN1D(**kwargs)
    elif name == "transformer":
        return Transformer1D(**kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")