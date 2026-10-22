# -*- coding: utf-8 -*-
"""
改进模型 v2: 用户无关手势表征学习
核心技术:
1. EMG+IMU 多模态融合
2. 域对抗训练 (DANN) - 强制学习用户无关特征
3. 跨用户对比学习 - 同手势不同用户特征拉近
4. 早停机制支持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================== 梯度反转层 (GRL) ====================
class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReverseLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReverse.apply(x, self.lambda_)


# ==================== EMG Encoder ====================
class EMGEncoder(nn.Module):
    """
    80通道EMG编码器
    使用分组卷积减少参数量，同时保持表达能力
    """
    def __init__(self, in_channels=80, feat_dim=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 128, 7, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.MaxPool1d(2))
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, feat_dim, 3, padding=1),
            nn.BatchNorm1d(feat_dim), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.flatten(1)


# ==================== IMU Encoder ====================
class IMUEncoder(nn.Module):
    """
    6通道IMU编码器 (3轴加速度 + 3轴角速度)
    IMU信号更用户无关，提供运动学先验
    """
    def __init__(self, in_channels=6, feat_dim=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, 7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2))
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, feat_dim, 3, padding=1),
            nn.BatchNorm1d(feat_dim), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.flatten(1)


# ==================== 多模态融合 ====================
class MultiModalFusion(nn.Module):
    """
    EMG + IMU 特征融合
    使用attention加权融合，让模型自动学习哪个模态更可靠
    """
    def __init__(self, emg_dim=256, imu_dim=128, fused_dim=256):
        super().__init__()
        self.emg_proj = nn.Linear(emg_dim, fused_dim)
        self.imu_proj = nn.Linear(imu_dim, fused_dim)
        self.attention = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.Tanh(),
            nn.Linear(fused_dim, 2),
            nn.Softmax(dim=1))
        self.fusion_bn = nn.BatchNorm1d(fused_dim)

    def forward(self, emg_feat, imu_feat):
        emg_p = self.emg_proj(emg_feat)
        imu_p = self.imu_proj(imu_feat)
        combined = torch.cat([emg_p, imu_p], dim=1)
        attn_weights = self.attention(combined)
        fused = attn_weights[:, 0:1] * emg_p + attn_weights[:, 1:2] * imu_p
        return self.fusion_bn(fused)


# ==================== 完整模型 (DANN + 对比学习) ====================
class UserInvariantModel(nn.Module):
    """
    用户无关手势识别模型
    包含:
    1. EMG编码器
    2. IMU编码器
    3. 多模态融合
    4. 手势分类头
    5. 用户分类头 (域对抗)
    """
    def __init__(self, num_classes=4, num_users=230,
                 emg_channels=80, imu_channels=6,
                 emg_dim=256, imu_dim=128, fused_dim=256,
                 grl_lambda=1.0, use_imu=True, dropout=0.0):
        super().__init__()
        self.use_imu = use_imu
        self.emg_encoder = EMGEncoder(in_channels=emg_channels, feat_dim=emg_dim)

        if use_imu:
            self.imu_encoder = IMUEncoder(in_channels=imu_channels, feat_dim=imu_dim)
            self.fusion = MultiModalFusion(emg_dim, imu_dim, fused_dim)
        else:
            self.imu_encoder = None
            self.fusion = None
            self.emg_proj = nn.Linear(emg_dim, fused_dim)
            self.fusion_bn = nn.BatchNorm1d(fused_dim)

        # 手势分类头
        self.gesture_classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, num_classes))

        # 域对抗头 (用户分类)
        self.grl = GradientReverseLayer(lambda_=grl_lambda)
        self.user_classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_users))

        self.fused_dim = fused_dim

    def forward(self, x_emg, x_imu=None, label=None, stage='train'):
        emg_feat = self.emg_encoder(x_emg)

        if self.use_imu and x_imu is not None:
            imu_feat = self.imu_encoder(x_imu)
            fused = self.fusion(emg_feat, imu_feat)
        else:
            fused = self.fusion_bn(self.emg_proj(emg_feat))

        gesture_logits = self.gesture_classifier(fused)

        if stage == 'train':
            # 域对抗: 梯度反转后分类用户
            fused_grl = self.grl(fused)
            user_logits = self.user_classifier(fused_grl)
            return {
                'gesture_logits': gesture_logits,
                'user_logits': user_logits,
                'feat': fused,
            }
        else:
            return gesture_logits

    def get_features(self, x_emg, x_imu=None):
        """提取融合特征用于对比学习"""
        emg_feat = self.emg_encoder(x_emg)
        if self.use_imu and x_imu is not None:
            imu_feat = self.imu_encoder(x_imu)
            return self.fusion(emg_feat, imu_feat)
        else:
            return self.fusion_bn(self.emg_proj(emg_feat))


# ==================== 特征解耦模型 (Disentangled) ====================
class DisentangledModel(nn.Module):
    """
    特征解耦模型
    将特征显式分离为:
    1. 手势特征 (只包含手势信息)
    2. 用户特征 (只包含用户信息)
    通过互信息最小化确保两个特征独立
    """
    def __init__(self, num_classes=4, num_users=230,
                 emg_channels=80, imu_channels=6,
                 emg_dim=256, imu_dim=128, gesture_dim=128, user_dim=128,
                 use_imu=True, mi_loss_type='hsic', dropout=0.0):
        super().__init__()
        self.use_imu = use_imu
        self.mi_loss_type = mi_loss_type

        # 共享编码器
        self.emg_encoder = EMGEncoder(in_channels=emg_channels, feat_dim=emg_dim)
        if use_imu:
            self.imu_encoder = IMUEncoder(in_channels=imu_channels, feat_dim=imu_dim)
            self.fusion = MultiModalFusion(emg_dim, imu_dim, emg_dim)
            input_dim = emg_dim
        else:
            self.imu_encoder = None
            self.fusion = None
            input_dim = emg_dim

        # 解耦投影头
        self.gesture_head = nn.Sequential(
            nn.Linear(input_dim, gesture_dim),
            nn.ReLU(),
            nn.BatchNorm1d(gesture_dim))

        self.user_head = nn.Sequential(
            nn.Linear(input_dim, user_dim),
            nn.ReLU(),
            nn.BatchNorm1d(user_dim))

        # 分类器
        self.gesture_classifier = nn.Sequential(
            nn.Linear(gesture_dim, gesture_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gesture_dim, num_classes))

        self.user_classifier = nn.Sequential(
            nn.Linear(user_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_users))

        self.gesture_dim = gesture_dim
        self.user_dim = user_dim

    def forward(self, x_emg, x_imu=None, label=None, stage='train'):
        # 提取共享特征
        emg_feat = self.emg_encoder(x_emg)
        if self.use_imu and x_imu is not None:
            imu_feat = self.imu_encoder(x_imu)
            fused = self.fusion(emg_feat, imu_feat)
        else:
            fused = emg_feat

        # 解耦
        gesture_feat = self.gesture_head(fused)
        user_feat = self.user_head(fused)

        # 分类
        gesture_logits = self.gesture_classifier(gesture_feat)

        if stage == 'train':
            user_logits = self.user_classifier(user_feat)
            return {
                'gesture_logits': gesture_logits,
                'user_logits': user_logits,
                'gesture_feat': gesture_feat,
                'user_feat': user_feat,
            }
        else:
            return gesture_logits

    def get_features(self, x_emg, x_imu=None):
        """提取手势特征用于对比学习"""
        emg_feat = self.emg_encoder(x_emg)
        if self.use_imu and x_imu is not None:
            imu_feat = self.imu_encoder(x_imu)
            fused = self.fusion(emg_feat, imu_feat)
        else:
            fused = emg_feat
        return self.gesture_head(fused)


def hsic_loss(x, y, sigma_x=None, sigma_y=None):
    """
    HSIC (Hilbert-Schmidt Independence Criterion) 损失
    使用RBF核，比线性核更敏感
    """
    n = x.size(0)
    if n < 2:
        return torch.tensor(0.0, device=x.device)

    # 计算距离矩阵
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    yy = torch.sum(y ** 2, dim=1, keepdim=True)
    dist_xx = torch.clamp(xx + xx.T - 2 * torch.matmul(x, x.T), min=0.0)
    dist_yy = torch.clamp(yy + yy.T - 2 * torch.matmul(y, y.T), min=0.0)

    # 使用中位数启发式设置sigma
    if sigma_x is None:
        sigma_x = torch.median(torch.sqrt(torch.clamp(dist_xx, min=1e-12)))
    if sigma_y is None:
        sigma_y = torch.median(torch.sqrt(torch.clamp(dist_yy, min=1e-12)))

    # RBF核
    K = torch.exp(-dist_xx / (2 * sigma_x ** 2 + 1e-12))
    L = torch.exp(-dist_yy / (2 * sigma_y ** 2 + 1e-12))

    # 中心化矩阵
    H = torch.eye(n, device=x.device) - 1.0 / n

    # HSIC
    K_centered = torch.matmul(H, torch.matmul(K, H))
    L_centered = torch.matmul(H, torch.matmul(L, H))

    hsic = torch.sum(K_centered * L_centered) / ((n - 1) ** 2)
    return torch.clamp(hsic, min=0.0, max=100.0)


def mutual_information_loss(gesture_feat, user_feat):
    """
    互信息损失: 使用HSIC衡量手势特征和用户特征的独立性
    """
    return hsic_loss(gesture_feat, user_feat)


# ==================== 纯CNN-1D Baseline (8通道) ====================
class CNN1D(nn.Module):
    def __init__(self, in_channels=8, num_classes=4, feat_dim=256, dropout=0.0):
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
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.fc(x)


# ==================== Transformer Baseline ====================
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
    elif name == "dann":
        return UserInvariantModel(**kwargs)
    elif name == "disentangled":
        return DisentangledModel(**kwargs)
    elif name == "imu_attn":
        return IMUAnchoredAttention(**kwargs)
    elif name == "imu_decompose":
        return IMUAnchoredDecomposition(**kwargs)
    elif name == "imu_crossmodal":
        return IMUCrossModalContrastive(**kwargs)
    elif name == "proto_contra":
        return PrototypeContrastive(**kwargs)
    elif name == "memory_contra":
        return MemoryBankContrastive(**kwargs)
    elif name == "user_norm_contra":
        return UserNormContrastive(**kwargs)
    elif name == "dual_branch_contra":
        return DualBranchContrastive(**kwargs)
    elif name == "multi_task":
        return MultiTaskModel(**kwargs)
    elif name == "self_supervised":
        return SelfSupervisedModel(**kwargs)
    elif name == "domain_gen":
        return DomainGeneralizationModel(**kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")


# ==================== IMU锚定方案1: IMU引导注意力 ====================
class IMUAnchoredAttention(nn.Module):
    """
    IMU引导的注意力机制
    核心思想: 用IMU特征作为query，从EMG特征中提取手势相关部分
    IMU作为"锚点"，因为它更用户无关
    """
    def __init__(self, num_classes=4, num_users=230,
                 emg_channels=80, imu_channels=6,
                 emg_dim=256, imu_dim=128, gesture_dim=128,
                 use_dann=False, align_weight=1.0, indep_weight=0.1, dropout=0.0):
        super().__init__()
        self.use_dann = use_dann
        self.align_weight = align_weight
        self.indep_weight = indep_weight

        self.emg_encoder = EMGEncoder(in_channels=emg_channels, feat_dim=emg_dim)
        self.imu_encoder = IMUEncoder(in_channels=imu_channels, feat_dim=imu_dim)

        # IMU投影到手势空间
        self.imu_to_gesture = nn.Sequential(
            nn.Linear(imu_dim, gesture_dim),
            nn.ReLU(),
            nn.BatchNorm1d(gesture_dim))

        # EMG投影到手势空间 (要被IMU引导)
        self.emg_to_gesture = nn.Sequential(
            nn.Linear(emg_dim, gesture_dim),
            nn.ReLU(),
            nn.BatchNorm1d(gesture_dim))

        # EMG投影到用户空间 (与IMU独立)
        self.emg_to_user = nn.Sequential(
            nn.Linear(emg_dim, gesture_dim),
            nn.ReLU(),
            nn.BatchNorm1d(gesture_dim))

        # IMU引导的注意力: 用IMU作为query查询EMG
        self.imu_query = nn.Linear(gesture_dim, gesture_dim)
        self.emg_key = nn.Linear(gesture_dim, gesture_dim)
        self.attn_scale = gesture_dim ** -0.5

        # 分类器
        self.gesture_classifier = nn.Sequential(
            nn.Linear(gesture_dim, gesture_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gesture_dim, num_classes))

        if use_dann:
            self.user_classifier = nn.Sequential(
                nn.Linear(gesture_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_users))

        self.gesture_dim = gesture_dim

    def forward(self, x_emg, x_imu=None, label=None, stage='train'):
        emg_feat = self.emg_encoder(x_emg)
        imu_feat = self.imu_encoder(x_imu)

        # 投影到共同空间
        emg_g = self.emg_to_gesture(emg_feat)  # [B, gesture_dim]
        emg_u = self.emg_to_user(emg_feat)     # [B, gesture_dim]
        imu_g = self.imu_to_gesture(imu_feat)  # [B, gesture_dim]

        # IMU引导的注意力
        query = self.imu_query(imu_g).unsqueeze(1)  # [B, 1, gesture_dim]
        key = self.emg_key(emg_g).unsqueeze(1)      # [B, 1, gesture_dim]
        attn_score = torch.sum(query * key, dim=2) * self.attn_scale  # [B, 1]
        attn_weight = torch.sigmoid(attn_score)     # [B, 1]

        # 融合: 用IMU引导的权重混合EMG手势特征和IMU特征
        gesture_feat = attn_weight * emg_g + (1 - attn_weight) * imu_g

        gesture_logits = self.gesture_classifier(gesture_feat)

        if stage == 'train':
            # 对齐损失: gesture_feat应该与imu_feat相似
            align_loss = 1 - F.cosine_similarity(gesture_feat, imu_g).mean()

            # 独立性损失: user_feat应该与imu_feat独立
            indep_loss = hsic_loss(emg_u, imu_g)

            result = {
                'gesture_logits': gesture_logits,
                'gesture_feat': gesture_feat,
                'user_feat': emg_u,
                'align_loss': align_loss,
                'indep_loss': indep_loss,
            }

            if self.use_dann:
                user_logits = self.user_classifier(emg_u)
                result['user_logits'] = user_logits

            return result
        else:
            return gesture_logits

    def get_features(self, x_emg, x_imu=None):
        emg_feat = self.emg_encoder(x_emg)
        imu_feat = self.imu_encoder(x_imu)
        emg_g = self.emg_to_gesture(emg_feat)
        imu_g = self.imu_to_gesture(imu_feat)
        query = self.imu_query(imu_g).unsqueeze(1)
        key = self.emg_key(emg_g).unsqueeze(1)
        attn_score = torch.sum(query * key, dim=2) * self.attn_scale
        attn_weight = torch.sigmoid(attn_score)
        gesture_feat = attn_weight * emg_g + (1 - attn_weight) * imu_g
        return gesture_feat


# ==================== IMU锚定方案2: 特征分解 ====================
class IMUAnchoredDecomposition(nn.Module):
    """
    IMU锚定的特征分解
    核心思想: EMG特征 = 手势特征(与IMU相关) + 用户特征(与IMU独立)
    通过约束实现解耦:
    1. 手势特征与IMU特征高度相关 (对齐损失)
    2. 用户特征与IMU特征独立 (HSIC损失)
    3. 手势特征 + 用户特征 ≈ 原始EMG特征 (重构损失)
    """
    def __init__(self, num_classes=4, num_users=230,
                 emg_channels=80, imu_channels=6,
                 emg_dim=256, imu_dim=128, gesture_dim=128,
                 use_dann=False, align_weight=1.0, indep_weight=0.1, recon_weight=0.5, dropout=0.0):
        super().__init__()
        self.use_dann = use_dann
        self.align_weight = align_weight
        self.indep_weight = indep_weight
        self.recon_weight = recon_weight

        self.emg_encoder = EMGEncoder(in_channels=emg_channels, feat_dim=emg_dim)
        self.imu_encoder = IMUEncoder(in_channels=imu_channels, feat_dim=imu_dim)

        # 特征分解头
        self.gesture_head = nn.Sequential(
            nn.Linear(emg_dim, gesture_dim),
            nn.ReLU(),
            nn.BatchNorm1d(gesture_dim))

        self.user_head = nn.Sequential(
            nn.Linear(emg_dim, gesture_dim),
            nn.ReLU(),
            nn.BatchNorm1d(gesture_dim))

        # IMU投影到手势空间 (用于对齐)
        self.imu_to_gesture = nn.Sequential(
            nn.Linear(imu_dim, gesture_dim),
            nn.ReLU(),
            nn.BatchNorm1d(gesture_dim))

        # 重构头: 手势 + 用户 → 原始EMG特征
        self.reconstructor = nn.Sequential(
            nn.Linear(gesture_dim * 2, emg_dim),
            nn.ReLU(),
            nn.Linear(emg_dim, emg_dim))

        # 分类器
        self.gesture_classifier = nn.Sequential(
            nn.Linear(gesture_dim, gesture_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gesture_dim, num_classes))

        if use_dann:
            self.grl = GradientReverseLayer(lambda_=1.0)
            self.user_classifier = nn.Sequential(
                nn.Linear(gesture_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_users))

        self.gesture_dim = gesture_dim

    def forward(self, x_emg, x_imu=None, label=None, stage='train'):
        emg_feat = self.emg_encoder(x_emg)
        imu_feat = self.imu_encoder(x_imu)

        # 分解
        gesture_feat = self.gesture_head(emg_feat)
        user_feat = self.user_head(emg_feat)
        imu_gesture = self.imu_to_gesture(imu_feat)

        # 重构
        recon_feat = self.reconstructor(torch.cat([gesture_feat, user_feat], dim=1))

        gesture_logits = self.gesture_classifier(gesture_feat)

        if stage == 'train':
            # 对齐损失: gesture与IMU对齐
            align_loss = 1 - F.cosine_similarity(gesture_feat, imu_gesture).mean()

            # 独立性损失: user与IMU独立
            indep_loss = hsic_loss(user_feat, imu_gesture)

            # 重构损失
            recon_loss = F.mse_loss(recon_feat, emg_feat)

            result = {
                'gesture_logits': gesture_logits,
                'gesture_feat': gesture_feat,
                'user_feat': user_feat,
                'align_loss': align_loss,
                'indep_loss': indep_loss,
                'recon_loss': recon_loss,
            }

            if self.use_dann:
                user_logits = self.user_classifier(self.grl(user_feat))
                result['user_logits'] = user_logits

            return result
        else:
            return gesture_logits

    def get_features(self, x_emg, x_imu=None):
        emg_feat = self.emg_encoder(x_emg)
        return self.gesture_head(emg_feat)


# ==================== IMU锚定方案3: 跨模态对比学习 ====================
class IMUCrossModalContrastive(nn.Module):
    """
    跨模态对比学习
    核心思想: 同一样本的EMG和IMU特征应该靠近 (正样本)
    不同样本的EMG和IMU特征应该远离 (负样本)
    这样EMG特征会被"拉向"用户无关的IMU空间
    """
    def __init__(self, num_classes=4, num_users=230,
                 emg_channels=80, imu_channels=6,
                 emg_dim=256, imu_dim=128, feat_dim=128,
                 use_dann=False, contrastive_weight=1.0, temperature=0.1, dropout=0.0):
        super().__init__()
        self.use_dann = use_dann
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

        self.emg_encoder = EMGEncoder(in_channels=emg_channels, feat_dim=emg_dim)
        self.imu_encoder = IMUEncoder(in_channels=imu_channels, feat_dim=imu_dim)

        # 投影到共同空间
        self.emg_proj = nn.Sequential(
            nn.Linear(emg_dim, feat_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feat_dim))

        self.imu_proj = nn.Sequential(
            nn.Linear(imu_dim, feat_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feat_dim))

        # 分类器 (使用EMG特征)
        self.gesture_classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes))

        if use_dann:
            self.grl = GradientReverseLayer(lambda_=1.0)
            self.user_classifier = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_users))

        self.feat_dim = feat_dim

    def forward(self, x_emg, x_imu=None, label=None, stage='train'):
        emg_feat = self.emg_encoder(x_emg)
        imu_feat = self.imu_encoder(x_imu)

        emg_proj = self.emg_proj(emg_feat)
        imu_proj = self.imu_proj(imu_feat)

        gesture_logits = self.gesture_classifier(emg_proj)

        if stage == 'train':
            # 跨模态对比损失
            contrastive_loss = self.cross_modal_contrastive_loss(
                emg_proj, imu_proj, self.temperature)

            result = {
                'gesture_logits': gesture_logits,
                'gesture_feat': emg_proj,
                'imu_feat': imu_proj,
                'contrastive_loss': contrastive_loss,
            }

            if self.use_dann:
                user_logits = self.user_classifier(self.grl(emg_proj))
                result['user_logits'] = user_logits

            return result
        else:
            return gesture_logits

    def cross_modal_contrastive_loss(self, emg_feat, imu_feat, temperature=0.1):
        """
        跨模态对比损失: 同一样本的EMG和IMU应该靠近
        使用InfoNCE损失
        """
        emg_norm = F.normalize(emg_feat, dim=1)
        imu_norm = F.normalize(imu_feat, dim=1)

        # 相似度矩阵
        sim = torch.matmul(emg_norm, imu_norm.T) / temperature

        # 对角线是正样本
        labels = torch.arange(emg_norm.size(0), device=emg_norm.device)

        # InfoNCE
        loss = F.cross_entropy(sim, labels)
        return loss

    def get_features(self, x_emg, x_imu=None):
        emg_feat = self.emg_encoder(x_emg)
        return self.emg_proj(emg_feat)


# ==================== 新方案1: 原型对比学习 ====================
class PrototypeContrastive(nn.Module):
    """
    原型对比学习模型
    核心思想:
    1. 为每个手势学习一个"原型"特征 (用户无关的理想表征)
    2. 所有用户的手势特征向对应原型靠近
    3. 不同手势原型之间远离
    4. 不需要用户标签，纯zero-shot友好
    """
    def __init__(self, num_classes=4, num_users=230,
                 emg_channels=80, imu_channels=6,
                 emg_dim=256, imu_dim=128, feat_dim=128,
                 use_imu=True, prototype_momentum=0.99, temperature=0.1,
                 contrastive_weight=1.0, dropout=0.0):
        super().__init__()
        self.use_imu = use_imu
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.prototype_momentum = prototype_momentum

        # 编码器
        self.emg_encoder = EMGEncoder(in_channels=emg_channels, feat_dim=emg_dim)
        if use_imu:
            self.imu_encoder = IMUEncoder(in_channels=imu_channels, feat_dim=imu_dim)
            self.fusion = MultiModalFusion(emg_dim, imu_dim, emg_dim)
            input_dim = emg_dim
        else:
            self.imu_encoder = None
            self.fusion = None
            input_dim = emg_dim

        # 投影头
        self.projection = nn.Sequential(
            nn.Linear(input_dim, feat_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feat_dim))

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes))

        # 可学习的原型 (num_classes × feat_dim)
        self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.kaiming_normal_(self.prototypes)

        self.feat_dim = feat_dim
        self.num_classes = num_classes

    def forward(self, x_emg, x_imu=None, label=None, stage='train'):
        # 提取特征
        emg_feat = self.emg_encoder(x_emg)
        if self.use_imu and x_imu is not None:
            imu_feat = self.imu_encoder(x_imu)
            fused = self.fusion(emg_feat, imu_feat)
        else:
            fused = emg_feat

        # 投影并归一化
        feat = self.projection(fused)
        feat_norm = F.normalize(feat, dim=1)

        # 分类
        logits = self.classifier(feat)

        if stage == 'train':
            # 原型对比损失
            proto_loss = self.prototype_contrastive_loss(feat_norm, label)

            return {
                'gesture_logits': logits,
                'gesture_feat': feat,
                'proto_loss': proto_loss,
                'prototypes': F.normalize(self.prototypes, dim=1),
            }
        else:
            return logits

    def prototype_contrastive_loss(self, feat_norm, label):
        """
        原型对比损失: InfoNCE风格
        - 正样本: 特征与对应原型的相似度
        - 负样本: 特征与其他原型的相似度
        """
        prototypes_norm = F.normalize(self.prototypes, dim=1)  # [C, D]

        # 计算所有特征与所有原型的相似度 [B, C]
        sim = torch.matmul(feat_norm, prototypes_norm.T) / self.temperature

        # InfoNCE损失
        loss = F.cross_entropy(sim, label)
        return loss

    def get_features(self, x_emg, x_imu=None):
        emg_feat = self.emg_encoder(x_emg)
        if self.use_imu and x_imu is not None:
            imu_feat = self.imu_encoder(x_imu)
            fused = self.fusion(emg_feat, imu_feat)
        else:
            fused = emg_feat
        feat = self.projection(fused)
        return F.normalize(feat, dim=1)


# ==================== 新方案2: 记忆库对比学习 ====================
class MemoryBankContrastive(nn.Module):
    """
    记忆库对比学习模型
    核心思想:
    1. 维护一个特征记忆库 (存储所有样本的特征)
    2. 训练时从记忆库采样负样本
    3. InfoNCE损失: 正样本靠近，负样本远离
    4. 记忆库动量更新
    """
    def __init__(self, num_classes=4, num_users=230,
                 emg_channels=80, imu_channels=6,
                 emg_dim=256, imu_dim=128, feat_dim=128,
                 use_imu=True, memory_size=10000, temperature=0.1,
                 contrastive_weight=1.0, momentum=0.5, dropout=0.0):
        super().__init__()
        self.use_imu = use_imu
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.memory_size = memory_size
        self.momentum = momentum

        # 编码器
        self.emg_encoder = EMGEncoder(in_channels=emg_channels, feat_dim=emg_dim)
        if use_imu:
            self.imu_encoder = IMUEncoder(in_channels=imu_channels, feat_dim=imu_dim)
            self.fusion = MultiModalFusion(emg_dim, imu_dim, emg_dim)
            input_dim = emg_dim
        else:
            self.imu_encoder = None
            self.fusion = None
            input_dim = emg_dim

        # 投影头
        self.projection = nn.Sequential(
            nn.Linear(input_dim, feat_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feat_dim))

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes))

        self.feat_dim = feat_dim
        self.num_classes = num_classes

        # 记忆库: 为每个类别存储多个特征
        self.register_buffer('memory', torch.randn(num_classes, memory_size // num_classes, feat_dim))
        self.register_buffer('memory_ptr', torch.zeros(num_classes, dtype=torch.long))
        self.memory = F.normalize(self.memory, dim=2)

    def forward(self, x_emg, x_imu=None, label=None, stage='train'):
        # 提取特征
        emg_feat = self.emg_encoder(x_emg)
        if self.use_imu and x_imu is not None:
            imu_feat = self.imu_encoder(x_imu)
            fused = self.fusion(emg_feat, imu_feat)
        else:
            fused = emg_feat

        # 投影并归一化
        feat = self.projection(fused)
        feat_norm = F.normalize(feat, dim=1)

        # 分类
        logits = self.classifier(feat)

        if stage == 'train':
            # 记忆库对比损失
            mem_loss = self.memory_contrastive_loss(feat_norm, label)

            # 更新记忆库
            self.update_memory(feat_norm.detach(), label)

            return {
                'gesture_logits': logits,
                'gesture_feat': feat,
                'memory_loss': mem_loss,
            }
        else:
            return logits

    def memory_contrastive_loss(self, feat_norm, label):
        """
        记忆库对比损失
        正样本: 特征与同类记忆的平均特征
        负样本: 特征与异类记忆的平均特征
        """
        B = feat_norm.size(0)
        C = self.num_classes

        # 计算每个类别的记忆平均特征 [C, D]
        memory_mean = self.memory.mean(dim=1)  # [C, D]
        memory_mean = F.normalize(memory_mean, dim=1)

        # 计算特征与所有类别记忆的相似度 [B, C]
        sim = torch.matmul(feat_norm, memory_mean.T) / self.temperature

        # InfoNCE损失
        loss = F.cross_entropy(sim, label)
        return loss

    @torch.no_grad()
    def update_memory(self, feat_norm, label):
        """动量更新记忆库"""
        B = feat_norm.size(0)
        for i in range(B):
            c = label[i].item()
            ptr = self.memory_ptr[c].item() % self.memory.size(1)

            # 动量更新
            self.memory[c, ptr] = (
                self.momentum * self.memory[c, ptr] +
                (1 - self.momentum) * feat_norm[i]
            )
            self.memory[c, ptr] = F.normalize(self.memory[c, ptr], dim=0)
            self.memory_ptr[c] += 1

    def get_features(self, x_emg, x_imu=None):
        emg_feat = self.emg_encoder(x_emg)
        if self.use_imu and x_imu is not None:
            imu_feat = self.imu_encoder(x_imu)
            fused = self.fusion(emg_feat, imu_feat)
        else:
            fused = emg_feat
        feat = self.projection(fused)
        return F.normalize(feat, dim=1)


# ==================== 新方案3: 用户归一化 + 对比学习 ====================
class UserNormContrastive(nn.Module):
    """
    用户归一化 + 对比学习
    核心思想:
    1. InstanceNorm去除用户特异性风格 (通道间的幅度差异)
    2. 对比学习保留手势语义
    3. 轻量且有效，不需要用户标签
    """
    def __init__(self, num_classes=4, num_users=230,
                 emg_channels=80, imu_channels=6,
                 emg_dim=256, imu_dim=128, feat_dim=128,
                 use_imu=True, temperature=0.1,
                 contrastive_weight=1.0, dropout=0.0):
        super().__init__()
        self.use_imu = use_imu
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight

        # 编码器
        self.emg_encoder = EMGEncoder(in_channels=emg_channels, feat_dim=emg_dim)
        if use_imu:
            self.imu_encoder = IMUEncoder(in_channels=imu_channels, feat_dim=imu_dim)
            self.fusion = MultiModalFusion(emg_dim, imu_dim, emg_dim)
            input_dim = emg_dim
        else:
            self.imu_encoder = None
            self.fusion = None
            input_dim = emg_dim

        # LayerNorm: 去除用户风格 (替代InstanceNorm，避免空间维度问题)
        self.layer_norm = nn.LayerNorm(input_dim)

        # 投影头
        self.projection = nn.Sequential(
            nn.Linear(input_dim, feat_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feat_dim))

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes))

        self.feat_dim = feat_dim
        self.num_classes = num_classes

    def forward(self, x_emg, x_imu=None, label=None, stage='train'):
        # 提取特征
        emg_feat = self.emg_encoder(x_emg)
        if self.use_imu and x_imu is not None:
            imu_feat = self.imu_encoder(x_imu)
            fused = self.fusion(emg_feat, imu_feat)
        else:
            fused = emg_feat

        # LayerNorm: 去除用户风格
        fused_norm = self.layer_norm(fused)

        # 投影并归一化
        feat = self.projection(fused_norm)
        feat_norm = F.normalize(feat, dim=1)

        # 分类
        logits = self.classifier(feat)

        if stage == 'train':
            # 类内对比损失
            contrastive_loss = self.class_contrastive_loss(feat_norm, label)

            return {
                'gesture_logits': logits,
                'gesture_feat': feat,
                'contrastive_loss': contrastive_loss,
            }
        else:
            return logits

    def class_contrastive_loss(self, feat_norm, label):
        """
        类内对比损失: 同类特征靠近，异类特征远离
        使用InfoNCE风格
        """
        B = feat_norm.size(0)
        C = self.num_classes

        # 计算相似度矩阵 [B, B]
        sim_matrix = torch.matmul(feat_norm, feat_norm.T) / self.temperature

        # 创建掩码: 同类为1，异类为0
        label_expanded = label.unsqueeze(1)  # [B, 1]
        pos_mask = (label_expanded == label_expanded.T).float()  # [B, B]
        pos_mask.fill_diagonal_(0)  # 排除自身

        # 对每个样本计算InfoNCE
        # 正样本: 同类其他样本
        # 负样本: 异类样本
        exp_sim = torch.exp(sim_matrix)

        # 每行的正样本数和负样本数
        pos_count = pos_mask.sum(dim=1)  # [B]

        # 避免除零
        valid_mask = (pos_count > 0).float()

        # 正样本相似度之和
        pos_sim = (exp_sim * pos_mask).sum(dim=1)  # [B]

        # 所有样本相似度之和 (排除自身)
        neg_sim = exp_sim.sum(dim=1) - torch.exp(torch.diag(sim_matrix))  # [B]

        # InfoNCE: -log(pos_sim / neg_sim)
        # 简化版: 使用cross_entropy
        # 这里用简化版本
        loss = -torch.log((pos_sim + 1e-8) / (neg_sim + 1e-8))
        loss = (loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        return loss

    def get_features(self, x_emg, x_imu=None):
        emg_feat = self.emg_encoder(x_emg)
        if self.use_imu and x_imu is not None:
            imu_feat = self.imu_encoder(x_imu)
            fused = self.fusion(emg_feat, imu_feat)
        else:
            fused = emg_feat
        fused_norm = self.layer_norm(fused)
        feat = self.projection(fused_norm)
        return F.normalize(feat, dim=1)


# ==================== 新方案5: 多任务学习 ====================
class MultiTaskModel(nn.Module):
    """
    多任务学习模型
    核心思想:
    1. 共享编码器提取通用特征
    2. 手势分类头 (主任务)
    3. 用户识别头 (辅助任务)
    4. 通过辅助任务帮助主任务学习更好的特征表示
    """
    def __init__(self, num_classes=4, num_users=230,
                 emg_channels=80, imu_channels=6,
                 emg_dim=256, imu_dim=128, feat_dim=128,
                 use_imu=True, user_weight=0.3, dropout=0.0):
        super().__init__()
        self.use_imu = use_imu
        self.user_weight = user_weight

        # 共享编码器
        self.emg_encoder = EMGEncoder(in_channels=emg_channels, feat_dim=emg_dim)
        if use_imu:
            self.imu_encoder = IMUEncoder(in_channels=imu_channels, feat_dim=imu_dim)
            self.fusion = MultiModalFusion(emg_dim, imu_dim, emg_dim)
            input_dim = emg_dim
        else:
            self.imu_encoder = None
            self.fusion = None
            input_dim = emg_dim

        # 投影头
        self.projection = nn.Sequential(
            nn.Linear(input_dim, feat_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feat_dim))

        # 手势分类器 (主任务)
        self.gesture_classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes))

        # 用户分类器 (辅助任务)
        self.user_classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_users))

    def forward(self, x_emg, x_imu=None, label=None, user=None, stage='train'):
        # 提取特征
        emg_feat = self.emg_encoder(x_emg)
        if self.use_imu and x_imu is not None:
            imu_feat = self.imu_encoder(x_imu)
            fused = self.fusion(emg_feat, imu_feat)
        else:
            fused = emg_feat

        feat = self.projection(fused)

        # 手势分类
        gesture_logits = self.gesture_classifier(feat)

        if stage == 'train':
            # 用户分类
            user_logits = self.user_classifier(feat)
            return {
                'gesture_logits': gesture_logits,
                'user_logits': user_logits,
                'gesture_feat': feat,
            }
        else:
            return gesture_logits

    def get_features(self, x_emg, x_imu=None):
        emg_feat = self.emg_encoder(x_emg)
        if self.use_imu and x_imu is not None:
            imu_feat = self.imu_encoder(x_imu)
            fused = self.fusion(emg_feat, imu_feat)
        else:
            fused = emg_feat
        feat = self.projection(fused)
        return feat


# ==================== 新方案4: 双分支对比学习 ====================
class DualBranchContrastive(nn.Module):
    """
    双分支对比学习模型
    核心思想:
    1. 两个编码器分支处理同一输入 (或数据增广版本)
    2. 同类手势的不同分支特征靠近
    3. 不同类手势特征远离
    4. 共享权重或独立权重
    """
    def __init__(self, num_classes=4, num_users=230,
                 emg_channels=80, imu_channels=6,
                 emg_dim=256, imu_dim=128, feat_dim=128,
                 use_imu=True, shared_encoder=True, temperature=0.1,
                 contrastive_weight=1.0, dropout=0.0):
        super().__init__()
        self.use_imu = use_imu
        self.shared_encoder = shared_encoder
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight

        # 编码器分支
        if shared_encoder:
            # 共享权重
            self.emg_encoder = EMGEncoder(in_channels=emg_channels, feat_dim=emg_dim)
            self.emg_encoder2 = None
            if use_imu:
                self.imu_encoder = IMUEncoder(in_channels=imu_channels, feat_dim=imu_dim)
                self.fusion = MultiModalFusion(emg_dim, imu_dim, emg_dim)
                self.imu_encoder2 = None
                self.fusion2 = None
        else:
            # 独立权重
            self.emg_encoder = EMGEncoder(in_channels=emg_channels, feat_dim=emg_dim)
            self.emg_encoder2 = EMGEncoder(in_channels=emg_channels, feat_dim=emg_dim)
            if use_imu:
                self.imu_encoder = IMUEncoder(in_channels=imu_channels, feat_dim=imu_dim)
                self.imu_encoder2 = IMUEncoder(in_channels=imu_channels, feat_dim=imu_dim)
                self.fusion = MultiModalFusion(emg_dim, imu_dim, emg_dim)
                self.fusion2 = MultiModalFusion(emg_dim, imu_dim, emg_dim)

        input_dim = emg_dim

        # 投影头
        self.projection = nn.Sequential(
            nn.Linear(input_dim, feat_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feat_dim))

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes))

        self.feat_dim = feat_dim
        self.num_classes = num_classes

    def encode_branch1(self, x_emg, x_imu=None):
        emg_feat = self.emg_encoder(x_emg)
        if self.use_imu and x_imu is not None:
            imu_encoder = self.imu_encoder
            fusion = self.fusion
            imu_feat = imu_encoder(x_imu)
            fused = fusion(emg_feat, imu_feat)
        else:
            fused = emg_feat
        return fused

    def encode_branch2(self, x_emg, x_imu=None):
        if self.shared_encoder:
            return self.encode_branch1(x_emg, x_imu)

        emg_encoder2 = self.emg_encoder2 if self.emg_encoder2 else self.emg_encoder
        emg_feat = emg_encoder2(x_emg)

        if self.use_imu and x_imu is not None:
            if self.imu_encoder2:
                imu_encoder2 = self.imu_encoder2
                fusion2 = self.fusion2
            else:
                imu_encoder2 = self.imu_encoder
                fusion2 = self.fusion
            imu_feat = imu_encoder2(x_imu)
            fused = fusion2(emg_feat, imu_feat)
        else:
            fused = emg_feat
        return fused

    def forward(self, x_emg, x_imu=None, label=None, stage='train', x_emg_aug=None, x_imu_aug=None):
        # 分支1: 原始输入
        feat1 = self.encode_branch1(x_emg, x_imu)
        feat1 = self.projection(feat1)
        feat1_norm = F.normalize(feat1, dim=1)

        # 分支2: 增广输入 (如果提供)
        if x_emg_aug is not None:
            feat2 = self.encode_branch2(x_emg_aug, x_imu_aug)
        else:
            feat2 = self.encode_branch2(x_emg, x_imu)
        feat2 = self.projection(feat2)
        feat2_norm = F.normalize(feat2, dim=1)

        # 分类 (使用分支1)
        logits = self.classifier(feat1)

        if stage == 'train':
            # 双分支对比损失
            if x_emg_aug is not None:
                # 有增广: 正样本对是同一样本的不同增广
                contrastive_loss = self.contrastive_loss(feat1_norm, feat2_norm)
            else:
                # 无增广: 使用类内对比
                contrastive_loss = self.class_contrastive_loss(feat1_norm, label)

            return {
                'gesture_logits': logits,
                'gesture_feat': feat1,
                'contrastive_loss': contrastive_loss,
            }
        else:
            return logits

    def contrastive_loss(self, feat1_norm, feat2_norm):
        """
        双分支对比损失 (SimCLR风格)
        正样本对: 同一样本的两个增广版本
        负样本对: 不同样本
        """
        B = feat1_norm.size(0)

        # 合并两个分支的特征 [2B, D]
        combined = torch.cat([feat1_norm, feat2_norm], dim=0)  # [2B, D]

        # 相似度矩阵 [2B, 2B]
        sim_matrix = torch.matmul(combined, combined.T) / self.temperature

        # 创建标签: 正样本对的位置
        # feat1[i] 的正样本是 feat2[i]，位置在 i+B
        # feat2[i] 的正样本是 feat1[i]，位置在 i-B
        labels = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(feat1_norm.device)

        # InfoNCE损失
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def class_contrastive_loss(self, feat_norm, label):
        """
        类内对比损失 (无增广时使用)
        """
        B = feat_norm.size(0)
        C = self.num_classes

        # 计算相似度矩阵 [B, B]
        sim_matrix = torch.matmul(feat_norm, feat_norm.T) / self.temperature

        # 创建掩码: 同类为1
        label_expanded = label.unsqueeze(1)
        pos_mask = (label_expanded == label_expanded.T).float()
        pos_mask.fill_diagonal_(0)

        # 简化版InfoNCE
        exp_sim = torch.exp(sim_matrix - torch.diag(sim_matrix).unsqueeze(1))
        pos_sim = (exp_sim * pos_mask).sum(dim=1)
        neg_sim = exp_sim.sum(dim=1) - 1  # 排除自身

        valid_mask = (pos_mask.sum(dim=1) > 0).float()
        loss = -torch.log((pos_sim + 1e-8) / (neg_sim + 1e-8))
        loss = (loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        return loss

    def get_features(self, x_emg, x_imu=None):
        feat = self.encode_branch1(x_emg, x_imu)
        feat = self.projection(feat)
        return F.normalize(feat, dim=1)


# ==================== 新方案6: 自监督预训练 + 微调 ====================
class SelfSupervisedModel(nn.Module):
    """
    自监督预训练 + 微调模型
    核心思想:
    1. 阶段1: 自监督对比学习，学习用户不变的特征表示
    2. 阶段2: 冻结编码器，添加分类头微调
    """
    def __init__(self, num_classes=4, num_users=230,
                 emg_channels=80, imu_channels=6,
                 emg_dim=256, imu_dim=128, feat_dim=128,
                 use_imu=True, user_weight=0.3, temperature=0.1,
                 dropout=0.0):
        super().__init__()
        self.use_imu = use_imu
        self.user_weight = user_weight
        self.temperature = temperature

        # 共享编码器
        self.emg_encoder = EMGEncoder(in_channels=emg_channels, feat_dim=emg_dim)
        if use_imu:
            self.imu_encoder = IMUEncoder(in_channels=imu_channels, feat_dim=imu_dim)
            self.fusion = MultiModalFusion(emg_dim, imu_dim, emg_dim)
            input_dim = emg_dim
        else:
            self.imu_encoder = None
            self.fusion = None
            input_dim = emg_dim

        # 投影头 (用于对比学习)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, feat_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feat_dim))

        # 分类头 (微调阶段添加)
        self.gesture_classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes))

        self.user_classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_users))

        self.feat_dim = feat_dim
        self.num_classes = num_classes

    def encode(self, x_emg, x_imu=None):
        """编码输入"""
        emg_feat = self.emg_encoder(x_emg)
        if self.use_imu and x_imu is not None:
            imu_feat = self.imu_encoder(x_imu)
            fused = self.fusion(emg_feat, imu_feat)
        else:
            fused = emg_feat
        return fused

    def forward(self, x_emg, x_imu=None, label=None, user=None, stage='train',
                x_emg_aug=None, x_imu_aug=None, mode='classify'):
        """
        mode: 'contrastive' (自监督) | 'classify' (分类)
        """
        if mode == 'contrastive':
            # 自监督对比学习模式
            assert x_emg_aug is not None, "需要增广输入进行对比学习"

            # 编码原始输入
            feat1 = self.encode(x_emg, x_imu)
            feat1 = self.projection(feat1)
            feat1_norm = F.normalize(feat1, dim=1)

            # 编码增广输入
            feat2 = self.encode(x_emg_aug, x_imu_aug)
            feat2 = self.projection(feat2)
            feat2_norm = F.normalize(feat2, dim=1)

            # 对比损失
            contra_loss = self.contrastive_loss(feat1_norm, feat2_norm)

            return {
                'contrastive_loss': contra_loss,
                'feat1': feat1_norm,
                'feat2': feat2_norm,
            }
        else:
            # 分类模式 (微调)
            fused = self.encode(x_emg, x_imu)
            feat = self.projection(fused)

            # 手势分类
            gesture_logits = self.gesture_classifier(feat)

            if stage == 'train':
                # 用户分类
                user_logits = self.user_classifier(feat)
                return {
                    'gesture_logits': gesture_logits,
                    'user_logits': user_logits,
                    'gesture_feat': feat,
                }
            else:
                return gesture_logits

    def contrastive_loss(self, feat1_norm, feat2_norm):
        """
        SimCLR风格对比损失
        """
        B = feat1_norm.size(0)

        # 合并两个分支的特征
        combined = torch.cat([feat1_norm, feat2_norm], dim=0)  # [2B, D]

        # 计算相似度矩阵 [2B, 2B]
        sim_matrix = torch.matmul(combined, combined.T) / self.temperature

        # 创建标签: 正样本对的位置
        labels = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(feat1_norm.device)

        # InfoNCE损失
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def get_features(self, x_emg, x_imu=None):
        fused = self.encode(x_emg, x_imu)
        feat = self.projection(fused)
        return feat


# ==================== 新方案7: 域泛化 (Domain Generalization) ====================
class DomainGeneralizationModel(nn.Module):
    """
    域泛化模型
    核心思想:
    1. 每个用户作为一个域
    2. 学习域不变特征表示
    3. 使用MMD (Maximum Mean Discrepancy) 对齐不同用户的特征分布
    4. 手势分类 + 域对齐损失
    """
    def __init__(self, num_classes=4, num_users=230,
                 emg_channels=80, imu_channels=6,
                 emg_dim=256, imu_dim=128, feat_dim=128,
                 use_imu=True, user_weight=0.3, mmd_weight=1.0,
                 dropout=0.0):
        super().__init__()
        self.use_imu = use_imu
        self.user_weight = user_weight
        self.mmd_weight = mmd_weight

        # 共享编码器
        self.emg_encoder = EMGEncoder(in_channels=emg_channels, feat_dim=emg_dim)
        if use_imu:
            self.imu_encoder = IMUEncoder(in_channels=imu_channels, feat_dim=imu_dim)
            self.fusion = MultiModalFusion(emg_dim, imu_dim, emg_dim)
            input_dim = emg_dim
        else:
            self.imu_encoder = None
            self.fusion = None
            input_dim = emg_dim

        # 投影头
        self.projection = nn.Sequential(
            nn.Linear(input_dim, feat_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feat_dim))

        # 手势分类器
        self.gesture_classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes))

        # 用户分类器 (辅助任务)
        self.user_classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_users))

        self.feat_dim = feat_dim
        self.num_classes = num_classes

    def encode(self, x_emg, x_imu=None):
        """编码输入"""
        emg_feat = self.emg_encoder(x_emg)
        if self.use_imu and x_imu is not None:
            imu_feat = self.imu_encoder(x_imu)
            fused = self.fusion(emg_feat, imu_feat)
        else:
            fused = emg_feat
        return fused

    def forward(self, x_emg, x_imu=None, label=None, user=None, stage='train'):
        fused = self.encode(x_emg, x_imu)
        feat = self.projection(fused)

        # 手势分类
        gesture_logits = self.gesture_classifier(feat)

        if stage == 'train':
            # 用户分类
            user_logits = self.user_classifier(feat)

            # MMD域对齐损失
            mmd_loss = self.mmd_loss(feat, user)

            return {
                'gesture_logits': gesture_logits,
                'user_logits': user_logits,
                'gesture_feat': feat,
                'mmd_loss': mmd_loss,
            }
        else:
            return gesture_logits

    def mmd_loss(self, feat, user):
        """
        多用户MMD损失
        对齐不同用户的特征分布
        """
        unique_users = user.unique()
        if len(unique_users) < 2:
            return torch.tensor(0.0, device=feat.device)

        # 随机采样用户对计算MMD
        n_pairs = min(4, len(unique_users) * (len(unique_users) - 1) // 2)
        total_mmd = 0.0
        count = 0

        # 使用RBF核
        for _ in range(n_pairs):
            idx = torch.randperm(len(unique_users))[:2]
            u1, u2 = unique_users[idx[0]], unique_users[idx[1]]

            mask1 = (user == u1)
            mask2 = (user == u2)

            if mask1.sum() > 0 and mask2.sum() > 0:
                feat1 = feat[mask1]
                feat2 = feat[mask2]

                # 计算MMD
                mmd = self._compute_mmd(feat1, feat2)
                total_mmd = total_mmd + mmd
                count += 1

        return total_mmd / (count + 1e-8)

    def _compute_mmd(self, x, y):
        """
        计算两个特征集的MMD (使用RBF核)
        """
        # 计算带宽
        xx = torch.matmul(x, x.T)
        yy = torch.matmul(y, y.T)
        xy = torch.matmul(x, y.T)

        # 使用中位数启发式计算带宽
        xx_diag = torch.diag(xx).unsqueeze(1)
        yy_diag = torch.diag(yy).unsqueeze(1)

        dist_xx = xx_diag + xx_diag.T - 2 * xx
        dist_yy = yy_diag + yy_diag.T - 2 * yy
        dist_xy = xx_diag + yy_diag.T - 2 * xy

        # 组合距离
        all_dist = torch.cat([dist_xx.flatten(), dist_yy.flatten(), dist_xy.flatten()])
        sigma = torch.median(all_dist[all_dist > 0]) + 1e-8

        # RBF核
        K_xx = torch.exp(-dist_xx / (2 * sigma))
        K_yy = torch.exp(-dist_yy / (2 * sigma))
        K_xy = torch.exp(-dist_xy / (2 * sigma))

        # MMD^2
        m = x.size(0)
        n = y.size(0)

        # 排除对角线
        K_xx_sum = K_xx.sum() - m
        K_yy_sum = K_yy.sum() - n

        mmd_sq = K_xx_sum / (m * (m - 1) + 1e-8) + \
                 K_yy_sum / (n * (n - 1) + 1e-8) - \
                 2 * K_xy.sum() / (m * n + 1e-8)

        return torch.clamp(mmd_sq, min=0.0)

    def get_features(self, x_emg, x_imu=None):
        fused = self.encode(x_emg, x_imu)
        feat = self.projection(fused)
        return feat