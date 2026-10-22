# -*- coding: utf-8 -*-
"""
models_v3.py - 跨用户手势识别消融实验模型
支持 5 个可选组件的消融:
  1. Feature Whitening  - UserStyleEncoder + 逐维度白化
  2. ArcFace            - 角度间隔损失分类器
  3. Slim               - 模型瘦身 (小维度 + dropout + 单层分类头)
  4. ContraFixed        - 修复的对比学习 (hard negative + 温度退火)
  5. V02                - v02 架构组件 (EMG 统计特征 + 协同注意力)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 基础编码器 (来自 v2) ====================
class EMGEncoder(nn.Module):
    def __init__(self, in_channels=80, feat_dim=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 128, 7, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2))
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, feat_dim, 3, padding=1),
            nn.BatchNorm1d(feat_dim), nn.ReLU(), nn.AdaptiveAvgPool1d(1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.flatten(1)


class IMUEncoder(nn.Module):
    def __init__(self, in_channels=6, feat_dim=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, 7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2))
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, feat_dim, 3, padding=1),
            nn.BatchNorm1d(feat_dim), nn.ReLU(), nn.AdaptiveAvgPool1d(1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.flatten(1)


class MultiModalFusion(nn.Module):
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


# ==================== 组件1: Feature Whitening ====================
class UserStyleEncoder(nn.Module):
    """预测用户风格参数 (mu, sigma)，逐维度"""
    def __init__(self, feat_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim * 2))

    def forward(self, x):
        params = self.net(x)
        mu, log_sigma = params.chunk(2, dim=1)
        sigma = torch.exp(log_sigma * 0.5) + 1e-6  # log_sigma -> sigma
        return mu, sigma


class FeatureWhitening(nn.Module):
    """逐维度特征白化: 去除用户风格，保留手势信息"""
    def __init__(self, feat_dim):
        super().__init__()
        self.style_encoder = UserStyleEncoder(feat_dim)
        self.register_buffer('global_mean', torch.zeros(feat_dim))
        self.register_buffer('global_std', torch.ones(feat_dim))
        self.register_buffer('global_user_mu', torch.zeros(feat_dim))
        self.register_buffer('global_user_sigma', torch.ones(feat_dim))

    def forward(self, feat, training=True):
        if training:
            mu, sigma = self.style_encoder(feat)
            z = (feat - mu) / sigma
        else:
            # 推理: batch 统计 + 全局先验混合
            B = feat.size(0)
            mu_batch = feat.mean(dim=0, keepdim=True).expand(B, -1)
            sigma_batch = feat.std(dim=0, keepdim=True).expand(B, -1) + 1e-6
            mu_global = self.global_user_mu.unsqueeze(0).expand(B, -1)
            sigma_global = self.global_user_sigma.unsqueeze(0).expand(B, -1)
            mu = 0.7 * mu_batch + 0.3 * mu_global
            sigma = 0.7 * sigma_batch + 0.3 * sigma_global
            z = (feat - mu) / sigma
        z = z * self.global_std + self.global_mean
        return z

    def update_global_stats(self, feat_list):
        """训练时更新全局统计量"""
        if len(feat_list) == 0:
            return
        z_all = torch.cat(feat_list, dim=0)
        device = self.style_encoder.net[0].weight.device
        z_all = z_all.to(device)
        self.global_mean = z_all.mean(dim=0).detach()
        self.global_std = (z_all.std(dim=0) + 1e-6).detach()
        with torch.no_grad():
            mu_all, sigma_all = self.style_encoder(z_all)
            self.global_user_mu = mu_all.mean(dim=0).detach()
            self.global_user_sigma = sigma_all.mean(dim=0).detach()


# ==================== 组件2: ArcFace 分类器 ====================
class ArcFaceClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes, margin=0.3, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.num_classes = num_classes

    def forward(self, feat, label=None):
        feat_norm = F.normalize(feat, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cos_theta = torch.matmul(feat_norm, weight_norm.T)
        cos_theta = cos_theta.clamp(-1, 1)

        if label is None or not self.training:
            return cos_theta * self.scale

        sin_theta = torch.sqrt(1.0 - torch.clamp(cos_theta ** 2, 0, 1) + 1e-8)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m, cos_theta - self.mm)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        logits = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        return logits * self.scale


# ==================== 组件4: 修复的对比学习 ====================
class CrossUserContrastiveLoss(nn.Module):
    """跨用户对比学习: 同手势不同用户拉近, hard negative mining
    支持 InfoNCE (默认) 和 SupCon 模式
    """
    def __init__(self, temperature=0.1, hard_neg_ratio=0.5, use_supcon=False):
        super().__init__()
        self.temperature = temperature
        self.hard_neg_ratio = hard_neg_ratio
        self.use_supcon = use_supcon

    def set_temperature(self, temp):
        """Temperature annealing 接口"""
        self.temperature = temp

    def forward(self, feat, labels, users):
        B = feat.size(0)
        if B < 2:
            return torch.tensor(0.0, device=feat.device)
        feat_norm = F.normalize(feat, dim=1)
        sim = torch.matmul(feat_norm, feat_norm.T) / self.temperature

        same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
        diff_user = users.unsqueeze(0) != users.unsqueeze(1)

        if self.use_supcon:
            # SupCon: 所有同手势样本都是正样本 (跨用户权重更高)
            pos_mask = same_label.clone()
            pos_mask.fill_diagonal_(False)
            # 跨用户正样本权重 1.0, 同用户正样本权重 0.3 (弱化同用户)
            same_user = ~diff_user
            same_user.fill_diagonal_(False)
            pos_weight = torch.ones_like(pos_mask, dtype=torch.float)
            pos_weight[same_user] = 0.3
        else:
            # InfoNCE: 正样本 = 同手势 & 不同用户
            pos_mask = same_label & diff_user
            pos_mask.fill_diagonal_(False)
            pos_weight = None

        # 负样本: 不同手势
        neg_mask = ~same_label
        neg_mask.fill_diagonal_(False)

        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=feat.device)

        # Hard negative mining
        if self.hard_neg_ratio < 1.0:
            with torch.no_grad():
                neg_sim = sim.masked_fill(~neg_mask, float('-inf'))
                k = max(1, int(neg_mask.sum(dim=1).float().mean().item() * self.hard_neg_ratio))
                topk_neg_vals, topk_neg_idx = neg_sim.topk(k, dim=1)
                hard_neg_mask = torch.zeros_like(neg_mask)
                hard_neg_mask.scatter_(1, topk_neg_idx, True)
                neg_mask = neg_mask & hard_neg_mask

        exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True)[0])
        neg_sum = (exp_sim * neg_mask).sum(dim=1)

        if self.use_supcon and pos_weight is not None:
            # SupCon: 对每个正样本单独计算 loss, 加权
            pos_exp = exp_sim * pos_mask.float() * pos_weight
            log_prob = torch.log(pos_exp / (pos_exp.sum(dim=1, keepdim=True) + neg_sum.unsqueeze(1) + 1e-8) + 1e-8)
            mean_log_prob = (log_prob * pos_mask.float()).sum(dim=1) / (pos_count + 1e-8)
            loss = -mean_log_prob
        else:
            pos_sum = (exp_sim * pos_mask).sum(dim=1)
            loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8) + 1e-8)

        loss = (loss * valid.float()).sum() / (valid.float().sum() + 1e-8)
        return loss


# ==================== 组件5: v02 架构组件 ====================
class EMGStatsExtractor(nn.Module):
    """提取 EMG 时域 + 频域统计特征 (参考 v02)"""
    def __init__(self, num_channels=80):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        # 时域特征
        mav = x.abs().mean(dim=2)                          # (B, C)
        rms = torch.sqrt((x ** 2).mean(dim=2) + 1e-8)      # (B, C)
        wl = x.diff(dim=2).abs().sum(dim=2) / T             # (B, C)
        zc = ((x[:, :, :-1] * x[:, :, 1:]) < 0).float().sum(dim=2) / T
        ssc = ((x.diff(dim=2)[:, :, :-1] * x.diff(dim=2)[:, :, 1:]) < 0).float().sum(dim=2) / T
        time_feat = torch.cat([mav, rms, wl, zc, ssc], dim=1)  # (B, 5C)

        # 频域特征 (FFT)
        x_fft = torch.fft.rfft(x, dim=2)
        power = (x_fft.abs() ** 2)                           # (B, C, F)
        power = power / (power.sum(dim=2, keepdim=True) + 1e-8)
        freqs = torch.fft.rfftfreq(T, d=1.0 / 500.0, device=x.device)
        mnf = (power * freqs).sum(dim=2) / (power.sum(dim=2) + 1e-8)
        cum_power = torch.cumsum(power, dim=2)
        total_power = power.sum(dim=2, keepdim=True)
        mdf_idx = (cum_power >= total_power * 0.5).float().argmax(dim=2)
        mdf = freqs[mdf_idx]
        # 频谱熵
        entropy = -(power * torch.log(power + 1e-8)).sum(dim=2)
        # 带宽
        bandwidth = torch.sqrt((power * (freqs.unsqueeze(0).unsqueeze(0) - mnf.unsqueeze(2)) ** 2).sum(dim=2) / (power.sum(dim=2) + 1e-8) + 1e-8)
        freq_feat = torch.cat([mnf, mdf, entropy, bandwidth], dim=1)  # (B, 4C)

        return torch.cat([time_feat, freq_feat], dim=1)  # (B, 9C)


class SynergyAttention(nn.Module):
    """肌肉协同注意力: 学习通道间权重 (参考 v02)"""
    def __init__(self, num_channels=80, feat_dim=64):
        super().__init__()
        self.query = nn.Linear(num_channels, feat_dim)
        self.key = nn.Linear(num_channels, feat_dim)
        self.value = nn.Linear(num_channels, num_channels)
        self.scale = feat_dim ** -0.5

    def forward(self, x):
        # x: (B, C, T) -> 对通道做注意力
        x_t = x.mean(dim=2)  # (B, C)
        q = self.query(x_t)  # (B, D)
        k = self.key(x_t)    # (B, D)
        v = self.value(x_t)  # (B, C)
        attn = torch.matmul(q, k.T) * self.scale  # (B, B)
        attn = F.softmax(attn, dim=1)
        out = torch.matmul(attn, v)  # (B, C)
        return out


# ==================== 组件6: 自监督预训练 (统计特征预测 + 对比学习) ====================
class StatsPredictionHead(nn.Module):
    """从编码特征预测 EMG 统计特征 (RMS/MAV/WL/ZC/SSC + 频域)
    目标: 让编码器学习高层手势语义, 而非逐像素重建
    """
    def __init__(self, feat_dim=256, num_channels=80, num_stats=9):
        super().__init__()
        self.num_channels = num_channels
        self.num_stats = num_stats  # 5 时域 + 4 频域
        out_dim = num_channels * num_stats  # 720
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(feat_dim * 2),
            nn.Linear(feat_dim * 2, out_dim))

    def forward(self, feat):
        return self.net(feat)


def compute_emg_stats(x):
    """从原始 EMG 信号计算统计特征 (无需梯度)
    x: (B, C, T) -> (B, C*num_stats)
    """
    B, C, T = x.shape
    # 时域
    mav = x.abs().mean(dim=2)
    rms = torch.sqrt((x ** 2).mean(dim=2) + 1e-8)
    wl = x.diff(dim=2).abs().sum(dim=2) / T
    zc = ((x[:, :, :-1] * x[:, :, 1:]) < 0).float().sum(dim=2) / T
    ssc = ((x.diff(dim=2)[:, :, :-1] * x.diff(dim=2)[:, :, 1:]) < 0).float().sum(dim=2) / T
    # 频域
    x_fft = torch.fft.rfft(x, dim=2)
    power = (x_fft.abs() ** 2)
    power = power / (power.sum(dim=2, keepdim=True) + 1e-8)
    freqs = torch.fft.rfftfreq(T, d=1.0 / 500.0, device=x.device)
    mnf = (power * freqs).sum(dim=2) / (power.sum(dim=2) + 1e-8)
    cum_power = torch.cumsum(power, dim=2)
    total_power = power.sum(dim=2, keepdim=True)
    mdf_idx = (cum_power >= total_power * 0.5).float().argmax(dim=2)
    mdf = freqs[mdf_idx]
    entropy = -(power * torch.log(power + 1e-8)).sum(dim=2)
    bandwidth = torch.sqrt((power * (freqs.unsqueeze(0).unsqueeze(0) - mnf.unsqueeze(2)) ** 2).sum(dim=2) / (power.sum(dim=2) + 1e-8) + 1e-8)
    return torch.cat([mav, rms, wl, zc, ssc, mnf, mdf, entropy, bandwidth], dim=1)


def random_time_mask(x, mask_ratio=0.75):
    """随机时间步掩码: x (B, C, T) -> masked_x, mask
    默认 mask_ratio=0.75 (比 0.4 更难, 迫使模型学习鲁棒特征)
    """
    B, C, T = x.shape
    mask = (torch.rand(B, T, device=x.device) < mask_ratio).unsqueeze(1).expand(B, C, T)
    x_masked = x.clone()
    x_masked[mask] = 0.0
    return x_masked, mask


# ==================== 组件: 频谱图掩码重建 (Yang 2025 / SPECTRE style) ====================
class SpectrogramReconstructionHead(nn.Module):
    """从编码特征重建 EMG 时频谱图 (STFT magnitude)
    目标: 让编码器学习肌肉频率特征 (比原始波形更用户不变)
    输出: (B, C * n_freq * n_frames)
    """
    def __init__(self, feat_dim=256, num_channels=80, n_fft=32, n_frames=6):
        super().__init__()
        self.num_channels = num_channels
        self.n_freq = n_fft // 2 + 1  # 17
        self.n_frames = n_frames      # 6
        out_dim = num_channels * self.n_freq * self.n_frames  # 80*17*6 = 8160
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(feat_dim * 2),
            nn.Linear(feat_dim * 2, out_dim))

    def forward(self, feat):
        return self.net(feat)  # (B, C*n_freq*n_frames)


def compute_stft_spectrogram(x, n_fft=32, hop_length=8, win_length=32, sr=500.0):
    """计算 EMG 的 STFT 幅度谱 (无需梯度)
    x: (B, C, T) -> (B, C, n_freq, n_frames)
    使用 log magnitude 压缩动态范围
    """
    B, C, T = x.shape
    # 合并 batch 和 channel 维度
    x_flat = x.reshape(B * C, T)
    # STFT: (B*C, n_freq, n_frames)
    spec = torch.stft(x_flat, n_fft=n_fft, hop_length=hop_length,
                      win_length=win_length, return_complex=True,
                      window=torch.hann_window(win_length, device=x.device))
    mag = spec.abs()  # (B*C, n_freq, n_frames)
    # log magnitude 压缩
    log_mag = torch.log1p(mag)
    # 还原维度
    n_freq = n_fft // 2 + 1
    n_frames = mag.shape[-1]
    return log_mag.reshape(B, C, n_freq, n_frames)


def random_patch_mask_spectrogram(spec, mask_ratio=0.5, patch_size=(4, 2)):
    """对频谱图做随机 patch 掩码 (Yang 2025 style)
    spec: (B, C, F, T) -> masked_spec, mask
    mask_ratio: 被掩码的 patch 比例
    patch_size: (freq_patch, time_patch)
    """
    B, C, F, T = spec.shape
    pf, pt = patch_size
    n_patches_f = F // pf
    n_patches_t = T // pt
    # 随机决定哪些 patch 被掩码
    patch_mask = (torch.rand(B, C, n_patches_f, n_patches_t, device=spec.device) < mask_ratio)
    # 扩展到像素级
    mask = patch_mask.repeat_interleave(pf, dim=2).repeat_interleave(pt, dim=3)
    # 处理边界
    mask = mask[:, :, :F, :T]
    spec_masked = spec.clone()
    spec_masked[mask] = 0.0
    return spec_masked, mask


# ==================== 组件7: 对抗域对齐 (Gradient Reversal Layer) ====================
class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层: 前向不变, 反向乘 -lambda"""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalLayer.apply(x, lambda_)


class DomainDiscriminator(nn.Module):
    """域判别器: 区分样本来自哪个用户 (域)
    配合 GRL 迫使编码器产生用户不变特征
    """
    def __init__(self, feat_dim, num_domains, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains))

    def forward(self, feat):
        return self.net(feat)


# ==================== 组件B: KL 用户均匀约束 ====================
def kl_user_uniform_loss(user_logits, num_users):
    """约束用户分类后验接近均匀分布 → 去除用户信息
    比 GRL 温和, 不破坏手势特征
    KL(p(u|f) || Uniform) = Σ p * log(p / (1/N)) = Σ p*log(p) + log(N)
    """
    log_probs = F.log_softmax(user_logits, dim=1)
    probs = torch.exp(log_probs)
    # KL(probs || uniform) = Σ p * log(p * N) = Σ p*log(p) + log(N)
    kl = (probs * log_probs).sum(dim=1).mean() + np.log(num_users)
    return kl


# ==================== 组件C: 数据增强 ====================
def cross_user_mixup(x_emg, x_imu, y, u, alpha=0.4):
    """跨用户 Mixup: 同手势不同用户的样本混合
    模糊用户边界, 让模型无法区分来源用户
    返回: mixed_x_emg, mixed_x_imu, y (不变), mixed_u (用于加权)
    """
    B = x_emg.size(0)
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    # 找同手势不同用户的配对
    perm = torch.zeros(B, dtype=torch.long, device=x_emg.device)
    for i in range(B):
        same_gesture = (y == y[i]) & (u != u[i])
        if same_gesture.sum() > 0:
            candidates = same_gesture.nonzero(as_tuple=True)[0]
            perm[i] = candidates[torch.randint(len(candidates), (1,))]
        else:
            perm[i] = i  # 找不到就和自己混 (无效果)

    mixed_emg = lam * x_emg + (1 - lam) * x_emg[perm]
    mixed_imu = None
    if x_imu is not None:
        mixed_imu = lam * x_imu + (1 - lam) * x_imu[perm]
    return mixed_emg, mixed_imu, y, perm, lam


def signal_augmentation(x_emg, aug_prob=0.5):
    """信号级数据增强: 幅值缩放 + 通道 dropout + 时间扭曲
    增加用户不变性
    """
    if torch.rand(1).item() > aug_prob:
        return x_emg

    B, C, T = x_emg.shape
    augmented = x_emg.clone()

    # 1. 幅值缩放 (0.8-1.2)
    scale = torch.empty(B, 1, 1, device=x_emg.device).uniform_(0.8, 1.2)
    augmented = augmented * scale

    # 2. 通道 dropout (随机置零 10% 通道)
    channel_mask = (torch.rand(B, C, 1, device=x_emg.device) > 0.1).float()
    augmented = augmented * channel_mask

    return augmented


# ==================== 统一消融模型 ====================
class UnifiedModel(nn.Module):
    """
    支持 5+ 个可选组件的统一模型
    baseline = multi_task + per_user 归一化 (57.39%)
    """
    def __init__(self, num_classes=4, num_users=175,
                 emg_channels=80, imu_channels=6,
                 use_imu=True, user_weight=0.3,
                 use_whitening=False, use_arcface=False,
                 use_slim=False, use_contra=False, use_v02=False,
                 use_domain_align=False, use_kl_uniform=False,
                 use_supcon=False, feat_dim_override=None,
                 arcface_margin=0.3, arcface_scale=30.0,
                 dropout=0.0):
        super().__init__()
        self.use_imu = use_imu
        self.user_weight = user_weight
        self.use_whitening = use_whitening
        self.use_arcface = use_arcface
        self.use_slim = use_slim
        self.use_contra = use_contra
        self.use_domain_align = use_domain_align
        self.use_kl_uniform = use_kl_uniform
        self.use_supcon = use_supcon
        self.use_v02 = use_v02
        self.num_users = num_users

        # 模型瘦身: 减小维度
        if use_slim:
            emg_dim, imu_dim, feat_dim = 128, 64, 64
            clf_dropout = 0.4
        else:
            emg_dim, imu_dim, feat_dim = 256, 128, 128
            clf_dropout = dropout

        # 信息瓶颈: 覆盖特征维度
        if feat_dim_override is not None:
            feat_dim = feat_dim_override

        # v02 组件
        self.emg_stats = EMGStatsExtractor(emg_channels) if use_v02 else None
        self.synergy_attn = SynergyAttention(emg_channels) if use_v02 else None
        v02_extra = 0
        if use_v02:
            v02_extra = 9 * emg_channels  # 9C 统计特征

        # 编码器
        self.emg_encoder = EMGEncoder(emg_channels, emg_dim)
        if use_imu:
            self.imu_encoder = IMUEncoder(imu_channels, imu_dim)
            self.fusion = MultiModalFusion(emg_dim, imu_dim, emg_dim)
            input_dim = emg_dim + v02_extra
        else:
            self.imu_encoder = None
            self.fusion = None
            input_dim = emg_dim + v02_extra

        # 投影头
        self.projection = nn.Sequential(
            nn.Linear(input_dim, feat_dim),
            nn.ReLU(),
            nn.BatchNorm1d(feat_dim))

        # Feature Whitening
        self.whitening = FeatureWhitening(feat_dim) if use_whitening else None

        # 分类器
        if use_arcface:
            self.gesture_classifier = ArcFaceClassifier(feat_dim, num_classes, arcface_margin, arcface_scale)
        else:
            if use_slim:
                self.gesture_classifier = nn.Linear(feat_dim, num_classes)
            else:
                self.gesture_classifier = nn.Sequential(
                    nn.Linear(feat_dim, feat_dim), nn.ReLU(),
                    nn.Dropout(clf_dropout), nn.Linear(feat_dim, num_classes))

        # 用户分类器 (辅助任务)
        self.user_classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Dropout(clf_dropout), nn.Linear(feat_dim, num_users))

        # 对比学习
        if use_contra:
            self.contra_criterion = CrossUserContrastiveLoss(
                temperature=0.1, hard_neg_ratio=0.5, use_supcon=use_supcon)

        # 对抗域对齐: 每个用户作为一个域, 编码器需产生用户不变特征
        if use_domain_align:
            self.domain_discriminator = DomainDiscriminator(feat_dim, num_users)
            self.domain_lambda = 1.0  # GRL 系数, 可退火

        # 自监督预训练解码器
        self.recon_decoder = None  # 延迟初始化，避免影响参数量统计

    def forward(self, x_emg, x_imu=None, label=None, user=None, stage='train'):
        # 编码
        emg_feat = self.emg_encoder(x_emg)
        if self.use_imu and x_imu is not None:
            imu_feat = self.imu_encoder(x_imu)
            fused = self.fusion(emg_feat, imu_feat)
        else:
            fused = emg_feat

        # v02: 统计特征 + 协同注意力
        if self.use_v02:
            stats = self.emg_stats(x_emg)
            fused = torch.cat([fused, stats], dim=1)

        feat = self.projection(fused)

        # Feature Whitening
        if self.use_whitening:
            feat = self.whitening(feat, training=self.training)

        # 手势分类
        if self.use_arcface:
            gesture_logits = self.gesture_classifier(feat, label if stage == 'train' else None)
        else:
            gesture_logits = self.gesture_classifier(feat)

        if stage == 'train':
            out = {'gesture_logits': gesture_logits, 'gesture_feat': feat}
            # 用户分类 (辅助任务)
            out['user_logits'] = self.user_classifier(feat)
            # 对比学习
            if self.use_contra and label is not None and user is not None:
                out['contra_loss'] = self.contra_criterion(feat, label, user)
            # 对抗域对齐: 编码器对抗域判别器 (迫使特征用户不变)
            if self.use_domain_align and user is not None:
                reversed_feat = grad_reverse(feat, self.domain_lambda)
                out['domain_logits'] = self.domain_discriminator(reversed_feat)
            # KL 用户均匀约束: 推动用户后验接近均匀 (温和去除用户信息)
            if self.use_kl_uniform:
                out['kl_uniform'] = kl_user_uniform_loss(out['user_logits'], self.num_users)
            return out
        return gesture_logits

    def update_whitening_stats(self, feat_list):
        if self.use_whitening:
            self.whitening.update_global_stats(feat_list)

    def init_pretrain_heads(self, emg_channels=80, emg_len=150, pretrain_target='stats',
                            n_fft=32, hop_length=8):
        """初始化预训练头
        pretrain_target: 'stats' (统计特征) 或 'spectrogram' (STFT 频谱图)
        """
        feat_dim = self.projection[0].out_features
        self.pretrain_target = pretrain_target
        if pretrain_target == 'stats':
            self.stats_predictor = StatsPredictionHead(feat_dim, emg_channels, num_stats=9)
        elif pretrain_target == 'spectrogram':
            # 动态计算 STFT 输出帧数 (torch.stft 需要 1D 输入)
            with torch.no_grad():
                dummy = torch.zeros(emg_len)
                spec = torch.stft(dummy, n_fft=n_fft, hop_length=hop_length,
                                  win_length=n_fft, return_complex=True,
                                  window=torch.hann_window(n_fft))
                n_frames = spec.shape[-1]
            self.spec_reconstructor = SpectrogramReconstructionHead(
                feat_dim, num_channels=emg_channels, n_fft=n_fft, n_frames=n_frames)
        # 对比损失 (跨用户: 同手势不同用户=正, 不同手势=负)
        self.pretrain_contra = CrossUserContrastiveLoss(
            temperature=0.1, hard_neg_ratio=0.5, use_supcon=getattr(self, 'use_supcon', False))

    def forward_pretrain(self, x_emg, x_imu=None, labels=None, users=None,
                         mask_ratio=0.75, contra_weight=0.5):
        """自监督预训练前向
        根据 pretrain_target 选择重建目标:
        - 'stats': 从 masked EMG 预测 9 组统计特征
        - 'spectrogram': 从 masked EMG 重建 STFT 幅度谱 (Yang 2025 style)
        辅任务: 跨用户对比学习 (同手势不同用户拉近)
        """
        x_emg_masked, mask = random_time_mask(x_emg, mask_ratio)

        # 编码 (用 masked 输入)
        emg_feat = self.emg_encoder(x_emg_masked)
        if self.use_imu and x_imu is not None:
            imu_feat = self.imu_encoder(x_imu)
            fused = self.fusion(emg_feat, imu_feat)
        else:
            fused = emg_feat
        if self.use_v02:
            stats = self.emg_stats(x_emg_masked)
            fused = torch.cat([fused, stats], dim=1)
        feat = self.projection(fused)

        # 主损失: 根据目标类型
        if self.pretrain_target == 'stats':
            # 统计特征预测 (从 masked 编码预测原始信号的统计量)
            stats_target = compute_emg_stats(x_emg)  # (B, 720), 从未 mask 的原始信号计算
            stats_pred = self.stats_predictor(feat)
            main_loss = F.smooth_l1_loss(stats_pred, stats_target)
        else:  # spectrogram
            # STFT 频谱图重建 (从 masked 编码预测原始信号的时频谱)
            spec_target = compute_stft_spectrogram(x_emg)  # (B, C, nF, nT), 从未 mask 计算
            B, C, nF, nT = spec_target.shape
            spec_pred = self.spec_reconstructor(feat)  # (B, C*nF*nT)
            spec_pred = spec_pred.reshape(B, C, nF, nT)
            main_loss = torch.nn.functional.mse_loss(spec_pred, spec_target)

        # 辅损失: 跨用户对比学习 (如果提供了标签和用户)
        contra_loss = torch.tensor(0.0, device=feat.device)
        if labels is not None and users is not None and contra_weight > 0:
            contra_loss = self.pretrain_contra(feat, labels, users)

        total_loss = main_loss + contra_weight * contra_loss
        return total_loss, main_loss, contra_loss


def build_model(name, **kwargs):
    if name == "unified":
        return UnifiedModel(**kwargs)
    raise ValueError(f"Unknown model: {name}")
