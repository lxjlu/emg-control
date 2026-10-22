# -*- coding: utf-8 -*-
"""
models_v4.py - 跨用户零样本手势识别 v4

核心设计哲学:
  主动去除用户信息, 而非避免添加。
  - 无用户分类头 (已验证: +0.87%)
  - GRL 对抗域对齐: 编码器愚弄用户判别器 → 特征用户不变
  - MMD 分布对齐: 跨用户特征分布矩匹配 (比 GRL 温和)
  - 特征白化: 消除用户特定协方差
  - 跨用户对比学习: 同手势不同用户拉近
  - 频谱预训练初始化

与 v3 的关键区别:
  v3 的 GRL 失败是因为与用户分类头梯度冲突 (一个保留, 一个去除)
  v4 去掉用户头, GRL 可以纯粹地去除用户信息
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 基础编码器 (同 v3) ====================
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


# ==================== 特征白化 ====================
class FeatureWhitening(nn.Module):
    """逐维度特征白化: 去除用户风格, 保留手势信息"""
    def __init__(self, feat_dim, momentum=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(feat_dim))
        self.register_buffer('running_var', torch.ones(feat_dim))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x, training=True):
        if training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)
                self.num_batches_tracked.add_(1)
        else:
            mean = self.running_mean
            var = self.running_var
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        return x_norm


# ==================== GRL 梯度反转 ====================
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalLayer.apply(x, lambda_)


# ==================== 域判别器 (用于 GRL 对抗去除用户信息) ====================
class DomainDiscriminator(nn.Module):
    """域判别器: 区分样本来自哪个用户
    配合 GRL, 编码器被训练为产生用户不可区分的特征
    """
    def __init__(self, feat_dim, num_domains, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_domains))

    def forward(self, feat):
        return self.net(feat)


# ==================== MMD 分布对齐 ====================
def compute_mmd(x, y, kernel_mul=2.0, kernel_num=5):
    """计算两批特征之间的 MMD (Maximum Mean Discrepancy)
    使用多核高斯核
    x: (n, d), y: (m, d)
    """
    n, m = x.size(0), y.size(0)
    xx = torch.mm(x, x.t())  # (n, n)
    yy = torch.mm(y, y.t())  # (m, m)
    xy = torch.mm(x, y.t())  # (n, m)

    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i·x_j
    rx = xx.diag().unsqueeze(1)  # (n, 1)
    ry = yy.diag().unsqueeze(0)  # (1, m)

    dxx = rx + rx.t() - 2. * xx  # (n, n)
    dyy = ry.t() + ry - 2. * yy  # (m, m)
    dxy = rx + ry - 2. * xy      # (n, m)

    XX = torch.zeros_like(xx)
    YY = torch.zeros_like(yy)
    XY = torch.zeros_like(xy)

    for i in range(kernel_num):
        bandwidth = 1.0 / (kernel_mul ** i)
        XX += torch.exp(-0.5 * dxx * bandwidth)
        YY += torch.exp(-0.5 * dyy * bandwidth)
        XY += torch.exp(-0.5 * dxy * bandwidth)

    return torch.mean(XX) + torch.mean(YY) - 2. * torch.mean(XY)


def mmd_loss_across_users(features, users, num_samples_per_user=6, max_pairs=12):
    """计算批次内跨用户特征分布的 MMD
    优化: 只采样 max_pairs 对用户, 每用户最多 num_samples_per_user 样本
    避免 O(U^2) 的全配对计算
    """
    unique_users = torch.unique(users)
    if len(unique_users) < 2:
        return torch.tensor(0.0, device=features.device)

    user_feats = {}
    for u in unique_users:
        mask = (users == u)
        feats = features[mask]
        if len(feats) >= 4:
            user_feats[u.item()] = feats

    if len(user_feats) < 2:
        return torch.tensor(0.0, device=features.device)

    user_list = list(user_feats.values())
    # 随机采样用户对, 限制计算量
    if len(user_list) * (len(user_list) - 1) // 2 > max_pairs:
        indices = torch.randperm(len(user_list))
        pairs = []
        for i in range(len(user_list)):
            for j in range(i + 1, len(user_list)):
                pairs.append((i, j))
                if len(pairs) >= max_pairs:
                    break
            if len(pairs) >= max_pairs:
                break
    else:
        pairs = [(i, j) for i in range(len(user_list)) for j in range(i + 1, len(user_list))]

    mmd_total = 0.0
    for i, j in pairs:
        fi = user_list[i]
        fj = user_list[j]
        if len(fi) > num_samples_per_user:
            idx = torch.randperm(len(fi))[:num_samples_per_user]
            fi = fi[idx]
        if len(fj) > num_samples_per_user:
            idx = torch.randperm(len(fj))[:num_samples_per_user]
            fj = fj[idx]
        mmd_total += compute_mmd(fi, fj, kernel_num=3)

    return mmd_total / max(len(pairs), 1)


# ==================== 跨用户对比学习 ====================
class CrossUserContrastiveLoss(nn.Module):
    """跨用户对比损失: 同手势不同用户=正样本, 不同手势=负样本"""
    def __init__(self, temperature=0.1, hard_neg_ratio=0.5):
        super().__init__()
        self.temperature = temperature
        self.hard_neg_ratio = hard_neg_ratio

    def forward(self, features, labels, users):
        B = features.size(0)
        features = F.normalize(features, dim=1)
        sim_matrix = torch.mm(features, features.t()) / self.temperature

        loss = 0.0
        valid_count = 0
        for i in range(B):
            pos_mask = (labels == labels[i]) & (users != users[i])
            neg_mask = labels != labels[i]
            n_pos = pos_mask.sum().item()
            n_neg = neg_mask.sum().item()
            if n_pos == 0 or n_neg == 0:
                continue

            # hard negative mining
            n_hard = max(1, int(n_neg * self.hard_neg_ratio))
            neg_sims = sim_matrix[i][neg_mask]
            hard_neg_idx = torch.topk(neg_sims, min(n_hard, len(neg_sims))).indices
            hard_neg_sims = neg_sims[hard_neg_idx]

            pos_sims = sim_matrix[i][pos_mask]
            numerator = torch.logsumexp(pos_sims, dim=0)
            denominator = torch.logsumexp(torch.cat([pos_sims, hard_neg_sims]), dim=0)
            loss += -(numerator - denominator)
            valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=features.device)
        return loss / valid_count


# ==================== V4 统一模型 ====================
class UnifiedModelV4(nn.Module):
    """
    v4: 主动用户不变性模型
    - 无用户分类头
    - GRL 对抗域对齐 (去除用户信息)
    - MMD 分布对齐 (温和去除用户信息)
    - 特征白化
    - 跨用户对比学习
    """
    def __init__(self, num_classes=4, num_users=175,
                 emg_channels=80, imu_channels=6,
                 use_imu=True,
                 use_whitening=True, use_contra=True,
                 use_grl=True, use_mmd=True,
                 grl_hidden_dim=256,
                 feat_dim=128, dropout=0.0):
        super().__init__()
        self.use_imu = use_imu
        self.use_whitening = use_whitening
        self.use_contra = use_contra
        self.use_grl = use_grl
        self.use_mmd = use_mmd
        self.num_users = num_users

        emg_dim, imu_dim = 256, 128

        # 编码器
        self.emg_encoder = EMGEncoder(emg_channels, emg_dim)
        if use_imu:
            self.imu_encoder = IMUEncoder(imu_channels, imu_dim)
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

        # 特征白化
        self.whitening = FeatureWhitening(feat_dim) if use_whitening else None

        # 手势分类器 (无用户分类头!)
        self.gesture_classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(feat_dim, num_classes))

        # 对比学习
        if use_contra:
            self.contra_criterion = CrossUserContrastiveLoss(
                temperature=0.1, hard_neg_ratio=0.5)

        # 对抗域对齐: 域判别器 + GRL
        # 注意: 这里的域判别器是被对抗的, 编码器要愚弄它 (与 v3 用户头相反)
        if use_grl:
            self.domain_discriminator = DomainDiscriminator(feat_dim, num_users, grl_hidden_dim)

        # 自监督预训练解码器 (延迟初始化)
        self.recon_decoder = None

    def forward(self, x_emg, x_imu=None, label=None, user=None, stage='train', grl_lambda=0.0):
        # 编码
        emg_feat = self.emg_encoder(x_emg)
        if self.use_imu and x_imu is not None:
            imu_feat = self.imu_encoder(x_imu)
            fused = self.fusion(emg_feat, imu_feat)
        else:
            fused = emg_feat

        feat = self.projection(fused)

        # 特征白化
        if self.use_whitening:
            feat = self.whitening(feat, training=self.training)

        # 手势分类
        gesture_logits = self.gesture_classifier(feat)

        if stage == 'train':
            out = {'gesture_logits': gesture_logits, 'gesture_feat': feat}

            # 对比学习
            if self.use_contra and label is not None and user is not None:
                out['contra_loss'] = self.contra_criterion(feat, label, user)

            # GRL 对抗域对齐: 编码器愚弄域判别器
            # grl_lambda=0 时不启用 (warmup 阶段)
            if self.use_grl and user is not None and grl_lambda > 0:
                reversed_feat = grad_reverse(feat, grl_lambda)
                out['domain_logits'] = self.domain_discriminator(reversed_feat)

            # MMD 分布对齐 (在特征上计算)
            if self.use_mmd and user is not None:
                out['mmd_loss'] = mmd_loss_across_users(feat, user)

            return out
        return gesture_logits

    def init_pretrain_heads(self, emg_channels=80, emg_len=150, pretrain_target='spectrogram',
                            n_fft=32, hop_length=8):
        """初始化预训练头 (频谱图重建)"""
        feat_dim = self.projection[0].out_features
        self.pretrain_target = pretrain_target
        with torch.no_grad():
            dummy = torch.zeros(emg_len)
            spec = torch.stft(dummy, n_fft=n_fft, hop_length=hop_length,
                              win_length=n_fft, return_complex=True,
                              window=torch.hann_window(n_fft))
            n_frames = spec.shape[-1]
        self.spec_reconstructor = SpectrogramReconstructionHead(
            feat_dim, num_channels=emg_channels, n_fft=n_fft, n_frames=n_frames)
        self.pretrain_contra = CrossUserContrastiveLoss(temperature=0.1, hard_neg_ratio=0.5)

    def forward_pretrain(self, x_emg, x_imu=None, labels=None, users=None,
                         mask_ratio=0.75, contra_weight=0.5):
        """自监督预训练: 频谱图重建 + 跨用户对比"""
        x_emg_masked, mask = random_time_mask(x_emg, mask_ratio)
        emg_feat = self.emg_encoder(x_emg_masked)
        if self.use_imu and x_imu is not None:
            imu_feat = self.imu_encoder(x_imu)
            fused = self.fusion(emg_feat, imu_feat)
        else:
            fused = emg_feat
        feat = self.projection(fused)

        spec_target = compute_stft_spectrogram(x_emg)
        B, C, nF, nT = spec_target.shape
        spec_pred = self.spec_reconstructor(feat).reshape(B, C, nF, nT)
        main_loss = F.mse_loss(spec_pred, spec_target)

        contra_loss = torch.tensor(0.0, device=feat.device)
        if labels is not None and users is not None and contra_weight > 0:
            contra_loss = self.pretrain_contra(feat, labels, users)

        total_loss = main_loss + contra_weight * contra_loss
        return total_loss, main_loss, contra_loss


# ==================== 预训练相关 (同 v3) ====================
def random_time_mask(x_emg, mask_ratio=0.75):
    B, C, T = x_emg.shape
    mask_len = int(T * mask_ratio)
    start = torch.randint(0, T - mask_len + 1, (B,), device=x_emg.device)
    mask = torch.ones(B, T, device=x_emg.device)
    for i in range(B):
        mask[i, start[i]:start[i] + mask_len] = 0
    x_masked = x_emg * mask.unsqueeze(1)
    return x_masked, mask


def compute_stft_spectrogram(x_emg, n_fft=32, hop_length=8):
    B, C, T = x_emg.shape
    x_flat = x_emg.reshape(B * C, T)
    spec = torch.stft(x_flat, n_fft=n_fft, hop_length=hop_length,
                      win_length=n_fft, return_complex=True,
                      window=torch.hann_window(n_fft, device=x_emg.device))
    spec = torch.log1p(torch.abs(spec))
    spec = spec.reshape(B, C, spec.shape[1], spec.shape[2])
    return spec


class SpectrogramReconstructionHead(nn.Module):
    def __init__(self, feat_dim, num_channels=80, n_fft=32, n_frames=19):
        super().__init__()
        self.num_channels = num_channels
        self.n_freq = n_fft // 2 + 1
        self.n_frames = n_frames
        output_dim = num_channels * self.n_freq * n_frames
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(),
            nn.Linear(512, output_dim))

    def forward(self, feat):
        return self.net(feat)


def build_model(name, **kwargs):
    if name == "unified":
        return UnifiedModelV4(**kwargs)
    raise ValueError(f"Unknown model: {name}")
