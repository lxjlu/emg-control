# -*- coding: utf-8 -*-
"""
features.py
56 维时域特征，逐行移植自 Android 端 GestureModel.features()。

每个通道 7 维：
    log1p(MAV)        平均绝对值
    log1p(RMS)        均方根
    log1p(WL)         波形长度（相邻差分绝对值均值）
    log1p(能量)       均方值
    过零率            ZC（带幅度阈值）
    斜率变号率        SSC（带双阈值）
    RMS 占比          该通道 RMS / 全部通道 RMS 之和

特征对增益不敏感，与 App 端 RBF SVM 的训练/推理保持一致。
"""

import math

from armband_ble import CHANNELS


def extract(window):
    """window: 形如 [[ch1..ch8], ...] 的整数样本列表，长度建议 >= 32。"""
    n = len(window)
    if n < 32:
        raise ValueError("动作窗口太短")

    result = [0.0] * (CHANNELS * 7)
    rms = [0.0] * CHANNELS
    total_rms = 1e-9

    for channel in range(CHANNELS):
        mean = sum(row[channel] for row in window) / n
        mav = 0.0
        square = 0.0
        length = 0.0
        zero = 0.0
        slope = 0.0

        previous = window[0][channel] - mean
        previous_delta = 0.0

        for i in range(n):
            value = window[i][channel] - mean
            mav += abs(value)
            square += value * value
            if i > 0:
                delta = value - previous
                length += abs(delta)
                if value * previous < 0 and abs(delta) > 2:
                    zero += 1
                if (i > 1 and delta * previous_delta < 0
                        and abs(delta) > 2 and abs(previous_delta) > 2):
                    slope += 1
                previous_delta = delta
            previous = value

        mav /= n
        rms[channel] = math.sqrt(square / n + 1e-9)
        total_rms += rms[channel]

        at = channel * 7
        result[at] = math.log1p(mav)
        result[at + 1] = math.log1p(rms[channel])
        result[at + 2] = math.log1p(length / max(1, n - 1))
        result[at + 3] = math.log1p(square / n)
        result[at + 4] = zero / max(1, n - 1.0)
        result[at + 5] = slope / max(1, n - 2.0)

    for channel in range(CHANNELS):
        result[channel * 7 + 6] = rms[channel] / total_rms

    return result
