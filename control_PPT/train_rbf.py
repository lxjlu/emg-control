# -*- coding: utf-8 -*-
"""
train_rbf.py
用 App 录制的 CSV 训练 RBF SVM 手势模型（端侧演示用）。

CSV 格式与 App 一致：
    # label=手掌
    # user=李鑫
    # hand=right
    sample_index,ch1,...,ch8
    0,2048,2031,...,1997

用法：
    python train_rbf.py --data ../recordings --out rbf_model.joblib
"""

import argparse
import glob
import os

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from armband_ble import CHANNELS, SAMPLE_RATE
from features import extract

REST_LABEL = "休息"


def read_recording_csv(path):
    """返回 (label, user, hand, rows)。rows 为 [[8 通道 int], ...]。"""
    label = None
    user = None
    hand = None
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                body = line[1:].strip()
                if "=" in body:
                    key, value = body.split("=", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key == "label":
                        label = value
                    elif key == "user":
                        user = value
                    elif key == "hand":
                        hand = value.lower()
                continue
            parts = line.split(",")
            if len(parts) < CHANNELS + 1:
                continue
            try:
                values = [float(item) for item in parts]
            except ValueError:
                continue   # 跳过表头
            rows.append([int(round(v)) for v in values[1:CHANNELS + 1]])

    return label, user, hand, rows


def collect_dataset(data_dir, window, hop):
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    if not files:
        raise RuntimeError(f"目录中没有 CSV：{os.path.abspath(data_dir)}")

    X, y, groups = [], [], []
    used = 0
    for file_index, path in enumerate(files):
        label, _, _, rows = read_recording_csv(path)
        if not label or len(rows) < window:
            continue
        used += 1
        for start in range(0, len(rows) - window + 1, hop):
            X.append(extract(rows[start:start + window]))
            y.append(label)
            groups.append(file_index)

    if not X:
        raise RuntimeError("没有可用的训练窗口，请检查 CSV 是否含 # label= 与足够的样本行")

    print(f"读取 {used} 个文件，生成 {len(X)} 个窗口，{len(set(y))} 个类别")
    for label in sorted(set(y)):
        print(f"  {label}: {y.count(label)} 个窗口")
    return np.asarray(X, dtype=np.float64), np.asarray(y), np.asarray(groups)


def train_model(data_dir, out_path, window=100, hop=20, C=10.0, gamma="scale", cv=5):
    """读取录制目录 → 训练 RBF SVM → 保存模型，返回 payload。"""
    X, y, groups = collect_dataset(data_dir, window, hop)

    if len(set(y)) < 2:
        raise RuntimeError("至少需要 2 个不同动作标签")

    if REST_LABEL not in set(y.tolist()):
        print(f"提示：数据中没有「{REST_LABEL}」类，休息态将只能靠低置信度判定。")

    # 交叉验证评估（按录制文件分组，避免同一文件的窗口同时出现在训练/测试集）
    eval_model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced"),
    )
    folds = min(
        cv,
        len(np.unique(groups)),
        int(np.min(np.bincount(np.unique(y, return_inverse=True)[1]))),
        int(min(len(np.unique(groups[y == label])) for label in np.unique(y))),
    )
    if folds >= 2:
        try:
            splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
            scores = cross_val_score(
                eval_model, X, y,
                cv=splitter.split(X, y, groups),
                scoring="accuracy",
            )
            print(f"{folds} 折分组交叉验证准确率：{scores.mean() * 100:.1f}% "
                  f"（{scores.min() * 100:.1f}% ~ {scores.max() * 100:.1f}%）")
        except Exception as error:
            print(f"交叉验证跳过（数据量不足以分组：{error.__class__.__name__}）")
    else:
        print("交叉验证跳过（每个类别需要至少 2 个独立录制文件）")

    # 全量训练，启用概率输出
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=C, gamma=gamma,
            probability=True, class_weight="balanced", random_state=42),
    )
    model.fit(X, y)
    print(f"训练集准确率：{model.score(X, y) * 100:.1f}%")

    payload = {
        "model": model,
        "labels": list(model.classes_),
        "window": window,
        "hop": hop,
        "sample_rate": SAMPLE_RATE,
    }
    joblib.dump(payload, out_path)
    print(f"模型已保存：{os.path.abspath(out_path)}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="训练 RBF SVM 手势模型")
    parser.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "recordings"),
                        help="录制 CSV 目录（默认 ../recordings）")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "rbf_model.joblib"),
                        help="模型输出路径")
    parser.add_argument("--window", type=int, default=100, help="窗口样本数（默认 100）")
    parser.add_argument("--hop", type=int, default=20, help="滑窗步长（默认 20）")
    parser.add_argument("--C", type=float, default=10.0, help="RBF SVM 的 C（默认 10）")
    parser.add_argument("--gamma", default="scale", help="RBF gamma（默认 scale）")
    parser.add_argument("--cv", type=int, default=5, help="交叉验证折数（默认 5）")
    args = parser.parse_args()

    train_model(args.data, args.out, args.window, args.hop, args.C, args.gamma, args.cv)


if __name__ == "__main__":
    main()
