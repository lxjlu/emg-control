# -*- coding: utf-8 -*-
"""
demo_control.py
RBF SVM 手势识别 + 稳定控制演示（独立于 Android App）。

两种数据源：
    实时：BLE 连接肌电臂环，读取并解析 STM32 帧
    回放：直接读取 App 录制的 CSV，用于没有硬件时验证

稳定性机制（演示优先）：
    1) 滑窗 100 / 步长 20，每 40 ms 推理一次
    2) 7 帧投票，同一标签至少 5 票才采用
    3) 置信度低于 0.48 视为「不确定」
    4) 中立锁存：必须先回「休息 / 不确定」，才允许切换到新动作
    5) 命令二次确认：新命令连续 2 次一致才执行
    6) 1 秒无有效识别 → 自动停止
    7) 生效后每 50 ms 重发当前命令（可选 UDP 输出）

用法：
    python demo_control.py --list                     # 扫描 BLE 设备
    python demo_control.py                            # BLE 实时
    python demo_control.py --address AA:BB:CC:DD:EE:FF
    python demo_control.py --csv ../recordings/xxx.csv --speed 0    # 回放验证
    python demo_control.py --udp-ip 192.168.123.1 --udp-port 8888   # 同时发 UDP
"""

import argparse
import asyncio
import os
import socket
import time
from collections import deque

import joblib
import numpy as np

from armband_ble import SAMPLE_RATE, ArmbandBle, list_devices
from features import extract
from train_rbf import read_recording_csv

# 手势标签 → 控制命令。
# 「休息」、低置信度、未映射手势都不产生动作（统一走中立分支）。
COMMAND_MAP = {
    "握拳": "backward", "fist": "backward", "后退": "backward",
    "手掌": "forward", "palm": "forward", "前进": "forward",
    "向左": "left", "左": "left", "左转": "left",
    "向右": "right", "右": "right", "右转": "right",
    "停止": "stop", "停": "stop", "stop": "stop",
}
REST_LABELS = {"休息", "rest"}


class UdpSender:
    def __init__(self, host, port, prefix=""):
        self.address = (host, int(port))
        self.prefix = prefix
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def __call__(self, command):
        self.sock.sendto((self.prefix + command).encode("utf-8"), self.address)


class StabilityController:
    """把逐帧预测转成稳定的设备命令。"""

    def __init__(self, send=None, log=print, conf=0.48, vote_window=7,
                 vote_needed=5, vote_min=3, confirm=2, timeout=1.0):
        self.send = send or (lambda command: None)
        self.log = log
        self.conf = conf
        self.vote_window = vote_window
        self.vote_needed = vote_needed
        self.vote_min = vote_min
        self.confirm = confirm
        self.timeout = timeout

        self.votes = deque(maxlen=vote_window)
        self.active = None
        self.neutral = True
        self.candidate = None
        self.candidate_count = 0
        self.last_result_at = time.time()
        self.last_send_at = 0.0
        self.executed = 0

    def on_prediction(self, label, confidence):
        self.votes.append(label)
        if len(self.votes) < self.vote_min:
            self._handle(label, confidence)
            return
        if self.votes.count(label) >= self.vote_needed:
            self._handle(label, confidence)

    def _handle(self, label, confidence):
        self.last_result_at = time.time()
        command = COMMAND_MAP.get(label) if confidence >= self.conf else None

        if command is None:
            # 休息 / 低置信度 / 未映射 → 一律不做动作
            was_active = self.active is not None
            self.candidate = None
            self.candidate_count = 0
            self.neutral = True
            if was_active:
                self.active = None
                self.send("stop")
                reason = "已回休息" if label in REST_LABELS else "低置信度/未映射"
                self.log(f"· {reason} → 停止")
            return

        if command == "stop":
            self.candidate = None
            self.candidate_count = 0
            self.neutral = True
            if self.active is not None:
                self.active = None
                self.send("stop")
                self.log("· 已识别停止")
            return

        if command == self.active:
            self.candidate = None
            self.candidate_count = 0
            return

        if not self.neutral:
            return   # 未经过中立，忽略直接跳变

        if command != self.candidate:
            self.candidate = command
            self.candidate_count = 1
            return

        self.candidate_count += 1
        if self.candidate_count < self.confirm:
            return

        self.candidate = None
        self.candidate_count = 0
        self.active = command
        self.neutral = False
        self.last_send_at = time.time()
        self.send(command)          # 生效即发一次，后续由 tick() 以 50 ms 周期重发
        self.executed += 1
        self.log(f"▶ 执行 {command}")

    def tick(self):
        if self.active is None:
            return
        now = time.time()
        if now - self.last_result_at > self.timeout:
            self.active = None
            self.send("stop")
            self.log("· 识别中断 → 停止")
            return
        if now - self.last_send_at >= 0.05:
            self.send(self.active)
            self.last_send_at = now

    def stop(self):
        if self.active is not None:
            self.active = None
            self.send("stop")
        else:
            self.send("stop")


def make_controller(args, log=print):
    sender = None
    if args.udp_ip:
        sender = UdpSender(args.udp_ip, args.udp_port, args.prefix)
        log(f"UDP 输出：{args.udp_ip}:{args.udp_port}（前缀 '{args.prefix}'）")
    return StabilityController(
        send=sender, log=log, conf=args.conf,
        vote_window=args.vote_window, vote_needed=args.vote_needed,
        confirm=args.confirm, timeout=args.timeout,
    )


def predict(payload, buffer):
    window = list(buffer)[-payload["window"]:]
    features = np.asarray([extract(window)], dtype=np.float64)
    model = payload["model"]
    probabilities = model.predict_proba(features)[0]
    index = int(probabilities.argmax())
    return payload["labels"][index], float(probabilities[index])


async def run_live(args, payload):
    controller = make_controller(args)
    buffer = deque(maxlen=payload["window"])
    pending = 0
    received = {"packets": 0}

    def on_samples(samples):
        nonlocal pending
        if received["packets"] == 0:
            print("收到首个数据包，开始识别…")
        received["packets"] += 1
        for row in samples:
            buffer.append(row)
            pending += 1
        while pending >= payload["hop"] and len(buffer) >= payload["window"]:
            pending -= payload["hop"]
            label, confidence = predict(payload, buffer)
            controller.on_prediction(label, confidence)

    arm = ArmbandBle(address=args.address, on_samples=on_samples, on_status=print)
    await arm.connect()
    print("按 Ctrl+C 退出。")
    try:
        while True:
            if not arm.is_connected:
                print("连接断开，正在重连…")
                if await arm.reconnect():
                    print("重连成功，继续识别。")
                else:
                    await asyncio.sleep(10)
            controller.tick()
            await asyncio.sleep(0.02)
    finally:
        await arm.disconnect()
        controller.stop()
        print("统计：" + arm.stats())
        print(f"共执行 {controller.executed} 次动作命令")


async def run_replay(args, payload):
    label, user, hand, rows = read_recording_csv(args.csv)
    if not rows:
        raise RuntimeError(f"CSV 中没有样本：{args.csv}")
    print(f"回放文件：{os.path.basename(args.csv)}")
    print(f"录制标签：{label or '未知'} · 用户：{user or '未知'} · 手别：{hand or '未知'} · "
          f"{len(rows)} 样本（{len(rows) / SAMPLE_RATE:.2f} 秒）")

    controller = make_controller(args)
    buffer = deque(maxlen=payload["window"])
    hop = payload["hop"]
    speed = args.speed if args.speed > 0 else 0.0

    for start in range(0, len(rows), hop):
        chunk = rows[start:start + hop]
        for row in chunk:
            buffer.append(row)
        if len(buffer) >= payload["window"]:
            predicted, confidence = predict(payload, buffer)
            controller.on_prediction(predicted, confidence)
        controller.tick()
        if speed > 0:
            await asyncio.sleep(len(chunk) / SAMPLE_RATE / speed)

    await asyncio.sleep(0.2)
    controller.stop()
    print(f"回放结束 · 共执行 {controller.executed} 次动作命令")


def main():
    parser = argparse.ArgumentParser(description="RBF SVM 手势识别与稳定控制演示")
    parser.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "rbf_model.joblib"),
                        help="模型文件（先运行 train_rbf.py）")
    parser.add_argument("--csv", default=None, help="回放模式：指定一个录制 CSV")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="回放速度倍率，1=实时，0=不限速（默认 1）")
    parser.add_argument("--address", default=None, help="BLE 设备地址；省略则自动扫描 BT-11")
    parser.add_argument("--list", action="store_true", help="扫描并列出 BLE 设备后退出")
    parser.add_argument("--scan-timeout", type=float, default=8.0, help="BLE 扫描时长（秒）")

    parser.add_argument("--conf", type=float, default=0.48, help="置信度阈值（默认 0.48）")
    parser.add_argument("--vote-window", type=int, default=7, help="投票窗口（默认 7）")
    parser.add_argument("--vote-needed", type=int, default=5, help="所需票数（默认 5）")
    parser.add_argument("--confirm", type=int, default=2, help="新命令确认次数（默认 2）")
    parser.add_argument("--timeout", type=float, default=1.0, help="无有效识别自动停止（秒）")

    parser.add_argument("--udp-ip", default=None, help="可选：把命令用 UDP 发到树莓派")
    parser.add_argument("--udp-port", type=int, default=8888, help="UDP 端口（默认 8888）")
    parser.add_argument("--prefix", default="", help="UDP 命令前缀，如 go2w:")
    args = parser.parse_args()

    if args.list:
        devices = list_devices(args.scan_timeout)
        if not devices:
            print("未扫描到 BLE 设备")
            return
        print(f"扫描到 {len(devices)} 个设备：")
        for item in devices:
            mark = " ← 可能是臂环（FFE0）" if item["likely_armband"] else ""
            services = ",".join(item.get("services_short") or [])
            tag = f"  [{services}]" if services else ""
            print(f"  {item['address']}  {item['name'] or '未命名'}{tag}{mark}")
        return

    if not os.path.exists(args.model):
        raise SystemExit(f"模型不存在：{args.model}\n请先运行：python train_rbf.py --data ../recordings")

    payload = joblib.load(args.model)
    print("模型信息：")
    print(f"  类别：{'、'.join(payload['labels'])}")
    print(f"  窗口 {payload['window']} · 步长 {payload['hop']} · 采样率 {payload['sample_rate']} Hz")
    print(f"  稳定策略：{args.vote_window} 帧需 {args.vote_needed} 票，"
          f"置信度 ≥ {args.conf}，新命令确认 {args.confirm} 次")

    try:
        if args.csv:
            asyncio.run(run_replay(args, payload))
        else:
            asyncio.run(run_live(args, payload))
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    main()
