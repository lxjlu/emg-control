# -*- coding: utf-8 -*-
"""
collect_and_train.py
一键流程：BLE 连接 → 引导式数据采集 → 训练 RBF SVM → 生成模型。

运行一次即可得到可用的 rbf_model.joblib：
    python collect_and_train.py

默认采集 5 个类别（休息 / 握拳 / 手掌 / 向左 / 向右），每类 5 轮 × 4 秒。
采集数据写入 control_PPT/dataset，模型写入 control_PPT/rbf_model.joblib。

常用参数：
    --seconds 4           每轮采集时长（默认 4 秒）
    --rounds 5            每类重复轮数（默认 5 轮）
    --labels 休息,握拳,手掌,向左,向右
    --data ./dataset      数据集目录（默认 control_PPT/dataset）
    --train-only          跳过采集，只用已有数据训练
    --diagnose            只诊断 BLE 数据流（排查断流/掉线）
"""

import argparse
import asyncio
import os
import time
from datetime import datetime

from armband_ble import DEVICE_NAME_HINT, SAMPLE_RATE, ArmbandBle
from record_csv import write_csv
from train_rbf import train_model

DEFAULT_LABELS = "休息,握拳,手掌,向左,向右"
SEGMENT_ATTEMPTS = 3      # 单段采样失败（掉线/断流）时的重试次数


def save_segment(out_dir, label, user, hand, rows):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    session = f"{user}_{label}_{stamp}"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stamp}_{label}.csv")
    write_csv(path, label, user, hand, session, rows)
    return path


async def collect_all(args, labels):
    current = {"rows": []}

    def on_samples(samples):
        current["rows"].extend(samples)

    arm = ArmbandBle(address=args.address, on_samples=on_samples, on_status=print)
    await arm.connect()

    total = len(labels) * args.rounds
    done = 0
    print("按提示做动作，采集期间保持姿势稳定。Ctrl+C 可中断。")

    try:
        for label in labels:
            for round_index in range(1, args.rounds + 1):
                done += 1
                print(f"\n[{done}/{total}] 准备「{label}」（第 {round_index} 轮）")

                rows = []
                for attempt in range(1, SEGMENT_ATTEMPTS + 1):
                    if not arm.is_connected:
                        if not await arm.reconnect():
                            print("  无法重连，终止采集。")
                            return

                    prepare = args.prepare if attempt == 1 else 1
                    for sec in range(prepare, 0, -1):
                        print(f"  {sec} 秒后开始…", end="\r", flush=True)
                        await asyncio.sleep(1)

                    current["rows"] = []
                    started = time.time()
                    last_print = 0.0
                    failed = False

                    while time.time() - started < args.seconds:
                        await asyncio.sleep(0.05)
                        now = time.time()

                        if not arm.is_connected:
                            failed = True
                            print("\n  连接断开，本段作废。")
                            break
                        if now - arm.last_notification_at > 1.5:
                            failed = True
                            print("\n  超过 1.5 秒没有数据，本段作废。")
                            break

                        if now - last_print >= 0.5:
                            last_print = now
                            print(f"\r  采集中 {len(current['rows']) / SAMPLE_RATE:.1f}"
                                  f" / {args.seconds:.0f} 秒", end="", flush=True)

                    print()
                    rows = list(current["rows"])
                    if not failed and len(rows) >= 200:
                        break

                    if attempt < SEGMENT_ATTEMPTS:
                        print(f"  重试本段（第 {attempt + 1}/{SEGMENT_ATTEMPTS} 次）…")
                        await asyncio.sleep(0.5)
                    else:
                        print(f"  本段连续 {SEGMENT_ATTEMPTS} 次失败，已跳过。")

                if len(rows) < 200:
                    continue

                path = save_segment(args.data, label, args.user, args.hand, rows)
                print(f"  已保存 {os.path.basename(path)}（{len(rows)} 样本）")
                await asyncio.sleep(args.gap)
    finally:
        await arm.disconnect()
        print("连接统计：" + arm.stats())


async def diagnose(args, seconds):
    arm = ArmbandBle(address=args.address, on_status=print)
    await arm.connect()
    print(f"开始诊断 {seconds:.0f} 秒，请长按 SW2 启动采集…")
    last_notes = 0
    last_bytes = 0
    for second in range(1, int(seconds) + 1):
        await asyncio.sleep(1)
        notes = arm.notifications - last_notes
        bytes_now = arm.bytes_received - last_bytes
        last_notes, last_bytes = arm.notifications, arm.bytes_received
        print(f"  {second:2d}s  通知 {notes:4d}/s  字节 {bytes_now:6d}/s  "
              f"CRC {arm.parser.crc_errors}  "
              f"连接={'是' if arm.is_connected else '否'}  MTU={arm.mtu}")
    await arm.disconnect()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="一键采集并训练 RBF SVM 手势模型")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help=f"类别列表（默认 {DEFAULT_LABELS}）")
    parser.add_argument("--seconds", type=float, default=4.0, help="每轮采集时长（默认 4 秒）")
    parser.add_argument("--rounds", type=int, default=5, help="每类重复轮数（默认 5 轮）")
    parser.add_argument("--prepare", type=int, default=3, help="每段前倒计时秒数（默认 3）")
    parser.add_argument("--gap", type=float, default=2.0, help="段间休息秒数（默认 2）")
    parser.add_argument("--user", default="demo", help="用户名（默认 demo）")
    parser.add_argument("--hand", default="right", choices=["left", "right"], help="手别")
    parser.add_argument("--data", default=os.path.join(here, "dataset"),
                        help="数据集目录（默认 control_PPT/dataset）")
    parser.add_argument("--out", default=os.path.join(here, "rbf_model.joblib"),
                        help="模型输出路径")
    parser.add_argument("--address", default=None,
                        help=f"BLE 地址；省略则自动扫描名称含 {DEVICE_NAME_HINT} 的设备")
    parser.add_argument("--C", type=float, default=10.0, help="RBF SVM 的 C")
    parser.add_argument("--gamma", default="scale", help="RBF gamma")
    parser.add_argument("--cv", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--train-only", action="store_true", help="跳过采集，仅用已有数据训练")
    parser.add_argument("--diagnose", action="store_true",
                        help="只诊断 BLE 数据流（每秒打印通知数与连接状态），不采集不训练")
    parser.add_argument("--diagnose-seconds", type=float, default=20.0,
                        help="诊断时长（默认 20 秒）")
    args = parser.parse_args()

    if args.diagnose:
        try:
            asyncio.run(diagnose(args, args.diagnose_seconds))
        except KeyboardInterrupt:
            print("\n已退出")
        return

    if not args.train_only:
        labels = [item.strip() for item in args.labels.split(",") if item.strip()]
        if not labels:
            raise SystemExit("类别列表为空")
        try:
            asyncio.run(collect_all(args, labels))
        except KeyboardInterrupt:
            print("\n采集中断，未生成模型。请重新运行。")
            return

    print("\n=== 开始训练 RBF SVM ===")
    payload = train_model(args.data, args.out, C=args.C, gamma=args.gamma, cv=args.cv)

    print("\n=== 完成 ===")
    print(f"模型：{os.path.abspath(args.out)}")
    print(f"类别：{'、'.join(payload['labels'])}")
    print("下一步可用手势控制 PPT：")
    print(f"  python ppt_control.py --delay 8")


if __name__ == "__main__":
    main()
