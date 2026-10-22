# -*- coding: utf-8 -*-
"""
record_csv.py
从 BLE 臂环录制带标签的 CSV，用于补充训练数据（尤其是「休息」类）。

输出格式与 App 录制完全一致，可直接放进 recordings/ 参与训练。

用法：
    python record_csv.py --label 休息 --seconds 30
    python record_csv.py --label 休息 --seconds 60 --hand right --user 李鑫
    python record_csv.py --label 休息 --address AA:BB:CC:DD:EE:FF
"""

import argparse
import asyncio
import os
import time
from datetime import datetime

from armband_ble import CHANNELS, DEVICE_NAME_HINT, SAMPLE_RATE, ArmbandBle


def write_csv(path, label, user, hand, session, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("# sEMG recording v1\n")
        handle.write(f"# label={label}\n")
        handle.write(f"# user={user}\n")
        handle.write(f"# session={session}\n")
        handle.write("# protocol=STM32\n")
        handle.write(f"# hand={hand}\n")
        handle.write("sample_index," + ",".join(f"ch{i + 1}" for i in range(CHANNELS)) + "\n")
        for index, row in enumerate(rows):
            handle.write(str(index) + "," + ",".join(str(value) for value in row) + "\n")


async def record(args):
    rows = []

    def on_samples(samples):
        rows.extend(samples)

    arm = ArmbandBle(address=args.address, on_samples=on_samples, on_status=print)
    await arm.connect()
    print(f"开始录制「{args.label}」{args.seconds:.0f} 秒…（Ctrl+C 提前结束）")

    started = time.time()
    last_print = 0.0
    try:
        while time.time() - started < args.seconds:
            await asyncio.sleep(0.05)
            now = time.time()
            if now - last_print >= 0.5:
                last_print = now
                print(f"\r已录制 {len(rows)} 样本 / {len(rows) / SAMPLE_RATE:.1f} 秒",
                      end="", flush=True)
    except KeyboardInterrupt:
        print("\n提前结束录制")
    finally:
        await arm.disconnect()

    print()
    if len(rows) < 100:
        raise SystemExit(f"样本太少（{len(rows)}），未保存。请确认手环已长按 SW2 启动采集。")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    session = f"{args.user}_{args.label}_{stamp}"
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{stamp}_{args.label}.csv")
    write_csv(path, args.label, args.user, args.hand, session, rows)

    print(f"已保存：{os.path.abspath(path)}（{len(rows)} 样本 / {len(rows) / SAMPLE_RATE:.1f} 秒）")
    print(f"下一步：python train_rbf.py --data {os.path.abspath(args.out)}")


def main():
    parser = argparse.ArgumentParser(description="从 BLE 臂环录制带标签 CSV")
    parser.add_argument("--label", required=True, help="动作标签，如 休息 / 握拳 / 手掌")
    parser.add_argument("--seconds", type=float, default=30.0, help="录制时长（秒，默认 30）")
    parser.add_argument("--user", default="python", help="用户名（默认 python）")
    parser.add_argument("--hand", default="right", choices=["left", "right"], help="手别（默认 right）")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "recordings"),
                        help="输出目录（默认 ../recordings）")
    parser.add_argument("--address", default=None,
                        help=f"BLE 地址；省略则自动扫描名称含 {DEVICE_NAME_HINT} 的设备")
    args = parser.parse_args()

    try:
        asyncio.run(record(args))
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    main()
