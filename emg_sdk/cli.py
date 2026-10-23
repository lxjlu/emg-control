# -*- coding: utf-8 -*-
"""命令行入口。

    python -m emg_sdk list
    python -m emg_sdk run --adapter go2 --port 8888
    python -m emg_sdk run --adapter ball --dry-run
    python -m emg_sdk run --adapter go2 --port 8888 --monitor 8080
    python -m emg_sdk monitor --port 8888 --http-port 8080
    python -m emg_sdk send --host 192.168.10.1 --act move --vx 0.6 --repeat 20
    python -m emg_sdk send --host 192.168.10.1 --act hello --wait
"""

import argparse
import json
import socket
import sys
import time

from .adapters import ADAPTERS
from .protocol import DISCRETE_ACTIONS, PROTOCOL_VERSION, encode_command
from .server import CommandServer


# ----------------------------------------------------------------------
def log(message=""):
    """行缓冲输出：重定向到文件 / systemd journal 时也能实时看到。"""
    print(message, flush=True)


def build_adapter(name, args, log=log):
    if name == "go2":
        return ADAPTERS[name](interface=args.interface, robot_ip=args.robot_ip,
                              dry_run=args.dry_run, log=log)
    if name == "go2w":
        return ADAPTERS[name](interface=args.interface, robot_ip=args.robot_ip,
                              balance_settle=args.balance_settle,
                              dry_run=args.dry_run, log=log)
    if name == "ball":
        return ADAPTERS[name](source_ip=args.source_ip, source_port=args.source_port,
                              ball_ip=args.ball_ip, ball_port=args.ball_port,
                              speed=args.speed, servo=args.servo,
                              turn_speed=args.turn_speed,
                              dry_run=args.dry_run, log=log)
    if name == "car":
        return ADAPTERS[name](target=args.target, dry_run=args.dry_run, log=log)
    if name == "ppt":
        return ADAPTERS[name](focus=not args.no_focus,
                              require_window=args.require_window,
                              restore_screen=args.restore_screen,
                              dry_run=args.dry_run, log=log)
    raise SystemExit(f"未知适配器: {name}")


# ----------------------------------------------------------------------
def cmd_list(_args):
    print(f"协议 v{PROTOCOL_VERSION}    可用适配器：\n")
    for name, cls in ADAPTERS.items():
        probe = cls(dry_run=True, log=lambda *_: None)
        caps = probe.capabilities()
        actions = "、".join(caps.get("actions", []))
        print(f"  {name:<6} {caps.get('name', ''):<18} 动作: {actions}")
        print(f"         心跳 {caps.get('publish_hz')}Hz  "
              f"超时停 {caps.get('deadman_ms')}ms  "
              + (f"速度上限 {caps['max_speed']}" if "max_speed" in caps else ""))
    print("\n示例：")
    print("  python -m emg_sdk run --adapter go2 --port 8888")
    print("  python -m emg_sdk send --host 192.168.10.1 --act move --vx 0.6 --wz 0")
    print("  python -m emg_sdk send --host 192.168.10.1 --act stop")
    return 0


def cmd_run(args):
    adapter = build_adapter(args.adapter, args)
    monitor = None
    if args.monitor:
        from .monitor import EmgMonitor
        monitor = EmgMonitor(http_port=args.monitor, log=log)
    server = CommandServer(adapter, host=args.host, port=args.port,
                           token=args.token, ack=not args.no_ack,
                           monitor=monitor, log=log)
    server.serve_forever()
    return 0


def cmd_monitor(args):
    from .monitor import EmgMonitor, MonitorAdapter
    monitor = EmgMonitor(http_port=args.http_port, log=log)
    server = CommandServer(MonitorAdapter(log=log), host=args.host, port=args.port,
                           token=args.token, monitor=monitor, log=log)
    server.serve_forever()
    return 0


def cmd_send(args):
    action = args.act.lower()
    arg = {}
    if action == "move":
        arg = {"vx": args.vx, "wz": args.wz}
    elif action not in DISCRETE_ACTIONS and action not in ("stop", "hello"):
        raise SystemExit(f"未知动作: {action}")

    payload = encode_command(action, arg, seq=args.seq, token=args.token)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(args.timeout)

    try:
        for index in range(args.repeat):
            sock.sendto(payload, (args.host, args.port))
            if args.repeat > 1:
                print(f"已发送 {index + 1}/{args.repeat}: {payload.decode('utf-8')}")
            if args.wait and index == 0:
                try:
                    data, addr = sock.recvfrom(4096)
                    print(f"回复 {addr[0]}:{addr[1]}: "
                          f"{json.dumps(json.loads(data.decode('utf-8')), ensure_ascii=False, indent=2)}")
                except socket.timeout:
                    print("等待回复超时（服务端可能未启用 ack，或地址/端口不对）")
                    return 1
            if index + 1 < args.repeat:
                time.sleep(args.interval)
    finally:
        sock.close()

    if args.repeat == 1 and not args.wait:
        print(f"已发送到 {args.host}:{args.port}: {payload.decode('utf-8')}")
    return 0


# ----------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(prog="python -m emg_sdk", description="手势控制统一 SDK")
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="列出可用适配器与其能力")
    listing.set_defaults(func=cmd_list)

    run = sub.add_parser("run", help="启动服务端")
    run.add_argument("--adapter", required=True, choices=sorted(ADAPTERS), help="被控对象类型")
    run.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    run.add_argument("--port", type=int, default=8888, help="监听端口（默认 8888）")
    run.add_argument("--token", default=None, help="共享 token；设置后只接受带同样 token 的报文")
    run.add_argument("--no-ack", action="store_true", help="不回复 ack")
    run.add_argument("--dry-run", action="store_true", help="只打印，不真的驱动设备")
    run.add_argument("--interface", default="eth0", help="Go2 的 Cyclone DDS 网卡")
    run.add_argument("--robot-ip", default=None, help="Go2 机器人 IP（默认自动）")
    run.add_argument("--balance-settle", type=float, default=1.5,
                     help="Go2-W BalanceStand 后等待秒数（默认 1.5）")
    run.add_argument("--source-ip", default="192.168.1.50", help="侦察球源 IP（eth0:1）")
    run.add_argument("--source-port", type=int, default=3020, help="侦察球源端口")
    run.add_argument("--ball-ip", default="192.168.1.11", help="侦察球目标 IP")
    run.add_argument("--ball-port", type=int, default=3020, help="侦察球目标端口")
    run.add_argument("--speed", type=int, default=1000, help="侦察球速度上限（默认 1000）")
    run.add_argument("--servo", type=int, default=0, help="侦察球舵机值")
    run.add_argument("--turn-speed", type=int, default=0x12, help="侦察球转向速度")
    run.add_argument("--target", default="192.168.2.10:8888", help="机械车目标 host:port")
    run.add_argument("--no-focus", action="store_true",
                     help="PPT：不自动把放映窗口切到前台（默认会切）")
    run.add_argument("--require-window", action="store_true",
                     help="PPT：找不到 PPT/WPS 窗口时不发键")
    run.add_argument("--restore-screen", action="store_true",
                     help="PPT：退出时按一次空格，尝试恢复黑屏/白屏")
    run.add_argument("--monitor", type=int, nargs="?", const=8080, default=None,
                     metavar="HTTP_PORT",
                     help="同时开启肌电波形网页（默认端口 8080）")
    run.set_defaults(func=cmd_run)

    watch = sub.add_parser("monitor", help="只看 8 通道肌电波形，不驱动设备")
    watch.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    watch.add_argument("--port", type=int, default=8888, help="接收 App 的 UDP 端口（默认 8888）")
    watch.add_argument("--http-port", type=int, default=8080, help="波形网页端口（默认 8080）")
    watch.add_argument("--token", default=None, help="与 App 约定的共享 token")
    watch.set_defaults(func=cmd_monitor)

    send = sub.add_parser("send", help="发送一条语义命令（调试用）")
    send.add_argument("--host", required=True, help="SDK 所在主机 IP")
    send.add_argument("--port", type=int, default=8888)
    send.add_argument("--act", required=True,
                      help="move / stop / hello / next / prev / start / exit / open / close")
    send.add_argument("--vx", type=float, default=0.0, help="前后速度归一化 -1..1（正为前进）")
    send.add_argument("--wz", type=float, default=0.0, help="转向归一化 -1..1（正为左转）")
    send.add_argument("--seq", type=int, default=None, help="序号，用于去重")
    send.add_argument("--token", default=None)
    send.add_argument("--repeat", type=int, default=1, help="重复发送次数（连续量建议 20）")
    send.add_argument("--interval", type=float, default=0.05, help="重复间隔秒")
    send.add_argument("--wait", action="store_true", help="等待一条回复（hello 用）")
    send.add_argument("--timeout", type=float, default=1.5, help="等待回复超时")
    send.set_defaults(func=cmd_send)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n已中断")
        return 130


if __name__ == "__main__":
    sys.exit(main())
