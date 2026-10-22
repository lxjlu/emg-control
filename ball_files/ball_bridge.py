#!/usr/bin/env python3
"""
ball_bridge.py
树莓派 UDP 中转：手机 App -> 侦察球

手机(App) --WiFi热点(192.168.123.1:8888)--> 树莓派 --eth0:1(192.168.1.50:3020)--> 侦察球(192.168.1.11:3020)

监听 8888 端口收到 App 的 11 字节二进制包，通过 192.168.1.50:3020 转发到机械球。
机械球要求源地址为 192.168.1.50，所以必须绑定到 eth0:1。

用法：
  python3 ball_bridge.py
  python3 ball_bridge.py --source-ip 192.168.1.50 --source-port 3020 --ball-ip 192.168.1.11 --ball-port 3020 --listen-port 8888
"""

import argparse
import logging
import socket
import struct
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BALL_MAGIC = b"\x01\x81"


def is_ball_packet(data):
    """检测是否是侦察球包：11 字节且前两字节为 0x01 0x81。"""
    return len(data) == 11 and data[:2] == BALL_MAGIC


def parse_ball_packet(data):
    """解析球包的可读信息，用于日志。"""
    if not is_ball_packet(data):
        return f"非球包({len(data)}字节)"
    control = data[2]
    steer = struct.unpack_from("<h", data, 3)[0]
    drive = struct.unpack_from("<h", data, 5)[0]
    ctrl_name = {0x00: "正常", 0x20: "锁定", 0x40: "解锁"}.get(control, f"0x{control:02X}")
    return f"{ctrl_name} steer={steer} drive={drive}"


def main():
    parser = argparse.ArgumentParser(description="侦察球 UDP 中转桥")
    parser.add_argument("--source-ip", default="192.168.1.50",
                        help="绑定到 eth0:1 的 IP（机械球要求源地址）")
    parser.add_argument("--source-port", type=int, default=3020,
                        help="源端口（和机械球一致）")
    parser.add_argument("--ball-ip", default="192.168.1.11",
                        help="机械球 IP")
    parser.add_argument("--ball-port", type=int, default=3020,
                        help="机械球端口")
    parser.add_argument("--listen-ip", default="0.0.0.0",
                        help="监听 App 命令的 IP")
    parser.add_argument("--listen-port", type=int, default=8888,
                        help="监听 App 命令的端口（手机 App 填的端口）")
    args = parser.parse_args()

    # ===== 球通信 socket（绑定到 192.168.1.50:3020，连接到 192.168.1.11:3020）=====
    ball_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        ball_sock.bind((args.source_ip, args.source_port))
        ball_sock.connect((args.ball_ip, args.ball_port))
    except OSError as e:
        logger.error(f"绑定 {args.source_ip}:{args.source_port} 失败: {e}")
        logger.error("请确认 eth0:1 已配置: sudo ifconfig eth0:1 192.168.1.50 netmask 255.255.255.0 up")
        return 1
    logger.info(f"球通信通道: {args.source_ip}:{args.source_port} -> {args.ball_ip}:{args.ball_port}")

    # ===== App 监听 socket =====
    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listen_sock.bind((args.listen_ip, args.listen_port))
    listen_sock.setblocking(False)
    logger.info(f"监听 App 命令: {args.listen_ip}:{args.listen_port}")

    # 统计
    packets_forwarded = 0
    last_forward_time = 0

    logger.info("=== 侦察球中转桥已启动 ===")
    logger.info("手机 App 填: IP=192.168.123.1  端口=8888  设备类型=侦察球")
    logger.info("等待 App 命令...")

    running = True
    while running:
        try:
            data, addr = listen_sock.recvfrom(1024)

            if is_ball_packet(data):
                # 转发到机械球
                ball_sock.send(data)
                packets_forwarded += 1
                last_forward_time = time.time()
                logger.info(f"转发 [{parse_ball_packet(data)}] {addr[0]}:{addr[1]} -> {args.ball_ip}:{args.ball_port} (#{packets_forwarded})")
            else:
                # 可能是 Go2 字符串命令，忽略
                try:
                    raw = data.decode("utf-8").strip()
                    logger.info(f"忽略非球包命令: {raw} from {addr}")
                except UnicodeDecodeError:
                    logger.info(f"忽略未知数据: {data.hex()} from {addr}")

        except BlockingIOError:
            time.sleep(0.001)
        except KeyboardInterrupt:
            logger.info("收到 Ctrl+C，退出")
            running = False
        except Exception as e:
            logger.warning(f"异常: {e}")

    ball_sock.close()
    listen_sock.close()
    logger.info(f"总计转发 {packets_forwarded} 个包")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
