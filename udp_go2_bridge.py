#!/usr/bin/env python3
"""
udp_go2_bridge.py
树莓派 UDP 中转控制宇树 Go2

手机(App) --WiFi热点(192.168.10.1)--> 树莓派 --网线(192.168.123.18)--> Go2

接收手机 sEMG App 发来的 UDP 字符串指令：
    机械车协议: forward / backward / left / right / stop / open / close
    智能车协议: 1 / 2 / 3 / 4 / 5 / 6 / 7
通过 unitree_sdk2py 直接控制 Go2 运动。
"""

import os
import socket
import time
import threading
import logging

# ================== 可配置参数 ==================
UDP_IP = "0.0.0.0"          # 监听所有网卡
UDP_PORT = 8888             # 与手机 App 中填写的端口一致

GO2_IP = os.environ.get("GO2_IP", "192.168.123.15")
NETWORK_INTERFACE = os.environ.get("NETWORK_INTERFACE", "eth0")

FORWARD_SPEED = 0.8
BACKWARD_SPEED = 0.8
LEFT_YAW = 2.0
RIGHT_YAW = -2.0

COMMAND_TIMEOUT_MS = 600    # 多久没收到新指令就自动停止
PUBLISH_PERIOD_MS = 100     # 持续向 Go2 发送运动指令的周期
# ===============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.sport.sport_client import SportClient
except ImportError as e:
    logger.error(f"导入 unitree_sdk2py 失败: {e}")
    logger.error("请确认已 source 宇树 SDK 环境，或已安装 unitree_sdk2_python")
    raise


class Go2Robot:
    def __init__(self):
        self.client = SportClient()
        self.client.SetTimeout(3.0)
        self.client.Init()
        logger.info("Go2 SportClient 初始化成功")

    def stop(self):
        try:
            self.client.StopMove()
        except Exception as e:
            logger.warning(f"停止指令异常: {e}")

    def forward(self):
        self.client.Move(FORWARD_SPEED, 0, 0)

    def backward(self):
        self.client.Move(-BACKWARD_SPEED, 0, 0)

    def left(self):
        self.client.Move(0, 0, LEFT_YAW)

    def right(self):
        self.client.Move(0, 0, RIGHT_YAW)


# UDP 套接字
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

# 全局状态
current_cmd = "stop"
last_cmd_time = time.time()
running = True
robot = None

# 兼容手机 App 两种协议
CMD_ALIASES = {
    "forward": "forward",
    "backward": "backward",
    "left": "left",
    "right": "right",
    "stop": "stop",
    "open": "open",
    "close": "close",
    "1": "forward",
    "2": "backward",
    "3": "left",
    "4": "right",
    "5": "stop",
    "6": "open",
    "7": "close",
}


def apply_command(cmd):
    """更新当前目标指令和最近接收时间"""
    global current_cmd, last_cmd_time
    current_cmd = cmd
    last_cmd_time = time.time()
    logger.info(f"当前动作: {cmd}")


def udp_receiver():
    """后台线程：接收手机 UDP 指令"""
    global running
    logger.info(f"UDP 监听已开启: {UDP_IP}:{UDP_PORT}")
    while running:
        try:
            data, addr = sock.recvfrom(1024)
            raw = data.decode("utf-8").strip()
            logger.info(f"收到指令: {raw} 来自 {addr}")
            alias = CMD_ALIASES.get(raw)
            if alias:
                apply_command(alias)
            else:
                logger.warning(f"未知指令: {raw}")
        except socket.error:
            time.sleep(0.001)
        except Exception as e:
            logger.warning(f"UDP 接收异常: {e}")


def control_loop():
    """主循环：持续向 Go2 发布速度指令"""
    global running, current_cmd, robot

    logger.info("开始控制循环，Go2 应保持已站立状态")

    while running:
        # 超时保护：太久没收到指令就停止
        if time.time() - last_cmd_time > COMMAND_TIMEOUT_MS / 1000.0:
            if current_cmd != "stop":
                logger.info("指令超时，自动停止")
                current_cmd = "stop"

        try:
            if current_cmd == "forward":
                robot.forward()
            elif current_cmd == "backward":
                robot.backward()
            elif current_cmd == "left":
                robot.left()
            elif current_cmd == "right":
                robot.right()
            elif current_cmd in ("open", "close"):
                # Go2 自身无夹爪，收到爪子指令时只记录并停止运动
                logger.info(f"Go2 不支持夹爪指令 '{current_cmd}'，执行停止")
                robot.stop()
            else:
                robot.stop()
        except Exception as e:
            logger.error(f"运动指令异常: {e}")

        time.sleep(PUBLISH_PERIOD_MS / 1000.0)

    # 退出前停止
    logger.info("停止控制...")
    try:
        robot.stop()
        time.sleep(0.5)
    except Exception as e:
        logger.warning(f"退出停止异常: {e}")


def main():
    global robot, running

    # 打印本机网卡信息，方便排查 Cyclone DDS 接口匹配问题
    try:
        import subprocess
        result = subprocess.run(["ip", "addr", "show"], capture_output=True, text=True)
        logger.info("本机网卡信息:\n" + result.stdout)
    except Exception as e:
        logger.warning(f"无法打印网卡信息: {e}")

    # Cyclone DDS 初始化：
    # 优先使用 NETWORK_INTERFACE（默认 eth0）指定的网卡，避免多网卡时选错接口。
    # 如需指定 Go2 IP，可设置环境变量 GO2_IP。
    if NETWORK_INTERFACE:
        dds_config = (
            f"<CycloneDDS><Domain><General>"
            f"<NetworkInterfaceAddress>{NETWORK_INTERFACE}</NetworkInterfaceAddress>"
            f"</General></Domain></CycloneDDS>"
        )
        os.environ["CYCLONEDDS_URI"] = dds_config
        logger.info(f"设置 Cyclone DDS 使用网卡: {NETWORK_INTERFACE}")

    explicit_go2_ip = os.environ.get("GO2_IP")
    if explicit_go2_ip:
        logger.info(f"初始化 Cyclone DDS，指定 Go2: {explicit_go2_ip}")
        ChannelFactoryInitialize(0, explicit_go2_ip)
    else:
        logger.info("初始化 Cyclone DDS，自动选择网卡")
        ChannelFactoryInitialize(0)

    robot = Go2Robot()

    # 启动 UDP 接收线程
    udp_thread = threading.Thread(target=udp_receiver, daemon=True)
    udp_thread.start()

    # 主线程运行控制循环
    try:
        control_loop()
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C")
    finally:
        running = False
        udp_thread.join(timeout=1.0)
        sock.close()


if __name__ == "__main__":
    main()
