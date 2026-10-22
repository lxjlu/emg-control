#!/usr/bin/env python3
"""
udp_go2_bridge_v2.py
树莓派 UDP 中转：手机 App -> 宇树 Go2 系列（机械狗 / 轮式 Go2-W）

架构：
    手机(App) --WiFi热点--> 树莓派(本脚本) --eth0--> Go2 机械狗 / Go2 轮式(Go2-W)

App 侧发送的 UDP 字符串指令（由 App 里选的设备类型决定前缀）：
    机械狗:  forward / backward / left / right / stop
    轮式  :  go2w:forward / go2w:backward / go2w:left / go2w:right / go2w:stop

本脚本按前缀自动选择对应的控制流程：
    无前缀 -> Go2 机械狗：直接 Move()
    go2w:  -> Go2-W    ：先 BalanceStand()（等效于在 Go2w_sport_client 里按 9，进入轮式平衡站立），再 Move()

备注：两种设备在 unitree_sdk2py 里用的是同一个 SportClient 类（C++ 的 go2w 示例同样 include
      unitree/robot/go2/sport/sport_client.hpp），差异只在初始化流程与状态话题
      （Go2 为 rt/sportmodestate，Go2-W 为 rt/lf/sportmodestate），因此一个进程即可同时服务两者。

用法：
    python3 udp_go2_bridge_v2.py                    # 自动按前缀识别
    python3 udp_go2_bridge_v2.py eth0               # 指定网卡（沿用 Go2w_sport_client eth0 的习惯）
    NETWORK_INTERFACE=eth0 python3 udp_go2_bridge_v2.py
    DEFAULT_MODE=go2w python3 udp_go2_bridge_v2.py  # 强制轮式流程（不带前缀也能控）
"""

import logging
import os
import socket
import sys
import threading
import time

# ================== 可配置参数 ==================
UDP_IP = "0.0.0.0"                      # 监听所有网卡
UDP_PORT = 8888                         # 与手机 App 中填写的端口一致

NETWORK_INTERFACE = os.environ.get("NETWORK_INTERFACE", "eth0")
GO2_IP = os.environ.get("GO2_IP", "")   # 可选：显式指定机器人 IP
DEFAULT_MODE = os.environ.get("DEFAULT_MODE", "").strip().lower()  # "" | "go2" | "go2w"

MODE_GO2 = "go2"                        # Go2 机械狗
MODE_GO2W = "go2w"                      # Go2 轮式

# 运动速度参数（按设备类型分开，可自行调整）
MOTION_PARAMS = {
    MODE_GO2: {
        "forward":  (0.8, 0.0, 0.0),
        "backward": (-0.8, 0.0, 0.0),
        "left":     (0.0, 0.0, 2.0),
        "right":    (0.0, 0.0, -2.0),
    },
    MODE_GO2W: {
        "forward":  (0.5, 0.0, 0.0),
        "backward": (-0.5, 0.0, 0.0),
        "left":     (0.0, 0.0, 1.0),
        "right":    (0.0, 0.0, -1.0),
    },
}

COMMAND_TIMEOUT_MS = 600        # 多久没收到新指令就自动停止
PUBLISH_PERIOD_MS = 100         # 持续下发运动指令的周期
BALANCE_SETTLE_S = 1.5          # BalanceStand 后等待机器人站稳的时间
# ===============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 兼容手机 App 智能车协议的数字指令
CMD_ALIASES = {
    "1": "forward",
    "2": "backward",
    "3": "left",
    "4": "right",
    "5": "stop",
    "6": "stop",   # Go2 无夹爪，开爪退化为停止
    "7": "stop",
}
VALID_CMDS = {"forward", "backward", "left", "right", "stop"}

# UDP 套接字
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 全局状态
state_lock = threading.Lock()
pending_mode = None
active_mode = None
current_cmd = "stop"
last_cmd_time = time.time()
running = True
robot = None


class RobotClient:
    """运动客户端封装。

    两种设备共用 unitree_sdk2py 的 SportClient，区别在于 Go2-W 必须先 BalanceStand
    （对应 Go2w_sport_client.cpp 菜单里的 id 9），否则 Move() 不会让轮足动起来。
    """

    def __init__(self):
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        self.client = SportClient()
        self.client.SetTimeout(3.0)
        self.client.Init()
        logger.info("SportClient 初始化成功")

    def prepare(self, mode):
        """进入某设备类型前的一次性初始化（App 启用控制时会先发一条 stop，正好触发这里）。"""
        if mode == MODE_GO2W:
            logger.info("Go2-W：执行 BalanceStand（等效按 9），进入轮式平衡站立")
            self.client.BalanceStand()
            time.sleep(BALANCE_SETTLE_S)
            logger.info("Go2-W：平衡站立完成，可以开始运动")
        else:
            logger.info("Go2 机械狗：无需额外初始化（机器人应已处于站立状态）")

    def stop(self):
        self.client.StopMove()

    def apply(self, mode, cmd):
        if cmd == "stop":
            return self.stop()
        params = MOTION_PARAMS.get(mode, {}).get(cmd)
        if params is None:
            return self.stop()
        return self.client.Move(*params)


def parse_command(raw):
    """把 App 指令解析为 (设备类型, 动作)。返回的动作无效时为 None。"""
    mode = DEFAULT_MODE or MODE_GO2
    cmd = raw
    if raw.startswith("go2w:"):
        mode, cmd = MODE_GO2W, raw[len("go2w:"):]
    elif raw.startswith("go2:"):
        mode, cmd = MODE_GO2, raw[len("go2:"):]
    cmd = CMD_ALIASES.get(cmd, cmd)
    if cmd not in VALID_CMDS:
        return mode, None
    return mode, cmd


def udp_receiver():
    """后台线程：接收手机 App 的 UDP 指令。"""
    global pending_mode, current_cmd, last_cmd_time
    logger.info(f"UDP 监听已开启: {UDP_IP}:{UDP_PORT}")
    while running:
        try:
            data, addr = sock.recvfrom(1024)
            raw = data.decode("utf-8", errors="ignore").strip()
            if not raw:
                continue
            mode, cmd = parse_command(raw)
            if cmd is None:
                logger.warning(f"未知指令: {raw!r} 来自 {addr}")
                continue
            with state_lock:
                pending_mode = mode
                current_cmd = cmd
                last_cmd_time = time.time()
            logger.info(f"收到指令: {raw} -> 设备={mode} 动作={cmd} 来自 {addr[0]}:{addr[1]}")
        except socket.error:
            time.sleep(0.001)
        except Exception as e:
            logger.warning(f"UDP 接收异常: {e}")


def control_loop():
    """主循环：按当前设备类型持续下发运动指令。"""
    global running, current_cmd, active_mode

    logger.info("开始控制循环，等待 App 指令...")

    while running:
        with state_lock:
            mode = pending_mode or DEFAULT_MODE or MODE_GO2
            cmd = current_cmd
            age = time.time() - last_cmd_time

        # 超时保护：太久没收到指令就停止
        if age > COMMAND_TIMEOUT_MS / 1000.0 and cmd != "stop":
            logger.info("指令超时，自动停止")
            cmd = "stop"
            with state_lock:
                current_cmd = "stop"

        # 设备类型首次出现或发生切换：先停住，再执行对应的初始化流程
        if mode != active_mode:
            if active_mode is not None:
                try:
                    robot.stop()
                except Exception as e:
                    logger.warning(f"切换设备时停止异常: {e}")
            try:
                robot.prepare(mode)
                active_mode = mode
            except Exception as e:
                logger.error(f"初始化 {mode} 失败: {e}")

        try:
            robot.apply(active_mode or mode, cmd)
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
    global robot, running, NETWORK_INTERFACE

    # 支持 `python3 udp_go2_bridge_v2.py eth0` 的写法
    if len(sys.argv) > 1 and sys.argv[1].strip():
        NETWORK_INTERFACE = sys.argv[1].strip()

    # 打印本机网卡信息，方便排查 Cyclone DDS 接口匹配问题
    try:
        import subprocess
        result = subprocess.run(["ip", "addr", "show"], capture_output=True, text=True)
        logger.info("本机网卡信息:\n" + result.stdout)
    except Exception as e:
        logger.warning(f"无法打印网卡信息: {e}")

    # Cyclone DDS 初始化：指定网卡，避免多网卡时选错接口
    if NETWORK_INTERFACE:
        os.environ["CYCLONEDDS_URI"] = (
            f"<CycloneDDS><Domain><General>"
            f"<NetworkInterfaceAddress>{NETWORK_INTERFACE}</NetworkInterfaceAddress>"
            f"</General></Domain></CycloneDDS>"
        )
        logger.info(f"设置 Cyclone DDS 使用网卡: {NETWORK_INTERFACE}")

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    except ImportError as e:
        logger.error(f"导入 unitree_sdk2py 失败: {e}")
        logger.error("请确认已 source 宇树 SDK 环境，或已安装 unitree_sdk2_python")
        return 1

    if GO2_IP:
        logger.info(f"初始化 Cyclone DDS，指定机器人: {GO2_IP}")
        ChannelFactoryInitialize(0, GO2_IP)
    else:
        logger.info("初始化 Cyclone DDS，自动选择网卡")
        ChannelFactoryInitialize(0)

    robot = RobotClient()

    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False)

    logger.info("=== Go2 中转桥已启动（支持机械狗 / 轮式）===")
    logger.info("手机 App 填: IP=树莓派热点地址(如 192.168.123.1)  端口=8888")
    logger.info("设备类型选「Go2机械狗」或「Go2轮式」，脚本会按指令前缀自动分流")

    udp_thread = threading.Thread(target=udp_receiver, daemon=True)
    udp_thread.start()

    try:
        control_loop()
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C")
    finally:
        running = False
        udp_thread.join(timeout=1.0)
        sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
