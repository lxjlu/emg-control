#!/usr/bin/env python3
"""
udp_bridge.py
树莓派 UDP 中转控制：支持 Go2 机械狗 + 侦察球

手机(App) --WiFi热点(192.168.10.1)--> 树莓派 --网线(192.168.123.18)--> Go2
                                       树莓派 --WiFi/LAN--> 侦察球(192.168.1.11:3020)

自动识别两种协议：
  1. 字符串命令（Go2）: forward / backward / left / right / stop / 1~7
  2. 二进制命令（侦察球）: 11 字节包，以 0x01 0x81 开头，直接转发到机械球

环境变量：
  GO2_AVOID=0          关闭 Go2 避障
  BALL_IP=192.168.1.11 侦察球目标 IP
  BALL_PORT=3020       侦察球目标端口
"""

import os
import socket
import time
import threading
import logging
import struct

os.environ["CYCLONEDDS_NETWORK_INTERFACE"] = os.environ.get("NETWORK_INTERFACE", "eth0")

# ================== 可配置参数 ==================
UDP_IP = "0.0.0.0"
UDP_PORT = 8888

GO2_IP = os.environ.get("GO2_IP", "192.168.123.161")
NETWORK_INTERFACE = os.environ.get("NETWORK_INTERFACE", "eth0")
AVOIDANCE_ENABLED = os.environ.get("GO2_AVOID", "1") != "0"

# 侦察球参数
BALL_IP = os.environ.get("BALL_IP", "192.168.1.11")
BALL_PORT = int(os.environ.get("BALL_PORT", "3020"))
BALL_MAGIC = b"\x01\x81"  # 侦察球包前两字节

FORWARD_SPEED = 1.0
BACKWARD_SPEED = 0.8
LEFT_YAW = 1.1
RIGHT_YAW = -1.1

COMMAND_TIMEOUT_MS = 300
PUBLISH_PERIOD_MS = 100
# ===============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ================== Go2 SDK（可选） ==================
go2_sdk_ok = False
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import (
        ObstaclesAvoidClient,
    )
    go2_sdk_ok = True
except ImportError as e:
    logger.warning(f"unitree_sdk2py 未安装，Go2 控制不可用: {e}")
    logger.warning("侦察球转发功能仍可正常使用")


class Go2Robot:
    """封装 Go2 运动 + 板载避障。"""

    def __init__(self, avoidance: bool = True):
        self.avoidance = avoidance
        self._api_control_active = False
        self._moving = False
        self._send_lock = threading.Lock()

        self.client = SportClient()
        self.client.SetTimeout(3.0)
        self.client.Init()

        self.avoid_client = None
        if self.avoidance:
            self.avoid_client = ObstaclesAvoidClient()
            self.avoid_client.SetTimeout(3.0)
            self.avoid_client.Init()
            code = self.avoid_client.SwitchSet(True)
            if code != 0:
                raise RuntimeError(f"开启 Go2 避障失败，错误码 {code}")
            logger.info("Go2 原生避障已开启")
        else:
            logger.info("避障已通过 GO2_AVOID=0 关闭")
        logger.info("Go2 SportClient 初始化成功")

    def _set_api_control(self, enabled: bool):
        if self.avoid_client is None or self._api_control_active == enabled:
            return
        code = self.avoid_client.UseRemoteCommandFromApi(enabled)
        if code != 0:
            action = "接管" if enabled else "释放"
            raise RuntimeError(f"避障服务{action}速度命令失败，错误码 {code}")
        self._api_control_active = enabled
        logger.info("Go2 避障控制权已%s", "接管" if enabled else "释放")

    def move(self, vx, vy, vyaw):
        with self._send_lock:
            if self.avoid_client is None:
                self.client.Move(vx, vy, vyaw)
            else:
                self._set_api_control(True)
                self.avoid_client.Move(vx, vy, vyaw)
            self._moving = True

    def stop(self):
        with self._send_lock:
            try:
                if self.avoid_client is not None and self._api_control_active:
                    self.avoid_client.Move(0.0, 0.0, 0.0)
                if self._moving:
                    self.client.StopMove()
                if self.avoid_client is not None and self._api_control_active:
                    self._set_api_control(False)
            except Exception as e:
                logger.warning(f"停止指令异常: {e}")
            finally:
                self._moving = False

    def forward(self):  self.move(FORWARD_SPEED, 0, 0)
    def backward(self): self.move(-BACKWARD_SPEED, 0, 0)
    def left(self):     self.move(0, 0, LEFT_YAW)
    def right(self):    self.move(0, 0, RIGHT_YAW)


# ================== 侦察球转发器 ==================
class BallForwarder:
    """将收到的机械球二进制包直接转发到机械球。"""

    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.connect((ip, port))
        logger.info(f"侦察球转发器就绪: -> {ip}:{port}")

    def forward(self, data):
        """转发原始二进制包。"""
        self.sock.send(data)

    def close(self):
        self.sock.close()

    @staticmethod
    def is_ball_packet(data):
        """检测是否是侦察球包：11 字节且前两字节为 0x01 0x81。"""
        return len(data) == 11 and data[:2] == BALL_MAGIC


# ================== UDP 接收 ==================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

# 全局状态
current_cmd = "stop"
last_cmd_time = time.time()
running = True
robot = None        # Go2 实例
ball_fwd = None     # 侦察球转发器实例

CMD_ALIASES = {
    "forward": "forward", "backward": "backward",
    "left": "left", "right": "right", "stop": "stop",
    "open": "open", "close": "close",
    "1": "forward", "2": "backward", "3": "left",
    "4": "right", "5": "stop", "6": "open", "7": "close",
}


def apply_command(cmd):
    global current_cmd, last_cmd_time
    current_cmd = cmd
    last_cmd_time = time.time()
    logger.info(f"Go2 当前动作: {cmd}")


def udp_receiver():
    """后台线程：接收手机 UDP 指令，自动区分机械球二进制包和 Go2 字符串命令。"""
    global running
    logger.info(f"UDP 监听已开启: {UDP_IP}:{UDP_PORT}")
    while running:
        try:
            data, addr = sock.recvfrom(1024)

            # 优先检测侦察球二进制包
            if BallForwarder.is_ball_packet(data):
                if ball_fwd:
                    ball_fwd.forward(data)
                    control = data[2]
                    steer = struct.unpack_from("<h", data, 3)[0]
                    drive = struct.unpack_from("<h", data, 5)[0]
                    ctrl_name = {0x00: "正常", 0x20: "锁定", 0x40: "解锁"}.get(control, f"0x{control:02X}")
                    logger.info(f"侦察球转发: {ctrl_name} steer={steer} drive={drive} -> {ball_fwd.ip}:{ball_fwd.port}")
                else:
                    logger.warning("收到侦察球包但转发器未初始化")
                continue

            # Go2 字符串命令
            raw = data.decode("utf-8").strip()
            logger.info(f"收到指令: {raw} 来自 {addr}")
            alias = CMD_ALIASES.get(raw)
            if alias:
                apply_command(alias)
            else:
                logger.warning(f"未知指令: {raw}")
        except socket.error:
            time.sleep(0.001)
        except UnicodeDecodeError:
            # 非二进制球包、非 UTF-8 字符串，忽略
            logger.warning(f"收到无法识别的数据包: {data.hex()}")
        except Exception as e:
            logger.warning(f"UDP 接收异常: {e}")


def control_loop():
    """主循环：持续向 Go2 发布速度指令（经过避障服务）。侦察球由 App 心跳直接驱动。"""
    global running, current_cmd, robot

    if robot:
        logger.info("开始 Go2 控制循环")
    else:
        logger.info("Go2 未初始化，仅侦察球转发模式")

    while running:
        if robot:
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
                    logger.info(f"Go2 不支持夹爪指令 '{current_cmd}'，执行停止")
                    robot.stop()
                else:
                    robot.stop()
            except Exception as e:
                logger.error(f"运动指令异常: {e}")

        time.sleep(PUBLISH_PERIOD_MS / 1000.0)

    if robot:
        logger.info("停止 Go2 控制...")
        try:
            robot.stop()
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"退出停止异常: {e}")


def main():
    global robot, ball_fwd, running

    try:
        import subprocess
        result = subprocess.run(["ip", "addr", "show"], capture_output=True, text=True)
        logger.info("本机网卡信息:\n" + result.stdout)
    except Exception as e:
        logger.warning(f"无法打印网卡信息: {e}")

    # 初始化侦察球转发器（始终可用）
    ball_fwd = BallForwarder(BALL_IP, BALL_PORT)

    # 初始化 Go2（SDK 可用时）
    if go2_sdk_ok:
        try:
            if NETWORK_INTERFACE:
                dds_config = (
                    f"<CycloneDDS><Domain><General>"
                    f"<NetworkInterfaceAddress>{NETWORK_INTERFACE}</NetworkInterfaceAddress>"
                    f"</General></Domain></CycloneDDS>"
                )
                os.environ["CYCLONEDDS_URI"] = dds_config
                logger.info(f"设置 Cyclone DDS 使用网卡: {NETWORK_INTERFACE}")

            if GO2_IP:
                logger.info(f"初始化 Cyclone DDS，指定 Go2: {GO2_IP}")
                ChannelFactoryInitialize(0, GO2_IP)
            else:
                logger.info("初始化 Cyclone DDS，自动选择网卡")
                ChannelFactoryInitialize(0)

            robot = Go2Robot(avoidance=AVOIDANCE_ENABLED)
        except Exception as e:
            logger.error(f"Go2 初始化失败: {e}")
            logger.warning("Go2 控制不可用，仅侦察球转发模式")
            robot = None
    else:
        logger.warning("Go2 SDK 未安装，仅侦察球转发模式")
        robot = None

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
        if ball_fwd:
            ball_fwd.close()


if __name__ == "__main__":
    main()
