# -*- coding: utf-8 -*-
"""侦察球适配器（11 字节 UDP 二进制协议）。

帧格式（小端）：
    [0:2]  01 81
    [2]    控制字 0x00 正常 / 0x20 锁定 / 0x40 解锁
    [3:5]  steer int16   （左为负、右为正）
    [5:7]  drive int16   （前进为正）
    [7]    servo uint8
    [8]    turn_speed uint8
    [9:11] CRC16/MAXIM（poly A001, init 0000, xor FFFF，大端）

球固件要求源地址为 192.168.1.50:3020，且单包 lease 约 0.4 秒，
所以必须 20Hz 持续重发——这正好由 server 的心跳统一处理。
"""

import socket
import struct
import time

from ..protocol import ACTION_MOVE, clamp
from .base import DeviceAdapter

BALL_ID = b"\x01\x81"
CONTROL_NORMAL = 0x00
CONTROL_LOCK = 0x20
CONTROL_UNLOCK = 0x40


def crc16_maxim(data):
    """CRC-16/MAXIM-DOW：poly 8005，初值 0000，反射输入输出，末异或 FFFF。"""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xA001 if crc & 1 else 0)
    return crc ^ 0xFFFF


def encode_packet(steer, drive, control=CONTROL_NORMAL, servo=0, turn_speed=0x12):
    steer = max(-1000, min(1000, int(steer)))
    drive = max(-1000, min(1000, int(drive)))
    payload = BALL_ID + bytes((control,)) + struct.pack("<hhBB", steer, drive, servo, turn_speed)
    return payload + crc16_maxim(payload).to_bytes(2, "big")


class BallAdapter(DeviceAdapter):
    name = "ball"
    display = "侦察球"
    continuous = True
    publish_hz = 20.0          # 固件 lease 约 0.4s，20Hz 足够稳
    deadman_ms = 400

    def __init__(self, source_ip="192.168.1.50", source_port=3020,
                 ball_ip="192.168.1.11", ball_port=3020, speed=1000,
                 servo=0, turn_speed=0x12, dry_run=False, log=print):
        super().__init__(dry_run=dry_run, log=log)
        self.source_ip = source_ip
        self.source_port = int(source_port)
        self.ball_ip = ball_ip
        self.ball_port = int(ball_port)
        self.speed = max(1, min(1000, int(speed)))
        self.servo = int(servo)
        self.turn_speed = int(turn_speed)
        self._sock = None
        self._last_axes = (0, 0)

    # ------------------------------------------------------------------
    def capabilities(self):
        caps = super().capabilities()
        caps["actions"] = ["move", "stop"]
        caps["max_speed"] = {"drive": self.speed, "steer": self.speed}
        caps["route"] = f"{self.source_ip}:{self.source_port} -> {self.ball_ip}:{self.ball_port}"
        return caps

    def start(self):
        if self.dry_run:
            self.log(f"[dry-run] {self.display} 就绪（{self.ball_ip}:{self.ball_port}）")
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((self.source_ip, self.source_port))
            sock.connect((self.ball_ip, self.ball_port))
        except OSError as error:
            sock.close()
            raise RuntimeError(
                f"绑定 {self.source_ip}:{self.source_port} 失败: {error}\n"
                "请确认已配置 eth0:1：sudo ifconfig eth0:1 192.168.1.50 netmask 255.255.255.0 up"
            ) from error
        self._sock = sock
        # 解锁（默认上电是锁定状态），再补一帧零速
        for _ in range(3):
            self._send_bytes(encode_packet(0, 0, CONTROL_UNLOCK, self.servo, self.turn_speed))
            time.sleep(0.02)
        self._send_bytes(encode_packet(0, 0, CONTROL_NORMAL, self.servo, self.turn_speed))
        self.log(f"球通道就绪：{self.source_ip}:{self.source_port} -> "
                 f"{self.ball_ip}:{self.ball_port}（已解锁）")

    # ------------------------------------------------------------------
    def apply(self, action, args):
        if action != ACTION_MOVE:
            return
        drive = int(round(clamp(args.get("vx", 0.0)) * self.speed))
        steer = int(round(-clamp(args.get("wz", 0.0)) * self.speed))   # 协议里左为负
        axes = (steer, drive)
        if self.dry_run:
            self.once("set_axes", steer=steer, drive=drive, hex=self._hex(steer, drive))
            self._last_axes = axes
            return
        self._send_bytes(encode_packet(steer, drive, CONTROL_NORMAL,
                                       self.servo, self.turn_speed))
        self._last_axes = axes

    def stop(self):
        if self.dry_run:
            if self._last_axes != (0, 0):
                self.once("set_axes", steer=0, drive=0, hex=self._hex(0, 0))
                self._last_axes = (0, 0)
            return
        self._send_bytes(encode_packet(0, 0, CONTROL_NORMAL, self.servo, self.turn_speed))
        self._last_axes = (0, 0)

    def handle_raw(self, data):
        """旧 App 直接发 11 字节球包时，原样转发，保持线路一字节不变。"""
        if len(data) != 11 or data[:2] != BALL_ID:
            return False
        if self.dry_run:
            self.once("转发原始球包", hex=data.hex(" ").upper())
            return True
        self._send_bytes(data)
        return True

    def close(self):
        if self.dry_run or self._sock is None:
            return
        try:
            for _ in range(3):
                self._send_bytes(encode_packet(0, 0, CONTROL_LOCK, self.servo, self.turn_speed))
        except OSError:
            pass
        finally:
            self._sock.close()
            self._sock = None

    # ------------------------------------------------------------------
    def _send_bytes(self, packet):
        if self._sock is not None:
            self._sock.send(packet)

    def _hex(self, steer, drive):
        return encode_packet(steer, drive, CONTROL_NORMAL, self.servo, self.turn_speed).hex(" ").upper()
