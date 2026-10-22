# -*- coding: utf-8 -*-
"""机械车适配器：把语义命令翻译成小车 ROS 节点认识的字符串协议。

沿用现有部署：树莓派上的 SDK 把 forward / backward / left / right / stop /
open / close 转发到小车的 8888 端口（小车上跑 udp_car_control.py 发 /cmd_vel）。
目标地址可用 --target 改成小车实际 IP。
"""

import socket

from ..protocol import ACTION_MOVE, clamp
from .base import DeviceAdapter


class MechanicalCarAdapter(DeviceAdapter):
    name = "car"
    display = "机械车（ROS 小车）"
    continuous = True
    publish_hz = 20.0
    deadman_ms = 500

    def __init__(self, target="192.168.2.10:8888", dry_run=False, log=print):
        super().__init__(dry_run=dry_run, log=log)
        host, _, port = target.partition(":")
        self.target_host = host or "192.168.2.10"
        self.target_port = int(port or 8888)
        self._sock = None

    # ------------------------------------------------------------------
    def capabilities(self):
        caps = super().capabilities()
        caps["actions"] = ["move", "stop", "open", "close"]
        caps["target"] = f"{self.target_host}:{self.target_port}"
        return caps

    def start(self):
        if self.dry_run:
            self.log(f"[dry-run] {self.display} 就绪（目标 {self.target_host}:{self.target_port}）")
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.log(f"机械车目标：{self.target_host}:{self.target_port}")

    # ------------------------------------------------------------------
    def apply(self, action, args):
        if action == ACTION_MOVE:
            vx = clamp(args.get("vx", 0.0))
            wz = clamp(args.get("wz", 0.0))
            if vx > 0.15:
                text = "forward"
            elif vx < -0.15:
                text = "backward"
            elif wz > 0.15:
                text = "left"
            elif wz < -0.15:
                text = "right"
            else:
                text = "stop"
        elif action in ("open", "close"):
            text = action
        else:
            return

        if self.dry_run:
            self.once("发送", cmd=text)
            return
        self._send(text)

    def stop(self):
        if self.dry_run:
            self.once("发送", cmd="stop")
            return
        self._send("stop")

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    # ------------------------------------------------------------------
    def _send(self, text):
        try:
            self._sock.sendto(text.encode("utf-8"),
                              (self.target_host, self.target_port))
        except OSError as error:
            self.log(f"发送失败: {error}")
