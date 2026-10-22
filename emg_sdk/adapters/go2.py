# -*- coding: utf-8 -*-
"""Go2 机械狗适配器（unitree_sdk2py）。

等价于原 udp_go2_bridge_v2.py 里「无前缀」那条分支：
    forward/backward → Move(±0.8, 0, 0)
    left/right       → Move(0, 0, ±2.0)
"""

import os
import time

from ..protocol import ACTION_MOVE, clamp
from .base import DeviceAdapter


class Go2Adapter(DeviceAdapter):
    name = "go2"
    display = "Go2 机械狗"
    continuous = True
    publish_hz = 10.0          # 原脚本 PUBLISH_PERIOD_MS = 100
    deadman_ms = 600           # 原脚本 COMMAND_TIMEOUT_MS = 600

    max_vx = 0.8               # m/s，对应原 FORWARD_SPEED / BACKWARD_SPEED
    max_wz = 2.0               # rad/s，对应原 LEFT_YAW / RIGHT_YAW

    def __init__(self, interface="eth0", robot_ip=None, dry_run=False, log=print):
        super().__init__(dry_run=dry_run, log=log)
        self.interface = interface
        self.robot_ip = robot_ip
        self._client = None
        self._moving = False

    # ------------------------------------------------------------------
    def capabilities(self):
        caps = super().capabilities()
        caps["actions"] = ["move", "stop"]
        caps["max_speed"] = {"vx": self.max_vx, "wz": self.max_wz}
        return caps

    def start(self):
        if self.dry_run:
            self.log(f"[dry-run] {self.display} 适配器就绪（不会真的驱动）")
            return

        if self.interface:
            # Cyclone DDS 在多网卡时必须显式指定，否则可能选到热点网卡
            os.environ["CYCLONEDDS_URI"] = (
                "<CycloneDDS><Domain><General>"
                f"<NetworkInterfaceAddress>{self.interface}</NetworkInterfaceAddress>"
                "</General></Domain></CycloneDDS>"
            )
            self.log(f"Cyclone DDS 使用网卡: {self.interface}")

        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.sport.sport_client import SportClient

        if self.robot_ip:
            ChannelFactoryInitialize(0, self.robot_ip)
        else:
            ChannelFactoryInitialize(0)

        self._client = SportClient()
        self._client.SetTimeout(3.0)
        self._client.Init()
        self.log(f"{self.display} SportClient 初始化成功（请确认狗已站立）")

    # ------------------------------------------------------------------
    def apply(self, action, args):
        if action != ACTION_MOVE:
            return
        vx = clamp(args.get("vx", 0.0)) * self.max_vx
        wz = clamp(args.get("wz", 0.0)) * self.max_wz

        if self.dry_run:
            self.once("Move", vx=f"{vx:+.2f}", wz=f"{wz:+.2f}")
            self._moving = True
            return

        self._client.Move(vx, 0.0, wz)
        self._moving = True

    def stop(self):
        if self.dry_run:
            if self._moving:
                self.once("StopMove")
                self._moving = False
            return
        if self._client is not None:
            self._client.StopMove()
        self._moving = False

    def close(self):
        if not self.dry_run:
            try:
                self.stop()
                time.sleep(0.2)
            except Exception as error:
                self.log(f"退出停止异常: {error}")
