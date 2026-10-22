# -*- coding: utf-8 -*-
"""适配器基类：把语义命令翻译成具体设备调用。"""


class DeviceAdapter:
    #: 设备标识，与 App 里的设备类型对应
    name = "base"
    #: 显示名
    display = "设备"
    #: 是否需要按 publish_hz 持续重发（连续运动类为 True，按键类为 False）
    continuous = True
    #: 心跳下发频率
    publish_hz = 20.0
    #: 多久没收到新命令就自动停止
    deadman_ms = 500

    def __init__(self, dry_run=False, log=print):
        self.dry_run = dry_run
        self.log = log
        self._last_key = None

    # ------------------------------------------------------------------
    def capabilities(self):
        """上报能力，供 App 决定手势映射与速度范围。"""
        return {
            "type": self.name,
            "name": self.display,
            "actions": ["move", "stop"],
            "continuous": self.continuous,
            "publish_hz": self.publish_hz,
            "deadman_ms": self.deadman_ms,
        }

    def start(self):
        """建立连接 / 初始化 SDK。"""

    def apply(self, action, args):
        """执行一个语义动作。"""
        raise NotImplementedError

    def stop(self):
        """立即停止。"""

    def close(self):
        """释放资源。"""

    def handle_raw(self, data):
        """处理旧版设备的原始二进制包，返回是否已处理。"""
        return False

    # ------------------------------------------------------------------
    def once(self, message, **fields):
        """dry-run 或调试用：同一个状态只打印一次，避免 20Hz 刷屏。"""
        key = (message, tuple(sorted((str(k), str(v)) for k, v in fields.items())))
        if key == self._last_key:
            return
        self._last_key = key
        detail = " ".join(f"{k}={v}" for k, v in fields.items())
        self.log(f"[{self.name}] {message} {detail}".rstrip())
