# -*- coding: utf-8 -*-
"""UDP 服务端：接收语义命令 → 心跳重发 → 交给设备适配器。

设计要点：
* 连续量（move）按适配器声明的频率重发，超过 deadman_ms 没收到新命令就自动停
* 停止只发一次，不反复刷 StopMove
* 离散量（next / prev / start / exit / open / close）收到就执行一次，不参与心跳
* 旧 App 的字符串命令和侦察球 11 字节二进制包都能直接吃，方便灰度
"""

import json
import select
import socket
import time

from .protocol import (ACTION_HELLO, ACTION_MOVE, ACTION_STOP, DISCRETE_ACTIONS,
                       PROTOCOL_VERSION, STREAM_TYPE, MessageError, parse)
from .stream import decode_stream


class CommandServer:
    def __init__(self, adapter, host="0.0.0.0", port=8888,
                 token=None, ack=True, monitor=None, log=print):
        self.adapter = adapter
        self.host = host
        self.port = port
        self.token = token
        self.ack = ack
        self.monitor = monitor
        self.log = log

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.setblocking(False)

        self._action = ACTION_STOP
        self._args = {}
        self._last_rx = time.monotonic()
        self._last_sent = None
        self._last_seq = None
        self._last_logged_motion = (None, None)
        self._raw_mode = False          # 收到旧版侦察球包时，交给 App 自己驱动心跳
        self._stream_warned = False
        self._running = True

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def serve_forever(self):
        adapter = self.adapter
        adapter.start()
        self.log(f"=== {adapter.display} 已就绪 ===")
        self.log(f"UDP 监听 {self.host}:{self.port}  "
                 f"类型={adapter.name}  心跳={adapter.publish_hz:g}Hz  "
                 f"超时停={adapter.deadman_ms}ms"
                 + (f"  token={'已启用' if self.token else '关闭'}"))

        period = 1.0 / max(1.0, float(adapter.publish_hz))
        next_tick = time.monotonic()

        try:
            if self.monitor is not None:
                self.monitor.start()
            while self._running:
                timeout = max(0.0, next_tick - time.monotonic())
                readable, _, _ = select.select([self._sock], [], [], min(timeout, 0.02))
                if readable:
                    self._drain()

                now = time.monotonic()
                if now >= next_tick:
                    next_tick = now + period
                    self._publish()
        except KeyboardInterrupt:
            self.log("收到 Ctrl+C")
        finally:
            self._running = False
            try:
                adapter.stop()
            except Exception as error:
                self.log(f"退出停止异常: {error}")
            try:
                adapter.close()
            except Exception as error:
                self.log(f"关闭适配器异常: {error}")
            self._sock.close()
            if self.monitor is not None:
                self.monitor.stop()
            self.log("已退出")

    def _drain(self):
        for _ in range(50):
            try:
                data, addr = self._sock.recvfrom(8192)
            except BlockingIOError:
                return
            except ConnectionResetError:
                continue
            except OSError:
                return
            self._handle(data, addr)

    # ------------------------------------------------------------------
    # 报文处理
    # ------------------------------------------------------------------
    def _handle(self, data, addr):
        try:
            message = parse(data)
        except MessageError as error:
            self.log(f"忽略无效数据（{len(data)} 字节）: {error}")
            self._reply(addr, {"ok": False, "error": str(error)})
            return

        if self.token and message["token"] != self.token:
            self.log(f"忽略 token 不匹配的报文 来自 {addr[0]}:{addr[1]}")
            return

        if message["kind"] == STREAM_TYPE:
            self._handle_stream(message)
            return

        if message["kind"] == "raw":
            if self.adapter.handle_raw(data):
                self._raw_mode = True
                self._last_rx = time.monotonic()
            else:
                self.log("收到侦察球二进制包，但当前适配器不支持")
            return

        action = message["action"]

        if action == ACTION_HELLO:
            self._reply(addr, self._hello())
            return

        seq = message["seq"]
        if seq is not None:
            if seq == self._last_seq:
                return                      # 重复包
            self._last_seq = seq

        self._raw_mode = False

        if action == ACTION_MOVE:
            args = {
                "vx": float(message["arg"].get("vx", 0.0)),
                "wz": float(message["arg"].get("wz", 0.0)),
            }
            self._action = ACTION_MOVE
            self._args = args
            self._last_rx = time.monotonic()
            key = (round(args["vx"], 2), round(args["wz"], 2))
            if key != self._last_logged_motion:
                self._last_logged_motion = key
                self.log(f"运动 vx={args['vx']:+.2f} wz={args['wz']:+.2f}  来自 {addr[0]}")
            if not self.adapter.continuous:
                self.adapter.apply(ACTION_MOVE, args)
        elif action == ACTION_STOP:
            self._action = ACTION_STOP
            self._args = {}
            self._last_rx = time.monotonic()
            self._last_logged_motion = (0.0, 0.0)
            self.adapter.stop()
            self._last_sent = ACTION_STOP
            self.log(f"停止  来自 {addr[0]}")
        elif action in DISCRETE_ACTIONS:
            self.adapter.apply(action, message["arg"])
            self.log(f"动作 {action}  来自 {addr[0]}")
        else:
            self.log(f"未知动作: {action}")
            self._reply(addr, {"ok": False, "error": f"未知动作: {action}"})
            return

        self._reply(addr, {"ok": True, "act": action})

    def _handle_stream(self, message):
        if self.monitor is None:
            if not self._stream_warned:
                self._stream_warned = True
                self.log("收到肌电流，但未启用监控（启动时加 --monitor 8080 可查看波形）")
            return
        try:
            samples, rate, channels = decode_stream(message["body"])
        except MessageError as error:
            self.log(f"忽略无效肌电流: {error}")
            return
        self.monitor.push(samples, rate, channels)

    def _publish(self):
        adapter = self.adapter
        if not adapter.continuous or self._raw_mode:
            return

        if self._action == ACTION_MOVE:
            if time.monotonic() - self._last_rx > adapter.deadman_ms / 1000.0:
                self._action = ACTION_STOP
                self._args = {}
                adapter.stop()
                self._last_sent = ACTION_STOP
                self._last_logged_motion = (0.0, 0.0)
                self.log("指令超时，自动停止")
                return
            adapter.apply(ACTION_MOVE, self._args)
            self._last_sent = ACTION_MOVE
        elif self._last_sent != ACTION_STOP:
            adapter.stop()
            self._last_sent = ACTION_STOP

    def _hello(self):
        capabilities = dict(self.adapter.capabilities())
        capabilities.update({"v": PROTOCOL_VERSION, "ok": True, "port": self.port,
                             "token_required": bool(self.token)})
        return capabilities

    def _reply(self, addr, payload):
        if not self.ack and payload.get("ok") is not False:
            return
        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._sock.sendto(data, addr)
        except OSError:
            pass
