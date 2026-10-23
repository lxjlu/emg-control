# -*- coding: utf-8 -*-
"""统一协议 v2。

线格式（UDP，UTF-8 文本）：

    {"v":2,"seq":1024,"act":"move","arg":{"vx":0.5,"wz":0.0},"token":"xxx"}
    {"v":2,"seq":1025,"act":"stop"}
    {"v":2,"seq":1026,"act":"next"}
    {"v":2,"seq":1027,"act":"hello"}
    {"v":2,"type":"stream","n":10,"ch":8,"rate":500,"data":"..."}   # 肌电波形流

* vx / wz 归一化到 -1..1，具体量纲（m/s、rad/s、-1000..1000）由适配器决定
* seq 可选，用于丢弃重复包
* token 可选，服务端配置了 token 才校验

同时兼容旧 App 的两种写法，便于不改 App 直接验证：
* 字符串：forward / backward / left / right / stop / open / close / 1~7 / go2w:forward
* 侦察球 11 字节二进制包（01 81 开头），由适配器原样转发
"""

import json

PROTOCOL_VERSION = 2

BALL_MAGIC = b"\x01\x81"
BALL_PACKET_LEN = 11

ACTION_MOVE = "move"
ACTION_STOP = "stop"
ACTION_HELLO = "hello"
STREAM_TYPE = "stream"

# 不参与心跳、收到就执行一次的动作
DISCRETE_ACTIONS = {"next", "prev", "start", "exit", "open", "close"}

# 旧字符串命令 → 语义命令（速度已归一化）
LEGACY_ALIASES = {
    "forward": (ACTION_MOVE, {"vx": 1.0, "wz": 0.0}),
    "backward": (ACTION_MOVE, {"vx": -1.0, "wz": 0.0}),
    "left": (ACTION_MOVE, {"vx": 0.0, "wz": 1.0}),
    "right": (ACTION_MOVE, {"vx": 0.0, "wz": -1.0}),
    "stop": (ACTION_STOP, {}),
    "open": ("open", {}),
    "close": ("close", {}),
    # 智能车数字协议
    "1": (ACTION_MOVE, {"vx": 1.0, "wz": 0.0}),
    "2": (ACTION_MOVE, {"vx": -1.0, "wz": 0.0}),
    "3": (ACTION_MOVE, {"vx": 0.0, "wz": 1.0}),
    "4": (ACTION_MOVE, {"vx": 0.0, "wz": -1.0}),
    "5": (ACTION_STOP, {}),
    "6": ("open", {}),
    "7": ("close", {}),
}


class MessageError(ValueError):
    """报文无法解析或字段非法。"""


def clamp(value, low=-1.0, high=1.0):
    return max(low, min(high, float(value)))


def encode_command(action, arg=None, seq=None, token=None):
    """打包一帧语义命令。"""
    payload = {"v": PROTOCOL_VERSION, "act": action}
    if arg:
        payload["arg"] = arg
    if seq is not None:
        payload["seq"] = int(seq)
    if token:
        payload["token"] = token
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def is_ball_packet(data):
    return len(data) == BALL_PACKET_LEN and data[:2] == BALL_MAGIC


def parse_legacy(text):
    """旧字符串命令解析，返回 (action, args)。"""
    raw = text.strip()
    for prefix in ("go2w:", "go2:"):
        if raw.lower().startswith(prefix):
            raw = raw[len(prefix):]
    alias = LEGACY_ALIASES.get(raw.lower())
    if alias is None:
        raise MessageError(f"未知指令: {text!r}")
    return alias[0], dict(alias[1])


def parse(data):
    """解析一帧 UDP 数据，返回统一结构。

    {"kind": "raw"|"json"|"legacy", "action": str, "arg": dict,
     "seq": int|None, "token": str|None}
    """
    if is_ball_packet(data):
        return {"kind": "raw", "action": "raw", "arg": {}, "seq": None, "token": None}

    try:
        text = data.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise MessageError(f"无法解码的数据（{len(data)} 字节）") from error

    if not text:
        raise MessageError("空数据")

    if text.startswith("{"):
        try:
            body = json.loads(text)
        except json.JSONDecodeError as error:
            raise MessageError(f"JSON 格式错误: {error}") from error
        if not isinstance(body, dict):
            raise MessageError("JSON 顶层必须是对象")

        version = body.get("v", PROTOCOL_VERSION)
        if version != PROTOCOL_VERSION:
            raise MessageError(f"协议版本不支持: {version}（本端为 {PROTOCOL_VERSION}）")

        msg_type = str(body.get("type", "")).strip().lower()
        if msg_type == STREAM_TYPE:
            seq = body.get("seq")
            if seq is not None and not isinstance(seq, int):
                raise MessageError("seq 必须是整数")
            return {"kind": STREAM_TYPE, "action": STREAM_TYPE, "arg": {},
                    "seq": seq, "token": body.get("token"), "body": body}

        action = str(body.get("act", "")).strip().lower()
        if not action:
            raise MessageError("缺少 act 字段")
        arg = body.get("arg") or {}
        if not isinstance(arg, dict):
            raise MessageError("arg 必须是对象")

        seq = body.get("seq")
        if seq is not None and not isinstance(seq, int):
            raise MessageError("seq 必须是整数")

        return {"kind": "json", "action": action, "arg": arg,
                "seq": seq, "token": body.get("token")}

    action, arg = parse_legacy(text)
    return {"kind": "legacy", "action": action, "arg": arg, "seq": None, "token": None}
