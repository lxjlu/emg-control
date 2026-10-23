# -*- coding: utf-8 -*-
"""肌电样本流：App → SDK 的实时波形通道。

    {"v":2,"type":"stream","seq":12,"rate":500,"ch":8,"n":10,
     "data":"<base64，n*ch 个小端 int16，按样本交错>"}

与命令报文共用同一个 UDP 端口，服务端按 type 分流：
流数据不参与心跳，也不会触发 deadman 或任何设备动作。
"""

import base64
import json
import struct

from .protocol import PROTOCOL_VERSION, STREAM_TYPE, MessageError

DEFAULT_RATE = 500
MAX_CHANNELS = 8
MAX_SAMPLES = 200          # 单包样本上限，防止异常大包


def encode_stream(samples, seq=None, rate=DEFAULT_RATE, channels=8):
    """把 [[ch1..chN], ...] 打成一条 stream 报文（测试与工具用）。"""
    if not samples:
        raise ValueError("samples 不能为空")
    if len(samples) > MAX_SAMPLES:
        raise ValueError(f"单包最多 {MAX_SAMPLES} 个样本")

    flat = []
    for sample in samples:
        if len(sample) != channels:
            raise ValueError(f"每个样本应有 {channels} 个通道")
        flat.extend(int(value) for value in sample)

    body = {
        "v": PROTOCOL_VERSION,
        "type": STREAM_TYPE,
        "rate": int(rate),
        "ch": int(channels),
        "n": len(samples),
        "data": base64.b64encode(struct.pack(f"<{len(flat)}h", *flat)).decode("ascii"),
    }
    if seq is not None:
        body["seq"] = int(seq)
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def decode_stream(body):
    """校验并解出一帧 stream，返回 (samples, rate, channels)。"""
    if not isinstance(body, dict):
        raise MessageError("stream 报文必须是 JSON 对象")

    channels = body.get("ch", MAX_CHANNELS)
    rate = body.get("rate", DEFAULT_RATE)
    count = body.get("n")
    data = body.get("data")

    if not isinstance(channels, int) or not 1 <= channels <= MAX_CHANNELS:
        raise MessageError(f"ch 非法: {channels!r}")
    if not isinstance(rate, (int, float)) or not 1 <= rate <= 5000:
        raise MessageError(f"rate 非法: {rate!r}")
    if not isinstance(count, int) or not 1 <= count <= MAX_SAMPLES:
        raise MessageError(f"n 非法: {count!r}")
    if not isinstance(data, str) or not data:
        raise MessageError("缺少 data 字段")

    try:
        raw = base64.b64decode(data, validate=True)
    except Exception as error:
        raise MessageError("data 不是合法的 base64") from error

    expected = count * channels * 2
    if len(raw) != expected:
        raise MessageError(f"data 长度 {len(raw)} 与 n*ch*2={expected} 不符")

    values = struct.unpack(f"<{count * channels}h", raw)
    samples = [list(values[index * channels:(index + 1) * channels])
               for index in range(count)]
    return samples, float(rate), channels
