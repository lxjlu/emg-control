# -*- coding: utf-8 -*-
"""emg_sdk —— 手势控制的统一中间层。

App 只发「语义命令」（move / stop / next / prev / ...），
每个被控对象装一个适配器，由 SDK 翻译成具体的 SDK 调用或设备协议。

    python -m emg_sdk list
    python -m emg_sdk run --adapter go2 --port 8888
    python -m emg_sdk send --host 192.168.10.1 --act move --vx 0.5
"""

from .protocol import PROTOCOL_VERSION, MessageError, encode_command, parse
from .server import CommandServer

__all__ = ["PROTOCOL_VERSION", "MessageError", "encode_command", "parse", "CommandServer"]
__version__ = "2.0.0"
