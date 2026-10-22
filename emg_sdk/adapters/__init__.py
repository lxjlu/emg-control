# -*- coding: utf-8 -*-
"""适配器注册表：新增被控对象时只需要在这里登记一行。"""

from .ball import BallAdapter
from .base import DeviceAdapter
from .go2 import Go2Adapter
from .go2w import Go2WAdapter
from .mechanical_car import MechanicalCarAdapter
from .ppt import PptAdapter

ADAPTERS = {
    "go2": Go2Adapter,
    "go2w": Go2WAdapter,
    "ball": BallAdapter,
    "car": MechanicalCarAdapter,
    "ppt": PptAdapter,
}

__all__ = ["ADAPTERS", "DeviceAdapter", "Go2Adapter", "Go2WAdapter",
           "BallAdapter", "MechanicalCarAdapter", "PptAdapter"]
