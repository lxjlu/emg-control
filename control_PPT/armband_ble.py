# -*- coding: utf-8 -*-
"""
armband_ble.py
肌电臂环 BLE 读取与 STM32 帧解析。

移植自项目 Android 端：
  - BleArmbandClient.java  : 服务 FFE0 / 通知 FFE2、设备名 BT-11
  - SemgPacketParser.java  : 帧头 A5 5A、8 通道 int16、CRC16-CCITT

帧格式（小端）：
    [0]     0xA5
    [1]     0x5A
    [2]     0x01
    [3]     0x01
    [4:8]   序号 sequence        uint32
    [8:12]  首个样本索引         uint32
    [12:14] 本帧样本数 1~10      uint16
    [14:]   样本数 × 8 通道 × int16
    [-2:]   CRC16（前面所有字节）uint16
"""

import asyncio
import struct
import time
from dataclasses import dataclass
from typing import List, Optional

CHANNELS = 8
SAMPLE_RATE = 500

STM32_SERVICE_UUID = "0000ffe0-0000-1000-8000-00805f9b34fb"
STM32_NOTIFY_UUID = "0000ffe2-0000-1000-8000-00805f9b34fb"
STM32_WRITE_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"
DEVICE_NAME_HINT = "BT-11"

STM32_HEADER = 14

# 连接参数：WinRT 全量 GATT 发现偏慢，给足超时并重试
CONNECT_TIMEOUT = 25.0
CONNECT_RETRIES = 3

# 扫描参数：臂环广播偶发丢失，单次扫描经常搜不到，因此多轮重试
SCAN_TIMEOUT = 6.0
SCAN_ROUNDS = 4
SCAN_PAUSE = 0.8


def is_armband_service(uuid: str) -> bool:
    """判断广播里的服务 UUID 是否为 FFE0（兼容 16 位与 128 位写法）。"""
    value = uuid.lower().replace("-", "")
    return value == "ffe0" or value.startswith("0000ffe0")


@dataclass
class Frame:
    sequence: int
    first_index: int
    samples: List[List[int]]   # [样本][8 通道]


def crc16_ccitt(data: bytes) -> int:
    """CRC16-CCITT：poly 0x1021，初值 0xFFFF，MSB first，无最终异或。"""
    crc = 0xFFFF
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


class Stm32Parser:
    """增量解析 BLE 通知流，输出完整帧。"""

    def __init__(self):
        self._buffer = bytearray()
        self.crc_errors = 0
        self.format_errors = 0
        self.discarded_bytes = 0

    def feed(self, data: bytes) -> List[Frame]:
        self._buffer.extend(data)
        frames: List[Frame] = []

        while True:
            sync = self._find_sync()
            if sync < 0:
                keep = 1 if self._buffer and self._buffer[-1] == 0xA5 else 0
                self.discarded_bytes += len(self._buffer) - keep
                if keep:
                    self._buffer = self._buffer[-1:]
                else:
                    self._buffer.clear()
                break

            if sync > 0:
                self.discarded_bytes += sync
                del self._buffer[:sync]

            if len(self._buffer) < STM32_HEADER:
                break

            if self._buffer[2] != 1 or self._buffer[3] != 1:
                self.format_errors += 1
                del self._buffer[:1]
                continue

            sample_count = self._buffer[12] | (self._buffer[13] << 8)
            if not 1 <= sample_count <= 10:
                self.format_errors += 1
                del self._buffer[:1]
                continue

            frame_size = STM32_HEADER + sample_count * CHANNELS * 2 + 2
            if len(self._buffer) < frame_size:
                break

            expected = self._buffer[frame_size - 2] | (self._buffer[frame_size - 1] << 8)
            actual = crc16_ccitt(bytes(self._buffer[:frame_size - 2]))
            if expected != actual:
                self.crc_errors += 1
                del self._buffer[:1]
                continue

            sequence = struct.unpack_from("<I", self._buffer, 4)[0]
            first_index = struct.unpack_from("<I", self._buffer, 8)[0]
            offset = STM32_HEADER
            samples = []
            for _ in range(sample_count):
                samples.append(list(struct.unpack_from("<8H", self._buffer, offset)))
                offset += CHANNELS * 2

            frames.append(Frame(sequence, first_index, samples))
            del self._buffer[:frame_size]

        return frames

    def _find_sync(self) -> int:
        for i in range(len(self._buffer) - 1):
            if self._buffer[i] == 0xA5 and self._buffer[i + 1] == 0x5A:
                return i
        return -1


def _short_uuid(uuid: str) -> str:
    value = uuid.lower().replace("-", "")
    return value[4:8] if value.startswith("0000") and len(value) >= 8 else value


def list_devices(timeout: float = 6.0) -> List[dict]:
    """扫描附近 BLE 设备（同步接口，便于命令行使用）。"""
    from bleak import BleakScanner

    async def _scan():
        found = await BleakScanner.discover(timeout=timeout, return_adv=True)
        result = []
        for device, adv in found.values():
            uuids = list(adv.service_uuids or []) if adv else []
            name = device.name or ((adv.local_name if adv else "") or "")
            result.append({
                "name": name,
                "address": device.address,
                "services": uuids,
                "services_short": [_short_uuid(u) for u in uuids],
                "likely_armband": any(is_armband_service(u) for u in uuids),
            })
        result.sort(key=lambda item: (not item["likely_armband"], item["name"]))
        return result

    return asyncio.run(_scan())


class ArmbandBle:
    """BLE 连接封装：连接后订阅 FFE2，把样本回调交给调用方。"""

    def __init__(self, address: Optional[str] = None, name_hint: str = DEVICE_NAME_HINT,
                 on_samples=None, on_status=None):
        self.address = address
        self.name_hint = name_hint
        self.on_samples = on_samples
        self.on_status = on_status or (lambda text: None)
        self.on_raw = None          # 调试用：原始通知字节回调
        self.parser = Stm32Parser()
        self.client = None
        self.notifications = 0
        self.bytes_received = 0
        self.disconnected = False
        self.last_notification_at = 0.0
        self.mtu = None

    @property
    def is_connected(self) -> bool:
        return (self.client is not None and self.client.is_connected
                and not self.disconnected)

    def _on_disconnect(self, _client):
        self.disconnected = True
        self.on_status("BLE 连接已断开")

    async def _find_device(self, timeout: float = SCAN_TIMEOUT, rounds: int = SCAN_ROUNDS):
        """按 FFE0 服务识别臂环；名称在未连接时可能为空，因此不依赖名称。

        臂环的广播包会偶发丢失，单次扫描经常搜不到，所以多轮重试，
        任意一轮命中就立刻返回。
        """
        from bleak import BleakScanner

        seen = {}
        for round_index in range(1, rounds + 1):
            self.on_status(f"正在扫描 BLE 设备（第 {round_index}/{rounds} 轮，按 FFE0 服务识别）…")
            found = await BleakScanner.discover(timeout=timeout, return_adv=True)
            entries = list(found.values())

            by_service = []
            by_name = []
            for device, adv in entries:
                uuids = (adv.service_uuids or []) if adv else []
                seen[device.address] = (
                    device.name or ((adv.local_name if adv else "") or ""),
                    uuids,
                )
                if any(is_armband_service(u) for u in uuids):
                    by_service.append(device)
                    continue
                name = device.name or ((adv.local_name if adv else "") or "")
                if self.name_hint.lower() in name.lower():
                    by_name.append(device)

            if by_service:
                device = by_service[0]
                self.on_status(f"按 FFE0 服务找到设备：{device.address}")
                return device
            if by_name:
                device = by_name[0]
                self.on_status(f"按名称找到设备：{device.name or '未命名'} ({device.address})")
                return device

            if round_index < rounds:
                self.on_status(f"本轮未找到，{SCAN_PAUSE:.1f} 秒后重试…")
                await asyncio.sleep(SCAN_PAUSE)

        summary = "、".join(
            f"{name or '未命名'}({address})[{','.join(_short_uuid(u) for u in uuids)}]"
            for address, (name, uuids) in list(seen.items())[:12]
        ) or "无"
        raise RuntimeError(
            f"扫描 {rounds} 轮仍未找到广播 FFE0 服务的设备，也没有名称包含 {self.name_hint} 的设备。\n"
            f"累计看到的设备：{summary}\n"
            "请依次检查：\n"
            "  1) 臂环是否开机（长按 SW2 3 秒会启动采集，指示灯应闪烁）\n"
            "  2) 是否被手机 App 占用（同一时间只能连一个中心设备，先在 App 里断开）\n"
            "  3) 靠近到 1 米内，或直接用 --address 指定地址跳过扫描"
        )

    async def connect(self):
        from bleak import BleakClient

        target = self.address
        if target is None:
            target = await self._find_device()
            self.address = target.address
        else:
            self.address = target if isinstance(target, str) else target.address

        last_error = None
        for attempt in range(1, CONNECT_RETRIES + 1):
            self.on_status(f"正在连接 {self.address}（第 {attempt}/{CONNECT_RETRIES} 次，"
                           f"超时 {CONNECT_TIMEOUT:.0f} 秒）…")
            try:
                # services 过滤：只发现 FFE0，避免 WinRT 全量 GATT 枚举超时
                self.client = BleakClient(
                    target,
                    disconnected_callback=self._on_disconnect,
                    timeout=CONNECT_TIMEOUT,
                    services=[STM32_SERVICE_UUID],
                )
                await self.client.connect(timeout=CONNECT_TIMEOUT)
                if not self.client.is_connected:
                    raise RuntimeError("连接未就绪")
                self.disconnected = False
                self.last_notification_at = time.time()
                try:
                    self.mtu = self.client.mtu_size
                except Exception:
                    self.mtu = None
                last_error = None
                break
            except Exception as error:
                last_error = error
                self.on_status(f"连接失败：{error}")
                try:
                    if self.client is not None:
                        await self.client.disconnect()
                except Exception:
                    pass
                self.client = None
                if attempt < CONNECT_RETRIES:
                    await asyncio.sleep(1.5)

        if last_error is not None:
            raise RuntimeError(
                f"BLE 连接失败：{last_error}\n"
                "请依次检查：\n"
                "  1) 臂环是否已被手机 App 占用（同一时间只能一个中心设备，先在 App 里断开）\n"
                "  2) Windows「蓝牙和其他设备」中删除该设备后重试（清除旧配对缓存）\n"
                "  3) 臂环是否开机并在广播，靠近到 1 米内\n"
                "  4) 关闭其他占用蓝牙的应用后重试"
            )

        await self.client.start_notify(STM32_NOTIFY_UUID, self._on_notify)
        self.on_status(f"已连接并订阅 FFE2（MTU={self.mtu}），"
                       "等待肌电数据（手环需长按 SW2 3 秒启动采集）")

    def _on_notify(self, *args):
        data = args[-1]
        if not data:
            return
        self.last_notification_at = time.time()
        self.notifications += 1
        self.bytes_received += len(data)
        if self.on_raw:
            self.on_raw(bytes(data))
        for frame in self.parser.feed(bytes(data)):
            if self.on_samples:
                self.on_samples(frame.samples)

    async def reconnect(self, attempts: int = 3, delay: float = 1.0) -> bool:
        """掉线后自动重连。实测该臂环重连后会继续推送数据，无需重新按 SW2。"""
        for attempt in range(1, attempts + 1):
            self.on_status(f"正在重连（第 {attempt}/{attempts} 次）…")
            try:
                await self.connect()
                return True
            except Exception as error:
                self.on_status(f"重连失败：{error}")
                if attempt < attempts:
                    await asyncio.sleep(delay)
        return False

    async def disconnect(self):
        if self.client is not None:
            try:
                if self.client.is_connected:
                    await self.client.stop_notify(STM32_NOTIFY_UUID)
            except Exception:
                pass
            try:
                await self.client.disconnect()
            except Exception:
                pass
        self.client = None

    def stats(self) -> str:
        return (f"收到 {self.notifications} 个通知 / {self.bytes_received} 字节，"
                f"CRC 错误 {self.parser.crc_errors}，格式错误 {self.parser.format_errors}")
