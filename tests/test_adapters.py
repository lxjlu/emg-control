# -*- coding: utf-8 -*-
"""适配器测试：报文编码、速度换算、设备映射，全部不接硬件。"""

import unittest

from emg_sdk.adapters import ADAPTERS
from emg_sdk.adapters.ball import BallAdapter, encode_packet
from emg_sdk.adapters.go2 import Go2Adapter
from emg_sdk.adapters.go2w import Go2WAdapter
from emg_sdk.adapters.mechanical_car import MechanicalCarAdapter
from emg_sdk.adapters.ppt import PptAdapter
from emg_sdk.protocol import ACTION_MOVE, ACTION_STOP


def silent(*_args, **_kwargs):
    pass


class FakeSportClient:
    def __init__(self):
        self.calls = []

    def Move(self, vx, vy, wz):
        self.calls.append(("move", vx, vy, wz))

    def StopMove(self):
        self.calls.append(("stop",))

    def BalanceStand(self):
        self.calls.append(("balance",))


class FakeSocket:
    def __init__(self):
        self.sent = []

    def sendto(self, data, address):
        self.sent.append((data, address))

    def send(self, data):
        self.sent.append((data, None))


class BallEncodeTests(unittest.TestCase):
    def test_forward_frame(self):
        # 与 Android 旧实现、文档示例逐字节一致
        self.assertEqual(encode_packet(0, 1000).hex(" "),
                         "01 81 00 00 00 e8 03 00 12 6e 7e")

    def test_left_frame(self):
        self.assertEqual(encode_packet(-600, 0).hex(" "),
                         "01 81 00 a8 fd 00 00 00 12 f2 cf")

    def test_axes_clamped(self):
        frame = encode_packet(5000, -5000)
        self.assertEqual(frame[3:5], (1000).to_bytes(2, "little", signed=True))
        self.assertEqual(frame[5:7], (-1000).to_bytes(2, "little", signed=True))


class BallAdapterTests(unittest.TestCase):
    def test_apply_maps_axes(self):
        adapter = BallAdapter(dry_run=True, log=silent)
        adapter.start()
        adapter.apply(ACTION_MOVE, {"vx": 1.0, "wz": 0.0})
        self.assertEqual(adapter._last_axes, (0, 1000))

        adapter.apply(ACTION_MOVE, {"vx": 0.0, "wz": 1.0})
        self.assertEqual(adapter._last_axes, (-1000, 0))   # 协议里左为负

        adapter.stop()
        self.assertEqual(adapter._last_axes, (0, 0))

    def test_handle_raw_forwards_only_ball_frames(self):
        adapter = BallAdapter(dry_run=True, log=silent)
        self.assertTrue(adapter.handle_raw(b"\x01\x81" + b"\x00" * 9))
        self.assertFalse(adapter.handle_raw(b"forward"))

    def test_capabilities(self):
        caps = BallAdapter(dry_run=True, log=silent).capabilities()
        self.assertEqual(caps["type"], "ball")
        self.assertEqual(caps["publish_hz"], 20.0)
        self.assertEqual(caps["deadman_ms"], 400)
        self.assertEqual(caps["max_speed"], {"drive": 1000, "steer": 1000})


class Go2AdapterTests(unittest.TestCase):
    def make(self):
        adapter = Go2Adapter(dry_run=False, log=silent)
        adapter._client = FakeSportClient()
        return adapter

    def test_normalized_to_physical(self):
        adapter = self.make()
        adapter.apply(ACTION_MOVE, {"vx": 0.5, "wz": 0.5})
        self.assertEqual(adapter._client.calls, [("move", 0.4, 0.0, 1.0)])

    def test_values_clamped(self):
        adapter = self.make()
        adapter.apply(ACTION_MOVE, {"vx": 5.0, "wz": -5.0})
        self.assertEqual(adapter._client.calls, [("move", 0.8, 0.0, -2.0)])

    def test_stop(self):
        adapter = self.make()
        adapter.stop()
        self.assertEqual(adapter._client.calls, [("stop",)])

    def test_stop_without_client_is_safe(self):
        adapter = Go2Adapter(dry_run=False, log=silent)
        adapter._client = None
        adapter.stop()          # 不应抛异常


class Go2WAdapterTests(unittest.TestCase):
    def test_balance_stand_before_move(self):
        adapter = Go2WAdapter(dry_run=False, balance_settle=0, log=silent)
        adapter._client = FakeSportClient()
        adapter.apply(ACTION_MOVE, {"vx": 1.0, "wz": 0.0})
        self.assertEqual(adapter._client.calls[0], ("balance",))
        self.assertEqual(adapter._client.calls[1], ("move", 0.5, 0.0, 0.0))

        # 只准备一次
        adapter.apply(ACTION_MOVE, {"vx": 0.0, "wz": 1.0})
        self.assertEqual([c[0] for c in adapter._client.calls],
                         ["balance", "move", "move"])


class MechanicalCarAdapterTests(unittest.TestCase):
    def make(self):
        adapter = MechanicalCarAdapter(target="127.0.0.1:9999",
                                       dry_run=False, log=silent)
        adapter._sock = FakeSocket()
        return adapter

    def payloads(self, adapter):
        return [data.decode("utf-8") for data, _ in adapter._sock.sent]

    def test_motion_mapping(self):
        adapter = self.make()
        adapter.apply(ACTION_MOVE, {"vx": 1.0, "wz": 0.0})
        adapter.apply(ACTION_MOVE, {"vx": -1.0, "wz": 0.0})
        adapter.apply(ACTION_MOVE, {"vx": 0.0, "wz": 1.0})
        adapter.apply(ACTION_MOVE, {"vx": 0.0, "wz": -1.0})
        adapter.stop()
        self.assertEqual(self.payloads(adapter),
                         ["forward", "backward", "left", "right", "stop"])

    def test_gripper_mapping(self):
        adapter = self.make()
        adapter.apply("open", {})
        adapter.apply("close", {})
        self.assertEqual(self.payloads(adapter), ["open", "close"])

    def test_target_parsed(self):
        adapter = self.make()
        adapter.apply(ACTION_MOVE, {"vx": 1.0, "wz": 0.0})
        self.assertEqual(adapter._sock.sent[0][1], ("127.0.0.1", 9999))


class PptAdapterTests(unittest.TestCase):
    def test_discrete_only(self):
        caps = PptAdapter(dry_run=True, log=silent).capabilities()
        self.assertFalse(caps["continuous"])
        self.assertEqual(caps["deadman_ms"], 0)
        self.assertEqual(set(caps["actions"]), {"next", "prev", "start", "exit"})

    def test_dry_run_records_action(self):
        logs = []
        adapter = PptAdapter(dry_run=True, log=logs.append)
        adapter.apply("next", {})
        self.assertTrue(any("next" in line for line in logs))

    def test_stop_does_not_exit_slideshow(self):
        logs = []
        adapter = PptAdapter(dry_run=True, log=logs.append)
        adapter.stop()
        self.assertEqual(logs, [])

    def test_move_action_ignored(self):
        logs = []
        adapter = PptAdapter(dry_run=True, log=logs.append)
        adapter.apply(ACTION_MOVE, {"vx": 1.0})
        self.assertEqual(logs, [])


class RegistryTests(unittest.TestCase):
    def test_all_adapters_constructible(self):
        for name, cls in ADAPTERS.items():
            adapter = cls(dry_run=True, log=silent)
            caps = adapter.capabilities()
            self.assertEqual(caps["type"], adapter.name)
            self.assertTrue(caps["actions"], f"{name} 没有声明动作")
            self.assertGreater(caps["publish_hz"], 0)

    def test_names_unique(self):
        names = [cls.name for cls in ADAPTERS.values()]
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
