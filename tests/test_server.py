# -*- coding: utf-8 -*-
"""服务端会话测试：心跳、看门狗、去重、token、能力发现。

全部直接调用 CommandServer 的内部方法，不启动线程、不依赖真实设备。
"""

import json
import socket
import time
import unittest

from emg_sdk.adapters.base import DeviceAdapter
from emg_sdk.protocol import encode_command
from emg_sdk.server import CommandServer

ADDR = ("127.0.0.1", 5555)
BALL_FRAME = b"\x01\x81" + b"\x00" * 9


def silent(*_args, **_kwargs):
    pass


class FakeAdapter(DeviceAdapter):
    name = "fake"
    display = "Fake 设备"
    continuous = True
    publish_hz = 20.0
    deadman_ms = 400

    def __init__(self):
        super().__init__(dry_run=True, log=silent)
        self.applied = []
        self.stops = 0
        self.raw = []
        self.starts = 0

    def start(self):
        self.starts += 1

    def apply(self, action, args):
        self.applied.append((action, dict(args)))

    def stop(self):
        self.stops += 1

    def handle_raw(self, data):
        self.raw.append(data)
        return True


class ServerTestCase(unittest.TestCase):
    def make_server(self, **kwargs):
        adapter = FakeAdapter()
        server = CommandServer(adapter, host="127.0.0.1", port=0,
                               log=silent, **kwargs)
        replies = []
        server._reply = lambda addr, payload: replies.append(payload)
        self.addCleanup(server._sock.close)
        return server, adapter, replies


class MotionTests(ServerTestCase):
    def test_move_is_not_sent_immediately(self):
        server, adapter, _ = self.make_server()
        server._handle(encode_command("move", {"vx": 1.0, "wz": 0.0}), ADDR)
        self.assertEqual(adapter.applied, [])
        self.assertEqual(server._action, "move")

    def test_publish_repeats_move(self):
        server, adapter, _ = self.make_server()
        server._handle(encode_command("move", {"vx": 0.5, "wz": 0.0}), ADDR)
        server._publish()
        server._publish()
        self.assertEqual([action for action, _ in adapter.applied], ["move", "move"])
        self.assertEqual(adapter.applied[0][1], {"vx": 0.5, "wz": 0.0})

    def test_deadman_stops_once(self):
        server, adapter, _ = self.make_server()
        server._handle(encode_command("move", {"vx": 1.0, "wz": 0.0}), ADDR)
        server._last_rx -= 10          # 模拟 10 秒没有新命令
        server._publish()
        self.assertEqual(adapter.stops, 1)
        self.assertEqual(server._action, "stop")

        server._publish()              # 不应重复 stop
        self.assertEqual(adapter.stops, 1)

    def test_stop_message_is_idempotent(self):
        server, adapter, _ = self.make_server()
        server._handle(encode_command("stop"), ADDR)
        self.assertEqual(adapter.stops, 1)
        server._publish()
        self.assertEqual(adapter.stops, 1)


class DedupeAndTokenTests(ServerTestCase):
    def test_duplicate_seq_ignored(self):
        server, adapter, _ = self.make_server()
        payload = encode_command("stop", seq=42)
        server._handle(payload, ADDR)
        server._handle(payload, ADDR)
        self.assertEqual(adapter.stops, 1)

    def test_token_mismatch_ignored(self):
        server, adapter, replies = self.make_server(token="secret")
        server._handle(encode_command("move", {"vx": 1.0}), ADDR)
        self.assertEqual(server._action, "stop")
        self.assertEqual(adapter.applied, [])
        self.assertEqual(replies, [])

    def test_token_match_accepted(self):
        server, adapter, _ = self.make_server(token="secret")
        server._handle(encode_command("move", {"vx": 1.0}, token="secret"), ADDR)
        self.assertEqual(server._action, "move")


class DiscoveryTests(ServerTestCase):
    def test_hello_returns_capabilities(self):
        server, _, replies = self.make_server()
        server._handle(encode_command("hello"), ADDR)
        self.assertEqual(len(replies), 1)
        reply = replies[0]
        self.assertTrue(reply["ok"])
        self.assertEqual(reply["type"], "fake")
        self.assertEqual(reply["v"], 2)
        self.assertIn("actions", reply)
        self.assertFalse(reply["token_required"])

    def test_hello_reports_token_required(self):
        server, _, replies = self.make_server(token="secret")
        server._handle(encode_command("hello", token="secret"), ADDR)
        self.assertTrue(replies[0]["token_required"])

    def test_unknown_action_replies_error(self):
        server, _, replies = self.make_server()
        server._handle(encode_command("jump"), ADDR)
        self.assertFalse(replies[0]["ok"])
        self.assertIn("error", replies[0])


class DiscreteAndRawTests(ServerTestCase):
    def test_discrete_action_executed_once(self):
        server, adapter, _ = self.make_server()
        server._handle(encode_command("next"), ADDR)
        self.assertEqual(adapter.applied, [("next", {})])
        server._publish()
        self.assertEqual(len(adapter.applied), 1)

    def test_raw_ball_frame_bypasses_heartbeat(self):
        server, adapter, _ = self.make_server()
        server._handle(BALL_FRAME, ADDR)
        self.assertEqual(adapter.raw, [BALL_FRAME])
        self.assertTrue(server._raw_mode)
        server._publish()
        self.assertEqual(adapter.applied, [])

    def test_json_command_exits_raw_mode(self):
        server, adapter, _ = self.make_server()
        server._handle(BALL_FRAME, ADDR)
        server._handle(encode_command("move", {"vx": 1.0}), ADDR)
        self.assertFalse(server._raw_mode)
        server._publish()
        self.assertEqual(adapter.applied[0][0], "move")


class AckTests(ServerTestCase):
    def test_invalid_packet_replies_error(self):
        server, _, replies = self.make_server()
        server._handle(b"not-a-command", ADDR)
        self.assertTrue(replies)
        self.assertFalse(replies[-1]["ok"])

    def test_ack_disabled_only_replies_errors(self):
        adapter = FakeAdapter()
        server = CommandServer(adapter, host="127.0.0.1", port=0,
                               ack=False, log=silent)
        self.addCleanup(server._sock.close)

        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.bind(("127.0.0.1", 0))
        client.settimeout(0.2)
        self.addCleanup(client.close)
        addr = client.getsockname()

        server._handle(encode_command("move", {"vx": 1.0}), addr)
        with self.assertRaises(socket.timeout):
            client.recvfrom(1024)

        server._handle(b"broken", addr)
        data, _ = client.recvfrom(1024)
        self.assertFalse(json.loads(data.decode("utf-8"))["ok"])


class TimingTests(unittest.TestCase):
    def test_monotonic_clock_used(self):
        # 看门狗依赖 monotonic，避免系统时间跳变导致误停
        server = CommandServer(FakeAdapter(), host="127.0.0.1", port=0, log=silent)
        self.addCleanup(server._sock.close)
        before = time.monotonic()
        server._handle(encode_command("move", {"vx": 1.0}), ADDR)
        self.assertGreaterEqual(server._last_rx, before)


if __name__ == "__main__":
    unittest.main()
