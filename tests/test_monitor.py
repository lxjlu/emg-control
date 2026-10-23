# -*- coding: utf-8 -*-
"""波形缓冲与网页监控测试：增量快照、环形覆盖、HTTP 接口、服务端接入。"""

import json
import unittest
import urllib.request

from emg_sdk.monitor import EmgMonitor, EmgStore
from emg_sdk.protocol import encode_command
from emg_sdk.server import CommandServer
from emg_sdk.stream import encode_stream
from tests.test_server import FakeAdapter, silent

ADDR = ("127.0.0.1", 5555)


def rows(count, channels=8, start=0):
    return [[start + index + channel for channel in range(channels)]
            for index in range(count)]


class StoreTests(unittest.TestCase):
    def test_incremental_snapshot(self):
        store = EmgStore()
        store.push(rows(10), 500, 8)
        first = store.snapshot(0)
        self.assertEqual(first["total"], 10)
        self.assertEqual(len(first["samples"]), 10)

        store.push(rows(5, start=10), 500, 8)
        second = store.snapshot(first["total"])
        self.assertEqual(second["total"], 15)
        self.assertEqual(len(second["samples"]), 5)
        self.assertEqual(second["samples"][0][0], 10)

    def test_values_are_preserved(self):
        store = EmgStore()
        store.push([[1, -2, 3, -4, 5, -6, 7, -8]], 500, 8)
        self.assertEqual(store.snapshot(0)["samples"][0], [1, -2, 3, -4, 5, -6, 7, -8])

    def test_ring_drops_oldest(self):
        store = EmgStore()
        store.push(rows(store.capacity + 100), 500, 8)
        snapshot = store.snapshot(0)
        self.assertEqual(snapshot["total"], store.capacity + 100)
        self.assertEqual(len(snapshot["samples"]), 2000)
        self.assertEqual(snapshot["from"], store.capacity + 100 - 2000)

    def test_since_beyond_total_returns_empty(self):
        store = EmgStore()
        store.push(rows(3), 500, 8)
        snapshot = store.snapshot(99)
        self.assertEqual(snapshot["samples"], [])

    def test_channel_change_resets(self):
        store = EmgStore()
        store.push(rows(4), 500, 8)
        store.push([[1], [2]], 500, 1)
        self.assertEqual(store.channels, 1)
        self.assertEqual(store.snapshot(0)["total"], 2)

    def test_stats_mark_connected(self):
        store = EmgStore()
        store.push(rows(2), 500, 8)
        self.assertTrue(store.snapshot(0)["stats"]["connected"])
        self.assertEqual(store.snapshot(0)["stats"]["packets"], 1)


class MonitorHttpTests(unittest.TestCase):
    def setUp(self):
        self.monitor = EmgMonitor(http_port=0, host="127.0.0.1", log=silent)
        self.monitor.start()
        self.addCleanup(self.monitor.stop)
        self.port = self.monitor._httpd.server_address[1]

    def get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=3) as response:
            return response.read().decode("utf-8")

    def test_page_is_served(self):
        page = self.get("/")
        self.assertIn("8 通道", page)
        self.assertIn("<canvas", page)

    def test_api_returns_json(self):
        self.monitor.push(rows(6), 500, 8)
        payload = json.loads(self.get("/api/emg?since=0"))
        self.assertEqual(payload["total"], 6)
        self.assertEqual(payload["channels"], 8)
        self.assertEqual(len(payload["samples"]), 6)
        self.assertTrue(payload["stats"]["connected"])

    def test_bad_since_falls_back_to_zero(self):
        self.monitor.push(rows(2), 500, 8)
        payload = json.loads(self.get("/api/emg?since=abc"))
        self.assertEqual(len(payload["samples"]), 2)


class ServerStreamTests(unittest.TestCase):
    def make_server(self):
        monitor = EmgMonitor(http_port=0, host="127.0.0.1", log=silent)
        server = CommandServer(FakeAdapter(), host="127.0.0.1", port=0,
                               monitor=monitor, log=silent)
        self.addCleanup(server._sock.close)
        return server, monitor

    def test_stream_reaches_store(self):
        server, monitor = self.make_server()
        server._handle(encode_stream(rows(4), seq=1), ADDR)
        self.assertEqual(monitor.store.snapshot(0)["total"], 4)

    def test_stream_does_not_touch_command_state(self):
        server, _ = self.make_server()
        server._handle(encode_command("move", {"vx": 1.0, "wz": 0.0}), ADDR)
        before = server._last_rx
        server._handle(encode_stream(rows(2), seq=2), ADDR)
        self.assertEqual(server._last_rx, before)
        self.assertEqual(server._action, "move")

    def test_bad_stream_is_ignored(self):
        server, monitor = self.make_server()
        body = json.loads(encode_stream(rows(2)).decode("utf-8"))
        body["n"] = 9
        server._handle(json.dumps(body).encode("utf-8"), ADDR)
        self.assertEqual(monitor.store.snapshot(0)["total"], 0)

    def test_stream_without_monitor_is_ignored(self):
        server = CommandServer(FakeAdapter(), host="127.0.0.1", port=0, log=silent)
        self.addCleanup(server._sock.close)
        server._handle(encode_stream(rows(2)), ADDR)
        self.assertEqual(server._action, "stop")


if __name__ == "__main__":
    unittest.main()
