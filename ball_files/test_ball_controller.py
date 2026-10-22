

import json
from pathlib import Path
import socket
import sys
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "host"))
from ball_controller import (BallConfig, BallController, CONTROL_LOCK,
                             crc16_maxim, encode_command, make_http_handler)
from http.server import ThreadingHTTPServer


EXAMPLES = (
    ((0, 0), "01 81 00 00 00 00 00 00 12 CE BB"),
    ((0, 1000), "01 81 00 00 00 E8 03 00 12 6E 7E"),
    ((0, -1000), "01 81 00 00 00 18 FC 00 12 5E 7D"),
    ((-1000, 0), "01 81 00 18 FC 00 00 00 12 02 E8"),
    ((1000, 0), "01 81 00 E8 03 00 00 00 12 E6 E8"),
)


class ProtocolTest(unittest.TestCase):
    def test_supplier_vectors(self):
        for axes, expected in EXAMPLES:
            with self.subTest(axes=axes):
                self.assertEqual(encode_command(*axes).hex(" ").upper(), expected)
                self.assertEqual(len(encode_command(*axes)), 11)

    def test_crc_standard_and_byte_order(self):
        self.assertEqual(crc16_maxim(b"123456789"), 0x44C2)
        self.assertEqual(encode_command(crc_byteorder="little")[-2:], bytes.fromhex("BB CE"))

    def test_rejects_invalid_commands(self):
        for kwargs in ({"steer": 1001}, {"drive": -1001}, {"servo": 101},
                       {"turn_speed": 64}, {"steer": True}, {"control": CONTROL_LOCK, "drive": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                encode_command(**kwargs)
        with self.assertRaises(ValueError):
            BallConfig(source_ip="224.0.0.1")

    def test_default_route_is_computer_to_sta(self):
        config = BallConfig()
        self.assertEqual((config.source_ip, config.source_port), ("192.168.1.50", 3020))
        self.assertEqual((config.sta_ip, config.sta_port), ("192.168.1.11", 3020))
        self.assertEqual(BallConfig(ap_ip="127.0.0.1", ap_port=3999).sta_ip, "127.0.0.1")
        self.assertEqual(BallConfig(ap_ip="127.0.0.1", ap_port=3999).sta_port, 3999)


class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver.bind(("127.0.0.1", 0))
        self.receiver.settimeout(1)
        self.controller = BallController(BallConfig(
            source_ip="127.0.0.1", source_port=0, sta_ip="127.0.0.1",
            sta_port=self.receiver.getsockname()[1], send_interval=0.025,
            lease_seconds=0.12))

    def tearDown(self):
        self.controller.close()
        self.receiver.close()

    def receive_until(self, predicate, timeout=1):
        limit = time.monotonic() + timeout
        while time.monotonic() < limit:
            packet, _ = self.receiver.recvfrom(64)
            self.assertEqual(len(packet), 11)
            self.assertEqual(packet[-2:], crc16_maxim(packet[:9]).to_bytes(2, "big"))
            if predicate(packet):
                return packet
        self.fail("expected UDP command did not arrive")

    def test_motion_lease_and_lock(self):
        self.assertEqual(self.controller._socket.getpeername(), self.receiver.getsockname())
        self.assertEqual(self.controller.status()["destination"],
                         "127.0.0.1:%d" % self.receiver.getsockname()[1])
        self.assertFalse(self.controller.status()["armed"])
        self.receiver.settimeout(0.05)
        with self.assertRaises(socket.timeout):
            self.receiver.recvfrom(64)
        with self.assertRaises(RuntimeError):
            self.controller.set_motion("forward", 1000)
        self.controller.arm()
        for _ in range(3):
            self.assertEqual(self.receiver.recvfrom(64)[0][2], 0x40)
        self.assertEqual(self.receiver.recvfrom(64)[0][2], 0x00)
        self.controller.set_motion("forward", 1000)
        self.receive_until(lambda p: p[5:7] == bytes.fromhex("E8 03"))
        time.sleep(0.16)
        self.receive_until(lambda p: p[2] == 0 and p[3:7] == b"\x00" * 4)
        self.assertEqual(self.controller.status()["drive"], 0)
        self.controller.set_axes(-1000, 0)
        self.receive_until(lambda p: p[3:5] == bytes.fromhex("18 FC"))
        self.controller.lock()
        self.receive_until(lambda p: p[2] == 0x20 and p[3:7] == b"\x00" * 4)
        self.assertFalse(self.controller.status()["armed"])
        self.controller.close()
        with self.assertRaises(RuntimeError):
            self.controller.arm()

    def test_http_api(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_http_handler(self.controller, "test-token"))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = "http://127.0.0.1:%d" % server.server_address[1]

        def request(path, data=None, token="test-token"):
            headers = {"X-Control-Token": token}
            if data is not None:
                headers["Content-Type"] = "application/json"
            body = json.dumps(data).encode() if data is not None else None
            return urlopen(Request(url + path, data=body, headers=headers), timeout=2)

        try:
            with self.assertRaises(HTTPError) as denied:
                request("/status", token="wrong")
            self.assertEqual(denied.exception.code, 401)
            with self.assertRaises(HTTPError) as locked:
                request("/motion", {"motion": "forward", "speed": 1000})
            self.assertEqual(locked.exception.code, 409)
            self.assertTrue(json.load(request("/arm", {}))["armed"])
            self.receive_until(lambda p: p[2] == 0x40)
            self.assertEqual(json.load(request("/motion", {"motion": "forward", "speed": 1000}))["drive"], 1000)
            self.receive_until(lambda p: p[5:7] == bytes.fromhex("E8 03"))
            with self.assertRaises(HTTPError) as invalid:
                request("/axes", {"steer": 2000, "drive": 0})
            self.assertEqual(invalid.exception.code, 400)
            self.assertEqual(json.load(request("/stop", {}))["drive"], 0)
            self.assertFalse(json.load(request("/lock", {}))["armed"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
