# -*- coding: utf-8 -*-
"""协议编解码与兼容性测试（不依赖硬件）。"""

import json
import unittest

from emg_sdk.protocol import (ACTION_MOVE, ACTION_STOP, PROTOCOL_VERSION,
                              MessageError, clamp, encode_command,
                              is_ball_packet, parse)


class EncodeDecodeTests(unittest.TestCase):
    def test_move_roundtrip(self):
        data = encode_command(ACTION_MOVE, {"vx": 0.6, "wz": -0.25}, seq=7)
        body = json.loads(data)
        self.assertEqual(body["v"], PROTOCOL_VERSION)
        self.assertEqual(body["seq"], 7)

        message = parse(data)
        self.assertEqual(message["kind"], "json")
        self.assertEqual(message["action"], "move")
        self.assertEqual(message["arg"], {"vx": 0.6, "wz": -0.25})
        self.assertEqual(message["seq"], 7)

    def test_stop_without_arg(self):
        message = parse(encode_command(ACTION_STOP))
        self.assertEqual(message["action"], "stop")
        self.assertEqual(message["arg"], {})

    def test_token_passthrough(self):
        message = parse(encode_command(ACTION_STOP, token="abc"))
        self.assertEqual(message["token"], "abc")

    def test_missing_version_defaults_to_v2(self):
        message = parse(b'{"act":"stop"}')
        self.assertEqual(message["action"], "stop")

    def test_wrong_version_rejected(self):
        with self.assertRaises(MessageError):
            parse(b'{"v":3,"act":"stop"}')

    def test_action_case_normalized(self):
        self.assertEqual(parse(b'{"v":2,"act":"MOVE","arg":{}}')["action"], "move")


class LegacyTests(unittest.TestCase):
    def test_motion_aliases(self):
        cases = {
            "forward": (1.0, 0.0),
            "backward": (-1.0, 0.0),
            "left": (0.0, 1.0),
            "right": (0.0, -1.0),
        }
        for text, (vx, wz) in cases.items():
            message = parse(text.encode("utf-8"))
            self.assertEqual(message["kind"], "legacy")
            self.assertEqual(message["action"], "move")
            self.assertEqual(message["arg"], {"vx": vx, "wz": wz})

    def test_discrete_aliases(self):
        for text, expected in (("stop", "stop"), ("open", "open"), ("close", "close")):
            self.assertEqual(parse(text.encode("utf-8"))["action"], expected)

    def test_numeric_aliases(self):
        expected = {"1": (1.0, 0.0), "2": (-1.0, 0.0),
                    "3": (0.0, 1.0), "4": (0.0, -1.0)}
        for text, (vx, wz) in expected.items():
            message = parse(text.encode("utf-8"))
            self.assertEqual(message["arg"], {"vx": vx, "wz": wz})
        self.assertEqual(parse(b"5")["action"], "stop")
        self.assertEqual(parse(b"6")["action"], "open")
        self.assertEqual(parse(b"7")["action"], "close")

    def test_device_prefix_stripped(self):
        for text in (b"go2w:left", b"go2:left"):
            self.assertEqual(parse(text)["arg"], {"vx": 0.0, "wz": 1.0})

    def test_case_and_whitespace(self):
        self.assertEqual(parse(b"  FORWARD \r\n")["action"], "move")

    def test_unknown_command_rejected(self):
        with self.assertRaises(MessageError):
            parse(b"jump")


class InvalidInputTests(unittest.TestCase):
    def assert_invalid(self, data):
        with self.assertRaises(MessageError):
            parse(data)

    def test_empty(self):
        self.assert_invalid(b"")

    def test_non_utf8(self):
        self.assert_invalid(b"\xff\xfe\x00")

    def test_json_array_rejected(self):
        self.assert_invalid(b"[1,2,3]")

    def test_missing_action(self):
        self.assert_invalid(b'{"v":2}')

    def test_arg_not_object(self):
        self.assert_invalid(b'{"v":2,"act":"move","arg":[1,2]}')

    def test_seq_not_int(self):
        self.assert_invalid(b'{"v":2,"act":"stop","seq":"x"}')

    def test_broken_json(self):
        self.assert_invalid(b'{"v":2,"act":')


class BallPacketTests(unittest.TestCase):
    def test_detection(self):
        self.assertTrue(is_ball_packet(b"\x01\x81" + b"\x00" * 9))
        self.assertFalse(is_ball_packet(b"\x01\x81" + b"\x00" * 8))
        self.assertFalse(is_ball_packet(b"\x01\x82" + b"\x00" * 9))

    def test_parse_marks_raw(self):
        message = parse(b"\x01\x81" + b"\x00" * 9)
        self.assertEqual(message["kind"], "raw")
        self.assertEqual(message["action"], "raw")


class ClampTests(unittest.TestCase):
    def test_bounds(self):
        self.assertEqual(clamp(2.0), 1.0)
        self.assertEqual(clamp(-2.0), -1.0)
        self.assertEqual(clamp(0.3), 0.3)


if __name__ == "__main__":
    unittest.main()
