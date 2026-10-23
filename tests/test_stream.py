# -*- coding: utf-8 -*-
"""肌电样本流（stream）报文测试：编解码、字段校验、与命令分流。"""

import base64
import json
import unittest

from emg_sdk.protocol import MessageError, parse
from emg_sdk.stream import MAX_CHANNELS, MAX_SAMPLES, decode_stream, encode_stream


class StreamCodecTests(unittest.TestCase):
    def test_roundtrip(self):
        samples = [[100, -200, 300, -400, 500, -600, 700, -800],
                   [101, -201, 301, -401, 501, -601, 701, -801]]
        message = encode_stream(samples, seq=7, rate=500, channels=8)
        body = json.loads(message.decode("utf-8"))

        self.assertEqual(body["type"], "stream")
        self.assertEqual(body["v"], 2)
        self.assertEqual(body["n"], 2)
        self.assertEqual(body["seq"], 7)

        decoded, rate, channels = decode_stream(body)
        self.assertEqual(decoded, samples)
        self.assertEqual(rate, 500.0)
        self.assertEqual(channels, 8)

    def test_single_channel(self):
        samples = [[-32768], [32767], [0]]
        _, _, channels = decode_stream(json.loads(
            encode_stream(samples, channels=1).decode("utf-8")))
        self.assertEqual(channels, 1)

    def test_empty_samples_rejected(self):
        with self.assertRaises(ValueError):
            encode_stream([])

    def test_too_many_samples_rejected(self):
        samples = [[0] * 8] * (MAX_SAMPLES + 1)
        with self.assertRaises(ValueError):
            encode_stream(samples)

    def test_wrong_channel_count_rejected(self):
        with self.assertRaises(ValueError):
            encode_stream([[1, 2, 3]], channels=8)


class StreamValidationTests(unittest.TestCase):
    def valid_body(self):
        return json.loads(encode_stream([[1] * 8] * 3).decode("utf-8"))

    def test_bad_channel_count(self):
        body = self.valid_body()
        body["ch"] = MAX_CHANNELS + 1
        with self.assertRaises(MessageError):
            decode_stream(body)

    def test_bad_rate(self):
        body = self.valid_body()
        body["rate"] = 0
        with self.assertRaises(MessageError):
            decode_stream(body)

    def test_bad_count(self):
        body = self.valid_body()
        body["n"] = 0
        with self.assertRaises(MessageError):
            decode_stream(body)

    def test_length_mismatch(self):
        body = self.valid_body()
        body["n"] = 5
        with self.assertRaises(MessageError):
            decode_stream(body)

    def test_bad_base64(self):
        body = self.valid_body()
        body["data"] = "@@@@"
        with self.assertRaises(MessageError):
            decode_stream(body)

    def test_missing_data(self):
        body = self.valid_body()
        del body["data"]
        with self.assertRaises(MessageError):
            decode_stream(body)

    def test_payload_is_little_endian(self):
        body = json.loads(encode_stream([[1, 2, 3, 4, 5, 6, 7, 8]]).decode("utf-8"))
        raw = base64.b64decode(body["data"])
        self.assertEqual(raw[:4], b"\x01\x00\x02\x00")


class StreamDispatchTests(unittest.TestCase):
    def test_parse_marks_stream(self):
        message = parse(encode_stream([[0] * 8], seq=3))
        self.assertEqual(message["kind"], "stream")
        self.assertEqual(message["seq"], 3)
        self.assertEqual(message["body"]["type"], "stream")

    def test_stream_without_act_is_accepted(self):
        body = json.loads(encode_stream([[0] * 8]).decode("utf-8"))
        body.pop("act", None)
        self.assertEqual(parse(json.dumps(body).encode("utf-8"))["kind"], "stream")

    def test_command_still_has_no_type(self):
        message = parse(b'{"v":2,"act":"stop"}')
        self.assertEqual(message["kind"], "json")
        self.assertNotIn("body", message)

    def test_bad_seq_rejected(self):
        body = json.loads(encode_stream([[0] * 8]).decode("utf-8"))
        body["seq"] = "abc"
        with self.assertRaises(MessageError):
            parse(json.dumps(body).encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
