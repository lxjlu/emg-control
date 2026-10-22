#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import ipaddress
import json
import socket
import struct
import threading
import time
from typing import Optional


CONTROL_NORMAL = 0x00
CONTROL_LOCK = 0x20
CONTROL_UNLOCK = 0x40
BALL_ID = b"\x01\x81"
MOTIONS = {"stop", "forward", "backward", "left", "right"}


def crc16_maxim(data: bytes) -> int:
    """CRC-16/MAXIM-DOW: poly 8005, init 0000, refin/refout, xorout FFFF."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xA001 if crc & 1 else 0)
    return crc ^ 0xFFFF


def _bounded_int(name: str, value: int, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError("%s must be an integer from %d to %d" % (name, low, high))
    return value


def encode_command(
    steer: int = 0,
    drive: int = 0,
    *,
    control: int = CONTROL_NORMAL,
    servo: int = 0,
    turn_speed: int = 0x12,
    crc_byteorder: str = "big",
) -> bytes:
    """Encode one 11-byte command; +steer is right, +drive is forward.

    `crc_byteorder='big'` follows the printed CEBB/6E7E examples. The PDF
    does not explicitly define wire order for those two CRC bytes.
    """
    _bounded_int("steer", steer, -1000, 1000)
    _bounded_int("drive", drive, -1000, 1000)
    _bounded_int("servo", servo, 0, 100)
    _bounded_int("turn_speed", turn_speed, 0, 63)
    if control not in (CONTROL_NORMAL, CONTROL_LOCK, CONTROL_UNLOCK) or isinstance(control, bool):
        raise ValueError("control must be 0x00, 0x20, or 0x40")
    if control != CONTROL_NORMAL and (steer or drive):
        raise ValueError("lock and unlock require zero axes")
    if crc_byteorder not in ("big", "little"):
        raise ValueError("crc_byteorder must be 'big' or 'little'")
    payload = BALL_ID + bytes((control,)) + struct.pack("<hhBB", steer, drive, servo, turn_speed)
    return payload + crc16_maxim(payload).to_bytes(2, crc_byteorder)


@dataclass(frozen=True)
class BallConfig:
    source_ip: str = "192.168.1.50"
    source_port: int = 3020
    sta_ip: str = "192.168.1.11"
    sta_port: int = 3020
    servo: int = 0
    turn_speed: int = 0x12
    crc_byteorder: str = "big"
    send_interval: float = 0.05
    lease_seconds: float = 0.4
    # Old callers used these names when the UDP destination was the AP.
    ap_ip: Optional[str] = None
    ap_port: Optional[int] = None

    def __post_init__(self) -> None:
        if self.ap_ip is not None:
            object.__setattr__(self, "sta_ip", self.ap_ip)
        if self.ap_port is not None:
            object.__setattr__(self, "sta_port", self.ap_port)
        for name in ("source_ip", "sta_ip"):
            address = ipaddress.ip_address(getattr(self, name))
            if address.version != 4 or address.is_multicast or address.is_unspecified or address.is_reserved:
                raise ValueError("%s must be a unicast IPv4 address" % name)
        _bounded_int("source_port", self.source_port, 0, 65535)
        _bounded_int("sta_port", self.sta_port, 1, 65535)
        encode_command(servo=self.servo, turn_speed=self.turn_speed, crc_byteorder=self.crc_byteorder)
        if not 0.01 <= self.send_interval <= 1.0:
            raise ValueError("send_interval must be 0.01..1 seconds")
        if not self.send_interval < self.lease_seconds <= 5.0:
            raise ValueError("lease_seconds must exceed send_interval and be at most 5 seconds")


class BallController:
    """Thread-safe UDP controller. Motion expires unless callers renew its lease.

    Construction only binds a socket. `arm()` explicitly requests unlock.
    Call `set_motion()` or `set_axes()` repeatedly while motion is wanted;
    the controller sends at approximately 20 Hz and returns to zero axes
    after the lease expires. `lock()` sends three zero-axis lock requests.
    UDP delivery and physical stopping cannot be confirmed by this API.
    """

    def __init__(self, config: BallConfig = BallConfig()) -> None:
        self.config = config
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._socket.bind((config.source_ip, config.source_port))
            self._socket.connect((config.sta_ip, config.sta_port))
            self._source_address = self._socket.getsockname()
        except Exception:
            self._socket.close()
            raise
        self._mutex = threading.RLock()
        self._closed = False
        self._armed = False
        self._axes = (0, 0)
        self._deadline = 0.0
        self._last_packet: Optional[bytes] = None
        self._last_error: Optional[str] = None
        self._halt = threading.Event()
        self._worker = threading.Thread(target=self._run, name="ball-udp-heartbeat", daemon=True)
        self._worker.start()

    def _packet(self, steer: int, drive: int, control: int = CONTROL_NORMAL) -> bytes:
        return encode_command(steer, drive, control=control, servo=self.config.servo,
                              turn_speed=self.config.turn_speed, crc_byteorder=self.config.crc_byteorder)

    def _send(self, packet: bytes) -> None:
        self._socket.send(packet)
        self._last_packet = packet

    def arm(self) -> None:
        with self._mutex:
            self._require_open()
            if self._armed:
                return
            self._axes = (0, 0)
            self._deadline = 0.0
            try:
                for _ in range(3):
                    self._send(self._packet(0, 0, CONTROL_UNLOCK))
                self._send(self._packet(0, 0))
            except OSError as exc:
                self._last_error = str(exc)
                raise
            self._armed = True
            self._last_error = None

    def set_axes(self, steer: int, drive: int) -> None:
        _bounded_int("steer", steer, -1000, 1000)
        _bounded_int("drive", drive, -1000, 1000)
        with self._mutex:
            self._require_open()
            if not self._armed:
                raise RuntimeError("ball is locked; call arm() first")
            self._axes = (steer, drive)
            self._deadline = time.monotonic() + self.config.lease_seconds
            try:
                self._send(self._packet(steer, drive))
            except OSError as exc:
                self._last_error = str(exc)
                self._axes = (0, 0)
                self._deadline = 0.0
                self._armed = False
                raise

    def set_motion(self, motion: str, speed: int = 250) -> None:
        if motion not in MOTIONS:
            raise ValueError("motion must be stop, forward, backward, left, or right")
        _bounded_int("speed", speed, 1, 1000)
        steer, drive = {
            "stop": (0, 0), "forward": (0, speed), "backward": (0, -speed),
            "left": (-speed, 0), "right": (speed, 0),
        }[motion]
        self.set_axes(steer, drive)

    def stop(self) -> None:
        with self._mutex:
            self._require_open()
            self._axes = (0, 0)
            self._deadline = 0.0
            if self._armed:
                self._send(self._packet(0, 0))

    def lock(self) -> None:
        with self._mutex:
            self._require_open()
            self._axes = (0, 0)
            self._deadline = 0.0
            if not self._armed:
                return
            self._armed = False
            error = None
            for _ in range(3):
                try:
                    self._send(self._packet(0, 0, CONTROL_LOCK))
                except OSError as exc:
                    error = exc
            if error is not None:
                self._last_error = str(error)
                raise error

    def status(self) -> dict:
        with self._mutex:
            now = time.monotonic()
            effective = self._axes if self._armed and now < self._deadline else (0, 0)
            return {
                "armed": self._armed, "closed": self._closed,
                "source": "%s:%d" % self._source_address,
                "destination": "%s:%d" % (self.config.sta_ip, self.config.sta_port),
                "steer": effective[0], "drive": effective[1],
                "lease_ms_remaining": max(0, int((self._deadline - now) * 1000)) if self._armed else 0,
                "last_packet_hex": self._last_packet.hex(" ").upper() if self._last_packet else None,
                "last_error": self._last_error,
                "note": "UDP send is not an acknowledgement from the ball",
            }

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("controller is closed")

    def _run(self) -> None:
        while not self._halt.wait(self.config.send_interval):
            with self._mutex:
                if not self._armed or self._closed:
                    continue
                if time.monotonic() >= self._deadline:
                    self._axes = (0, 0)
                try:
                    self._send(self._packet(*self._axes))
                except OSError as exc:
                    self._last_error = str(exc)
                    self._axes = (0, 0)
                    self._armed = False
                    try:
                        for _ in range(3):
                            self._send(self._packet(0, 0, CONTROL_LOCK))
                    except OSError:
                        pass

    def close(self) -> None:
        with self._mutex:
            if self._closed:
                return
            try:
                self.lock()
            except OSError as exc:
                self._last_error = str(exc)
            finally:
                self._closed = True
                self._halt.set()
                self._socket.close()
        if threading.current_thread() is not self._worker:
            self._worker.join(timeout=1.0)

    def __enter__(self) -> BallController:
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


def make_http_handler(controller: BallController, token: Optional[str] = None):
    """Create JSON API: GET /status; POST /arm, /motion, /axes, /stop, /lock."""
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            if token is None or hmac.compare_digest(self.headers.get("X-Control-Token", ""), token):
                return True
            self._reply(401, {"error": "invalid control token"})
            return False

        def do_GET(self) -> None:
            if not self._authorized():
                return
            if self.path != "/status":
                self._reply(404, {"error": "unknown path"})
                return
            self._reply(200, controller.status())

        def do_POST(self) -> None:
            if not self._authorized():
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > 4096:
                    self._reply(413, {"error": "request body too large"})
                    return
                body = json.loads(self.rfile.read(length)) if length else {}
                if not isinstance(body, dict):
                    raise ValueError("JSON object required")
                if self.path == "/arm":
                    controller.arm()
                elif self.path == "/motion":
                    controller.set_motion(body["motion"], body.get("speed", 250))
                elif self.path == "/axes":
                    controller.set_axes(body["steer"], body["drive"])
                elif self.path == "/stop":
                    controller.stop()
                elif self.path == "/lock":
                    controller.lock()
                else:
                    self._reply(404, {"error": "unknown path"})
                    return
                self._reply(200, controller.status())
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self._reply(400, {"error": str(exc)})
            except RuntimeError as exc:
                self._reply(409, {"error": str(exc)})
            except OSError as exc:
                self._reply(503, {"error": str(exc)})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="V1.2 reconnaissance-ball UDP controller")
    sub = parser.add_subparsers(dest="action", required=True)
    preview = sub.add_parser("preview", help="print example frames without opening a socket")
    preview.add_argument("--speed", type=int, default=1000)
    server_parser = sub.add_parser("serve", help="start the JSON HTTP API; no ball command sent until /arm")
    server_parser.add_argument("--source-ip", default="192.168.1.50")
    server_parser.add_argument("--source-port", type=int, default=3020)
    server_parser.add_argument("--sta-ip", "--ap-ip", dest="sta_ip", default="192.168.1.11")
    server_parser.add_argument("--sta-port", "--ap-port", dest="sta_port", type=int, default=3020)
    server_parser.add_argument("--listen", default="127.0.0.1")
    server_parser.add_argument("--http-port", type=int, default=18765)
    server_parser.add_argument("--token", help="required when HTTP listen address is non-loopback")
    server_parser.add_argument("--crc-byteorder", choices=("big", "little"), default="big")
    server_parser.add_argument("--servo", type=int, default=0)
    server_parser.add_argument("--turn-speed", type=int, default=18)
    args = parser.parse_args()

    if args.action == "preview":
        for name, axes in (("stop", (0, 0)), ("forward", (0, args.speed)),
                           ("backward", (0, -args.speed)), ("left", (-args.speed, 0)),
                           ("right", (args.speed, 0))):
            print("%s: %s" % (name, encode_command(*axes).hex(" ").upper()))
        return 0

    listen_address = ipaddress.ip_address(args.listen)
    if not listen_address.is_loopback and not args.token:
        parser.error("--token is required when --listen is not a loopback address")
    config = BallConfig(source_ip=args.source_ip, source_port=args.source_port,
                        sta_ip=args.sta_ip, sta_port=args.sta_port, servo=args.servo,
                        turn_speed=args.turn_speed, crc_byteorder=args.crc_byteorder)
    try:
        controller = BallController(config)
    except OSError as exc:
        parser.exit(2, "UDP bind/connect failed: %s\n" % exc)
    with controller:
        server = ThreadingHTTPServer((args.listen, args.http_port), make_http_handler(controller, args.token))
        server.daemon_threads = True
        route = controller.status()
        print("HTTP control at http://%s:%d; UDP source %s -> STA %s" %
              (args.listen, server.server_address[1], route["source"], route["destination"]))
        try:
            server.serve_forever(poll_interval=0.1)
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
