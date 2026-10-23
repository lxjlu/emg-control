# -*- coding: utf-8 -*-
"""肌电波形监控：接收 App 推送的 stream 报文，用浏览器看 8 通道实时波形。

    python -m emg_sdk run --adapter ppt --port 8888 --monitor 8080
    python -m emg_sdk monitor --port 8888 --http-port 8080

HTTP 接口：
    GET /                 波形页面（自带 Canvas 绘制，无外部依赖）
    GET /api/emg?since=N  返回全局序号 N 之后的样本
"""

import json
import socket
import threading
import time
from array import array
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .adapters.base import DeviceAdapter

MAX_RETURN = 2000          # 单次接口最多返回的样本数
RING_SECONDS = 16          # 服务端保留的波形时长


class EmgStore:
    """8 通道环形缓冲，按全局样本序号取增量。"""

    def __init__(self, channels=8, sample_rate=500):
        self.channels = channels
        self.capacity = max(MAX_RETURN, int(sample_rate * RING_SECONDS))
        self._rings = [array("h", [0]) * self.capacity for _ in range(channels)]
        self._head = 0
        self._total = 0
        self._lock = threading.Lock()

        self.rate = float(sample_rate)
        self.measured_rate = 0.0
        self.packets = 0
        self.samples = 0
        self.last_rx = 0.0
        self._rate_t0 = time.time()
        self._rate_samples = 0

    # ------------------------------------------------------------------
    def push(self, samples, rate, channels):
        with self._lock:
            if channels != self.channels:
                self.channels = channels
                self._rings = [array("h", [0]) * self.capacity for _ in range(channels)]
                self._head = 0
                self._total = 0
            self.rate = float(rate)

            for sample in samples:
                for index in range(self.channels):
                    value = sample[index] if index < len(sample) else 0
                    if value > 32767:
                        value = 32767
                    elif value < -32768:
                        value = -32768
                    self._rings[index][self._head] = value
                self._head = (self._head + 1) % self.capacity
                self._total += 1

            self.packets += 1
            self.samples += len(samples)
            self.last_rx = time.time()

            elapsed = self.last_rx - self._rate_t0
            if elapsed >= 1.0:
                self.measured_rate = (self.samples - self._rate_samples) / elapsed
                self._rate_t0 = self.last_rx
                self._rate_samples = self.samples

    # ------------------------------------------------------------------
    def snapshot(self, since=0, limit=MAX_RETURN):
        with self._lock:
            total = self._total
            oldest = max(0, total - self.capacity)
            start = max(int(since), oldest, total - limit)
            head = self._head

            rows = []
            for index in range(start, total):
                position = (head - (total - index)) % self.capacity
                rows.append([self._rings[channel][position]
                             for channel in range(self.channels)])

            return {
                "total": total,
                "from": start,
                "channels": self.channels,
                "rate": round(self.measured_rate or self.rate, 1),
                "samples": rows,
                "stats": {
                    "packets": self.packets,
                    "samples": self.samples,
                    "connected": bool(self.last_rx) and (time.time() - self.last_rx) < 1.5,
                },
            }


class _Handler(BaseHTTPRequestHandler):
    store = None
    server_version = "EmgMonitor/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send("text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif parsed.path == "/api/emg":
            query = parse_qs(parsed.query)
            try:
                since = int(query.get("since", ["0"])[0])
            except ValueError:
                since = 0
            body = json.dumps(self.store.snapshot(since), separators=(",", ":"))
            self._send("application/json", body.encode("utf-8"))
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def _send(self, content_type, body):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt, *args):
        pass


class MonitorAdapter(DeviceAdapter):
    """`emg-sdk monitor` 用的空适配器：只收波形，不驱动任何设备。"""

    name = "monitor"
    display = "肌电监控"
    continuous = False
    publish_hz = 1.0
    deadman_ms = 500

    def capabilities(self):
        caps = super().capabilities()
        caps.update({"actions": [], "display": self.display})
        return caps

    def apply(self, action, args):
        self.log(f"监控模式：忽略动作 {action}")


class EmgMonitor:
    def __init__(self, http_port=8080, host="0.0.0.0", log=print):
        self.store = EmgStore()
        self.http_port = int(http_port)
        self.host = host
        self.log = log
        self._httpd = None
        self._thread = None
        self._last_summary = time.time()
        self._last_packets = 0

    def start(self):
        handler = type("BoundHandler", (_Handler,), {"store": self.store})
        self._httpd = ThreadingHTTPServer((self.host, self.http_port), handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="emg-monitor", daemon=True)
        self._thread.start()

        address = _guess_lan_ip()
        suffix = f"  或 http://{address}:{self.http_port}" if address not in ("", "127.0.0.1") else ""
        self.log(f"肌电波形监控：http://127.0.0.1:{self.http_port}{suffix}")
        return self

    def stop(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    def push(self, samples, rate, channels):
        self.store.push(samples, rate, channels)
        now = time.time()
        if now - self._last_summary >= 2.0:
            self._last_summary = now
            packets = self.store.packets
            delta = packets - self._last_packets
            self._last_packets = packets
            self.log(f"肌电流 {delta / 2.0:.0f} 包/s · "
                     f"{self.store.measured_rate:.0f} 样本/s · "
                     f"{self.store.channels} 通道")


def _guess_lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"


PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sEMG 8 通道实时波形</title>
<style>
  :root{--bg:#0b1116;--panel:#111c24;--line:#1d2b35;--ink:#e6f1f7;
        --muted:#7b909d;--ok:#38d39f;--warn:#ffb020;--bad:#ff5f7e;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:14px/1.5 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
  header{display:flex;flex-wrap:wrap;align-items:center;gap:20px;
         padding:12px 18px;border-bottom:1px solid var(--line);background:var(--panel)}
  h1{font-size:15px;font-weight:600;margin:0;letter-spacing:.4px}
  .stat{display:flex;align-items:baseline;gap:6px;color:var(--muted);font-size:12px}
  .stat b{color:var(--ink);font-size:15px;font-weight:600;font-variant-numeric:tabular-nums}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--bad);display:inline-block}
  .dot.on{background:var(--ok);box-shadow:0 0 8px var(--ok)}
  #wrap{position:relative;padding:10px 12px 16px}
  canvas{width:100%;height:calc(100vh - 92px);display:block;border-radius:10px;
         background:var(--panel);border:1px solid var(--line)}
  #empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
         color:var(--muted);font-size:13px;pointer-events:none}
  button{background:#16232c;color:var(--ink);border:1px solid var(--line);
         border-radius:8px;padding:6px 12px;cursor:pointer;font-size:12px}
  button:hover{border-color:#2f4653}
</style>
</head>
<body>
<header>
  <h1>sEMG 8 通道实时波形</h1>
  <div class="stat"><span>采样率</span><b id="rate">--</b><span>sample/s</span></div>
  <div class="stat"><span>样本</span><b id="count">0</b></div>
  <div class="stat"><span>数据包</span><b id="packets">0</b></div>
  <div class="stat"><span class="dot" id="dot"></span><b id="state">等待数据</b></div>
  <div style="margin-left:auto"><button id="toggle">暂停</button></div>
</header>
<div id="wrap">
  <canvas id="wave"></canvas>
  <div id="empty">等待 App 推送肌电数据…</div>
</div>
<script>
const CH = 8;
const COLORS = ['#4cc9f0','#f7b32b','#f72585','#7ae582','#b388eb','#ff8fa3','#3ddc97','#8ecae6'];
const MAX_ROWS = 2500;

const canvas = document.getElementById('wave');
const ctx = canvas.getContext('2d');
const empty = document.getElementById('empty');
const elRate = document.getElementById('rate');
const elCount = document.getElementById('count');
const elPackets = document.getElementById('packets');
const elState = document.getElementById('state');
const elDot = document.getElementById('dot');
const toggle = document.getElementById('toggle');

let total = 0;
let rows = [];
let paused = false;
let timer = null;

async function poll() {
  if (paused) return;
  try {
    const response = await fetch('/api/emg?since=' + total, {cache: 'no-store'});
    const data = await response.json();
    if (data.total < total) rows = [];
    total = data.total;
    if (data.samples && data.samples.length) {
      rows.push(...data.samples);
      if (rows.length > MAX_ROWS) rows.splice(0, rows.length - MAX_ROWS);
    }
    elRate.textContent = data.rate ? data.rate.toFixed(0) : '--';
    elCount.textContent = data.stats.samples;
    elPackets.textContent = data.stats.packets;
    const live = data.stats.connected && data.samples && data.samples.length > 0;
    elDot.className = 'dot' + (live ? ' on' : '');
    elState.textContent = live ? '接收中' : '等待数据';
    empty.style.display = rows.length ? 'none' : 'flex';
    draw(data.channels || CH);
  } catch (error) {
    elDot.className = 'dot';
    elState.textContent = '连接中断';
  }
}

function draw(channels) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const lane = height / channels;
  ctx.lineWidth = 1;
  ctx.font = '11px ui-monospace, Consolas, monospace';
  ctx.textBaseline = 'middle';

  for (let ch = 0; ch < channels; ch++) {
    const top = lane * ch;
    const center = top + lane / 2;

    ctx.strokeStyle = '#1d2b35';
    ctx.beginPath();
    ctx.moveTo(0, top + lane);
    ctx.lineTo(width, top + lane);
    ctx.stroke();

    ctx.fillStyle = COLORS[ch % COLORS.length];
    ctx.fillText('CH' + (ch + 1), 8, top + 14);

    if (!rows.length) continue;

    let peak = 1;
    for (let i = 0; i < rows.length; i++) {
      const v = Math.abs(rows[i][ch] || 0);
      if (v > peak) peak = v;
    }
    const scale = Math.max(200, peak * 1.25);
    const gain = (lane * 0.42) / scale;

    ctx.strokeStyle = COLORS[ch % COLORS.length];
    ctx.lineWidth = 1.1;
    ctx.beginPath();
    for (let x = 0; x < width; x++) {
      const index = Math.min(rows.length - 1, Math.floor(x * rows.length / width));
      const value = rows[index][ch] || 0;
      const y = center - value * gain;
      if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

function restart() {
  if (timer) clearInterval(timer);
  timer = setInterval(poll, 100);
}

toggle.addEventListener('click', () => {
  paused = !paused;
  toggle.textContent = paused ? '继续' : '暂停';
  if (!paused) poll();
});

restart();
poll();
window.addEventListener('resize', () => draw(CH));
</script>
</body>
</html>
"""
