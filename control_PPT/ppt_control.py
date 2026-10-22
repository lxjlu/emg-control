# -*- coding: utf-8 -*-
"""
ppt_control.py
用手势控制 PPT / WPS 翻页。

映射：
    握拳  → 全屏放映（Shift+F5，编辑态生效；已在放映则忽略）
    手掌  → 退出全屏（ESC，放映态生效）
    向左  → 上一页（PageUp）
    向右  → 下一页（PageDown）
    休息 / 低置信度 / 未映射 → 不动作

按键策略（Windows，移植自旧版 control_ppt.py）：
    - 每次发键前查找 PowerPoint / WPS 窗口并切到前台，避免按键落到别的程序
    - 放映中自动用 Space / Backspace（Office、WPS 放映下最稳），编辑态用 PageDown / PageUp
    - 用 keybd_event 发送虚拟键码，比 pynput 在放映模式下更可靠
    - 非 Windows 平台回退为 pynput 发送 ← / →

其他：
    - 一次手势只翻一页：翻页后必须回到「休息」才能再次触发，避免连续跳页
    - 7 帧投票需 5 票 + 连续 2 次确认 + 置信度阈值，抑制误翻页
    - 启动延时，留出时间切换到 PPT 全屏放映

用法：
    python ppt_control.py --delay 8                              # 默认开启置顶浮窗
    python ppt_control.py --delay 8 --hud-corner br              # 浮窗放右下角
    python ppt_control.py --delay 8 --no-hud                     # 关闭浮窗
    python ppt_control.py --diagnose --diagnose-seconds 15       # 只诊断静息是否被判成手势
    python ppt_control.py --delay 8 --dry-run                    # 只打印按键，不真的按
    python ppt_control.py --delay 8 --require-window             # 找不到 PPT 窗口就不发键
    python ppt_control.py --delay 8 --restore-screen             # 退出时按空格恢复黑屏
    python ppt_control.py --csv ../dataset/xxx.csv --dry-run     # 回放验证
"""

import argparse
import asyncio
import ctypes
import gc
import glob
import os
import threading
import time
from collections import deque

try:                    # 只有 Windows 才提供 wintypes，其他平台导入会失败
    import ctypes.wintypes as wt
except Exception:
    wt = None

import joblib
import numpy as np

from armband_ble import SAMPLE_RATE, ArmbandBle
from features import extract
from train_rbf import read_recording_csv

# 手势 → PPT 动作。休息 / 低置信度 / 未映射不产生动作。
#   握拳 → 全屏放映，手掌 → 退出全屏，向左 → 上一页，向右 → 下一页
ACTION_MAP = {
    "握拳": "start", "fist": "start", "全屏": "start", "开始放映": "start", "start": "start",
    "手掌": "exit", "palm": "exit", "退出放映": "exit", "结束放映": "exit", "exit": "exit",
    "向左": "prev", "左": "prev", "左转": "prev", "prev": "prev",
    "向右": "next", "右": "next", "右转": "next", "next": "next",
}
REST_LABELS = {"休息", "rest"}

ACTION_TEXT = {"next": "下一页", "prev": "上一页", "start": "全屏放映", "exit": "退出全屏"}

IS_WINDOWS = os.name == "nt"

# 虚拟键码
VK = {
    "space": 0x20, "backspace": 0x08,
    "pagedown": 0x22, "pageup": 0x21,
    "f5": 0x74, "esc": 0x1B, "shift": 0x10,
}
KEYEVENTF_KEYUP = 0x0002

# 放映态用 Space / Backspace（旧版实测在 Office / WPS 放映下最稳）
SLIDESHOW_KEYS = {"next": "space", "prev": "backspace"}
EDIT_KEYS = {"next": "pagedown", "prev": "pageup"}

PPT_CLASSES = {"screenclass", "pptframeclass", "pptclass", "pptlayerclass"}
PPT_TITLE_HINTS = ("powerpoint", "演示文稿", "幻灯片放映", "wps 演示", "wpp")
SCREEN_TOLERANCE = 10   # 判定全屏时与屏幕尺寸的容差（最大化窗口会差 16 像素，不会被误判）

# 信号“几乎为零”时直接判静息：训练集里最弱的静息也有 11 左右，
# 而平坦信号会让 SVM 落到“未知区”（概率最大的一类常常是手掌），所以单独兜底。
ENERGY_FLOOR = 2.0


class SlideController:
    """把逐帧预测转换为「一次手势翻一页」的动作。"""

    def __init__(self, send, log=print, conf=0.6, vote_window=7,
                 vote_needed=5, vote_min=3, confirm=2, on_action=None):
        self.send = send
        self.log = log
        self.on_action = on_action
        self.conf = conf
        self.vote_window = vote_window
        self.vote_needed = vote_needed
        self.vote_min = vote_min
        self.confirm = confirm

        self.votes = deque(maxlen=vote_window)
        self.neutral = True
        self.candidate = None
        self.candidate_count = 0
        self.count = 0

    def on_prediction(self, label, confidence):
        self.votes.append(label)
        if len(self.votes) < self.vote_min:
            self._handle(label, confidence)
        elif self.votes.count(label) >= self.vote_needed:
            self._handle(label, confidence)

    def _handle(self, label, confidence):
        action = ACTION_MAP.get(label) if confidence >= self.conf else None

        if action is None:
            self.neutral = True
            self.candidate = None
            self.candidate_count = 0
            return

        if not self.neutral:
            return   # 翻页后必须先回休息，避免连续跳页

        if action != self.candidate:
            self.candidate = action
            self.candidate_count = 1
            return

        self.candidate_count += 1
        if self.candidate_count < self.confirm:
            return

        self.candidate = None
        self.candidate_count = 0
        self.neutral = False
        self.count += 1
        self.send(action)
        if self.on_action:
            self.on_action(action, self.count)
        self.log(f">> {ACTION_TEXT.get(action, action)}（第 {self.count} 次）")


HUD_CORNERS = {"tl": (24, 24), "tr": (-324, 24), "bl": (24, -120), "br": (-324, -120)}


class Hud:
    """置顶半透明浮窗：放映时也能看到最近手势、置信度和翻页结果。

    Tk 在独立线程里跑自己的消息循环；主线程只改 `_state` 字典。
    """

    def __init__(self, corner="tl", conf_min=0.48, log=print):
        self.corner = corner if corner in HUD_CORNERS else "tl"
        self.log = log
        self.hwnd = None
        self._state = {"label": "-", "conf": 0.0, "neutral": True,
                       "count": 0, "action": "", "action_time": 0.0,
                       "link": False, "last_data": 0.0, "conf_min": conf_min}
        self._ready = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, **fields):
        self._state.update(fields)

    def wait_ready(self, timeout=3.0):
        self._ready.wait(timeout)

    def close(self):
        self._stop = True
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # 以下都在 HUD 自己的线程里执行
    # ------------------------------------------------------------------
    def _run(self):
        try:
            import tkinter as tk
        except ImportError:
            self.log("  未找到 tkinter，跳过悬浮窗")
            self._ready.set()
            return

        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.92)
        root.configure(bg="#0f1419", highlightthickness=2, highlightbackground="#4aa3df")

        label = tk.Label(root, text="", justify="left", anchor="nw",
                         bg="#0f1419", fg="#e6eef5", font=("Microsoft YaHei", 11),
                         padx=12, pady=8)
        label.pack(fill="both", expand=True)

        width, height = 300, 96
        dx, dy = HUD_CORNERS[self.corner]
        x = dx if dx >= 0 else root.winfo_screenwidth() + dx
        y = dy if dy >= 0 else root.winfo_screenheight() + dy
        root.geometry(f"{width}x{height}+{x}+{y}")
        root.update_idletasks()

        self._pin_on_top(root)
        self._ready.set()

        def tick():
            if self._stop:
                root.destroy()
                return
            label.config(text=self._format(self._state))
            self._pin_on_top(root)      # _activate 会把 PPT 抬到最上层，这里周期性抢回来
            root.after(250, tick)

        root.after(250, tick)
        try:
            root.mainloop()
        except Exception:
            pass
        finally:
            # 必须在创建 Tk 的线程里销毁，否则退出时会报
            # Tcl_AsyncDelete: async handler deleted by the wrong thread
            try:
                label.destroy()
                root.destroy()
            except Exception:
                pass
            del label, root
            gc.collect()

    @staticmethod
    def _format(state):
        now = time.time()
        fresh = now - state["action_time"] < 2.0
        action = state["action"] if fresh and state["action"] else "—"

        conf_min = state.get("conf_min", 0.48)
        if not state["link"]:
            status = "蓝牙未连接"
        elif now - state["last_data"] > 1.5:
            status = "无数据（手环未推流）"
        elif state["conf"] < conf_min:
            status = "置信度不足，不动作"
        else:
            status = "可触发" if state["neutral"] else "已翻页，请回休息"

        stale = (not state["link"]) or (now - state["last_data"] > 1.5)
        if stale:
            recent = "—   --"
        elif state["conf"] < conf_min:
            recent = f"未确定（{state['label']} {state['conf']:.2f}）"
        else:
            recent = f"{state['label']}   {state['conf']:.2f}"
        return (f"手势控制   {status}\n"
                f"最近 {recent}\n"
                f"动作 {action}    共 {state['count']} 次")

    def _pin_on_top(self, root):
        if not IS_WINDOWS:
            return
        try:
            user32 = ctypes.windll.user32
            if self.hwnd is None:
                self.hwnd = user32.GetParent(root.winfo_id()) or root.winfo_id()
                GWL_EXSTYLE = -20
                WS_EX_NOACTIVATE = 0x08000000
                WS_EX_TOOLWINDOW = 0x00000080
                style = user32.GetWindowLongW(self.hwnd, GWL_EXSTYLE)
                user32.SetWindowLongW(self.hwnd, GWL_EXSTYLE,
                                      style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
            HWND_TOPMOST = -1
            SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = 0x0001, 0x0002, 0x0010
            user32.SetWindowPos(self.hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                                SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE)
        except Exception:
            pass


class PptSender:
    """把 next / prev / start / exit 变成真实的 PPT 按键。

    Windows 下会先找到 PPT / WPS 窗口并切到前台，再按放映态或编辑态选择按键；
    其他平台直接发 ← / →。
    """

    def __init__(self, dry_run=False, focus=True, require_window=False,
                 restore_screen=False, log=print):
        self.dry_run = dry_run
        self.focus = focus
        self.require_window = require_window
        self.restore_screen = restore_screen
        self.log = log
        self._keyboard = None
        self._last_state = None

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def __call__(self, action):
        if self.dry_run:
            self.log(f"[按键] {ACTION_TEXT.get(action, action)}")
            return
        if action not in ACTION_TEXT:
            return
        if IS_WINDOWS:
            self._send_windows(action)
        else:
            self._send_portable(action)

    def close(self):
        """退出时按一次空格，尝试把黑屏/白屏的放映画面恢复正常。"""
        if self.dry_run or not self.restore_screen:
            return
        try:
            if IS_WINDOWS:
                hwnd = self._find_window()
                if hwnd is None:
                    return
                if self.focus:
                    self._activate(hwnd)
                self._press("space")
            else:
                self._keyboard_controller().tap(self._portable_key("space"))
            self.log("  已按一次空格（若放映处于黑屏/白屏会恢复正常画面）")
        except Exception as error:
            self.log(f"  恢复画面失败：{error}")

    # ------------------------------------------------------------------
    # Windows 实现
    # ------------------------------------------------------------------
    def _send_windows(self, action):
        hwnd = self._find_window()
        if hwnd is None:
            if self.require_window:
                self.log("  未找到 PPT/WPS 窗口，本次按键已跳过（--require-window）")
                return
        else:
            if self.focus:
                self._activate(hwnd)
            self._note_state(hwnd)

        slideshow = self._is_fullscreen(hwnd) if hwnd else False

        if action == "start":
            if not slideshow:      # 已在放映就不重复 Shift+F5，避免从头再放
                self._press_combo(["shift", "f5"])
            return
        if action == "exit":
            if slideshow:
                self._press("esc")
            return

        self._press(SLIDESHOW_KEYS[action] if slideshow else EDIT_KEYS[action])

    def _note_state(self, hwnd):
        """窗口或放映状态变化时打印一次，便于现场确认焦点没跑。"""
        state = (self._window_title(hwnd), self._is_fullscreen(hwnd))
        if state == self._last_state:
            return
        self._last_state = state
        self.log(f"  PPT 窗口：{state[0][:30] or '(无标题)'} · "
                 f"{'放映中' if state[1] else '编辑态'}")

    @staticmethod
    def _user32():
        return ctypes.windll.user32

    def _window_title(self, hwnd):
        user32 = self._user32()
        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        return buffer.value

    def _class_name(self, hwnd):
        buffer = ctypes.create_unicode_buffer(256)
        self._user32().GetClassNameW(hwnd, buffer, 256)
        return buffer.value

    def _is_fullscreen(self, hwnd):
        """窗口尺寸与屏幕一致即认为在放映（最大化编辑窗口会差 16 像素，不会误判）。"""
        rect = wt.RECT()
        user32 = self._user32()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        width = user32.GetSystemMetrics(0)
        height = user32.GetSystemMetrics(1)
        return (abs((rect.right - rect.left) - width) < SCREEN_TOLERANCE
                and abs((rect.bottom - rect.top) - height) < SCREEN_TOLERANCE)

    def _find_window(self):
        """查找 PowerPoint / WPS 窗口，优先前台窗口，其次全屏放映窗口。"""
        user32 = self._user32()
        candidates = []

        @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
        def _callback(hwnd, _lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            title = self._window_title(hwnd).lower()
            class_name = self._class_name(hwnd).lower()
            if class_name in PPT_CLASSES or any(hint in title for hint in PPT_TITLE_HINTS):
                candidates.append(hwnd)
            return True

        user32.EnumWindows(_callback, 0)
        if not candidates:
            return None

        foreground = user32.GetForegroundWindow()
        if foreground in candidates:
            return foreground
        for hwnd in candidates:
            if self._is_fullscreen(hwnd):
                return hwnd
        return candidates[0]

    def _activate(self, hwnd):
        """把窗口切到前台。Windows 限制后台进程抢焦点，用 AttachThreadInput 绕过。"""
        user32 = self._user32()
        try:
            if user32.GetForegroundWindow() == hwnd:
                return
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, 9)      # SW_RESTORE，只对最小化窗口做
            user32.BringWindowToTop(hwnd)
            foreground = user32.GetForegroundWindow()
            target_thread = user32.GetWindowThreadProcessId(foreground, None)
            own_thread = ctypes.windll.kernel32.GetCurrentThreadId()
            if target_thread and target_thread != own_thread:
                user32.AttachThreadInput(target_thread, own_thread, True)
                user32.SetForegroundWindow(hwnd)
                user32.AttachThreadInput(target_thread, own_thread, False)
            else:
                user32.SetForegroundWindow(hwnd)
            time.sleep(0.05)    # 留出切窗口时间，避免按键丢失
        except Exception as error:
            self.log(f"  窗口切换失败（仍继续发键）：{error}")

    def _press(self, name, hold=0.03):
        user32 = self._user32()
        code = VK[name]
        user32.keybd_event(code, 0, 0, 0)
        time.sleep(hold)
        user32.keybd_event(code, 0, KEYEVENTF_KEYUP, 0)

    def _press_combo(self, names):
        user32 = self._user32()
        for name in names:
            user32.keybd_event(VK[name], 0, 0, 0)
            time.sleep(0.02)
        for name in reversed(names):
            user32.keybd_event(VK[name], 0, KEYEVENTF_KEYUP, 0)
            time.sleep(0.02)

    # ------------------------------------------------------------------
    # 非 Windows：退回 pynput
    # ------------------------------------------------------------------
    def _keyboard_controller(self):
        if self._keyboard is None:
            try:
                from pynput.keyboard import Controller
            except ImportError as error:
                raise SystemExit("缺少 pynput，请先执行：pip install pynput") from error
            self._keyboard = Controller()
        return self._keyboard

    @staticmethod
    def _portable_key(name):
        from pynput.keyboard import Key

        return {
            "next": Key.right, "prev": Key.left, "start": Key.f5,
            "exit": Key.esc, "space": Key.space,
        }[name]

    def _send_portable(self, action):
        self._keyboard_controller().tap(self._portable_key(action))


def make_sender(dry_run, focus=True, require_window=False, restore_screen=False):
    return PptSender(dry_run=dry_run, focus=focus, require_window=require_window,
                     restore_screen=restore_screen)


def window_energy(window):
    """窗口平均幅值（相对 2048 基线），用来识别“几乎没信号”的静息态。"""
    return float(np.mean(np.abs(np.asarray(window, dtype=np.float64) - 2048)))


def predict(payload, buffer):
    window = list(buffer)[-payload["window"]:]
    if window_energy(window) < ENERGY_FLOOR:
        return "休息", 1.0          # 信号几乎为零，不可能是手势
    features = np.asarray([extract(window)], dtype=np.float64)
    probabilities = payload["model"].predict_proba(features)[0]
    index = int(probabilities.argmax())
    return payload["labels"][index], float(probabilities[index])


async def run_live(args, payload):
    sender = make_sender(args.dry_run, focus=not args.no_focus,
                         require_window=args.require_window,
                         restore_screen=args.restore_screen)

    hud = Hud(args.hud_corner, conf_min=args.conf) if args.hud else None

    def on_action(action, count):
        if hud:
            hud.update(action=ACTION_TEXT.get(action, action), count=count,
                       action_time=time.time())

    controller = SlideController(
        send=sender, conf=args.conf, on_action=on_action,
        vote_window=args.vote_window, vote_needed=args.vote_needed, confirm=args.confirm,
    )
    buffer = deque(maxlen=payload["window"])
    pending = 0

    def on_samples(samples):
        nonlocal pending
        if not samples:
            return
        buffer.extend(samples)          # deque(maxlen=window)，自动丢弃最旧样本
        pending += len(samples)
        if pending < payload["hop"] or len(buffer) < payload["window"]:
            return
        # 一次通知最多预测一次：BLE 偶发卡顿后可能一次送来几百个样本，
        # 若按 hop 逐个补算，会连发几十次相同的预测，投票瞬间被灌满 → 动作扎堆。
        pending %= payload["hop"]
        label, confidence = predict(payload, buffer)
        controller.on_prediction(label, confidence)
        if hud:
            hud.update(label=label, conf=confidence, neutral=controller.neutral,
                       last_data=time.time())

    arm = ArmbandBle(address=args.address, on_samples=on_samples, on_status=print)
    await arm.connect()

    print()
    print("手势映射：握拳 = 全屏放映，手掌 = 退出全屏，向左 = 上一页，"
          "向右 = 下一页，休息 = 不动")
    if hud:
        print(f"置顶浮窗已开启（{args.hud_corner}），放映时也能看到识别结果；关闭用 --no-hud")
    if IS_WINDOWS and not args.dry_run:
        print(f"发键前会自动切到 PPT 窗口（{'已启用' if not args.no_focus else '已关闭'}），"
              f"放映态用 Space/Backspace，编辑态用 PageDown/PageUp")
    if args.delay > 0:
        print(f"请在 {args.delay} 秒内切换到 PPT 并开始放映（F5）…")
        for sec in range(int(args.delay), 0, -1):
            print(f"  {sec} …", end="\r", flush=True)
            await asyncio.sleep(1)
    print("开始接受手势。Ctrl+C 退出。          ")

    try:
        while True:
            if not arm.is_connected:
                print("连接断开，正在重连…")
                if await arm.reconnect():
                    print("重连成功，继续接收手势。")
                else:
                    print("重连失败，10 秒后重试。")
                    await asyncio.sleep(10)
            if hud:
                hud.update(link=arm.is_connected, last_data=arm.last_notification_at)
            await asyncio.sleep(0.05)
    finally:
        await arm.disconnect()
        sender.close()
        if hud:
            hud.close()
        print("\n连接统计：" + arm.stats())
        print(f"共翻页 {controller.count} 次")


async def run_replay(args, payload):
    label, user, hand, rows = read_recording_csv(args.csv)
    if not rows:
        raise RuntimeError(f"CSV 中没有样本：{args.csv}")
    print(f"回放：{os.path.basename(args.csv)}  录制标签：{label or '未知'}  "
          f"{len(rows)} 样本（{len(rows) / SAMPLE_RATE:.2f} 秒）")

    sender = make_sender(args.dry_run, focus=not args.no_focus,
                         require_window=args.require_window)
    controller = SlideController(
        send=sender, conf=args.conf,
        vote_window=args.vote_window, vote_needed=args.vote_needed, confirm=args.confirm,
    )
    buffer = deque(maxlen=payload["window"])
    hop = payload["hop"]

    for start in range(0, len(rows), hop):
        for row in rows[start:start + hop]:
            buffer.append(row)
        if len(buffer) >= payload["window"]:
            predicted, confidence = predict(payload, buffer)
            controller.on_prediction(predicted, confidence)
        if args.speed > 0:
            await asyncio.sleep(hop / SAMPLE_RATE / args.speed)

    print(f"回放结束，共翻页 {controller.count} 次")


def dataset_energy(data_dir, label="休息"):
    """统计训练集某个类别的平均幅值，用来和实时数据对比。"""
    values = []
    for path in glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True):
        current, _user, _hand, rows = read_recording_csv(path)
        if current != label or not rows:
            continue
        array = np.asarray(rows, dtype=np.float64)
        values.append(float(np.mean(np.abs(array - 2048))))
    if not values:
        return 0.0, 0
    return float(np.mean(values)), len(values)


async def diagnose_live(args, payload):
    """不发送任何按键，只打印实时预测和信号能量，用于定位“静息被判成手势”。

    每秒由主循环打印一次（不是靠数据回调触发），这样没数据、掉线时也能看到状态；
    掉线会自动重连，避免一次断开就整段诊断作废。
    """
    arm = ArmbandBle(address=args.address, on_status=print)
    buffer = deque(maxlen=payload["window"])
    pending = 0
    counts = {}
    confs = []
    energies = []
    samples_total = 0

    def on_samples(samples):
        nonlocal pending, samples_total
        if not samples:
            return
        samples_total += len(samples)
        array = np.asarray(samples, dtype=np.float64)
        energies.append(float(np.mean(np.abs(array - 2048))))
        buffer.extend(samples)
        pending += len(samples)
        if pending >= payload["hop"] and len(buffer) >= payload["window"]:
            pending %= payload["hop"]
            label, confidence = predict(payload, buffer)
            counts[label] = counts.get(label, 0) + 1
            confs.append(confidence)

    raw_dumps = []

    def on_raw(data):
        if len(raw_dumps) < 3:
            raw_dumps.append((len(data), data[:96].hex(" ")))

    arm.on_raw = on_raw
    await arm.connect()
    print(f"\n开始诊断 {args.diagnose_seconds:.0f} 秒：请保持手臂静止、不要做任何动作。")

    last_notes = 0
    last_bytes = 0
    last_samples = 0
    try:
        for second in range(1, int(args.diagnose_seconds) + 1):
            await asyncio.sleep(1)
            notes = arm.notifications - last_notes
            bytes_now = arm.bytes_received - last_bytes
            samples_now = samples_total - last_samples
            last_notes, last_bytes, last_samples = (
                arm.notifications, arm.bytes_received, samples_total)

            top = sorted(counts.items(), key=lambda item: -item[1])[:3]
            summary = "、".join(f"{name}{count}" for name, count in top) or "无"
            print(f"  {second:2d}s 连接={'是' if arm.is_connected else '否'}  "
                  f"通知 {notes:3d}/s  样本 {samples_now:4d}/s  字节 {bytes_now:6d}/s  "
                  f"丢弃 {arm.parser.discarded_bytes:7d}B  格式错 {arm.parser.format_errors}  "
                  f"预测 {summary:<20} 能量 {np.mean(energies) if energies else 0:5.1f}")
            counts.clear()
            confs.clear()
            energies.clear()

            if raw_dumps and second == 1:
                for index, (length, dump) in enumerate(raw_dumps, 1):
                    print(f"     原始包{index}（长度 {length}，前 96 字节）: {dump}")

            if not arm.is_connected:
                print("     连接断开，尝试重连…")
                if await arm.reconnect():
                    print("     重连成功，继续诊断（数据应自动恢复，无需重按 SW2）。")
                    last_notes = arm.notifications
                    last_bytes = arm.bytes_received
                    last_samples = samples_total
                else:
                    print("     重连失败，下一轮再试。")
    finally:
        await arm.disconnect()

    rest_energy, rest_files = dataset_energy(args.data, "休息")
    print(f"\n参考：训练集「休息」平均能量 {rest_energy:.1f}（{rest_files} 个文件）")
    print(f"本次共收到 {samples_total} 个样本，CRC 错误 {arm.parser.crc_errors}，"
          f"格式错误 {arm.parser.format_errors}")
    if samples_total == 0:
        print("一个样本都没收到：手环连上了但没有推流，请长按 SW2 3 秒启动采集后重试。")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="手势控制 PPT 翻页")
    parser.add_argument("--model", default=os.path.join(here, "rbf_model.joblib"),
                        help="模型文件（先运行 collect_and_train.py）")
    parser.add_argument("--address", default=None, help="BLE 地址；省略则自动扫描")
    parser.add_argument("--delay", type=float, default=8.0,
                        help="启动后等待多少秒开始接受手势（默认 8）")
    parser.add_argument("--dry-run", action="store_true", help="只打印按键，不真的发送")
    parser.add_argument("--csv", default=None, help="回放模式：指定 CSV 验证翻页逻辑")
    parser.add_argument("--speed", type=float, default=1.0, help="回放速度倍率，0=不限速")
    parser.add_argument("--conf", type=float, default=0.6, help="置信度阈值")
    parser.add_argument("--vote-window", type=int, default=7, help="投票窗口")
    parser.add_argument("--vote-needed", type=int, default=5, help="所需票数")
    parser.add_argument("--confirm", type=int, default=2, help="动作确认次数")
    parser.add_argument("--no-focus", action="store_true",
                        help="不自动把 PPT 窗口切到前台（默认会切）")
    parser.add_argument("--require-window", action="store_true",
                        help="找不到 PPT/WPS 窗口时不发送按键（防止按键落到别的程序）")
    parser.add_argument("--restore-screen", action="store_true",
                        help="退出时按一次空格，尝试恢复黑屏/白屏的放映画面")
    parser.add_argument("--hud", dest="hud", action="store_true", default=True,
                        help="显示置顶浮窗（默认开启）")
    parser.add_argument("--no-hud", dest="hud", action="store_false",
                        help="关闭置顶浮窗")
    parser.add_argument("--hud-corner", default="tl", choices=["tl", "tr", "bl", "br"],
                        help="浮窗位置：tl 左上 / tr 右上 / bl 左下 / br 右下（默认 tl）")
    parser.add_argument("--data", default=os.path.join(here, "dataset"),
                        help="训练数据目录（诊断时用来对比静息能量）")
    parser.add_argument("--diagnose", action="store_true",
                        help="只诊断：打印实时预测与能量，不发任何按键")
    parser.add_argument("--diagnose-seconds", type=float, default=12.0,
                        help="诊断时长（默认 12 秒）")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"模型不存在：{args.model}\n请先运行：python collect_and_train.py")

    payload = joblib.load(args.model)
    print(f"模型类别：{'、'.join(payload['labels'])}")
    print(f"窗口 {payload['window']} · 步长 {payload['hop']} · "
          f"投票 {args.vote_window}/{args.vote_needed} · 置信度 ≥ {args.conf}")

    try:
        if args.diagnose:
            asyncio.run(diagnose_live(args, payload))
        elif args.csv:
            asyncio.run(run_replay(args, payload))
        else:
            asyncio.run(run_live(args, payload))
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    main()
