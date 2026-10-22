# -*- coding: utf-8 -*-
"""PPT / WPS 放映适配器：把语义命令翻译成键盘按键。

    next  → 放映中 Space / 编辑态 PageDown
    prev  → 放映中 Backspace / 编辑态 PageUp
    start → Shift+F5（已在放映则忽略）
    exit  → ESC（仅在放映态发送）

不依赖 PowerPoint COM：发键前先找到 PPT/WPS 窗口并切到前台，再用 keybd_event
注入虚拟键码，Office 与 WPS 都能响应；非 Windows 平台退回 pynput。
"""

import ctypes
import os
import sys
import time

from .base import DeviceAdapter

IS_WINDOWS = os.name == "nt"

try:                    # 只有 Windows 才提供 wintypes
    import ctypes.wintypes as wt
except Exception:
    wt = None

VK = {
    "space": 0x20, "backspace": 0x08,
    "pagedown": 0x22, "pageup": 0x21,
    "f5": 0x74, "esc": 0x1B, "shift": 0x10,
}
KEYEVENTF_KEYUP = 0x0002

SLIDESHOW_KEYS = {"next": "space", "prev": "backspace"}
EDIT_KEYS = {"next": "pagedown", "prev": "pageup"}

PPT_CLASSES = {"screenclass", "pptframeclass", "pptclass", "pptlayerclass"}
PPT_TITLE_HINTS = ("powerpoint", "演示文稿", "幻灯片放映", "wps 演示", "wpp")
SCREEN_TOLERANCE = 10

DISCRETE_ACTIONS = ("next", "prev", "start", "exit")


class PptAdapter(DeviceAdapter):
    name = "ppt"
    display = "PPT / WPS 放映"
    continuous = False          # 离散动作，不参与心跳
    publish_hz = 10.0
    deadman_ms = 0

    def __init__(self, focus=True, require_window=False, restore_screen=False,
                 dry_run=False, log=print):
        super().__init__(dry_run=dry_run, log=log)
        self.focus = focus
        self.require_window = require_window
        self.restore_screen = restore_screen
        self.hwnd = None
        self._keyboard = None
        self._last_state = None

    # ------------------------------------------------------------------
    def capabilities(self):
        caps = super().capabilities()
        caps["actions"] = list(DISCRETE_ACTIONS)
        caps["deadman_ms"] = 0
        caps["platform"] = ("windows" if IS_WINDOWS
                            else "macos" if sys.platform == "darwin" else "linux")
        return caps

    def start(self):
        mode = "Windows keybd_event" if IS_WINDOWS else "pynput"
        self.log(f"{self.display} 就绪（{mode}）")
        if not IS_WINDOWS:
            self.log("提示：macOS 需要在「系统设置 → 隐私与安全性 → 辅助功能」里允许终端")

    # ------------------------------------------------------------------
    def apply(self, action, args):
        if action not in DISCRETE_ACTIONS:
            return
        if self.dry_run:
            self.log(f"[ppt] 按键 {action}")
            return
        if IS_WINDOWS:
            self._send_windows(action)
        else:
            self._send_portable(action)

    def stop(self):
        """PPT 的 stop 不做任何事：退出放映属于显式的 exit 动作。"""

    def close(self):
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
            self.log("已按一次空格（若放映处于黑屏/白屏会恢复正常画面）")
        except Exception as error:
            self.log(f"恢复画面失败：{error}")

    # ------------------------------------------------------------------
    # Windows 实现
    # ------------------------------------------------------------------
    def _send_windows(self, action):
        hwnd = self._find_window()
        if hwnd is None:
            if self.require_window:
                self.log("未找到 PPT/WPS 窗口，本次按键已跳过（--require-window）")
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
        state = (self._window_title(hwnd), self._is_fullscreen(hwnd))
        if state == self._last_state:
            return
        self._last_state = state
        self.log(f"PPT 窗口：{state[0][:30] or '(无标题)'} · "
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
        rect = wt.RECT()
        user32 = self._user32()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        width = user32.GetSystemMetrics(0)
        height = user32.GetSystemMetrics(1)
        return (abs((rect.right - rect.left) - width) < SCREEN_TOLERANCE
                and abs((rect.bottom - rect.top) - height) < SCREEN_TOLERANCE)

    def _find_window(self):
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
            time.sleep(0.05)
        except Exception as error:
            self.log(f"窗口切换失败（仍继续发键）：{error}")

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
    # 非 Windows：pynput
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
            "next": Key.right, "prev": Key.left,
            "start": Key.f5, "exit": Key.esc, "space": Key.space,
        }[name]

    def _send_portable(self, action):
        self._keyboard_controller().tap(self._portable_key(action))
