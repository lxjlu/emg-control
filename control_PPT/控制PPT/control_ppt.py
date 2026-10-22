#----------------------------------------------------------------
# 参数	说明	默认值
# --th1	动作激活上限阈值	90000
# --th2	动作释放下限阈值	89999
# --vote	投票窗口长度（连续 N 次相同才识别）	10
# --cooldown	同一命令冷却秒数	1.0
#----------------------------------------------------------------
import socket
import struct
import numpy as np
from scipy.signal import butter, filtfilt
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import serial
import os
import argparse
import logging
import pyautogui  # 新增：模拟键盘控制PPT
import win32gui
import win32api
import win32con
from pynput.keyboard import Key, Controller as KeyboardController
# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------
# 命令行参数配置
# ----------------------
parser = argparse.ArgumentParser(description="EMG 手势控制PPT翻页（可配置灵敏度）")
parser.add_argument('--th1', type=int, default=90000, help='动作激活阈值 (默认 90000)')
parser.add_argument('--th2', type=int, default=89999, help='动作释放阈值 (默认 89999)')
parser.add_argument('--vote', type=int, default=10, help='投票窗口长度 (默认10)')
parser.add_argument('--cooldown', type=float, default=1.5, help='命令重复冷却秒 (默认1.5，全屏模式建议增加)')
parser.add_argument('--debug', action='store_true', help='启用调试模式，输出详细信息')
args = parser.parse_args()

# 根据调试模式调整日志级别
if args.debug:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------
# 核心配置（删除小车UDP配置）
# ----------------------
SERIAL_PORT = 'COM5'
BAUDRATE = 115200
REAL_TIME_MONITOR = True
MODEL_PATH = 'best_lx_model.pth'
config = {
    "device": torch.device("cpu")
}

# PPT快捷键映射（手势0/1/2/3对应）
# 根据PPT模式自动选择合适的快捷键
PPT_SHORTCUT_NORMAL = {      # 非全屏编辑模式
    0: ["shift", "f5"],   # 当前页全屏
    1: "esc",             # 退出全屏
    2: "pageup",          # 上一页
    3: "pagedown"         # 下一页
}

PPT_SHORTCUT_SLIDESHOW = {   # 全屏放映模式
    0: ["shift", "f5"],   # 当前页全屏（已经全屏时无效）
    1: "esc",             # 退出全屏
    2: "backspace",       # 上一页（全屏模式Backspace更可靠）
    3: "space"            # 下一页（全屏模式Space更可靠）
}

# 键盘控制器（使用pynput替代pyautogui，全屏模式更稳定）
keyboard = KeyboardController()

# ----------------------
# 模型定义（完全不变）
# ----------------------
class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(8, 64, kernel_size=10, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.AvgPool1d(5)
        self.conv2 = nn.Conv1d(64, 96, kernel_size=5, padding=1)
        self.bn2 = nn.BatchNorm1d(96)
        self.conv3 = nn.Conv1d(96, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc = nn.Linear(2048, 4)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x).view(x.size(0), -1)
        return self.fc(x)

# ----------------------
# 工具函数（修改send_command为PPT按键）
# ----------------------
def check_port_exists(port_path):
    return os.path.exists(port_path)

def all_same(lst):
    return all(x == lst[0] for x in lst)

def list_all_windows():
    """列出所有可见窗口（用于调试）"""
    def callback(handle, extra):
        if win32gui.IsWindowVisible(handle):
            window_text = win32gui.GetWindowText(handle)
            class_name = win32gui.GetClassName(handle)
            rect = win32gui.GetWindowRect(handle)
            w = rect[2] - rect[0]
            h = rect[3] - rect[1]
            is_foreground = (handle == win32gui.GetForegroundWindow())
            extra.append({
                'hwnd': handle,
                'title': window_text,
                'class': class_name,
                'width': w,
                'height': h,
                'is_foreground': is_foreground
            })
        return True
    
    windows = []
    win32gui.EnumWindows(callback, windows)
    return windows

def find_ppt_window():
    """查找PPT窗口并返回句柄（支持全屏放映模式）"""
    screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
    
    # 调试模式：列出所有窗口
    if args.debug:
        all_windows = list_all_windows()
        print("\n=== 所有可见窗口 ===")
        for w in all_windows:
            fg = "*" if w['is_foreground'] else " "
            print(f"{fg} HWND: {w['hwnd']:08X}, 标题: '{w['title']}', 类名: {w['class']}, 大小: {w['width']}x{w['height']}")
        print("=== 窗口列表结束 ===\n")
    
    def callback(handle, extra):
        window_text = win32gui.GetWindowText(handle)
        class_name = win32gui.GetClassName(handle)
        
        # 检查是否是PowerPoint窗口
        is_ppt_title = "PowerPoint" in window_text or "PPT" in window_text or "演示文稿" in window_text
        is_ppt_class = class_name == "screenClass" or class_name == "PPTFrameClass" or class_name == "PPTLayerClass"
        
        if is_ppt_title or is_ppt_class:
            rect = win32gui.GetWindowRect(handle)
            w = rect[2] - rect[0]
            h = rect[3] - rect[1]
            is_full = abs(w - screen_width) < 10 and abs(h - screen_height) < 10
            extra.append((handle, window_text, class_name, is_full))
        return True
    
    windows = []
    win32gui.EnumWindows(callback, windows)
    
    if windows:
        # 优先选择全屏窗口
        for hwnd, text, cls, is_full in windows:
            if is_full:
                logger.debug(f"找到全屏PPT窗口: HWND={hwnd:08X}, 标题='{text}', 类名={cls}")
                return hwnd
        
        # 否则返回第一个找到的窗口
        hwnd, text, cls, _ = windows[0]
        logger.debug(f"找到PPT窗口: HWND={hwnd:08X}, 标题='{text}', 类名={cls}")
        return hwnd
    
    logger.warning("未找到PPT窗口")
    return None

def is_ppt_fullscreen():
    """检测PPT是否处于全屏放映模式"""
    hwnd = find_ppt_window()
    if not hwnd:
        return False
    
    # 获取窗口矩形
    rect = win32gui.GetWindowRect(hwnd)
    left, top, right, bottom = rect
    
    # 获取屏幕分辨率
    screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
    
    # 如果窗口大小接近屏幕大小，认为是全屏模式
    window_width = right - left
    window_height = bottom - top
    
    is_full = (abs(window_width - screen_width) < 10) and (abs(window_height - screen_height) < 10)
    
    # 检查窗口标题（全屏模式下标题通常为空或不同）
    window_text = win32gui.GetWindowText(hwnd)
    
    # 全屏放映模式下，窗口标题通常包含"幻灯片放映"或为空
    is_slideshow = "幻灯片放映" in window_text or len(window_text) == 0
    
    return is_full or is_slideshow

def send_input_key(key_code, press=True):
    """使用win32api.keybd_event模拟键盘输入（最可靠）"""
    import win32api
    import win32con
    import time
    
    if press:
        win32api.keybd_event(key_code, 0, 0, 0)
    else:
        win32api.keybd_event(key_code, 0, win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(0.01)

# 虚拟键码映射
VK_CODES = {
    "right": 0x27,    # VK_RIGHT
    "left": 0x25,     # VK_LEFT
    "up": 0x26,       # VK_UP
    "down": 0x28,     # VK_DOWN
    "pageup": 0x21,   # VK_PRIOR
    "pagedown": 0x22, # VK_NEXT
    "f5": 0x74,       # VK_F5
    "esc": 0x1B,      # VK_ESCAPE
    "space": 0x20,    # VK_SPACE
    "shift": 0x10,    # VK_SHIFT
    "backspace": 0x08 # VK_BACK
}

def send_ppt_cmd(gesture_id):
    """根据手势ID发送PPT快捷键（智能检测PPT模式）"""
    try:
        # 智能检测PPT模式，选择合适的快捷键
        is_slideshow = is_ppt_fullscreen()
        current_shortcut = PPT_SHORTCUT_SLIDESHOW if is_slideshow else PPT_SHORTCUT_NORMAL
        key_data = current_shortcut[gesture_id]
        
        # 尝试激活PPT窗口（如果存在）
        ppt_handle = find_ppt_window()
        if ppt_handle:
            # 使用 SetForegroundWindow 将窗口带到前台（不使用 SetFocus 避免权限问题）
            try:
                win32gui.SetForegroundWindow(ppt_handle)
                time.sleep(0.15)  # 全屏模式需要更长时间激活
            except Exception as e:
                logger.warning(f"窗口激活失败（权限限制）: {e}")
        
        # 全屏模式使用win32api.keybd_event（最可靠）
        if is_slideshow:
            if isinstance(key_data, list):
                # 组合键处理（如 Shift+F5）
                if len(key_data) == 2 and key_data[0] == "shift":
                    # 按下Shift
                    send_input_key(VK_CODES["shift"], True)
                    time.sleep(0.02)
                    # 按下主键
                    send_input_key(VK_CODES[key_data[1]], True)
                    time.sleep(0.02)
                    # 释放主键
                    send_input_key(VK_CODES[key_data[1]], False)
                    time.sleep(0.02)
                    # 释放Shift
                    send_input_key(VK_CODES["shift"], False)
                    time.sleep(0.03)
                    logger.info(f"PPT组合键触发: {'+'.join(key_data)} (全屏模式-keybd_event)")
            else:
                # 单键处理
                key_str = key_data
                if key_str in VK_CODES:
                    send_input_key(VK_CODES[key_str], True)
                    time.sleep(0.03)
                    send_input_key(VK_CODES[key_str], False)
                    time.sleep(0.03)
                    logger.info(f"PPT按键触发: {key_str} (全屏模式-keybd_event)")
                else:
                    # 字符按键（如 'b', 'w'）
                    char_code = ord(key_str.lower())
                    send_input_key(char_code, True)
                    time.sleep(0.03)
                    send_input_key(char_code, False)
                    time.sleep(0.03)
                    logger.info(f"PPT字符触发: {key_str} (全屏模式-SendInput)")
        else:
            # 非全屏模式使用pynput
            key_map = {
                "right": Key.right,
                "left": Key.left,
                "up": Key.up,
                "down": Key.down,
                "pageup": Key.page_up,
                "pagedown": Key.page_down,
                "f5": Key.f5,
                "esc": Key.esc,
                "space": Key.space,
                "shift": Key.shift,
                "backspace": Key.backspace
            }
            
            if isinstance(key_data, list):
                modifiers = []
                main_key = None
                
                for key_str in key_data:
                    if key_str in key_map:
                        if key_str in ["shift", "ctrl", "alt"]:
                            modifiers.append(key_map[key_str])
                        else:
                            main_key = key_map[key_str]
                    else:
                        main_key = key_str
                
                for mod in modifiers:
                    keyboard.press(mod)
                time.sleep(0.02)
                
                if main_key:
                    if isinstance(main_key, Key):
                        keyboard.press(main_key)
                        time.sleep(0.02)
                        keyboard.release(main_key)
                    else:
                        keyboard.press(main_key)
                        time.sleep(0.02)
                        keyboard.release(main_key)
                time.sleep(0.02)
                
                for mod in modifiers:
                    keyboard.release(mod)
                
                logger.info(f"PPT组合键触发: {'+'.join(key_data)} (编辑模式)")
            else:
                key_str = key_data
                if key_str in key_map:
                    keyboard.press(key_map[key_str])
                    time.sleep(0.02)
                    keyboard.release(key_map[key_str])
                else:
                    keyboard.press(key_str)
                    time.sleep(0.02)
                    keyboard.release(key_str)
                
                logger.info(f"PPT按键触发: {key_str} (编辑模式)")
    except Exception as e:
        logger.error(f"PPT按键模拟失败: {e}")

def exit_ppt_blank():
    """退出黑屏/白屏模式"""
    try:
        # 按空格键恢复PPT画面
        keyboard.press(Key.space)
        time.sleep(0.02)
        keyboard.release(Key.space)
    except Exception as e:
        logger.error(f"退出黑屏失败: {e}")

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def parse_sensor_data(packet):
    try:
        if len(packet) != 98:
            logger.debug(f"无效数据包长度: {len(packet)}字节, 期望98字节")
            return None
        if packet[0] != 0xAA or packet[1] != 0xAA:
            logger.debug("帧头错误")
            return None
        if packet[2] != 95:
            logger.debug(f"长度帧错误: 收到{packet[2]}, 期望95")
            return None
        if packet[97] != 0x55:
            logger.debug("帧尾错误")
            return None
        payload = packet[3:97]
        timestamp = struct.unpack('>I', payload[0:4])[0]
        acc = (payload[4], payload[5], payload[6])
        gyro = (payload[7], payload[8], payload[9])
        angle = (payload[10], payload[11], payload[12])
        emg = []
        emg_start = 13
        for i in range(10):
            emg_group = list(payload[emg_start:emg_start+8])
            emg.append(emg_group)
            emg_start += 8
        battery = payload[93]
        logger.debug(f"解析成功 - 时间戳:{timestamp}, 电池:{battery}%")
        return {
            'timestamp': timestamp, 'acc': acc, 'gyro': gyro, 'angle': angle,
            'emg': emg, 'battery': battery
        }
    except Exception as e:
        logger.error(f"解析数据包异常: {e}")
        logger.error(f"原始数据包: {packet.hex()}")
        return None

# ----------------------
# EMG 处理器（采用固定窗口，参考control_car1.py）
# ----------------------
class EMGSerialProcessor:
    def __init__(self, model, config, args):
        self.model = model
        self.config = config
        self.ser = None
        self.buffer = bytearray()
        
        # 信号处理参数（与训练代码一致）
        self.fs = 500
        self.lowcut = 20
        self.highcut = 249.999
        self.order = 4
        
        # 【修改点】采用固定窗口（参考control_car1.py）
        self.emgwindow = np.zeros((100, 8))  # 模型输入窗口（100个时间步）
        self.slidewindow = np.zeros((20, 8))  # 滑动窗口（20个时间步）
        self.MVwindow = [-1] * args.vote      # 移动平均窗口
        
        # 参数配置
        self.th1 = args.th1
        self.th2 = args.th2
        self.cmd_cooldown = args.cooldown
        
        # 状态变量
        self.slidenum = 0
        self.activeflag = 0
        self.prepredicted = -1
        self.last_cmd_time = 0
        self.valid_packet_cnt = 0
        self.lost_packet_cnt = 0
        self.monitor_last_refresh = 0

    def connect_serial(self):
        try:
            self.ser = serial.Serial(
                port=SERIAL_PORT,
                baudrate=BAUDRATE,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.5,
                write_timeout=1,
                dsrdtr=False,
                rtscts=False,
                xonxoff=False
            )
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            print(f"✅ 串口成功打开: {SERIAL_PORT}")
            print("💡 提示：请确保 ble_to_serial_bridge.py 已启动并连接BLE EMG设备")
            return True
        except Exception as e:
            print(f"❌ 串口打开失败: {e}")
            print("排查建议：")
            print("1. 确认 ble_to_serial_bridge.py 已启动虚拟串口")
            print("2. Linux：sudo chmod 666 /dev/ttyUSB0")
            print("3. 确认EMG手环BLE已连接")
            return False

    def refresh_monitor(self, current_energy):
        if not REAL_TIME_MONITOR:
            return
        now = time.time()
        if now - self.monitor_last_refresh < 0.1:
            return
        self.monitor_last_refresh = now
        port_ok = "正常" if check_port_exists(SERIAL_PORT) else "断开"
        ser_ok = "已连接" if (self.ser and self.ser.is_open) else "关闭"
        energy_alert = "↑超阈值" if current_energy > self.th1 else "↓正常"
        msg = (f"\r📡 EMG:{ser_ok} | 能量:{current_energy:.0f}({energy_alert}) | "
               f"阈值:{self.th1}/{self.th2} | 投票:{len(self.MVwindow)} | 有效包:{self.valid_packet_cnt} | 丢包:{self.lost_packet_cnt}    ")
        print(msg, end='')

    def process_serial_data(self):
        packet_size = 98
        current_energy = 0
        try:
            while True:
                self.refresh_monitor(current_energy)
                if not check_port_exists(SERIAL_PORT) or (self.ser and not self.ser.is_open):
                    print("\n❌ EMG设备断开！")
                    break
                if self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting)
                    self.buffer.extend(data)
                while len(self.buffer) >= packet_size:
                    start_pos = self.buffer.find(b'\xAA\xAA')
                    if start_pos == -1:
                        self.buffer.clear()
                        self.lost_packet_cnt +=1
                        break
                    if len(self.buffer) < start_pos + packet_size:
                        self.buffer = self.buffer[start_pos:]
                        break
                    packet = self.buffer[start_pos:start_pos+packet_size]
                    self.buffer = self.buffer[start_pos+packet_size:]
                    sensor_data = parse_sensor_data(packet)
                    if sensor_data is not None:
                        self.valid_packet_cnt +=1
                        self.process_emg_data(sensor_data)
                    else:
                        self.lost_packet_cnt +=1
                current_energy = np.sum(np.abs(self.emgwindow))
                time.sleep(0.001)
        except KeyboardInterrupt:
            print("\n🛑 停止程序")
        finally:
            if self.ser: self.ser.close()
            exit_ppt_blank()
            print("\n✅ PPT恢复正常画面，程序退出")

    def process_emg_data(self, sensor_data):
        emg = sensor_data['emg']
        
        # 1. EMG数据转换（与采集代码一致：电压转换→微伏）
        for i in range(10):
            emg_data = (np.array(emg[i])/255*3.3 - 1.65)/200*1000000
            emg_data = np.clip(emg_data, -20000, 20000)
            self.slidewindow[self.slidenum, :] = emg_data
            self.slidenum += 1
        
        # 2. 滑动窗口更新（每20个时间步更新一次100时间步的模型输入窗口）
        if self.slidenum == 20:
            self.slidenum = 0
            # 窗口滑动：保留前80个时间步，新增后20个时间步
            self.emgwindow[:80, :] = self.emgwindow[-80:, :]
            self.emgwindow[-20:, :] = self.slidewindow
            
            # 3. 活动状态判断（基于EMG信号能量）
            sum_arr = np.sum(np.abs(self.emgwindow))
            if sum_arr > self.th1:
                self.activeflag = 1
                print(f"\r📌 激活状态 | 信号能量: {sum_arr:.0f} | 上一次命令: {self.prepredicted}", end='')
            if sum_arr < self.th2:
                self.activeflag = 0
                print(f"\r📌 放松状态 | 信号能量: {sum_arr:.0f} | 发送停止命令", end='')
        
        # 4. 激活状态：模型预测+PPT控制
        if self.activeflag == 1:
            filtered_signal = np.zeros((100, 8))
            for i in range(8):
                filtered_signal[:, i] = bandpass_filter(
                    self.emgwindow[:, i], self.lowcut, self.highcut, self.fs, self.order
                )
            filtered_signal = np.swapaxes(filtered_signal, 0, 1)
            inputs = torch.tensor(np.array([filtered_signal])).float().to(self.config["device"])
            with torch.no_grad():
                _, predicted = torch.max(self.model(inputs).data, 1)
            
            # 移动平均滤波
            self.MVwindow[:-1] = self.MVwindow[1:]
            self.MVwindow[-1] = predicted.item()
            
            # 发送控制命令（仅当预测结果稳定且与上一次不同时，或超过冷却时间）
            if all_same(self.MVwindow):
                now = time.time()
                if predicted.item() != self.prepredicted or (now - self.last_cmd_time) > self.cmd_cooldown:
                    gid = predicted.item()
                    if gid == 0:
                        print("\n🚀 预测结果：开始放映 → 发送 F5")
                        send_ppt_cmd(0)
                    elif gid == 1:
                        print("\n🔙 预测结果：结束放映 → 发送 ESC")
                        send_ppt_cmd(1)
                    elif gid == 2:
                        print("\n⬅️  预测结果：上一页 → 发送 Backspace")
                        send_ppt_cmd(2)
                    elif gid == 3:
                        print("\n➡️  预测结果：下一页 → 发送 Space")
                        send_ppt_cmd(3)
                    self.prepredicted = gid
                    self.last_cmd_time = now
        
        # 5. 放松状态：重置状态（不发送任何按键）
        if self.activeflag == 0:
            if self.prepredicted != -1:
                print("\n🛑 手部放松，PPT恢复正常")
                self.prepredicted = -1
                self.MVwindow = [-1] * len(self.MVwindow)
                self.last_cmd_time = 0

# ----------------------
# 主函数
# ----------------------
def main():
    print("="*50)
    print("📶 EMG 手势控制PPT翻页程序")
    print(f"阈值 th1={args.th1}, th2={args.th2} | 投票窗口={args.vote} | 冷却={args.cooldown}s")
    print("手势映射：0当前页全屏(Shift+F5) / 1退出放映(ESC) / 2上一页(PageUp) / 3下一页(PageDown)")
    print("="*50)
    processor = None
    try:
        model = CNN1D().to(config["device"])
        model.load_state_dict(torch.load(MODEL_PATH, map_location=config["device"]))
        model.eval()
        print("✅ CNN肌电模型加载成功")
        
        processor = EMGSerialProcessor(model, config, args)
        if not processor.connect_serial():
            return
        
        print("\n🎮 手势控制PPT已启动，请打开PPT")
        processor.process_serial_data()

    except KeyboardInterrupt:
        print("\n🛑 用户手动终止程序")
    except serial.SerialException as e:
        print(f"\n❌ 串口异常: {e}")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        logger.error(f"程序异常详细信息: {e}", exc_info=True)
    finally:
        print("\n🔄 恢复PPT正常画面...")
        exit_ppt_blank()
        if processor and processor.ser and processor.ser.is_open:
            processor.ser.close()
        print("✅ PPT控制程序退出")

if __name__ == "__main__":
    main()