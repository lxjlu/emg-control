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

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------
# 命令行参数配置
# ----------------------
parser = argparse.ArgumentParser(description="EMG 手势控制小车（可配置灵敏度）")
parser.add_argument('--th1', type=int, default=90000, help='动作激活阈值 (默认 90000)')
parser.add_argument('--th2', type=int, default=89999, help='动作释放阈值 (默认 89999)')
parser.add_argument('--vote', type=int, default=10, help='投票窗口长度 (默认10)')
parser.add_argument('--cooldown', type=float, default=1.0, help='命令重复冷却秒 (默认1.0)')
parser.add_argument('--window_time', type=float, default=0.04, help='滑动窗口时间(秒),替代固定大小窗口 (默认0.04s)')
parser.add_argument('--model_samples', type=int, default=100, help='模型输入所需样本数 (默认100)')
parser.add_argument('--model_samples', type=int, default=100, help='模型输入所需样本数 (默认100)')
args = parser.parse_args()

# ----------------------
# 核心配置
# ----------------------
host = '192.168.2.10'
port = 8888

import os

# 使用虚拟串口（由 ble_to_serial_bridge.py 创建，放在用户目录下）
# SERIAL_PORT = os.path.expanduser('~/virtual_serial/emg_port')
SERIAL_PORT = '/dev/ttyUSB0'
BAUDRATE = 115200
REAL_TIME_MONITOR = True

MODEL_PATH = 'best_lx_model.pth'
config = {
    "device": torch.device("cpu")
}

# ----------------------
# 模型定义（不变）
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
# 工具函数
# ----------------------
def check_port_exists(port_path):
    return os.path.exists(port_path)

def all_same(lst):
    return all(x == lst[0] for x in lst)

def send_command(cmd):
    try:
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_sock.sendto(cmd.encode('utf-8'), (host, port))
        udp_sock.close()
    except Exception as e:
        pass

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
# EMG 处理器（修复锁死问题）
# ----------------------
class EMGSerialProcessor:
    def __init__(self, model, config, args):
        self.model = model
        self.config = config
        self.ser = None
        self.buffer = bytearray()

        self.fs = 500
        self.lowcut = 20
        self.highcut = 249.999
        self.order = 4

        self.model_samples = args.model_samples  # 模型输入所需样本数
        self.emgwindow = np.zeros((self.model_samples, 8))
        
        # 固定时间窗口相关变量
        self.window_time = args.window_time  # 窗口时间（秒）
        self.emg_time_queue = []  # 存储每个样本的时间戳
        self.emg_data_queue = []  # 存储EMG数据
        
        # 动态频率检测
        self.freq_history = []  # 存储最近的时间间隔用于计算频率
        self.current_freq = 500  # 当前检测到的频率（默认500Hz）
        self.last_packet_time = 0  # 上一个数据包的时间
        
        # 动态调整的窗口大小
        self.vote_len_base = args.vote  # 基准投票窗口长度
        self.vote_len = args.vote
        self.MVwindow = [-1] * self.vote_len

        self.th1 = args.th1
        self.th2 = args.th2
        self.cmd_cooldown = args.cooldown

        self.slidenum = 0
        self.slide_samples = 20  # 每次处理的样本数（会动态调整）
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
            print(f"✅ 虚拟串口成功打开: {SERIAL_PORT}")
            print("💡 提示：请确保 ble_to_serial_bridge.py 已启动并连接BLE设备")
            return True
        except Exception as e:
            print(f"❌ 串口打开失败: {e}")
            print("排查建议：")
            print("1. 确认 ble_to_serial_bridge.py 已启动并成功创建虚拟串口")
            print("2. 检查串口权限（Linux：sudo chmod 666 /dev/rfcomm0）")
            print("3. 确认BLE设备已连接")
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
        msg = (f"\r📡 蓝牙:{ser_ok} | 频率:{self.current_freq:.0f}Hz | 能量:{current_energy:.0f}({energy_alert}) | "
               f"阈值:{self.th1}/{self.th2} | 投票:{self.vote_len} | 有效包:{self.valid_packet_cnt} | 丢包:{self.lost_packet_cnt}    ")
        print(msg, end='')

    def process_serial_data(self):
        packet_size = 98
        current_energy = 0
        try:
            while True:
                self.refresh_monitor(current_energy)

                if not check_port_exists(SERIAL_PORT) or (self.ser and not self.ser.is_open):
                    print("\n❌ 蓝牙已断开！")
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
            print("\n🛑 停止")
        finally:
            if self.ser: self.ser.close()
            send_command('stop')
            print("\n✅ 小车已停止")

    def update_dynamic_params(self):
        """根据检测到的频率动态调整窗口参数"""
        if len(self.freq_history) >= 10:
            avg_interval = np.mean(self.freq_history)
            if avg_interval > 0:
                self.current_freq = 1.0 / avg_interval
                
                # 动态调整投票窗口长度（与频率成正比）
                # 500Hz时vote_len=10，50Hz时vote_len=1
                self.vote_len = max(1, int(self.vote_len_base * self.current_freq / 500))
                
                # 调整MVwindow大小
                self.MVwindow = [-1] * self.vote_len
                
                # 调整每次处理的样本数
                self.slide_samples = max(2, int(20 * self.current_freq / 500))
                
                logger.info(f"频率检测: {self.current_freq:.1f}Hz, 调整投票窗口: {self.vote_len}, 滑动样本数: {self.slide_samples}")

    def process_emg_data(self, sensor_data):
        emg = sensor_data['emg']
        current_time = time.time()
        
        # 频率检测
        if self.last_packet_time > 0:
            interval = current_time - self.last_packet_time
            self.freq_history.append(interval)
            if len(self.freq_history) > 20:
                self.freq_history.pop(0)
            self.update_dynamic_params()
        self.last_packet_time = current_time
        
        for i in range(10):
            emg_data = (np.array(emg[i])/255*3.3 - 1.65)/200*1000000
            emg_data = np.clip(emg_data, -20000, 20000)
            
            # 直接添加到队列
            self.emg_data_queue.append(emg_data)
            self.emg_time_queue.append(current_time)
            self.slidenum += 1

        # 根据动态调整的样本数触发处理
        if self.slidenum >= self.slide_samples:
            self.slidenum = 0
            
            # 裁剪超出时间窗口的数据
            while self.emg_time_queue and (current_time - self.emg_time_queue[0]) > self.window_time:
                self.emg_time_queue.pop(0)
                self.emg_data_queue.pop(0)
            
            # 更新emgwindow（保持模型所需样本数）
            if len(self.emg_data_queue) > 0:
                data_array = np.array(self.emg_data_queue)
                # 确保emgwindow至少有模型所需的样本数
                if len(data_array) >= self.model_samples:
                    self.emgwindow = data_array[-self.model_samples:]
                else:
                    # 数据不足时，用最新数据填充
                    self.emgwindow[:-len(data_array), :] = self.emgwindow[len(data_array):, :]
                    self.emgwindow[-len(data_array):, :] = data_array

            sum_arr = np.sum(np.abs(self.emgwindow))
            if sum_arr > self.th1:
                self.activeflag = 1
            if sum_arr < self.th2:
                # 关键修复：能量低于阈值时，立刻停止小车并清空状态
                self.activeflag = 0
                if self.prepredicted != -1:
                    send_command("stop")
                    print("\n🛑 手已放松，小车停止")
                self.MVwindow = [-1] * self.vote_len
                self.prepredicted = -1
                self.last_cmd_time = 0

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

            self.MVwindow[:-1] = self.MVwindow[1:]
            self.MVwindow[-1] = predicted.item()

            if all_same(self.MVwindow):
                now = time.time()
                if predicted.item() != self.prepredicted or (now - self.last_cmd_time) > self.cmd_cooldown:
                    if predicted.item() == 0:
                        print("\n➡ 前进")
                        send_command("forward")
                    elif predicted.item() == 1:
                        print("\n⬅ 后退")
                        send_command("backward")
                    elif predicted.item() == 2:
                        print("\n⬅ 左转")
                        send_command("left")
                    elif predicted.item() == 3:
                        print("\n➡ 右转")
                        send_command("right")
                    self.prepredicted = predicted.item()
                    self.last_cmd_time = now

# ----------------------
# 主函数
# ----------------------
def main():
    print("="*50)
    print("📶 EMG 手势控制小车（修复锁死问题版）")
    print(f"阈值 th1={args.th1}, th2={args.th2} | 投票窗口={args.vote} | 冷却={args.cooldown}s")
    print("="*50)

    processor = None
    try:
        # 加载模型
        model = CNN1D().to(config["device"])
        model.load_state_dict(torch.load(MODEL_PATH, map_location=config["device"]))
        model.eval()
        print("✅ 模型加载成功")

        # 初始化处理器并连接串口
        processor = EMGSerialProcessor(model, config, args)
        if not processor.connect_serial():
            return

        print("\n🎮 开始控制 | 实时能量监控已开启")
        print("提示：手放松后，能量低于阈值会自动停止小车")
        
        # 主循环
        processor.process_serial_data()
        
    except KeyboardInterrupt:
        print("\n🛑 用户手动停止")
    except serial.SerialException as e:
        print(f"\n❌ 串口异常: {e}")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        logger.error(f"程序异常详细信息: {e}", exc_info=True)
    finally:
        # 确保退出时发送停止命令给小车
        print("\n🔄 正在停止小车...")
        send_command('stop')
        # 关闭串口
        if processor and processor.ser and processor.ser.is_open:
            processor.ser.close()
        print("✅ 小车已停止，程序退出")

if __name__ == "__main__":
    main()