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

# ----------------------
# 命令行参数配置
# ----------------------
parser = argparse.ArgumentParser(description="EMG 手势控制小车（可配置灵敏度）")
parser.add_argument('--th1', type=int, default=90000, help='动作激活阈值 (默认 90000)')
parser.add_argument('--th2', type=int, default=89999, help='动作释放阈值 (默认 89999)')
parser.add_argument('--vote', type=int, default=10, help='投票窗口长度 (默认10)')
parser.add_argument('--cooldown', type=float, default=1.0, help='命令重复冷却秒 (默认1.0)')
args = parser.parse_args()

# ----------------------
# 核心配置
# ----------------------
host = '192.168.2.10'
port = 8888

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
    if len(packet) != 98:
        return None
    if packet[0] != 0xAA or packet[1] != 0xAA:
        return None
    if packet[2] != 95:
        return None
    if packet[97] != 0x55:
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

    return {
        'timestamp': timestamp, 'acc': acc, 'gyro': gyro, 'angle': angle,
        'emg': emg, 'battery': battery
    }

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

        self.emgwindow = np.zeros((100, 8))
        self.slidewindow = np.zeros((20, 8))
        
        self.vote_len = args.vote
        self.MVwindow = [-1] * self.vote_len

        self.th1 = args.th1
        self.th2 = args.th2
        self.cmd_cooldown = args.cooldown

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
            return True
        except Exception as e:
            print(f"❌ 串口打开失败: {e}")
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
        msg = (f"\r📡 蓝牙:{ser_ok} | 能量:{current_energy:.0f}({energy_alert}) | "
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

    def process_emg_data(self, sensor_data):
        emg = sensor_data['emg']
        for i in range(10):
            emg_data = (np.array(emg[i])/255*3.3 - 1.65)/200*1000000
            emg_data = np.clip(emg_data, -20000, 20000)
            self.slidewindow[self.slidenum, :] = emg_data
            self.slidenum += 1

        if self.slidenum == 20:
            self.slidenum = 0
            self.emgwindow[:80, :] = self.emgwindow[-80:, :]
            self.emgwindow[-20:, :] = self.slidewindow

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

    try:
        model = CNN1D().to(config["device"])
        model.load_state_dict(torch.load(MODEL_PATH, map_location=config["device"]))
        model.eval()
        print("✅ 模型加载成功")
    except:
        print("❌ 模型加载失败")
        return

    processor = EMGSerialProcessor(model, config, args)
    if not processor.connect_serial():
        return

    print("\n🎮 开始控制 | 实时能量监控已开启")
    print("提示：手放松后，能量低于阈值会自动停止小车")
    try:
        processor.process_serial_data()
    except KeyboardInterrupt:
        send_command('stop')
        print("\n🛑 程序退出")

if __name__ == "__main__":
    main()