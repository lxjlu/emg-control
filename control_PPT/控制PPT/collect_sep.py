import serial
import struct
import time
import numpy as np
import os
import sys
import argparse
import logging
from datetime import datetime

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------
# 配置参数
# ----------------------
import os

# 使用虚拟串口（由 ble_to_serial_bridge.py 创建，放在用户目录下）
# SERIAL_PORT = os.path.expanduser('~/virtual_serial/emg_port')
# SERIAL_PORT = '/dev/ttyUSB0'
SERIAL_PORT = 'COM5'
BAUDRATE = 115200
SENSOR_DATA_PRINT = False
REAL_TIME_MONITOR = True

SAVE_DIR = "Sep_EMG_Data"
EACH_GES_GROUP_NUM = 5      # 每个动作固定 5 组
PREPARE_DURATION = 3
RECORD_DURATION = 5

# ----------------------
# 命令行参数 --ges 1~4
# ----------------------
parser = argparse.ArgumentParser(description="EMG 采集程序")
parser.add_argument('--ges', type=int, required=True, choices=[1,2,3,4], help="选择动作：1/2/3/4")
args = parser.parse_args()
SELECTED_GES = args.ges

# 每个动作内部都是 1~5 编号
MAX_MOTION_GROUPS = EACH_GES_GROUP_NUM
# 自动创建当前动作的保存目录
GESTURE_SAVE_DIR = os.path.join(SAVE_DIR, f"gesture_{SELECTED_GES}")

# ----------------------
# 工具函数
# ----------------------
def check_port_exists(port_path):
    return os.path.exists(port_path)

def create_save_dir():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    if not os.path.exists(GESTURE_SAVE_DIR):
        os.makedirs(GESTURE_SAVE_DIR)

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
        acc_x, acc_y, acc_z = payload[4], payload[5], payload[6]
        gyro_x, gyro_y, gyro_z = payload[7], payload[8], payload[9]
        pitch, roll, yaw = payload[10], payload[11], payload[12]

        emg = []
        emg_start = 13
        for i in range(10):
            emg_group = list(payload[emg_start:emg_start+8])
            emg.append(emg_group)
            emg_start += 8

        battery = payload[93]

        logger.debug(f"解析成功 - 时间戳:{timestamp}, 电池:{battery}%, 加速度:({acc_x},{acc_y},{acc_z})")

        return {
            'timestamp': timestamp,
            'acc': (acc_x, acc_y, acc_z),
            'gyro': (gyro_x, gyro_y, gyro_z),
            'angle': (pitch, roll, yaw),
            'emg': emg,
            'battery': battery
        }
    except Exception as e:
        logger.error(f"解析数据包异常: {e}")
        logger.error(f"原始数据包: {packet.hex()}")
        return None

def print_countdown(message, duration):
    for i in range(duration, 0, -1):
        print(f"\r⚠️  {message} - 倒计时 {i} 秒...", end='')
        sys.stdout.flush()
        time.sleep(1)
    print(f"\r✅ {message} - 倒计时结束！")
    sys.stdout.flush()

# ----------------------
# 数据处理类
# ----------------------s
class EMGDataProcessor:
    def __init__(self):
        self.buffer = bytearray()
        self.group_start_time = None
        self.motion_group = 1
        self.current_stage = "READY"
        self.valid_packet_cnt = 0
        self.lost_packet_cnt = 0
        self.monitor_last_refresh = 0

        # 每组都是 1~5
        self.memory_buffer = {i: [] for i in range(1, MAX_MOTION_GROUPS + 1)}
        create_save_dir()

    def reset_group_timer(self):
        self.group_start_time = time.time()
        self.current_stage = "PREPARE"
        print(f"\n" + "="*60)
        print(f"📌 动作 {SELECTED_GES} → 第 {self.motion_group}/{MAX_MOTION_GROUPS} 组")
        print(f"📂 保存路径：{GESTURE_SAVE_DIR}/{self.motion_group}.txt")
        print(f"⏳ 准备倒计时")
        print("="*60)
        sys.stdout.flush()
        
        print_countdown("准备阶段", PREPARE_DURATION)
        
        self.current_stage = "RECORD"
        self.record_start_time = time.time()
        print(f"\n📝 开始记录 {RECORD_DURATION} 秒")
        sys.stdout.flush()

    def process_buffer(self, new_data):
        self.buffer.extend(new_data)
        packet_size = 98

        while len(self.buffer) >= packet_size:
            start_pos = self.buffer.find(b'\xAA\xAA')
            if start_pos == -1:
                self.buffer.clear()
                self.lost_packet_cnt += 1
                break

            if len(self.buffer) < start_pos + packet_size:
                self.buffer = self.buffer[start_pos:]
                break

            packet = self.buffer[start_pos:start_pos+packet_size]
            self.buffer = self.buffer[start_pos+packet_size:]

            sensor_data = parse_sensor_data(packet)
            if not sensor_data:
                self.lost_packet_cnt += 1
                continue

            self.valid_packet_cnt += 1
            self.process_timer_and_save(sensor_data)

    def process_timer_and_save(self, sensor_data):
        if self.current_stage == "READY":
            self.reset_group_timer()
            return

        current_time = time.time()
        record_elapsed = current_time - self.record_start_time

        if self.current_stage == "RECORD":
            remaining = max(0, RECORD_DURATION - record_elapsed)
            print(f"\r📊 记录中 → 剩余 {remaining:.1f}s | 保存: {self.motion_group}.txt", end='')
            sys.stdout.flush()

            emg = sensor_data['emg']
            for i in range(10):
                emg_data = (np.array(emg[i])/255*3.3 - 1.65)/200*1000000
                emg_data = emg_data.reshape(1, -1)
                self.memory_buffer[self.motion_group].append(emg_data)

            if record_elapsed >= RECORD_DURATION:
                print(f"\n✅ 第 {self.motion_group} 组采集完成")
                sys.stdout.flush()

                if self.motion_group < MAX_MOTION_GROUPS:
                    self.motion_group += 1
                    self.current_stage = "READY"
                    print(f"\n" + "-"*60)
                    print(f"⌛ 休息 3 秒")
                    print("-"*60)
                    sys.stdout.flush()
                    time.sleep(3)
                else:
                    self.current_stage = "FINISH"
                    print(f"\n🎉 动作 {SELECTED_GES} 全部 5 组完成！")
                    print("\n💾 开始保存文件...")
                    sys.stdout.flush()
                    self.save_all_to_disk()

    def save_all_to_disk(self):
        for idx in range(1, MAX_MOTION_GROUPS + 1):
            data_list = self.memory_buffer[idx]
            if not data_list:
                continue
            file_path = os.path.join(GESTURE_SAVE_DIR, f"{idx}.txt")
            all_data = np.vstack(data_list)
            np.savetxt(file_path, all_data, fmt="%.2f", delimiter=",")
        print(f"✅ 全部保存到：{GESTURE_SAVE_DIR}/1~5.txt")
        sys.stdout.flush()

    def refresh_monitor(self, ser):
        if not REAL_TIME_MONITOR:
            return
        if self.current_stage == "FINISH":
            return
            
        now = time.time()
        if now - self.monitor_last_refresh < 0.1:
            return
        self.monitor_last_refresh = now

        port_ok = "正常" if check_port_exists(SERIAL_PORT) else "断开"
        ser_ok = "已连接" if (ser and ser.is_open) else "关闭"
        msg = f"\r📡 串口:{ser_ok} | 有效包:{self.valid_packet_cnt} | 丢包:{self.lost_packet_cnt}   "
        sys.stdout.write(msg)
        sys.stdout.flush()

# ----------------------
# 主函数
# ----------------------
def main():
    ser = None
    processor = EMGDataProcessor()

    try:
        if not check_port_exists(SERIAL_PORT):
            print(f"❌ 错误：{SERIAL_PORT} 不存在！")
            return

        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUDRATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=2,
            write_timeout=2,
            dsrdtr=False,
            rtscts=False,
            xonxoff=False
        )
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        print(f"📶 串口连接成功 → 动作 {SELECTED_GES}")
        print(f"📂 保存目录：{GESTURE_SAVE_DIR}")
        print(f"📋 每组文件：1.txt ~ 5.txt")
        print("="*60)
        print("🚀 程序准备就绪")
        print("="*60)
        sys.stdout.flush()
    except Exception as e:
        print(f"❌ 串口打开失败: {e}")
        return

    try:
        while True:
            if not check_port_exists(SERIAL_PORT) or (ser and not ser.is_open):
                print("\n❌ 蓝牙断开！")
                break

            if processor.current_stage == "FINISH":
                print("\n✅ 全部完成，程序自动退出！")
                break

            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                processor.process_buffer(data)

            processor.refresh_monitor(ser)
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n\n🛑 手动终止")
        processor.save_all_to_disk()
    finally:
        if ser and ser.is_open:
            ser.close()
            print("\n🔌 串口已关闭")

if __name__ == "__main__":
    main()