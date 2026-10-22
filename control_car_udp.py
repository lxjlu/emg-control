import socket
import struct
import numpy as np
from scipy.signal import butter, filtfilt
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import serial

# ----------------------
# 核心配置（用户需根据实际情况修改）
# ----------------------
# 小车UDP通信配置
# host = '192.168.0.52' # 办公室
host = '192.168.110.62'
port = 8888

# 【修改点1】真实蓝牙适配器串口配置
SERIAL_PORT = '/dev/ttyUSB0'          # Windows: COM3/COM4  Linux: /dev/ttyUSB0 / /dev/rfcomm0
BAUDRATE = 115200             # 波特率保持不变

# 模型配置
MODEL_PATH = 'best_lx_model.pth'  # 训练好的模型路径
config = {
    "device": torch.device("cpu")  # 默认为CPU，若有GPU可改为cuda
}

# ----------------------
# 模型定义（与训练代码一致，必须保留）
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
        self.fc = nn.Linear(2048, 4)  # 4个动作分类（前进/倒退/左转/右转）

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
# 工具函数（复用原逻辑，仅修改数据输入源）
# ----------------------
def all_same(lst):
    """判断列表中所有元素是否相同（用于移动平均滤波）"""
    return all(x == lst[0] for x in lst)

def send_command(cmd):
    """通过UDP向小车发送控制命令"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(5)
            s.sendto(cmd.encode('utf-8'), (host, port))
    except socket.timeout:
        print("❌ 发送命令超时（5秒无响应）")
    except Exception as e:
        print(f"❌ 命令发送异常: {str(e)}")

def butter_bandpass(lowcut, highcut, fs, order=5):
    """带通滤波器系数计算"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """应用带通滤波（去除EMG信号噪声）"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def parse_sensor_data(packet):
    """复用数据包解析函数（与采集/训练代码一致）"""
    if len(packet) != 98:
        # print(f"无效数据包长度: {len(packet)}字节, 期望98字节")
        return None
    
    if packet[0] != 0xAA or packet[1] != 0xAA:
        # print("帧头错误")
        return None
    
    length_byte = packet[2]
    if length_byte != 95:
        # print(f"长度帧错误: 收到{length_byte}, 期望95")
        return None
    
    if packet[97] != 0x55:
        # print("帧尾错误")
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
        'timestamp': timestamp,
        'acc': acc,
        'gyro': gyro,
        'angle': angle,
        'emg': emg,
        'battery': battery
    }

# ----------------------
# EMG数据处理器（核心逻辑，适配真实物理串口）
# ----------------------
class EMGSerialProcessor:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # 串口相关
        self.ser = None
        self.buffer = bytearray()  # 数据缓冲区（处理粘包/分包）
        
        # 信号处理参数（与训练代码一致）
        self.fs = 500
        self.lowcut = 20
        self.highcut = 249.999
        self.order = 4
        
        # 数据窗口（滑动窗口+预测窗口）
        self.emgwindow = np.zeros((100, 8))  # 模型输入窗口（100个时间步）
        self.slidewindow = np.zeros((20, 8))  # 滑动窗口（20个时间步）
        self.MVwindow = [-1] * 10             # 移动平均窗口（10次预测结果）
        
        # 活动状态判断阈值
        self.th1 = 90000   # 激活阈值（大于此值进入激活状态）
        self.th2 = 89999   # 放松阈值（小于此值进入放松状态）
        self.slidenum = 0  # 滑动窗口计数
        self.activeflag = 0# 活动状态标记（0=放松，1=激活）
        self.prepredicted = -1  # 上一次有效预测结果

    def connect_serial(self):
        """【修改点2】连接真实蓝牙串口（标准物理串口配置）"""
        try:
            self.ser = serial.Serial(
                port=SERIAL_PORT,
                baudrate=BAUDRATE,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.5  # 标准串口超时
            )
            print(f"✅ 成功连接真实蓝牙串口 {SERIAL_PORT}（波特率 {BAUDRATE}）")
            return True
        except Exception as e:
            print(f"❌ 串口连接失败: {e}")
            print("排查建议：")
            print("1. 查看设备管理器/终端，确认正确串口名称")
            print("2. 检查串口权限（Linux：sudo chmod 666 串口名）")
            print("3. 确认蓝牙适配器已插入、驱动正常")
            print("4. 确认波特率与设备一致")
            return False

    def process_serial_data(self):
        """【修改点3】标准物理串口数据读取逻辑"""
        if not self.ser or not self.ser.is_open:
            print("❌ 串口未连接，无法读取数据")
            return

        packet_size = 98
        try:
            while True:
                # 物理串口稳定读取逻辑
                if self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting)
                    self.buffer.extend(data)

                # 从缓冲区提取完整数据包
                while len(self.buffer) >= packet_size:
                    # 查找帧头（0xAA 0xAA）
                    start_pos = self.buffer.find(b'\xAA\xAA')
                    if start_pos == -1:
                        self.buffer.clear()
                        break

                    # 检查是否有完整数据包
                    if len(self.buffer) < start_pos + packet_size:
                        self.buffer = self.buffer[start_pos:]
                        break

                    # 提取并删除完整数据包
                    packet = self.buffer[start_pos:start_pos+packet_size]
                    self.buffer = self.buffer[start_pos+packet_size:]

                    # 解析数据并处理
                    sensor_data = parse_sensor_data(packet)
                    if sensor_data is not None:
                        self.process_emg_data(sensor_data)

                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n🛑 用户终止程序")
        except serial.SerialException as e:
            print(f"\n❌ 串口读取错误: {e}")
            print("建议：检查蓝牙设备是否断开，重新插拔适配器")
        finally:
            if self.ser and self.ser.is_open:
                self.ser.close()
                print("🔌 串口已关闭")
            # 发送停止命令给小车
            send_command('5')
            print("✅ 已发送停止命令给小车")

    def process_emg_data(self, sensor_data):
        """处理EMG数据，包括滑动窗口、滤波、模型预测、小车控制"""
        emg = sensor_data['emg']
        
        # 1. EMG数据转换（与采集代码一致：电压转换→微伏）
        for i in range(10):
            emg_data = (np.array(emg[i])/255*3.3 - 1.65)/200*1000000
            emg_data = np.clip(emg_data, -20000, 20000)  # 限制异常值
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
                print(f"📌 激活状态 | 信号能量: {sum_arr:.0f} | 上一次命令: {self.prepredicted}", end='\r')
            if sum_arr < self.th2:
                self.activeflag = 0
                print(f"📌 放松状态 | 信号能量: {sum_arr:.0f} | 发送停止命令", end='\r')
        
        # 4. 激活状态：模型预测+小车控制
        if self.activeflag == 1:
            # 信号滤波（与训练代码一致的带通滤波）
            filtered_signal = np.zeros((100, 8))
            for i in range(8):
                filtered_signal[:, i] = bandpass_filter(
                    self.emgwindow[:, i], 
                    self.lowcut, 
                    self.highcut, 
                    self.fs, 
                    self.order
                )
            
            # 模型输入准备（维度调整：(100,8) → (1,8,100)，适配CNN1D输入）
            filtered_signal = np.swapaxes(filtered_signal, 0, 1)  # (8,100)
            inputs = np.array([filtered_signal])  # (1,8,100)
            inputs = torch.tensor(inputs).float().to(self.config["device"])
            
            # 模型预测（关闭梯度计算，提高速度）
            with torch.no_grad():
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)  # 取概率最大的类别
            
            # 移动平均滤波（避免抖动，连续10次预测一致才生效）
            self.MVwindow[:-1] = self.MVwindow[1:]
            self.MVwindow[-1] = predicted.item()
            
            # 发送控制命令（仅当预测结果稳定且与上一次不同时）
            if all_same(self.MVwindow):
                if predicted.item() != self.prepredicted:
                    if predicted.item() == 0:
                        print("\n🚀 预测结果：前进 → 发送命令 '1'")
                        send_command('1')
                    elif predicted.item() == 1:
                        print("\n🔙 预测结果：倒退 → 发送命令 '2'")
                        send_command('2')
                    elif predicted.item() == 2:
                        print("\n⬅️  预测结果：左转 → 发送命令 '3'")
                        send_command('3')
                    elif predicted.item() == 3:
                        print("\n➡️  预测结果：右转 → 发送命令 '4'")
                        send_command('4')
                    self.prepredicted = predicted.item()
        
        # 5. 放松状态：发送停止命令
        if self.activeflag == 0:
            if self.prepredicted != -1:  # 仅当之前是激活状态时发送停止命令
                send_command('5')
                self.prepredicted = -1
                self.MVwindow = [-1] * 10  # 重置移动平均窗口

# ----------------------
# 主函数（整合模型加载、串口连接、数据处理）
# ----------------------
def main():
    print("="*60)
    print("🚗 EMG手势控制小车（真实蓝牙串口版）")
    print("="*60)

    # 1. 加载训练好的模型
    print(f"\n📦 正在加载模型: {MODEL_PATH}")
    try:
        model = CNN1D().to(config["device"])
        model.load_state_dict(torch.load(MODEL_PATH, map_location=config["device"]))
        model.eval()  # 设置为评估模式（关闭Dropout/BatchNorm训练模式）
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("排查建议：1. 确认模型文件路径正确 2. 模型结构与训练代码一致")
        return

    # 2. 初始化EMG处理器并连接串口
    processor = EMGSerialProcessor(model, config)
    if not processor.connect_serial():
        return

    # 3. 开始处理串口数据并控制小车
    print("\n📡 开始接收EMG数据，等待手势指令...")
    print("💡 提示：做出手势激活设备，放松时自动停止")
    print("🔑 按 Ctrl+C 终止程序")
    print("="*60)
    try:
        processor.process_serial_data()
    except KeyboardInterrupt:
        print("\n\n🛑 程序终止")
    finally:
        # 确保串口关闭和小车停止
        if processor.ser and processor.ser.is_open:
            processor.ser.close()
        send_command('5')
        print("✅ 小车已停止")

if __name__ == "__main__":
    main()