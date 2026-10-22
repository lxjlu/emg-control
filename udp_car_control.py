#!/usr/bin/env python3
import socket
import rospy
import math
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState

# UDP监听配置：监听本机所有网卡8888端口
UDP_IP = "0.0.0.0"
UDP_PORT = 8888

# 创建UDP套接字
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)  # 非阻塞，不卡死ROS循环

# ROS速度话题发布器
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
pub_joint = rospy.Publisher('/joint_states', JointState, queue_size=10)
rospy.init_node('udp_to_ros_control')

rate = rospy.Rate(50)  # 50Hz控制刷新率
twist = Twist()

# ========== 小车运动速度参数（可后续微调） ==========
LINEAR_SPEED = 0.5    # 前进/后退线速度
ANGULAR_SPEED = 0.6   # 左右转角角速度
# =====================================================

# 动作函数定义
def cmd_stop():
    twist.linear.x = 0.0
    twist.angular.z = 0.0

def cmd_forward():
    twist.linear.x = LINEAR_SPEED
    twist.angular.z = 0.0

def cmd_backward():
    twist.linear.x = -LINEAR_SPEED
    twist.angular.z = 0.0

def cmd_left():
    twist.linear.x = LINEAR_SPEED * 0.5
    twist.angular.z = ANGULAR_SPEED

def cmd_right():
    twist.linear.x = LINEAR_SPEED * 0.5
    twist.angular.z = -ANGULAR_SPEED
# jiazhuakongzhi
def gripper_open():
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.name = ["a", "b", "c", "d", "x", "y"]
    msg.position = [0,0,math.radians(32), math.radians(-32),0,0]
    msg.velocity = [0.0]*6
    msg.effort = [0.0]*6
    pub_joint.publish(msg)
def gripper_close():
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.name = ["a", "b", "c", "d", "x", "y"]
    msg.position = [0,0,math.radians(32), math.radians(32),0,0]
    msg.velocity = [0.0]*6
    msg.effort = [0.0]*6
    pub_joint.publish(msg)

# 字符串命令 ↔ 动作函数映射表
cmd_dict = {
    "forward": cmd_forward,
    "backward": cmd_backward,
    "left": cmd_left,
    "right": cmd_right,
    "stop": cmd_stop,
    "open": gripper_open,
    "close": gripper_close
}

print("✅ UDP监听已开启，等待树莓派手势指令...")
# 初始化停车
cmd_stop()

# ROS主循环
while not rospy.is_shutdown():
    try:
        # 接收树莓派发来的UDP命令
        data, addr = sock.recvfrom(1024)
        cmd = data.decode().strip()
        print(f"📥 收到指令: {cmd}")
        
        # 匹配执行对应动作
        if cmd in cmd_dict:
            cmd_dict[cmd]()
    
    except socket.error:
        # 无数据时跳过，不阻塞
        pass

    # 持续发布速度指令
    pub.publish(twist)
    rate.sleep()
sock.close()
