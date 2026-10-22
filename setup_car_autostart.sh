:
#!/usr/bin/env bash
# 一键配置机械车 UDP 控制脚本开机自启动
# 适配 Ubuntu + ROS1 (noetic/melodic/kinetic) 环境
# 用法：sudo ./setup_car_autostart.sh

set -e

SERVICE_NAME="udp-car-control"
USER_NAME="${SUDO_USER:-$USER}"
# 默认把 udp_car_control.py 直接放到机械车桌面上
PROJECT_DIR="/home/$USER_NAME/Desktop"
SCRIPT_NAME="udp_car_control.py"
START_SCRIPT="$PROJECT_DIR/start_udp_car_control.sh"

# 1. 检查 root 权限
if [ "$EUID" -ne 0 ]; then
    echo "错误：请使用 sudo 运行此脚本"
    exit 1
fi

# 2. 检查项目目录和脚本
if [ ! -d "$PROJECT_DIR" ]; then
    echo "错误：桌面目录不存在：$PROJECT_DIR"
    exit 1
fi

if [ ! -f "$PROJECT_DIR/$SCRIPT_NAME" ]; then
    echo "错误：找不到文件 $PROJECT_DIR/$SCRIPT_NAME"
    exit 1
fi

chmod +x "$PROJECT_DIR/$SCRIPT_NAME"

# 3. 自动检测 ROS 版本，并强制使用系统 Python3
# ROS1 的 rospy 通常只安装在系统 Python 中，pyenv 等环境可能找不到。
PYTHON_PATH="/usr/bin/python3"
if [ ! -x "$PYTHON_PATH" ]; then
    echo "错误：找不到系统 Python3: $PYTHON_PATH"
    exit 1
fi

echo "使用系统 Python3: $PYTHON_PATH"

ROS_DISTRO=""
for distro in noetic melodic kinetic; do
    if [ -f "/opt/ros/$distro/setup.bash" ]; then
        ROS_DISTRO="$distro"
        break
    fi
done

if [ -z "$ROS_DISTRO" ]; then
    echo "错误：未检测到 ROS 环境"
    echo "请确认已安装 ROS1 并 source 了 setup.bash"
    exit 1
fi

echo "检测到 ROS 版本: $ROS_DISTRO"

# 4. 创建启动脚本（负责启动 roscore 和 udp_car_control.py）
cat > "$START_SCRIPT" <<EOF
#!/bin/bash
# 自动启动 roscore 和 udp_car_control.py
set -e

# 加载 ROS 环境
source /opt/ros/$ROS_DISTRO/setup.bash

# 重要：systemd 不会加载 ~/.bashrc，需要手动提取用户 .bashrc 里的
# ROS 网络配置（ROS_MASTER_URI / ROS_HOSTNAME / ROS_IP），
# 否则 rospy 会连到 http://localhost:11311，和底盘节点不在同一个 master。
USER_BASHRC="/home/$USER_NAME/.bashrc"
if [ -f "\$USER_BASHRC" ]; then
    eval "\$(grep -E '^\\s*(export\\s+)?ROS_(MASTER_URI|HOSTNAME|IP)=' "\$USER_BASHRC" 2>/dev/null)"
fi
echo "ROS_MASTER_URI=\${ROS_MASTER_URI:-http://localhost:11311}"
echo "ROS_HOSTNAME=\${ROS_HOSTNAME:-\$(hostname)}"
echo "ROS_IP=\${ROS_IP:-}"

# 等待 ROS master 可达（最多 60 秒，机械车开机后底盘可能先于本服务就绪）
for i in {1..60}; do
    if rosnode list >/dev/null 2>&1; then
        echo "ROS master 可达"
        break
    fi
    sleep 1
done

# 如果 roscore 没运行，则后台启动
if ! rosnode list >/dev/null 2>&1; then
    echo "启动 roscore..."
    roscore > /tmp/roscore.log 2>&1 &
    # 等待 roscore 就绪
    for i in {1..20}; do
        if rosnode list >/dev/null 2>&1; then
            echo "roscore 已就绪"
            break
        fi
        sleep 0.5
    done
fi

# 启动 UDP 控制节点（使用系统 Python3，确保能导入 rospy）
exec $PYTHON_PATH $PROJECT_DIR/$SCRIPT_NAME
EOF

chmod +x "$START_SCRIPT"

# 5. 创建 systemd 服务文件
cat > "/etc/systemd/system/$SERVICE_NAME.service" <<EOF
[Unit]
Description=UDP to ROS Car Control for sEMG App
After=network.target
Wants=network.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$PROJECT_DIR
ExecStart=$START_SCRIPT
Restart=always
RestartSec=5
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
EOF

# 6. 重载、启用、启动
systemctl daemon-reload
systemctl enable "$SERVICE_NAME.service"
systemctl start "$SERVICE_NAME.service"
sleep 1

# 7. 检查状态
status=$(systemctl is-active "$SERVICE_NAME.service")
echo ""
echo "=============================================="
if [ "$status" = "active" ]; then
    echo " ✅ 机械车开机自启动配置成功！"
    echo ""
    echo " 服务名称: $SERVICE_NAME"
    echo " 服务状态: 运行中"
    echo " 监听端口: UDP 8888"
    echo " ROS 版本: $ROS_DISTRO"
    echo ""
    echo " 常用命令："
    echo "   查看状态: sudo systemctl status $SERVICE_NAME"
    echo "   查看日志: sudo journalctl -u $SERVICE_NAME -f"
    echo "   停止服务: sudo systemctl stop $SERVICE_NAME"
    echo "   启动服务: sudo systemctl start $SERVICE_NAME"
    echo "   重启服务: sudo systemctl restart $SERVICE_NAME"
    echo "   禁用自启: sudo systemctl disable $SERVICE_NAME"
else
    echo " ⚠️ 服务启动失败，状态：$status"
    echo " 请运行：sudo systemctl status $SERVICE_NAME"
    echo " 或：sudo journalctl -u $SERVICE_NAME -n 50"
fi
echo "=============================================="
