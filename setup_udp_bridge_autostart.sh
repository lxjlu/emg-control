#!/usr/bin/env bash
# 一键配置 udp_go2_bridge.py 开机自启动
# 手机连接树莓派热点后，即可通过 UDP 8888 端口控制 Go2

set -e

SERVICE_NAME="udp-go2-bridge"
USER_NAME="${SUDO_USER:-$USER}"
PROJECT_DIR="/home/$USER_NAME/Desktop/go2"

# 1. 检查 root 权限
if [ "$EUID" -ne 0 ]; then
    echo "错误：请使用 sudo 运行此脚本"
    exit 1
fi

# 2. 检查项目目录
if [ ! -d "$PROJECT_DIR" ]; then
    echo "错误：项目目录不存在：$PROJECT_DIR"
    echo "请确认 udp_go2_bridge.py 已上传到 /home/$USER_NAME/Desktop/go2/"
    exit 1
fi

if [ ! -f "$PROJECT_DIR/udp_go2_bridge.py" ]; then
    echo "错误：找不到文件 $PROJECT_DIR/udp_go2_bridge.py"
    echo "请先上传 udp_go2_bridge.py"
    exit 1
fi

# 3. 查找 Python3 路径（优先使用当前用户的 pyenv）
PYENV_ROOT="${PYENV_ROOT:-/home/$USER_NAME/.pyenv}"
PYTHON_PATH=""
PYENV_ROOT_LINE=""
PYENV_PATH_LINE=""

if [ -d "$PYENV_ROOT" ] && [ -x "$PYENV_ROOT/shims/python3" ]; then
    echo "检测到 pyenv 环境：$PYENV_ROOT"
    PYTHON_PATH=$(sudo -u "$USER_NAME" "$PYENV_ROOT/shims/python3" -c "import sys; print(sys.executable)" 2>/dev/null || echo "$PYENV_ROOT/shims/python3")
    PYENV_ROOT_LINE="Environment=\"PYENV_ROOT=$PYENV_ROOT\""
    PYENV_PATH_LINE="Environment=\"PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:/usr/local/bin:/usr/bin:/bin\""
else
    PYTHON_PATH=$(sudo -u "$USER_NAME" which python3)
fi

if [ -z "$PYTHON_PATH" ]; then
    echo "错误：未找到 python3"
    exit 1
fi

echo "使用 Python: $PYTHON_PATH"

# 4. 验证能导入 unitree_sdk2py
echo "检查宇树 SDK 依赖..."
if ! sudo -u "$USER_NAME" "$PYTHON_PATH" -c "import unitree_sdk2py" 2>/dev/null; then
    echo "警告：当前 Python 无法导入 unitree_sdk2py"
    echo "      请确认已安装宇树 SDK2 Python 绑定"
fi

# 5. 创建 systemd 服务文件
echo "创建 systemd 服务: $SERVICE_NAME"
cat > "/etc/systemd/system/$SERVICE_NAME.service" <<EOF
[Unit]
Description=UDP to Go2 Bridge for sEMG App
After=network.target
Wants=network.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$PROJECT_DIR
$PYENV_ROOT_LINE
$PYENV_PATH_LINE
ExecStart=$PYTHON_PATH $PROJECT_DIR/udp_go2_bridge.py
Restart=always
RestartSec=5
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
EOF

# 6. 重载并启用启动
systemctl daemon-reload
systemctl enable "$SERVICE_NAME.service"

# 7. 启动服务
systemctl start "$SERVICE_NAME.service"
sleep 1

# 8. 检查状态
status=$(systemctl is-active "$SERVICE_NAME.service")
echo ""
echo "=============================================="
if [ "$status" = "active" ]; then
    echo " ✅ 开机自启动配置成功！"
    echo ""
    echo " 服务名称: $SERVICE_NAME"
    echo " 服务状态: 运行中"
    echo " 监听端口: UDP 8888"
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
