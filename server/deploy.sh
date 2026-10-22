#!/bin/bash
# EMG 云端训练服务 - 一键部署脚本
# 适用于 Alibaba Cloud Linux / CentOS / Ubuntu
# 用法：bash deploy.sh

set -e

echo "=== EMG 云端训练服务部署 ==="

# 1. 检查 Python（需 3.7-3.10；云端为 3.8，torch 1.13.1 不支持 3.11+）
if ! command -v python3 &> /dev/null; then
    echo "[1/4] 安装 Python..."
    if command -v yum &> /dev/null; then
        sudo yum install -y python3 python3-pip
    elif command -v apt &> /dev/null; then
        sudo apt update && sudo apt install -y python3 python3-pip
    fi
else
    echo "[1/4] Python 已安装: $(python3 --version)"
fi

PY_VER=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
case "$PY_VER" in
    3.7|3.8|3.9|3.10) echo "     版本校验通过：Python $PY_VER" ;;
    *) echo "     错误：torch 1.13.1 不支持 Python $PY_VER，请使用 3.7-3.10（云端为 3.8）"; exit 1 ;;
esac

# 2. 安装依赖
echo "[2/4] 安装 Python 依赖..."
python3 -m pip install -r requirements.txt

# 3. 创建 systemd 服务
echo "[3/4] 创建 systemd 服务..."
SERVICE_FILE="/etc/systemd/system/emg-train.service"
sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=EMG Cloud Training Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$(pwd)
ExecStart=$(which python3) $(pwd)/app.py
Restart=always
RestartSec=5
Environment=PORT=5000

[Install]
WantedBy=multi-user.target
EOF

# 4. 启动服务
echo "[4/4] 启动服务..."
sudo systemctl daemon-reload
sudo systemctl enable emg-train
sudo systemctl restart emg-train

echo ""
echo "=== 部署完成 ==="
echo "服务状态: sudo systemctl status emg-train"
echo "查看日志: sudo journalctl -u emg-train -f"
echo "API 地址: http://$(hostname -I | awk '{print $1}'):5000/api/health"
echo ""
echo "防火墙开放 5000 端口:"
echo "  sudo firewall-cmd --add-port=5000/tcp --permanent && sudo firewall-cmd --reload"
