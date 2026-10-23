# sEMG 手势控制系统

用肌电臂环（8 通道 / 500 Hz）做手势识别，并把识别结果映射成不同被控对象的命令：
宇树 Go2、Go2-W、侦察球、ROS 机械车、PPT/WPS 翻页。

系统由三部分组成：**Android App**（采集 / 训练 / 识别）、**Python SDK**（协议与会话）、
以及每个被控对象自己的适配器。App 与被控对象之间只传**语义命令**，不传设备专属报文，
因此新增一个被控对象不需要改 App。

```
┌──────────────┐   BLE (FFE0 / FFE2)   ┌──────────────────────┐
│  肌电臂环     │ ────────────────────▶ │  Android App          │
│  BT-11       │    8ch @ 500 Hz       │  采集 · 训练 · 识别     │
└──────────────┘                       └──────────┬───────────┘
                                                  │ UDP · 语义命令
                                                  ▼
                                       ┌──────────────────────┐
                                       │  emg_sdk (Python)     │
                                       │  协议 · 心跳 · 看门狗   │
                                       └──────────┬───────────┘
                                                  │ 适配器
              ┌───────────┬───────────────┬───────┴───────┬───────────┐
              ▼           ▼               ▼               ▼           ▼
            Go2        Go2-W           侦察球          机械车       PPT/WPS
```

## 支持的能力

| 被控对象 | 适配器 | 依赖 | 说明 |
|---|---|---|---|
| Go2 机械狗 | `go2` | `unitree_sdk2py` | `Move` / `StopMove`，10 Hz 心跳 |
| Go2 轮式 | `go2w` | `unitree_sdk2py` | 先 `BalanceStand` 再 `Move` |
| 侦察球 | `ball` | 无 | 11 字节 UDP + CRC16，20 Hz 心跳 |
| 机械车 | `car` | 无 | 转发 `forward/left/...` 给 ROS 节点 |
| PPT / WPS | `ppt` | Windows 或 `pynput` | `next/prev/start/exit` 按键注入 |

> 智能车的 `1~7` 数字协议尚未内置适配器；需要时复制 `adapters/mechanical_car.py`，
> 把 payload 换成数字即可（约 20 行改动）。

App 端可选的本地模型：最近质心、线性 SVM、RBF SVM（LIBSVM）；也可从云端下载
TorchScript 模型（CNN-Transformer）。

## 目录结构

```
.
├─ emg_sdk/            Python SDK：协议、UDP 服务端、设备适配器
├─ android/            Android App 源码（Java）
├─ server/             Flask 训练服务（上传 CSV → 训练 → 下发模型）
├─ control_PPT/        离线演示套件：BLE 采集 + RBF 训练 + 翻页/控制
├─ base_cross_user/    跨用户 / 零样本研究的训练与消融脚本
├─ ball_files/         侦察球协议的参考实现（Python）
├─ docs/protocol.md    App ↔ SDK 协议（v2），跨语言契约
├─ pyproject.toml      SDK 打包配置
└─ udp_go2_bridge*.py  旧版中转脚本（可被 emg_sdk 替代）
```

---

## 三条使用路径

### A. 只想控制现成设备（最常见）

**1）装 SDK**（树莓派 / Linux / Windows 均可）

```bash
git clone <repo-url> && cd <repo>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
emg-sdk list                       # 查看已注册适配器
```

Go2 / Go2-W 还需要宇树官方 SDK：

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python
cd unitree_sdk2_python && pip install -e .
```

**2）启动服务端**（一台设备一个进程）

```bash
# Go2 机械狗
emg-sdk run --adapter go2 --port 8888 --interface eth0

# Go2 轮式
emg-sdk run --adapter go2w --port 8888 --interface eth0

# 侦察球（前提：eth0:1 已配置 192.168.1.50）
emg-sdk run --adapter ball --port 8888 --source-ip 192.168.1.50

# 机械车（转发给小车上的 ROS 节点）
emg-sdk run --adapter car --port 8888 --target 192.168.2.10:8888

# PPT（在放 PPT 的那台 Windows 上）
emg-sdk run --adapter ppt --port 8888 --require-window
```

**3）没有硬件也能先验证**

```bash
emg-sdk run --adapter ball --port 8899 --dry-run
# 另开一个终端
emg-sdk send --host 127.0.0.1 --port 8899 --act move --vx 0.6 --repeat 20
emg-sdk send --host 127.0.0.1 --port 8899 --act hello --wait
```

**4）可选：浏览器查看 8 通道肌电波形**

App 采集时可以顺手把原始样本推给 SDK，用来确认电极接触和信号质量：

```bash
# 在任意适配器上附加网页监控（默认端口 8080）
emg-sdk run --adapter go2 --port 8888 --monitor

# 或者只想看波形，不驱动设备
emg-sdk monitor --port 8888 --http-port 8080
```

然后在 App 控制页勾选「同步肌电到 SDK」，浏览器打开 `http://<SDK 主机 IP>:8080`。
页面会实时画出 8 条波形，并显示采样率、样本计数与连接状态。推送的是原始
int16，不参与识别和控制，取消勾选即停止。

**5）手机端**

安装 `app-debug.apk`，在控制页填：

- 设备类型：Go2 / Go2 轮式 / 侦察球 / 机械车
- IP：树莓派热点地址（如 `192.168.10.1`）
- 端口：`8888`（与 SDK 启动参数一致）

App 默认发送旧格式字符串，SDK **同时兼容**旧格式与 v2 JSON，因此不改 App 也能直接接上。

### B. 加一个新的被控对象

在 `emg_sdk/adapters/` 新建一个类，实现三个方法即可：

```python
from .base import DeviceAdapter
from ..protocol import ACTION_MOVE

class MyRobotAdapter(DeviceAdapter):
    name = "myrobot"
    display = "我的机器人"
    publish_hz = 20.0        # 心跳频率
    deadman_ms = 500         # 超时自动停

    def start(self):
        self.api = MyRobotApi()          # 厂商 SDK

    def apply(self, action, args):
        if action == ACTION_MOVE:
            self.api.drive(args["vx"] * 0.5, args["wz"] * 1.0)

    def stop(self):
        self.api.brake()
```

在 `emg_sdk/adapters/__init__.py` 的 `ADAPTERS` 里登记一行，然后：

```bash
emg-sdk run --adapter myrobot --port 8888
```

App 侧不需要改代码：设备类型按后端返回的 `capabilities.actions` 裁剪手势映射。

### C. 采集 / 训练（研究者）

- **手机本地**：App 内选算法（质心 / 线性 SVM / RBF SVM）直接训练；
- **上传云端**：`server/app.py` 起服务，App 填地址后上传 CSV，服务端训练
  CNN-Transformer 并下发 TorchScript；
- **纯离线**：`control_PPT/collect_and_train.py` 一条命令完成 BLE 采集 + RBF 训练 +
  生成 `rbf_model.joblib`，再用 `control_PPT/ppt_control.py` 翻页。

---

## 协议速查

完整定义见 [docs/protocol.md](docs/protocol.md)。

```json
// 连续运动（vx 正为前进，wz 正为左转，均归一化到 -1..1）
{"v":2,"seq":12,"act":"move","arg":{"vx":0.60,"wz":0.00}}

// 停止
{"v":2,"seq":13,"act":"stop"}

// 离散动作：next / prev / start / exit / open / close
{"v":2,"act":"next"}

// 能力查询
{"v":2,"act":"hello"}

// 肌电波形流（可选，不参与心跳）
{"v":2,"type":"stream","seq":1,"rate":500,"ch":8,"n":25,"data":"<base64 int16>"}
```

三条时序规则：

1. `move` 需要按 `publish_hz`（10~20 Hz）持续发送；
2. 超过 `deadman_ms` 没收到新 `move`，SDK 自动 `stop`；
3. `stop` 幂等，发一次即可。

旧格式（`forward`、`1`~`7`、`go2w:left`、侦察球 11 字节包）仍然可用，便于平滑迁移。

---

## 构建与部署

### Android App

需要 Android SDK 34 与 JDK 17。

**Windows 一键脚本**（推荐，双击或在仓库根目录运行）：

```bat
build_app.bat            :: 只编译
build_app.bat /install   :: 编译并安装到已连接的手机
```

脚本会自动查找 JDK 17（环境变量 / Android Studio 自带 JBR / 常见安装路径）和
Android SDK，生成 `android\local.properties` 后执行 Gradle 构建，最后打印 APK
路径与体积。找不到依赖时会提示改怎么装。

**手动构建**：

```bash
cd android
export ANDROID_HOME=/path/to/Android/Sdk
export JAVA_HOME=/path/to/jdk-17
./gradlew :app:assembleDebug
# 产物：app/build/outputs/apk/debug/app-debug.apk
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

> `android/local.properties` 被 `.gitignore` 排除，克隆后不存在；不设
> `ANDROID_HOME` 也没写这个文件时，Gradle 会报 `SDK location not found`。

Android 12+ 首次启动需授予「附近的设备」权限。

### 训练服务

```bash
cd server
pip install -r requirements.txt
python3.8 app.py          # 默认 5000 端口，可用 PORT 环境变量覆盖
```

### 运行测试

SDK 的协议、适配器与会话逻辑都有单元测试，不需要连接任何硬件：

```bash
python -m unittest discover -s tests -v     # 只依赖标准库
# 或者
pip install -e ".[dev]" && pytest
```

### 开机自启（树莓派）

```ini
# /etc/systemd/system/emg-sdk.service
[Unit]
Description=EMG SDK (Go2)
After=network-online.target

[Service]
WorkingDirectory=/home/pi/emg-control
ExecStart=/home/pi/emg-control/.venv/bin/emg-sdk run --adapter go2 --port 8888 --interface eth0
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

---

## 排错

| 现象 | 排查 |
|---|---|
| App 搜不到臂环 | 长按 SW2 3 秒启动采集；确认手机 App 已断开（BLE 同时只能一个中心）；或 `--address` 指定 MAC |
| 连上但样本数为 0 | 通知有字节但解析不出帧：确认固件帧头是 `A5 5A`、`sampleCount` 2 字节小端 |
| Go2 不动 | 狗需处于站立状态；`--interface eth0` 与狗同网段；命令需持续发送（600ms 内） |
| 侦察球不动 | 确认 `eth0:1` 为 `192.168.1.50`；固件 lease 约 0.4s，需 20Hz 续发；速度上限 1000 |
| PPT 不翻页 | 焦点是否在放映窗口（加 `--require-window`）；是否已按 F5 进入放映 |
| 卡顿后动作扎堆 | App 已改为「一次通知最多预测一次」，若仍有，检查 2.4G 干扰与 BLE 距离 |

**安全提示**：首次带载前请架空设备、清空周边并准备物理急停；`deadman_ms` 建议
300~800 ms，不要为了「跟手」调得过大。

## License

[MIT](LICENSE)
