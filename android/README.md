# 智为光电sEMG测试

`android/` 是与本仓库 STM32 固件配套的原生 Android 应用。应用直接连接
BT-11 的 BLE 透传服务，订阅 `FFE2` 通知，并解析固件的 `A5 5A` 二进制帧。

## 功能

- 扫描并连接 BT-11（服务 `FFE0`、通知 `FFE2`、写入 `FFE1`）；
- 实时绘制 8 通道、最近 2 秒的原始 ADC 波形；
- 显示当前 8 通道原始值、采样率、帧数、丢帧数和 CRC 错误数；
- 按动作标签将原始数据保存为 CSV；
- 管理历史采集记录，查看末尾波形、导出 CSV 或删除记录；
- 在手机端从 200 ms 窗口提取 56 维 sEMG 特征并训练分类模型；
- 命名并保存多个模型，可切换当前模型、导出 JSON 或删除模型；
- 以 50 ms 步长实时识别动作，模型及数据只保存在手机本地；
- 可将“左、右、前进、倒退、停止”识别结果转换为同机 G20 控制 DEMO 的
  HTTP 控制流，识别中断、蓝牙断开或应用退出会自动停止。

训练使用每通道的平均绝对值、RMS、波形长度、方差、过零率、斜率变化率及
通道能量占比。特征先按数据集标准化，再为每种动作拟合质心；推理时通过
标准化特征距离分类并输出归一化置信度。建议每种动作采集至少 3 次、每次
3–5 秒，并加入 `rest`（放松）动作。

## 构建与测试

需要 Android SDK 34 和 JDK 17：

```sh
cd android
export ANDROID_HOME="$HOME/Library/Android/sdk"
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
./run_core_tests.sh
./gradlew :app:assembleDebug
```

调试 APK 位于 `app/build/outputs/apk/debug/app-debug.apk`。安装到已启用无线
调试的设备：

```sh
adb connect 192.168.11.162:39301
adb -s 192.168.11.162:39301 install -r app/build/outputs/apk/debug/app-debug.apk
```

Android 12 及以上首次启动需要允许“附近的设备”权限。Android 11 及以下
因系统 BLE 扫描规则需要位置权限，但应用不会读取或保存位置。

## G20 控制

先在同一台 Android 设备启动“智为光电远程控制DEMO”，保持默认 HTTP 端口
`8080` 和令牌 `zhiwei-demo`。当前模型须包含停止动作；点击“启用G20控制”并
确认安全提示后，应用通过 `127.0.0.1` 调用 `/control` 和 `/stop`。运动命令以
20 Hz 续发以满足 DEMO 的 500 ms 看门狗，控制通道映射为：

- 左/右：CH4 = 1200/1800；
- 前进/倒退：CH3 = 1800/1200；
- 停止：调用 `/stop`。

首次带载前请架空机器狗、清空周边并准备物理急停；若现场方向与标签相反，
应先停机并校准 CH3/CH4 的方向配置。

## 数据格式

采集文件保存在应用私有目录 `files/recordings/`，每行包含：

```text
sample_index,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8
```

模型保存在 `files/gesture_model.json`。调试版本可通过 ADB 检查或导出：

```sh
adb shell run-as cn.semg.armband ls files/recordings
adb exec-out run-as cn.semg.armband tar -cf - files/recordings > semg-recordings.tar
```

## 联调检查

连接成功但没有波形时，先确认手环已经长按 SW2 三秒进入采集状态。BT-11
必须为 BLE 从机/双模从机，且没有被其他主机占用。正常状态应接近
`500 sample/s`、`50 帧/s`，CRC 错误和丢帧应持续为 0。
