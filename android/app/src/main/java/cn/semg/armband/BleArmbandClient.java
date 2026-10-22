package cn.semg.armband;

import android.annotation.SuppressLint;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothGatt;
import android.bluetooth.BluetoothGattCallback;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattDescriptor;
import android.bluetooth.BluetoothGattService;
import android.bluetooth.BluetoothManager;
import android.bluetooth.BluetoothProfile;
import android.bluetooth.le.BluetoothLeScanner;
import android.bluetooth.le.ScanCallback;
import android.bluetooth.le.ScanFilter;
import android.bluetooth.le.ScanResult;
import android.bluetooth.le.ScanSettings;
import android.content.Context;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/** BLE central for the BT-11 transparent UART service. */
final class BleArmbandClient {
    private static final String TAG = "SemgBle";
    static final UUID SERVICE_UUID = uuid16("FFE0");
    static final UUID WRITE_UUID = uuid16("FFE1");
    static final UUID NOTIFY_UUID = uuid16("FFE2");
    private static final UUID CCCD_UUID = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb");

    interface Listener {
        void onStatus(String text, boolean connected);
        void onBytes(byte[] bytes);
        void onScanFinished(List<DeviceItem> devices);
        /** 蓝牙意外断开后开始自动重连时回调（attempt 从 1 开始）。 */
        default void onReconnecting(String deviceName, int attempt) { }
    }

    static final class DeviceItem {
        final BluetoothDevice device;
        final String name;
        final int rssi;

        DeviceItem(BluetoothDevice device, String name, int rssi) {
            this.device = device;
            this.name = name;
            this.rssi = rssi;
        }

        @Override public String toString() {
            return name + "\n" + device.getAddress() + "   " + rssi + " dBm";
        }
    }

    private final Context context;
    private final Listener listener;
    private final Handler main = new Handler(Looper.getMainLooper());
    private final BluetoothAdapter adapter;
    private final BleEventLog eventLog;

    BleEventLog eventLog() { return eventLog; }
    private final Map<String, DeviceItem> scanResults = new LinkedHashMap<>();
    private BluetoothLeScanner scanner;
    private BluetoothGatt gatt;
    private boolean scanning;
    private boolean servicesDiscovered;
    private boolean subscriptionReady;
    private boolean announcedFirstData;
    private long receivedBytes;
    private long notificationCount;
    private int smallestNotification;
    private int largestNotification;
    private long invalidWireFrames;
    private int discoveryAttempts;

    // ---- 连接质量监控：RSSI 轮询 + 周期性传输统计 ----
    private int lastRssi = Integer.MIN_VALUE;
    private int rssiHistoryIndex = 0;
    private final int[] rssiHistory = new int[60]; // 最近60次RSSI记录（约2分钟）
    private long connectedAtMs;
    private long statsAtMs;
    private long statsBytesAtMark;
    private long statsNotificationsAtMark;
    private static final long RSSI_POLL_MS = 2000;
    private static final long RSSI_LOG_MS = 10000;
    private static final long STATS_LOG_MS = 30000;
    private long lastRssiLogMs;
    
    // ---- 深层诊断：数据包间隔监控 ----
    private long lastPacketTimeMs = 0;
    private long maxPacketIntervalMs = 0;
    private long totalPacketIntervals = 0;
    private long silentPeriods = 0; // 超过500ms无数据的静默期
    private static final long MAX_NORMAL_INTERVAL_MS = 500;
    
    // ---- 深层诊断：RSSI趋势分析 ----
    private int rssiSamplesInWindow = 0;
    private int rssiSumInWindow = 0;
    private int lastWindowAvgRssi = Integer.MIN_VALUE;
    private int consecutiveRssiDrop = 0; // 连续信号恶化次数
    private int peakRssi = Integer.MIN_VALUE; // 连接期间峰值信号
    private int minRssi = Integer.MAX_VALUE; // 连接期间最低信号
    
    // ---- 深层诊断：GATT操作统计 ----
    private int gattErrorCount = 0;
    private int serviceDiscoveryFailures = 0;
    private int cccdWriteFailures = 0;
    private int mtuNegotiationFailures = 0;
    
    // ---- 深层诊断：通知数据质量 ----
    private long lastNotificationTimeMs = 0;
    private long notificationGaps = 0; // 超过1秒无通知的次数
    private int[] notificationSizeHistory = new int[100]; // 最近100个包的大小
    private int notificationSizeIndex = 0;
    private final Runnable rssiPollRunnable = new Runnable() {
        @SuppressLint("MissingPermission")
        @Override public void run() {
            BluetoothGatt current = gatt;
            if (current == null || !subscriptionReady) {
                main.postDelayed(this, RSSI_POLL_MS);
                return;
            }
            try { current.readRemoteRssi(); } catch (SecurityException ignored) { }
            main.postDelayed(this, RSSI_POLL_MS);
        }
    };

    /** GATT 状态码可读化，便于事后分析断链原因。 */
    private static String gattStatusText(int status) {
        switch (status) {
            case 0: return "成功";
            case 5: return "认证不足(5)";
            case 8: return "连接超时(8)——设备距离过远或被遮挡";
            case 13: return "连接码不对(13)";
            case 15: return "连接已被设备忙拒绝(15)";
            case 16: return "连接异常终止(16)";
            case 19: return "远端主动断开(19)——臂环重启或关机";
            case 22: return "本机控制器断开(22)";
            case 34: return "加密校验失败(34)";
            case 62: return "连接失败堆积(62)";
            case 133: return "通用GATT错误(133)——常见于系统蓝牙栈节流或设备未广播";
            case 257: return "内部错误(257)";
            default: return "状态码 " + status;
        }
    }

    private void blog(String message) { eventLog.log("BLE", message); }
    
    /** 深层诊断：记录RSSI历史并分析趋势 */
    private void recordRssiHistory(int rssi) {
        rssiHistory[rssiHistoryIndex % rssiHistory.length] = rssi;
        rssiHistoryIndex++;
        
        // 更新峰值和最低值
        if (rssi > peakRssi) peakRssi = rssi;
        if (rssi < minRssi) minRssi = rssi;
        
        // 计算滑动窗口平均值（最近5次）
        rssiSamplesInWindow++;
        rssiSumInWindow += rssi;
        if (rssiSamplesInWindow >= 5) {
            int currentAvg = rssiSumInWindow / rssiSamplesInWindow;
            if (lastWindowAvgRssi != Integer.MIN_VALUE) {
                // 检测信号持续恶化
                if (currentAvg < lastWindowAvgRssi - 5) {
                    consecutiveRssiDrop++;
                    if (consecutiveRssiDrop >= 3) {
                        blog(String.format(java.util.Locale.CHINA,
                                "⚠️ 信号持续恶化：连续%d次下降，当前平均%d dBm，峰值%d，最低%d",
                                consecutiveRssiDrop, currentAvg, peakRssi, minRssi));
                    }
                } else {
                    consecutiveRssiDrop = 0;
                }
            }
            lastWindowAvgRssi = currentAvg;
            rssiSamplesInWindow = 0;
            rssiSumInWindow = 0;
        }
    }
    
    /** 深层诊断：分析RSSI历史趋势 */
    private String analyzeRssiTrend() {
        if (rssiHistoryIndex == 0) return "无RSSI记录";
        
        int count = Math.min(rssiHistoryIndex, rssiHistory.length);
        int start = Math.max(0, rssiHistoryIndex - count);
        
        int sum = 0;
        int max = Integer.MIN_VALUE;
        int min = Integer.MAX_VALUE;
        int first = rssiHistory[start];
        int last = rssiHistory[start + count - 1];
        
        for (int i = start; i < start + count; i++) {
            int rssi = rssiHistory[i];
            sum += rssi;
            if (rssi > max) max = rssi;
            if (rssi < min) min = rssi;
        }
        
        int avg = sum / count;
        int trend = last - first;
        String trendText = trend > 5 ? "上升" : trend < -5 ? "下降" : "稳定";
        
        // 计算波动幅度
        int fluctuation = max - min;
        String fluctuationText = fluctuation < 10 ? "稳定" : fluctuation < 20 ? "轻微波动" : "剧烈波动";
        
        return String.format(java.util.Locale.CHINA,
                "RSSI趋势：平均%d dBm（范围%d~%d），趋势%s（变化%d dB），波动%s",
                avg, min, max, trendText, trend, fluctuationText);
    }
    
    /** 深层诊断：分析通知包大小分布 */
    private String analyzeNotificationSizes() {
        if (notificationCount == 0) return "无数据";
        
        int count = Math.min((int) notificationCount, notificationSizeHistory.length);
        int sum = 0;
        int max = 0;
        int min = Integer.MAX_VALUE;
        
        for (int i = 0; i < count; i++) {
            int idx = (notificationSizeIndex - count + i + notificationSizeHistory.length) % notificationSizeHistory.length;
            int size = notificationSizeHistory[idx];
            sum += size;
            if (size > max) max = size;
            if (size < min) min = size;
        }
        
        int avg = count > 0 ? sum / count : 0;
        return String.format(java.util.Locale.CHINA,
                "包大小分布：平均%d字节（范围%d~%d），共%d包",
                avg, min, max, notificationCount);
    }

    // ---- 自动重连状态：先扫描确认设备在广播，再发起直连 ----
    // 直接对关机的设备 connectGatt 会让部分手机蓝牙栈无回调并触发连接节流，
    // 导致重连循环死掉。因此每轮先按 MAC 扫描，扫到设备才连接。
    private static final long RECONNECT_SCAN_MS = 10000;
    private static final long CONNECT_TIMEOUT_MS = 12000;
    private BluetoothDevice reconnectDevice;
    private String reconnectName;
    private boolean autoReconnect;
    private int reconnectAttempt;
    private boolean connectionPending;
    private boolean reconnectScanning;

    /** 重连扫描超时：没扫到设备（可能没开机），进入下一轮 */
    private final Runnable reconnectScanTimeoutRunnable = new Runnable() {
        @Override public void run() {
            if (!reconnectScanning) return;
            stopReconnectScan();
            if (autoReconnect) {
                blog("重连扫描超时：设备未在广播（可能未开机），进入下一轮");
                scheduleReconnect();
            }
        }
    };

    /** 直连看门狗：connectGatt 后超时仍无结果则关闭并进入下一轮 */
    private final Runnable connectTimeoutRunnable = new Runnable() {
        @SuppressLint("MissingPermission")
        @Override public void run() {
            BluetoothGatt current = gatt;
            if (connectionPending && current != null) {
                Log.w(TAG, "connect timeout, closing gatt to retry");
                blog("连接超时（" + (CONNECT_TIMEOUT_MS / 1000) + " 秒无回调），关闭后进入下一轮重连");
                connectionPending = false;
                gatt = null;
                try { current.disconnect(); current.close(); } catch (SecurityException ignored) { }
                if (autoReconnect) scheduleReconnect();
            }
        }
    };

    /** 重连专用扫描回调：只认目标设备的 MAC 地址 */
    private final ScanCallback reconnectScanCallback = new ScanCallback() {
        @Override public void onScanResult(int callbackType, ScanResult result) {
            if (!autoReconnect || gatt != null || reconnectDevice == null) return;
            if (!reconnectDevice.getAddress().equals(result.getDevice().getAddress())) return;
            Log.i(TAG, "target device advertising again, connecting");
            blog("重连扫描：发现设备在广播（信号 " + result.getRssi() + " dBm），发起连接");
            stopReconnectScan();
            connectSavedDevice();
        }

        @Override public void onScanFailed(int errorCode) {
            reconnectScanning = false;
            Log.w(TAG, "reconnect scan failed: " + errorCode);
            if (autoReconnect) main.postDelayed(() -> {
                if (autoReconnect) scheduleReconnect();
            }, RECONNECT_SCAN_MS);
        }
    };

    BleArmbandClient(Context context, Listener listener) {
        this(context, listener, null);
    }

    /** sharedLog 非空时多个客户端共用同一日志文件（如 BLE 与经典蓝牙并存）。 */
    BleArmbandClient(Context context, Listener listener, BleEventLog sharedLog) {
        this.context = context.getApplicationContext();
        this.listener = listener;
        this.eventLog = sharedLog != null ? sharedLog : new BleEventLog(this.context);
        BluetoothManager manager = context.getSystemService(BluetoothManager.class);
        adapter = manager == null ? null : manager.getAdapter();
        blog("App 启动，BLE 日志系统就绪");
    }

    boolean isBluetoothEnabled() {
        return adapter != null && adapter.isEnabled();
    }

    @SuppressLint("MissingPermission")
    void scan() {
        if (!isBluetoothEnabled()) {
            listener.onStatus("请先开启手机蓝牙", false);
            listener.onScanFinished(new ArrayList<>());
            return;
        }
        stopScan(false);
        scanResults.clear();
        scanner = adapter.getBluetoothLeScanner();
        if (scanner == null) {
            listener.onStatus("BLE 扫描器不可用", false);
            return;
        }
        scanning = true;
        listener.onStatus("正在扫描附近的肌电手环…", false);
        blog("开始扫描 BLE 设备");
        try {
            scanner.startScan(scanCallback);
            main.postDelayed(() -> stopScan(true), 7000);
        } catch (SecurityException error) {
            scanning = false;
            listener.onStatus("缺少蓝牙扫描权限", false);
            eventLog.logError("BLE", "启动扫描失败：缺少权限", error);
        }
    }

    @SuppressLint("MissingPermission")
    private void stopScan(boolean report) {
        boolean wasScanning = scanning;
        scanning = false;
        if (report) {
            blog("扫描完成：发现 " + scanResults.size() + " 个设备");
            listener.onScanFinished(new ArrayList<>(scanResults.values()));
        }
        // Some vendor Bluetooth stacks can block in the stopScan Binder call. Never
        // let that freeze the activity or prevent the device chooser from opening.
        BluetoothLeScanner activeScanner = scanner;
        if (wasScanning && activeScanner != null) {
            Thread stopThread = new Thread(() -> {
                try { activeScanner.stopScan(scanCallback); } catch (SecurityException ignored) { }
            }, "semg-stop-ble-scan");
            stopThread.setDaemon(true);
            stopThread.start();
        }
    }

    @SuppressLint("MissingPermission")
    void connect(DeviceItem item) {
        stopScan(false);
        disconnect();
        // 记住设备以便断开后自动重连
        reconnectDevice = item.device;
        reconnectName = item.name;
        autoReconnect = true;
        reconnectAttempt = 0;
        listener.onStatus("正在连接 " + item.name + "…", false);
        blog("发起连接：" + item.name + " (" + item.device.getAddress()
                + ")，扫描时信号 " + item.rssi + " dBm");
        subscriptionReady = false;
        servicesDiscovered = false;
        announcedFirstData = false;
        receivedBytes = 0;
        notificationCount = 0;
        smallestNotification = Integer.MAX_VALUE;
        largestNotification = 0;
        invalidWireFrames = 0;
        discoveryAttempts = 0;
        lastRssi = Integer.MIN_VALUE;
        connectedAtMs = 0;
        statsAtMs = 0;
        statsBytesAtMark = 0;
        statsNotificationsAtMark = 0;
        // 深层诊断：重置监控状态
        rssiHistoryIndex = 0;
        java.util.Arrays.fill(rssiHistory, 0);
        lastPacketTimeMs = 0;
        maxPacketIntervalMs = 0;
        totalPacketIntervals = 0;
        silentPeriods = 0;
        rssiSamplesInWindow = 0;
        rssiSumInWindow = 0;
        lastWindowAvgRssi = Integer.MIN_VALUE;
        consecutiveRssiDrop = 0;
        peakRssi = Integer.MIN_VALUE;
        minRssi = Integer.MAX_VALUE;
        gattErrorCount = 0;
        serviceDiscoveryFailures = 0;
        cccdWriteFailures = 0;
        mtuNegotiationFailures = 0;
        lastNotificationTimeMs = 0;
        notificationGaps = 0;
        java.util.Arrays.fill(notificationSizeHistory, 0);
        notificationSizeIndex = 0;
        try {
            connectionPending = true;
            main.removeCallbacks(connectTimeoutRunnable);
            main.postDelayed(connectTimeoutRunnable, CONNECT_TIMEOUT_MS);
            gatt = item.device.connectGatt(context, false, gattCallback, BluetoothDevice.TRANSPORT_LE);
        } catch (SecurityException error) {
            connectionPending = false;
            listener.onStatus("缺少蓝牙连接权限", false);
            eventLog.logError("BLE", "connectGatt 抛出异常：缺少权限", error);
        }
    }

    @SuppressLint("MissingPermission")
    void disconnect() {
        autoReconnect = false;
        connectionPending = false;
        stopReconnectScan();
        main.removeCallbacks(connectTimeoutRunnable);
        main.removeCallbacks(rssiPollRunnable);
        BluetoothGatt old = gatt;
        gatt = null;
        if (old != null) {
            blog("用户手动断开连接");
            try { old.disconnect(); old.close(); } catch (SecurityException ignored) { }
        }
    }

    @SuppressLint("MissingPermission")
    void close() {
        stopScan(false);
        disconnect();
        main.removeCallbacksAndMessages(null);
    }

    /** 安排下一轮重连：先扫描确认设备在广播，扫到才连接。 */
    @SuppressLint("MissingPermission")
    private void scheduleReconnect() {
        if (!autoReconnect || reconnectDevice == null) return;
        reconnectAttempt++;
        final int attempt = reconnectAttempt;
        final String name = reconnectName;
        blog("第 " + attempt + " 轮自动重连：开始搜索设备（" + (RECONNECT_SCAN_MS / 1000) + " 秒）");
        main.post(() -> listener.onReconnecting(name == null ? "臂环" : name, attempt));
        startReconnectScan();
    }

    @SuppressLint("MissingPermission")
    private void startReconnectScan() {
        BluetoothLeScanner current = null;
        if (isBluetoothEnabled() && adapter != null) current = adapter.getBluetoothLeScanner();
        if (current == null) {
            // 蓝牙被关掉或扫描器暂不可用：稍后再试
            main.postDelayed(() -> {
                if (autoReconnect) scheduleReconnect();
            }, RECONNECT_SCAN_MS);
            return;
        }
        reconnectScanning = true;
        main.removeCallbacks(reconnectScanTimeoutRunnable);
        main.postDelayed(reconnectScanTimeoutRunnable, RECONNECT_SCAN_MS);
        try {
            List<ScanFilter> filters = new ArrayList<>();
            filters.add(new ScanFilter.Builder()
                    .setDeviceAddress(reconnectDevice.getAddress()).build());
            ScanSettings settings = new ScanSettings.Builder()
                    .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY).build();
            current.startScan(filters, settings, reconnectScanCallback);
        } catch (SecurityException error) {
            Log.w(TAG, "reconnect scan start failed", error);
            reconnectScanning = false;
        }
    }

    private void stopReconnectScan() {
        main.removeCallbacks(reconnectScanTimeoutRunnable);
        if (!reconnectScanning) return;
        reconnectScanning = false;
        BluetoothLeScanner current = adapter == null ? null : adapter.getBluetoothLeScanner();
        if (current != null) {
            // 与手动扫描一致：stopScan 可能在个别机型上阻塞，放到后台线程执行
            Thread stopThread = new Thread(() -> {
                try { current.stopScan(reconnectScanCallback); } catch (SecurityException ignored) { }
            }, "semg-stop-reconnect-scan");
            stopThread.setDaemon(true);
            stopThread.start();
        }
    }

    /** 扫描确认设备在广播后，发起直连。 */
    @SuppressLint("MissingPermission")
    private void connectSavedDevice() {
        if (!autoReconnect || gatt != null || reconnectDevice == null) return;
        Log.i(TAG, "reconnect attempt " + reconnectAttempt);
        try {
            connectionPending = true;
            main.removeCallbacks(connectTimeoutRunnable);
            main.postDelayed(connectTimeoutRunnable, CONNECT_TIMEOUT_MS);
            gatt = reconnectDevice.connectGatt(context, false, gattCallback, BluetoothDevice.TRANSPORT_LE);
        } catch (SecurityException error) {
            Log.w(TAG, "reconnect connectGatt failed", error);
            connectionPending = false;
            scheduleReconnect();
        }
    }

    private final ScanCallback scanCallback = new ScanCallback() {
        @SuppressLint("MissingPermission")
        @Override public void onScanResult(int callbackType, ScanResult result) {
            BluetoothDevice device = result.getDevice();
            String name;
            try { name = device.getName(); } catch (SecurityException error) { name = null; }
            if (name == null || name.trim().isEmpty()) name = "未命名 BLE 设备";
            boolean likelyArmband = name.toUpperCase().contains("BT-11")
                    || (result.getScanRecord() != null
                    && result.getScanRecord().getServiceUuids() != null
                    && result.getScanRecord().getServiceUuids().stream()
                    .anyMatch(value -> SERVICE_UUID.equals(value.getUuid())));
            // Prefer the armband, but keep named devices available for custom BT-11 names.
            if (likelyArmband || !"未命名 BLE 设备".equals(name)) {
                scanResults.put(device.getAddress(), new DeviceItem(device, name, result.getRssi()));
            }
        }

        @Override public void onScanFailed(int errorCode) {
            scanning = false;
            blog("BLE 扫描失败，错误码 " + errorCode);
            listener.onStatus("BLE 扫描失败（" + errorCode + "）", false);
        }
    };

    private final BluetoothGattCallback gattCallback = new BluetoothGattCallback() {
        @SuppressLint("MissingPermission")
        @Override public void onConnectionStateChange(BluetoothGatt current, int status, int newState) {
            Log.i(TAG, "connection state status=" + status + " state=" + newState);
            if (newState == BluetoothProfile.STATE_CONNECTED && status == BluetoothGatt.GATT_SUCCESS) {
                reconnectAttempt = 0;
                connectionPending = false;
                stopReconnectScan();
                main.removeCallbacks(connectTimeoutRunnable);
                connectedAtMs = System.currentTimeMillis();
                statsAtMs = connectedAtMs;
                blog("GATT 已连接，开始发现服务");
                main.post(() -> listener.onStatus("已连接，正在发现数据服务…", false));
                // GATT operations must be serialized. In particular, several vendor
                // stacks drop discoverServices() when requestMtu() is still pending.
                beginServiceDiscovery(current);
            } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                logDisconnectSnapshot(current, status);
                current.close();
                if (gatt == current) gatt = null;
                connectionPending = false;
                main.removeCallbacks(connectTimeoutRunnable);
                if (autoReconnect && reconnectDevice != null) {
                    // 意外断开：自动重连
                    Log.i(TAG, "unexpected disconnect, scheduling reconnect");
                    scheduleReconnect();
                } else {
                    main.post(() -> listener.onStatus("连接已断开", false));
                }
            }
        }

        /** 断链时输出完整快照：原因、时长、RSSI、传输量、CRC 错误率。 */
        private void logDisconnectSnapshot(BluetoothGatt current, int status) {
            long now = System.currentTimeMillis();
            StringBuilder snapshot = new StringBuilder();
            
            // ===== 第一部分：断开原因 =====
            snapshot.append("========== 连接断开诊断 ==========\n");
            snapshot.append("【断开原因】").append(gattStatusText(status));
            
            // ===== 第二部分：连接时长 =====
            if (connectedAtMs > 0) {
                double duration = (now - connectedAtMs) / 1000.0;
                snapshot.append(String.format(java.util.Locale.CHINA, "\n【连接时长】%.1f 秒", duration));
                if (duration < 10) {
                    snapshot.append("（极短——可能设备未就绪或干扰严重）");
                } else if (duration < 60) {
                    snapshot.append("（较短）");
                }
            }
            
            // ===== 第三部分：信号强度深度分析 =====
            snapshot.append("\n【信号强度】");
            if (lastRssi != Integer.MIN_VALUE) {
                snapshot.append(String.format(java.util.Locale.CHINA,
                        "最后 %d dBm", lastRssi));
                if (lastRssi > -60) snapshot.append("（强）");
                else if (lastRssi > -75) snapshot.append("（中等）");
                else if (lastRssi > -85) snapshot.append("（偏弱——距离过远或遮挡）");
                else snapshot.append("（很弱——即将断连）");
            } else {
                snapshot.append("无RSSI记录（可能未开始轮询或连接极短）");
            }
            
            // RSSI趋势分析
            snapshot.append("\n").append(analyzeRssiTrend());
            
            // ===== 第四部分：数据传输统计 =====
            snapshot.append(String.format(java.util.Locale.CHINA,
                    "\n【数据传输】共收 %d 字节 / %d 包", receivedBytes, notificationCount));
            if (notificationCount > 0) {
                snapshot.append(String.format(java.util.Locale.CHINA,
                        "\n  %s", analyzeNotificationSizes()));
                if (invalidWireFrames > 0) {
                    double errorRate = invalidWireFrames * 100.0 / notificationCount;
                    snapshot.append(String.format(java.util.Locale.CHINA,
                            "\n  CRC错误 %d 次（%.2f%%）", invalidWireFrames, errorRate));
                    if (errorRate > 5) {
                        snapshot.append("——无线干扰严重");
                    } else if (errorRate > 1) {
                        snapshot.append("——存在轻微干扰");
                    }
                }
            }
            if (connectedAtMs > 0 && receivedBytes > 0) {
                double seconds = Math.max(0.001, (now - connectedAtMs) / 1000.0);
                snapshot.append(String.format(java.util.Locale.CHINA,
                        "\n【吞吐率】平均 %.1f KB/s", receivedBytes / 1024.0 / seconds));
            }
            
            // ===== 第五部分：数据包间隔分析 =====
            snapshot.append(String.format(java.util.Locale.CHINA,
                    "\n【包间隔】最大间隔 %d ms", maxPacketIntervalMs));
            if (silentPeriods > 0) {
                snapshot.append(String.format(java.util.Locale.CHINA,
                        "，静默期 %d 次（>500ms无数据）", silentPeriods));
                if (silentPeriods > 5) {
                    snapshot.append("——数据流极不稳定");
                } else if (silentPeriods > 2) {
                    snapshot.append("——数据流偶有中断");
                }
            }
            if (notificationGaps > 0) {
                snapshot.append(String.format(java.util.Locale.CHINA,
                        "，数据流中断 %d 次（>1秒无数据）", notificationGaps));
            }
            
            // ===== 第六部分：GATT操作统计 =====
            snapshot.append(String.format(java.util.Locale.CHINA,
                    "\n【GATT操作】服务发现失败 %d 次", serviceDiscoveryFailures));
            snapshot.append(String.format(java.util.Locale.CHINA,
                    "，CCCD写入失败 %d 次", cccdWriteFailures));
            snapshot.append(String.format(java.util.Locale.CHINA,
                    "，MTU协商失败 %d 次", mtuNegotiationFailures));
            snapshot.append(String.format(java.util.Locale.CHINA,
                    "，GATT错误 %d 次", gattErrorCount));
            
            // ===== 第七部分：断开原因推断 =====
            snapshot.append("\n\n【断开原因推断】");
            String reason = inferDisconnectReason(status);
            snapshot.append(reason);
            
            // ===== 第八部分：解决建议 =====
            snapshot.append("\n\n【解决建议】");
            String suggestion = getSuggestion(status);
            snapshot.append(suggestion);
            
            snapshot.append("\n====================================");
            blog(snapshot.toString());
        }
        
        /** 深层诊断：推断断开原因 */
        private String inferDisconnectReason(int status) {
            StringBuilder reason = new StringBuilder();
            
            // 基于GATT状态码
            switch (status) {
                case 8:
                    reason.append("连接超时(8)：");
                    if (lastRssi < -85) {
                        reason.append("信号极弱导致超时");
                    } else {
                        reason.append("设备可能距离过远或被遮挡");
                    }
                    break;
                case 19:
                    reason.append("远端主动断开(19)：");
                    if (receivedBytes == 0) {
                        reason.append("臂环可能未开机或固件异常");
                    } else if (notificationCount < 10) {
                        reason.append("臂环重启、关机或进入休眠");
                    } else {
                        reason.append("臂环主动断开（正常关机或异常重启）");
                    }
                    break;
                case 133:
                    reason.append("通用GATT错误(133)：");
                    if (serviceDiscoveryFailures > 0) {
                        reason.append("服务发现失败，系统蓝牙栈可能节流");
                    } else {
                        reason.append("常见于系统蓝牙栈节流或设备未广播");
                    }
                    break;
                case 22:
                    reason.append("本机控制器断开(22)：");
                    if (gattErrorCount > 5) {
                        reason.append("GATT操作频繁失败导致控制器断开");
                    } else {
                        reason.append("手机蓝牙栈异常或系统回收");
                    }
                    break;
                default:
                    reason.append("状态码").append(status).append("：");
                    if (status == 0 && lastRssi < -90) {
                        reason.append("虽然状态成功但信号极弱，可能随后断开");
                    } else {
                        reason.append("需要结合其他指标分析");
                    }
            }
            
            // 结合信号趋势
            if (consecutiveRssiDrop >= 3) {
                reason.append("\n  + 信号持续恶化（").append(consecutiveRssiDrop).append("次连续下降）");
            }
            
            // 结合数据流质量
            if (silentPeriods > 5) {
                reason.append("\n  + 数据流严重不稳定（").append(silentPeriods).append("次静默期）");
            } else if (notificationGaps > 3) {
                reason.append("\n  + 数据流频繁中断（").append(notificationGaps).append("次）");
            }
            
            // 结合CRC错误
            if (invalidWireFrames > 10) {
                reason.append("\n  + CRC错误率高（").append(invalidWireFrames).append("次），无线干扰严重");
            }
            
            return reason.toString();
        }
        
        /** 深层诊断：提供解决建议 */
        private String getSuggestion(int status) {
            StringBuilder suggestion = new StringBuilder();
            
            // 信号相关
            if (lastRssi < -85 || minRssi < -90) {
                suggestion.append("1. 缩短手机与臂环距离（建议<3米）\n");
                suggestion.append("   当前最低信号 ").append(minRssi).append(" dBm，过弱\n");
            }
            if (consecutiveRssiDrop >= 3) {
                suggestion.append("2. 检查是否有遮挡物（人体、金属、墙壁）\n");
                suggestion.append("   信号持续恶化趋势明显\n");
            }
            
            // 干扰相关
            if (invalidWireFrames > 5) {
                suggestion.append("3. 远离WiFi路由器、微波炉等2.4GHz干扰源\n");
                suggestion.append("   CRC错误 ").append(invalidWireFrames).append(" 次\n");
            }
            if (silentPeriods > 3) {
                suggestion.append("4. 数据流不稳定，可能存在同频段设备干扰\n");
            }
            
            // 设备相关
            if (status == 19 && receivedBytes == 0) {
                suggestion.append("5. 确认臂环已开机且工作在BLE模式\n");
                suggestion.append("   未收到任何数据\n");
            }
            if (serviceDiscoveryFailures > 0) {
                suggestion.append("6. 尝试关闭手机蓝牙后重新打开\n");
                suggestion.append("   服务发现失败 ").append(serviceDiscoveryFailures).append(" 次\n");
            }
            if (gattErrorCount > 5) {
                suggestion.append("7. GATT操作频繁失败，建议重启手机蓝牙\n");
            }
            
            // 系统相关
            if (status == 133) {
                suggestion.append("8. 系统蓝牙栈节流：关闭后台其他蓝牙应用\n");
                suggestion.append("   或尝试重启手机\n");
            }
            
            if (suggestion.length() == 0) {
                suggestion.append("断开原因不明确，请查看完整日志分析");
            }
            
            return suggestion.toString();
        }

        @SuppressLint("MissingPermission")
        @Override public void onServicesDiscovered(BluetoothGatt current, int status) {
            servicesDiscovered = true;
            Log.i(TAG, "services discovered status=" + status + " count=" + current.getServices().size());
            if (status != BluetoothGatt.GATT_SUCCESS) {
                serviceDiscoveryFailures++;
                blog("服务发现失败：" + gattStatusText(status) + "（累计" + serviceDiscoveryFailures + "次）");
                main.post(() -> listener.onStatus("数据服务发现失败（GATT " + status + "）", false));
                current.disconnect();
                return;
            }
            // Match the desktop/Bleak implementation: locate FFE2 across every
            // service instead of rejecting modules whose service UUID was customized.
            BluetoothGattCharacteristic notify = findCharacteristic(current, NOTIFY_UUID);
            if (notify == null) {
                String available = availableCharacteristics(current);
                Log.w(TAG, "FFE2 missing; characteristics=" + available);
                blog("服务发现完成（" + current.getServices().size() + " 个服务），但没有 FFE2 特征，实际特征：" + available);
                main.post(() -> listener.onStatus("设备没有 FFE2 通知特征；实际特征：" + available, false));
                current.disconnect();
                return;
            }
            blog("服务发现完成（" + current.getServices().size() + " 个服务），正在订阅 FFE2");
            boolean enabled = current.setCharacteristicNotification(notify, true);
            BluetoothGattDescriptor cccd = notify.getDescriptor(CCCD_UUID);
            if (!enabled || cccd == null) {
                blog("订阅 FFE2 失败：setCharacteristicNotification=" + enabled + " cccd=" + (cccd != null));
                main.post(() -> listener.onStatus("无法订阅 FFE2 数据通知", false));
                return;
            }
            byte[] cccdValue = (notify.getProperties() & BluetoothGattCharacteristic.PROPERTY_NOTIFY) != 0
                    ? BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
                    : BluetoothGattDescriptor.ENABLE_INDICATION_VALUE;
            boolean started;
            if (Build.VERSION.SDK_INT >= 33) {
                started = current.writeDescriptor(cccd, cccdValue) == BluetoothGatt.GATT_SUCCESS;
            } else {
                cccd.setValue(cccdValue);
                started = current.writeDescriptor(cccd);
            }
            if (!started) {
                cccdWriteFailures++;
                blog("写入 CCCD 失败（启动数据通知失败，累计" + cccdWriteFailures + "次）");
                main.post(() -> listener.onStatus("启动数据通知失败", false));
            }
        }

        @Override public void onDescriptorWrite(BluetoothGatt current, BluetoothGattDescriptor descriptor, int status) {
            Log.i(TAG, "CCCD write status=" + status + " uuid=" + descriptor.getUuid());
            subscriptionReady = status == BluetoothGatt.GATT_SUCCESS;
            blog("CCCD 写入" + (subscriptionReady ? "成功，已订阅数据通知" : "失败：" + gattStatusText(status)));
            main.post(() -> listener.onStatus(status == BluetoothGatt.GATT_SUCCESS
                    ? "已连接 · 已订阅 FFE2，等待肌电数据" : "订阅数据失败（" + status + "）",
                    status == BluetoothGatt.GATT_SUCCESS));
            if (status == BluetoothGatt.GATT_SUCCESS) {
                // Now that service discovery and CCCD writing are complete, MTU
                // negotiation is safe. A larger ATT payload and short connection
                // interval reduce notification loss at the 8.8 kB/s stream rate.
                if (!current.requestMtu(247)) {
                    current.requestConnectionPriority(BluetoothGatt.CONNECTION_PRIORITY_HIGH);
                }
                // 订阅就绪后启动 RSSI 轮询与周期统计
                main.removeCallbacks(rssiPollRunnable);
                main.postDelayed(rssiPollRunnable, RSSI_POLL_MS);
                main.postDelayed(() -> {
                    if (gatt == current && subscriptionReady && receivedBytes == 0) {
                        listener.onStatus("FFE2 已订阅，但没有数据；请长按手环 SW2 三秒启动采集", true);
                    }
                }, 4000);
            }
        }

        @SuppressLint("MissingPermission")
        @Override public void onMtuChanged(BluetoothGatt current, int mtu, int status) {
            Log.i(TAG, "MTU changed mtu=" + mtu + " status=" + status);
            if (status == BluetoothGatt.GATT_SUCCESS) {
                blog("MTU 协商成功：" + mtu + " 字节");
            } else {
                mtuNegotiationFailures++;
                blog("⚠️ MTU 协商失败：" + gattStatusText(status) + "（累计" + mtuNegotiationFailures + "次）");
            }
            current.requestConnectionPriority(BluetoothGatt.CONNECTION_PRIORITY_HIGH);
        }

        @SuppressLint("MissingPermission")
        @Override public void onReadRemoteRssi(BluetoothGatt current, int rssi, int status) {
            if (status != BluetoothGatt.GATT_SUCCESS) {
                gattErrorCount++;
                blog("⚠️ RSSI读取失败（GATT错误，累计" + gattErrorCount + "次）");
                return;
            }
            lastRssi = rssi;
            recordRssiHistory(rssi);
            long now = System.currentTimeMillis();
            // 每 10 秒记录一次信号强度；信号快速恶化时立即记录
            if (lastRssiLogMs == 0 || now - lastRssiLogMs >= RSSI_LOG_MS
                    || (rssi < -85 && lastRssiLogMs > 0 && now - lastRssiLogMs >= 2000)) {
                lastRssiLogMs = now;
                String quality = rssi > -60 ? "强" : rssi > -75 ? "中等" : rssi > -85 ? "偏弱" : "很弱";
                blog(String.format(java.util.Locale.CHINA, "信号强度 %d dBm（%s）", rssi, quality));
            }
            // 信号突然恶化立即告警
            if (rssi < -85 && peakRssi > -70) {
                blog(String.format(java.util.Locale.CHINA,
                        "⚠️ 信号骤降：从峰值%d dBm降至%d dBm（下降%d dB），可能存在遮挡或干扰",
                        peakRssi, rssi, peakRssi - rssi));
            }
        }

        @Override public void onCharacteristicChanged(BluetoothGatt current,
                                                       BluetoothGattCharacteristic characteristic) {
            if (NOTIFY_UUID.equals(characteristic.getUuid())) deliver(characteristic.getValue());
        }

        @Override public void onCharacteristicChanged(BluetoothGatt current,
                                                       BluetoothGattCharacteristic characteristic,
                                                       byte[] value) {
            if (NOTIFY_UUID.equals(characteristic.getUuid())) deliver(value);
        }

        private void deliver(byte[] value) {
            if (value == null || value.length == 0) return;
            
            // 深层诊断：监控数据包间隔
            long now = System.currentTimeMillis();
            if (lastPacketTimeMs > 0) {
                long interval = now - lastPacketTimeMs;
                totalPacketIntervals++;
                if (interval > maxPacketIntervalMs) {
                    maxPacketIntervalMs = interval;
                    if (interval > MAX_NORMAL_INTERVAL_MS) {
                        silentPeriods++;
                        blog(String.format(java.util.Locale.CHINA,
                                "⚠️ 数据包间隔异常：本次%d ms（最大%d ms），静默期累计%d次",
                                interval, maxPacketIntervalMs, silentPeriods));
                    }
                }
            }
            lastPacketTimeMs = now;
            
            // 深层诊断：监控通知间隙
            if (lastNotificationTimeMs > 0) {
                long gap = now - lastNotificationTimeMs;
                if (gap > 1000) { // 超过1秒
                    notificationGaps++;
                    blog(String.format(java.util.Locale.CHINA,
                            "⚠️ 数据流中断：%d ms无数据（累计%d次）", gap, notificationGaps));
                }
            }
            lastNotificationTimeMs = now;
            
            // 深层诊断：记录包大小历史
            notificationSizeHistory[notificationSizeIndex % notificationSizeHistory.length] = value.length;
            notificationSizeIndex++;
            
            receivedBytes += value.length;
            notificationCount++;
            smallestNotification = Math.min(smallestNotification, value.length);
            largestNotification = Math.max(largestNotification, value.length);
            logPeriodicStats();
            if (!announcedFirstData) {
                announcedFirstData = true;
                blog("收到首包数据（" + value.length + " 字节），数据链路正常");
                main.post(() -> listener.onStatus("已连接 · 正在接收肌电数据", true));
            }
            byte[] copy = value.clone();
            diagnoseCompleteWireFrame(copy);
            listener.onBytes(copy);
        }

        /** 每 30 秒记录一次传输统计：吞吐率、包长分布、CRC 错误率。 */
        private void logPeriodicStats() {
            long now = System.currentTimeMillis();
            if (statsAtMs == 0 || now - statsAtMs < STATS_LOG_MS) return;
            double seconds = (now - statsAtMs) / 1000.0;
            long bytes = receivedBytes - statsBytesAtMark;
            long notifications = notificationCount - statsNotificationsAtMark;
            blog(String.format(java.util.Locale.CHINA,
                    "传输统计：%.1f KB/s · %d 包/秒 · 包长 %d–%d · 累计CRC错误 %d",
                    bytes / 1024.0 / seconds, Math.round(notifications / seconds),
                    smallestNotification, largestNotification, invalidWireFrames));
            statsAtMs = now;
            statsBytesAtMark = receivedBytes;
            statsNotificationsAtMark = notificationCount;
        }
    };

    @SuppressLint("MissingPermission")
    private void beginServiceDiscovery(BluetoothGatt current) {
        if (gatt != current) return;
        discoveryAttempts++;
        boolean started = current.discoverServices();
        Log.i(TAG, "discoverServices attempt=" + discoveryAttempts + " started=" + started);
        if (!started) {
            main.post(() -> listener.onStatus("无法启动 BLE 数据服务发现", false));
            current.disconnect();
            return;
        }
        int attempt = discoveryAttempts;
        main.postDelayed(() -> {
            if (gatt != current || servicesDiscovered || discoveryAttempts != attempt) return;
            if (attempt < 2) {
                listener.onStatus("数据服务发现超时，正在重试…", false);
                beginServiceDiscovery(current);
            } else {
                listener.onStatus("数据服务发现超时，请断开后重试", false);
                current.disconnect();
            }
        }, 7000);
    }

    private static BluetoothGattCharacteristic findCharacteristic(BluetoothGatt current, UUID uuid) {
        for (BluetoothGattService service : current.getServices()) {
            BluetoothGattCharacteristic characteristic = service.getCharacteristic(uuid);
            if (characteristic != null) return characteristic;
        }
        return null;
    }

    private static String availableCharacteristics(BluetoothGatt current) {
        List<String> values = new ArrayList<>();
        for (BluetoothGattService service : current.getServices()) {
            for (BluetoothGattCharacteristic characteristic : service.getCharacteristics()) {
                String uuid = characteristic.getUuid().toString();
                values.add(uuid.length() >= 8 ? uuid.substring(4, 8).toUpperCase() : uuid);
            }
        }
        return values.isEmpty() ? "无" : String.join(", ", values);
    }

    private void diagnoseCompleteWireFrame(byte[] frame) {
        if (frame.length < 16 || (frame[0] & 0xFF) != 0xA5 || (frame[1] & 0xFF) != 0x5A) return;
        int sampleCount = (frame[12] & 0xFF) | (frame[13] & 0xFF) << 8;
        int frameSize = 14 + sampleCount * SemgPacketParser.CHANNELS * 2 + 2;
        if (frameSize != frame.length) return;
        int expected = (frame[frame.length - 2] & 0xFF) | (frame[frame.length - 1] & 0xFF) << 8;
        int actual = SemgPacketParser.crc16(frame, 0, frame.length - 2);
        if (expected != actual) {
            invalidWireFrames++;
            if (invalidWireFrames <= 5 || invalidWireFrames % 100 == 0) {
                long sequence = (long) (frame[4] & 0xFF) | (long) (frame[5] & 0xFF) << 8
                        | (long) (frame[6] & 0xFF) << 16 | (long) (frame[7] & 0xFF) << 24;
                Log.w(TAG, "wire CRC mismatch #" + invalidWireFrames + " seq=" + sequence
                        + " expected=" + expected + " actual=" + actual);
                blog(String.format(java.util.Locale.CHINA,
                        "数据帧 CRC 错误 #%d（seq=%d）——可能存在无线干扰或丢包", invalidWireFrames, sequence));
            }
        }
    }

    private static UUID uuid16(String value) {
        return UUID.fromString("0000" + value.toLowerCase() + "-0000-1000-8000-00805f9b34fb");
    }
}