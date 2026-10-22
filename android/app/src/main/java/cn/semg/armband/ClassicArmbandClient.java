package cn.semg.armband;

import android.annotation.SuppressLint;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothManager;
import android.bluetooth.BluetoothSocket;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * 经典蓝牙 SPP（RFCOMM 串口）客户端，面向经典蓝牙臂环。
 * 数据以字节流方式从 InputStream 读出，直接交给与 BLE 相同的解析器。
 */
final class ClassicArmbandClient {
    private static final String TAG = "SemgClassic";
    /** 标准 SPP 串口 UUID，绝大多数经典蓝牙串口模块（HC-05、BT-04 等）使用该 UUID。 */
    private static final UUID SPP_UUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB");
    private static final long DISCOVERY_TIMEOUT_MS = 12000;
    private static final long RECONNECT_INTERVAL_MS = 3000;

    interface Listener {
        void onStatus(String text, boolean connected);
        void onBytes(byte[] bytes);
        void onClassicScanFinished(List<DeviceItem> devices);
        /** 意外断开后开始自动重连时回调（attempt 从 1 开始）。 */
        default void onReconnecting(String deviceName, int attempt) { }
    }

    static final class DeviceItem {
        final BluetoothDevice device;
        final String name;
        final boolean bonded;

        DeviceItem(BluetoothDevice device, String name, boolean bonded) {
            this.device = device;
            this.name = name;
            this.bonded = bonded;
        }

        @Override public String toString() {
            return name + "\n" + device.getAddress() + (bonded ? "   已配对" : "   未配对");
        }
    }

    private final Context context;
    private final Listener listener;
    private final Handler main = new Handler(Looper.getMainLooper());
    private final BluetoothAdapter adapter;
    private final BleEventLog eventLog;

    BleEventLog eventLog() { return eventLog; }

    private final Map<String, DeviceItem> scanResults = new LinkedHashMap<>();
    private boolean discovering;
    private BluetoothDevice targetDevice;
    private String targetName;
    private volatile BluetoothSocket socket;
    private volatile boolean userInitiatedClose;
    private volatile boolean reconnecting;
    private volatile boolean ioRunning;
    private Thread connectThread;
    private Thread ioThread;
    private int reconnectAttempt;

    ClassicArmbandClient(Context context, Listener listener, BleEventLog sharedLog) {
        this.context = context.getApplicationContext();
        this.listener = listener;
        BluetoothManager manager = context.getSystemService(BluetoothManager.class);
        adapter = manager == null ? null : manager.getAdapter();
        eventLog = sharedLog;
    }

    boolean isBluetoothEnabled() {
        return adapter != null && adapter.isEnabled();
    }

    private void blog(String message) { eventLog.log("经典蓝牙", message); }

    // ==================== 扫描（已配对设备 + 周边发现） ====================

    private final BroadcastReceiver discoveryReceiver = new BroadcastReceiver() {
        @Override public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (BluetoothDevice.ACTION_FOUND.equals(action)) {
                BluetoothDevice device = intent.getParcelableExtra(BluetoothDevice.EXTRA_DEVICE);
                if (device == null) return;
                String name;
                try { name = device.getName(); } catch (SecurityException ignored) { name = null; }
                boolean bonded;
                try { bonded = device.getBondState() == BluetoothDevice.BOND_BONDED; }
                catch (SecurityException ignored) { bonded = false; }
                if (name == null || name.trim().isEmpty()) name = device.getAddress();
                // 音频类设备对 SPP 无意义，跳过明显的耳机/音箱
                String lower = name.toLowerCase(java.util.Locale.ROOT);
                if (lower.contains("headset") || lower.contains("headphone") || lower.contains("耳机")
                        || lower.contains("speaker") || lower.contains("音箱")) return;
                scanResults.put(device.getAddress(), new DeviceItem(device, name, bonded));
            } else if (BluetoothAdapter.ACTION_DISCOVERY_FINISHED.equals(action)) {
                finishDiscovery();
            }
        }
    };

    @SuppressLint("MissingPermission")
    void scan() {
        if (adapter == null) { listener.onStatus("本机不支持经典蓝牙", false); return; }
        if (discovering) return;
        scanResults.clear();
        // 已配对设备立即可连
        try {
            for (BluetoothDevice device : adapter.getBondedDevices()) {
                String name = device.getName();
                if (name == null || name.trim().isEmpty()) name = device.getAddress();
                scanResults.put(device.getAddress(), new DeviceItem(device, name, true));
            }
        } catch (SecurityException error) {
            blog("读取已配对设备失败：缺少权限");
        }
        discovering = true;
        listener.onStatus("正在搜索经典蓝牙设备…（已配对设备可直接选择）", false);
        blog("开始扫描经典蓝牙设备，已配对 " + scanResults.size() + " 台");
        IntentFilter filter = new IntentFilter(BluetoothDevice.ACTION_FOUND);
        filter.addAction(BluetoothAdapter.ACTION_DISCOVERY_FINISHED);
        try {
            context.registerReceiver(discoveryReceiver, filter);
            if (!adapter.startDiscovery()) blog("启动设备发现失败（将只显示已配对设备）");
        } catch (SecurityException error) {
            blog("启动设备发现失败：缺少权限（将只显示已配对设备）");
        }
        main.postDelayed(() -> finishDiscovery(), DISCOVERY_TIMEOUT_MS);
    }

    @SuppressLint("MissingPermission")
    private void finishDiscovery() {
        if (!discovering) return;
        discovering = false;
        main.removeCallbacksAndMessages(null);
        try { context.unregisterReceiver(discoveryReceiver); } catch (IllegalArgumentException ignored) { }
        try { adapter.cancelDiscovery(); } catch (SecurityException ignored) { }
        blog("扫描完成：共发现 " + scanResults.size() + " 台设备");
        listener.onClassicScanFinished(new ArrayList<>(scanResults.values()));
    }

    // ==================== 连接与数据收发 ====================

    void connect(DeviceItem item) {
        finishDiscovery();
        disconnect();
        targetDevice = item.device;
        targetName = item.name;
        userInitiatedClose = false;
        reconnecting = false;
        reconnectAttempt = 0;
        listener.onStatus("正在连接 " + item.name + "…", false);
        blog("发起经典蓝牙连接：" + item.name + " (" + item.device.getAddress() + ")，"
                + (item.bonded ? "已配对" : "未配对（连接时可能弹出配对请求）"));
        openSocket(false);
    }

    /** openSocket 在后台线程执行；reconnect 为 true 时静默失败进入重连循环。 */
    private void openSocket(boolean reconnect) {
        final BluetoothDevice device = targetDevice;
        if (device == null) return;
        if (connectThread != null && connectThread.isAlive()) return;
        connectThread = new Thread(() -> {
            try {
                BluetoothSocket newSocket = device.createRfcommSocketToServiceRecord(SPP_UUID);
                long start = System.currentTimeMillis();
                newSocket.connect();
                socket = newSocket;
                ioRunning = true;
                reconnecting = false;
                reconnectAttempt = 0;
                blog("SPP 连接建立，耗时 " + (System.currentTimeMillis() - start) + " ms");
                main.post(() -> listener.onStatus("已连接 · 经典蓝牙 SPP 串口已建立", true));
                startReadLoop(newSocket);
            } catch (IOException | SecurityException error) {
                try { socket = null; } catch (Throwable ignored) { }
                if (reconnect && !userInitiatedClose) {
                    scheduleReconnect();
                } else if (!userInitiatedClose) {
                    blog("SPP 连接失败：" + error.getMessage());
                    main.post(() -> listener.onStatus("经典蓝牙连接失败：" + error.getMessage()
                            + "\n请确认设备已开机并完成配对", false));
                }
            }
        }, "spp-connect");
        connectThread.start();
    }

    /** 读循环：阻塞读 InputStream，字节流交给解析器；读到 -1 或异常即视为断开。 */
    private void startReadLoop(BluetoothSocket activeSocket) {
        ioThread = new Thread(() -> {
            byte[] buffer = new byte[4096];
            long bytes = 0;
            try {
                InputStream in = activeSocket.getInputStream();
                while (ioRunning && socket == activeSocket) {
                    int read = in.read(buffer);
                    if (read < 0) break;
                    if (read > 0) {
                        bytes += read;
                        byte[] chunk = new byte[read];
                        System.arraycopy(buffer, 0, chunk, 0, read);
                        listener.onBytes(chunk);
                    }
                }
            } catch (IOException | SecurityException error) {
                Log.w(TAG, "SPP read loop ended", error);
            }
            boolean unexpected = !userInitiatedClose && socket == activeSocket;
            try { activeSocket.close(); } catch (IOException ignored) { }
            if (socket == activeSocket) socket = null;
            ioRunning = false;
            if (unexpected) {
                blog("连接断开（共收 " + bytes + " 字节），开始自动重连");
                scheduleReconnect();
            }
        }, "spp-read");
        ioThread.start();
    }

    // ==================== 自动重连 ====================

    private void scheduleReconnect() {
        if (userInitiatedClose || targetDevice == null) return;
        reconnecting = true;
        reconnectAttempt++;
        final int attempt = reconnectAttempt;
        final String name = targetName;
        blog("第 " + attempt + " 次自动重连（每 3 秒重试）");
        main.post(() -> listener.onReconnecting(name == null ? "臂环" : name, attempt));
        main.postDelayed(() -> {
            if (!userInitiatedClose && reconnecting) openSocket(true);
        }, RECONNECT_INTERVAL_MS);
    }

    // ==================== 断开 ====================

    void disconnect() {
        userInitiatedClose = true;
        reconnecting = false;
        reconnectAttempt = 0;
        main.removeCallbacksAndMessages(null);
        BluetoothSocket current = socket;
        socket = null;
        ioRunning = false;
        if (current != null) {
            blog("用户手动断开经典蓝牙连接");
            try { current.close(); } catch (IOException ignored) { }
        }
    }
}
