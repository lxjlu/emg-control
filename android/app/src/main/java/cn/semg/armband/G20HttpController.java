package cn.semg.armband;

import android.content.Context;
import android.content.SharedPreferences;

import cn.semg.armband.control.ControlCommand;
import cn.semg.armband.control.ControlEndpoint;
import cn.semg.armband.control.ControlLink;
import cn.semg.armband.control.LegacyUdpLink;
import cn.semg.armband.control.UdpControlLink;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * 控制门面：负责手势映射、锁存与心跳，具体报文交给 {@link ControlLink}。
 *
 * <p>对外 API 与旧版保持一致，{@code MainActivity} 无需改动。
 */
final class G20HttpController {
    interface Listener { void onState(boolean enabled, String message, boolean warning); }

    enum DeviceType {
        MECHANICAL_CAR("机械车"),
        SMART_CAR("智能车"),
        GO2("Go2机械狗"),
        RECON_BALL("侦察球"),
        GO2W("Go2轮式");

        final String display;
        DeviceType(String display) { this.display = display; }
    }

    enum Command {
        LEFT, RIGHT, FORWARD, BACKWARD, STOP, GRIPPER_OPEN, GRIPPER_CLOSE
    }

    private static final long SEND_PERIOD_MS = 50;
    private static final long RECOGNITION_TIMEOUT_MS = 3000;  // 识别超时：3秒无新结果才停止
    private static final long DEFAULT_CONTROL_TIMEOUT_MS = 30000;
    private static final double MIN_CONFIDENCE = 0.48;
    private static final String PREFS_NAME = "car_controller_prefs";
    private static final String KEY_IP = "target_ip";
    private static final String KEY_PORT = "target_port";
    private static final String KEY_TYPE = "device_type";
    private static final String KEY_MAP_PREFIX = "map_cmd_";
    private static final String KEY_CONTROL_TIMEOUT_MS = "control_timeout_ms";
    private static final String KEY_PROTOCOL = "control_protocol";

    // v2 = 对接 emg_sdk；legacy = 兼容旧中转脚本（默认，行为与重构前一致）
    private static final String PROTOCOL_V2 = "v2";
    private static final String PROTOCOL_LEGACY = "legacy";

    private static final String BALL_DEFAULT_IP = "192.168.1.11";
    private static final int BALL_DEFAULT_PORT = 3020;

    private final Context context;
    private final Listener listener;
    private final ScheduledExecutorService io = Executors.newSingleThreadScheduledExecutor();
    private final SharedPreferences prefs;
    private final Map<String, Command> labelCommandMap = new LinkedHashMap<>();

    private volatile boolean enabled;
    private volatile Command active;
    private volatile long lastRecognitionAt;
    private volatile long lastAnyRecognitionAt;
    private volatile boolean stopSent = true;
    private volatile boolean manualControl = false;  // 手动控制模式，不受识别超时限制
    private volatile boolean neutralConfirmed;   // 是否回到过"休息/不确定"中立位，未回中立前忽略手势跳变

    private String targetIp = "192.168.2.10";
    private int targetPort = 8888;
    private DeviceType deviceType = DeviceType.MECHANICAL_CAR;
    private long controlTimeoutMs = DEFAULT_CONTROL_TIMEOUT_MS;
    private String protocol = PROTOCOL_LEGACY;
    private ControlLink link;

    G20HttpController(Context context, Listener listener) {
        this.context = context.getApplicationContext();
        this.listener = listener;
        this.prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        loadConfig();
        loadMappingConfig();
        io.scheduleAtFixedRate(this::heartbeat, 0, SEND_PERIOD_MS, TimeUnit.MILLISECONDS);
    }

    void setConfig(String ip, int port, DeviceType type) {
        this.targetIp = ip;
        this.targetPort = port;
        this.deviceType = type;
        saveConfig();
    }

    String getTargetIp() { return targetIp; }
    int getTargetPort() { return targetPort; }
    DeviceType getDeviceType() { return deviceType; }
    long getControlTimeoutMs() { return controlTimeoutMs; }

    void setControlTimeoutMs(long timeoutMs) {
        this.controlTimeoutMs = Math.max(1000, timeoutMs);
        saveConfig();
    }

    boolean isEnabled() { return enabled; }

    void enable() {
        io.execute(() -> {
            try {
                closeLink();
                link = createLink();
                link.open(new ControlEndpoint(targetIp, targetPort, linkType(deviceType)));
                link.onEnable();
                enabled = true;
                active = null;
                stopSent = true;
                neutralConfirmed = true;   // 启用后首个动作可直接执行，无需先回中立
                long now = System.currentTimeMillis();
                lastRecognitionAt = now;
                lastAnyRecognitionAt = now;
                listener.onState(true, "控制已启用 · " + deviceType.display + " · " + targetIp + ":" + targetPort, false);
            } catch (Exception error) {
                closeLink();
                enabled = false;
                listener.onState(false, "连接失败：" + shortMessage(error), true);
            }
        });
    }

    void sendManualCommand(Command command) {
        io.execute(() -> {
            try {
                if (command == Command.STOP) {
                    manualControl = false;
                    active = null;
                    stopSent = false;
                    safeSendStop();
                    stopSent = true;
                    listener.onState(enabled, "手动控制 · 停止", false);
                } else {
                    // 持续发送：设置 active 让 heartbeat 线程每 50ms 重发
                    manualControl = true;
                    active = command;
                    stopSent = false;
                    lastRecognitionAt = System.currentTimeMillis();
                    lastAnyRecognitionAt = System.currentTimeMillis();
                    sendCommand(toControlCommand(command));
                    listener.onState(enabled, "手动控制 · " + displayFor(command), false);
                }
            } catch (Exception error) {
                listener.onState(enabled, "发送失败：" + shortMessage(error), true);
            }
        });
    }

    void disable(String reason) {
        boolean wasEnabled = enabled;
        enabled = false;
        active = null;
        stopSent = true;
        manualControl = false;
        neutralConfirmed = false;
        io.execute(() -> {
            if (wasEnabled && link != null) {
                link.onDisable();
            }
            listener.onState(false, reason, false);
        });
    }

    void stopMotion(String reason) {
        if (!enabled) return;
        active = null;
        if (!stopSent) io.execute(this::safeSendStop);
        stopSent = true;
        listener.onState(true, reason, false);
    }

    void onRecognition(String label, double confidence) {
        if (!enabled) return;
        manualControl = false;  // 收到手势识别结果，切换回手势控制模式
        long now = System.currentTimeMillis();
        lastAnyRecognitionAt = now;
        Command command = confidence >= MIN_CONFIDENCE ? commandFor(label) : null;

        if (command == null) {
            // 中性位：识别到"休息"/低置信度"不确定"/未映射手势。
            // 回到中立：停止动作并解锁锁存，之后才允许切换到新命令。
            neutralConfirmed = true;
            if (active != null && !stopSent) io.execute(this::safeSendStop);
            if (active != null) stopSent = true;
            active = null;
            listener.onState(true, "控制中 · 已回休息/不确定，已停止", false);
            return;
        }

        if (command == Command.STOP) {
            active = null;
            if (!stopSent) io.execute(this::safeSendStop);
            stopSent = true;
            neutralConfirmed = true;   // 停止命令同样视为中立，允许后续换手势
            listener.onState(true, "控制中 · 已识别停止", false);
            return;
        }

        if (command == active) { lastRecognitionAt = now; return; }   // 仍是同一命令：保持并续期，避免误报"识别中断"
        if (!neutralConfirmed) return;   // 未经过休息/不确定 → 忽略手势直接跳变（如 左->右）
        active = command;
        neutralConfirmed = false;        // 进入新命令后必须再回中立才能再换手势
        lastRecognitionAt = now;
        stopSent = false;
        listener.onState(true, "控制中 · " + displayFor(command), false);
    }

    private void heartbeat() {
        if (!enabled) return;
        long now = System.currentTimeMillis();
        // 手动控制模式跳过识别超时检查，持续发送命令
        if (!manualControl && now - lastAnyRecognitionAt > controlTimeoutMs) {
            disable("控制超时（" + (controlTimeoutMs / 1000) + " s 无识别结果），已停止");
            return;
        }
        Command command = active;
        if (command == null) {
            if (!stopSent) {
                safeSendStop();
                stopSent = true;
            }
            return;
        }
        if (!manualControl && System.currentTimeMillis() - lastRecognitionAt > RECOGNITION_TIMEOUT_MS) {
            active = null;
            if (!stopSent) safeSendStop();
            stopSent = true;
            listener.onState(true, "控制中 · 识别中断，已停止", true);
            return;
        }
        try {
            sendCommand(toControlCommand(command));
        } catch (Exception error) {
            enabled = false;
            active = null;
            safeSendStop();
            listener.onState(false, "发送失败：" + shortMessage(error), true);
        }
    }

    private void safeSendStop() {
        try {
            sendCommand(ControlCommand.stop());
        } catch (Exception ignored) {
        }
    }

    /** 发一条语义命令；通道未建立时静默忽略（与旧版行为一致）。 */
    private void sendCommand(ControlCommand command) throws Exception {
        ControlLink current = link;
        if (current == null) return;
        current.send(command);
    }

    /** 把界面/手势层的 Command 翻译成语义命令。 */
    private static ControlCommand toControlCommand(Command command) {
        switch (command) {
            case FORWARD:
                return ControlCommand.move(1f, 0f);
            case BACKWARD:
                return ControlCommand.move(-1f, 0f);
            case LEFT:
                return ControlCommand.move(0f, 1f);
            case RIGHT:
                return ControlCommand.move(0f, -1f);
            case GRIPPER_OPEN:
                return ControlCommand.of(ControlCommand.OPEN);
            case GRIPPER_CLOSE:
                return ControlCommand.of(ControlCommand.CLOSE);
            case STOP:
            default:
                return ControlCommand.stop();
        }
    }

    private ControlLink createLink() {
        return PROTOCOL_V2.equals(protocol) ? new UdpControlLink() : new LegacyUdpLink();
    }

    /** 设备类型 → 通道类型（旧协议编码用；v2 只用于日志与能力兜底）。 */
    private static String linkType(DeviceType type) {
        switch (type) {
            case SMART_CAR:
                return "smart_car";
            case GO2:
                return "go2";
            case GO2W:
                return "go2w";
            case RECON_BALL:
                return "ball";
            case MECHANICAL_CAR:
            default:
                return "car";
        }
    }

    private void closeLink() {
        ControlLink current = link;
        link = null;
        if (current != null) {
            current.close();
        }
    }

    private Command commandFor(String raw) {
        if (raw == null) return null;
        String label = normalizeLabel(raw);
        Command mapped = labelCommandMap.get(label);
        if (mapped != null && supportsCommand(mapped)) return mapped;
        // 标准手势标签默认映射：休息→停止, 握拳→后退, 手掌→前进, 向右→右转, 向左→左转
        if (label.equals("休息") || label.equals("rest")) return Command.STOP;
        if (label.equals("握拳") || label.equals("fist")) return Command.BACKWARD;
        if (label.equals("手掌") || label.equals("palm") || label.equals("openhand")) return Command.FORWARD;
        if (label.equals("向左") || label.equals("左") || label.equals("左转") || label.equals("left")) return Command.LEFT;
        if (label.equals("向右") || label.equals("右") || label.equals("右转") || label.equals("right")) return Command.RIGHT;
        // 兼容旧版别名
        if (label.equals("前进") || label.equals("向前") || label.equals("forward")) return Command.FORWARD;
        if (label.equals("倒退") || label.equals("后退") || label.equals("向后") || label.equals("backward")) return Command.BACKWARD;
        if (label.equals("停止") || label.equals("停") || label.equals("stop")) return Command.STOP;
        if (supportsCommand(Command.GRIPPER_OPEN)
                && (label.equals("张开") || label.equals("开爪") || label.equals("open"))) return Command.GRIPPER_OPEN;
        if (supportsCommand(Command.GRIPPER_CLOSE)
                && (label.equals("并拢") || label.equals("合爪") || label.equals("关闭") || label.equals("close"))) return Command.GRIPPER_CLOSE;
        return null;
    }

    private boolean supportsCommand(Command command) {
        if (command == Command.GRIPPER_OPEN || command == Command.GRIPPER_CLOSE) {
            return deviceType == DeviceType.MECHANICAL_CAR;
        }
        return true;
    }

    private static String displayFor(Command command) {
        switch (command) {
            case LEFT: return "左转";
            case RIGHT: return "右转";
            case FORWARD: return "前进";
            case BACKWARD: return "后退";
            case STOP: return "停止";
            case GRIPPER_OPEN: return "爪子张开";
            case GRIPPER_CLOSE: return "爪子并拢";
        }
        return "未知";
    }

    boolean supportsLabel(String label) { return commandFor(label) != null; }
    boolean isStopLabel(String label) { return commandFor(label) == Command.STOP; }

    boolean hasStopMapping() {
        return labelCommandMap.containsValue(Command.STOP);
    }

    Map<Command, String> getMappingConfig() {
        Map<Command, String> result = new LinkedHashMap<>();
        for (Command cmd : Command.values()) {
            String raw = prefs.getString(KEY_MAP_PREFIX + cmd.name(), null);
            if (raw != null) result.put(cmd, raw);
        }
        return result;
    }

    void saveMappingConfig(Map<Command, String> mapping) {
        labelCommandMap.clear();
        SharedPreferences.Editor editor = prefs.edit();
        for (Command cmd : Command.values()) {
            editor.remove(KEY_MAP_PREFIX + cmd.name());
        }
        for (Map.Entry<Command, String> entry : mapping.entrySet()) {
            String raw = entry.getValue();
            if (raw != null && !raw.trim().isEmpty()) {
                labelCommandMap.put(normalizeLabel(raw), entry.getKey());
                editor.putString(KEY_MAP_PREFIX + entry.getKey().name(), raw);
            }
        }
        editor.apply();
    }

    private void loadMappingConfig() {
        labelCommandMap.clear();
        for (Command cmd : Command.values()) {
            String raw = prefs.getString(KEY_MAP_PREFIX + cmd.name(), null);
            if (raw != null && !raw.trim().isEmpty()) {
                labelCommandMap.put(normalizeLabel(raw), cmd);
            }
        }
    }

    private static String normalizeLabel(String raw) {
        if (raw == null) return null;
        return raw.trim().toLowerCase(Locale.ROOT).replace(" ", "");
    }

    private void loadConfig() {
        String typeName = prefs.getString(KEY_TYPE, DeviceType.MECHANICAL_CAR.name());
        try {
            deviceType = DeviceType.valueOf(typeName);
        } catch (Exception e) {
            deviceType = DeviceType.MECHANICAL_CAR;
        }
        // 侦察球使用固定默认 IP/端口
        if (deviceType == DeviceType.RECON_BALL) {
            targetIp = prefs.getString(KEY_IP, BALL_DEFAULT_IP);
            targetPort = prefs.getInt(KEY_PORT, BALL_DEFAULT_PORT);
        } else {
            targetIp = prefs.getString(KEY_IP, "192.168.2.10");
            targetPort = prefs.getInt(KEY_PORT, 8888);
        }
        controlTimeoutMs = prefs.getLong(KEY_CONTROL_TIMEOUT_MS, DEFAULT_CONTROL_TIMEOUT_MS);
        String saved = prefs.getString(KEY_PROTOCOL, PROTOCOL_LEGACY);
        protocol = PROTOCOL_V2.equalsIgnoreCase(saved) ? PROTOCOL_V2 : PROTOCOL_LEGACY;
    }

    private void saveConfig() {
        prefs.edit().putString(KEY_IP, targetIp)
                .putInt(KEY_PORT, targetPort)
                .putString(KEY_TYPE, deviceType.name())
                .putLong(KEY_CONTROL_TIMEOUT_MS, controlTimeoutMs)
                .putString(KEY_PROTOCOL, protocol)
                .apply();
    }

    /** 当前控制协议：v2（对接 emg_sdk）或 legacy（旧中转脚本）。 */
    String getProtocol() { return protocol; }

    void setProtocol(String value) {
        protocol = PROTOCOL_V2.equalsIgnoreCase(value) ? PROTOCOL_V2 : PROTOCOL_LEGACY;
        prefs.edit().putString(KEY_PROTOCOL, protocol).apply();
    }

    private static String shortMessage(Exception error) {
        String message = error.getMessage();
        return message == null || message.isEmpty() ? error.getClass().getSimpleName() : message;
    }

    void close() {
        boolean wasEnabled = enabled;
        enabled = false;
        active = null;
        io.execute(() -> {
            if (wasEnabled && link != null) {
                link.onDisable();
            } else if (wasEnabled) {
                safeSendStop();
            }
            closeLink();
        });
        io.shutdown();
    }
}