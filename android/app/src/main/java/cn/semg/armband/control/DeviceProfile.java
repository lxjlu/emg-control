package cn.semg.armband.control;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * 被控对象的能力描述。
 *
 * <p>首选来源是 SDK 的 {@code hello} 应答；SDK 还没部署时用本地兜底值，
 * 保证行为与旧版一致。
 */
public final class DeviceProfile {
    public final String type;
    public final String displayName;
    public final Set<String> actions;
    public final int deadmanMs;
    public final float maxVx;
    public final float maxWz;

    public DeviceProfile(String type, String displayName, Set<String> actions,
                         int deadmanMs, float maxVx, float maxWz) {
        this.type = type;
        this.displayName = displayName;
        this.actions = Collections.unmodifiableSet(new LinkedHashSet<>(actions));
        this.deadmanMs = deadmanMs;
        this.maxVx = maxVx;
        this.maxWz = maxWz;
    }

    public boolean supports(String action) {
        return actions.contains(action);
    }

    /** SDK 不可达时的本地默认能力，数值与各适配器保持一致。 */
    public static DeviceProfile fallback(String type) {
        if ("go2".equals(type)) {
            return new DeviceProfile(type, "Go2 机械狗",
                    actions(ControlCommand.MOVE, ControlCommand.STOP), 600, 0.8f, 2.0f);
        }
        if ("go2w".equals(type)) {
            return new DeviceProfile(type, "Go2 轮式",
                    actions(ControlCommand.MOVE, ControlCommand.STOP), 600, 0.5f, 1.0f);
        }
        if ("ball".equals(type)) {
            return new DeviceProfile(type, "侦察球",
                    actions(ControlCommand.MOVE, ControlCommand.STOP), 400, 1f, 1f);
        }
        if ("smart_car".equals(type)) {
            return new DeviceProfile(type, "智能车",
                    actions(ControlCommand.MOVE, ControlCommand.STOP,
                            ControlCommand.OPEN, ControlCommand.CLOSE), 500, 1f, 1f);
        }
        if ("ppt".equals(type)) {
            return new DeviceProfile(type, "PPT / WPS",
                    actions("next", "prev", "start", "exit"), 0, 0f, 0f);
        }
        return new DeviceProfile(type, "机械车",
                actions(ControlCommand.MOVE, ControlCommand.STOP,
                        ControlCommand.OPEN, ControlCommand.CLOSE), 500, 1f, 1f);
    }

    private static Set<String> actions(String... values) {
        return new LinkedHashSet<>(Arrays.asList(values));
    }
}
