package cn.semg.armband.control;

/**
 * 语义命令：与具体被控对象无关。
 *
 * <p>vx / wz 归一化到 -1..1，具体量纲（m/s、rad/s、-1000..1000）由 SDK 侧的适配器决定。
 * 离散动作（open / close / next / prev ...）只填 action。
 */
public final class ControlCommand {
    public static final String MOVE = "move";
    public static final String STOP = "stop";
    public static final String OPEN = "open";
    public static final String CLOSE = "close";

    public final String action;
    public final float vx;
    public final float wz;

    private ControlCommand(String action, float vx, float wz) {
        this.action = action;
        this.vx = normalize(vx);
        this.wz = normalize(wz);
    }

    public static ControlCommand move(float vx, float wz) {
        return new ControlCommand(MOVE, vx, wz);
    }

    public static ControlCommand stop() {
        return new ControlCommand(STOP, 0f, 0f);
    }

    public static ControlCommand of(String action) {
        return new ControlCommand(action, 0f, 0f);
    }

    public boolean isMove() {
        return MOVE.equals(action);
    }

    private static float normalize(float value) {
        if (Float.isNaN(value)) return 0f;
        return Math.max(-1f, Math.min(1f, value));
    }

    @Override
    public String toString() {
        return isMove()
                ? String.format(java.util.Locale.US, "move(vx=%.2f, wz=%.2f)", vx, wz)
                : action;
    }
}
