package cn.semg.armband.control;

/**
 * App 与被控对象之间的唯一通道抽象。
 *
 * <p>上层只发 {@link ControlCommand}，不再拼任何设备专属报文。
 * 目前有两个实现：v2 JSON（对接 emg_sdk）与 legacy（兼容旧中转脚本）。
 */
public interface ControlLink extends AutoCloseable {

    /** 建立通道（打开 socket、记录地址）。 */
    void open(ControlEndpoint endpoint) throws Exception;

    /** 被控对象能力；未知时返回本地兜底值。 */
    DeviceProfile profile();

    /** 发送一条语义命令。连续量需要调用方按固定周期重复调用。 */
    void send(ControlCommand command) throws Exception;

    /** 控制启用时的一次性动作（例如侦察球解锁）。 */
    void onEnable() throws Exception;

    /** 控制关闭时的一次性动作（例如侦察球锁定、补发停止）。 */
    void onDisable();

    @Override
    void close();
}
