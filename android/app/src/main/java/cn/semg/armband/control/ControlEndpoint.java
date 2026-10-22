package cn.semg.armband.control;

/** 被控对象地址 + 类型。类型用于旧协议编码，v2 协议下仅作日志与能力兜底。 */
public final class ControlEndpoint {
    public final String host;
    public final int port;
    public final String deviceType;

    public ControlEndpoint(String host, int port, String deviceType) {
        this.host = host;
        this.port = port;
        this.deviceType = deviceType;
    }

    @Override
    public String toString() {
        return host + ":" + port + " (" + deviceType + ")";
    }
}
