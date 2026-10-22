package cn.semg.armband.control;

import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

/**
 * v2 协议实现：直接向 emg_sdk 发语义命令（JSON over UDP）。
 *
 * <pre>
 * {"v":2,"seq":12,"act":"move","arg":{"vx":1.00,"wz":0.00}}
 * {"v":2,"seq":13,"act":"stop"}
 * </pre>
 */
public final class UdpControlLink implements ControlLink {
    private DatagramSocket socket;
    private ControlEndpoint endpoint;
    private DeviceProfile profile;
    private int seq;

    @Override
    public void open(ControlEndpoint endpoint) throws Exception {
        this.endpoint = endpoint;
        this.profile = DeviceProfile.fallback(endpoint.deviceType);
        this.seq = 0;
        this.socket = new DatagramSocket();
    }

    @Override
    public DeviceProfile profile() {
        return profile;
    }

    @Override
    public void send(ControlCommand command) throws Exception {
        if (socket == null || socket.isClosed()) return;
        sendRaw(encode(command).getBytes(StandardCharsets.UTF_8));
    }

    @Override
    public void onEnable() throws Exception {
        send(ControlCommand.stop());
    }

    @Override
    public void onDisable() {
        try {
            send(ControlCommand.stop());
        } catch (Exception ignored) {
        }
    }

    @Override
    public void close() {
        if (socket != null) {
            socket.close();
            socket = null;
        }
    }

    // ------------------------------------------------------------------
    private void sendRaw(byte[] data) throws Exception {
        InetAddress address = InetAddress.getByName(endpoint.host);
        socket.send(new DatagramPacket(data, data.length, address, endpoint.port));
    }

    private String encode(ControlCommand command) {
        StringBuilder builder = new StringBuilder(96);
        builder.append("{\"v\":2,\"seq\":").append(++seq)
                .append(",\"act\":\"").append(command.action).append('"');
        if (command.isMove()) {
            builder.append(",\"arg\":{\"vx\":")
                    .append(String.format(Locale.US, "%.2f", command.vx))
                    .append(",\"wz\":")
                    .append(String.format(Locale.US, "%.2f", command.wz))
                    .append('}');
        }
        return builder.append('}').toString();
    }
}
