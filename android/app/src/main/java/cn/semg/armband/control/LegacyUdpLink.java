package cn.semg.armband.control;

import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/**
 * 旧协议实现：把语义命令翻译成各设备原来认识的报文。
 *
 * <p>用于兼容尚未替换成 emg_sdk 的中转脚本：
 * 机械车/Go2/Go2-W 发字符串，智能车发 1~7，侦察球发 11 字节二进制包。
 * 设备差异全部留在这个实现里，上层只看到 {@link ControlCommand}。
 */
public final class LegacyUdpLink implements ControlLink {
    private static final byte[] BALL_ID = {0x01, (byte) 0x81};
    private static final byte CONTROL_NORMAL = 0x00;
    private static final byte CONTROL_LOCK = 0x20;
    private static final byte CONTROL_UNLOCK = 0x40;
    private static final int BALL_STEER_SPEED = 600;
    private static final int BALL_DRIVE_SPEED = 1000;
    private static final byte BALL_SERVO = 0;
    private static final byte BALL_TURN_SPEED = 0x12;

    private DatagramSocket socket;
    private ControlEndpoint endpoint;
    private DeviceProfile profile;

    @Override
    public void open(ControlEndpoint endpoint) throws Exception {
        this.endpoint = endpoint;
        this.profile = DeviceProfile.fallback(endpoint.deviceType);
        this.socket = new DatagramSocket();
    }

    @Override
    public DeviceProfile profile() {
        return profile;
    }

    @Override
    public void send(ControlCommand command) throws Exception {
        if (socket == null || socket.isClosed()) return;
        if (isBall()) {
            sendRaw(encodeBallPacket(command));
        } else {
            sendRaw(payloadFor(command).getBytes(StandardCharsets.UTF_8));
        }
    }

    @Override
    public void onEnable() throws Exception {
        if (isBall()) {
            ballArm();
        } else {
            send(ControlCommand.stop());
        }
    }

    @Override
    public void onDisable() {
        try {
            if (isBall()) {
                ballLock();
            } else {
                send(ControlCommand.stop());
            }
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
    private boolean isBall() {
        return endpoint != null && "ball".equals(endpoint.deviceType);
    }

    private boolean isSmartCar() {
        return endpoint != null && "smart_car".equals(endpoint.deviceType);
    }

    private void sendRaw(byte[] data) throws Exception {
        InetAddress address = InetAddress.getByName(endpoint.host);
        socket.send(new DatagramPacket(data, data.length, address, endpoint.port));
    }

    private String payloadFor(ControlCommand command) {
        if (command.isMove()) {
            if (command.vx > 0.1f) return isSmartCar() ? "1" : "forward";
            if (command.vx < -0.1f) return isSmartCar() ? "2" : "backward";
            if (command.wz > 0.1f) return isSmartCar() ? "3" : "left";
            if (command.wz < -0.1f) return isSmartCar() ? "4" : "right";
            return isSmartCar() ? "5" : "stop";
        }
        switch (command.action) {
            case ControlCommand.STOP:
                return isSmartCar() ? "5" : "stop";
            case ControlCommand.OPEN:
                return isSmartCar() ? "6" : "open";
            case ControlCommand.CLOSE:
                return isSmartCar() ? "7" : "close";
            default:
                return isSmartCar() ? "5" : "stop";
        }
    }

    // ===== 侦察球二进制协议 =====

    private byte[] encodeBallPacket(ControlCommand command) {
        int drive = Math.round(command.vx * BALL_DRIVE_SPEED);
        int steer = Math.round(-command.wz * BALL_STEER_SPEED);   // 协议里左为负
        if (!command.isMove()) {
            steer = 0;
            drive = 0;
        }
        return encodeBallRaw(steer, drive, CONTROL_NORMAL);
    }

    private byte[] encodeBallRaw(int steer, int drive, byte control) {
        ByteBuffer buf = ByteBuffer.allocate(11);
        buf.put(BALL_ID);
        buf.put(control);
        buf.order(ByteOrder.LITTLE_ENDIAN);
        buf.putShort((short) steer);
        buf.putShort((short) drive);
        buf.order(ByteOrder.BIG_ENDIAN);
        buf.put(BALL_SERVO);
        buf.put(BALL_TURN_SPEED);
        int crc = crc16Maxim(buf.array(), 0, 9);
        buf.putShort((short) crc);
        return buf.array();
    }

    /** CRC-16/MAXIM-DOW: poly 8005, init 0000, refin/refout, xorout FFFF */
    private static int crc16Maxim(byte[] data, int offset, int length) {
        int crc = 0;
        for (int i = offset; i < offset + length; i++) {
            crc ^= (data[i] & 0xFF);
            for (int j = 0; j < 8; j++) {
                crc = (crc >>> 1) ^ (0xA001 & -(crc & 1));
            }
        }
        return crc ^ 0xFFFF;
    }

    private void ballArm() throws Exception {
        byte[] unlock = encodeBallRaw(0, 0, CONTROL_UNLOCK);
        byte[] normal = encodeBallRaw(0, 0, CONTROL_NORMAL);
        for (int i = 0; i < 3; i++) {
            sendRaw(unlock);
            Thread.sleep(10);
        }
        sendRaw(normal);
    }

    private void ballLock() throws Exception {
        byte[] lock = encodeBallRaw(0, 0, CONTROL_LOCK);
        for (int i = 0; i < 3; i++) {
            sendRaw(lock);
            Thread.sleep(10);
        }
    }
}
