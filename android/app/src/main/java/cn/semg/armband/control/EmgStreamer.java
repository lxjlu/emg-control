package cn.semg.armband.control;

import android.util.Base64;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.charset.StandardCharsets;

/**
 * 把原始肌电样本通过 UDP 推给 SDK，供浏览器查看 8 通道实时波形。
 *
 * <p>报文格式见 docs/protocol.md：
 * {"v":2,"type":"stream","seq":1,"rate":500,"ch":8,"n":25,"data":"&lt;base64&gt;"}
 */
public final class EmgStreamer {
    /** 每包样本数，25 个约 50ms，正好是 8 通道 × 25 × 2 字节 = 400 字节。 */
    private static final int SAMPLES_PER_PACKET = 25;

    private final DatagramSocket socket;
    private final InetAddress address;
    private final int port;
    private final int channels;
    private final int sampleRate;
    private final ByteArrayOutputStream buffer = new ByteArrayOutputStream(1024);

    private int buffered;
    private int seq;

    public EmgStreamer(String host, int port, int channels, int sampleRate) throws IOException {
        this.address = InetAddress.getByName(host);
        this.port = port;
        this.channels = channels;
        this.sampleRate = sampleRate;
        this.socket = new DatagramSocket();
    }

    /** 输入一帧样本（[样本][通道]），每满一包就发送。 */
    public void feed(int[][] samples) {
        for (int[] sample : samples) {
            for (int channel = 0; channel < channels; channel++) {
                int value = channel < sample.length ? sample[channel] : 0;
                if (value > 32767) value = 32767;
                if (value < -32768) value = -32768;
                buffer.write(value & 0xFF);
                buffer.write((value >> 8) & 0xFF);
            }
            buffered++;
            if (buffered >= SAMPLES_PER_PACKET) flush();
        }
    }

    private void flush() {
        if (buffered == 0) return;
        String json = "{\"v\":2,\"type\":\"stream\",\"seq\":" + (++seq)
                + ",\"rate\":" + sampleRate
                + ",\"ch\":" + channels
                + ",\"n\":" + buffered
                + ",\"data\":\"" + Base64.encodeToString(buffer.toByteArray(), Base64.NO_WRAP)
                + "\"}";
        byte[] payload = json.getBytes(StandardCharsets.UTF_8);
        try {
            socket.send(new DatagramPacket(payload, payload.length, address, port));
        } catch (IOException ignored) {
            // 网络暂时不可用时丢弃这一包，不影响识别和控制
        }
        buffer.reset();
        buffered = 0;
    }

    public void close() {
        flush();
        socket.close();
    }
}
