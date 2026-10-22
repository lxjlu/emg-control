package cn.semg.armband;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Dependency-free smoke tests for protocol framing and the local classifier. */
public final class CoreSelfTest {
    public static void main(String[] args) {
        testParserAcrossBleChunks();
        testClassifier();
        System.out.println("CoreSelfTest: protocol and classifier passed");
    }

    private static void testParserAcrossBleChunks() {
        byte[] frame = frame(7, 120, 3);
        SemgPacketParser parser = new SemgPacketParser();
        require(parser.feed(slice(frame, 0, 17)).isEmpty(), "partial frame decoded too early");
        List<SemgFrame> decoded = parser.feed(slice(frame, 17, frame.length));
        require(decoded.size() == 1, "split BLE chunks did not produce one frame");
        SemgFrame value = decoded.get(0);
        require(value.sequence == 7 && value.firstSampleIndex == 120, "header mismatch");
        require(value.samples[2][7] == 2048 + 20 + 7, "channel data mismatch");

        byte[] bad = frame.clone();
        bad[20] ^= 0x22;
        byte[] stream = new byte[3 + bad.length + frame.length];
        System.arraycopy(new byte[]{1, 2, 3}, 0, stream, 0, 3);
        System.arraycopy(bad, 0, stream, 3, bad.length);
        System.arraycopy(frame, 0, stream, 3 + bad.length, frame.length);
        decoded = parser.feed(stream);
        require(decoded.size() == 1 && parser.crcErrors == 1, "CRC resynchronization failed");
    }

    private static void testClassifier() {
        Map<String, List<double[]>> dataset = new LinkedHashMap<>();
        dataset.put("rest", new ArrayList<>());
        dataset.put("fist", new ArrayList<>());
        for (int variation = 0; variation < 8; variation++) {
            dataset.get("rest").add(GestureModel.features(signal(false, variation)));
            dataset.get("fist").add(GestureModel.features(signal(true, variation)));
        }
        GestureModel model = GestureModel.train(dataset);
        require("rest".equals(model.predict(signal(false, 20)).label), "rest classification failed");
        require("fist".equals(model.predict(signal(true, 20)).label), "fist classification failed");
    }

    private static int[][] signal(boolean active, int variation) {
        int[][] samples = new int[GestureModel.WINDOW][SemgPacketParser.CHANNELS];
        for (int i = 0; i < samples.length; i++) for (int channel = 0; channel < 8; channel++) {
            int quiet = ((i * 17 + channel * 11 + variation) % 7) - 3;
            int burst = active ? (((i + channel) % 4 < 2 ? 1 : -1) * (110 + channel * 9 + variation)) : 0;
            samples[i][channel] = 2048 + quiet + burst;
        }
        return samples;
    }

    private static byte[] frame(int sequence, int firstIndex, int sampleCount) {
        ByteBuffer body = ByteBuffer.allocate(14 + sampleCount * 8 * 2).order(ByteOrder.LITTLE_ENDIAN);
        body.put((byte) 0xA5).put((byte) 0x5A).put((byte) 1).put((byte) 1);
        body.putInt(sequence).putInt(firstIndex).putShort((short) sampleCount);
        for (int row = 0; row < sampleCount; row++) for (int channel = 0; channel < 8; channel++) {
            body.putShort((short) (2048 + row * 10 + channel));
        }
        byte[] frame = new byte[body.capacity() + 2];
        System.arraycopy(body.array(), 0, frame, 0, body.capacity());
        int crc = SemgPacketParser.crc16(frame, 0, body.capacity());
        frame[frame.length - 2] = (byte) crc;
        frame[frame.length - 1] = (byte) (crc >> 8);
        return frame;
    }

    private static byte[] slice(byte[] source, int start, int end) {
        byte[] result = new byte[end - start];
        System.arraycopy(source, start, result, 0, result.length);
        return result;
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}
