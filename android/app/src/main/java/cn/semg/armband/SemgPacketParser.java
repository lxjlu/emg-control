package cn.semg.armband;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Incremental decoder for sEMG byte streams.
 * 仅支持 STM32 手环：帧头 A5 5A，8 通道 12-bit 原始 ADC（0-4095），500 Hz。
 */
final class SemgPacketParser {
    enum Protocol { STM32 }

    static final int CHANNELS = 8;
    static final int SAMPLE_RATE = 500;
    static final int STM32_SAMPLE_RATE = 500;
    private static final int STM32_HEADER = 14;

    private Protocol protocol = Protocol.STM32;
    private byte[] buffer = new byte[4096];
    private int size;
    long crcErrors;
    long formatErrors;
    long discardedBytes;

    void setProtocol(Protocol protocol) {
        this.protocol = protocol;
        reset();
    }

    Protocol getProtocol() {
        return protocol;
    }

    int getSampleRate() {
        return STM32_SAMPLE_RATE;
    }

    synchronized List<SemgFrame> feed(byte[] data) {
        ensureCapacity(size + data.length);
        System.arraycopy(data, 0, buffer, size, data.length);
        size += data.length;
        List<SemgFrame> frames = new ArrayList<>();
        parseStm32Frames(frames);
        return frames;
    }

    private void parseStm32Frames(List<SemgFrame> out) {
        while (true) {
            int sync = findStm32Sync();
            if (sync < 0) {
                int keep = size > 0 && unsigned(buffer[size - 1]) == 0xA5 ? 1 : 0;
                discardedBytes += size - keep;
                if (keep == 1) buffer[0] = buffer[size - 1];
                size = keep;
                break;
            }
            if (sync > 0) {
                discardedBytes += sync;
                consume(sync);
            }
            if (size < STM32_HEADER) break;

            int sampleCount = u16(12);
            if (unsigned(buffer[2]) != 1 || unsigned(buffer[3]) != 1
                    || sampleCount < 1 || sampleCount > 10) {
                formatErrors++;
                consume(1);
                continue;
            }
            int frameSize = STM32_HEADER + sampleCount * CHANNELS * 2 + 2;
            if (size < frameSize) break;
            int expected = u16(frameSize - 2);
            int actual = crc16(buffer, 0, frameSize - 2);
            if (expected != actual) {
                crcErrors++;
                consume(1);
                continue;
            }

            int[][] samples = new int[sampleCount][CHANNELS];
            int offset = STM32_HEADER;
            for (int row = 0; row < sampleCount; row++) {
                for (int channel = 0; channel < CHANNELS; channel++) {
                    samples[row][channel] = u16(offset);
                    offset += 2;
                }
            }
            out.add(new SemgFrame(u32(4), u32(8), samples));
            consume(frameSize);
        }
    }

    synchronized void reset() {
        size = 0;
        crcErrors = formatErrors = discardedBytes = 0;
    }

    static int crc16(byte[] bytes, int offset, int length) {
        int crc = 0xFFFF;
        for (int i = offset; i < offset + length; i++) {
            crc ^= unsigned(bytes[i]) << 8;
            for (int bit = 0; bit < 8; bit++) {
                crc = (crc & 0x8000) != 0 ? ((crc << 1) ^ 0x1021) & 0xFFFF
                        : (crc << 1) & 0xFFFF;
            }
        }
        return crc;
    }

    private int findStm32Sync() {
        for (int i = 0; i + 1 < size; i++) {
            if (unsigned(buffer[i]) == 0xA5 && unsigned(buffer[i + 1]) == 0x5A) return i;
        }
        return -1;
    }

    private int u16(int offset) {
        return unsigned(buffer[offset]) | unsigned(buffer[offset + 1]) << 8;
    }

    private long u32(int offset) {
        return (long) unsigned(buffer[offset])
                | (long) unsigned(buffer[offset + 1]) << 8
                | (long) unsigned(buffer[offset + 2]) << 16
                | (long) unsigned(buffer[offset + 3]) << 24;
    }

    private void consume(int count) {
        System.arraycopy(buffer, count, buffer, 0, size - count);
        size -= count;
    }

    private void ensureCapacity(int wanted) {
        if (wanted > buffer.length) buffer = Arrays.copyOf(buffer, Math.max(wanted, buffer.length * 2));
    }

    private static int unsigned(byte value) {
        return value & 0xFF;
    }
}
