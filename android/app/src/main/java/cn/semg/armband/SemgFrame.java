package cn.semg.armband;

final class SemgFrame {
    final long sequence;
    final long firstSampleIndex;
    final int[][] samples;

    SemgFrame(long sequence, long firstSampleIndex, int[][] samples) {
        this.sequence = sequence;
        this.firstSampleIndex = firstSampleIndex;
        this.samples = samples;
    }
}
