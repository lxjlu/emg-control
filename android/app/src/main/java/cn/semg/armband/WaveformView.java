package cn.semg.armband;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.view.View;

/** Eight-lane oscilloscope view with a two-second circular history. */
public final class WaveformView extends View {
    private static final int CHANNELS = 8;
    private static final int CAPACITY = 1000;
    private static final int[] COLORS = {
            0xFF42E8B4, 0xFF4BC6FF, 0xFFFFCF5C, 0xFFFF7895,
            0xFFB889FF, 0xFF63E6E2, 0xFFFF9E57, 0xFF9BDD62
    };
    private final Object lock = new Object();
    private final int[][] history = new int[CHANNELS][CAPACITY];
    private final Paint grid = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint trace = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint label = new Paint(Paint.ANTI_ALIAS_FLAG);
    private int writeAt;
    private int count;
    private long lastInvalidate;

    public WaveformView(Context context) { this(context, null); }

    public WaveformView(Context context, AttributeSet attributes) {
        super(context, attributes);
        setBackgroundColor(0xFF0B151B);
        grid.setColor(0xFF23323A);
        grid.setStrokeWidth(dp(1));
        trace.setStyle(Paint.Style.STROKE);
        trace.setStrokeWidth(dp(1.15f));
        trace.setStrokeJoin(Paint.Join.ROUND);
        label.setColor(0xFF80929D);
        label.setTextSize(dp(10));
    }

    void addSamples(int[][] samples) {
        synchronized (lock) {
            for (int[] sample : samples) {
                for (int channel = 0; channel < CHANNELS; channel++) {
                    history[channel][writeAt] = sample[channel];
                }
                writeAt = (writeAt + 1) % CAPACITY;
                if (count < CAPACITY) count++;
            }
        }
        long now = System.currentTimeMillis();
        if (now - lastInvalidate > 32) {
            lastInvalidate = now;
            postInvalidateOnAnimation();
        }
    }

    void clear() {
        synchronized (lock) { writeAt = count = 0; }
        invalidate();
    }

    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        float width = getWidth();
        float height = getHeight();
        float left = dp(29);
        float plotWidth = Math.max(1, width - left - dp(7));
        float laneHeight = height / CHANNELS;

        for (int channel = 0; channel < CHANNELS; channel++) {
            float center = laneHeight * (channel + 0.5f);
            canvas.drawLine(left, center, width, center, grid);
            canvas.drawText("CH" + (channel + 1), dp(4), center + dp(3), label);
        }
        for (int division = 1; division < 4; division++) {
            float x = left + plotWidth * division / 4f;
            canvas.drawLine(x, 0, x, height, grid);
        }

        int localCount;
        int localWrite;
        int[][] copy;
        synchronized (lock) {
            localCount = count;
            localWrite = writeAt;
            copy = new int[CHANNELS][localCount];
            int start = (localWrite - localCount + CAPACITY) % CAPACITY;
            for (int channel = 0; channel < CHANNELS; channel++) {
                for (int i = 0; i < localCount; i++) {
                    copy[channel][i] = history[channel][(start + i) % CAPACITY];
                }
            }
        }
        if (localCount < 2) return;

        for (int channel = 0; channel < CHANNELS; channel++) {
            int min = Integer.MAX_VALUE;
            int max = Integer.MIN_VALUE;
            for (int value : copy[channel]) { min = Math.min(min, value); max = Math.max(max, value); }
            float mid = (min + max) * 0.5f;
            float range = Math.max(80f, max - min);
            float amplitude = laneHeight * 0.36f;
            Path path = new Path();
            for (int i = 0; i < localCount; i++) {
                float x = left + plotWidth * i / (localCount - 1f);
                float y = laneHeight * (channel + 0.5f)
                        - (copy[channel][i] - mid) / range * amplitude * 2f;
                if (i == 0) path.moveTo(x, y); else path.lineTo(x, y);
            }
            trace.setColor(COLORS[channel]);
            canvas.drawPath(path, trace);
        }
    }

    private float dp(float value) { return value * getResources().getDisplayMetrics().density; }
}
