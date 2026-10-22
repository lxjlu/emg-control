package cn.semg.armband;

/**
 * Butterworth 带通滤波器 + filtfilt 零相位实现。
 * 与 scipy.signal.butter(order, [lowcut, highcut], btype='band', fs=fs)
 * 及 scipy.signal.filtfilt(b, a, data, axis=0) 输出对齐，用于 PyTorch CNN 推理前的 EMG 预处理。
 *
 * 设计流程（与 scipy.signal.iirfilter 一致）：
 *   1. 预扭曲（prewarp）数字频率到模拟域
 *   2. 设计 N 阶 Butterworth 模拟低通原型（cutoff = 1 rad/s，极点 s_k = exp(i*π/2 + i*(2k-1)π/(2N))）
 *   3. 低通 → 带通变换：s → (s^2 + w0^2) / (s * bw)，每个低通极点 → 2 个带通极点；
 *      低通零点（s=∞，共 N 个）→ 带通零点 s=0（N 个）+ s=∞（N 个）
 *   4. 双线性变换 s → 2*fs*(z-1)/(z+1)
 *   5. 由零点/极点构建分子分母多项式（系数按 z^{-k} 升序排列，与 scipy 一致）
 *
 * 系数约定：b[k] 为 z^{-k} 的系数；a[0] = 1。
 */
final class ButterworthFilter {
    private final double[] b;  // 分子系数，b[k] 为 z^{-k} 的系数
    private final double[] a;  // 分母系数，a[0] = 1
    private final double[] zi; // 稳态初始条件（scipy.signal.lfilter_zi），null 表示用零初始条件

    /** 默认配置：4 阶带通 20-249.999Hz @ 500Hz（与 train_model_sep.py / control_car_sep.py 一致）。 */
    ButterworthFilter() {
        this(20.0, 249.999, 500.0, 4);
    }

    /**
     * 直接使用外部提供的 b/a 系数（来自云端训练端 scipy.signal.butter）。
     * 这是保证手机端与训练端滤波完全一致的首选方式：自行重算滤波器容易漏掉
     * scipy 的增益归一化，实测滤波后幅度会偏大约 1.9 倍，导致推理结果错乱。
     */
    ButterworthFilter(double[] b, double[] a) {
        this(b, a, null);
    }

    /**
     * 直接使用外部提供的 b/a 系数与稳态初始条件 zi。
     * 传入 zi 后可完全复现 scipy.signal.filtfilt 的行为（含边缘瞬态）。
     */
    ButterworthFilter(double[] b, double[] a, double[] zi) {
        if (b == null || a == null || b.length == 0 || a.length == 0) {
            throw new IllegalArgumentException("滤波器系数不能为空");
        }
        this.b = b.clone();
        this.a = a.clone();
        this.zi = (zi != null && zi.length > 0) ? zi.clone() : null;
    }

    ButterworthFilter(double lowcut, double highcut, double fs, int order) {
        // 1. 预扭曲到模拟频率
        double wL = 2.0 * fs * Math.tan(Math.PI * lowcut / fs);
        double wH = 2.0 * fs * Math.tan(Math.PI * highcut / fs);
        double w0 = Math.sqrt(wL * wH);
        double bw = wH - wL;

        // 2. N 阶 Butterworth 模拟低通原型极点（cutoff = 1 rad/s）
        //    s_k = exp(i * (π/2 + (2k-1)π/(2N)))，k = 1..N
        Complex[] lpPoles = new Complex[order];
        for (int k = 0; k < order; k++) {
            double angle = Math.PI / 2.0 + (2 * k + 1) * Math.PI / (2.0 * order);
            lpPoles[k] = new Complex(Math.cos(angle), Math.sin(angle));
        }

        // 3. 低通 → 带通变换：每个极点 s_p → 2 个带通极点
        //    s_bp = (s_p * bw ± sqrt(s_p^2 * bw^2 - 4 * w0^2)) / 2
        Complex[] bpPoles = new Complex[2 * order];
        for (int k = 0; k < order; k++) {
            Complex sp = lpPoles[k];
            Complex spBw = sp.multiply(bw);
            Complex disc = sp.multiply(sp).multiply(bw * bw).subtract(new Complex(4.0 * w0 * w0, 0)).sqrt();
            bpPoles[2 * k] = spBw.add(disc).multiply(0.5);
            bpPoles[2 * k + 1] = spBw.subtract(disc).multiply(0.5);
        }

        // 4. 双线性变换：s → 2*fs*(z-1)/(z+1)，即 z = (2*fs + s) / (2*fs - s)
        Complex[] dPoles = new Complex[2 * order];
        for (int k = 0; k < 2 * order; k++) {
            dPoles[k] = bilinear(bpPoles[k], fs);
        }
        // 带通零点：N 个在 s=0（→ z=1），N 个在 s=∞（→ z=-1）
        Complex[] dZeros = new Complex[2 * order];
        for (int k = 0; k < order; k++) {
            dZeros[k] = new Complex(1, 0);       // z = 1
            dZeros[order + k] = new Complex(-1, 0); // z = -1
        }

        // 5. 由零点/极点构建多项式，系数按 z^{-k} 升序（与 scipy.signal.butter 一致）
        //    polyFromRoots 返回 [c_N, ..., c_0]（降幂），
        //    而 scipy 的 b[k] 实际上 = 多项式中 z^{N-k} 的系数，索引与降幂同序，无需反转。
        //    验证：对带通 (zeros at z=±1)，多项式为 z^N*(1-z^-2)^...，DC 处 B(z=1)=0 ✓
        double[] numPoly = polyFromRoots(dZeros);
        double[] denPoly = polyFromRoots(dPoles);
        int len = numPoly.length;  // = 2*order + 1
        b = new double[len];
        a = new double[len];
        for (int i = 0; i < len; i++) {
            b[i] = numPoly[i];
            a[i] = denPoly[i];
        }

        // 6. 归一化：a[0] = 1（与 scipy 一致）
        double a0 = a[0];
        for (int i = 0; i < len; i++) {
            b[i] /= a0;
            a[i] /= a0;
        }
        // 注意：此处未复现 scipy 的中心频率增益校准，滤波后幅度会比 scipy 偏大约 1.9 倍。
        // 云端训练会把 scipy 的 b/a/zi 一并下发，推理优先使用外部系数；
        // 本构造器仅作为旧模型（无系数元数据）的兜底。
        this.zi = null;
    }

    /**
     * 零相位前向+反向滤波，与 scipy.signal.filtfilt 一致。
     * - edge 长度 = 3 * max(len(a), len(b))，与 scipy 默认 padlen 相同
     * - padding 使用 odd 模式（边缘反射）
     * - 若提供了 zi，则以 zi * 首样本作为初始状态，完全复现 scipy 的边缘行为
     */
    double[] filtfilt(double[] signal) {
        int n = signal.length;
        if (n == 0) return new double[0];
        int nfilt = Math.max(a.length, b.length);
        int edgeLen = 3 * nfilt;
        if (edgeLen >= n) edgeLen = Math.max(0, n - 1);

        // odd padding：边缘反射，padded[edgeLen-1-i] = 2*signal[0] - signal[i+1]
        double[] padded = new double[n + 2 * edgeLen];
        double first = signal[0];
        double last = signal[n - 1];
        for (int i = 0; i < edgeLen; i++) {
            padded[edgeLen - 1 - i] = 2.0 * first - signal[i + 1];
        }
        System.arraycopy(signal, 0, padded, edgeLen, n);
        for (int i = 0; i < edgeLen; i++) {
            padded[edgeLen + n + i] = 2.0 * last - signal[n - 2 - i];
        }

        // 前向滤波（直接型 II 转置），初始状态 zi * padded[0]
        double[] forward = filter(padded, new double[padded.length], initialZ(padded[0]));

        // 反向：反转 → 滤波 → 反转，初始状态 zi * forward[-1]
        double[] reversed = new double[forward.length];
        for (int i = 0; i < forward.length; i++) reversed[i] = forward[forward.length - 1 - i];
        double[] backward = filter(reversed, new double[reversed.length], initialZ(reversed[0]));

        // 取出中间部分（去掉 padding），并反转
        double[] result = new double[n];
        for (int i = 0; i < n; i++) result[i] = backward[backward.length - 1 - edgeLen - i];
        return result;
    }

    /** 由 zi 与首样本值构造初始状态；zi 为 null 时返回 null（零初始条件）。 */
    private double[] initialZ(double firstSample) {
        if (zi == null) return null;
        double[] z = new double[zi.length];
        for (int i = 0; i < zi.length; i++) z[i] = zi[i] * firstSample;
        return z;
    }

    /** 直接型 II 转置 IIR 滤波：y[n] = b[0]*x[n] + z[0]; z[j-1] = b[j]*x[n] - a[j]*y[n] + z[j]。 */
    private double[] filter(double[] x, double[] y, double[] z0) {
        int N = b.length - 1;
        double[] z = (z0 != null && z0.length == N) ? z0 : new double[N];
        for (int i = 0; i < x.length; i++) {
            y[i] = b[0] * x[i] + (N > 0 ? z[0] : 0);
            for (int j = 1; j < N; j++) {
                z[j - 1] = b[j] * x[i] - a[j] * y[i] + z[j];
            }
            if (N > 0) z[N - 1] = b[N] * x[i] - a[N] * y[i];
        }
        return y;
    }

    private static Complex bilinear(Complex s, double fs) {
        Complex num = new Complex(2.0 * fs, 0).add(s);
        Complex den = new Complex(2.0 * fs, 0).subtract(s);
        return num.divide(den);
    }

    /** 由根构建多项式系数（降幂 [c_N, c_{N-1}, ..., c_0]），与 numpy.poly 一致。 */
    private static double[] polyFromRoots(Complex[] roots) {
        int n = roots.length;
        Complex[] poly = new Complex[n + 1];
        poly[0] = new Complex(1, 0);
        for (int i = 1; i <= n; i++) poly[i] = new Complex(0, 0);
        for (Complex root : roots) {
            // 乘以 (z - root)，原地更新，从底向上以保留旧值
            for (int i = n; i >= 1; i--) {
                poly[i] = poly[i].subtract(root.multiply(poly[i - 1]));
            }
        }
        double[] real = new double[n + 1];
        for (int i = 0; i <= n; i++) real[i] = poly[i].real;
        return real;
    }

    /** 简单复数类（不可变）。 */
    private static final class Complex {
        final double real, imag;
        Complex(double r, double i) { real = r; imag = i; }
        Complex add(Complex o) { return new Complex(real + o.real, imag + o.imag); }
        Complex subtract(Complex o) { return new Complex(real - o.real, imag - o.imag); }
        Complex multiply(double s) { return new Complex(real * s, imag * s); }
        Complex multiply(Complex o) {
            return new Complex(real * o.real - imag * o.imag, real * o.imag + imag * o.real);
        }
        Complex sqrt() {
            double r = Math.sqrt(real * real + imag * imag);
            double theta = Math.atan2(imag, real) / 2.0;
            return new Complex(r * Math.cos(theta), r * Math.sin(theta));
        }
        Complex divide(Complex o) {
            double denom = o.real * o.real + o.imag * o.imag;
            return new Complex((real * o.real + imag * o.imag) / denom,
                    (imag * o.real - real * o.imag) / denom);
        }
    }
}
