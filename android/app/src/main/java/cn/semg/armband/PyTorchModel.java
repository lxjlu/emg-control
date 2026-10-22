package cn.semg.armband;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * PyTorch CNN1D 推理引擎，与 PC 端 control_car_sep.py 的推理流程对齐。
 * 加载 TorchScript (.pt) 模型，对 8 通道 × 100 样本的 EMG 窗口做带通滤波 + 归一化 + CNN 前向 + softmax，
 * 输出 5 类手势标签 + 置信度（4个动作 + 1个休息状态）。
 *
 * 模型架构（与 train_model_sep.py 中 CNN1D 一致，由 TorchScript 序列化保存）：
 *   Conv1d(8→64,k10) → BN → ReLU → AvgPool(5) → Dropout
 *   → Conv1d(64→96,k5) → BN → ReLU → Dropout
 *   → Conv1d(96→128,k3) → BN → ReLU → Dropout → Flatten
 *   → Linear(2048→5)
 */
final class PyTorchModel {
    static final int WINDOW = 100;  // 200 ms @ 500 Hz，与训练端一致
    static final int HOP = 20;     // 训练端 step_size=20

    static final class Prediction {
        final String label;        // 映射后的动作标签
        final double confidence;   // softmax 最大概率
        Prediction(String label, double confidence) {
            this.label = label;
            this.confidence = confidence;
        }
    }

    private final Module module;
    private final ButterworthFilter filter;
    final Map<Integer, String> labelMapping;  // {0: "前进", 1: "后退", ...}
    final SemgPacketParser.Protocol protocol;
    final String modelName;
    final long createdAt;
    final File sourceFile;
    // 归一化统计：per-channel mean 和 std（长度 8），null 表示不做归一化（兼容旧模型）
    final float[] normMeans;
    final float[] normStds;
    // 滤波器系数：来自训练端 scipy，null 表示回退到内置 Butterworth 设计（兼容旧模型）
    final double[] filterB;
    final double[] filterA;
    final double[] filterZi;

    private PyTorchModel(Module module, Map<Integer, String> mapping,
                         SemgPacketParser.Protocol protocol, String name,
                         long createdAt, File sourceFile,
                         float[] normMeans, float[] normStds,
                         double[] filterB, double[] filterA, double[] filterZi) {
        this.module = module;
        this.labelMapping = mapping;
        this.protocol = protocol;
        this.modelName = name;
        this.createdAt = createdAt;
        this.sourceFile = sourceFile;
        this.normMeans = normMeans;
        this.normStds = normStds;
        this.filterB = filterB;
        this.filterA = filterA;
        this.filterZi = filterZi;
        // 优先使用云端下发的系数，保证与训练端滤波完全一致
        if (filterB != null && filterA != null && filterB.length > 0 && filterA.length > 0) {
            this.filter = new ButterworthFilter(filterB, filterA, filterZi);
        } else {
            this.filter = new ButterworthFilter();
        }
    }

    /**
     * 从 .pt 文件加载 TorchScript 模型。必须在 worker 线程调用（Module.load 阻塞）。
     */
    static PyTorchModel load(File ptFile, Map<Integer, String> mapping,
                             SemgPacketParser.Protocol protocol, String name,
                             long createdAt) throws RuntimeException {
        return load(ptFile, mapping, protocol, name, createdAt, null, null, null, null, null);
    }

    /**
     * 从 .pt 文件加载 TorchScript 模型，附带归一化统计。
     */
    static PyTorchModel load(File ptFile, Map<Integer, String> mapping,
                             SemgPacketParser.Protocol protocol, String name,
                             long createdAt,
                             float[] normMeans, float[] normStds) throws RuntimeException {
        return load(ptFile, mapping, protocol, name, createdAt, normMeans, normStds,
                null, null, null);
    }

    /**
     * 从 .pt 文件加载 TorchScript 模型，附带归一化统计和滤波器系数。
     */
    static PyTorchModel load(File ptFile, Map<Integer, String> mapping,
                             SemgPacketParser.Protocol protocol, String name,
                             long createdAt,
                             float[] normMeans, float[] normStds,
                             double[] filterB, double[] filterA,
                             double[] filterZi) throws RuntimeException {
        Module module = Module.load(ptFile.getAbsolutePath());
        return new PyTorchModel(module, mapping, protocol, name, createdAt, ptFile,
                normMeans, normStds, filterB, filterA, filterZi);
    }

    /**
     * 推理：int[100][8] 窗口 → 滤波 → 归一化 → (1, 8, 100) 张量 → CNN → softmax → 标签。
     */
    Prediction predict(int[][] window) {
        if (window == null || window.length < WINDOW) {
            return new Prediction("—", 0.0);
        }

        // 1. int[100][8] → 按通道做 Butterworth 带通滤波（每通道独立）
        double[][] filtered = new double[WINDOW][SemgPacketParser.CHANNELS];
        for (int ch = 0; ch < SemgPacketParser.CHANNELS; ch++) {
            double[] column = new double[WINDOW];
            for (int i = 0; i < WINDOW; i++) column[i] = window[i][ch];
            double[] filt = filter.filtfilt(column);
            for (int i = 0; i < WINDOW; i++) filtered[i][ch] = filt[i];
        }

        // 2. 转置为 (8, 100) 并展平为一维 float[800]，构造 Tensor (1, 8, 100)
        //    与 train_model_sep.py 中 np.swapaxes 一致：(N,100,8) → (N,8,100)
        //    同时应用 per-channel Z-score 归一化（与训练端一致）
        float[] flat = new float[SemgPacketParser.CHANNELS * WINDOW];
        int idx = 0;
        for (int ch = 0; ch < SemgPacketParser.CHANNELS; ch++) {
            float mean = (normMeans != null && ch < normMeans.length) ? normMeans[ch] : 0f;
            float std = (normStds != null && ch < normStds.length) ? normStds[ch] : 1f;
            if (std < 1e-8f) std = 1f;
            for (int i = 0; i < WINDOW; i++) {
                float val = (float) filtered[i][ch];
                if (normMeans != null) {
                    val = (val - mean) / std;
                }
                flat[idx++] = val;
            }
        }
        
        // 调试日志：打印第一个样本的统计信息
        if (idx == SemgPacketParser.CHANNELS * WINDOW) {
            float min = flat[0], max = flat[0], sum = 0;
            for (float v : flat) {
                if (v < min) min = v;
                if (v > max) max = v;
                sum += v;
            }
            android.util.Log.d("PyTorchModel", String.format(
                "推理输入: min=%.2f, max=%.2f, mean=%.2f, normMeans=%s",
                min, max, sum / flat.length,
                normMeans != null ? java.util.Arrays.toString(normMeans) : "NULL"));
        }
        
        Tensor input = Tensor.fromBlob(flat, new long[]{1, SemgPacketParser.CHANNELS, WINDOW});

        // 3. 前向推理 + softmax
        Tensor output = module.forward(IValue.from(input)).toTensor();
        float[] logits = output.getDataAsFloatArray();

        // 4. softmax → 概率
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (float v : logits) if (v > maxLogit) maxLogit = v;
        double sumExp = 0;
        double[] exps = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exps[i] = Math.exp(logits[i] - maxLogit);
            sumExp += exps[i];
        }
        int argmax = 0;
        double maxProb = 0;
        for (int i = 0; i < exps.length; i++) {
            double p = exps[i] / sumExp;
            if (p > maxProb) { maxProb = p; argmax = i; }
        }

        // 5. 映射到动作标签
        String label = labelMapping.get(argmax);
        if (label == null) label = "gesture_" + (argmax + 1);
        return new Prediction(label, maxProb);
    }

    /** 保存映射关系到 JSON（与 .pt 同名 .json 旁车文件）。 */
    JSONObject toJson() throws JSONException {
        JSONObject json = new JSONObject();
        json.put("formatVersion", 2);
        json.put("engineType", "pytorch");
        json.put("modelName", modelName);
        json.put("createdAt", createdAt);
        json.put("protocol", protocol.name());
        json.put("windowSamples", WINDOW);
        json.put("hopSamples", HOP);
        JSONObject mapping = new JSONObject();
        for (Map.Entry<Integer, String> entry : labelMapping.entrySet()) {
            mapping.put(String.valueOf(entry.getKey()), entry.getValue());
        }
        json.put("labelMapping", mapping);
        // 保存归一化统计
        if (normMeans != null && normStds != null) {
            JSONArray meansArr = new JSONArray();
            for (float v : normMeans) meansArr.put(v);
            JSONArray stdsArr = new JSONArray();
            for (float v : normStds) stdsArr.put(v);
            json.put("normMeans", meansArr);
            json.put("normStds", stdsArr);
        }
        // 保存滤波器系数
        if (filterB != null && filterA != null) {
            JSONArray bArr = new JSONArray();
            for (double v : filterB) bArr.put(v);
            JSONArray aArr = new JSONArray();
            for (double v : filterA) aArr.put(v);
            json.put("filterB", bArr);
            json.put("filterA", aArr);
            if (filterZi != null) {
                JSONArray ziArr = new JSONArray();
                for (double v : filterZi) ziArr.put(v);
                json.put("filterZi", ziArr);
            }
        }
        return json;
    }

    /** 从 JSON 读取映射关系（不重新加载模型，模型由 load() 单独加载）。 */
    static Map<Integer, String> mappingFromJson(JSONObject json) throws JSONException {
        Map<Integer, String> mapping = new LinkedHashMap<>();
        JSONObject obj = json.getJSONObject("labelMapping");
        JSONArray keys = obj.names();
        if (keys != null) {
            for (int i = 0; i < keys.length(); i++) {
                String key = keys.getString(i);
                mapping.put(Integer.parseInt(key), obj.getString(key));
            }
        }
        return mapping;
    }

    /** 从 JSON 读取归一化统计，返回 float[8] 或 null。 */
    static float[] readNormArray(JSONObject json, String key) throws JSONException {
        if (!json.has(key)) return null;
        JSONArray arr = json.getJSONArray(key);
        float[] result = new float[arr.length()];
        for (int i = 0; i < arr.length(); i++) result[i] = (float) arr.getDouble(i);
        return result;
    }

    /** 从 JSON 读取 double 数组（滤波器系数），null 表示不存在。 */
    static double[] readDoubleArray(JSONObject json, String key) throws JSONException {
        if (!json.has(key)) return null;
        JSONArray arr = json.getJSONArray(key);
        double[] result = new double[arr.length()];
        for (int i = 0; i < arr.length(); i++) result[i] = arr.getDouble(i);
        return result;
    }
}