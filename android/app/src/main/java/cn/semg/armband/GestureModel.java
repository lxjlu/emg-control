package cn.semg.armband;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_print_interface;
import libsvm.svm_problem;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.StringReader;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** On-device gesture classifier: nearest centroid, linear SVM or RBF SVM (LIBSVM). */
final class GestureModel {
    static final int WINDOW = 100; // 200 ms at 500 Hz
    static final int HOP = 25;

    enum Algorithm {
        CENTROID("简单模型1"),
        LINEAR_SVM("线性模型1"),
        RBF_SVM("非线性模型1");
        final String display;
        Algorithm(String display) { this.display = display; }

        static Algorithm fromName(String name) {
            if (name != null) for (Algorithm algorithm : values())
                if (algorithm.name().equals(name)) return algorithm;
            return CENTROID;
        }
    }

    static {
        // Silence LIBSVM training output on Android.
        svm.svm_set_print_string_function(new svm_print_interface() {
            @Override public void print(String s) { }
        });
    }

    static final class Prediction {
        final String label;
        final double confidence;
        Prediction(String label, double confidence) { this.label = label; this.confidence = confidence; }
    }

    final Algorithm algorithm;
    final double[] mean;
    final double[] scale;
    final Map<String, double[]> centroids; // CENTROID only, otherwise null
    final svm_model svmModel;              // SVM only, otherwise null
    final List<String> labels;             // sorted class labels
    final int trainingWindows;
    final SemgPacketParser.Protocol protocol;

    private GestureModel(Algorithm algorithm, double[] mean, double[] scale,
                         Map<String, double[]> centroids, svm_model svmModel, List<String> labels,
                         int trainingWindows, SemgPacketParser.Protocol protocol) {
        this.algorithm = algorithm;
        this.mean = mean;
        this.scale = scale;
        this.centroids = centroids;
        this.svmModel = svmModel;
        this.labels = labels;
        this.trainingWindows = trainingWindows;
        this.protocol = protocol;
    }

    List<String> labels() { return labels; }

    int labelCount() { return labels.size(); }

    static GestureModel train(Map<String, List<double[]>> byLabel, SemgPacketParser.Protocol protocol, Algorithm algorithm) {
        if (byLabel.size() < 2) throw new IllegalArgumentException("至少需要两种动作数据（建议包含 rest）");
        List<double[]> all = new ArrayList<>();
        for (Map.Entry<String, List<double[]>> entry : byLabel.entrySet()) {
            if (entry.getValue().size() < 3) throw new IllegalArgumentException(
                    "动作“" + entry.getKey() + "”有效窗口不足，请至少采集 1 秒");
            all.addAll(entry.getValue());
        }
        int dimensions = all.get(0).length;
        double[] mean = new double[dimensions];
        double[] scale = new double[dimensions];
        for (double[] row : all) for (int j = 0; j < dimensions; j++) mean[j] += row[j];
        for (int j = 0; j < dimensions; j++) mean[j] /= all.size();
        for (double[] row : all) for (int j = 0; j < dimensions; j++) {
            double delta = row[j] - mean[j]; scale[j] += delta * delta;
        }
        for (int j = 0; j < dimensions; j++) scale[j] = Math.max(1e-6, Math.sqrt(scale[j] / all.size()));

        List<String> labels = new ArrayList<>(byLabel.keySet());
        Collections.sort(labels);

        if (algorithm == Algorithm.CENTROID) {
            Map<String, double[]> centroids = new LinkedHashMap<>();
            for (String label : labels) {
                double[] centroid = new double[dimensions];
                List<double[]> rows = byLabel.get(label);
                for (double[] row : rows) for (int j = 0; j < dimensions; j++) {
                    centroid[j] += (row[j] - mean[j]) / scale[j];
                }
                for (int j = 0; j < dimensions; j++) centroid[j] /= rows.size();
                centroids.put(label, centroid);
            }
            return new GestureModel(Algorithm.CENTROID, mean, scale, centroids, null, labels, all.size(), protocol);
        }

        svm_model svmModel = trainSvm(byLabel, labels, mean, scale, dimensions, algorithm);
        return new GestureModel(algorithm, mean, scale, null, svmModel, labels, all.size(), protocol);
    }

    private static svm_model trainSvm(Map<String, List<double[]>> byLabel, List<String> labels,
                                      double[] mean, double[] scale, int dimensions, Algorithm algorithm) {
        int total = 0;
        for (List<double[]> rows : byLabel.values()) total += rows.size();
        svm_problem problem = new svm_problem();
        problem.l = total;
        problem.y = new double[total];
        problem.x = new svm_node[total][];
        int at = 0;
        for (int labelIndex = 0; labelIndex < labels.size(); labelIndex++) {
            for (double[] row : byLabel.get(labels.get(labelIndex))) {
                problem.y[at] = labelIndex + 1;
                svm_node[] nodes = new svm_node[dimensions];
                for (int j = 0; j < dimensions; j++) {
                    svm_node node = new svm_node();
                    node.index = j + 1;
                    node.value = (row[j] - mean[j]) / scale[j];
                    nodes[j] = node;
                }
                problem.x[at] = nodes;
                at++;
            }
        }
        svm_parameter param = new svm_parameter();
        param.svm_type = svm_parameter.C_SVC;
        param.kernel_type = algorithm == Algorithm.RBF_SVM ? svm_parameter.RBF : svm_parameter.LINEAR;
        param.degree = 3;
        param.gamma = 1.0 / dimensions;
        param.coef0 = 0;
        param.cache_size = 100;
        param.eps = 1e-3;
        param.C = 1;
        param.nr_weight = 0;
        param.weight_label = null;
        param.weight = null;
        param.nu = 0.5;
        param.p = 0.1;
        param.shrinking = 1;
        param.probability = 1;
        String check = svm.svm_check_parameter(problem, param);
        if (check != null) throw new IllegalArgumentException("SVM 参数错误：" + check);
        return svm.svm_train(problem, param);
    }

    Prediction predict(int[][] window) {
        double[] feature = features(window);
        if (algorithm != Algorithm.CENTROID) return predictSvm(feature);
        double minDistance = Double.POSITIVE_INFINITY;
        String winner = "—";
        Map<String, Double> distances = new LinkedHashMap<>();
        for (Map.Entry<String, double[]> entry : centroids.entrySet()) {
            double distance = 0;
            for (int j = 0; j < feature.length; j++) {
                double delta = (feature[j] - mean[j]) / scale[j] - entry.getValue()[j];
                distance += delta * delta;
            }
            distance /= feature.length;
            distances.put(entry.getKey(), distance);
            if (distance < minDistance) { minDistance = distance; winner = entry.getKey(); }
        }
        double total = 0;
        double winnerScore = 0;
        for (Map.Entry<String, Double> entry : distances.entrySet()) {
            double score = Math.exp(-Math.min(50, (entry.getValue() - minDistance) * 1.5));
            total += score;
            if (entry.getKey().equals(winner)) winnerScore = score;
        }
        return new Prediction(winner, total == 0 ? 0 : winnerScore / total);
    }

    private Prediction predictSvm(double[] feature) {
        svm_node[] nodes = new svm_node[feature.length];
        for (int j = 0; j < feature.length; j++) {
            svm_node node = new svm_node();
            node.index = j + 1;
            node.value = (feature[j] - mean[j]) / scale[j];
            nodes[j] = node;
        }
        double[] probabilities = new double[svmModel.nr_class];
        double predicted = svm.svm_predict_probability(svmModel, nodes, probabilities);
        int index = (int) predicted - 1;
        if (index < 0 || index >= labels.size()) return new Prediction("—", 0);
        double confidence = 0;
        for (int i = 0; i < svmModel.nr_class; i++) {
            if (svmModel.label != null && svmModel.label[i] == (int) predicted) confidence = probabilities[i];
        }
        return new Prediction(labels.get(index), confidence);
    }

    /** Gain-tolerant time-domain sEMG features: six measures per channel plus energy ratios. */
    static double[] features(int[][] window) {
        if (window.length < 32) throw new IllegalArgumentException("动作窗口太短");
        double[] result = new double[SemgPacketParser.CHANNELS * 7];
        double[] rms = new double[SemgPacketParser.CHANNELS];
        double totalRms = 1e-9;
        for (int channel = 0; channel < SemgPacketParser.CHANNELS; channel++) {
            double mean = 0;
            for (int[] sample : window) mean += sample[channel];
            mean /= window.length;
            double mav = 0, square = 0, length = 0, zero = 0, slope = 0;
            double previous = window[0][channel] - mean;
            double previousDelta = 0;
            for (int i = 0; i < window.length; i++) {
                double value = window[i][channel] - mean;
                mav += Math.abs(value);
                square += value * value;
                if (i > 0) {
                    double delta = value - previous;
                    length += Math.abs(delta);
                    if (value * previous < 0 && Math.abs(delta) > 2) zero++;
                    if (i > 1 && delta * previousDelta < 0
                            && Math.abs(delta) > 2 && Math.abs(previousDelta) > 2) slope++;
                    previousDelta = delta;
                }
                previous = value;
            }
            mav /= window.length;
            rms[channel] = Math.sqrt(square / window.length + 1e-9);
            totalRms += rms[channel];
            int at = channel * 7;
            result[at] = Math.log1p(mav);
            result[at + 1] = Math.log1p(rms[channel]);
            result[at + 2] = Math.log1p(length / Math.max(1, window.length - 1));
            result[at + 3] = Math.log1p(square / window.length);
            result[at + 4] = zero / Math.max(1, window.length - 1.0);
            result[at + 5] = slope / Math.max(1, window.length - 2.0);
        }
        for (int channel = 0; channel < SemgPacketParser.CHANNELS; channel++) {
            result[channel * 7 + 6] = rms[channel] / totalRms;
        }
        return result;
    }

    JSONObject toJson() throws JSONException {
        JSONObject json = new JSONObject();
        json.put("formatVersion", 2);
        json.put("algorithm", algorithm.name());
        json.put("windowSamples", WINDOW);
        json.put("trainingWindows", trainingWindows);
        json.put("protocol", protocol.name());
        json.put("mean", array(mean));
        json.put("scale", array(scale));
        json.put("labels", new JSONArray(labels));
        if (algorithm == Algorithm.CENTROID) {
            JSONObject storedCentroids = new JSONObject();
            for (Map.Entry<String, double[]> entry : centroids.entrySet()) {
                storedCentroids.put(entry.getKey(), array(entry.getValue()));
            }
            json.put("centroids", storedCentroids);
        } else {
            try {
                StringWriter writer = new StringWriter();
                svm.svm_save_model(writer, svmModel);
                json.put("svmModel", writer.toString());
            } catch (Exception error) {
                throw new JSONException("SVM 模型序列化失败");
            }
        }
        return json;
    }

    static GestureModel fromJson(JSONObject json) throws JSONException {
        Algorithm algorithm = Algorithm.fromName(json.optString("algorithm", ""));
        double[] mean = doubles(json.getJSONArray("mean"));
        double[] scale = doubles(json.getJSONArray("scale"));
        List<String> labels = new ArrayList<>();
        JSONArray storedLabels = json.optJSONArray("labels");
        if (storedLabels != null) for (int i = 0; i < storedLabels.length(); i++) labels.add(storedLabels.getString(i));
        SemgPacketParser.Protocol protocol = SemgPacketParser.Protocol.STM32;
        String protocolName = json.optString("protocol", "");
        if (!protocolName.isEmpty()) {
            try { protocol = SemgPacketParser.Protocol.valueOf(protocolName); }
            catch (IllegalArgumentException ignored) { protocol = SemgPacketParser.Protocol.STM32; }
        }
        if (algorithm == Algorithm.CENTROID) {
            JSONObject stored = json.getJSONObject("centroids");
            Map<String, double[]> centroids = new LinkedHashMap<>();
            JSONArray names = stored.names();
            if (names != null) for (int i = 0; i < names.length(); i++) {
                String name = names.getString(i);
                centroids.put(name, doubles(stored.getJSONArray(name)));
            }
            if (labels.isEmpty()) labels.addAll(centroids.keySet());
            return new GestureModel(Algorithm.CENTROID, mean, scale, centroids, null, labels,
                    json.optInt("trainingWindows", 0), protocol);
        }
        try {
            BufferedReader reader = new BufferedReader(new StringReader(json.getString("svmModel")));
            svm_model svmModel = svm.svm_load_model(reader);
            if (labels.isEmpty() && svmModel.label != null) {
                for (int value : svmModel.label) labels.add("类别" + value);
            }
            return new GestureModel(algorithm, mean, scale, null, svmModel, labels,
                    json.optInt("trainingWindows", 0), protocol);
        } catch (Exception error) {
            throw new JSONException("SVM 模型加载失败");
        }
    }

    private static JSONArray array(double[] values) throws JSONException {
        JSONArray array = new JSONArray();
        for (double value : values) array.put(value);
        return array;
    }

    private static double[] doubles(JSONArray array) throws JSONException {
        double[] values = new double[array.length()];
        for (int i = 0; i < values.length; i++) values[i] = array.getDouble(i);
        return values;
    }
}