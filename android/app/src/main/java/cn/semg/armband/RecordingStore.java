package cn.semg.armband;

import android.content.Context;
import android.content.SharedPreferences;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

/** CSV recording persistence plus versioned, switchable local models. */
final class RecordingStore {
    private static final String ACTIVE_MODEL = "active_model";
    private static final String ACTIVE_PYTORCH_MODEL = "active_pytorch_model";
    /** "记录数据"按钮产生的原始数据标签；只用于本地回看，不参与模型训练与云端上传。 */
    static final String RAW_LABEL = "原始数据";

    static final class Summary {
        final Map<String, Integer> recordings;
        final long samples;
        Summary(Map<String, Integer> recordings, long samples) {
            this.recordings = recordings;
            this.samples = samples;
        }
    }

    static final class RecordingInfo {
        final File file;
        final String label;
        final String user;
        final long sampleCount;
        final long createdAt;
        final SemgPacketParser.Protocol protocol;

        RecordingInfo(File file, String label, String user, long sampleCount, long createdAt, SemgPacketParser.Protocol protocol) {
            this.file = file;
            this.label = label;
            this.user = user;
            this.sampleCount = sampleCount;
            this.createdAt = createdAt;
            this.protocol = protocol;
        }

        double durationSeconds() { return sampleCount / (double) SemgPacketParser.SAMPLE_RATE; }
    }

    static final class ModelInfo {
        final File file;
        final String name;
        final long createdAt;
        final GestureModel model;
        final boolean active;
        final SemgPacketParser.Protocol protocol;
        /** 训练所用用户名列表；null 表示全部用户（或旧版模型未记录）。 */
        final List<String> users;

        ModelInfo(File file, String name, long createdAt, GestureModel model, boolean active,
                  SemgPacketParser.Protocol protocol, List<String> users) {
            this.file = file;
            this.name = name;
            this.createdAt = createdAt;
            this.model = model;
            this.active = active;
            this.protocol = protocol;
            this.users = users;
        }

        String usersDisplay() {
            if (users == null) return "全部用户";
            return users.isEmpty() ? "全部用户" : String.join("、", users);
        }
    }

    /** PyTorch (.pt) 模型信息；module 在 list 时为 null（懒加载），激活后由 MainActivity 显式 load。 */
    static final class PyTorchModelInfo {
        final File ptFile;
        final File jsonFile;
        final String name;
        final long createdAt;
        final boolean active;
        final SemgPacketParser.Protocol protocol;
        final Map<Integer, String> labelMapping;
        final float[] normMeans;
        final float[] normStds;
        final double[] filterB;
        final double[] filterA;
        final double[] filterZi;
        PyTorchModel model;  // 加载后的实例，可能为 null

        PyTorchModelInfo(File ptFile, File jsonFile, String name, long createdAt, boolean active,
                         SemgPacketParser.Protocol protocol, Map<Integer, String> labelMapping) {
            this(ptFile, jsonFile, name, createdAt, active, protocol, labelMapping,
                    null, null, null, null, null);
        }

        PyTorchModelInfo(File ptFile, File jsonFile, String name, long createdAt, boolean active,
                         SemgPacketParser.Protocol protocol, Map<Integer, String> labelMapping,
                         float[] normMeans, float[] normStds) {
            this(ptFile, jsonFile, name, createdAt, active, protocol, labelMapping,
                    normMeans, normStds, null, null, null);
        }

        PyTorchModelInfo(File ptFile, File jsonFile, String name, long createdAt, boolean active,
                         SemgPacketParser.Protocol protocol, Map<Integer, String> labelMapping,
                         float[] normMeans, float[] normStds,
                         double[] filterB, double[] filterA, double[] filterZi) {
            this.ptFile = ptFile;
            this.jsonFile = jsonFile;
            this.name = name;
            this.createdAt = createdAt;
            this.active = active;
            this.protocol = protocol;
            this.labelMapping = labelMapping;
            this.normMeans = normMeans;
            this.normStds = normStds;
            this.filterB = filterB;
            this.filterA = filterA;
            this.filterZi = filterZi;
        }
    }

    final class Session {
        private final BufferedWriter writer;
        private final File file;
        private final SemgPacketParser.Protocol protocol;
        private long sampleCount;
        private boolean closed;
        private final String sessionId;

        Session(String label, String user, SemgPacketParser.Protocol protocol, String hand) throws IOException {
            String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(new Date());
            // 生成 session 标识：用户 + 递增计数器（每次开始新采集批次算新 session）
            // 使用 SharedPreferences 存储每个用户的 session 计数器
            String counterKey = "session_counter_" + safeName(user);
            int counter = preferences.getInt(counterKey, 0) + 1;
            preferences.edit().putInt(counterKey, counter).apply();
            sessionId = user + "_session_" + String.format("%03d", counter);
            file = new File(recordingsDir, timestamp + "_" + safeName(label) + ".csv");
            writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), StandardCharsets.UTF_8), 32768);
            writer.write("# sEMG recording v1\n");
            writer.write("# label=" + label.replace("\n", " ").replace("\r", " ") + "\n");
            writer.write("# user=" + user.replace("\n", " ").replace("\r", " ") + "\n");
            writer.write("# session=" + sessionId.replace("\n", " ").replace("\r", " ") + "\n");
            writer.write("# protocol=" + protocol.name() + "\n");
            writer.write("# hand=" + (hand == null ? "right" : hand) + "\n");
            writer.write("sample_index,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8\n");
            this.protocol = protocol;
        }

        /** 向后兼容：无 hand 参数时默认右手 */
        Session(String label, String user, SemgPacketParser.Protocol protocol) throws IOException {
            this(label, user, protocol, "right");
        }

        synchronized void append(long firstIndex, int[][] samples) {
            if (closed) return;
            try {
                StringBuilder line = new StringBuilder(96);
                for (int row = 0; row < samples.length; row++) {
                    line.setLength(0);
                    line.append(firstIndex + row);
                    for (int value : samples[row]) line.append(',').append(value);
                    line.append('\n');
                    writer.write(line.toString());
                    sampleCount++;
                }
            } catch (IOException error) {
                throw new IllegalStateException("写入采集数据失败", error);
            }
        }

        synchronized long close() throws IOException {
            if (!closed) {
                closed = true;
                writer.close();
            }
            if (sampleCount == 0) file.delete();
            return sampleCount;
        }
    }

    private final File recordingsDir;
    private final File modelsDir;
    private final File pytorchDir;
    private final File legacyModelFile;
    private final SharedPreferences preferences;

    RecordingStore(Context context) {
        // 使用外部公共存储目录，卸载 App 时数据不会自动删除
        // 路径：/storage/emulated/0/Android/data/cn.semg.armband/files/recordings
        File externalBase = context.getExternalFilesDir(null);
        if (externalBase == null) {
            // 降级到内部存储
            externalBase = context.getFilesDir();
        }
        
        recordingsDir = new File(externalBase, "recordings");
        modelsDir = new File(externalBase, "models");
        pytorchDir = new File(externalBase, "pytorch_models");
        legacyModelFile = new File(externalBase, "gesture_model.json");
        if (!recordingsDir.exists()) recordingsDir.mkdirs();
        if (!modelsDir.exists()) modelsDir.mkdirs();
        if (!pytorchDir.exists()) pytorchDir.mkdirs();
        preferences = context.getSharedPreferences("semg_storage", Context.MODE_PRIVATE);
        migrateLegacyModel();
    }

    Session start(String label, String user, SemgPacketParser.Protocol protocol) throws IOException {
        return new Session(label.trim(), user.trim(), protocol);
    }

    Session start(String label, String user, SemgPacketParser.Protocol protocol, String hand) throws IOException {
        return new Session(label.trim(), user.trim(), protocol, hand);
    }

    List<RecordingInfo> listRecordings() {
        List<RecordingInfo> result = new ArrayList<>();
        File[] files = recordingsDir.listFiles((dir, name) -> name.endsWith(".csv"));
        if (files == null) return result;
        Arrays.sort(files, (left, right) -> Long.compare(right.lastModified(), left.lastModified()));
        for (File file : files) {
            RecordingInfo info = inspectRecording(file);
            if (info != null) result.add(info);
        }
        return result;
    }

    Summary summary() {
        Map<String, Integer> counts = new LinkedHashMap<>();
        long samples = 0;
        for (RecordingInfo info : listRecordings()) {
            counts.put(info.label, counts.getOrDefault(info.label, 0) + 1);
            samples += info.sampleCount;
        }
        return new Summary(counts, samples);
    }

    int[][] loadPreview(RecordingInfo info, int maxSamples) throws IOException {
        ArrayDeque<int[]> recent = new ArrayDeque<>(maxSamples);
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(info.file), StandardCharsets.UTF_8))) {
            readHeader(reader);
            String line;
            while ((line = reader.readLine()) != null) {
                int[] row = parseSample(line);
                if (row == null) continue;
                if (recent.size() == maxSamples) recent.removeFirst();
                recent.addLast(row);
            }
        }
        return recent.toArray(new int[0][]);
    }

    boolean deleteRecording(RecordingInfo info) {
        return info.file.isFile() && info.file.delete();
    }

    /** 批量删除历史记录，返回成功删除的条数。 */
    int deleteRecordings(List<RecordingInfo> infos) {
        int deleted = 0;
        for (RecordingInfo info : infos) {
            if (deleteRecording(info)) deleted++;
        }
        return deleted;
    }

    /**
     * 训练模型。users 为 null 表示使用全部用户的数据；
     * 传入集合则只使用集合内用户名的录制数据（空集合会报错）。
     */
    GestureModel train(String modelName, SemgPacketParser.Protocol protocol,
                       GestureModel.Algorithm algorithm, java.util.Set<String> users) throws IOException {
        Map<String, List<double[]>> byLabel = new LinkedHashMap<>();
        File[] files = recordingsDir.listFiles((dir, name) -> name.endsWith(".csv"));
        if (files == null) files = new File[0];
        int usedFiles = 0;
        for (File file : files) {
            RecordingInfo info = inspectRecording(file);
            if (info == null || info.protocol != protocol) continue;
            if (RAW_LABEL.equals(info.label)) continue;   // 原始记录不参与训练
            if (users != null && !users.contains(info.user)) continue;
            readFeatures(file, byLabel);
            usedFiles++;
        }
        if (usedFiles == 0) {
            throw new IOException("未找到所选用户" + protocol.name() + "协议的录制数据，请切换协议或重新选择用户");
        }
        GestureModel model = GestureModel.train(byLabel, protocol, algorithm);
        long createdAt = System.currentTimeMillis();
        String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(new Date(createdAt));
        File destination = new File(modelsDir, stamp + "_" + safeName(modelName) + ".json");
        try {
            JSONObject json = model.toJson();
            json.put("modelName", modelName.trim());
            json.put("createdAt", createdAt);
            json.put("protocol", protocol.name());
            if (users != null) {
                org.json.JSONArray userArray = new org.json.JSONArray();
                for (String user : users) userArray.put(user);
                json.put("users", userArray);
            }
            writeJson(destination, json);
            preferences.edit().putString(ACTIVE_MODEL, destination.getName()).apply();
        } catch (Exception error) {
            throw new IOException("保存模型失败", error);
        }
        return model;
    }

    List<ModelInfo> listModels() {
        List<ModelInfo> result = new ArrayList<>();
        File[] files = modelsDir.listFiles((dir, name) -> name.endsWith(".json"));
        if (files == null) return result;
        Arrays.sort(files, (left, right) -> Long.compare(right.lastModified(), left.lastModified()));
        String activeName = preferences.getString(ACTIVE_MODEL, "");
        for (File file : files) {
            try {
                JSONObject json = readJson(file);
                GestureModel model = GestureModel.fromJson(json);
                String name = json.optString("modelName", stripExtension(file.getName()));
                long createdAt = json.optLong("createdAt", file.lastModified());
                SemgPacketParser.Protocol protocol = model.protocol;
                List<String> users = null;
                if (json.has("users")) {
                    users = new ArrayList<>();
                    org.json.JSONArray array = json.optJSONArray("users");
                    if (array != null) for (int i = 0; i < array.length(); i++) users.add(array.optString(i));
                }
                result.add(new ModelInfo(file, name, createdAt, model, file.getName().equals(activeName), protocol, users));
            } catch (Exception ignored) { }
        }
        if (!result.isEmpty() && result.stream().noneMatch(item -> item.active)) {
            preferences.edit().putString(ACTIVE_MODEL, result.get(0).file.getName()).apply();
            ModelInfo first = result.get(0);
            result.set(0, new ModelInfo(first.file, first.name, first.createdAt, first.model, true, first.protocol, first.users));
        }
        return result;
    }

    GestureModel loadModel() {
        for (ModelInfo info : listModels()) if (info.active) return info.model;
        return null;
    }

    String activeModelName() {
        for (ModelInfo info : listModels()) if (info.active) return info.name;
        return "";
    }

    void activateModel(ModelInfo info) {
        preferences.edit().putString(ACTIVE_MODEL, info.file.getName()).apply();
    }

    GestureModel deleteModel(ModelInfo info) {
        boolean wasActive = info.active;
        if (info.file.isFile()) info.file.delete();
        if (wasActive) preferences.edit().remove(ACTIVE_MODEL).apply();
        return loadModel();
    }

    // ---------------- PyTorch 模型管理 ----------------

    /**
     * 导入 .pt 文件到应用私有目录，并保存同名 .json 元数据（含标签映射）。
     * 调用者负责传入已配置好的标签映射；此方法做 IO 复制和持久化。
     */
    PyTorchModelInfo importPytorchModel(File sourceFile, String name,
                                        Map<Integer, String> mapping,
                                        SemgPacketParser.Protocol protocol) throws IOException {
        return importPytorchModel(sourceFile, name, mapping, protocol, null, null);
    }

    /**
     * 导入 .pt 文件到应用私有目录，并保存同名 .json 元数据（含标签映射和归一化统计）。
     */
    PyTorchModelInfo importPytorchModel(File sourceFile, String name,
                                        Map<Integer, String> mapping,
                                        SemgPacketParser.Protocol protocol,
                                        float[] normMeans, float[] normStds) throws IOException {
        return importPytorchModel(sourceFile, name, mapping, protocol,
                normMeans, normStds, null, null, null);
    }

    /**
     * 导入 .pt 文件到应用私有目录，并保存同名 .json 元数据
     * （含标签映射、归一化统计、滤波器系数）。
     */
    PyTorchModelInfo importPytorchModel(File sourceFile, String name,
                                        Map<Integer, String> mapping,
                                        SemgPacketParser.Protocol protocol,
                                        float[] normMeans, float[] normStds,
                                        double[] filterB, double[] filterA,
                                        double[] filterZi) throws IOException {
        long createdAt = System.currentTimeMillis();
        String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(new Date(createdAt));
        String baseName = stamp + "_" + safeName(name);
        File ptDest = new File(pytorchDir, baseName + ".pt");
        File jsonDest = new File(pytorchDir, baseName + ".json");

        // 复制 .pt 文件
        try (InputStream input = new FileInputStream(sourceFile);
             OutputStream output = new FileOutputStream(ptDest)) {
            byte[] buffer = new byte[32768];
            int count;
            while ((count = input.read(buffer)) >= 0) output.write(buffer, 0, count);
        }

        // 写元数据 JSON
        try {
            JSONObject json = new JSONObject();
            json.put("formatVersion", 2);
            json.put("engineType", "pytorch");
            json.put("modelName", name.trim());
            json.put("createdAt", createdAt);
            json.put("protocol", protocol.name());
            json.put("ptFileName", ptDest.getName());
            JSONObject mappingJson = new JSONObject();
            for (Map.Entry<Integer, String> entry : mapping.entrySet()) {
                mappingJson.put(String.valueOf(entry.getKey()), entry.getValue());
            }
            json.put("labelMapping", mappingJson);
            // 保存归一化统计
            if (normMeans != null && normStds != null) {
                JSONArray meansArr = new JSONArray();
                for (float v : normMeans) meansArr.put(v);
                JSONArray stdsArr = new JSONArray();
                for (float v : normStds) stdsArr.put(v);
                json.put("normMeans", meansArr);
                json.put("normStds", stdsArr);
            }
            // 保存滤波器系数（云端下发，保证与训练端滤波一致）
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
            writeJson(jsonDest, json);
        } catch (Exception error) {
            ptDest.delete();
            throw new IOException("保存 PyTorch 模型元数据失败", error);
        }

        // 自动激活新导入的模型，清除质心激活标记
        preferences.edit().putString(ACTIVE_PYTORCH_MODEL, ptDest.getName())
                .remove(ACTIVE_MODEL).apply();
        return new PyTorchModelInfo(ptDest, jsonDest, name.trim(), createdAt, true, protocol, mapping,
                normMeans, normStds, filterB, filterA, filterZi);
    }

    /**
     * 扫描 pytorchDir，列出所有 .pt 模型（不加载 Module，懒加载由调用者完成）。
     */
    List<PyTorchModelInfo> listPytorchModels() {
        List<PyTorchModelInfo> result = new ArrayList<>();
        File[] ptFiles = pytorchDir.listFiles((dir, n) -> n.endsWith(".pt"));
        if (ptFiles == null) return result;
        Arrays.sort(ptFiles, (l, r) -> Long.compare(r.lastModified(), l.lastModified()));
        String activeName = preferences.getString(ACTIVE_PYTORCH_MODEL, "");
        for (File ptFile : ptFiles) {
            String baseName = ptFile.getName().substring(0, ptFile.getName().length() - 3);
            File jsonFile = new File(pytorchDir, baseName + ".json");
            String name = baseName;
            long createdAt = ptFile.lastModified();
            SemgPacketParser.Protocol protocol = SemgPacketParser.Protocol.STM32;
            Map<Integer, String> mapping = new LinkedHashMap<>();
            float[] normMeans = null;
            float[] normStds = null;
            double[] filterB = null;
            double[] filterA = null;
            double[] filterZi = null;
            try {
                if (jsonFile.isFile()) {
                    JSONObject json = readJson(jsonFile);
                    name = json.optString("modelName", baseName);
                    createdAt = json.optLong("createdAt", ptFile.lastModified());
                    String protocolName = json.optString("protocol", "STM32");
                    try { protocol = SemgPacketParser.Protocol.valueOf(protocolName); }
                    catch (IllegalArgumentException ignored) { }
                    mapping = PyTorchModel.mappingFromJson(json);
                    normMeans = PyTorchModel.readNormArray(json, "normMeans");
                    normStds = PyTorchModel.readNormArray(json, "normStds");
                    filterB = PyTorchModel.readDoubleArray(json, "filterB");
                    filterA = PyTorchModel.readDoubleArray(json, "filterA");
                    filterZi = PyTorchModel.readDoubleArray(json, "filterZi");
                }
            } catch (Exception ignored) { }
            result.add(new PyTorchModelInfo(ptFile, jsonFile, name, createdAt,
                    ptFile.getName().equals(activeName), protocol, mapping,
                    normMeans, normStds, filterB, filterA, filterZi));
        }
        return result;
    }

    /**
     * 激活 PyTorch 模型；清除质心模型激活标记。
     */
    void activatePytorchModel(PyTorchModelInfo info) {
        preferences.edit()
                .putString(ACTIVE_PYTORCH_MODEL, info.ptFile.getName())
                .remove(ACTIVE_MODEL)
                .apply();
    }

    /**
     * 删除 PyTorch 模型文件（.pt + .json）。返回是否仍存在其它已激活的 PyTorch 模型。
     */
    boolean deletePytorchModel(PyTorchModelInfo info) {
        boolean wasActive = info.active;
        if (info.ptFile.isFile()) info.ptFile.delete();
        if (info.jsonFile.isFile()) info.jsonFile.delete();
        if (wasActive) preferences.edit().remove(ACTIVE_PYTORCH_MODEL).apply();
        return wasActive;
    }

    /**
     * 返回当前激活的 PyTorch 模型信息（如未激活返回 null）。Module 不预加载。
     */
    PyTorchModelInfo activePytorchModel() {
        for (PyTorchModelInfo info : listPytorchModels()) if (info.active) return info;
        return null;
    }

    /**
     * 返回当前激活引擎类型："pytorch" 或 "centroid"。
     * 用于 MainActivity 决定加载哪种引擎。
     */
    String activeEngineType() {
        if (preferences.contains(ACTIVE_PYTORCH_MODEL)) return "pytorch";
        if (preferences.contains(ACTIVE_MODEL)) return "centroid";
        // 默认回退：仅有 PyTorch 模型时选 pytorch，否则 centroid
        return listModels().isEmpty() && !listPytorchModels().isEmpty() ? "pytorch" : "centroid";
    }

    private RecordingInfo inspectRecording(File file) {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8))) {
            Map<String, String> header = readHeader(reader);
            long count = 0;
            while (reader.readLine() != null) count++;
            String label = header.getOrDefault("label", "unknown");
            String user = header.getOrDefault("user", "未署名");
            SemgPacketParser.Protocol protocol = SemgPacketParser.Protocol.STM32;
            try { protocol = SemgPacketParser.Protocol.valueOf(header.getOrDefault("protocol", "STM32")); }
            catch (IllegalArgumentException ignored) { }
            return new RecordingInfo(file, label, user, count, file.lastModified(), protocol);
        } catch (IOException ignored) {
            return null;
        }
    }

    /** 读取全部 "# key=value" 头部行，遇到首个非注释行停止。兼容新旧格式（无 user 行）。 */
    private static Map<String, String> readHeader(BufferedReader reader) throws IOException {
        Map<String, String> header = new LinkedHashMap<>();
        reader.mark(4096);
        String line;
        while ((line = reader.readLine()) != null) {
            if (!line.startsWith("#")) break;
            int eq = line.indexOf('=');
            if (eq > 2) header.put(line.substring(2, eq).trim(), line.substring(eq + 1));
            reader.mark(4096);
        }
        reader.reset();
        return header;
    }

    private void readFeatures(File file, Map<String, List<double[]>> byLabel) throws IOException {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8))) {
            Map<String, String> header = readHeader(reader);
            String label = header.getOrDefault("label", "").trim();
            if (label.isEmpty()) return;
            List<int[]> samples = new ArrayList<>();
            String line;
            long previousIndex = -1;
            while ((line = reader.readLine()) != null) {
                String[] fields = line.split(",");
                if (fields.length != 9) continue;
                long index;
                int[] row = parseSample(fields);
                try { index = Long.parseLong(fields[0]); }
                catch (NumberFormatException ignored) { continue; }
                if (row == null) continue;
                if (previousIndex >= 0 && index != previousIndex + 1) {
                    addWindows(label, samples, byLabel);
                    samples.clear();
                }
                samples.add(row);
                previousIndex = index;
            }
            addWindows(label, samples, byLabel);
        }
    }

    private void addWindows(String label, List<int[]> samples, Map<String, List<double[]>> byLabel) {
        List<double[]> target = byLabel.computeIfAbsent(label, unused -> new ArrayList<>());
        int trim = Math.min(125, Math.max(0, (samples.size() - GestureModel.WINDOW) / 4));
        for (int start = trim; start + GestureModel.WINDOW <= samples.size() - trim; start += GestureModel.HOP) {
            int[][] window = new int[GestureModel.WINDOW][];
            for (int i = 0; i < window.length; i++) window[i] = samples.get(start + i);
            target.add(GestureModel.features(window));
        }
    }

    private void migrateLegacyModel() {
        if (preferences.getBoolean("legacy_migrated", false)) return;
        File[] current = modelsDir.listFiles((dir, name) -> name.endsWith(".json"));
        if (!legacyModelFile.isFile() || (current != null && current.length > 0)) {
            preferences.edit().putBoolean("legacy_migrated", true).apply();
            return;
        }
        try {
            JSONObject json = readJson(legacyModelFile);
            long createdAt = legacyModelFile.lastModified();
            json.put("modelName", "默认模型（旧版本）");
            json.put("createdAt", createdAt);
            File migrated = new File(modelsDir, "legacy_" + createdAt + ".json");
            writeJson(migrated, json);
            preferences.edit().putString(ACTIVE_MODEL, migrated.getName())
                    .putBoolean("legacy_migrated", true).apply();
        } catch (Exception ignored) { }
    }

    private static int[] parseSample(String line) {
        return parseSample(line.split(","));
    }

    private static int[] parseSample(String[] fields) {
        if (fields.length != 9) return null;
        int[] row = new int[SemgPacketParser.CHANNELS];
        try {
            for (int channel = 0; channel < row.length; channel++) {
                row[channel] = Integer.parseInt(fields[channel + 1]);
            }
            return row;
        } catch (NumberFormatException ignored) {
            return null;
        }
    }

    private static JSONObject readJson(File file) throws Exception {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8))) {
            StringBuilder content = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) content.append(line);
            return new JSONObject(content.toString());
        }
    }

    private static void writeJson(File file, JSONObject json) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), StandardCharsets.UTF_8))) {
            writer.write(json.toString(2));
        } catch (Exception error) {
            throw new IOException(error);
        }
    }

    private static String stripExtension(String name) {
        int dot = name.lastIndexOf('.');
        return dot > 0 ? name.substring(0, dot) : name;
    }

    // ---------------- 数据迁移与导入 ----------------

    /**
     * 将旧版内部存储的数据迁移到外部存储目录。
     * 仅在首次启动时执行一次，迁移完成后删除旧数据。
     * @return 迁移的文件数量，0 表示无需迁移
     */
    int migrateFromInternalStorage(Context context) {
        if (preferences.getBoolean("internal_migrated", false)) return 0;

        File internalBase = context.getFilesDir();
        File internalRecordings = new File(internalBase, "recordings");
        File internalModels = new File(internalBase, "models");
        File internalPytorch = new File(internalBase, "pytorch_models");
        File internalLegacy = new File(internalBase, "gesture_model.json");

        int migrated = 0;

        // 迁移采集数据
        if (internalRecordings.isDirectory()) {
            File[] csvFiles = internalRecordings.listFiles((dir, name) -> name.endsWith(".csv"));
            if (csvFiles != null) {
                for (File csvFile : csvFiles) {
                    File dest = new File(recordingsDir, csvFile.getName());
                    if (!dest.exists()) {
                        if (copyFile(csvFile, dest)) {
                            migrated++;
                        }
                    }
                }
            }
        }

        // 迁移质心/SVM 模型
        if (internalModels.isDirectory()) {
            File[] jsonFiles = internalModels.listFiles((dir, name) -> name.endsWith(".json"));
            if (jsonFiles != null) {
                for (File jsonFile : jsonFiles) {
                    File dest = new File(modelsDir, jsonFile.getName());
                    if (!dest.exists()) {
                        if (copyFile(jsonFile, dest)) {
                            migrated++;
                        }
                    }
                }
            }
        }

        // 迁移 PyTorch 模型
        if (internalPytorch.isDirectory()) {
            File[] ptFiles = internalPytorch.listFiles((dir, name) -> name.endsWith(".pt"));
            if (ptFiles != null) {
                for (File ptFile : ptFiles) {
                    File dest = new File(pytorchDir, ptFile.getName());
                    if (!dest.exists()) {
                        if (copyFile(ptFile, dest)) {
                            migrated++;
                        }
                    }
                    // 同时迁移对应的 .json 元数据
                    String baseName = ptFile.getName().substring(0, ptFile.getName().length() - 3);
                    File jsonFile = new File(internalPytorch, baseName + ".json");
                    if (jsonFile.exists()) {
                        File jsonDest = new File(pytorchDir, baseName + ".json");
                        if (!jsonDest.exists()) {
                            copyFile(jsonFile, jsonDest);
                        }
                    }
                }
            }
        }

        // 迁移旧版模型文件
        if (internalLegacy.isFile() && !legacyModelFile.exists()) {
            copyFile(internalLegacy, legacyModelFile);
            migrated++;
        }

        // 标记已迁移，下次不再执行
        preferences.edit().putBoolean("internal_migrated", true).apply();

        return migrated;
    }

    /**
     * 从外部导入 CSV 数据文件。
     * @param sourceFiles 要导入的 CSV 文件列表
     * @return 成功导入的文件数量
     */
    int importRecordings(List<File> sourceFiles) {
        int imported = 0;
        for (File source : sourceFiles) {
            if (source == null || !source.exists() || !source.getName().endsWith(".csv")) continue;
            String destName = source.getName();
            File dest = new File(recordingsDir, destName);
            // 文件名冲突时自动追加序号
            int index = 1;
            while (dest.exists()) {
                String baseName = stripExtension(destName);
                dest = new File(recordingsDir, baseName + "_" + index + ".csv");
                index++;
            }
            if (copyFile(source, dest)) {
                imported++;
            }
        }
        return imported;
    }

    /**
     * 从外部导入 CSV 数据文件（通过 InputStream）。
     * @param inputStream 文件输入流
     * @param fileName 原始文件名
     * @return 是否导入成功
     */
    boolean importRecording(InputStream inputStream, String fileName) {
        if (inputStream == null || fileName == null || !fileName.endsWith(".csv")) return false;
        String destName = fileName;
        File dest = new File(recordingsDir, destName);
        int index = 1;
        while (dest.exists()) {
            String baseName = stripExtension(destName);
            dest = new File(recordingsDir, baseName + "_" + index + ".csv");
            index++;
        }
        return copyStream(inputStream, dest);
    }

    // ---------------- 打包备份与恢复 ----------------

    /** 备份包内的清单文件名。 */
    private static final String BACKUP_MANIFEST = "backup_manifest.json";

    static final class BackupResult {
        final int recordings;
        final int models;
        final int pytorchModels;
        BackupResult(int recordings, int models, int pytorchModels) {
            this.recordings = recordings;
            this.models = models;
            this.pytorchModels = pytorchModels;
        }
        int total() { return recordings + models + pytorchModels; }
    }

    /**
     * 把采集数据与全部模型打包成 zip 写入 output，便于换机、重装或归档。
     * 目录结构：recordings/*.csv、models/*.json、pytorch_models/*.pt|*.json
     * @return 打包的文件数量
     */
    int exportBackup(OutputStream output) throws IOException {
        int count = 0;
        try (ZipOutputStream zip = new ZipOutputStream(new BufferedOutputStream(output))) {
            count += zipDirectory(zip, recordingsDir, "recordings/");
            count += zipDirectory(zip, modelsDir, "models/");
            count += zipDirectory(zip, pytorchDir, "pytorch_models/");

            // 清单用最简 JSON 手写，避免 JSONObject.put 抛受检异常
            String manifest = "{\"formatVersion\":1,\"createdAt\":" + System.currentTimeMillis()
                    + ",\"fileCount\":" + count + "}";
            zip.putNextEntry(new ZipEntry(BACKUP_MANIFEST));
            zip.write(manifest.getBytes(StandardCharsets.UTF_8));
            zip.closeEntry();
        }
        return count;
    }

    /** 把目录下所有文件写入 zip 的指定前缀下，返回写入数量。 */
    private static int zipDirectory(ZipOutputStream zip, File dir, String prefix) throws IOException {
        File[] files = dir.listFiles(File::isFile);
        if (files == null) return 0;
        int count = 0;
        for (File file : files) {
            zip.putNextEntry(new ZipEntry(prefix + file.getName()));
            try (InputStream in = new FileInputStream(file)) {
                byte[] buffer = new byte[32768];
                int read;
                while ((read = in.read(buffer)) >= 0) zip.write(buffer, 0, read);
            }
            zip.closeEntry();
            count++;
        }
        return count;
    }

    /**
     * 从 zip 备份恢复数据与模型。同名文件不覆盖，自动追加序号。
     * PyTorch 模型是 .pt + 同名 .json 配对，重命名时两者保持一致。
     */
    BackupResult importBackup(InputStream input) throws IOException {
        int recordings = 0, models = 0, pytorchModels = 0;
        Map<String, String> pytorchRenames = new LinkedHashMap<>();
        try (ZipInputStream zip = new ZipInputStream(new BufferedInputStream(input))) {
            ZipEntry entry;
            byte[] buffer = new byte[32768];
            while ((entry = zip.getNextEntry()) != null) {
                if (entry.isDirectory()) continue;
                String name = entry.getName();
                if (name.equals(BACKUP_MANIFEST) || name.startsWith("__MACOSX/")) continue;

                File target = null;
                if (name.startsWith("recordings/")) {
                    target = uniqueFile(recordingsDir, baseName(name));
                    recordings++;
                } else if (name.startsWith("models/")) {
                    target = uniqueFile(modelsDir, baseName(name));
                    models++;
                } else if (name.startsWith("pytorch_models/")) {
                    String base = baseName(name);
                    String stem = stripExtension(base);
                    String ext = extensionOf(base);
                    String mapped = pytorchRenames.get(stem);
                    if (mapped == null) {
                        mapped = uniqueModelBase(pytorchDir, stem);
                        pytorchRenames.put(stem, mapped);
                    }
                    target = new File(pytorchDir, mapped + ext);
                    pytorchModels++;
                }
                if (target == null) continue;

                File parent = target.getParentFile();
                if (parent != null) parent.mkdirs();
                try (OutputStream out = new FileOutputStream(target)) {
                    int read;
                    while ((read = zip.read(buffer)) >= 0) out.write(buffer, 0, read);
                }
                zip.closeEntry();
            }
        }
        return new BackupResult(recordings, models, pytorchModels);
    }

    /** 取 zip 条目路径中的文件名。 */
    private static String baseName(String path) {
        int slash = path.lastIndexOf('/');
        return slash >= 0 ? path.substring(slash + 1) : path;
    }

    /** 取文件扩展名（含点）；无扩展名时返回空串。 */
    private static String extensionOf(String name) {
        int dot = name.lastIndexOf('.');
        return dot >= 0 ? name.substring(dot) : "";
    }

    /** 目标目录下不存在同名文件时返回 name，否则追加 _1、_2 … */
    private static File uniqueFile(File dir, String name) {
        File dest = new File(dir, name);
        if (!dest.exists()) return dest;
        String stem = stripExtension(name);
        String ext = extensionOf(name);
        int index = 1;
        while (true) {
            dest = new File(dir, stem + "_" + index + ext);
            if (!dest.exists()) return dest;
            index++;
        }
    }

    /** 为 PyTorch 模型找一组不冲突的基名（.pt 与 .json 必须同名）。 */
    private static String uniqueModelBase(File dir, String stem) {
        if (!new File(dir, stem + ".pt").exists() && !new File(dir, stem + ".json").exists()) return stem;
        int index = 1;
        while (true) {
            String candidate = stem + "_" + index;
            if (!new File(dir, candidate + ".pt").exists()
                    && !new File(dir, candidate + ".json").exists()) {
                return candidate;
            }
            index++;
        }
    }

    /** 复制文件 */
    private static boolean copyFile(File source, File dest) {
        try (InputStream input = new FileInputStream(source);
             OutputStream output = new FileOutputStream(dest)) {
            byte[] buffer = new byte[32768];
            int count;
            while ((count = input.read(buffer)) >= 0) {
                output.write(buffer, 0, count);
            }
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    /** 复制输入流到文件 */
    private static boolean copyStream(InputStream input, File dest) {
        try (OutputStream output = new FileOutputStream(dest)) {
            byte[] buffer = new byte[32768];
            int count;
            while ((count = input.read(buffer)) >= 0) {
                output.write(buffer, 0, count);
            }
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    private static String safeName(String label) {
        String safe = label.replaceAll("[^a-zA-Z0-9\\p{L}_-]+", "_");
        return safe.isEmpty() ? "item" : safe.substring(0, Math.min(30, safe.length()));
    }
}