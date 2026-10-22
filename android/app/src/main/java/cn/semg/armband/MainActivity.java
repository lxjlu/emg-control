package cn.semg.armband;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.bluetooth.BluetoothAdapter;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.InputType;
import android.text.TextUtils;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ArrayAdapter;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.FileOutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLDecoder;
import java.text.SimpleDateFormat;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity implements BleArmbandClient.Listener, ClassicArmbandClient.Listener {
    private static final int REQUEST_BLUETOOTH = 41;
    private static final int REQUEST_EXPORT = 42;
    private static final int REQUEST_IMPORT_PT = 43;
    private static final int REQUEST_IMPORT_DATA = 44;
    private static final int REQUEST_EXPORT_BACKUP = 45;
    private static final int REQUEST_IMPORT_BACKUP = 46;
    private static final int INK = 0xFFE9F0F4;
    private static final int MUTED = 0xFF8FA0AA;
    private static final int SURFACE = 0xFF111D24;
    private static final int ACCENT = 0xFF42E8B4;

    private final Handler main = new Handler(Looper.getMainLooper());
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final SemgPacketParser parser = new SemgPacketParser();
    private final Object inferenceLock = new Object();
    private final ArrayDeque<int[]> inferenceWindow = new ArrayDeque<>();
    private final ArrayDeque<String> recentPredictions = new ArrayDeque<>();

    private Spinner linkModeSpinner;
    private boolean classicMode;
    private BleArmbandClient ble;
    private ClassicArmbandClient classic;
    private G20HttpController g20;
    private RecordingStore store;
    private RecordingStore.Session recording;
    private boolean rawRecording;          // 当前记录是否为"记录数据"（原始数据，不选标签）
    private GestureModel model;            // 当前质心/SVM 模型（如 PyTorch 激活则为 null）
    private PyTorchModel pytorchModel;     // 当前 PyTorch 模型（如质心激活则为 null）
    private Uri pendingImportUri;          // 待导入的 .pt 文件 Uri（在配置完映射后处理）
    private final List<String> cloudModelLabels = new ArrayList<>();  // 云端返回的类别标签顺序
    private float[] cloudNormMeans;        // 云端返回的归一化均值
    private float[] cloudNormStds;         // 云端返回的归一化标准差
    private double[] cloudFilterB;         // 云端返回的滤波器分子系数 b
    private double[] cloudFilterA;         // 云端返回的滤波器分母系数 a
    private double[] cloudFilterZi;        // 云端返回的滤波器稳态初始条件 zi
    private boolean connected;
    private boolean recognizing;
    private boolean validDataAnnounced;
    private int inferenceHop;
    private boolean inferenceHasResult;   // 是否已产出过一次推理结果（避免"准备中"反复覆盖结果导致闪烁）

    private WaveformView waveform;
    private TextView statusView;
    private TextView statsView;
    private TextView rawView;
    private TextView datasetView;
    private TextView modelView;
    private TextView resultView;
    private EditText userInput;
    private String authToken;              // 用户认证 Token
    private String currentUsername;        // 当前登录用户名
    private int currentUserId;             // 当前登录用户 ID
    private boolean isAdmin;               // 是否为管理员
    private Button userBtn;                // 顶部用户状态按钮
    private Spinner labelSpinner;
    private Spinner handSpinner;
    private EditText customLabelInput;
    private Button connectButton;
    private Button recordButton;
    private Button recordRawButton;
    private Button recognizeButton;
    private Button g20Button;
    private TextView g20StatusView;
    private EditText ipInput;
    private EditText portInput;
    private EditText timeoutInput;
    private Spinner deviceTypeSpinner;
    private Spinner algorithmSpinner;
    private TextView localIpView;
    private LinearLayout rawCardHeader;
    private AlertDialog historyDialog;
    private LinearLayout historyListContent;
    private final java.util.LinkedHashSet<String> selectedRecordingPaths = new java.util.LinkedHashSet<>();
    private Button historySelectAllButton;
    private Button historyDeleteSelectedButton;
    private AlertDialog modelDialog;
    private LinearLayout modelListContent;
    private AlertDialog reconnectDialog;
    private TextView reconnectMessageView;
    private File pendingExportFile;

    private long frames;
    private long samples;
    private long missingFrames;
    private long previousSequence = -1;
    private long meterStartMs;
    private long meterStartSamples;
    private long lastUiMs;
    private long lastSampleIndex;
    private int[] lastValues = new int[SemgPacketParser.CHANNELS];

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        installCrashLogger();
        getWindow().setStatusBarColor(0xFF091116);
        getWindow().setNavigationBarColor(0xFF091116);
        
        // 加载保存的 Token
        android.content.SharedPreferences prefs = getSharedPreferences("semg_prefs", MODE_PRIVATE);
        authToken = prefs.getString("auth_token", null);
        currentUsername = prefs.getString("current_username", null);
        currentUserId = prefs.getInt("current_user_id", 0);
        isAdmin = prefs.getBoolean("is_admin", false);
        
        store = new RecordingStore(this);
        // 自动迁移旧版内部存储数据到外部存储
        int migrated = store.migrateFromInternalStorage(this);
        if (migrated > 0) {
            main.post(() -> toast("已迁移 " + migrated + " 个文件到外部存储"));
        }
        // 根据 SharedPreferences 中的激活标记，决定加载质心模型还是 PyTorch 模型
        if ("pytorch".equals(store.activeEngineType())) {
            loadActivePytorchModelAsync();
        } else {
            model = store.loadModel();
        }
        BleEventLog sharedLog = new BleEventLog(this);
        ble = new BleArmbandClient(this, this, sharedLog);
        classic = new ClassicArmbandClient(this, this, sharedLog);
        g20 = new G20HttpController(this, (enabled, message, warning) -> main.post(() -> {
            g20Button.setEnabled(true);
            g20Button.setText(enabled ? "停用控制" : "启用控制");
            g20Button.setBackground(rounded(enabled ? 0xFFFF6F85 : ACCENT, 10, 0));
            g20StatusView.setText(message);
            g20StatusView.setTextColor(warning ? 0xFFFF8395 : (enabled ? ACCENT : MUTED));
        }));
        setContentView(buildUi());
        // 优先使用登录用户名，否则使用上次输入的用户名
        String defaultUser = currentUsername != null ? currentUsername : 
                getSharedPreferences("semg_prefs", MODE_PRIVATE).getString("last_user", "");
        userInput.setText(defaultUser);
        if (currentUsername != null) {
            userInput.setEnabled(false);
            userInput.setAlpha(0.55f);
            userInput.setHint("已登录用户");
        }
        updateRawCardSubtitle();
        loadAlgorithmPreference();
        refreshDatasetAsync();
        updateModelText();
        String localIp = getLocalIpAddress();
        localIpView.setText("本机IP: " + localIp + "（请确保手机与小车在同一网络）");
    }

    /** 全局未捕获异常：先写入 BLE 日志文件（含堆栈），再交回系统默认处理。 */
    private void installCrashLogger() {
        final Thread.UncaughtExceptionHandler systemHandler = Thread.getDefaultUncaughtExceptionHandler();
        final BleEventLog crashLog = new BleEventLog(this);
        Thread.setDefaultUncaughtExceptionHandler((thread, error) -> {
            try {
                crashLog.logError("崩溃", "未捕获异常（线程 " + thread.getName() + "）", error);
            } catch (Throwable ignored) { }
            if (systemHandler != null) systemHandler.uncaughtException(thread, error);
        });
    }

    /** 在 worker 线程加载激活的 PyTorch 模型（Module.load 阻塞）。 */
    private void loadActivePytorchModelAsync() {
        worker.execute(() -> {
            try {
                RecordingStore.PyTorchModelInfo info = store.activePytorchModel();
                if (info == null) {
                    main.post(() -> modelView.setText("PyTorch 模型未找到，已回退到质心模型"));
                    model = store.loadModel();
                    return;
                }
                PyTorchModel loaded = PyTorchModel.load(info.ptFile, info.labelMapping,
                        info.protocol, info.name, info.createdAt, info.normMeans, info.normStds,
                        info.filterB, info.filterA, info.filterZi);
                pytorchModel = loaded;
                main.post(this::updateModelText);
            } catch (Exception error) {
                main.post(() -> {
                    modelView.setText("PyTorch 模型加载失败：" + error.getMessage());
                    // 回退到质心模型
                    model = store.loadModel();
                    updateModelText();
                });
            }
        });
    }

    private View buildUi() {
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        scroll.setBackgroundColor(0xFF091116);
        LinearLayout root = column();
        root.setPadding(dp(16), dp(16), dp(16), dp(28));
        scroll.addView(root);

        TextView eyebrow = text("ZHIWEIGUANG · sEMG · 8 CHANNEL", 11, ACCENT);
        eyebrow.setTypeface(Typeface.DEFAULT_BOLD);
        root.addView(eyebrow);
        LinearLayout titleRow = new LinearLayout(this);
        titleRow.setGravity(Gravity.CENTER_VERTICAL);
        TextView title = text("智为光电sEMG测试", 25, INK);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        titleRow.addView(title, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        
        // 用户状态按钮
        userBtn = button(authToken != null ? currentUsername : "登录", false);
        userBtn.setTextColor(authToken != null ? ACCENT : MUTED);
        userBtn.setOnClickListener(view -> {
            if (authToken != null) {
                // 已登录，显示菜单
                String[] menuItems = isAdmin 
                    ? new String[]{"管理员功能", "退出登录"}
                    : new String[]{"升级为管理员", "退出登录"};
                new AlertDialog.Builder(MainActivity.this)
                    .setTitle("用户: " + currentUsername + (isAdmin ? " [管理员]" : ""))
                    .setItems(menuItems, (d, which) -> {
                        if (isAdmin && which == 0) {
                            showAdminPanel();
                        } else if (!isAdmin && which == 0) {
                            showUpgradeToAdminDialog();
                        } else {
                            logout();
                        }
                    })
                    .setNegativeButton("关闭", null)
                    .show();
            } else {
                showLoginDialog(null);
            }
        });
        titleRow.addView(userBtn, new LinearLayout.LayoutParams(dp(88), dp(42)));
        
        Button about = button("关于", false);
        about.setOnClickListener(view -> showAbout());
        titleRow.addView(about, new LinearLayout.LayoutParams(dp(88), dp(42)));
        root.addView(titleRow, matchWrap(dp(2), dp(14)));

        LinearLayout connection = card();
        LinearLayout statusRow = new LinearLayout(this);
        statusRow.setGravity(Gravity.CENTER_VERTICAL);
        statusView = text("尚未连接手环", 14, INK);
        statusView.setSingleLine(false);
        statusRow.addView(statusView, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        connectButton = button("扫描并连接", true);
        connectButton.setOnClickListener(view -> {
            if (connected) { disconnectActiveLink(); onStatus("连接已断开", false); }
            else beginScan();
        });
        statusRow.addView(connectButton, new LinearLayout.LayoutParams(dp(124), dp(44)));
        connection.addView(statusRow);

        LinearLayout linkRow = new LinearLayout(this);
        linkRow.setGravity(Gravity.CENTER_VERTICAL);
        TextView linkLabel = text("连接方式:", 12, INK);
        linkLabel.setPadding(0, 0, dp(8), 0);
        linkRow.addView(linkLabel);
        linkModeSpinner = new Spinner(this);
        ArrayAdapter<String> linkAdapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item,
                new String[]{"蓝牙BLE（低功耗）", "经典蓝牙（SPP 串口）"});
        linkAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        linkModeSpinner.setAdapter(linkAdapter);
        linkModeSpinner.setBackground(rounded(0xFF0B151B, 8, 0xFF2A3A43));
        linkModeSpinner.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(android.widget.AdapterView<?> parent, View view, int position, long id) {
                boolean newMode = position == 1;
                if (newMode != classicMode) {
                    classicMode = newMode;
                    getSharedPreferences("semg_prefs", MODE_PRIVATE).edit()
                            .putBoolean("classic_mode", classicMode).apply();
                    if (connected) { disconnectActiveLink(); onStatus("连接方式已切换，请重新连接", false); }
                }
            }
            @Override public void onNothingSelected(android.widget.AdapterView<?> parent) { }
        });
        linkModeSpinner.setSelection(getSharedPreferences("semg_prefs", MODE_PRIVATE)
                .getBoolean("classic_mode", false) ? 1 : 0);
        classicMode = linkModeSpinner.getSelectedItemPosition() == 1;
        linkRow.addView(linkModeSpinner, new LinearLayout.LayoutParams(0, dp(40), 1));
        connection.addView(linkRow, matchWrap(dp(10), 0));

        LinearLayout logRow = new LinearLayout(this);
        logRow.setGravity(Gravity.CENTER_VERTICAL);
        TextView logHint = text("连接诊断:", 12, MUTED);
        logHint.setPadding(0, 0, dp(8), 0);
        logRow.addView(logHint);
        Button bleLog = button("查看 BLE 日志", false);
        bleLog.setOnClickListener(view -> showBleLog());
        logRow.addView(bleLog, new LinearLayout.LayoutParams(0, dp(38), 1));
        connection.addView(logRow, matchWrap(dp(10), 0));
        root.addView(connection, matchWrap(0, dp(12)));

        LinearLayout signalCard = card();
        signalCard.addView(sectionHeader("实时波形", "最近 2 秒 · 自动量程"));
        waveform = new WaveformView(this);
        signalCard.addView(waveform, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(320)));
        statsView = text("0 sample/s   ·   0 帧   ·   CRC 0", 12, MUTED);
        signalCard.addView(statsView, matchWrap(dp(10), dp(4)));
        root.addView(signalCard, matchWrap(0, dp(12)));

        LinearLayout rawCard = card();
        rawCardHeader = sectionHeader("原始数据", "ADC 12-bit · 0–4095");
        rawCard.addView(rawCardHeader);
        rawView = text("等待有效数据帧…", 12, INK);
        rawView.setTypeface(Typeface.MONOSPACE);
        rawView.setTextIsSelectable(true);
        rawCard.addView(rawView, matchWrap(dp(8), dp(4)));
        root.addView(rawCard, matchWrap(0, dp(12)));

        LinearLayout collectCard = card();
        collectCard.addView(sectionHeader("动作数据采集", "每种动作建议采集 3 次，每次 3–5 秒"));
        LinearLayout labelUserRow = new LinearLayout(this);
        labelUserRow.setOrientation(LinearLayout.HORIZONTAL);
        userInput = new EditText(this);
        userInput.setHint("用户名，例如 张三");
        userInput.setHintTextColor(0xFF657781);
        userInput.setTextColor(INK);
        userInput.setTextSize(14);
        userInput.setSingleLine(true);
        userInput.setInputType(InputType.TYPE_CLASS_TEXT);
        userInput.setBackground(rounded(0xFF0B151B, 10, 0xFF2A3A43));
        userInput.setPadding(dp(12), 0, dp(12), 0);
        LinearLayout.LayoutParams userParams = new LinearLayout.LayoutParams(0, dp(48), 1);
        userParams.rightMargin = dp(5);
        labelUserRow.addView(userInput, userParams);
        // 标签下拉选择器
        labelSpinner = new Spinner(this);
        String[] presetLabels = {"休息", "握拳", "手掌", "向左", "向右", "自定义"};
        ArrayAdapter<String> labelAdapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, presetLabels);
        labelAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        labelSpinner.setAdapter(labelAdapter);
        labelSpinner.setBackground(rounded(0xFF0B151B, 10, 0xFF2A3A43));
        labelSpinner.setPadding(dp(12), 0, dp(12), 0);
        LinearLayout.LayoutParams spinnerParams = new LinearLayout.LayoutParams(0, dp(48), 1.2f);
        spinnerParams.leftMargin = dp(5);
        labelUserRow.addView(labelSpinner, spinnerParams);
        collectCard.addView(labelUserRow, matchWrap(dp(8), dp(10)));
        // 自定义标签输入框（默认隐藏）
        customLabelInput = new EditText(this);
        customLabelInput.setHint("请输入自定义标签名称");
        customLabelInput.setHintTextColor(0xFF657781);
        customLabelInput.setTextColor(INK);
        customLabelInput.setTextSize(14);
        customLabelInput.setSingleLine(true);
        customLabelInput.setInputType(InputType.TYPE_CLASS_TEXT);
        customLabelInput.setBackground(rounded(0xFF0B151B, 10, 0xFF2A3A43));
        customLabelInput.setPadding(dp(12), 0, dp(12), 0);
        customLabelInput.setVisibility(View.GONE);
        LinearLayout.LayoutParams customParams = new LinearLayout.LayoutParams(0, dp(48), 1.4f);
        customParams.leftMargin = dp(5);
        customLabelInput.setLayoutParams(customParams);
        collectCard.addView(customLabelInput, matchWrap(dp(8), dp(10)));
        // 采集手别选择：向左/向右在不同手语义相反，每次采集都记录
        LinearLayout handRow = new LinearLayout(this);
        handRow.setOrientation(LinearLayout.HORIZONTAL);
        handRow.setGravity(Gravity.CENTER_VERTICAL);
        TextView handLabel = text("采集手别:", 13, INK);
        handRow.addView(handLabel);
        handSpinner = new Spinner(this);
        ArrayAdapter<String> handAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, new String[]{"右手", "左手"});
        handAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        handSpinner.setAdapter(handAdapter);
        handSpinner.setBackground(rounded(0xFF0B151B, 10, 0xFF2A3A43));
        handSpinner.setPadding(dp(12), 0, dp(12), 0);
        // 默认沿用注册时填写的手别
        String registeredHand = getSharedPreferences("semg_prefs", MODE_PRIVATE).getString("hand", "right");
        handSpinner.setSelection("left".equals(registeredHand) ? 1 : 0);
        LinearLayout.LayoutParams handParams = new LinearLayout.LayoutParams(0, dp(48), 1f);
        handParams.leftMargin = dp(8);
        handRow.addView(handSpinner, handParams);
        collectCard.addView(handRow, matchWrap(dp(8), dp(10)));
        // 监听下拉选择变化
        labelSpinner.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(android.widget.AdapterView<?> parent, View view, int position, long id) {
                if (position == presetLabels.length - 1) {
                    customLabelInput.setVisibility(View.VISIBLE);
                } else {
                    customLabelInput.setVisibility(View.GONE);
                }
            }
            @Override public void onNothingSelected(android.widget.AdapterView<?> parent) {}
        });
        recordButton = button("开始采集", true);
        recordButton.setOnClickListener(view -> toggleRecording());
        collectCard.addView(recordButton, matchHeight(dp(46), 0, dp(10)));
        // "记录数据"：不选动作标签，直接录制原始 EMG，保存后可在"历史数据管理"中回看
        recordRawButton = button("记录数据", false);
        recordRawButton.setTextColor(ACCENT);
        recordRawButton.setOnClickListener(view -> toggleRawRecording());
        collectCard.addView(recordRawButton, matchHeight(dp(46), 0, dp(10)));
        datasetView = text("正在读取本地数据集…", 12, MUTED);
        collectCard.addView(datasetView);
        Button history = button("历史数据管理", false);
        history.setOnClickListener(view -> showHistoryManager());
        collectCard.addView(history, matchHeight(dp(44), dp(10), 0));
        root.addView(collectCard, matchWrap(0, dp(12)));

        LinearLayout trainCard = card();
        trainCard.addView(sectionHeader("训练与识别", "训练和推理均在本机完成，不上传数据"));
        LinearLayout algorithmRow = new LinearLayout(this);
        algorithmRow.setGravity(Gravity.CENTER_VERTICAL);
        TextView algorithmLabel = text("识别算法:", 12, INK);
        algorithmLabel.setPadding(0, 0, dp(8), 0);
        algorithmRow.addView(algorithmLabel);
        algorithmSpinner = new Spinner(this);
        ArrayAdapter<String> algorithmAdapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item,
                new String[]{GestureModel.Algorithm.CENTROID.display,
                        GestureModel.Algorithm.LINEAR_SVM.display,
                        GestureModel.Algorithm.RBF_SVM.display});
        algorithmAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        algorithmSpinner.setAdapter(algorithmAdapter);
        algorithmSpinner.setBackground(rounded(0xFF0B151B, 8, 0xFF2A3A43));
        algorithmSpinner.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(android.widget.AdapterView<?> parent, View view, int position, long id) {
                GestureModel.Algorithm selected = position == 1
                        ? GestureModel.Algorithm.LINEAR_SVM
                        : (position == 2 ? GestureModel.Algorithm.RBF_SVM : GestureModel.Algorithm.CENTROID);
                getSharedPreferences("semg_prefs", MODE_PRIVATE).edit()
                        .putString("algorithm", selected.name()).apply();
            }
            @Override public void onNothingSelected(android.widget.AdapterView<?> parent) { }
        });
        algorithmRow.addView(algorithmSpinner, new LinearLayout.LayoutParams(0, dp(40), 1));
        trainCard.addView(algorithmRow, matchWrap(dp(4), dp(8)));
        LinearLayout modelActions = new LinearLayout(this);
        modelActions.setOrientation(LinearLayout.HORIZONTAL);
        Button train = button("训练新模型", false);
        train.setOnClickListener(view -> trainModel(train));
        LinearLayout.LayoutParams trainParams = new LinearLayout.LayoutParams(0, dp(46), 1);
        trainParams.rightMargin = dp(5);
        modelActions.addView(train, trainParams);
        Button manageModels = button("模型管理", false);
        manageModels.setOnClickListener(view -> showModelManager());
        LinearLayout.LayoutParams manageParams = new LinearLayout.LayoutParams(0, dp(46), 1);
        manageParams.leftMargin = dp(5);
        modelActions.addView(manageModels, manageParams);
        trainCard.addView(modelActions, matchWrap(dp(8), dp(8)));
        Button importPytorch = button("导入 PyTorch", false);
        importPytorch.setOnClickListener(view -> beginImportPytorchModel());
        trainCard.addView(importPytorch, matchHeight(dp(44), dp(4), dp(8)));
        Button cloudUpload = button("云端上传", false);
        cloudUpload.setTextColor(0xFF4BC6FF);
        cloudUpload.setOnClickListener(view -> showCloudUploadDialog());
        trainCard.addView(cloudUpload, matchHeight(dp(44), dp(4), dp(8)));
        Button cloudTrain = button("云端训练", false);
        cloudTrain.setTextColor(0xFF42E8B4);
        cloudTrain.setOnClickListener(view -> showCloudTrainDialog());
        trainCard.addView(cloudTrain, matchHeight(dp(44), dp(4), dp(8)));
        Button downloadMyModel = button("下载我的模型", false);
        downloadMyModel.setTextColor(0xFFFFB74D);
        downloadMyModel.setOnClickListener(view -> showDownloadMyModelDialog());
        trainCard.addView(downloadMyModel, matchHeight(dp(44), dp(4), dp(8)));
        modelView = text("尚未训练模型", 12, MUTED);
        trainCard.addView(modelView, matchWrap(0, dp(10)));
        resultView = text("—", 32, INK);
        resultView.setTypeface(Typeface.DEFAULT_BOLD);
        resultView.setGravity(Gravity.CENTER);
        resultView.setBackground(rounded(0xFF0B151B, 12, 0xFF23343D));
        trainCard.addView(resultView, matchHeight(dp(86), 0, dp(10)));
        recognizeButton = button("开始实时识别", true);
        recognizeButton.setOnClickListener(view -> toggleRecognition());
        trainCard.addView(recognizeButton, matchHeight(dp(46), 0, 0));
        root.addView(trainCard, matchWrap(0, dp(12)));

        LinearLayout g20Card = card();
        g20Card.addView(sectionHeader("WiFi 小车控制", "UDP · 支持机械车/智能车/Go2/Go2轮式/侦察球"));

        localIpView = text("本机IP: 获取中...", 12, MUTED);
        g20Card.addView(localIpView, matchWrap(0, dp(6)));

        LinearLayout ipRow = new LinearLayout(this);
        ipRow.setOrientation(LinearLayout.HORIZONTAL);
        TextView ipLabel = text("目标IP:", 12, INK);
        ipLabel.setPadding(0, 0, dp(8), 0);
        ipRow.addView(ipLabel);
        ipInput = new EditText(this);
        ipInput.setText(g20.getTargetIp());
        ipInput.setHint("192.168.x.x");
        ipInput.setHintTextColor(0xFF657781);
        ipInput.setTextColor(INK);
        ipInput.setTextSize(13);
        ipInput.setSingleLine(true);
        ipInput.setInputType(InputType.TYPE_CLASS_TEXT);
        ipInput.setBackground(rounded(0xFF0B151B, 8, 0xFF2A3A43));
        ipInput.setPadding(dp(10), 0, dp(10), 0);
        ipRow.addView(ipInput, new LinearLayout.LayoutParams(0, dp(40), 2));

        TextView portLabel = text(" 端口:", 12, INK);
        portLabel.setPadding(dp(8), 0, dp(8), 0);
        ipRow.addView(portLabel);
        portInput = new EditText(this);
        portInput.setText(String.valueOf(g20.getTargetPort()));
        portInput.setHint("8888");
        portInput.setHintTextColor(0xFF657781);
        portInput.setTextColor(INK);
        portInput.setTextSize(13);
        portInput.setSingleLine(true);
        portInput.setInputType(InputType.TYPE_CLASS_NUMBER);
        portInput.setBackground(rounded(0xFF0B151B, 8, 0xFF2A3A43));
        portInput.setPadding(dp(10), 0, dp(10), 0);
        ipRow.addView(portInput, new LinearLayout.LayoutParams(0, dp(40), 1));
        g20Card.addView(ipRow, matchWrap(0, dp(8)));

        TextView typeLabel = text("设备类型:", 12, INK);
        g20Card.addView(typeLabel, matchWrap(0, dp(4)));
        deviceTypeSpinner = new Spinner(this);
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item,
                new String[]{G20HttpController.DeviceType.MECHANICAL_CAR.display,
                        G20HttpController.DeviceType.SMART_CAR.display,
                        G20HttpController.DeviceType.GO2.display,
                        G20HttpController.DeviceType.RECON_BALL.display,
                        G20HttpController.DeviceType.GO2W.display});
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        deviceTypeSpinner.setAdapter(adapter);
        deviceTypeSpinner.setBackground(rounded(0xFF0B151B, 8, 0xFF2A3A43));
        if (g20.getDeviceType() == G20HttpController.DeviceType.SMART_CAR) {
            deviceTypeSpinner.setSelection(1);
        } else if (g20.getDeviceType() == G20HttpController.DeviceType.GO2) {
            deviceTypeSpinner.setSelection(2);
        } else if (g20.getDeviceType() == G20HttpController.DeviceType.RECON_BALL) {
            deviceTypeSpinner.setSelection(3);
        } else if (g20.getDeviceType() == G20HttpController.DeviceType.GO2W) {
            deviceTypeSpinner.setSelection(4);
        }
        g20Card.addView(deviceTypeSpinner, matchWrap(0, dp(8)));

        LinearLayout timeoutRow = new LinearLayout(this);
        timeoutRow.setOrientation(LinearLayout.HORIZONTAL);
        timeoutRow.setGravity(Gravity.CENTER_VERTICAL);
        TextView timeoutLabel = text("控制超时(s):", 12, INK);
        timeoutLabel.setPadding(0, 0, dp(8), 0);
        timeoutRow.addView(timeoutLabel);
        timeoutInput = new EditText(this);
        timeoutInput.setText(String.valueOf(g20.getControlTimeoutMs() / 1000));
        timeoutInput.setHint("30");
        timeoutInput.setHintTextColor(0xFF657781);
        timeoutInput.setTextColor(INK);
        timeoutInput.setTextSize(13);
        timeoutInput.setSingleLine(true);
        timeoutInput.setInputType(InputType.TYPE_CLASS_NUMBER);
        timeoutInput.setBackground(rounded(0xFF0B151B, 8, 0xFF2A3A43));
        timeoutInput.setPadding(dp(10), 0, dp(10), 0);
        timeoutRow.addView(timeoutInput, new LinearLayout.LayoutParams(0, dp(40), 1));
        g20Card.addView(timeoutRow, matchWrap(0, dp(8)));

        Button mappingButton = button("配置动作映射", false);
        mappingButton.setOnClickListener(view -> showMappingConfig());
        g20Card.addView(mappingButton, matchHeight(dp(44), dp(8), dp(8)));

        g20StatusView = text("控制未启用", 12, MUTED);
        g20Card.addView(g20StatusView, matchWrap(0, dp(10)));
        g20Button = button("启用控制", true);
        g20Button.setOnClickListener(view -> toggleG20Control());
        g20Card.addView(g20Button, matchHeight(dp(46), 0, dp(8)));
        g20Card.addView(text("提示：请先配置动作映射。识别不确定、数据中断或退出应用时自动停止。", 11, MUTED));

        // 手动控制面板
        LinearLayout manualCard = card();
        manualCard.addView(sectionHeader("手动控制", "测试连接 · 直接发送控制指令"));

        LinearLayout manualButtons = new LinearLayout(this);
        manualButtons.setOrientation(LinearLayout.VERTICAL);
        manualButtons.setPadding(0, dp(8), 0, 0);

        // 第一行：前进
        Button btnForward = button("⬆ 前进", true);
        btnForward.setOnClickListener(view -> sendManualCommand(G20HttpController.Command.FORWARD));
        manualButtons.addView(btnForward, matchHeight(dp(48), 0, dp(6)));

        // 第二行：左转 + 停止 + 右转
        LinearLayout middleRow = new LinearLayout(this);
        middleRow.setOrientation(LinearLayout.HORIZONTAL);
        Button btnLeft = button("⬅ 左转", false);
        btnLeft.setOnClickListener(view -> sendManualCommand(G20HttpController.Command.LEFT));
        middleRow.addView(btnLeft, weightedButtonParams(0, 4));
        Button btnStop = button("⏹ 停止", false);
        btnStop.setTextColor(0xFFFF6F85);
        btnStop.setOnClickListener(view -> sendManualCommand(G20HttpController.Command.STOP));
        middleRow.addView(btnStop, weightedButtonParams(4, 4));
        Button btnRight = button(" 右转", false);
        btnRight.setOnClickListener(view -> sendManualCommand(G20HttpController.Command.RIGHT));
        middleRow.addView(btnRight, weightedButtonParams(4, 0));
        manualButtons.addView(middleRow, matchWrap(0, dp(6)));

        // 第三行：后退
        Button btnBackward = button("⬇ 后退", true);
        btnBackward.setOnClickListener(view -> sendManualCommand(G20HttpController.Command.BACKWARD));
        manualButtons.addView(btnBackward, matchHeight(dp(48), 0, 0));

        manualCard.addView(manualButtons);

        // 机械车额外按钮：爪子张开/并拢
        Button btnGripperOpen = button("🤏 爪子张开", false);
        btnGripperOpen.setOnClickListener(view -> sendManualCommand(G20HttpController.Command.GRIPPER_OPEN));
        manualCard.addView(btnGripperOpen, matchHeight(dp(44), dp(8), dp(4)));
        Button btnGripperClose = button(" 爪子并拢", false);
        btnGripperClose.setOnClickListener(view -> sendManualCommand(G20HttpController.Command.GRIPPER_CLOSE));
        manualCard.addView(btnGripperClose, matchHeight(dp(44), 0, 0));

        manualCard.addView(text("提示：手动控制不需要启用自动识别，可直接测试连接。", 11, MUTED), matchWrap(dp(10), 0));
        root.addView(manualCard, matchWrap(0, dp(12)));

        root.addView(g20Card);
        return scroll;
    }

    private void beginScan() {
        if (!hasBluetoothPermissions()) {
            if (Build.VERSION.SDK_INT >= 31) requestPermissions(new String[]{
                    Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT
            }, REQUEST_BLUETOOTH);
            else requestPermissions(new String[]{Manifest.permission.ACCESS_FINE_LOCATION}, REQUEST_BLUETOOTH);
            return;
        }
        if (classicMode) {
            if (!classic.isBluetoothEnabled()) {
                try { startActivity(new Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE)); }
                catch (Exception ignored) { toast("请在系统设置中开启蓝牙"); }
                return;
            }
            classic.scan();
            return;
        }
        if (!ble.isBluetoothEnabled()) {
            try { startActivity(new Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE)); }
            catch (Exception ignored) { toast("请在系统设置中开启蓝牙"); }
            return;
        }
        ble.scan();
    }

    /** 断开当前使用的连接方式（BLE 或经典蓝牙）。 */
    private void disconnectActiveLink() {
        if (classicMode) classic.disconnect();
        ble.disconnect();
    }

    private boolean hasBluetoothPermissions() {
        if (Build.VERSION.SDK_INT >= 31) return checkSelfPermission(Manifest.permission.BLUETOOTH_SCAN) == PackageManager.PERMISSION_GRANTED
                && checkSelfPermission(Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED;
        return checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED;
    }

    @Override public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(requestCode, permissions, results);
        if (requestCode != REQUEST_BLUETOOTH) return;
        boolean granted = results.length > 0;
        for (int result : results) granted &= result == PackageManager.PERMISSION_GRANTED;
        if (granted) beginScan(); else toast("需要附近设备权限才能发现并连接手环");
    }

    @Override public void onStatus(String text, boolean isConnected) {
        main.post(() -> {
            boolean newlyConnected = isConnected && !connected;
            connected = isConnected;
            statusView.setText(text);
            statusView.setTextColor(isConnected ? ACCENT : INK);
            connectButton.setText(isConnected ? "断开" : "扫描并连接");
            if (isConnected) dismissReconnectDialog();
            if (!isConnected && g20 != null) g20.stopMotion("G20 控制已启用 · 蓝牙断开，已停止");
            if (newlyConnected) {
                parser.reset();
                frames = samples = missingFrames = 0;
                previousSequence = -1;
                meterStartMs = System.currentTimeMillis();
                meterStartSamples = 0;
                waveform.clear();
                validDataAnnounced = false;
            }
        });
    }

    /** 蓝牙意外断开：弹出浮窗提示并自动重连（由 BleArmbandClient 驱动）。 */
    @Override public void onReconnecting(String deviceName, int attempt) {
        main.post(() -> {
            if (isFinishing() || isDestroyed()) return;
            if (reconnectDialog == null || !reconnectDialog.isShowing()) {
                LinearLayout body = column();
                body.setPadding(dp(20), dp(10), dp(20), 0);
                reconnectMessageView = text("", 14, INK);
                body.addView(reconnectMessageView, matchWrap(0, dp(8)));
                reconnectDialog = new AlertDialog.Builder(MainActivity.this)
                        .setTitle("蓝牙已断开")
                        .setView(body)
                        .setCancelable(false)
                        .setPositiveButton("停止重连", (dialog, which) -> {
                            dismissReconnectDialog();
                            disconnectActiveLink();
                            toast("已停止自动重连");
                        })
                        .create();
                reconnectDialog.show();
            }
            if (reconnectMessageView != null) {
                reconnectMessageView.setText("“" + deviceName + "”连接已断开\n正在搜索设备并自动重连…（第 " + attempt + " 次尝试）\n设备重新开机后会自动恢复连接");
            }
        });
    }

    private void dismissReconnectDialog() {
        if (reconnectDialog != null && reconnectDialog.isShowing()) reconnectDialog.dismiss();
        reconnectDialog = null;
        reconnectMessageView = null;
    }

    @Override public void onScanFinished(List<BleArmbandClient.DeviceItem> devices) {
        main.post(() -> {
            if (devices.isEmpty()) {
                statusView.setText("未发现肌电手环，请检查手环状态");
                new AlertDialog.Builder(this).setTitle("未发现可连接设备")
                        .setMessage("确认手环已开机、工作在 BLE 从机模式，且没有连接到其他手机或电脑。")
                        .setPositiveButton("重新扫描", (dialog, which) -> beginScan())
                        .setNegativeButton("取消", null).show();
                return;
            }
            statusView.setText("发现 " + devices.size() + " 台设备，请选择手环");
            devices.sort(Comparator.comparing((BleArmbandClient.DeviceItem item) ->
                    !item.name.toUpperCase(Locale.ROOT).contains("BT-11")).thenComparingInt(item -> -item.rssi));
            String[] labels = new String[devices.size()];
            for (int i = 0; i < devices.size(); i++) labels[i] = devices.get(i).toString();
            new AlertDialog.Builder(this).setTitle("选择肌电手环")
                    .setItems(labels, (dialog, which) -> ble.connect(devices.get(which)))
                    .setNegativeButton("取消", null).show();
        });
    }

    @Override public void onBytes(byte[] bytes) {
        List<SemgFrame> decoded = parser.feed(bytes);
        for (SemgFrame frame : decoded) processFrame(frame);
    }

    /** 经典蓝牙扫描完成：列出已配对及发现的设备，选择后走 SPP 连接。 */
    @Override public void onClassicScanFinished(List<ClassicArmbandClient.DeviceItem> devices) {
        main.post(() -> {
            if (devices.isEmpty()) {
                statusView.setText("未发现经典蓝牙设备");
                new AlertDialog.Builder(this).setTitle("未发现设备")
                        .setMessage("经典蓝牙设备需要先在系统蓝牙设置中配对。请先配对臂环，或在臂环开机状态下重新扫描。")
                        .setPositiveButton("重新扫描", (dialog, which) -> beginScan())
                        .setNegativeButton("取消", null).show();
                return;
            }
            statusView.setText("发现 " + devices.size() + " 台设备，请选择臂环");
            devices.sort(Comparator.comparing((ClassicArmbandClient.DeviceItem item) -> !item.bonded));
            String[] labels = new String[devices.size()];
            for (int i = 0; i < devices.size(); i++) labels[i] = devices.get(i).toString();
            new AlertDialog.Builder(this).setTitle("选择经典蓝牙臂环")
                    .setItems(labels, (dialog, which) -> classic.connect(devices.get(which)))
                    .setNegativeButton("取消", null).show();
        });
    }

    private void processFrame(SemgFrame frame) {
        if (!validDataAnnounced) {
            validDataAnnounced = true;
            main.post(() -> statusView.setText("已连接 · 正在接收有效肌电数据"));
        }
        frames++;
        if (previousSequence >= 0 && frame.sequence > previousSequence + 1) {
            missingFrames += frame.sequence - previousSequence - 1;
        }
        previousSequence = frame.sequence;
        samples += frame.samples.length;
        lastSampleIndex = frame.firstSampleIndex + frame.samples.length - 1;
        lastValues = frame.samples[frame.samples.length - 1].clone();
        waveform.addSamples(frame.samples);
        RecordingStore.Session active = recording;
        if (active != null) {
            try { active.append(frame.firstSampleIndex, frame.samples); }
            catch (RuntimeException error) { main.post(() -> stopRecordingWithError(error.getMessage())); }
        }
        processInference(frame.samples);

        long now = System.currentTimeMillis();
        if (now - lastUiMs > 250) {
            lastUiMs = now;
            main.post(this::updateSignalUi);
        }
    }

    private void updateSignalUi() {
        long now = System.currentTimeMillis();
        double rate = (samples - meterStartSamples) * 1000.0 / Math.max(1, now - meterStartMs);
        if (now - meterStartMs > 4000) { meterStartMs = now; meterStartSamples = samples; }
        statsView.setText(String.format(Locale.CHINA,
                "%.0f sample/s   ·   %d 帧   ·   丢帧 %d   ·   CRC %d",
                rate, frames, missingFrames, parser.crcErrors));
        StringBuilder raw = new StringBuilder();
        raw.append("sample_index ").append(lastSampleIndex).append("\n");
        for (int i = 0; i < lastValues.length; i++) {
            raw.append(String.format(Locale.CHINA, "CH%d %4d", i + 1, lastValues[i]));
            raw.append(i % 4 == 3 ? '\n' : "     ");
        }
        rawView.setText(raw.toString().trim());
    }

    private String getSelectedLabel() {
        int pos = labelSpinner.getSelectedItemPosition();
        String[] presetLabels = {"休息", "握拳", "手掌", "向左", "向右", "自定义"};
        if (pos >= 0 && pos < presetLabels.length - 1) {
            return presetLabels[pos];
        }
        String custom = customLabelInput.getText().toString().trim();
        return custom.isEmpty() ? "" : custom;
    }

    private void toggleRecording() {
        if (recording != null) { finishRecording(); return; }
        String label = getSelectedLabel();
        if (label.isEmpty()) { toast("请先选择或输入动作标签"); return; }
        String user;
        if (authToken != null && currentUsername != null) {
            user = currentUsername;
        } else {
            user = userInput.getText().toString().trim();
            if (user.isEmpty()) { toast("请先输入用户名"); return; }
        }
        if (!connected || samples == 0) { toast("请先连接手环并确认已有数据"); return; }
        try {
            String hand = handSpinner.getSelectedItemPosition() == 1 ? "left" : "right";
            recording = store.start(label, user, parser.getProtocol(), hand);
            getSharedPreferences("semg_prefs", MODE_PRIVATE).edit().putString("last_user", user).apply();
            recordButton.setText("停止并保存");
            recordButton.setBackground(rounded(0xFFFF6F85, 10, 0));
            recordRawButton.setEnabled(false);
            labelSpinner.setEnabled(false);
            customLabelInput.setEnabled(false);
            handSpinner.setEnabled(false);
            userInput.setEnabled(false);
            datasetView.setText("正在采集用户“" + user + "”的“" + label + "”…");
        } catch (IOException error) { toast(error.getMessage()); }
    }

    /** "记录数据"：不选动作标签，直接录制原始 EMG；再次点击保存为 CSV，可在历史数据管理中回看。 */
    private void toggleRawRecording() {
        if (recording != null) { finishRecording(); return; }
        String user;
        if (authToken != null && currentUsername != null) {
            user = currentUsername;
        } else {
            user = userInput.getText().toString().trim();
            if (user.isEmpty()) { toast("请先输入用户名"); return; }
        }
        if (!connected || samples == 0) { toast("请先连接手环并确认已有数据"); return; }
        try {
            String hand = handSpinner.getSelectedItemPosition() == 1 ? "left" : "right";
            recording = store.start(RecordingStore.RAW_LABEL, user, parser.getProtocol(), hand);
            rawRecording = true;
            getSharedPreferences("semg_prefs", MODE_PRIVATE).edit().putString("last_user", user).apply();
            recordRawButton.setText("停止记录");
            recordRawButton.setTextColor(0xFF06140F);
            recordRawButton.setBackground(rounded(0xFFFF6F85, 10, 0));
            recordButton.setEnabled(false);
            labelSpinner.setEnabled(false);
            customLabelInput.setEnabled(false);
            handSpinner.setEnabled(false);
            userInput.setEnabled(false);
            datasetView.setText("正在记录原始数据…保存后可在“历史数据管理”中回看");
        } catch (IOException error) { toast(error.getMessage()); }
    }

    private void finishRecording() {
        RecordingStore.Session active = recording;
        boolean wasRaw = rawRecording;
        recording = null;
        rawRecording = false;
        long count = 0;
        try { if (active != null) count = active.close(); }
        catch (IOException error) { toast("保存失败：" + error.getMessage()); }
        recordButton.setText("开始采集");
        recordButton.setBackground(rounded(ACCENT, 10, 0));
        recordButton.setEnabled(true);
        recordRawButton.setText("记录数据");
        recordRawButton.setTextColor(ACCENT);
        recordRawButton.setBackground(rounded(0xFF1B2B34, 10, 0xFF344954));
        recordRawButton.setEnabled(true);
        labelSpinner.setEnabled(true);
        customLabelInput.setEnabled(true);
        handSpinner.setEnabled(true);
        userInput.setEnabled(true);
        toast(wasRaw
                ? String.format(Locale.CHINA, "原始数据已保存 %.1f 秒，可在“历史数据管理”中回看", count / 500.0)
                : String.format(Locale.CHINA, "已保存 %.1f 秒原始数据", count / 500.0));
        refreshDatasetAsync();
    }

    private void stopRecordingWithError(String message) {
        if (recording == null) return;
        finishRecording();
        toast(message);
    }

    private void refreshDatasetAsync() {
        worker.execute(() -> {
            List<RecordingStore.RecordingInfo> recordings = store.listRecordings();
            Map<String, Integer> labelCounts = new LinkedHashMap<>();
            Map<String, Integer> userCounts = new LinkedHashMap<>();
            long total = 0;
            for (RecordingStore.RecordingInfo info : recordings) {
                labelCounts.put(info.label, labelCounts.getOrDefault(info.label, 0) + 1);
                userCounts.put(info.user, userCounts.getOrDefault(info.user, 0) + 1);
                total += info.sampleCount;
            }
            final long samples = total;
            final boolean empty = recordings.isEmpty();
            main.post(() -> {
                if (empty) datasetView.setText("本地数据集为空");
                else {
                    List<String> parts = new ArrayList<>();
                    for (String label : labelCounts.keySet()) parts.add(label + " × " + labelCounts.get(label));
                    List<String> userParts = new ArrayList<>();
                    for (String user : userCounts.keySet()) userParts.add(user + "(" + userCounts.get(user) + ")");
                    datasetView.setText("用户：" + String.join("、", userParts) + "\n"
                            + String.join("   ·   ", parts) + String.format(Locale.CHINA,
                            "\n共 %.1f 秒 / %d 样本", samples / 500.0, samples));
                }
            });
        });
    }

    /** 过滤掉"记录数据"产生的原始记录：它们只用于本地回看，不参与训练与云端上传。 */
    private static List<RecordingStore.RecordingInfo> trainableRecordings(
            List<RecordingStore.RecordingInfo> all) {
        List<RecordingStore.RecordingInfo> result = new ArrayList<>(all.size());
        for (RecordingStore.RecordingInfo info : all) {
            if (!RecordingStore.RAW_LABEL.equals(info.label)) result.add(info);
        }
        return result;
    }

    private void trainModel(Button button) {
        if (recording != null) { toast("请先停止当前采集"); return; }
        worker.execute(() -> {
            // 收集各用户的记录数（按当前协议）
            Map<String, Integer> userCounts = new LinkedHashMap<>();
            for (RecordingStore.RecordingInfo info : trainableRecordings(store.listRecordings())) {
                if (info.protocol != parser.getProtocol()) continue;
                userCounts.put(info.user, userCounts.getOrDefault(info.user, 0) + 1);
            }
            main.post(() -> showTrainDialog(button, userCounts));
        });
    }

    private void showTrainDialog(Button button, Map<String, Integer> userCounts) {
        LinearLayout body = column();
        body.setPadding(dp(8), 0, dp(8), 0);
        TextView userLabel = text(userCounts.isEmpty() ? "暂无当前协议的录制数据" : "选择参与训练的用户（全部勾选＝使用全部数据）：", 13, INK);
        body.addView(userLabel, matchWrap(dp(4), dp(8)));
        List<CheckBox> userBoxes = new ArrayList<>();
        List<String> userNames = new ArrayList<>(userCounts.keySet());
        for (String user : userNames) {
            CheckBox box = new CheckBox(this);
            box.setText(user + "（" + userCounts.get(user) + " 条）");
            box.setTextColor(INK);
            box.setTextSize(14);
            box.setChecked(true);
            body.addView(box, matchWrap(0, dp(2)));
            userBoxes.add(box);
        }
        Button selectAllUsers = button("全部勾选 / 全部取消", false);
        body.addView(selectAllUsers, matchHeight(dp(40), dp(6), dp(8)));
        selectAllUsers.setOnClickListener(view -> {
            boolean anyUnchecked = false;
            for (CheckBox box : userBoxes) if (!box.isChecked()) anyUnchecked = true;
            for (CheckBox box : userBoxes) box.setChecked(anyUnchecked);
        });
        EditText nameInput = new EditText(this);
        nameInput.setSingleLine(true);
        nameInput.setText(new SimpleDateFormat("动作模型 yyyy-MM-dd HH:mm", Locale.CHINA).format(new Date()));
        nameInput.setSelectAllOnFocus(true);
        nameInput.setPadding(dp(8), 0, dp(8), 0);
        body.addView(nameInput, matchHeight(dp(46), dp(6), dp(4)));
        new AlertDialog.Builder(this)
                .setTitle("命名新模型")
                .setMessage("训练完成后会保留现有模型，并自动启用新模型。")
                .setView(body)
                .setPositiveButton("开始训练", (dialog, which) -> {
                    String name = nameInput.getText().toString().trim();
                    if (name.isEmpty()) name = "未命名模型";
                    java.util.Set<String> selectedUsers = new java.util.LinkedHashSet<>();
                    for (int i = 0; i < userBoxes.size(); i++) {
                        if (userBoxes.get(i).isChecked()) selectedUsers.add(userNames.get(i));
                    }
                    if (!userNames.isEmpty() && selectedUsers.isEmpty()) {
                        toast("请至少选择一个用户");
                        return;
                    }
                    runModelTraining(button, name, selectedUsers);
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private void runModelTraining(Button button, String modelName, java.util.Set<String> users) {
        button.setEnabled(false);
        button.setText("正在训练…");
        GestureModel.Algorithm algorithm = selectedAlgorithm();
        String scope = users.isEmpty() ? "全部用户" : (users.size() + " 位用户");
        modelView.setText("正在用" + scope + "的数据提取特征（" + algorithm.display + "）");
        worker.execute(() -> {
            try {
                GestureModel trained = store.train(modelName, parser.getProtocol(), algorithm,
                        users.isEmpty() ? null : users);
                model = trained;
                main.post(() -> { toast("模型训练完成"); updateModelText(); });
            } catch (Exception error) {
                main.post(() -> { modelView.setText("训练失败：" + error.getMessage()); toast(error.getMessage()); });
            } finally {
                main.post(() -> { button.setEnabled(true); button.setText("训练新模型"); });
            }
        });
    }

    private GestureModel.Algorithm selectedAlgorithm() {
        if (algorithmSpinner == null) return GestureModel.Algorithm.CENTROID;
        int position = algorithmSpinner.getSelectedItemPosition();
        if (position == 1) return GestureModel.Algorithm.LINEAR_SVM;
        if (position == 2) return GestureModel.Algorithm.RBF_SVM;
        return GestureModel.Algorithm.CENTROID;
    }

    private void loadAlgorithmPreference() {
        String saved = getSharedPreferences("semg_prefs", MODE_PRIVATE).getString("algorithm", GestureModel.Algorithm.CENTROID.name());
        GestureModel.Algorithm algorithm = GestureModel.Algorithm.fromName(saved);
        algorithmSpinner.setSelection(algorithm == GestureModel.Algorithm.LINEAR_SVM ? 1
                : (algorithm == GestureModel.Algorithm.RBF_SVM ? 2 : 0));
    }

    private void updateModelText() {
        if (pytorchModel != null) {
            boolean protocolMismatch = pytorchModel.protocol != parser.getProtocol();
            String protocolTag = protocolMismatch
                    ? " ️协议不匹配(" + pytorchModel.protocol.name() + ")"
                    : " (" + pytorchModel.protocol.name() + ")";
            StringBuilder labels = new StringBuilder();
            for (Map.Entry<Integer, String> entry : pytorchModel.labelMapping.entrySet()) {
                if (labels.length() > 0) labels.append("  /  ");
                labels.append(entry.getKey()).append(":").append(entry.getValue());
            }
            modelView.setText("当前模型（PyTorch CNN）：" + pytorchModel.modelName + protocolTag + "\n"
                    + pytorchModel.labelMapping.size() + " 类 · 输入 8×100 → 4 类\n" + labels);
            modelView.setTextColor(protocolMismatch ? 0xFFFF8395 : ACCENT);
            return;
        }
        if (model == null) {
            modelView.setText("尚未训练模型");
            modelView.setTextColor(MUTED);
            return;
        }
        boolean protocolMismatch = model.protocol != parser.getProtocol();
        String protocolTag = protocolMismatch ? " ️协议不匹配(" + model.protocol.name() + ")" : " (" + model.protocol.name() + ")";
        modelView.setText("当前模型：" + store.activeModelName() + protocolTag + "\n"
                + model.algorithm.display + " · " + model.labelCount() + " 类动作 · "
                + model.trainingWindows + " 个窗口\n" + String.join("  /  ", model.labels()));
        if (protocolMismatch) {
            modelView.setTextColor(0xFFFF8395);
        } else {
            modelView.setTextColor(MUTED);
        }
    }

    private void showHistoryManager() {
        if (recording != null) { toast("请先停止当前采集"); return; }
        refreshHistoryManager();
    }

    private void refreshHistoryManager() {
        worker.execute(() -> {
            List<RecordingStore.RecordingInfo> recordings = store.listRecordings();
            main.post(() -> presentHistoryManager(recordings));
        });
    }

    private void presentHistoryManager(List<RecordingStore.RecordingInfo> recordings) {
        boolean reopen = historyDialog == null || !historyDialog.isShowing();
        if (reopen) {
            if (modelDialog != null && modelDialog.isShowing()) modelDialog.dismiss();
            ScrollView scroll = new ScrollView(this);
            historyListContent = column();
            historyListContent.setPadding(dp(12), dp(6), dp(12), dp(12));
            scroll.addView(historyListContent);
            historyDialog = new AlertDialog.Builder(this)
                    .setTitle("历史数据管理")
                    .setView(scroll)
                    .setNegativeButton("关闭", null)
                    .create();
            historyDialog.setOnDismissListener(dialog -> {
                if (historyDialog == dialog) {
                    historyDialog = null;
                    historyListContent = null;
                    historySelectAllButton = null;
                    historyDeleteSelectedButton = null;
                    selectedRecordingPaths.clear();
                }
            });
            historyDialog.show();
        }
        populateHistoryManager(recordings);
    }

    private void populateHistoryManager(List<RecordingStore.RecordingInfo> recordings) {
        if (historyListContent == null) return;
        historyListContent.removeAllViews();
        if (recordings.isEmpty()) {
            selectedRecordingPaths.clear();
            TextView empty = text("暂无历史数据。连接手环并完成一次动作采集后，记录会显示在这里。", 14, MUTED);
            empty.setPadding(dp(8), dp(20), dp(8), dp(20));
            historyListContent.addView(empty);
            return;
        }
        long totalSamples = 0;
        for (RecordingStore.RecordingInfo info : recordings) totalSamples += info.sampleCount;
        selectedRecordingPaths.retainAll(new java.util.HashSet<>(pathsOf(recordings)));
        boolean allSelected = selectedRecordingPaths.size() == recordings.size();
        LinearLayout toolbar = new LinearLayout(this);
        toolbar.setOrientation(LinearLayout.HORIZONTAL);
        toolbar.setGravity(Gravity.CENTER_VERTICAL);
        toolbar.addView(text(String.format(Locale.CHINA, "%d 条记录 · %.1f 秒 · %d 个样本",
                recordings.size(), totalSamples / 500.0, totalSamples), 12, MUTED),
                new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        Button selectAll = button(allSelected ? "取消全选" : "全选", false);
        historySelectAllButton = selectAll;
        int totalForToggle = recordings.size();
        selectAll.setOnClickListener(view -> {
            if (selectedRecordingPaths.size() == totalForToggle) selectedRecordingPaths.clear();
            else {
                selectedRecordingPaths.clear();
                for (RecordingStore.RecordingInfo info : recordings)
                    selectedRecordingPaths.add(info.file.getAbsolutePath());
            }
            refreshHistoryManager();
        });
        toolbar.addView(selectAll, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, dp(36)));
        Button deleteSelected = button("删除所选(" + selectedRecordingPaths.size() + ")", false);
        deleteSelected.setTextColor(0xFFFF8395);
        deleteSelected.setEnabled(!selectedRecordingPaths.isEmpty());
        historyDeleteSelectedButton = deleteSelected;
        deleteSelected.setOnClickListener(view -> {
            int current = selectedRecordingPaths.size();
            if (current > 0) confirmDeleteSelectedRecordings(current);
        });
        LinearLayout.LayoutParams deleteParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, dp(36));
        deleteParams.leftMargin = dp(6);
        toolbar.addView(deleteSelected, deleteParams);

        // 导入数据按钮
        Button importData = button("导入数据", false);
        importData.setTextColor(ACCENT);
        importData.setOnClickListener(view -> beginImportData());
        LinearLayout.LayoutParams importParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, dp(36));
        importParams.leftMargin = dp(6);
        toolbar.addView(importData, importParams);

        historyListContent.addView(toolbar, matchWrap(dp(4), dp(10)));

        // 数据备份与恢复：把采集数据与模型打包成 zip，便于换机、重装或归档
        LinearLayout backupRow = new LinearLayout(this);
        backupRow.setOrientation(LinearLayout.HORIZONTAL);
        Button backupExport = button("备份全部数据", false);
        backupExport.setTextColor(ACCENT);
        backupExport.setOnClickListener(view -> beginBackupExport());
        backupRow.addView(backupExport, weightedButtonParams(0, 6));
        Button backupImport = button("从备份恢复", false);
        backupImport.setOnClickListener(view -> beginBackupImport());
        backupRow.addView(backupImport, weightedButtonParams(6, 0));
        historyListContent.addView(backupRow, matchWrap(0, dp(10)));

        for (RecordingStore.RecordingInfo info : recordings) {
            LinearLayout item = column();
            item.setPadding(dp(12), dp(10), dp(12), dp(10));
            item.setBackground(rounded(0xFF111D24, 12, 0xFF2A3A43));
            LinearLayout titleRow = new LinearLayout(this);
            titleRow.setOrientation(LinearLayout.HORIZONTAL);
            titleRow.setGravity(Gravity.CENTER_VERTICAL);
            CheckBox check = new CheckBox(this);
            String path = info.file.getAbsolutePath();
            check.setChecked(selectedRecordingPaths.contains(path));
            check.setOnCheckedChangeListener((buttonView, isChecked) -> {
                if (isChecked) selectedRecordingPaths.add(path);
                else selectedRecordingPaths.remove(path);
                updateHistoryToolbarButtons(recordings.size());
            });
            titleRow.addView(check, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT));
            TextView name = text(info.label, 16, INK);
            name.setTypeface(Typeface.DEFAULT_BOLD);
            titleRow.addView(name, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
            item.addView(titleRow);
            item.addView(text(String.format(Locale.CHINA, "用户 %s   ·   %s   ·   %.1f 秒   ·   %d 样本   ·   %s",
                    info.user, formatDate(info.createdAt), info.durationSeconds(), info.sampleCount, info.protocol.name()), 11, MUTED),
                    matchWrap(dp(3), dp(8)));
            LinearLayout actions = new LinearLayout(this);
            actions.setOrientation(LinearLayout.HORIZONTAL);
            Button preview = button("波形", false);
            preview.setOnClickListener(view -> previewRecording(info));
            actions.addView(preview, weightedButtonParams(0, 4));
            Button export = button("导出 CSV", false);
            export.setOnClickListener(view -> beginExport(info.file, "text/csv"));
            actions.addView(export, weightedButtonParams(4, 4));
            Button delete = button("删除", false);
            delete.setTextColor(0xFFFF8395);
            delete.setOnClickListener(view -> confirmDeleteRecording(info));
            actions.addView(delete, weightedButtonParams(4, 0));
            item.addView(actions, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(42)));
            historyListContent.addView(item, matchWrap(0, dp(8)));
        }
    }

    private static List<String> pathsOf(List<RecordingStore.RecordingInfo> recordings) {
        List<String> paths = new ArrayList<>(recordings.size());
        for (RecordingStore.RecordingInfo info : recordings) paths.add(info.file.getAbsolutePath());
        return paths;
    }

    private void updateHistoryToolbarButtons(int total) {
        if (historySelectAllButton != null)
            historySelectAllButton.setText(selectedRecordingPaths.size() == total ? "取消全选" : "全选");
        if (historyDeleteSelectedButton != null) {
            int count = selectedRecordingPaths.size();
            historyDeleteSelectedButton.setText("删除所选(" + count + ")");
            historyDeleteSelectedButton.setEnabled(count > 0);
        }
    }

    private void confirmDeleteSelectedRecordings(int count) {
        new AlertDialog.Builder(this).setTitle("批量删除历史记录？")
                .setMessage("将永久删除所选的 " + count + " 条记录，此操作不可恢复。")
                .setPositiveButton("删除", (dialog, which) -> worker.execute(() -> {
                    List<RecordingStore.RecordingInfo> targets = new ArrayList<>();
                    for (RecordingStore.RecordingInfo info : store.listRecordings()) {
                        if (selectedRecordingPaths.contains(info.file.getAbsolutePath())) targets.add(info);
                    }
                    int deleted = store.deleteRecordings(targets);
                    main.post(() -> {
                        toast(deleted > 0 ? "已删除 " + deleted + " 条记录" : "没有可删除的记录");
                        refreshDatasetAsync();
                        refreshHistoryManager();
                    });
                }))
                .setNegativeButton("取消", null).show();
    }

    private void previewRecording(RecordingStore.RecordingInfo info) {
        worker.execute(() -> {
            try {
                int[][] preview = store.loadPreview(info, 1000);
                main.post(() -> {
                    LinearLayout body = column();
                    body.setPadding(dp(8), dp(4), dp(8), 0);
                    body.addView(text(String.format(Locale.CHINA,
                            "%s · %.1f 秒 · %d 样本\n显示记录末尾最多 2 秒原始波形",
                            formatDate(info.createdAt), info.durationSeconds(), info.sampleCount), 12, MUTED),
                            matchWrap(0, dp(8)));
                    WaveformView previewView = new WaveformView(this);
                    previewView.addSamples(preview);
                    body.addView(previewView, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(190)));
                    new AlertDialog.Builder(this).setTitle(info.label + " · 历史波形")
                            .setView(body).setPositiveButton("关闭", null).show();
                });
            } catch (IOException error) {
                main.post(() -> toast("读取记录失败：" + error.getMessage()));
            }
        });
    }

    private void confirmDeleteRecording(RecordingStore.RecordingInfo info) {
        new AlertDialog.Builder(this).setTitle("删除历史记录？")
                .setMessage("将永久删除“" + info.label + "”的 "
                        + String.format(Locale.CHINA, "%.1f 秒数据。", info.durationSeconds()))
                .setPositiveButton("删除", (dialog, which) -> worker.execute(() -> {
                    boolean deleted = store.deleteRecording(info);
                    main.post(() -> {
                        toast(deleted ? "记录已删除" : "删除失败");
                        selectedRecordingPaths.remove(info.file.getAbsolutePath());
                        refreshDatasetAsync();
                        refreshHistoryManager();
                    });
                }))
                .setNegativeButton("取消", null).show();
    }

    private void showModelManager() {
        refreshModelManager();
    }

    private void refreshModelManager() {
        worker.execute(() -> {
            List<RecordingStore.ModelInfo> models = store.listModels();
            List<RecordingStore.PyTorchModelInfo> ptModels = store.listPytorchModels();
            main.post(() -> presentModelManager(models, ptModels));
        });
    }

    private void presentModelManager(List<RecordingStore.ModelInfo> models,
                                     List<RecordingStore.PyTorchModelInfo> ptModels) {
        boolean reopen = modelDialog == null || !modelDialog.isShowing();
        if (reopen) {
            if (historyDialog != null && historyDialog.isShowing()) historyDialog.dismiss();
            ScrollView scroll = new ScrollView(this);
            modelListContent = column();
            modelListContent.setPadding(dp(12), dp(6), dp(12), dp(12));
            scroll.addView(modelListContent);
            modelDialog = new AlertDialog.Builder(this).setTitle("数据模型管理")
                    .setView(scroll).setNegativeButton("关闭", null).create();
            modelDialog.setOnDismissListener(dialog -> {
                if (modelDialog == dialog) {
                    modelDialog = null;
                    modelListContent = null;
                }
            });
            modelDialog.show();
        }
        populateModelManager(models, ptModels);
    }

    private void populateModelManager(List<RecordingStore.ModelInfo> models,
                                      List<RecordingStore.PyTorchModelInfo> ptModels) {
        if (modelListContent == null) return;
        modelListContent.removeAllViews();
        boolean anyModel = !models.isEmpty() || !ptModels.isEmpty();
        if (!anyModel) {
            TextView empty = text("暂无模型。可点击\"训练新模型\"训练质心/SVM 模型，或\"导入 PyTorch\"加载 CNN 模型。", 14, MUTED);
            empty.setPadding(dp(8), dp(20), dp(8), dp(20));
            modelListContent.addView(empty);
            return;
        }

        // 质心/SVM 模型区段
        if (!models.isEmpty()) {
            modelListContent.addView(text("质心 / SVM 模型", 12, MUTED), matchWrap(0, dp(6)));
            for (RecordingStore.ModelInfo info : models) {
                modelListContent.addView(buildCentroidModelItem(info), matchWrap(0, dp(8)));
            }
        }

        // PyTorch 模型区段
        if (!ptModels.isEmpty()) {
            modelListContent.addView(text("PyTorch CNN 模型", 12, MUTED), matchWrap(0, dp(10)));
            for (RecordingStore.PyTorchModelInfo info : ptModels) {
                modelListContent.addView(buildPytorchModelItem(info), matchWrap(0, dp(8)));
            }
        }
    }

    private View buildCentroidModelItem(RecordingStore.ModelInfo info) {
        LinearLayout item = column();
        item.setPadding(dp(12), dp(10), dp(12), dp(10));
        item.setBackground(rounded(info.active ? 0xFF142820 : 0xFF111D24, 12,
                info.active ? ACCENT : 0xFF2A3A43));
        TextView name = text((info.active ? "● 当前  " : "") + info.name, 16,
                info.active ? ACCENT : INK);
        name.setTypeface(Typeface.DEFAULT_BOLD);
        item.addView(name);
        item.addView(text(formatDate(info.createdAt) + "   ·   " + info.model.algorithm.display
                + "   ·   " + info.usersDisplay()
                + "   ·   " + info.model.trainingWindows
                + " 个窗口   ·   " + info.protocol.name() + "\n" + String.join(" / ", info.model.labels()), 11, MUTED),
                matchWrap(dp(3), dp(8)));
        LinearLayout actions = new LinearLayout(this);
        actions.setOrientation(LinearLayout.HORIZONTAL);
        Button activate = button(info.active ? "已启用" : "启用", false);
        activate.setEnabled(!info.active);
        activate.setOnClickListener(view -> switchToCentroidModel(info));
        actions.addView(activate, weightedButtonParams(0, 4));
        Button export = button("导出 JSON", false);
        export.setOnClickListener(view -> beginExport(info.file, "application/json"));
        actions.addView(export, weightedButtonParams(4, 4));
        Button delete = button("删除", false);
        delete.setTextColor(0xFFFF8395);
        delete.setOnClickListener(view -> confirmDeleteModel(info));
        actions.addView(delete, weightedButtonParams(4, 0));
        item.addView(actions, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(42)));
        return item;
    }

    private View buildPytorchModelItem(RecordingStore.PyTorchModelInfo info) {
        LinearLayout item = column();
        item.setPadding(dp(12), dp(10), dp(12), dp(10));
        item.setBackground(rounded(info.active ? 0xFF142820 : 0xFF111D24, 12,
                info.active ? ACCENT : 0xFF2A3A43));
        TextView name = text((info.active ? "● 当前  " : "") + info.name + "  [PyTorch]", 16,
                info.active ? ACCENT : INK);
        name.setTypeface(Typeface.DEFAULT_BOLD);
        item.addView(name);
        StringBuilder labels = new StringBuilder();
        for (Map.Entry<Integer, String> entry : info.labelMapping.entrySet()) {
            if (labels.length() > 0) labels.append(" / ");
            labels.append(entry.getKey()).append(":").append(entry.getValue());
        }
        item.addView(text(formatDate(info.createdAt) + "   ·   " + info.protocol.name()
                + "   ·   8×100 → 4 类\n" + labels, 11, MUTED), matchWrap(dp(3), dp(8)));
        LinearLayout actions = new LinearLayout(this);
        actions.setOrientation(LinearLayout.HORIZONTAL);
        Button activate = button(info.active ? "已启用" : "启用", false);
        activate.setEnabled(!info.active);
        activate.setOnClickListener(view -> switchToPytorchModel(info));
        actions.addView(activate, weightedButtonParams(0, 4));
        Button export = button("导出元数据", false);
        export.setOnClickListener(view -> beginExport(info.jsonFile, "application/json"));
        actions.addView(export, weightedButtonParams(4, 4));
        Button delete = button("删除", false);
        delete.setTextColor(0xFFFF8395);
        delete.setOnClickListener(view -> confirmDeletePytorchModel(info));
        actions.addView(delete, weightedButtonParams(4, 0));
        item.addView(actions, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(42)));
        return item;
    }

    /** 在 worker 线程切换到质心/SVM 模型（清空 PyTorch 激活状态）。 */
    private void switchToCentroidModel(RecordingStore.ModelInfo info) {
        worker.execute(() -> {
            store.activateModel(info);
            model = info.model;
            pytorchModel = null;
            main.post(() -> {
                stopRecognitionForModelChange();
                updateModelText();
                toast("已启用模型：“" + info.name + "”");
                refreshModelManager();
            });
        });
    }

    /** 在 worker 线程切换到 PyTorch 模型（清空质心激活状态，加载 Module）。 */
    private void switchToPytorchModel(RecordingStore.PyTorchModelInfo info) {
        worker.execute(() -> {
            try {
                PyTorchModel loaded = PyTorchModel.load(info.ptFile, info.labelMapping,
                        info.protocol, info.name, info.createdAt, info.normMeans, info.normStds,
                        info.filterB, info.filterA, info.filterZi);
                store.activatePytorchModel(info);
                pytorchModel = loaded;
                model = null;
                main.post(() -> {
                    stopRecognitionForModelChange();
                    updateModelText();
                    toast("已启用 PyTorch 模型：“" + info.name + "”");
                    refreshModelManager();
                });
            } catch (Exception error) {
                main.post(() -> {
                    toast("PyTorch 模型加载失败：" + error.getMessage());
                    refreshModelManager();
                });
            }
        });
    }

    private void confirmDeleteModel(RecordingStore.ModelInfo info) {
        new AlertDialog.Builder(this).setTitle("删除模型？")
                .setMessage("将永久删除模型“" + info.name + "”。历史采集数据不会受影响。")
                .setPositiveButton("删除", (dialog, which) -> worker.execute(() -> {
                    GestureModel remaining = store.deleteModel(info);
                    model = remaining;
                    main.post(() -> {
                        stopRecognitionForModelChange();
                        updateModelText();
                        toast("模型已删除");
                        refreshModelManager();
                    });
                }))
                .setNegativeButton("取消", null).show();
    }

    private void confirmDeletePytorchModel(RecordingStore.PyTorchModelInfo info) {
        new AlertDialog.Builder(this).setTitle("删除 PyTorch 模型？")
                .setMessage("将永久删除 PyTorch 模型“" + info.name + "”及其 .pt 权重文件。")
                .setPositiveButton("删除", (dialog, which) -> worker.execute(() -> {
                    boolean wasActive = store.deletePytorchModel(info);
                    if (wasActive) {
                        pytorchModel = null;
                        // 回退到质心模型
                        model = store.loadModel();
                    }
                    main.post(() -> {
                        stopRecognitionForModelChange();
                        updateModelText();
                        toast("PyTorch 模型已删除");
                        refreshModelManager();
                    });
                }))
                .setNegativeButton("取消", null).show();
    }

    private void stopRecognitionForModelChange() {
        recognizing = false;
        if (g20 != null) g20.stopMotion("G20 控制已启用 · 模型已切换，已停止");
        recognizeButton.setText("开始实时识别");
        resultView.setText("—");
        synchronized (inferenceLock) {
            inferenceWindow.clear();
            recentPredictions.clear();
        }
    }

    private void beginExport(File file, String mime) {
        pendingExportFile = file;
        Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType(mime);
        intent.putExtra(Intent.EXTRA_TITLE, file.getName());
        try { startActivityForResult(intent, REQUEST_EXPORT); }
        catch (Exception error) { toast("没有可用的文件保存应用"); }
    }

    /** 触发系统文件选择器，让用户选择 .pt 文件。 */
    private void beginImportPytorchModel() {
        if (recording != null) { toast("请先停止当前采集"); return; }
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        try { startActivityForResult(intent, REQUEST_IMPORT_PT); }
        catch (Exception error) { toast("没有可用的文件选择应用"); }
    }

    /** 触发系统文件选择器，让用户选择 CSV 数据文件导入。 */
    private void beginImportData() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("text/csv");
        // 允许多选
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
        try { startActivityForResult(intent, REQUEST_IMPORT_DATA); }
        catch (Exception error) { toast("没有可用的文件选择应用"); }
    }

    /** 备份全部采集数据与模型为一个 zip 文件，便于换机、重装或归档。 */
    private void beginBackupExport() {
        String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.CHINA).format(new Date());
        Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("application/zip");
        intent.putExtra(Intent.EXTRA_TITLE, "emg_backup_" + stamp + ".zip");
        try { startActivityForResult(intent, REQUEST_EXPORT_BACKUP); }
        catch (Exception error) { toast("没有可用的文件保存应用"); }
    }

    /** 从 zip 备份恢复采集数据与模型（同名文件不覆盖，自动追加序号）。 */
    private void beginBackupImport() {
        if (recording != null) { toast("请先停止当前采集"); return; }
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        try { startActivityForResult(intent, REQUEST_IMPORT_BACKUP); }
        catch (Exception error) { toast("没有可用的文件选择应用"); }
    }

    /**
     * 让用户为 PyTorch CNN 输出的 4 个类别（0/1/2/3）配置对应的动作标签。
     * 选择项包含小车控制命令（前进/后退/左转/右转/停止）和常见手势名（握拳/张手/上翻/下翻）。
     */
    private void showPytorchLabelMappingDialog() {
        if (pendingImportUri == null) { toast("导入失败：未获取文件"); return; }

        LinearLayout content = column();
        content.setPadding(dp(12), dp(6), dp(12), dp(12));

        // 模型名称输入
        EditText nameInput = new EditText(this);
        nameInput.setHint("模型名称，例如 lx_model_v1");
        nameInput.setText("PyTorch " + new SimpleDateFormat("MM-dd HH:mm", Locale.CHINA).format(new Date()));
        nameInput.setSingleLine(true);
        nameInput.setTextColor(INK);
        nameInput.setTextSize(13);
        nameInput.setBackground(rounded(0xFF0B151B, 8, 0xFF2A3A43));
        nameInput.setPadding(dp(10), 0, dp(10), 0);
        content.addView(nameInput, matchHeight(dp(40), 0, dp(10)));

        // 默认选项：与 control_car_sep.py 一致 0=前进 1=后退 2=左转 3=右转
        String[] options = {"前进", "后退", "左转", "右转", "停止", "握拳", "张手", "上翻", "下翻", "rest"};
        String[] defaults = {"前进", "后退", "左转", "右转"};
        Spinner[] spinners = new Spinner[4];
        for (int i = 0; i < 4; i++) {
            TextView labelView = text("类别 " + i + ":", 13, INK);
            content.addView(labelView, matchWrap(0, dp(4)));
            Spinner spinner = new Spinner(this);
            ArrayAdapter<String> adapter = new ArrayAdapter<>(this,
                    android.R.layout.simple_spinner_item, options);
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
            spinner.setAdapter(adapter);
            spinner.setBackground(rounded(0xFF0B151B, 8, 0xFF2A3A43));
            for (int j = 0; j < options.length; j++) {
                if (options[j].equals(defaults[i])) { spinner.setSelection(j); break; }
            }
            spinners[i] = spinner;
            content.addView(spinner, matchWrap(0, dp(8)));
        }

        new AlertDialog.Builder(this)
                .setTitle("配置类别映射")
                .setView(content)
                .setPositiveButton("导入", (dialog, which) -> {
                    String name = nameInput.getText().toString().trim();
                    if (name.isEmpty()) name = "PyTorch 模型";
                    Map<Integer, String> mapping = new LinkedHashMap<>();
                    for (int i = 0; i < 4; i++) mapping.put(i, (String) spinners[i].getSelectedItem());
                    performPytorchImport(name, mapping);
                })
                .setNegativeButton("取消", (dialog, which) -> pendingImportUri = null)
                .show();
    }

    /** 在 worker 线程把 Uri 内容复制到缓存文件 → 调用 store.importPytorchModel → 加载 Module。 */
    private void performPytorchImport(String name, Map<Integer, String> mapping) {
        Uri source = pendingImportUri;
        pendingImportUri = null;
        if (source == null) { toast("导入失败：未获取文件"); return; }
        modelView.setText("正在导入 PyTorch 模型…");
        worker.execute(() -> {
            File cacheFile = new File(getCacheDir(), "import_pt_" + System.currentTimeMillis() + ".pt");
            try {
                try (InputStream input = getContentResolver().openInputStream(source);
                     OutputStream output = new FileOutputStream(cacheFile)) {
                    if (input == null) throw new IOException("无法打开源文件");
                    byte[] buffer = new byte[32768];
                    int count;
                    while ((count = input.read(buffer)) >= 0) output.write(buffer, 0, count);
                }
                RecordingStore.PyTorchModelInfo info = store.importPytorchModel(
                        cacheFile, name, mapping, parser.getProtocol());
                // 加载 Module
                PyTorchModel loaded = PyTorchModel.load(info.ptFile, info.labelMapping,
                        info.protocol, info.name, info.createdAt, info.normMeans, info.normStds,
                        info.filterB, info.filterA, info.filterZi);
                pytorchModel = loaded;
                model = null;  // 切换到 PyTorch 引擎
                main.post(() -> {
                    stopRecognitionForModelChange();
                    updateModelText();
                    toast("PyTorch 模型已导入并激活");
                });
            } catch (Exception error) {
                main.post(() -> {
                    modelView.setText("PyTorch 导入失败：" + error.getMessage());
                    toast("导入失败：" + error.getMessage());
                });
            } finally {
                cacheFile.delete();
            }
        });
    }

    /** 云端上传对话框：增量上传数据到服务器（不训练） */
    private void showCloudUploadDialog() {
        if (recording != null) {
            toast("请先停止当前采集");
            return;
        }

        // 检查是否有采集数据
        List<RecordingStore.RecordingInfo> recordings = trainableRecordings(store.listRecordings());
        if (recordings.isEmpty()) {
            toast("暂无采集数据，请先采集数据");
            return;
        }

        // 先检查登录状态
        showLoginDialog(() -> performCloudUploadWithAuth());
    }

    /** 执行云端上传（已登录） */
    private void performCloudUploadWithAuth() {
        List<RecordingStore.RecordingInfo> recordings = trainableRecordings(store.listRecordings());
        if (recordings.isEmpty()) {
            toast("暂无采集数据");
            return;
        }

        // 获取服务器地址
        android.content.SharedPreferences prefs = getSharedPreferences("semg_prefs", MODE_PRIVATE);
        String serverUrl = prefs.getString("server_url", "http://123.57.250.63:5000");

        LinearLayout content = column();
        content.setPadding(dp(12), dp(6), dp(12), dp(12));

        // 显示当前用户信息
        TextView userInfo = text("当前用户: " + currentUsername, 13, ACCENT);
        userInfo.setPadding(0, 0, 0, dp(8));
        content.addView(userInfo, matchWrap(0, dp(4)));

        // 状态显示
        TextView statusView = text("准备上传数据...", 12, MUTED);
        statusView.setPadding(0, dp(8), 0, dp(8));
        content.addView(statusView, matchWrap(0, dp(4)));

        final AlertDialog dialog = new AlertDialog.Builder(this)
                .setTitle("云端上传（增量）")
                .setView(content)
                .setPositiveButton("开始上传", null)
                .setNegativeButton("取消", null)
                .create();

        dialog.setOnShowListener(d -> {
            Button positive = dialog.getButton(AlertDialog.BUTTON_POSITIVE);
            positive.setOnClickListener(v -> {
                if (serverUrl.isEmpty()) {
                    toast("服务器地址未配置");
                    return;
                }

                // 禁用按钮，开始增量上传流程
                positive.setEnabled(false);
                statusView.setText("正在查询云端已有文件...");
                statusView.setTextColor(ACCENT);

                final String finalServerUrl = serverUrl;
                worker.execute(() -> performCloudUploadWithToken(finalServerUrl, recordings, statusView, dialog));
            });
        });

        dialog.show();
    }

    /** 执行云端增量上传（使用 Token 认证）：查询云端文件 → 对比本地 → 上传新文件 */
    private void performCloudUploadWithToken(String serverUrl,
                                             List<RecordingStore.RecordingInfo> recordings,
                                             TextView statusView, AlertDialog dialog) {
        try {
            // 1. 查询云端已有文件列表
            main.post(() -> statusView.setText("正在查询云端已有文件..."));
            Set<String> cloudFiles = queryCloudFilesWithToken(serverUrl);
            final Set<String> finalCloudFiles = (cloudFiles != null) ? cloudFiles : new HashSet<>();
            if (cloudFiles == null) {
                main.post(() -> statusView.setText("查询云端文件失败，将上传所有文件"));
            }

            // 2. 对比本地文件，筛选出需要上传的新文件
            List<RecordingStore.RecordingInfo> filesToUpload = new ArrayList<>();
            int skippedCount = 0;
            for (RecordingStore.RecordingInfo rec : recordings) {
                File csvFile = rec.file;
                if (csvFile == null || !csvFile.exists() || !csvFile.getName().endsWith(".csv")) continue;
                
                // 如果云端已有该文件，则跳过
                if (finalCloudFiles.contains(csvFile.getName())) {
                    skippedCount++;
                    continue;
                }
                filesToUpload.add(rec);
            }

            if (filesToUpload.isEmpty()) {
                main.post(() -> {
                    statusView.setText("所有文件已在云端，无需上传");
                    statusView.setTextColor(MUTED);
                    dialog.getButton(AlertDialog.BUTTON_POSITIVE).setEnabled(true);
                });
                return;
            }

            // 3. 上传新文件（增量模式）
            final int finalSkipped = skippedCount;
            main.post(() -> statusView.setText("正在上传 " + filesToUpload.size() + " 个新文件（跳过 " + finalSkipped + " 个已存在）..."));
            String uploadResult = uploadDataToCloudWithToken(serverUrl, filesToUpload);
            if (uploadResult == null) {
                main.post(() -> {
                    statusView.setText("上传失败");
                    statusView.setTextColor(0xFFFF8395);
                    dialog.getButton(AlertDialog.BUTTON_POSITIVE).setEnabled(true);
                });
                return;
            }

            main.post(() -> {
                statusView.setText("上传成功！新增 " + filesToUpload.size() + " 个文件，跳过 " + finalSkipped + " 个已存在文件");
                statusView.setTextColor(ACCENT);
                dialog.getButton(AlertDialog.BUTTON_POSITIVE).setEnabled(true);
            });

        } catch (Exception error) {
            final String errMsg = error.getMessage() != null ? error.getMessage() : error.toString();
            main.post(() -> {
                statusView.setText("错误: " + errMsg);
                statusView.setTextColor(0xFFFF8395);
                dialog.getButton(AlertDialog.BUTTON_POSITIVE).setEnabled(true);
            });
        }
    }

    /** 查询云端已有文件列表 */
    private Set<String> queryCloudFiles(String serverUrl, String user) {
        try {
            if (serverUrl.endsWith("/")) serverUrl = serverUrl.substring(0, serverUrl.length() - 1);
            URL url = new URL(serverUrl + "/api/files/" + Uri.encode(user));
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setConnectTimeout(10000);
            conn.setReadTimeout(10000);

            if (conn.getResponseCode() == 200) {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getInputStream(), "UTF-8"))) {
                    StringBuilder sb = new StringBuilder();
                    String line;
                    while ((line = br.readLine()) != null) sb.append(line);
                    
                    // 解析 JSON 获取文件列表
                    JSONObject json = new JSONObject(sb.toString());
                    JSONArray filesArray = json.getJSONArray("files");
                    Set<String> files = new HashSet<>();
                    for (int i = 0; i < filesArray.length(); i++) {
                        files.add(filesArray.getString(i));
                    }
                    return files;
                }
            }
            return null;
        } catch (Exception e) {
            return null;
        }
    }

    /** 增量上传数据到云端（只上传新文件） */
    private String uploadDataToCloudIncremental(String serverUrl, String user,
                                                List<RecordingStore.RecordingInfo> recordings) {
        try {
            if (serverUrl.endsWith("/")) serverUrl = serverUrl.substring(0, serverUrl.length() - 1);
            URL url = new URL(serverUrl + "/api/upload");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setDoOutput(true);
            conn.setConnectTimeout(30000);
            conn.setReadTimeout(120000);

            String boundary = "----EMGUpload" + System.currentTimeMillis();
            conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);

            try (OutputStream os = conn.getOutputStream()) {
                // 添加 user 字段
                os.write(("--" + boundary + "\r\n").getBytes("UTF-8"));
                os.write("Content-Disposition: form-data; name=\"user\"\r\n\r\n".getBytes("UTF-8"));
                os.write(user.getBytes("UTF-8"));
                os.write("\r\n".getBytes("UTF-8"));

                // 添加 mode=incremental 字段
                os.write(("--" + boundary + "\r\n").getBytes("UTF-8"));
                os.write("Content-Disposition: form-data; name=\"mode\"\r\n\r\n".getBytes("UTF-8"));
                os.write("incremental".getBytes("UTF-8"));
                os.write("\r\n".getBytes("UTF-8"));

                // 添加所有 CSV 文件
                for (RecordingStore.RecordingInfo rec : recordings) {
                    File csvFile = rec.file;
                    if (csvFile == null || !csvFile.exists() || !csvFile.getName().endsWith(".csv")) continue;
                    
                    os.write(("--" + boundary + "\r\n").getBytes("UTF-8"));
                    os.write(("Content-Disposition: form-data; name=\"files\"; filename=\""
                            + csvFile.getName() + "\"\r\n").getBytes("UTF-8"));
                    os.write("Content-Type: text/csv\r\n\r\n".getBytes("UTF-8"));

                    try (FileInputStream fis = new FileInputStream(csvFile)) {
                        byte[] buffer = new byte[8192];
                        int len;
                        while ((len = fis.read(buffer)) != -1) {
                            os.write(buffer, 0, len);
                        }
                    }
                    os.write("\r\n".getBytes("UTF-8"));
                }

                os.write(("--" + boundary + "--\r\n").getBytes("UTF-8"));
                os.flush();
            }

            int responseCode = conn.getResponseCode();
            if (responseCode == 200) {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getInputStream(), "UTF-8"))) {
                    StringBuilder sb = new StringBuilder();
                    String line;
                    while ((line = br.readLine()) != null) sb.append(line);
                    return sb.toString();
                }
            } else {
                InputStream errStream = conn.getErrorStream();
                if (errStream != null) {
                    try (BufferedReader br = new BufferedReader(new InputStreamReader(errStream, "UTF-8"))) {
                        StringBuilder sb = new StringBuilder();
                        String line;
                        while ((line = br.readLine()) != null) sb.append(line);
                        throw new IOException("服务器返回 " + responseCode + ": " + sb.toString());
                    }
                }
                throw new IOException("服务器返回 HTTP " + responseCode);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private String loadServerUrlPref() {
        return getSharedPreferences("semg_prefs", MODE_PRIVATE)
            .getString("server_url", "http://123.57.250.63:5000");
    }

    /** 显示登录/注册入口选择对话框 */
    private void showLoginDialog(Runnable onSuccess) {
        if (authToken != null && currentUsername != null) {
            if (onSuccess != null) onSuccess.run();
            return;
        }

        new AlertDialog.Builder(this)
            .setTitle("用户认证")
            .setMessage("请选择操作")
            .setPositiveButton("登录", (d, w) -> showLoginOnlyDialog(onSuccess))
            .setNegativeButton("注册", (d, w) -> showRegisterDialog(onSuccess))
            .setNeutralButton("取消", null)
            .show();
    }

    private void showLoginOnlyDialog(Runnable onSuccess) {
        LinearLayout content = column();
        content.setPadding(dp(12), dp(6), dp(12), dp(12));

        EditText serverInput = buildServerInput();
        EditText userInput = buildTextInput("请输入用户名", 13);
        EditText passInput = buildPassInput();

        content.addView(text("服务器地址:", 13, INK), matchWrap(0, dp(4)));
        content.addView(serverInput, matchHeight(dp(40), 0, dp(10)));
        content.addView(text("用户名:", 13, INK), matchWrap(0, dp(4)));
        content.addView(userInput, matchHeight(dp(40), 0, dp(10)));
        content.addView(text("密码:", 13, INK), matchWrap(0, dp(4)));
        content.addView(passInput, matchHeight(dp(40), 0, dp(10)));

        TextView statusView = text("请输入账号密码登录", 12, MUTED);
        statusView.setPadding(0, dp(8), 0, dp(8));
        content.addView(statusView, matchWrap(0, dp(4)));

        AlertDialog dialog = new AlertDialog.Builder(this)
            .setTitle("登录")
            .setView(content)
            .setPositiveButton("登录", null)
            .setNegativeButton("去注册", (d, w) -> showRegisterDialog(onSuccess))
            .setNeutralButton("取消", null)
            .create();

        dialog.setOnShowListener(d -> {
            Button btn = dialog.getButton(AlertDialog.BUTTON_POSITIVE);
            btn.setOnClickListener(v -> {
                String serverUrl = serverInput.getText().toString().trim();
                String username = userInput.getText().toString().trim();
                String password = passInput.getText().toString().trim();

                if (username.isEmpty() || password.isEmpty()) { toast("请输入用户名和密码"); return; }
                if (serverUrl.isEmpty()) { toast("请输入服务器地址"); return; }

                btn.setEnabled(false);
                statusView.setText("正在登录...");
                statusView.setTextColor(ACCENT);
                worker.execute(() -> performLogin(serverUrl, username, password, statusView, dialog, btn, btn, onSuccess));
            });
        });

        dialog.show();
    }

    private void showRegisterDialog(Runnable onSuccess) {
        LinearLayout content = column();
        content.setPadding(dp(12), dp(6), dp(12), dp(12));

        EditText serverInput = buildServerInput();
        EditText userInput = buildTextInput("3-20 个字符", 13);
        EditText passInput = buildPassInput();

        Spinner genderSpinner = new Spinner(this);
        genderSpinner.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_item,
            new String[]{"男", "女"}));
        genderSpinner.setBackground(rounded(0xFF0B151B, 8, 0xFF2A3A43));

        EditText weightInput = new EditText(this);
        weightInput.setHint("例如 65（可留空）");
        weightInput.setInputType(android.text.InputType.TYPE_CLASS_NUMBER | android.text.InputType.TYPE_NUMBER_FLAG_DECIMAL);
        weightInput.setSingleLine(true);
        weightInput.setTextColor(INK);
        weightInput.setTextSize(13);
        weightInput.setBackground(rounded(0xFF0B151B, 8, 0xFF2A3A43));
        weightInput.setPadding(dp(10), 0, dp(10), 0);

        Spinner handSpinner = new Spinner(this);
        handSpinner.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_item,
            new String[]{"右手", "左手"}));
        handSpinner.setBackground(rounded(0xFF0B151B, 8, 0xFF2A3A43));

        content.addView(text("服务器地址:", 13, INK), matchWrap(0, dp(4)));
        content.addView(serverInput, matchHeight(dp(40), 0, dp(10)));
        content.addView(text("用户名:", 13, INK), matchWrap(0, dp(4)));
        content.addView(userInput, matchHeight(dp(40), 0, dp(10)));
        content.addView(text("密码:", 13, INK), matchWrap(0, dp(4)));
        content.addView(passInput, matchHeight(dp(40), 0, dp(10)));
        content.addView(text("性别:", 13, INK), matchWrap(0, dp(4)));
        content.addView(genderSpinner, matchHeight(dp(40), 0, dp(10)));
        content.addView(text("体重(kg):", 13, INK), matchWrap(0, dp(4)));
        content.addView(weightInput, matchHeight(dp(40), 0, dp(10)));
        content.addView(text("习惯手:", 13, INK), matchWrap(0, dp(4)));
        content.addView(handSpinner, matchHeight(dp(40), 0, dp(10)));

        TextView statusView = text("填写信息后点击注册", 12, MUTED);
        statusView.setPadding(0, dp(8), 0, dp(8));
        content.addView(statusView, matchWrap(0, dp(4)));

        AlertDialog dialog = new AlertDialog.Builder(this)
            .setTitle("注册新账号")
            .setView(content)
            .setPositiveButton("注册", null)
            .setNegativeButton("已有账号登录", (d, w) -> showLoginOnlyDialog(onSuccess))
            .setNeutralButton("取消", null)
            .create();

        dialog.setOnShowListener(d -> {
            Button btn = dialog.getButton(AlertDialog.BUTTON_POSITIVE);
            btn.setOnClickListener(v -> {
                String serverUrl = serverInput.getText().toString().trim();
                String username = userInput.getText().toString().trim();
                String password = passInput.getText().toString().trim();

                if (username.isEmpty() || password.isEmpty()) { toast("请输入用户名和密码"); return; }
                if (username.length() < 3 || username.length() > 20) { toast("用户名长度 3-20 个字符"); return; }
                if (password.length() < 6) { toast("密码长度至少 6 个字符"); return; }
                if (serverUrl.isEmpty()) { toast("请输入服务器地址"); return; }

                btn.setEnabled(false);
                statusView.setText("正在注册...");
                statusView.setTextColor(ACCENT);
                String gender = genderSpinner.getSelectedItemPosition() == 0 ? "male" : "female";
                String weightStr = weightInput.getText().toString().trim();
                String hand = handSpinner.getSelectedItemPosition() == 0 ? "right" : "left";
                worker.execute(() -> performRegister(serverUrl, username, password,
                    gender, weightStr, hand, statusView, dialog, btn, btn, onSuccess));
            });
        });

        dialog.show();
    }

    private EditText buildServerInput() {
        EditText et = new EditText(this);
        et.setHint("http://123.57.250.63:5000");
        et.setText(loadServerUrlPref());
        et.setSingleLine(true);
        et.setTextColor(INK);
        et.setTextSize(13);
        et.setBackground(rounded(0xFF0B151B, 8, 0xFF2A3A43));
        et.setPadding(dp(10), 0, dp(10), 0);
        return et;
    }

    private EditText buildTextInput(String hint, int size) {
        EditText et = new EditText(this);
        et.setHint(hint);
        et.setSingleLine(true);
        et.setTextColor(INK);
        et.setTextSize(size);
        et.setBackground(rounded(0xFF0B151B, 8, 0xFF2A3A43));
        et.setPadding(dp(10), 0, dp(10), 0);
        return et;
    }

    private EditText buildPassInput() {
        EditText et = new EditText(this);
        et.setHint("请输入密码");
        et.setInputType(android.text.InputType.TYPE_CLASS_TEXT | android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD);
        et.setSingleLine(true);
        et.setTextColor(INK);
        et.setTextSize(13);
        et.setBackground(rounded(0xFF0B151B, 8, 0xFF2A3A43));
        et.setPadding(dp(10), 0, dp(10), 0);
        return et;
    }

    /** 执行登录 */
    private void performLogin(String serverUrl, String username, String password,
                              TextView statusView, AlertDialog dialog,
                              Button loginBtn, Button registerBtn, Runnable onSuccess) {
        try {
            if (serverUrl.endsWith("/")) serverUrl = serverUrl.substring(0, serverUrl.length() - 1);
            URL url = new URL(serverUrl + "/api/login");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
            conn.setDoOutput(true);
            conn.setConnectTimeout(15000);
            conn.setReadTimeout(15000);

            String json = "{\"username\":\"" + username + "\",\"password\":\"" + password + "\"}";
            try (java.io.OutputStream os = conn.getOutputStream()) {
                os.write(json.getBytes("UTF-8"));
                os.flush();
            }

            if (conn.getResponseCode() == 200) {
                StringBuilder sb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), "UTF-8"))) {
                    String line;
                    while ((line = br.readLine()) != null) sb.append(line);
                }
                org.json.JSONObject jsonObj = new org.json.JSONObject(sb.toString());
                authToken = jsonObj.getString("token");
                currentUsername = jsonObj.getString("username");
                currentUserId = jsonObj.getInt("user_id");
                isAdmin = jsonObj.optBoolean("is_admin", false) || "admin".equals(jsonObj.optString("role", ""));

                // 保存 Token 和用户元数据
                String gender = jsonObj.optString("gender", "");
                double weight = jsonObj.optDouble("weight", 0.0);
                String hand = jsonObj.optString("hand", "");
                android.content.SharedPreferences prefs = getSharedPreferences("semg_prefs", MODE_PRIVATE);
                prefs.edit()
                    .putString("auth_token", authToken)
                    .putString("current_username", currentUsername)
                    .putInt("current_user_id", currentUserId)
                    .putBoolean("is_admin", isAdmin)
                    .putString("server_url", serverUrl)
                    .putString("gender", gender)
                    .putString("hand", hand)
                    .putFloat("weight", (float) weight)
                    .apply();

                main.post(() -> {
                    toast("登录成功: " + currentUsername);
                    updateUserButton();
                    userInput.setText(currentUsername);
                    userInput.setEnabled(false);
                    userInput.setAlpha(0.55f);
                    userInput.setHint("已登录用户");
                    dialog.dismiss();
                    if (onSuccess != null) onSuccess.run();
                });
            } else {
                StringBuilder sb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getErrorStream(), "UTF-8"))) {
                    String line;
                    while ((line = br.readLine()) != null) sb.append(line);
                }
                String errorMsg = "登录失败";
                try {
                    org.json.JSONObject errJson = new org.json.JSONObject(sb.toString());
                    errorMsg = errJson.optString("error", errorMsg);
                } catch (Exception ignored) {}
                final String finalErrorMsg = errorMsg;
                main.post(() -> {
                    statusView.setText(finalErrorMsg);
                    statusView.setTextColor(0xFFFF8395);
                    loginBtn.setEnabled(true);
                    registerBtn.setEnabled(true);
                });
            }
        } catch (Exception e) {
            main.post(() -> {
                statusView.setText("登录失败: " + e.getMessage());
                statusView.setTextColor(0xFFFF8395);
                loginBtn.setEnabled(true);
                registerBtn.setEnabled(true);
            });
        }
    }

    /** 执行注册 */
    private void performRegister(String serverUrl, String username, String password,
                              String gender, String weightStr, String hand,
                              TextView statusView, AlertDialog dialog,
                              Button loginBtn, Button registerBtn, Runnable onSuccess) {
        try {
            if (serverUrl.endsWith("/")) serverUrl = serverUrl.substring(0, serverUrl.length() - 1);
            URL url = new URL(serverUrl + "/api/register");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
            conn.setDoOutput(true);
            conn.setConnectTimeout(15000);
            conn.setReadTimeout(15000);

            String weightJson = weightStr.isEmpty() ? "null" : weightStr;
            String json = "{\"username\":\"" + username + "\",\"password\":\"" + password
                + "\",\"gender\":\"" + gender + "\",\"weight\":" + weightJson
                + ",\"hand\":\"" + hand + "\"}";
            try (java.io.OutputStream os = conn.getOutputStream()) {
                os.write(json.getBytes("UTF-8"));
                os.flush();
            }

            if (conn.getResponseCode() == 201) {
                StringBuilder sb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), "UTF-8"))) {
                    String line;
                    while ((line = br.readLine()) != null) sb.append(line);
                }
                org.json.JSONObject jsonObj = new org.json.JSONObject(sb.toString());
                authToken = jsonObj.getString("token");
                currentUsername = jsonObj.getString("username");
                currentUserId = jsonObj.getInt("user_id");
                isAdmin = jsonObj.optBoolean("is_admin", false) || "admin".equals(jsonObj.optString("role", ""));

                // 保存 Token 和用户元数据
                String regGender = jsonObj.optString("gender", gender);
                double regWeight = jsonObj.optDouble("weight", 0.0);
                String regHand = jsonObj.optString("hand", hand);
                android.content.SharedPreferences prefs = getSharedPreferences("semg_prefs", MODE_PRIVATE);
                prefs.edit()
                    .putString("auth_token", authToken)
                    .putString("current_username", currentUsername)
                    .putInt("current_user_id", currentUserId)
                    .putBoolean("is_admin", isAdmin)
                    .putString("server_url", serverUrl)
                    .putString("gender", regGender)
                    .putString("hand", regHand)
                    .putFloat("weight", (float) regWeight)
                    .apply();

                main.post(() -> {
                    toast("注册成功: " + currentUsername);
                    updateUserButton();
                    userInput.setText(currentUsername);
                    userInput.setEnabled(false);
                    userInput.setAlpha(0.55f);
                    userInput.setHint("已登录用户");
                    dialog.dismiss();
                    if (onSuccess != null) onSuccess.run();
                });
            } else {
                StringBuilder sb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getErrorStream(), "UTF-8"))) {
                    String line;
                    while ((line = br.readLine()) != null) sb.append(line);
                }
                String errorMsg = "注册失败";
                try {
                    org.json.JSONObject errJson = new org.json.JSONObject(sb.toString());
                    errorMsg = errJson.optString("error", errorMsg);
                } catch (Exception ignored) {}
                final String finalErrorMsg = errorMsg;
                main.post(() -> {
                    statusView.setText(finalErrorMsg);
                    statusView.setTextColor(0xFFFF8395);
                    loginBtn.setEnabled(true);
                    registerBtn.setEnabled(true);
                });
            }
        } catch (Exception e) {
            main.post(() -> {
                statusView.setText("注册失败: " + e.getMessage());
                statusView.setTextColor(0xFFFF8395);
                loginBtn.setEnabled(true);
                registerBtn.setEnabled(true);
            });
        }
    }

    /** 退出登录 */
    private void logout() {
        authToken = null;
        currentUsername = null;
        currentUserId = 0;
        isAdmin = false;
        android.content.SharedPreferences prefs = getSharedPreferences("semg_prefs", MODE_PRIVATE);
        prefs.edit()
            .remove("auth_token")
            .remove("current_username")
            .remove("current_user_id")
            .remove("is_admin")
            .apply();
        updateUserButton();
        userInput.setEnabled(true);
        userInput.setAlpha(1.0f);
        userInput.setHint("用户名，例如 张三");
        userInput.setText("");
        toast("已退出登录");
    }

    /** 显示升级为管理员对话框 */
    private void showUpgradeToAdminDialog() {
        if (isAdmin) {
            toast("您已经是管理员");
            return;
        }

        LinearLayout content = column();
        content.setPadding(dp(12), dp(6), dp(12), dp(12));

        TextView hint = text("输入管理员密码以升级为管理员", 13, INK);
        hint.setPadding(0, 0, 0, dp(12));
        content.addView(hint, matchWrap(0, dp(8)));

        EditText passwordInput = new EditText(this);
        passwordInput.setHint("请输入管理员密码");
        passwordInput.setInputType(android.text.InputType.TYPE_CLASS_TEXT | android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD);
        passwordInput.setSingleLine(true);
        passwordInput.setTextColor(INK);
        passwordInput.setTextSize(14);
        passwordInput.setBackground(rounded(0xFF0B151B, 8, 0xFF2A3A43));
        passwordInput.setPadding(dp(10), 0, dp(10), 0);
        content.addView(passwordInput, matchHeight(dp(42), 0, dp(8)));

        TextView statusView = text("", 12, MUTED);
        statusView.setPadding(0, dp(4), 0, dp(4));
        content.addView(statusView, matchWrap(0, dp(4)));

        AlertDialog dialog = new AlertDialog.Builder(this)
            .setTitle("升级为管理员")
            .setView(content)
            .setPositiveButton("升级", null)
            .setNegativeButton("取消", null)
            .create();

        dialog.setOnShowListener(d -> {
            Button upgradeBtn = dialog.getButton(AlertDialog.BUTTON_POSITIVE);
            upgradeBtn.setOnClickListener(v -> {
                String adminPassword = passwordInput.getText().toString().trim();
                if (adminPassword.isEmpty()) {
                    toast("请输入管理员密码");
                    return;
                }

                upgradeBtn.setEnabled(false);
                statusView.setText("正在验证...");
                statusView.setTextColor(ACCENT);

                worker.execute(() -> upgradeToAdmin(adminPassword, statusView, dialog, upgradeBtn));
            });
        });

        dialog.show();
    }

    /** 执行升级为管理员操作 */
    private void upgradeToAdmin(String adminPassword, TextView statusView, AlertDialog dialog, Button upgradeBtn) {
        try {
            String serverUrl = getSharedPreferences("semg_prefs", MODE_PRIVATE)
                .getString("server_url", "http://123.57.250.63:5000");
            if (serverUrl.endsWith("/")) serverUrl = serverUrl.substring(0, serverUrl.length() - 1);

            URL url = new URL(serverUrl + "/api/admin/set_admin");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
            conn.setDoOutput(true);
            conn.setConnectTimeout(15000);
            conn.setReadTimeout(15000);

            String json = "{\"username\":\"" + currentUsername + "\",\"is_admin\":true," +
                "\"admin_password\":\"" + adminPassword + "\"}";
            try (java.io.OutputStream os = conn.getOutputStream()) {
                os.write(json.getBytes("UTF-8"));
                os.flush();
            }

            if (conn.getResponseCode() == 200) {
                main.post(() -> {
                    isAdmin = true;
                    getSharedPreferences("semg_prefs", MODE_PRIVATE)
                        .edit()
                        .putBoolean("is_admin", true)
                        .apply();
                    updateUserButton();
                    dialog.dismiss();
                    toast("升级成功！您现在是管理员");
                });
            } else {
                StringBuilder sb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getErrorStream(), "UTF-8"))) {
                    String line;
                    while ((line = br.readLine()) != null) sb.append(line);
                }
                final String errorMsg = sb.length() > 0 ? sb.toString() : "升级失败";
                main.post(() -> {
                    statusView.setText("错误: " + errorMsg);
                    statusView.setTextColor(0xFFFF8395);
                    upgradeBtn.setEnabled(true);
                });
            }
        } catch (Exception e) {
            main.post(() -> {
                statusView.setText("网络错误: " + e.getMessage());
                statusView.setTextColor(0xFFFF8395);
                upgradeBtn.setEnabled(true);
            });
        }
    }

    /** 显示管理员功能面板 */
    private void showAdminPanel() {
        if (!isAdmin) {
            toast("您没有管理员权限");
            return;
        }

        LinearLayout content = column();
        content.setPadding(dp(12), dp(6), dp(12), dp(12));

        TextView hint = text("管理员功能：", 14, INK);
        hint.setPadding(0, 0, 0, dp(12));
        content.addView(hint, matchWrap(0, dp(8)));

        Button btnUserList = button("用户管理", false);
        btnUserList.setOnClickListener(v -> {
            showUserManagement();
        });
        content.addView(btnUserList, matchHeight(dp(46), 0, dp(8)));

        Button btnModelList = button("全部模型管理", false);
        btnModelList.setOnClickListener(v -> showModelManagement());
        content.addView(btnModelList, matchHeight(dp(46), 0, dp(8)));

        Button btnDataStats = button("数据统计", false);
        btnDataStats.setOnClickListener(v -> showDataStats());
        content.addView(btnDataStats, matchHeight(dp(46), 0, 0));

        new AlertDialog.Builder(this)
            .setTitle("管理员功能")
            .setView(content)
            .setPositiveButton("关闭", null)
            .show();
    }

    private final java.util.Set<String> selectedModelKeys = new java.util.HashSet<>();
    private final java.util.List<CheckBox> modelCheckboxes = new java.util.ArrayList<>();
    private CheckBox selectAllCheckBox;
    private TextView modelStatusText;
    private LinearLayout modelListContainer;

    /** 显示全部模型管理界面 */
    private void showModelManagement() {
        selectedModelKeys.clear();
        modelCheckboxes.clear();

        LinearLayout content = column();
        content.setPadding(dp(12), dp(6), dp(12), dp(12));

        LinearLayout topBar = new LinearLayout(this);
        topBar.setOrientation(LinearLayout.HORIZONTAL);
        topBar.setGravity(android.view.Gravity.CENTER_VERTICAL);

        selectAllCheckBox = new CheckBox(this);
        selectAllCheckBox.setText("全选");
        selectAllCheckBox.setTextColor(INK);
        selectAllCheckBox.setOnCheckedChangeListener((buttonView, isChecked) -> {
            for (CheckBox cb : modelCheckboxes) {
                cb.setChecked(isChecked);
            }
        });
        LinearLayout.LayoutParams selectAllParams = new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        selectAllParams.weight = 1;
        topBar.addView(selectAllCheckBox, selectAllParams);

        Button batchDeleteBtn = button("批量删除", false);
        batchDeleteBtn.setTextColor(0xFFFF6F85);
        batchDeleteBtn.setOnClickListener(v -> {
            if (selectedModelKeys.isEmpty()) {
                toast("请先勾选要删除的模型");
                return;
            }
            confirmBatchDeleteModels();
        });
        topBar.addView(batchDeleteBtn);

        content.addView(topBar, matchWrap(0, dp(4)));

        modelStatusText = text("正在加载模型列表...", 13, MUTED);
        modelStatusText.setPadding(0, dp(4), 0, dp(8));
        content.addView(modelStatusText, matchWrap(0, dp(4)));

        ScrollView scroll = new ScrollView(this);
        modelListContainer = column();
        scroll.addView(modelListContainer);
        content.addView(scroll, new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, dp(400)));

        AlertDialog dialog = new AlertDialog.Builder(this)
            .setTitle("全部模型管理")
            .setView(content)
            .setPositiveButton("刷新", null)
            .setNegativeButton("关闭", null)
            .create();

        dialog.setOnShowListener(d -> {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener(v -> {
                loadAllModels();
            });
        });

        dialog.show();
        loadAllModels();
    }

    /** 加载全部模型列表 */
    private void loadAllModels() {
        modelStatusText.setText("正在加载模型列表...");
        modelStatusText.setTextColor(ACCENT);
        modelListContainer.removeAllViews();
        selectedModelKeys.clear();
        modelCheckboxes.clear();
        if (selectAllCheckBox != null) selectAllCheckBox.setChecked(false);

        worker.execute(() -> {
            try {
                String serverUrl = getSharedPreferences("semg_prefs", MODE_PRIVATE)
                    .getString("server_url", "http://123.57.250.63:5000");
                if (serverUrl.endsWith("/")) serverUrl = serverUrl.substring(0, serverUrl.length() - 1);

                URL url = new URL(serverUrl + "/api/admin/models?admin_password=" + 
                    java.net.URLEncoder.encode("zhiwei8888", "UTF-8"));
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("GET");
                conn.setConnectTimeout(15000);
                conn.setReadTimeout(15000);

                if (conn.getResponseCode() == 200) {
                    StringBuilder sb = new StringBuilder();
                    try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), "UTF-8"))) {
                        String line;
                        while ((line = br.readLine()) != null) sb.append(line);
                    }
                    org.json.JSONObject json = new org.json.JSONObject(sb.toString());
                    org.json.JSONArray modelsArray = json.getJSONArray("models");
                    final int total = json.getInt("total");

                    main.post(() -> {
                        try {
                            modelStatusText.setText("共 " + total + " 个模型");
                            modelStatusText.setTextColor(INK);
                            modelListContainer.removeAllViews();

                            if (total == 0) {
                                TextView empty = text("暂无训练模型", 13, MUTED);
                                empty.setGravity(android.view.Gravity.CENTER);
                                modelListContainer.addView(empty, new LinearLayout.LayoutParams(
                                    ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
                                return;
                            }

                            for (int i = 0; i < modelsArray.length(); i++) {
                                org.json.JSONObject model = modelsArray.getJSONObject(i);
                                addModelListItem(model);
                            }
                        } catch (Exception e) {
                            modelStatusText.setText("解析数据失败: " + e.getMessage());
                            modelStatusText.setTextColor(0xFFFF8395);
                        }
                    });
                } else {
                    main.post(() -> {
                        modelStatusText.setText("加载失败，请检查权限");
                        modelStatusText.setTextColor(0xFFFF8395);
                    });
                }
            } catch (Exception e) {
                main.post(() -> {
                    modelStatusText.setText("加载失败: " + e.getMessage());
                    modelStatusText.setTextColor(0xFFFF8395);
                });
            }
        });
    }

    /** 添加模型列表项（带勾选框，统一训练模型+用户副本） */
    private void addModelListItem(org.json.JSONObject model) {
        try {
            String key = model.optString("key", "");
            String storage = model.optString("storage", "train");
            String modelId = model.optString("model_id", "unknown");
            boolean hasPt = model.optBoolean("has_pt", false);
            long ptSize = model.optLong("pt_size", 0);
            String sizeStr = ptSize > 0 ? String.format("%.2f MB", ptSize / (1024.0 * 1024.0)) : "未知";
            String owner = model.optString("owner", "未知用户");
            double accuracy = model.optDouble("accuracy", 0);
            String createdAt = model.optString("created_at", "");
            org.json.JSONArray labels = model.optJSONArray("labels");

            LinearLayout item = new LinearLayout(this);
            item.setOrientation(LinearLayout.HORIZONTAL);
            item.setBackground(rounded(0xFF0B151B, 8, 0xFF2A3A43));
            item.setPadding(dp(8), dp(8), dp(8), dp(8));
            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
            params.bottomMargin = dp(8);
            item.setLayoutParams(params);

            CheckBox cb = new CheckBox(this);
            cb.setOnCheckedChangeListener((buttonView, isChecked) -> {
                if (isChecked) selectedModelKeys.add(key);
                else selectedModelKeys.remove(key);
                updateSelectAllState();
            });
            LinearLayout.LayoutParams cbParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
            cbParams.gravity = android.view.Gravity.CENTER_VERTICAL;
            item.addView(cb, cbParams);
            modelCheckboxes.add(cb);

            LinearLayout infoCol = new LinearLayout(this);
            infoCol.setOrientation(LinearLayout.VERTICAL);
            LinearLayout.LayoutParams infoParams = new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1);
            infoCol.setLayoutParams(infoParams);

            LinearLayout nameRow = new LinearLayout(this);
            nameRow.setOrientation(LinearLayout.HORIZONTAL);
            nameRow.setGravity(android.view.Gravity.CENTER_VERTICAL);

            TextView nameText = text(modelId, 13, INK);
            nameText.setTypeface(Typeface.DEFAULT_BOLD);
            LinearLayout.LayoutParams nameParams = new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WRAP_CONTENT, 1);
            nameRow.addView(nameText, nameParams);

            TextView storageBadge;
            if ("train".equals(storage)) {
                storageBadge = text("训练", 10, 0xFF4ECDC4);
            } else {
                storageBadge = text("用户", 10, 0xFFFFA07A);
            }
            storageBadge.setBackground(rounded(0xFF1A2E36, 6, 0xFF4ECDC4));
            storageBadge.setPadding(dp(6), dp(2), dp(6), dp(2));
            nameRow.addView(storageBadge);

            infoCol.addView(nameRow);

            StringBuilder info = new StringBuilder();
            info.append(owner);
            info.append(" | ").append(sizeStr);
            if (accuracy > 0) info.append(" | 准确率: ").append(String.format("%.1f%%", accuracy));
            if (!hasPt) info.append(" | ⚠ PT缺失");

            TextView infoText = text(info.toString(), 11, MUTED);
            infoCol.addView(infoText);

            if (labels != null && labels.length() > 0) {
                StringBuilder lbStr = new StringBuilder();
                for (int i = 0; i < Math.min(labels.length(), 4); i++) {
                    if (i > 0) lbStr.append(", ");
                    lbStr.append(labels.getString(i));
                }
                if (labels.length() > 4) lbStr.append("...");
                TextView labelText = text(lbStr.toString(), 10, 0xFF8899AA);
                infoCol.addView(labelText);
            }

            if (!createdAt.isEmpty()) {
                String shortTime = createdAt.length() > 19 ? createdAt.substring(0, 19).replace("T", " ") : createdAt;
                TextView timeText = text(shortTime, 10, 0xFF5A7A8A);
                infoCol.addView(timeText);
            }

            item.addView(infoCol);
            modelListContainer.addView(item);
        } catch (Exception e) {
            // 忽略解析错误
        }
    }

    /** 更新全选框状态 */
    private void updateSelectAllState() {
        if (selectAllCheckBox == null) return;
        if (modelCheckboxes.isEmpty()) {
            selectAllCheckBox.setChecked(false);
            return;
        }
        boolean allChecked = true;
        for (CheckBox cb : modelCheckboxes) {
            if (!cb.isChecked()) {
                allChecked = false;
                break;
            }
        }
        selectAllCheckBox.setOnCheckedChangeListener(null);
        selectAllCheckBox.setChecked(allChecked);
        selectAllCheckBox.setOnCheckedChangeListener((buttonView, isChecked) -> {
            for (CheckBox cb : modelCheckboxes) {
                cb.setChecked(isChecked);
            }
        });
        modelStatusText.setText("共 " + modelCheckboxes.size() + " 个模型 | 已选 " + selectedModelKeys.size());
    }

    /** 确认批量删除 */
    private void confirmBatchDeleteModels() {
        new AlertDialog.Builder(this)
            .setTitle("确认批量删除")
            .setMessage("确定要删除选中的 " + selectedModelKeys.size() + " 个模型吗？\n\n此操作不可恢复！")
            .setPositiveButton("删除全部", (dialog, which) -> deleteModelsBatch())
            .setNegativeButton("取消", null)
            .show();
    }

    /** 批量删除模型 */
    private void deleteModelsBatch() {
        java.util.List<String> keys = new java.util.ArrayList<>(selectedModelKeys);
        worker.execute(() -> {
            try {
                String serverUrl = getSharedPreferences("semg_prefs", MODE_PRIVATE)
                    .getString("server_url", "http://123.57.250.63:5000");
                if (serverUrl.endsWith("/")) serverUrl = serverUrl.substring(0, serverUrl.length() - 1);

                URL url = new URL(serverUrl + "/api/admin/models/batch?admin_password=" + 
                    java.net.URLEncoder.encode("zhiwei8888", "UTF-8"));
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("DELETE");
                conn.setDoOutput(true);
                conn.setRequestProperty("Content-Type", "application/json");
                conn.setConnectTimeout(15000);
                conn.setReadTimeout(15000);

                org.json.JSONObject body = new org.json.JSONObject();
                body.put("keys", new org.json.JSONArray(keys));
                try (java.io.OutputStream os = conn.getOutputStream()) {
                    os.write(body.toString().getBytes("UTF-8"));
                }

                if (conn.getResponseCode() == 200) {
                    StringBuilder sb = new StringBuilder();
                    try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), "UTF-8"))) {
                        String line;
                        while ((line = br.readLine()) != null) sb.append(line);
                    }
                    org.json.JSONObject resp = new org.json.JSONObject(sb.toString());
                    final String msg = resp.optString("message", "删除成功");

                    main.post(() -> {
                        toast(msg);
                        loadAllModels();
                    });
                } else {
                    main.post(() -> toast("批量删除失败"));
                }
            } catch (Exception e) {
                main.post(() -> toast("批量删除失败: " + e.getMessage()));
            }
        });
    }

    /** 显示数据统计界面 */
    private void showDataStats() {
        LinearLayout content = column();
        content.setPadding(dp(12), dp(6), dp(12), dp(12));

        TextView statusText = text("正在加载统计数据...", 13, MUTED);
        statusText.setPadding(0, 0, 0, dp(8));
        content.addView(statusText, matchWrap(0, dp(8)));

        ScrollView scroll = new ScrollView(this);
        LinearLayout statsContainer = column();
        scroll.addView(statsContainer);
        content.addView(scroll, new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, dp(400)));

        AlertDialog dialog = new AlertDialog.Builder(this)
            .setTitle("数据统计")
            .setView(content)
            .setPositiveButton("刷新", null)
            .setNegativeButton("关闭", null)
            .create();

        dialog.setOnShowListener(d -> {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener(v -> {
                loadDataStats(statsContainer, statusText);
            });
        });

        dialog.show();
        loadDataStats(statsContainer, statusText);
    }

    /** 加载数据统计 */
    private void loadDataStats(LinearLayout container, TextView statusText) {
        statusText.setText("正在加载统计数据...");
        statusText.setTextColor(ACCENT);
        container.removeAllViews();

        worker.execute(() -> {
            try {
                String serverUrl = getSharedPreferences("semg_prefs", MODE_PRIVATE)
                    .getString("server_url", "http://123.57.250.63:5000");
                if (serverUrl.endsWith("/")) serverUrl = serverUrl.substring(0, serverUrl.length() - 1);

                URL url = new URL(serverUrl + "/api/admin/stats?admin_password=" + 
                    java.net.URLEncoder.encode("zhiwei8888", "UTF-8"));
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("GET");
                conn.setConnectTimeout(15000);
                conn.setReadTimeout(15000);

                if (conn.getResponseCode() == 200) {
                    StringBuilder sb = new StringBuilder();
                    try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), "UTF-8"))) {
                        String line;
                        while ((line = br.readLine()) != null) sb.append(line);
                    }
                    final org.json.JSONObject json = new org.json.JSONObject(sb.toString());

                    main.post(() -> {
                        try {
                            container.removeAllViews();
                            renderStats(container, json);
                            String serverTime = json.optString("server_time", "");
                            statusText.setText("服务器时间: " + serverTime);
                            statusText.setTextColor(MUTED);
                        } catch (Exception e) {
                            statusText.setText("解析数据失败: " + e.getMessage());
                            statusText.setTextColor(0xFFFF8395);
                        }
                    });
                } else {
                    main.post(() -> {
                        statusText.setText("加载失败，请检查权限");
                        statusText.setTextColor(0xFFFF8395);
                    });
                }
            } catch (Exception e) {
                main.post(() -> {
                    statusText.setText("加载失败: " + e.getMessage());
                    statusText.setTextColor(0xFFFF8395);
                });
            }
        });
    }

    /** 渲染统计卡片 */
    private void renderStats(LinearLayout container, org.json.JSONObject json) {
        org.json.JSONObject users = json.optJSONObject("users");
        org.json.JSONObject files = json.optJSONObject("files");
        org.json.JSONObject models = json.optJSONObject("models");

        int totalUsers = users != null ? users.optInt("total", 0) : 0;
        int activeUsers = users != null ? users.optInt("active", 0) : 0;
        int adminUsers = users != null ? users.optInt("admins", 0) : 0;
        int totalCsv = files != null ? files.optInt("total_csv", 0) : 0;
        int uploadDirs = files != null ? files.optInt("upload_dirs", 0) : 0;
        double uploadSize = files != null ? files.optDouble("upload_size_mb", 0) : 0;
        int totalModels = models != null ? models.optInt("total", 0) : 0;
        double modelSize = models != null ? models.optDouble("size_mb", 0) : 0;

        container.addView(buildStatCard("👥 用户统计", new String[][]{
            {"总用户数", String.valueOf(totalUsers)},
            {"活跃用户", String.valueOf(activeUsers)},
            {"管理员", String.valueOf(adminUsers)},
        }), matchWrap(0, dp(8)));

        container.addView(buildStatCard("📁 数据文件", new String[][]{
            {"用户目录", String.valueOf(uploadDirs)},
            {"CSV文件数", String.valueOf(totalCsv)},
            {"上传总大小", String.format("%.2f MB", uploadSize)},
        }), matchWrap(0, dp(8)));

        container.addView(buildStatCard("🧠 训练模型", new String[][]{
            {"模型总数", String.valueOf(totalModels)},
            {"模型总大小", String.format("%.2f MB", modelSize)},
        }), matchWrap(0, dp(8)));
    }

    /** 构建统计卡片 */
    private LinearLayout buildStatCard(String title, String[][] rows) {
        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setBackground(rounded(0xFF0B151B, 10, 0xFF2A3A43));
        card.setPadding(dp(14), dp(12), dp(14), dp(12));
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        params.bottomMargin = dp(8);
        card.setLayoutParams(params);

        TextView titleText = text(title, 14, ACCENT);
        titleText.setTypeface(Typeface.DEFAULT_BOLD);
        card.addView(titleText);

        for (String[] row : rows) {
            LinearLayout rowLayout = new LinearLayout(this);
            rowLayout.setOrientation(LinearLayout.HORIZONTAL);
            rowLayout.setPadding(0, dp(6), 0, 0);

            TextView label = text(row[0], 13, MUTED);
            rowLayout.addView(label, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));

            TextView value = text(row[1], 13, INK);
            value.setTypeface(Typeface.DEFAULT_BOLD);
            rowLayout.addView(value, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT));

            card.addView(rowLayout);
        }

        return card;
    }

    /** 显示用户管理界面 */
    private void showUserManagement() {
        LinearLayout content = column();
        content.setPadding(dp(12), dp(6), dp(12), dp(12));

        TextView statusText = text("正在加载用户列表...", 13, MUTED);
        statusText.setPadding(0, 0, 0, dp(8));
        content.addView(statusText, matchWrap(0, dp(8)));

        ScrollView scroll = new ScrollView(this);
        LinearLayout userList = column();
        scroll.addView(userList);
        content.addView(scroll, new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, dp(400)));

        AlertDialog dialog = new AlertDialog.Builder(this)
            .setTitle("用户管理")
            .setView(content)
            .setPositiveButton("刷新", null)
            .setNegativeButton("关闭", null)
            .create();

        dialog.setOnShowListener(d -> {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener(v -> {
                loadUserList(userList, statusText, dialog);
            });
        });

        dialog.show();
        loadUserList(userList, statusText, dialog);
    }

    /** 加载用户列表 */
    private void loadUserList(LinearLayout userList, TextView statusText, AlertDialog dialog) {
        statusText.setText("正在加载用户列表...");
        statusText.setTextColor(ACCENT);
        userList.removeAllViews();

        worker.execute(() -> {
            try {
                String serverUrl = getSharedPreferences("semg_prefs", MODE_PRIVATE)
                    .getString("server_url", "http://123.57.250.63:5000");
                if (serverUrl.endsWith("/")) serverUrl = serverUrl.substring(0, serverUrl.length() - 1);

                URL url = new URL(serverUrl + "/api/admin/users?admin_password=" + 
                    java.net.URLEncoder.encode("zhiwei8888", "UTF-8"));
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("GET");
                conn.setConnectTimeout(15000);
                conn.setReadTimeout(15000);

                if (conn.getResponseCode() == 200) {
                    StringBuilder sb = new StringBuilder();
                    try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), "UTF-8"))) {
                        String line;
                        while ((line = br.readLine()) != null) sb.append(line);
                    }
                    org.json.JSONObject json = new org.json.JSONObject(sb.toString());
                    org.json.JSONArray usersArray = json.getJSONArray("users");
                    final int total = json.getInt("total");

                    main.post(() -> {
                        try {
                            statusText.setText("共 " + total + " 个用户");
                            statusText.setTextColor(INK);
                            userList.removeAllViews();

                            for (int i = 0; i < usersArray.length(); i++) {
                                org.json.JSONObject user = usersArray.getJSONObject(i);
                                addUserListItem(userList, user);
                            }
                        } catch (Exception e) {
                            statusText.setText("解析数据失败: " + e.getMessage());
                            statusText.setTextColor(0xFFFF8395);
                        }
                    });
                } else {
                    main.post(() -> {
                        statusText.setText("加载失败，请检查权限");
                        statusText.setTextColor(0xFFFF8395);
                    });
                }
            } catch (Exception e) {
                main.post(() -> {
                    statusText.setText("加载失败: " + e.getMessage());
                    statusText.setTextColor(0xFFFF8395);
                });
            }
        });
    }

    /** 添加用户列表项 */
    private void addUserListItem(LinearLayout parent, org.json.JSONObject user) {
        try {
            LinearLayout item = new LinearLayout(this);
            item.setOrientation(LinearLayout.VERTICAL);
            item.setBackground(rounded(0xFF0B151B, 8, 0xFF2A3A43));
            item.setPadding(dp(12), dp(10), dp(12), dp(10));

            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
            params.bottomMargin = dp(8);
            item.setLayoutParams(params);

            String username = user.getString("username");
            boolean isAdmin = user.optBoolean("is_admin", false);
            String email = user.optString("email", "未设置");
            int fileCount = user.optInt("file_count", 0);
            int modelCount = user.optInt("model_count", 0);
            int userId = user.getInt("id");

            TextView nameText = text(username + (isAdmin ? " [管理员]" : ""), 14, 
                isAdmin ? 0xFFFFB74D : INK);
            nameText.setTypeface(Typeface.DEFAULT_BOLD);
            item.addView(nameText);

            TextView infoText = text("邮箱: " + email + " | 数据: " + fileCount + " | 模型: " + modelCount, 
                12, MUTED);
            item.addView(infoText, matchWrap(0, dp(4)));

            LinearLayout btnRow = new LinearLayout(this);
            btnRow.setOrientation(LinearLayout.HORIZONTAL);
            
            if (!isAdmin) {
                Button setAdminBtn = button("设为管理员", false);
                setAdminBtn.setTextColor(0xFFFFB74D);
                setAdminBtn.setOnClickListener(v -> {
                    toggleAdminStatus(userId, username, true, parent);
                });
                LinearLayout.LayoutParams params1 = new LinearLayout.LayoutParams(0, dp(36), 1);
                params1.rightMargin = dp(4);
                setAdminBtn.setLayoutParams(params1);
                btnRow.addView(setAdminBtn);
                
                Button deleteBtn = button("删除用户", false);
                deleteBtn.setTextColor(0xFFFF6F85);
                deleteBtn.setOnClickListener(v -> {
                    confirmDeleteUser(userId, username, parent);
                });
                LinearLayout.LayoutParams params2 = new LinearLayout.LayoutParams(0, dp(36), 1);
                deleteBtn.setLayoutParams(params2);
                btnRow.addView(deleteBtn);
            } else {
                Button cancelAdminBtn = button("取消管理员", false);
                cancelAdminBtn.setTextColor(0xFFFF8395);
                cancelAdminBtn.setOnClickListener(v -> {
                    toggleAdminStatus(userId, username, false, parent);
                });
                btnRow.addView(cancelAdminBtn, new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, dp(36)));
            }
            
            item.addView(btnRow, matchWrap(0, dp(4)));
            parent.addView(item);
        } catch (Exception e) {
            // 忽略解析错误
        }
    }

    /** 确认删除用户 */
    private void confirmDeleteUser(int userId, String username, LinearLayout parent) {
        new AlertDialog.Builder(this)
            .setTitle("确认删除")
            .setMessage("确定要删除用户 \"" + username + "\" 吗？\n\n这将删除该用户的所有数据，包括：\n• 所有采集数据\n• 所有训练模型\n• 用户账户信息\n\n此操作不可恢复！")
            .setPositiveButton("删除", (dialog, which) -> {
                deleteUser(userId, username, parent);
            })
            .setNegativeButton("取消", null)
            .show();
    }

    /** 切换用户管理员状态 */
    private void toggleAdminStatus(int userId, String username, boolean setAsAdmin, LinearLayout parent) {
        String action = setAsAdmin ? "设为管理员" : "取消管理员";
        new AlertDialog.Builder(this)
            .setTitle("确认操作")
            .setMessage("确定要" + action + "用户 \"" + username + "\" 吗？")
            .setPositiveButton("确定", (dialog, which) -> {
                setAdminStatus(userId, username, setAsAdmin, parent);
            })
            .setNegativeButton("取消", null)
            .show();
    }

    /** 设置用户管理员权限 */
    private void setAdminStatus(int userId, String username, boolean isAdmin, LinearLayout parent) {
        worker.execute(() -> {
            try {
                String serverUrl = getSharedPreferences("semg_prefs", MODE_PRIVATE)
                    .getString("server_url", "http://123.57.250.63:5000");
                if (serverUrl.endsWith("/")) serverUrl = serverUrl.substring(0, serverUrl.length() - 1);

                URL url = new URL(serverUrl + "/api/admin/set_admin");
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("POST");
                conn.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
                conn.setDoOutput(true);
                conn.setConnectTimeout(15000);
                conn.setReadTimeout(15000);

                String json = "{\"username\":\"" + username + "\",\"is_admin\":" + isAdmin + 
                    ",\"admin_password\":\"zhiwei8888\"}";
                try (java.io.OutputStream os = conn.getOutputStream()) {
                    os.write(json.getBytes("UTF-8"));
                    os.flush();
                }

                if (conn.getResponseCode() == 200) {
                    StringBuilder sb = new StringBuilder();
                    try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), "UTF-8"))) {
                        String line;
                        while ((line = br.readLine()) != null) sb.append(line);
                    }
                    org.json.JSONObject jsonResp = new org.json.JSONObject(sb.toString());
                    final String msg = jsonResp.optString("message", "操作成功");

                    main.post(() -> {
                        toast(msg);
                        // 重新加载用户列表
                        refreshUserList(parent);
                    });
                } else {
                    StringBuilder sb = new StringBuilder();
                    try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getErrorStream(), "UTF-8"))) {
                        String line;
                        while ((line = br.readLine()) != null) sb.append(line);
                    }
                    String errorMsg = "操作失败";
                    try {
                        org.json.JSONObject errJson = new org.json.JSONObject(sb.toString());
                        errorMsg = errJson.optString("error", errorMsg);
                    } catch (Exception ignored) {}
                    final String finalErrorMsg = errorMsg;
                    main.post(() -> toast(finalErrorMsg));
                }
            } catch (Exception e) {
                main.post(() -> toast("操作失败: " + e.getMessage()));
            }
        });
    }

    /** 刷新用户列表 */
    private void refreshUserList(LinearLayout parent) {
        try {
            View grandParent = (View) parent.getParent();
            if (grandParent instanceof LinearLayout) {
                LinearLayout container = (LinearLayout) grandParent;
                for (int i = 0; i < container.getChildCount(); i++) {
                    View child = container.getChildAt(i);
                    if (child instanceof TextView) {
                        TextView statusText = (TextView) child;
                        loadUserList((LinearLayout) parent, statusText, null);
                        break;
                    }
                }
            }
        } catch (Exception e) {
            // 忽略刷新失败
        }
    }

    /** 删除用户 */
    private void deleteUser(int userId, String username, LinearLayout parent) {
        worker.execute(() -> {
            try {
                String serverUrl = getSharedPreferences("semg_prefs", MODE_PRIVATE)
                    .getString("server_url", "http://123.57.250.63:5000");
                if (serverUrl.endsWith("/")) serverUrl = serverUrl.substring(0, serverUrl.length() - 1);

                URL url = new URL(serverUrl + "/api/admin/users/" + userId + 
                    "?admin_password=" + java.net.URLEncoder.encode("zhiwei8888", "UTF-8"));
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("DELETE");
                conn.setConnectTimeout(15000);
                conn.setReadTimeout(15000);

                if (conn.getResponseCode() == 200) {
                    StringBuilder sb = new StringBuilder();
                    try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), "UTF-8"))) {
                        String line;
                        while ((line = br.readLine()) != null) sb.append(line);
                    }
                    org.json.JSONObject json = new org.json.JSONObject(sb.toString());
                    final String msg = json.optString("message", "删除成功");

                    main.post(() -> {
                        toast(msg);
                        // 重新加载用户列表
                        TextView statusText = null;
                        for (int i = 0; i < ((LinearLayout)parent.getParent()).getChildCount(); i++) {
                            View child = ((LinearLayout)parent.getParent()).getChildAt(i);
                            if (child instanceof TextView) {
                                statusText = (TextView) child;
                                break;
                            }
                        }
                        if (statusText != null) {
                            loadUserList((LinearLayout) parent, statusText, null);
                        }
                    });
                } else {
                    StringBuilder sb = new StringBuilder();
                    try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getErrorStream(), "UTF-8"))) {
                        String line;
                        while ((line = br.readLine()) != null) sb.append(line);
                    }
                    String errorMsg = "删除失败";
                    try {
                        org.json.JSONObject errJson = new org.json.JSONObject(sb.toString());
                        errorMsg = errJson.optString("error", errorMsg);
                    } catch (Exception ignored) {}
                    final String finalErrorMsg = errorMsg;
                    main.post(() -> toast(finalErrorMsg));
                }
            } catch (Exception e) {
                main.post(() -> toast("删除失败: " + e.getMessage()));
            }
        });
    }

    /** 更新顶部用户按钮显示 */
    private void updateUserButton() {
        if (userBtn == null) return;
        main.post(() -> {
            userBtn.setText(authToken != null ? currentUsername : "登录");
            userBtn.setTextColor(authToken != null ? ACCENT : MUTED);
        });
    }

    /** 云端训练对话框：上传数据 → 触发训练 → 轮询状态 → 下载模型 */
    private void showCloudTrainDialog() {
        if (recording != null) {
            toast("请先停止当前采集");
            return;
        }

        // 检查是否有采集数据
        List<RecordingStore.RecordingInfo> recordings = trainableRecordings(store.listRecordings());
        if (recordings.isEmpty()) {
            toast("暂无采集数据，请先采集数据");
            return;
        }

        // 已登录直接训练，未登录才弹登录框
        if (authToken != null) {
            performCloudTrainWithAuth();
        } else {
            showLoginDialog(() -> performCloudTrainWithAuth());
        }
    }

    /** 执行云端训练（已登录） */
    private void performCloudTrainWithAuth() {
        List<RecordingStore.RecordingInfo> recordings = trainableRecordings(store.listRecordings());
        if (recordings.isEmpty()) {
            toast("暂无采集数据");
            return;
        }

        // 获取服务器地址
        android.content.SharedPreferences prefs = getSharedPreferences("semg_prefs", MODE_PRIVATE);
        String serverUrl = prefs.getString("server_url", "http://123.57.250.63:5000");

        LinearLayout content = column();
        content.setPadding(dp(12), dp(6), dp(12), dp(12));

        // 显示当前用户信息
        TextView userInfo = text("当前用户: " + currentUsername, 13, ACCENT);
        userInfo.setPadding(0, 0, 0, dp(8));
        content.addView(userInfo, matchWrap(0, dp(4)));

        // 模型类型选择
        TextView modelLabel = text("模型类型:", 13, INK);
        modelLabel.setPadding(0, dp(4), 0, dp(4));
        content.addView(modelLabel, matchWrap(0, dp(4)));

        Spinner modelSpinner = new Spinner(this);
        ArrayAdapter<String> modelAdapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item,
                new String[]{"CNN-1D（基础）", "CNN-Transformer（元学习）"});
        modelAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        modelSpinner.setAdapter(modelAdapter);
        modelSpinner.setPadding(0, 0, 0, dp(8));
        content.addView(modelSpinner, matchWrap(0, dp(4)));

        // 模型类型说明
        TextView modelDesc = text("CNN-1D：适合单用户、数据量少\nCNN-Transformer：两阶段元学习，首次训练基础模型，后续用户直接微调，数据越多越好", 11, MUTED);
        modelDesc.setPadding(0, 0, 0, dp(8));
        content.addView(modelDesc, matchWrap(0, dp(4)));

        // 基础模型状态
        TextView baseModelStatus = text("基础模型状态: 检查中...", 11, MUTED);
        baseModelStatus.setPadding(0, 0, 0, dp(4));
        content.addView(baseModelStatus, matchWrap(0, dp(4)));

        Button retrainBtn = new Button(this);
        retrainBtn.setText("重新训练基础模型");
        retrainBtn.setTextSize(11);
        retrainBtn.setPadding(0, 0, 0, dp(8));
        retrainBtn.setVisibility(android.view.View.GONE);
        content.addView(retrainBtn, matchWrap(0, dp(4)));

        // 零样本下载按钮：新用户无需采集数据，直接下载基础模型使用
        Button zeroShotBtn = new Button(this);
        zeroShotBtn.setText("零样本下载（基础模型）");
        zeroShotBtn.setTextSize(11);
        zeroShotBtn.setPadding(0, 0, 0, dp(8));
        zeroShotBtn.setVisibility(android.view.View.GONE);
        content.addView(zeroShotBtn, matchWrap(0, dp(4)));

        // 异步查询基础模型状态
        worker.execute(() -> {
            try {
                URL url = new URL(serverUrl + "/api/base_model_status");
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("GET");
                conn.setRequestProperty("Authorization", "Bearer " + authToken);
                conn.setConnectTimeout(10000);
                conn.setReadTimeout(10000);
                
                int code = conn.getResponseCode();
                if (code == 200) {
                    try (BufferedReader br = new BufferedReader(
                            new InputStreamReader(conn.getInputStream(), "UTF-8"))) {
                        StringBuilder sb = new StringBuilder();
                        String line;
                        while ((line = br.readLine()) != null) sb.append(line);
                        String json = sb.toString();
                        
                        boolean hasBase = json.contains("\"has_base_model\":true");
                        boolean needsRetrain = json.contains("\"needs_retrain\":true");
                        
                        main.post(() -> {
                            if (hasBase) {
                                baseModelStatus.setText("基础模型状态: 已就绪" + 
                                    (needsRetrain ? "（有新用户，建议重新训练）" : ""));
                                baseModelStatus.setTextColor(needsRetrain ? 0xFFFFA500 : 0xFF4CAF50);
                                // 基础模型已就绪时开放零样本下载
                                zeroShotBtn.setVisibility(android.view.View.VISIBLE);
                                if (needsRetrain) {
                                    retrainBtn.setVisibility(android.view.View.VISIBLE);
                                }
                            } else {
                                baseModelStatus.setText("基础模型状态: 首次训练将创建");
                                baseModelStatus.setTextColor(0xFFFFA500);
                            }
                        });
                    }
                }
            } catch (Exception e) {
                main.post(() -> {
                    baseModelStatus.setText("基础模型状态: 查询失败");
                    baseModelStatus.setTextColor(0xFFFF8395);
                });
            }
        });

        // 重新训练按钮点击事件
        retrainBtn.setOnClickListener(v -> {
            retrainBtn.setEnabled(false);
            retrainBtn.setText("重新训练中...");
            worker.execute(() -> {
                try {
                    URL url = new URL(serverUrl + "/api/retrain_base_model");
                    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                    conn.setRequestMethod("POST");
                    conn.setRequestProperty("Authorization", "Bearer " + authToken);
                    conn.setRequestProperty("Content-Type", "application/json");
                    conn.setConnectTimeout(10000);
                    conn.setReadTimeout(10000);

                    int code = conn.getResponseCode();
                    if (code == 200) {
                        StringBuilder sb = new StringBuilder();
                        try (BufferedReader br = new BufferedReader(
                                new InputStreamReader(conn.getInputStream(), "UTF-8"))) {
                            String line;
                            while ((line = br.readLine()) != null) sb.append(line);
                        }
                        String json = sb.toString();
                        String taskId = jsonField(json, "task_id");
                        main.post(() -> {
                            toast("基础模型重新训练已启动");
                            retrainBtn.setVisibility(android.view.View.GONE);
                            baseModelStatus.setText("基础模型状态: 重新训练中...");
                            baseModelStatus.setTextColor(0xFFFFA500);
                        });
                        // 轮询训练状态，完成后更新 UI
                        if (taskId != null) {
                            pollTrainingStatusWithToken(serverUrl, taskId, baseModelStatus);
                            main.post(() -> {
                                baseModelStatus.setText("基础模型状态: 已就绪");
                                baseModelStatus.setTextColor(0xFF4CAF50);
                                toast("基础模型训练完成");
                            });
                        }
                    } else if (code == 409) {
                        main.post(() -> {
                            toast("基础模型正在训练中，请稍候");
                            retrainBtn.setEnabled(true);
                            retrainBtn.setText("重新训练基础模型");
                        });
                    } else {
                        main.post(() -> {
                            toast("重新训练启动失败 (" + code + ")");
                            retrainBtn.setEnabled(true);
                            retrainBtn.setText("重新训练基础模型");
                        });
                    }
                } catch (Exception e) {
                    main.post(() -> {
                        String msg = e.getMessage();
                        if (msg != null && msg.startsWith("服务端训练失败：")) {
                            baseModelStatus.setText("基础模型状态: 训练失败");
                            baseModelStatus.setTextColor(0xFFFF8395);
                        }
                        toast("重新训练失败: " + msg);
                        retrainBtn.setEnabled(true);
                        retrainBtn.setText("重新训练基础模型");
                    });
                }
            });
        });

        // 零样本下载按钮点击事件
        zeroShotBtn.setOnClickListener(v -> {
            zeroShotBtn.setEnabled(false);
            zeroShotBtn.setText("零样本模型下载中...");
            worker.execute(() -> {
                try {
                    File ptFile = downloadCloudModelWithToken(serverUrl, "base_model");
                    if (ptFile == null) {
                        main.post(() -> {
                            toast("零样本模型下载失败，请确认基础模型已训练完成");
                            zeroShotBtn.setEnabled(true);
                            zeroShotBtn.setText("零样本下载（基础模型）");
                        });
                        return;
                    }
                    Map<Integer, String> mapping = new LinkedHashMap<>();
                    if (cloudModelLabels.isEmpty()) {
                        mapping.put(0, "休息");
                        mapping.put(1, "握拳");
                        mapping.put(2, "手掌");
                        mapping.put(3, "向右");
                        mapping.put(4, "向左");
                    } else {
                        for (int i = 0; i < cloudModelLabels.size(); i++) {
                            mapping.put(i, cloudModelLabels.get(i));
                        }
                    }
                    RecordingStore.PyTorchModelInfo info = store.importPytorchModel(
                            ptFile, "零样本基础模型_" + new SimpleDateFormat("MM-dd HH:mm", Locale.CHINA).format(new Date()),
                            mapping, parser.getProtocol(), cloudNormMeans, cloudNormStds,
                            cloudFilterB, cloudFilterA, cloudFilterZi);
                    PyTorchModel loaded = PyTorchModel.load(info.ptFile, info.labelMapping,
                            info.protocol, info.name, info.createdAt, info.normMeans, info.normStds,
                            info.filterB, info.filterA, info.filterZi);
                    pytorchModel = loaded;
                    model = null;

                    main.post(() -> {
                        stopRecognitionForModelChange();
                        updateModelText();
                        toast("零样本模型已激活，可直接识别");
                        zeroShotBtn.setEnabled(true);
                        zeroShotBtn.setText("零样本下载（基础模型）");
                    });
                } catch (Exception e) {
                    main.post(() -> {
                        toast("零样本模型导入失败: " + e.getMessage());
                        zeroShotBtn.setEnabled(true);
                        zeroShotBtn.setText("零样本下载（基础模型）");
                    });
                }
            });
        });

        // 状态显示
        TextView statusView = text("准备上传数据...", 12, MUTED);
        statusView.setPadding(0, dp(8), 0, dp(8));
        content.addView(statusView, matchWrap(0, dp(4)));

        final AlertDialog dialog = new AlertDialog.Builder(this)
                .setTitle("云端训练")
                .setView(content)
                .setPositiveButton("开始训练", null)
                .setNegativeButton("取消", null)
                .create();

        dialog.setOnShowListener(d -> {
            Button positive = dialog.getButton(AlertDialog.BUTTON_POSITIVE);
            positive.setOnClickListener(v -> {
                if (serverUrl.isEmpty()) {
                    toast("服务器地址未配置");
                    return;
                }

                // 获取选择的模型类型
                String modelType = modelSpinner.getSelectedItemPosition() == 0 ? "cnn1d" : "cnn_transformer";

                // 禁用按钮，开始云端训练流程
                positive.setEnabled(false);
                statusView.setText("正在上传数据...");
                statusView.setTextColor(ACCENT);

                // 在后台线程执行云端训练
                final String finalServerUrl = serverUrl;
                final String finalModelType = modelType;
                worker.execute(() -> performCloudTrainWithToken(finalServerUrl, recordings, statusView, dialog, finalModelType));
            });
        });

        dialog.show();
    }

    /** 执行云端训练（使用 Token 认证）：增量上传 → 训练 → 下载 → 导入 */
    private void performCloudTrainWithToken(String serverUrl,
                                            List<RecordingStore.RecordingInfo> recordings,
                                            TextView statusView, AlertDialog dialog, String modelType) {
        final String finalServerUrl = serverUrl;
        try {
            // 1. 增量上传数据（先查询云端文件，再上传新文件）
            main.post(() -> statusView.setText("正在查询云端已有文件..."));
            Set<String> cloudFiles = queryCloudFilesWithToken(serverUrl);
            int skippedCount = 0;
            List<RecordingStore.RecordingInfo> filesToUpload = new ArrayList<>();
            
            if (cloudFiles != null) {
                for (RecordingStore.RecordingInfo rec : recordings) {
                    File csvFile = rec.file;
                    if (csvFile == null || !csvFile.exists() || !csvFile.getName().endsWith(".csv")) continue;
                    
                    if (cloudFiles.contains(csvFile.getName())) {
                        skippedCount++;
                        continue;
                    }
                    filesToUpload.add(rec);
                }
            } else {
                filesToUpload.addAll(recordings);
            }

            final int finalSkippedCount = skippedCount;
            if (!filesToUpload.isEmpty()) {
                main.post(() -> statusView.setText("正在上传 " + filesToUpload.size() + " 个新文件（跳过 " + finalSkippedCount + " 个已存在）..."));
                String uploadResult = uploadDataToCloudWithToken(serverUrl, filesToUpload);
                if (uploadResult == null) {
                    main.post(() -> {
                        statusView.setText("上传失败");
                        statusView.setTextColor(0xFFFF8395);
                        dialog.getButton(AlertDialog.BUTTON_POSITIVE).setEnabled(true);
                    });
                    return;
                }
            } else {
                main.post(() -> statusView.setText("所有文件已在云端，直接开始训练..."));
            }

            // 2. 触发训练
            main.post(() -> statusView.setText("触发训练任务..."));
            String taskId = startCloudTrainingWithToken(serverUrl, modelType);
            if (taskId == null) {
                main.post(() -> {
                    statusView.setText("触发训练失败");
                    statusView.setTextColor(0xFFFF8395);
                    dialog.getButton(AlertDialog.BUTTON_POSITIVE).setEnabled(true);
                });
                return;
            }

            // 3. 轮询训练状态
            main.post(() -> statusView.setText("训练中，请稍候..."));
            String modelId = pollTrainingStatusWithToken(serverUrl, taskId, statusView);
            if (modelId == null) {
                main.post(() -> {
                    statusView.setText("训练失败");
                    statusView.setTextColor(0xFFFF8395);
                    dialog.getButton(AlertDialog.BUTTON_POSITIVE).setEnabled(true);
                });
                return;
            }

            // 4. 下载模型
            main.post(() -> statusView.setText("下载模型..."));
            File ptFile = downloadCloudModelWithToken(serverUrl, modelId);
            if (ptFile == null) {
                main.post(() -> {
                    statusView.setText("下载失败");
                    statusView.setTextColor(0xFFFF8395);
                    dialog.getButton(AlertDialog.BUTTON_POSITIVE).setEnabled(true);
                });
                return;
            }

            // 5. 导入模型（类别顺序以服务端返回的 X-Model-Labels 为准）
            main.post(() -> statusView.setText("导入模型..."));
            Map<Integer, String> mapping = new LinkedHashMap<>();
            if (cloudModelLabels.isEmpty()) {
                // 服务端未返回标签时的兜底默认顺序（5类：休息 + 4个动作）
                mapping.put(0, "休息");
                mapping.put(1, "握拳");
                mapping.put(2, "手掌");
                mapping.put(3, "向右");
                mapping.put(4, "向左");
            } else {
                for (int i = 0; i < cloudModelLabels.size(); i++) {
                    mapping.put(i, cloudModelLabels.get(i));
                }
            }

            RecordingStore.PyTorchModelInfo info = store.importPytorchModel(
                    ptFile, "云端模型_" + new SimpleDateFormat("MM-dd HH:mm", Locale.CHINA).format(new Date()),
                    mapping, parser.getProtocol(), cloudNormMeans, cloudNormStds,
                    cloudFilterB, cloudFilterA, cloudFilterZi);
            PyTorchModel loaded = PyTorchModel.load(info.ptFile, info.labelMapping,
                    info.protocol, info.name, info.createdAt, info.normMeans, info.normStds,
                    info.filterB, info.filterA, info.filterZi);
            pytorchModel = loaded;
            model = null;

            // 6. 询问用户是否保存模型到云端
            final String finalModelId = modelId;
            main.post(() -> {
                stopRecognitionForModelChange();
                updateModelText();
                dialog.dismiss();
                
                // 弹出对话框询问是否保存模型
                new AlertDialog.Builder(MainActivity.this)
                    .setTitle("训练完成")
                    .setMessage("模型已激活。是否保存此模型到云端？\n保存后可随时下载复用。")
                    .setPositiveButton("保存", (d, which) -> {
                        worker.execute(() -> saveModelToCloudWithToken(finalServerUrl, finalModelId));
                    })
                    .setNegativeButton("不保存", null)
                    .show();
            });

        } catch (Exception error) {
            final String errMsg = error.getMessage() != null ? error.getMessage() : error.toString();
            main.post(() -> {
                statusView.setText("错误: " + errMsg);
                statusView.setTextColor(0xFFFF8395);
                dialog.getButton(AlertDialog.BUTTON_POSITIVE).setEnabled(true);
            });
        }
    }

    /** 保存模型到云端用户文件夹（使用 Token 认证） */
    private void saveModelToCloudWithToken(String serverUrl, String modelId) {
        try {
            if (serverUrl.endsWith("/")) serverUrl = serverUrl.substring(0, serverUrl.length() - 1);
            URL url = new URL(serverUrl + "/api/save_user_model");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
            conn.setRequestProperty("Authorization", "Bearer " + authToken);
            conn.setDoOutput(true);
            conn.setConnectTimeout(15000);
            conn.setReadTimeout(30000);

            String json = "{\"model_id\":\"" + modelId + "\"}";
            try (java.io.OutputStream os = conn.getOutputStream()) {
                os.write(json.getBytes("UTF-8"));
                os.flush();
            }

            if (conn.getResponseCode() == 200) {
                main.post(() -> toast("模型已保存到云端"));
            } else {
                main.post(() -> toast("保存失败"));
            }
        } catch (Exception e) {
            main.post(() -> toast("保存失败: " + e.getMessage()));
        }
    }

    /** 下载我的模型对话框 */
    private void showDownloadMyModelDialog() {
        if (recording != null) {
            toast("请先停止当前采集");
            return;
        }

        // 先检查登录状态
        showLoginDialog(() -> performDownloadMyModelWithAuth());
    }

    /** 执行下载我的模型（已登录） */
    private void performDownloadMyModelWithAuth() {
        // 获取服务器地址
        android.content.SharedPreferences prefs = getSharedPreferences("semg_prefs", MODE_PRIVATE);
        String serverUrl = prefs.getString("server_url", "http://123.57.250.63:5000");

        LinearLayout content = column();
        content.setPadding(dp(12), dp(6), dp(12), dp(12));

        // 显示当前用户信息
        TextView userInfo = text("当前用户: " + currentUsername, 13, ACCENT);
        userInfo.setPadding(0, 0, 0, dp(8));
        content.addView(userInfo, matchWrap(0, dp(4)));

        // 状态显示
        TextView statusView = text("准备查询模型...", 12, MUTED);
        statusView.setPadding(0, dp(8), 0, dp(8));
        content.addView(statusView, matchWrap(0, dp(4)));

        // 模型列表显示区域
        LinearLayout modelListLayout = column();
        modelListLayout.setPadding(0, dp(4), 0, 0);
        content.addView(modelListLayout, matchHeight(dp(200), 0, dp(4)));

        final AlertDialog dialog = new AlertDialog.Builder(this)
                .setTitle("下载我的模型")
                .setView(content)
                .setPositiveButton("查询模型", null)
                .setNegativeButton("取消", null)
                .create();

        dialog.setOnShowListener(d -> {
            Button positive = dialog.getButton(AlertDialog.BUTTON_POSITIVE);
            positive.setOnClickListener(v -> {
                if (serverUrl.isEmpty()) {
                    toast("服务器地址未配置");
                    return;
                }

                positive.setEnabled(false);
                statusView.setText("正在查询模型...");
                statusView.setTextColor(ACCENT);
                modelListLayout.removeAllViews();

                final String finalServerUrl = serverUrl;
                worker.execute(() -> queryUserModelsWithToken(finalServerUrl, statusView, modelListLayout, dialog, positive));
            });
        });

        dialog.show();
    }

    /** 查询用户保存的模型列表（使用 Token 认证） */
    private void queryUserModelsWithToken(String serverUrl, TextView statusView,
                                          LinearLayout modelListLayout, AlertDialog dialog, Button queryButton) {
        final String finalServerUrl = serverUrl;
        try {
            if (serverUrl.endsWith("/")) serverUrl = serverUrl.substring(0, serverUrl.length() - 1);
            URL url = new URL(serverUrl + "/api/user_models");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setRequestProperty("Authorization", "Bearer " + authToken);
            conn.setConnectTimeout(10000);
            conn.setReadTimeout(10000);

            if (conn.getResponseCode() != 200) {
                main.post(() -> {
                    statusView.setText("查询失败");
                    statusView.setTextColor(0xFFFF8395);
                    queryButton.setEnabled(true);
                });
                return;
            }

            StringBuilder sb = new StringBuilder();
            try (BufferedReader br = new BufferedReader(
                    new InputStreamReader(conn.getInputStream(), "UTF-8"))) {
                String line;
                while ((line = br.readLine()) != null) sb.append(line);
            }
            String json = sb.toString();

            // 解析模型列表
            org.json.JSONObject jsonObj = new org.json.JSONObject(json);
            org.json.JSONArray modelsArr = jsonObj.optJSONArray("models");

            if (modelsArr == null || modelsArr.length() == 0) {
                main.post(() -> {
                    statusView.setText("未找到保存的模型");
                    statusView.setTextColor(MUTED);
                    queryButton.setEnabled(true);
                });
                return;
            }

            // 在主线程更新 UI
            main.post(() -> {
                statusView.setText("找到 " + modelsArr.length() + " 个模型，请选择下载:");
                statusView.setTextColor(ACCENT);

                for (int i = 0; i < modelsArr.length(); i++) {
                    org.json.JSONObject modelObj = modelsArr.optJSONObject(i);
                    if (modelObj == null) continue;

                    String modelName = modelObj.optString("name", "未知");
                    String timestamp = modelObj.optString("timestamp", "");
                    double accuracy = modelObj.optDouble("accuracy", 0.0);
                    org.json.JSONArray labelsArr = modelObj.optJSONArray("labels");

                    // 格式化时间戳
                    String timeStr = "";
                    if (!timestamp.isEmpty()) {
                        try {
                            java.util.Date date = java.util.Date.from(
                                java.time.Instant.parse(timestamp)
                            );
                            timeStr = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.CHINA).format(date);
                        } catch (Exception e) {
                            timeStr = timestamp;
                        }
                    }

                    // 格式化标签
                    String labelsStr = "";
                    if (labelsArr != null) {
                        List<String> labels = new ArrayList<>();
                        for (int j = 0; j < labelsArr.length(); j++) {
                            labels.add(labelsArr.optString(j, ""));
                        }
                        labelsStr = TextUtils.join("/", labels);
                    }

                    // 创建模型项按钮
                    Button modelBtn = new Button(this);
                    String btnText = "模型: " + modelName + "\n" +
                                    "时间: " + timeStr + "\n" +
                                    "准确率: " + String.format("%.1f%%", accuracy) + "\n" +
                                    "类别: " + labelsStr;
                    modelBtn.setText(btnText);
                    modelBtn.setTextSize(11);
                    modelBtn.setTextColor(INK);
                    modelBtn.setGravity(android.view.Gravity.START | android.view.Gravity.CENTER_VERTICAL);
                    modelBtn.setPadding(dp(10), dp(8), dp(10), dp(8));
                    modelBtn.setBackground(rounded(0xFF0B151B, 6, 0xFF2A3A43));
                    
                    LinearLayout.LayoutParams btnParams = new LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
                    btnParams.setMargins(0, dp(4), 0, dp(4));
                    modelBtn.setLayoutParams(btnParams);

                    final String finalModelName = modelName;
                    modelBtn.setOnClickListener(btn -> {
                        modelBtn.setEnabled(false);
                        modelBtn.setTextColor(MUTED);
                        statusView.setText("正在下载模型: " + finalModelName + "...");
                        worker.execute(() -> downloadUserModelWithToken(finalServerUrl, finalModelName, statusView, dialog));
                    });

                    modelListLayout.addView(modelBtn);
                }

                queryButton.setEnabled(true);
                queryButton.setText("重新查询");
            });

        } catch (Exception e) {
            main.post(() -> {
                statusView.setText("查询失败: " + e.getMessage());
                statusView.setTextColor(0xFFFF8395);
                queryButton.setEnabled(true);
            });
        }
    }

    /** 下载用户保存的模型（使用 Token 认证） */
    private void downloadUserModelWithToken(String serverUrl, String modelName,
                                            TextView statusView, AlertDialog dialog) {
        try {
            if (serverUrl.endsWith("/")) serverUrl = serverUrl.substring(0, serverUrl.length() - 1);
            URL url = new URL(serverUrl + "/api/download_user_model/" + modelName);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setRequestProperty("Authorization", "Bearer " + authToken);
            conn.setConnectTimeout(30000);
            conn.setReadTimeout(180000);

            if (conn.getResponseCode() != 200) {
                main.post(() -> {
                    statusView.setText("下载失败");
                    statusView.setTextColor(0xFFFF8395);
                });
                return;
            }

            // 解析响应头
            cloudModelLabels.clear();
            String header = conn.getHeaderField("X-Model-Labels");
            if (header != null && !header.isEmpty()) {
                String decoded = URLDecoder.decode(header, "UTF-8");
                for (String item : decoded.split(",")) {
                    if (!item.trim().isEmpty()) cloudModelLabels.add(item.trim());
                }
            }

            cloudNormMeans = null;
            cloudNormStds = null;
            String normHeader = conn.getHeaderField("X-Model-NormStats");
            if (normHeader != null && !normHeader.isEmpty()) {
                try {
                    String decodedNorm = URLDecoder.decode(normHeader, "UTF-8");
                    JSONObject normJson = new JSONObject(decodedNorm);
                    JSONArray meansArr = normJson.getJSONArray("means");
                    JSONArray stdsArr = normJson.getJSONArray("stds");
                    cloudNormMeans = new float[meansArr.length()];
                    cloudNormStds = new float[stdsArr.length()];
                    for (int i = 0; i < meansArr.length(); i++) {
                        cloudNormMeans[i] = (float) meansArr.getDouble(i);
                        cloudNormStds[i] = (float) stdsArr.getDouble(i);
                    }
                } catch (Exception ignored) { }
            }

            cloudFilterB = null;
            cloudFilterA = null;
            cloudFilterZi = null;
            String filterHeader = conn.getHeaderField("X-Model-FilterCoeffs");
            if (filterHeader != null && !filterHeader.isEmpty()) {
                try {
                    String decodedFilter = URLDecoder.decode(filterHeader, "UTF-8");
                    JSONObject filterJson = new JSONObject(decodedFilter);
                    JSONArray bArr = filterJson.getJSONArray("b");
                    JSONArray aArr = filterJson.getJSONArray("a");
                    JSONArray ziArr = filterJson.optJSONArray("zi");
                    cloudFilterB = new double[bArr.length()];
                    cloudFilterA = new double[aArr.length()];
                    for (int i = 0; i < bArr.length(); i++) cloudFilterB[i] = bArr.getDouble(i);
                    for (int i = 0; i < aArr.length(); i++) cloudFilterA[i] = aArr.getDouble(i);
                    if (ziArr != null) {
                        cloudFilterZi = new double[ziArr.length()];
                        for (int i = 0; i < ziArr.length(); i++) cloudFilterZi[i] = ziArr.getDouble(i);
                    }
                } catch (Exception ignored) { }
            }

            // 保存文件
            File ptFile = new File(getCacheDir(), modelName + ".pt");
            try (InputStream is = conn.getInputStream();
                 FileOutputStream fos = new FileOutputStream(ptFile)) {
                byte[] buffer = new byte[8192];
                int len;
                while ((len = is.read(buffer)) != -1) {
                    fos.write(buffer, 0, len);
                }
            }

            // 导入模型
            Map<Integer, String> mapping = new LinkedHashMap<>();
            if (cloudModelLabels.isEmpty()) {
                mapping.put(0, "休息");
                mapping.put(1, "握拳");
                mapping.put(2, "手掌");
                mapping.put(3, "向右");
                mapping.put(4, "向左");
            } else {
                for (int i = 0; i < cloudModelLabels.size(); i++) {
                    mapping.put(i, cloudModelLabels.get(i));
                }
            }

            RecordingStore.PyTorchModelInfo info = store.importPytorchModel(
                    ptFile, "我的模型_" + modelName,
                    mapping, parser.getProtocol(), cloudNormMeans, cloudNormStds,
                    cloudFilterB, cloudFilterA, cloudFilterZi);
            
            // 调试日志：验证系数是否正确传递
            android.util.Log.d("PyTorchModel", "=== 模型加载调试 ===");
            android.util.Log.d("PyTorchModel", "归一化均值: " + java.util.Arrays.toString(cloudNormMeans));
            android.util.Log.d("PyTorchModel", "归一化标准差: " + java.util.Arrays.toString(cloudNormStds));
            android.util.Log.d("PyTorchModel", "滤波器系数 b: " + (cloudFilterB != null ? java.util.Arrays.toString(cloudFilterB) : "NULL"));
            android.util.Log.d("PyTorchModel", "滤波器系数 a: " + (cloudFilterA != null ? java.util.Arrays.toString(cloudFilterA) : "NULL"));
            android.util.Log.d("PyTorchModel", "滤波器系数 zi: " + (cloudFilterZi != null ? java.util.Arrays.toString(cloudFilterZi) : "NULL"));
            
            PyTorchModel loaded = PyTorchModel.load(info.ptFile, info.labelMapping,
                    info.protocol, info.name, info.createdAt, info.normMeans, info.normStds,
                    info.filterB, info.filterA, info.filterZi);
            pytorchModel = loaded;
            model = null;

            main.post(() -> {
                stopRecognitionForModelChange();
                updateModelText();
                toast("模型下载完成，已激活");
                dialog.dismiss();
            });

        } catch (Exception e) {
            main.post(() -> {
                statusView.setText("下载失败: " + e.getMessage());
                statusView.setTextColor(0xFFFF8395);
            });
        }
    }

    /** 上传数据到云端 */
    private String uploadDataToCloud(String serverUrl, String user,
                                     List<RecordingStore.RecordingInfo> recordings) {
        try {
            // 确保 URL 末尾没有斜杠
            if (serverUrl.endsWith("/")) serverUrl = serverUrl.substring(0, serverUrl.length() - 1);
            URL url = new URL(serverUrl + "/api/upload");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setDoOutput(true);
            conn.setConnectTimeout(30000);
            conn.setReadTimeout(120000);

            String boundary = "----EMGUpload" + System.currentTimeMillis();
            conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);

            // 先收集要上传的文件
            List<File> filesToUpload = new ArrayList<>();
            for (RecordingStore.RecordingInfo rec : recordings) {
                File csvFile = rec.file;
                if (csvFile == null || !csvFile.exists() || !csvFile.getName().endsWith(".csv")) continue;
                filesToUpload.add(csvFile);
            }
            if (filesToUpload.isEmpty()) {
                throw new IOException("没有可上传的采集数据");
            }

            try (OutputStream os = conn.getOutputStream()) {
                // 添加 user 字段
                os.write(("--" + boundary + "\r\n").getBytes("UTF-8"));
                os.write("Content-Disposition: form-data; name=\"user\"\r\n\r\n".getBytes("UTF-8"));
                os.write(user.getBytes("UTF-8"));
                os.write("\r\n".getBytes("UTF-8"));

                // 添加所有 CSV 文件
                for (File csvFile : filesToUpload) {
                    os.write(("--" + boundary + "\r\n").getBytes("UTF-8"));
                    os.write(("Content-Disposition: form-data; name=\"files\"; filename=\""
                            + csvFile.getName() + "\"\r\n").getBytes("UTF-8"));
                    os.write("Content-Type: text/csv\r\n\r\n".getBytes("UTF-8"));

                    try (FileInputStream fis = new FileInputStream(csvFile)) {
                        byte[] buffer = new byte[8192];
                        int len;
                        while ((len = fis.read(buffer)) != -1) {
                            os.write(buffer, 0, len);
                        }
                    }
                    os.write("\r\n".getBytes("UTF-8"));
                }

                os.write(("--" + boundary + "--\r\n").getBytes("UTF-8"));
                os.flush();
            }

            int responseCode = conn.getResponseCode();
            if (responseCode == 200) {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getInputStream(), "UTF-8"))) {
                    StringBuilder sb = new StringBuilder();
                    String line;
                    while ((line = br.readLine()) != null) sb.append(line);
                    return sb.toString();
                }
            } else {
                // 读取错误响应
                InputStream errStream = conn.getErrorStream();
                if (errStream != null) {
                    try (BufferedReader br = new BufferedReader(new InputStreamReader(errStream, "UTF-8"))) {
                        StringBuilder sb = new StringBuilder();
                        String line;
                        while ((line = br.readLine()) != null) sb.append(line);
                        throw new IOException("服务器返回 " + responseCode + ": " + sb.toString());
                    }
                }
                throw new IOException("服务器返回 HTTP " + responseCode);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /** 触发云端训练 */
    private String startCloudTraining(String serverUrl) {
        try {
            URL url = new URL(serverUrl + "/api/train");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setDoOutput(true);
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setConnectTimeout(30000);
            conn.setReadTimeout(60000);

            String body = "{\"protocol\":\"" + parser.getProtocol().name() + "\"}";
            try (OutputStream os = conn.getOutputStream()) {
                os.write(body.getBytes("UTF-8"));
            }

            int responseCode = conn.getResponseCode();
            if (responseCode == 200) {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getInputStream(), "UTF-8"))) {
                    StringBuilder sb = new StringBuilder();
                    String line;
                    while ((line = br.readLine()) != null) {
                        sb.append(line);
                    }
                    // 简单解析 JSON 获取 task_id
                    String json = sb.toString();
                    int idx = json.indexOf("\"task_id\":\"");
                    if (idx >= 0) {
                        int start = idx + 11;
                        int end = json.indexOf("\"", start);
                        return json.substring(start, end);
                    }
                }
            }
            return null;
        } catch (Exception e) {
            return null;
        }
    }

    /** 从服务端返回的 JSON 中取字符串字段（中文会被转义，值中不含裸引号）。 */
    private static String jsonField(String json, String key) {
        if (json == null) return null;
        String needle = "\"" + key + "\":\"";
        int start = json.indexOf(needle);
        if (start < 0) return null;
        start += needle.length();
        int end = json.indexOf('"', start);
        if (end < 0) return null;
        return json.substring(start, end);
    }

    /** 轮询训练状态；2 核 CPU 训练 100 epoch 需数分钟，最多等 15 分钟 */
    private String pollTrainingStatus(String serverUrl, String taskId, TextView statusView) {
        try {
            for (int i = 0; i < 900; i++) {
                Thread.sleep(1000);

                URL url = new URL(serverUrl + "/api/status/" + taskId);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("GET");
                conn.setConnectTimeout(10000);
                conn.setReadTimeout(10000);

                if (conn.getResponseCode() != 200) continue;

                StringBuilder sb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getInputStream(), "UTF-8"))) {
                    String line;
                    while ((line = br.readLine()) != null) sb.append(line);
                }
                String json = sb.toString();

                String status = jsonField(json, "status");
                if ("done".equals(status)) {
                    return jsonField(json, "model_id");
                }
                if ("failed".equals(status)) {
                    String serverError = jsonField(json, "error");
                    throw new IOException("服务端训练失败：" + (serverError == null ? "未知原因" : serverError));
                }

                String progress = jsonField(json, "progress");
                if (progress != null) {
                    main.post(() -> statusView.setText(progress));
                }
            }
            throw new IOException("等待训练超时（超过 15 分钟）");
        } catch (IOException error) {
            throw new RuntimeException(error.getMessage());
        } catch (Exception error) {
            throw new RuntimeException(error.toString());
        }
    }

    /** 下载云端模型，同时解析响应头中的类别标签顺序和归一化统计 */
    private File downloadCloudModel(String serverUrl, String modelId) {
        try {
            URL url = new URL(serverUrl + "/api/download/" + modelId);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setConnectTimeout(30000);
            conn.setReadTimeout(180000);

            if (conn.getResponseCode() == 200) {
                // 服务端把类别顺序放在 X-Model-Labels 头里（URL 编码，可能含中文）
                cloudModelLabels.clear();
                String header = conn.getHeaderField("X-Model-Labels");
                if (header != null && !header.isEmpty()) {
                    String decoded = URLDecoder.decode(header, "UTF-8");
                    for (String item : decoded.split(",")) {
                        if (!item.trim().isEmpty()) cloudModelLabels.add(item.trim());
                    }
                }

                // 解析归一化统计（per-channel mean/std）
                cloudNormMeans = null;
                cloudNormStds = null;
                String normHeader = conn.getHeaderField("X-Model-NormStats");
                if (normHeader != null && !normHeader.isEmpty()) {
                    try {
                        String decodedNorm = URLDecoder.decode(normHeader, "UTF-8");
                        JSONObject normJson = new JSONObject(decodedNorm);
                        JSONArray meansArr = normJson.getJSONArray("means");
                        JSONArray stdsArr = normJson.getJSONArray("stds");
                        cloudNormMeans = new float[meansArr.length()];
                        cloudNormStds = new float[stdsArr.length()];
                        for (int i = 0; i < meansArr.length(); i++) {
                            cloudNormMeans[i] = (float) meansArr.getDouble(i);
                            cloudNormStds[i] = (float) stdsArr.getDouble(i);
                        }
                    } catch (Exception ignored) { }
                }

                // 解析滤波器系数（训练端 scipy 设计，手机端直接复用）
                cloudFilterB = null;
                cloudFilterA = null;
                cloudFilterZi = null;
                String filterHeader = conn.getHeaderField("X-Model-FilterCoeffs");
                if (filterHeader != null && !filterHeader.isEmpty()) {
                    try {
                        String decodedFilter = URLDecoder.decode(filterHeader, "UTF-8");
                        JSONObject filterJson = new JSONObject(decodedFilter);
                        JSONArray bArr = filterJson.getJSONArray("b");
                        JSONArray aArr = filterJson.getJSONArray("a");
                        cloudFilterB = new double[bArr.length()];
                        cloudFilterA = new double[aArr.length()];
                        for (int i = 0; i < bArr.length(); i++) cloudFilterB[i] = bArr.getDouble(i);
                        for (int i = 0; i < aArr.length(); i++) cloudFilterA[i] = aArr.getDouble(i);
                        if (filterJson.has("zi")) {
                            JSONArray ziArr = filterJson.getJSONArray("zi");
                            cloudFilterZi = new double[ziArr.length()];
                            for (int i = 0; i < ziArr.length(); i++) cloudFilterZi[i] = ziArr.getDouble(i);
                        }
                    } catch (Exception ignored) { }
                }

                File cacheFile = new File(getCacheDir(), "cloud_" + modelId + ".pt");
                try (InputStream is = conn.getInputStream();
                     FileOutputStream fos = new FileOutputStream(cacheFile)) {
                    byte[] buffer = new byte[8192];
                    int len;
                    while ((len = is.read(buffer)) != -1) {
                        fos.write(buffer, 0, len);
                    }
                }
                return cacheFile;
            }
            return null;
        } catch (Exception e) {
            return null;
        }
    }

    /** 从 Uri 获取文件名 */
    private String getFileNameFromUri(Uri uri) {
        if (uri == null) return null;
        String fileName = null;
        android.database.Cursor cursor = null;
        try {
            cursor = getContentResolver().query(uri, null, null, null, null);
            if (cursor != null && cursor.moveToFirst()) {
                int displayNameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME);
                if (displayNameIndex >= 0) {
                    fileName = cursor.getString(displayNameIndex);
                }
            }
        } catch (Exception ignored) { }
        finally {
            if (cursor != null) cursor.close();
        }
        // 降级：从 Uri 路径提取文件名
        if (fileName == null) {
            String path = uri.getPath();
            if (path != null) {
                int lastSlash = path.lastIndexOf('/');
                if (lastSlash >= 0 && lastSlash < path.length() - 1) {
                    fileName = path.substring(lastSlash + 1);
                }
            }
        }
        return fileName;
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_IMPORT_PT) {
            if (resultCode != RESULT_OK || data == null) { pendingImportUri = null; return; }
            Uri source = data.getData();
            if (source == null) { toast("未选择文件"); return; }
            pendingImportUri = source;
            showPytorchLabelMappingDialog();
            return;
        }
        if (requestCode == REQUEST_IMPORT_DATA) {
            if (resultCode != RESULT_OK || data == null) return;
            List<Uri> uris = new ArrayList<>();
            // 处理多选
            if (data.getClipData() != null) {
                for (int i = 0; i < data.getClipData().getItemCount(); i++) {
                    Uri uri = data.getClipData().getItemAt(i).getUri();
                    if (uri != null) uris.add(uri);
                }
            } else if (data.getData() != null) {
                uris.add(data.getData());
            }
            if (uris.isEmpty()) { toast("未选择文件"); return; }
            worker.execute(() -> {
                int imported = 0;
                for (Uri uri : uris) {
                    try {
                        String fileName = getFileNameFromUri(uri);
                        if (fileName == null || !fileName.endsWith(".csv")) continue;
                        try (InputStream input = getContentResolver().openInputStream(uri)) {
                            if (input != null && store.importRecording(input, fileName)) {
                                imported++;
                            }
                        }
                    } catch (Exception ignored) { }
                }
                int finalImported = imported;
                main.post(() -> {
                    if (finalImported > 0) {
                        toast("已导入 " + finalImported + " 个数据文件");
                        refreshDatasetAsync();
                        refreshHistoryManager();
                    } else {
                        toast("导入失败：请选择有效的 CSV 文件");
                    }
                });
            });
            return;
        }
        if (requestCode == REQUEST_EXPORT_BACKUP) {
            if (resultCode != RESULT_OK || data == null || data.getData() == null) return;
            Uri destination = data.getData();
            worker.execute(() -> {
                try (OutputStream output = getContentResolver().openOutputStream(destination, "w")) {
                    if (output == null) throw new IOException("无法打开目标文件");
                    int count = store.exportBackup(output);
                    main.post(() -> toast("已备份 " + count + " 个文件"));
                } catch (IOException error) {
                    main.post(() -> toast("备份失败：" + error.getMessage()));
                }
            });
            return;
        }
        if (requestCode == REQUEST_IMPORT_BACKUP) {
            if (resultCode != RESULT_OK || data == null || data.getData() == null) return;
            Uri source = data.getData();
            worker.execute(() -> {
                try (InputStream input = getContentResolver().openInputStream(source)) {
                    if (input == null) throw new IOException("无法读取备份文件");
                    RecordingStore.BackupResult result = store.importBackup(input);
                    main.post(() -> {
                        toast("已恢复 " + result.recordings + " 条采集、"
                                + result.models + " 个模型、"
                                + result.pytorchModels + " 个 PyTorch 文件");
                        refreshDatasetAsync();
                        refreshHistoryManager();
                        refreshModelManager();
                    });
                } catch (IOException error) {
                    main.post(() -> toast("恢复失败：" + error.getMessage()));
                }
            });
            return;
        }
        if (requestCode != REQUEST_EXPORT || resultCode != RESULT_OK || data == null) return;
        Uri destination = data.getData();
        File source = pendingExportFile;
        pendingExportFile = null;
        if (destination == null || source == null) return;
        worker.execute(() -> {
            try (FileInputStream input = new FileInputStream(source);
                 OutputStream output = getContentResolver().openOutputStream(destination, "w")) {
                if (output == null) throw new IOException("无法打开目标文件");
                byte[] buffer = new byte[32768];
                int count;
                while ((count = input.read(buffer)) >= 0) output.write(buffer, 0, count);
                main.post(() -> toast("已导出：" + source.getName()));
            } catch (IOException error) {
                main.post(() -> toast("导出失败：" + error.getMessage()));
            }
        });
    }

    /** BLE 连接日志：记录连接/断开、RSSI、传输统计、异常与崩溃，便于分析断链原因。 */
    private void showBleLog() {
        worker.execute(() -> {
            final String content = ble.eventLog().recent(30000);
            main.post(() -> {
                ScrollView scroll = new ScrollView(this);
                TextView logView = text(content.isEmpty() ? "暂无日志。连接臂环后日志会自动记录在这里。" : content, 11, INK);
                logView.setTypeface(Typeface.MONOSPACE);
                logView.setTextIsSelectable(true);
                logView.setPadding(dp(14), dp(10), dp(8), dp(16));
                scroll.addView(logView);
                new AlertDialog.Builder(this)
                        .setTitle("BLE 连接日志")
                        .setView(scroll)
                        .setPositiveButton("导出", (dialog, which) ->
                                beginExport(ble.eventLog().exportFile(), "text/plain"))
                        .setNeutralButton("清空", (dialog, which) ->
                                new AlertDialog.Builder(this).setTitle("清空日志？")
                                        .setMessage("将删除全部已记录的 BLE 日志，此操作不可恢复。")
                                        .setPositiveButton("清空", (d2, w2) -> {
                                            ble.eventLog().clear();
                                            toast("日志已清空");
                                        })
                                        .setNegativeButton("取消", null).show())
                        .setNegativeButton("关闭", null)
                        .show();
            });
        });
    }

    private void showAbout() {
        String version = "未知";
        long versionCode = 0;
        try {
            android.content.pm.PackageInfo info = getPackageManager().getPackageInfo(getPackageName(), 0);
            version = info.versionName;
            versionCode = Build.VERSION.SDK_INT >= 28 ? info.getLongVersionCode() : info.versionCode;
        } catch (PackageManager.NameNotFoundException ignored) { }
        String message = "版本：v" + version + "（" + versionCode + "）\n"
                + "开发者：智为光电\n\n"
                + "适配硬件：STM32F411 8通道 sEMG 手环\n"
                + "通信协议：BT-11 BLE FFE0 / FFE2\n"
                + "采样规格：8通道 · 500 Hz · 12-bit ADC（原始值 0–4095）\n\n"
                + "数据与模型默认仅保存在本机；仅在启用 G20 控制后访问同机控制服务。\n\n"
                + "© 2026 智为光电";
        new AlertDialog.Builder(this).setIcon(R.drawable.ic_launcher)
                .setTitle("智为光电sEMG测试")
                .setMessage(message).setPositiveButton("关闭", null).show();
    }

    private void showMappingConfig() {
        if (model == null) { toast("请先训练模型"); return; }
        List<String> labels = new ArrayList<>(model.labels());
        Collections.sort(labels);
        labels.add(0, "（未选择）");

        Map<G20HttpController.Command, String> current = g20.getMappingConfig();
        boolean hasGripper = g20.getDeviceType() == G20HttpController.DeviceType.MECHANICAL_CAR;
        String[] cmdNames = hasGripper
                ? new String[]{"前进", "后退", "左转", "右转", "停止", "爪子张开", "爪子并拢"}
                : new String[]{"前进", "后退", "左转", "右转", "停止"};
        G20HttpController.Command[] commands = hasGripper
                ? new G20HttpController.Command[]{
                        G20HttpController.Command.FORWARD,
                        G20HttpController.Command.BACKWARD,
                        G20HttpController.Command.LEFT,
                        G20HttpController.Command.RIGHT,
                        G20HttpController.Command.STOP,
                        G20HttpController.Command.GRIPPER_OPEN,
                        G20HttpController.Command.GRIPPER_CLOSE}
                : new G20HttpController.Command[]{
                        G20HttpController.Command.FORWARD,
                        G20HttpController.Command.BACKWARD,
                        G20HttpController.Command.LEFT,
                        G20HttpController.Command.RIGHT,
                        G20HttpController.Command.STOP};

        LinearLayout content = column();
        content.setPadding(dp(12), dp(6), dp(12), dp(12));
        Spinner[] spinners = new Spinner[commands.length];
        for (int i = 0; i < commands.length; i++) {
            TextView labelView = text(cmdNames[i] + ":", 13, INK);
            content.addView(labelView, matchWrap(0, dp(4)));
            Spinner spinner = new Spinner(this);
            ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, labels);
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
            spinner.setAdapter(adapter);
            spinner.setBackground(rounded(0xFF0B151B, 8, 0xFF2A3A43));
            String currentLabel = current.get(commands[i]);
            if (currentLabel != null) {
                int pos = labels.indexOf(currentLabel);
                if (pos >= 0) spinner.setSelection(pos);
            }
            spinners[i] = spinner;
            content.addView(spinner, matchWrap(0, dp(8)));
        }

        new AlertDialog.Builder(this)
                .setTitle("配置动作映射")
                .setView(content)
                .setPositiveButton("保存", (dialog, which) -> {
                    Map<G20HttpController.Command, String> mapping = new LinkedHashMap<>();
                    for (int i = 0; i < commands.length; i++) {
                        String selected = (String) spinners[i].getSelectedItem();
                        if (!"（未选择）".equals(selected)) {
                            mapping.put(commands[i], selected);
                        }
                    }
                    g20.saveMappingConfig(mapping);
                    toast("映射已保存");
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private void toggleG20Control() {
        if (g20.isEnabled()) {
            g20Button.setEnabled(false);
            g20.disable("控制未启用 · 已发送停止指令");
            return;
        }
        if (model == null) { toast("请先训练包含控制动作的模型"); return; }
        if (!g20.hasStopMapping()) { toast("未配置“停止”动作映射，请先点击“配置动作映射”"); return; }

        String ip = ipInput.getText().toString().trim();
        if (ip.isEmpty()) { toast("请输入目标IP地址"); return; }
        String portStr = portInput.getText().toString().trim();
        int port;
        try {
            port = Integer.parseInt(portStr);
            if (port < 1 || port > 65535) throw new NumberFormatException();
        } catch (NumberFormatException e) {
            toast("端口格式错误（1-65535）"); return;
        }

        int typeIndex = deviceTypeSpinner.getSelectedItemPosition();
        G20HttpController.DeviceType type;
        if (typeIndex == 1) {
            type = G20HttpController.DeviceType.SMART_CAR;
        } else if (typeIndex == 2) {
            type = G20HttpController.DeviceType.GO2;
        } else if (typeIndex == 3) {
            type = G20HttpController.DeviceType.RECON_BALL;
        } else if (typeIndex == 4) {
            type = G20HttpController.DeviceType.GO2W;
        } else {
            type = G20HttpController.DeviceType.MECHANICAL_CAR;
        }

        String timeoutStr = timeoutInput.getText().toString().trim();
        long timeoutMs;
        try {
            int seconds = timeoutStr.isEmpty() ? 30 : Integer.parseInt(timeoutStr);
            if (seconds < 1 || seconds > 3600) throw new NumberFormatException();
            timeoutMs = seconds * 1000L;
        } catch (NumberFormatException e) {
            toast("超时时间格式错误（1-3600 秒）"); return;
        }
        g20.setControlTimeoutMs(timeoutMs);

        g20.setConfig(ip, port, type);
        new AlertDialog.Builder(this)
                .setTitle("启用 WiFi 小车控制？")
                .setMessage("请确认小车周围安全。启用后，稳定识别到左、右、前进、倒退、爪子张开或爪子并拢会持续发送 UDP 控制；停止、低置信度、蓝牙断开和应用退出会停止。")
                .setPositiveButton("确认启用", (dialog, which) -> {
                    g20Button.setEnabled(false);
                    g20StatusView.setText("正在连接小车控制服务…");
                    g20.enable();
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private void sendManualCommand(G20HttpController.Command command) {
        if (g20 == null) {
            toast("控制器未初始化");
            return;
        }
        String ip = ipInput.getText().toString().trim();
        if (ip.isEmpty()) {
            toast("请先输入目标IP地址");
            return;
        }
        String portStr = portInput.getText().toString().trim();
        int port;
        try {
            port = Integer.parseInt(portStr);
            if (port < 1 || port > 65535) throw new NumberFormatException();
        } catch (NumberFormatException e) {
            toast("端口格式错误（1-65535）");
            return;
        }

        int typeIndex = deviceTypeSpinner.getSelectedItemPosition();
        G20HttpController.DeviceType type;
        if (typeIndex == 1) {
            type = G20HttpController.DeviceType.SMART_CAR;
        } else if (typeIndex == 2) {
            type = G20HttpController.DeviceType.GO2;
        } else if (typeIndex == 3) {
            type = G20HttpController.DeviceType.RECON_BALL;
        } else if (typeIndex == 4) {
            type = G20HttpController.DeviceType.GO2W;
        } else {
            type = G20HttpController.DeviceType.MECHANICAL_CAR;
        }

        g20.setConfig(ip, port, type);
        g20.sendManualCommand(command);
    }

    private String getLocalIpAddress() {
        try {
            java.util.Enumeration<java.net.NetworkInterface> interfaces =
                    java.net.NetworkInterface.getNetworkInterfaces();
            while (interfaces.hasMoreElements()) {
                java.net.NetworkInterface ni = interfaces.nextElement();
                if (ni.isLoopback() || !ni.isUp()) continue;
                java.util.Enumeration<java.net.InetAddress> addrs = ni.getInetAddresses();
                while (addrs.hasMoreElements()) {
                    java.net.InetAddress addr = addrs.nextElement();
                    if (!addr.isLoopbackAddress() && addr.getHostAddress().indexOf(':') < 0) {
                        return addr.getHostAddress();
                    }
                }
            }
        } catch (Exception e) {
            // ignore
        }
        return "未知（请检查网络连接）";
    }

    private LinearLayout.LayoutParams weightedButtonParams(int leftMargin, int rightMargin) {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(0, dp(42), 1);
        params.leftMargin = dp(leftMargin);
        params.rightMargin = dp(rightMargin);
        return params;
    }

    private String formatDate(long timestamp) {
        return new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.CHINA).format(new Date(timestamp));
    }

    private void updateRawCardSubtitle() {
        if (rawCardHeader == null || rawCardHeader.getChildCount() < 2) return;
        View subtitleView = rawCardHeader.getChildAt(1);
        if (!(subtitleView instanceof TextView)) return;
        ((TextView) subtitleView).setText("ADC 12-bit · 0–4095 · 500 Hz");
    }

    private void toggleRecognition() {
        if (recognizing) {
            recognizing = false;
            g20.stopMotion("G20 控制已启用 · 实时识别已停止");
            recognizeButton.setText("开始实时识别");
            resultView.setText("—");
            synchronized (inferenceLock) { inferenceWindow.clear(); recentPredictions.clear(); }
            return;
        }
        if (model == null && pytorchModel == null) {
            toast("请先采集并训练模型，或导入 PyTorch 模型");
            return;
        }
        if (!connected) { toast("请先连接手环"); return; }
        SemgPacketParser.Protocol active = parser.getProtocol();
        if (pytorchModel != null && pytorchModel.protocol != active) {
            toast("协议不匹配！当前 " + active.name() + " 协议，PyTorch 模型基于 " + pytorchModel.protocol.name());
            return;
        }
        if (model != null && model.protocol != active) {
            toast("协议不匹配！当前使用 " + active.name() + " 协议，但模型基于 " + model.protocol.name() + " 协议训练。请切换协议或重新训练模型。");
            return;
        }
        recognizing = true;
        inferenceHop = 0;
        inferenceHasResult = false;
        synchronized (inferenceLock) { inferenceWindow.clear(); recentPredictions.clear(); }
        recognizeButton.setText("停止实时识别");
        resultView.setText("准备中…");
    }

    private void processInference(int[][] incoming) {
        if (!recognizing) return;
        SemgPacketParser.Protocol active = parser.getProtocol();
        int windowSize = 0, hopSize = 0;
        boolean protocolMatch = true;
        if (pytorchModel != null) {
            if (pytorchModel.protocol != active) protocolMatch = false;
            windowSize = PyTorchModel.WINDOW;
            hopSize = PyTorchModel.HOP;
        } else if (model != null) {
            if (model.protocol != active) protocolMatch = false;
            windowSize = GestureModel.WINDOW;
            hopSize = GestureModel.HOP;
        } else {
            return;
        }
        int[][] window = null;
        synchronized (inferenceLock) {
            for (int[] sample : incoming) {
                inferenceWindow.addLast(sample);
                while (inferenceWindow.size() > windowSize) inferenceWindow.removeFirst();
                inferenceHop++;
            }
            int curSize = inferenceWindow.size();
            if (curSize < windowSize || inferenceHop < hopSize) {
                if (!protocolMatch) {
                    String msg = "协议不匹配\n当前 " + active.name() + " · 模型 " +
                        (pytorchModel != null ? pytorchModel.protocol : model.protocol) +
                        "\n已收 " + curSize + "/" + windowSize;
                    main.post(() -> resultView.setText(msg));
                } else if (!inferenceHasResult) {
                    // 已产出过结果时不再用"准备中"覆盖结果，避免闪烁；
                    // 只在还没出过结果时提示采集进度。
                    String msg = "准备中… (" + curSize + "/" + windowSize + ")";
                    main.post(() -> resultView.setText(msg));
                }
                return;
            }
            inferenceHop = 0;
            window = inferenceWindow.toArray(new int[0][]);
        }
        if (!protocolMatch) return;
        String label;
        double confidence;
        try {
            if (pytorchModel != null) {
                PyTorchModel.Prediction prediction = pytorchModel.predict(window);
                label = prediction.label;
                confidence = prediction.confidence;
            } else {
                GestureModel.Prediction prediction = model.predict(window);
                label = prediction.label;
                confidence = prediction.confidence;
            }
        } catch (RuntimeException inferenceError) {
            // 推理异常不能打挂数据线程，否则界面会一直卡在"准备中"。
            // 打印完整异常，并在界面提示，便于定位模型/输入不匹配等问题。
            android.util.Log.e("MainActivity", "实时推理失败", inferenceError);
            synchronized (inferenceLock) {
                inferenceWindow.clear();
                inferenceHop = 0;
            }
            main.post(() -> resultView.setText("推理失败：" +
                    (inferenceError.getMessage() != null ? inferenceError.getMessage() : inferenceError.getClass().getSimpleName())));
            return;
        }
        synchronized (inferenceLock) {
            recentPredictions.addLast(label);
            inferenceHasResult = true;   // 已产出过结果，后续不再用"准备中"覆盖
            while (recentPredictions.size() > 5) recentPredictions.removeFirst();
            int votes = 0;
            for (String item : recentPredictions) if (item.equals(label)) votes++;
            if (votes >= 3 || recentPredictions.size() < 3) {
                String displayLabel = confidence < 0.48 ? "不确定" : label;
                String displayText = String.format(Locale.CHINA, "%s\n%.0f%%",
                        displayLabel, confidence * 100);
                main.post(() -> resultView.setText(displayText));
                g20.onRecognition(label, confidence);
            }
        }
    }

    @Override protected void onPause() {
        if (g20 != null) g20.stopMotion("G20 控制已启用 · 应用进入后台，已停止");
        super.onPause();
    }

    @Override protected void onDestroy() {
        if (recording != null) finishRecording();
        if (g20 != null) g20.close();
        ble.close();
        classic.disconnect();
        dismissReconnectDialog();
        worker.shutdownNow();
        super.onDestroy();
    }

    private LinearLayout column() {
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        return layout;
    }

    private LinearLayout card() {
        LinearLayout card = column();
        card.setPadding(dp(14), dp(14), dp(14), dp(14));
        card.setBackground(rounded(SURFACE, 15, 0xFF1B2B34));
        return card;
    }

    private LinearLayout sectionHeader(String title, String subtitle) {
        LinearLayout row = new LinearLayout(this);
        row.setGravity(Gravity.BOTTOM);
        TextView name = text(title, 17, INK);
        name.setTypeface(Typeface.DEFAULT_BOLD);
        row.addView(name, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        TextView detail = text(subtitle, 10, MUTED);
        detail.setGravity(Gravity.RIGHT);
        row.addView(detail);
        row.setPadding(0, 0, 0, dp(10));
        return row;
    }

    private TextView text(String value, float size, int color) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextSize(size);
        view.setTextColor(color);
        view.setLineSpacing(0, 1.12f);
        return view;
    }

    private Button button(String value, boolean primary) {
        Button button = new Button(this);
        button.setText(value);
        button.setTextSize(13);
        button.setAllCaps(false);
        button.setTypeface(Typeface.DEFAULT_BOLD);
        button.setTextColor(primary ? 0xFF06140F : INK);
        button.setPadding(dp(8), 0, dp(8), 0);
        button.setBackground(rounded(primary ? ACCENT : 0xFF1B2B34, 10, primary ? 0 : 0xFF344954));
        return button;
    }

    private GradientDrawable rounded(int color, int radius, int stroke) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(color);
        drawable.setCornerRadius(dp(radius));
        if (stroke != 0) drawable.setStroke(dp(1), stroke);
        return drawable;
    }

    private LinearLayout.LayoutParams matchWrap(int top, int bottom) {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        params.topMargin = top; params.bottomMargin = bottom;
        return params;
    }

    private LinearLayout.LayoutParams matchHeight(int height, int top, int bottom) {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, height);
        params.topMargin = top; params.bottomMargin = bottom;
        return params;
    }

    private int dp(float value) { return (int) (value * getResources().getDisplayMetrics().density + 0.5f); }
    private void toast(String text) { Toast.makeText(this, text, Toast.LENGTH_SHORT).show(); }

    // ==================== Token 认证的云端辅助方法 ====================

    /** 查询云端文件列表（使用 Token 认证） */
    private Set<String> queryCloudFilesWithToken(String serverUrl) {
        try {
            if (serverUrl.endsWith("/")) serverUrl = serverUrl.substring(0, serverUrl.length() - 1);
            URL url = new URL(serverUrl + "/api/files");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setRequestProperty("Authorization", "Bearer " + authToken);
            conn.setConnectTimeout(10000);
            conn.setReadTimeout(10000);

            if (conn.getResponseCode() == 200) {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getInputStream(), "UTF-8"))) {
                    StringBuilder sb = new StringBuilder();
                    String line;
                    while ((line = br.readLine()) != null) sb.append(line);
                    
                    JSONObject json = new JSONObject(sb.toString());
                    JSONArray filesArray = json.getJSONArray("files");
                    Set<String> files = new HashSet<>();
                    for (int i = 0; i < filesArray.length(); i++) {
                        files.add(filesArray.getString(i));
                    }
                    return files;
                }
            }
            return null;
        } catch (Exception e) {
            return null;
        }
    }

    /** 增量上传数据到云端（使用 Token 认证） */
    private String uploadDataToCloudWithToken(String serverUrl,
                                              List<RecordingStore.RecordingInfo> recordings) {
        try {
            if (serverUrl.endsWith("/")) serverUrl = serverUrl.substring(0, serverUrl.length() - 1);
            URL url = new URL(serverUrl + "/api/upload");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Authorization", "Bearer " + authToken);
            conn.setDoOutput(true);
            conn.setConnectTimeout(30000);
            conn.setReadTimeout(120000);

            String boundary = "----EMGUpload" + System.currentTimeMillis();
            conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);

            List<File> filesToUpload = new ArrayList<>();
            for (RecordingStore.RecordingInfo rec : recordings) {
                File csvFile = rec.file;
                if (csvFile == null || !csvFile.exists() || !csvFile.getName().endsWith(".csv")) continue;
                filesToUpload.add(csvFile);
            }
            if (filesToUpload.isEmpty()) {
                throw new IOException("没有可上传的采集数据");
            }

            try (OutputStream os = conn.getOutputStream()) {
                for (File csvFile : filesToUpload) {
                    os.write(("--" + boundary + "\r\n").getBytes("UTF-8"));
                    os.write(("Content-Disposition: form-data; name=\"files\"; filename=\""
                            + csvFile.getName() + "\"\r\n").getBytes("UTF-8"));
                    os.write("Content-Type: text/csv\r\n\r\n".getBytes("UTF-8"));

                    try (FileInputStream fis = new FileInputStream(csvFile)) {
                        byte[] buffer = new byte[8192];
                        int len;
                        while ((len = fis.read(buffer)) != -1) {
                            os.write(buffer, 0, len);
                        }
                    }
                    os.write("\r\n".getBytes("UTF-8"));
                }

                os.write(("--" + boundary + "--\r\n").getBytes("UTF-8"));
                os.flush();
            }

            int responseCode = conn.getResponseCode();
            if (responseCode == 200) {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getInputStream(), "UTF-8"))) {
                    StringBuilder sb = new StringBuilder();
                    String line;
                    while ((line = br.readLine()) != null) sb.append(line);
                    return sb.toString();
                }
            } else {
                InputStream errStream = conn.getErrorStream();
                if (errStream != null) {
                    try (BufferedReader br = new BufferedReader(new InputStreamReader(errStream, "UTF-8"))) {
                        StringBuilder sb = new StringBuilder();
                        String line;
                        while ((line = br.readLine()) != null) sb.append(line);
                        throw new IOException("服务器返回 " + responseCode + ": " + sb.toString());
                    }
                }
                throw new IOException("服务器返回 HTTP " + responseCode);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /** 触发云端训练（使用 Token 认证） */
    private String startCloudTrainingWithToken(String serverUrl, String modelType) {
        try {
            URL url = new URL(serverUrl + "/api/train");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Authorization", "Bearer " + authToken);
            conn.setDoOutput(true);
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setConnectTimeout(30000);
            conn.setReadTimeout(60000);

            String body = "{\"protocol\":\"" + parser.getProtocol().name() + "\",\"model_type\":\"" + modelType + "\"}";
            try (OutputStream os = conn.getOutputStream()) {
                os.write(body.getBytes("UTF-8"));
            }

            int responseCode = conn.getResponseCode();
            if (responseCode == 200) {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getInputStream(), "UTF-8"))) {
                    StringBuilder sb = new StringBuilder();
                    String line;
                    while ((line = br.readLine()) != null) {
                        sb.append(line);
                    }
                    String json = sb.toString();
                    int idx = json.indexOf("\"task_id\":\"");
                    if (idx >= 0) {
                        int start = idx + 11;
                        int end = json.indexOf("\"", start);
                        return json.substring(start, end);
                    }
                }
            }
            return null;
        } catch (Exception e) {
            return null;
        }
    }

    /** 轮询训练状态（使用 Token 认证） */
    private String pollTrainingStatusWithToken(String serverUrl, String taskId, TextView statusView) {
        try {
            for (int i = 0; i < 900; i++) {
                Thread.sleep(1000);

                URL url = new URL(serverUrl + "/api/status/" + taskId);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("GET");
                conn.setRequestProperty("Authorization", "Bearer " + authToken);
                conn.setConnectTimeout(10000);
                conn.setReadTimeout(10000);

                if (conn.getResponseCode() != 200) continue;

                StringBuilder sb = new StringBuilder();
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getInputStream(), "UTF-8"))) {
                    String line;
                    while ((line = br.readLine()) != null) sb.append(line);
                }
                String json = sb.toString();

                String status = jsonField(json, "status");
                if ("done".equals(status)) {
                    return jsonField(json, "model_id");
                }
                if ("failed".equals(status)) {
                    String serverError = jsonField(json, "error");
                    throw new IOException("服务端训练失败：" + (serverError == null ? "未知原因" : serverError));
                }

                String progress = jsonField(json, "progress");
                if (progress != null) {
                    main.post(() -> statusView.setText(progress));
                }
            }
            throw new IOException("等待训练超时（超过 15 分钟）");
        } catch (IOException error) {
            throw new RuntimeException(error.getMessage());
        } catch (Exception error) {
            throw new RuntimeException(error);
        }
    }

    /** 下载云端模型（使用 Token 认证） */
    private File downloadCloudModelWithToken(String serverUrl, String modelId) {
        try {
            // base_model 走零样本下载接口，其余走常规模型下载接口
            String endpoint = "base_model".equals(modelId)
                    ? "/api/download_base_model"
                    : "/api/model/" + modelId;
            URL url = new URL(serverUrl + endpoint);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setRequestProperty("Authorization", "Bearer " + authToken);
            conn.setConnectTimeout(30000);
            conn.setReadTimeout(180000);

            if (conn.getResponseCode() != 200) {
                throw new IOException("下载模型失败");
            }

            // 解析响应头
            cloudModelLabels.clear();
            String header = conn.getHeaderField("X-Model-Labels");
            if (header != null && !header.isEmpty()) {
                String decoded = URLDecoder.decode(header, "UTF-8");
                for (String item : decoded.split(",")) {
                    if (!item.trim().isEmpty()) cloudModelLabels.add(item.trim());
                }
            }

            cloudNormMeans = null;
            cloudNormStds = null;
            String normHeader = conn.getHeaderField("X-Model-NormStats");
            if (normHeader != null && !normHeader.isEmpty()) {
                try {
                    String decodedNorm = URLDecoder.decode(normHeader, "UTF-8");
                    JSONObject normJson = new JSONObject(decodedNorm);
                    JSONArray meansArr = normJson.getJSONArray("means");
                    JSONArray stdsArr = normJson.getJSONArray("stds");
                    cloudNormMeans = new float[meansArr.length()];
                    cloudNormStds = new float[stdsArr.length()];
                    for (int i = 0; i < meansArr.length(); i++) {
                        cloudNormMeans[i] = (float) meansArr.getDouble(i);
                        cloudNormStds[i] = (float) stdsArr.getDouble(i);
                    }
                } catch (Exception ignored) { }
            }

            cloudFilterB = null;
            cloudFilterA = null;
            cloudFilterZi = null;
            String filterHeader = conn.getHeaderField("X-Model-FilterCoeffs");
            if (filterHeader != null && !filterHeader.isEmpty()) {
                try {
                    String decodedFilter = URLDecoder.decode(filterHeader, "UTF-8");
                    JSONObject filterJson = new JSONObject(decodedFilter);
                    JSONArray bArr = filterJson.getJSONArray("b");
                    JSONArray aArr = filterJson.getJSONArray("a");
                    JSONArray ziArr = filterJson.optJSONArray("zi");
                    cloudFilterB = new double[bArr.length()];
                    cloudFilterA = new double[aArr.length()];
                    for (int i = 0; i < bArr.length(); i++) cloudFilterB[i] = bArr.getDouble(i);
                    for (int i = 0; i < aArr.length(); i++) cloudFilterA[i] = aArr.getDouble(i);
                    if (ziArr != null) {
                        cloudFilterZi = new double[ziArr.length()];
                        for (int i = 0; i < ziArr.length(); i++) cloudFilterZi[i] = ziArr.getDouble(i);
                    }
                } catch (Exception ignored) { }
            }

            File ptFile = new File(getCacheDir(), "cloud_model_" + modelId + ".pt");
            try (InputStream is = conn.getInputStream();
                 FileOutputStream fos = new FileOutputStream(ptFile)) {
                byte[] buffer = new byte[8192];
                int len;
                while ((len = is.read(buffer)) != -1) {
                    fos.write(buffer, 0, len);
                }
            }

            return ptFile;
        } catch (Exception e) {
            return null;
        }
    }
}