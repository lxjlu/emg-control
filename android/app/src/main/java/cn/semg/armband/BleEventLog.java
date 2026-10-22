package cn.semg.armband;

import android.content.Context;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * BLE 事件日志：记录连接/断开、RSSI、传输统计与异常，持久化到应用私有目录，
 * 用于事后分析断链原因。文件超过上限时从头部裁剪（环形保留最近记录）。
 * 所有方法线程安全（BLE 回调来自 binder 线程）。
 */
final class BleEventLog {
    private static final long MAX_FILE_BYTES = 512 * 1024;
    private static final String LINE_PREFIX = "==== ";

    private final File file;
    private final Object lock = new Object();
    private final SimpleDateFormat stamp =
            new SimpleDateFormat("MM-dd HH:mm:ss.SSS", Locale.CHINA);

    BleEventLog(Context context) {
        file = new File(context.getFilesDir(), "ble_log.txt");
    }

    File getFile() { return file; }

    void log(String tag, String message) {
        append(tag, message, null);
    }

    void logError(String tag, String message, Throwable error) {
        append(tag, message, error);
    }

    /** 记录一次异常堆栈（不写消息体）。 */
    void logThrowable(String tag, Throwable error) {
        append(tag, "异常捕获", error);
    }

    void clear() {
        synchronized (lock) {
            try (FileOutputStream out = new FileOutputStream(file)) {
                out.write(header("日志已清空").getBytes(StandardCharsets.UTF_8));
            } catch (IOException ignored) { }
        }
    }

    /** 读取最近的日志内容（最多 maxChars 个字符，按记录倒序：最新在最上面）。 */
    String recent(int maxChars) {
        synchronized (lock) {
            byte[] all = readAll();
            if (all.length == 0) return "";
            // 简单截取尾部（UTF-8 中文按 3 字节，宽松一点不碍事）
            int keep = Math.min(all.length, maxChars * 3);
            int start = all.length - keep;
            while (start < all.length && start > 0 && all[start] != '\n') start++;
            String text = new String(all, start, all.length - start, StandardCharsets.UTF_8);
            if (text.length() > maxChars) text = text.substring(text.length() - maxChars);
            // 以时间戳行（"MM-dd HH:mm:ss.SSS [tag] ..."）为一条记录的起点，
            // 按记录倒序排列，让最新记录显示在最上面；异常堆栈行跟随其所属记录。
            java.util.List<String> entries = new java.util.ArrayList<>();
            String current = null;
            for (String line : text.split("\n", -1)) {
                if (isTimestampLine(line)) {
                    if (current != null) entries.add(current);
                    current = line;
                } else {
                    if (current == null) current = ""; // 文件被裁剪后残缺的首条
                    current = current.isEmpty() ? line : current + "\n" + line;
                }
            }
            if (current != null) entries.add(current);
            java.util.Collections.reverse(entries);
            return String.join("\n", entries);
        }
    }

    /** 判断是否为带时间戳的记录首行，格式如 "08-28 12:04:09.318 [BLE] ..."。 */
    private static boolean isTimestampLine(String line) {
        if (line.length() < 18) return false;
        return line.charAt(2) == '-' && line.charAt(5) == ' '
                && line.charAt(8) == ':' && line.charAt(11) == ':'
                && line.charAt(14) == ':' && line.charAt(17) == ' ';
    }

    /** 生成用于导出的倒序日志文件（最新记录在最上面），返回导出用的临时文件。 */
    File exportFile() {
        synchronized (lock) {
            File export = new File(file.getParentFile(), "ble_log_export.txt");
            try (FileOutputStream out = new FileOutputStream(export)) {
                out.write(recent(2_000_000).getBytes(StandardCharsets.UTF_8));
            } catch (IOException ignored) { }
            return export;
        }
    }

    private void append(String tag, String message, Throwable error) {
        StringBuilder line = new StringBuilder();
        synchronized (lock) {
            line.append(stamp.format(new Date())).append(" [").append(tag).append("] ")
                    .append(message).append('\n');
            if (error != null) {
                StringWriter trace = new StringWriter();
                error.printStackTrace(new PrintWriter(trace));
                for (String part : trace.toString().split("\n")) {
                    if (part.trim().isEmpty()) continue;
                    line.append("    ").append(part.trim()).append('\n');
                }
            }
            try (FileOutputStream out = new FileOutputStream(file, true)) {
                out.write(line.toString().getBytes(StandardCharsets.UTF_8));
            } catch (IOException ignored) { }
            trimIfNeeded();
        }
    }

    private void trimIfNeeded() {
        if (file.length() <= MAX_FILE_BYTES) return;
        byte[] all = readAll();
        int keep = (int) (MAX_FILE_BYTES / 2);
        int start = all.length - keep;
        while (start < all.length && start > 0 && all[start] != '\n') start++;
        if (start >= all.length) return;
        try (FileOutputStream out = new FileOutputStream(file)) {
            out.write(header("日志达到上限，已裁剪保留最近记录").getBytes(StandardCharsets.UTF_8));
            out.write(all, start, all.length - start);
        } catch (IOException ignored) { }
    }

    private byte[] readAll() {
        if (!file.isFile()) return new byte[0];
        try (InputStream in = new FileInputStream(file)) {
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            byte[] chunk = new byte[8192];
            int read;
            while ((read = in.read(chunk)) > 0) buffer.write(chunk, 0, read);
            return buffer.toByteArray();
        } catch (IOException error) {
            return new byte[0];
        }
    }

    private static String header(String message) {
        return LINE_PREFIX + message + " " +
                new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.CHINA).format(new Date())
                + " " + LINE_PREFIX + "\n";
    }
}
