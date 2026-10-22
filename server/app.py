"""
EMG 云端训练服务 - Flask API
功能：接收手机上传的采集 CSV → 转成训练脚本要求的 txt 格式 → 训练 CNN1D → 导出 .pt → 提供下载

数据格式说明（方案 A：训练与推理统一使用原始 ADC 值，不做 µV 换算）：
  手机录制 CSV 格式：
      # sEMG recording v1
      # label=前进
      # user=zhang
      # protocol=STM32
      sample_index,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8
      0,2048,2049,...
  训练脚本要求格式：
      Sep_EMG_Data/<user>/gesture_1/1.txt   （纯数字文件名，8 列数值，无表头无索引列）

  Android 端 PyTorchModel.predict() 直接用原始 ADC 值做带通滤波后送入 CNN，
  因此这里转换时同样不做任何缩放，保证训练/推理量纲一致。

部署：python3.8 app.py
"""

import os
import re
import sys
import json
import time
import shutil
import hashlib
import threading
import subprocess
import sqlite3
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps

from flask import Flask, request, jsonify, send_file
from urllib.parse import quote

app = Flask(__name__)
# 固定 SECRET_KEY，避免服务重启后之前登录的 Token 失效
# 生产环境建议从环境变量或配置文件读取
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'emg_armband_secret_key_2026_fixed')

# ==================== 数据库初始化 ====================
DB_PATH = BASE_DIR = Path(__file__).parent.resolve()
DB_FILE = BASE_DIR / "users.db"

def get_db():
    """获取数据库连接"""
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """初始化数据库表"""
    conn = get_db()
    cursor = conn.cursor()
    
    # 用户表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT,
            is_admin INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            status TEXT DEFAULT 'active'
        )
    ''')
    
    # 检查并添加 is_admin 列（兼容旧数据库）
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
        print("[数据库] 已添加 is_admin 列", flush=True)
    except Exception:
        pass  # 列已存在，忽略错误

    # 检查并添加 gender/weight/hand 列（用户元数据，用于 GRL 不变性训练）
    for col, decl in [
        ("gender", "TEXT"),   # 'male' | 'female' | NULL
        ("weight", "REAL"),   # kg, NULL
        ("hand",   "TEXT"),   # 'left' | 'right' | NULL
    ]:
        try:
            cursor.execute(f"ALTER TABLE users ADD COLUMN {col} {decl}")
            print(f"[数据库] 已添加 {col} 列", flush=True)
        except Exception:
            pass

    # 用户数据文件表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            label TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_size INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # 用户模型表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            model_id TEXT,
            accuracy REAL,
            labels TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()

# 启动时初始化数据库
init_db()

# ==================== JWT Token 工具函数 ====================
def generate_token(user_id, username):
    """生成简单的 JWT-like Token（只存 user_id，避免中文编码问题）"""
    payload = {
        'user_id': user_id,
        'exp': (datetime.now() + timedelta(days=30)).timestamp()
    }
    payload_json = json.dumps(payload)
    signature = hashlib.sha256((payload_json + app.config['SECRET_KEY']).encode()).hexdigest()
    return f"{payload_json}.{signature}"

def verify_token(token):
    """验证 Token 并返回用户信息"""
    try:
        # 从右边找最后一个 '.'，避免 exp 浮点数小数点干扰
        dot_index = token.rfind('.')
        if dot_index == -1:
            return None
        payload_json = token[:dot_index]
        signature = token[dot_index + 1:]
        
        expected_sig = hashlib.sha256((payload_json + app.config['SECRET_KEY']).encode()).hexdigest()
        if signature != expected_sig:
            return None
        payload = json.loads(payload_json)
        if payload['exp'] < datetime.now().timestamp():
            return None
        
        # 从数据库获取用户名
        conn = get_db()
        user = conn.execute("SELECT username FROM users WHERE id = ?", (payload['user_id'],)).fetchone()
        conn.close()
        
        if not user:
            return None
        
        payload['username'] = user['username']
        return payload
    except Exception:
        return None

def token_required(f):
    """Token 验证装饰器"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            token = request.args.get('token')
        if not token:
            return jsonify({'error': '未提供认证令牌'}), 401
        
        # 处理 "Bearer " 前缀
        if token.startswith('Bearer '):
            token = token[7:]
        
        payload = verify_token(token)
        if not payload:
            return jsonify({'error': '无效的认证令牌'}), 401
        
        request.user_id = payload['user_id']
        request.username = payload['username']
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    """管理员权限验证装饰器"""
    @wraps(f)
    def decorated(*args, **kwargs):
        # 方式 1：通过 Token 验证（管理员账号登录）
        token = request.headers.get('Authorization')
        if token:
            if token.startswith('Bearer '):
                token = token[7:]
            payload = verify_token(token)
            if payload:
                conn = get_db()
                user = conn.execute("SELECT is_admin FROM users WHERE id = ?", 
                                   (payload['user_id'],)).fetchone()
                conn.close()
                if user and user['is_admin']:
                    request.is_admin = True
                    request.admin_username = payload['username']
                    return f(*args, **kwargs)
        
        # 方式 2：通过管理员密码验证（App 端直接传密码）
        payload_data = request.get_json(silent=True) or {}
        admin_password = payload_data.get('admin_password') or request.args.get('admin_password')
        
        if admin_password:
            password_hash = hashlib.sha256(admin_password.encode()).hexdigest()
            if password_hash == ADMIN_PASSWORD_HASH:
                request.is_admin = True
                request.admin_username = "admin"
                return f(*args, **kwargs)
        
        return jsonify({'error': '需要管理员权限'}), 403
    return decorated

# ==================== 配置 ====================
UPLOAD_DIR = BASE_DIR / "uploads"      # 原始上传的 CSV
MODEL_DIR = BASE_DIR / "models"        # 训练产出的 .pt + 元数据
DATA_DIR = BASE_DIR / "data"           # 每次训练的临时工作目录
TRAIN_SCRIPT = BASE_DIR / "train_model_sep.py"

# 管理员密码（生产环境建议从环境变量读取）
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "zhiwei8888")
ADMIN_PASSWORD_HASH = hashlib.sha256(ADMIN_PASSWORD.encode()).hexdigest()

# 基础模型相关配置
BASE_MODEL_PATH = BASE_DIR / "base_model_transformer.pth"
BASE_MODEL_USERS_FILE = BASE_DIR / "base_model_users.txt"
# 零样本下载：基础模型导出的 scripted .pt + 元数据
BASE_MODEL_SCRIPTED_PATH = BASE_DIR / "base_model_scripted.pt"
BASE_MODEL_NORM_STATS = BASE_DIR / "base_model_norm_stats.json"
BASE_MODEL_FILTER_COEFFS = BASE_DIR / "base_model_filter_coeffs.json"
BASE_MODEL_RETRAIN_LOCK = threading.Lock()  # 防止并发重新训练

# 验证训练脚本是否存在
if not TRAIN_SCRIPT.exists():
    raise FileNotFoundError(f"训练脚本不存在: {TRAIN_SCRIPT}")

for d in (UPLOAD_DIR, MODEL_DIR, DATA_DIR):
    d.mkdir(exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

CHANNELS = 8
# 训练脚本 CNN1D 的 fc 输出固定为 5 类，映射到 gesture_0..gesture_4（包含休息状态）
MAX_CLASSES = 5
# 规范标签顺序：与手机端"配置类别映射"的默认顺序一致，
# 保证训练出的类别下标 i 对应第 i 个标签
CANONICAL_LABELS = ["休息", "握拳", "手掌", "向右", "向左"]

# ==================== 任务状态 ====================
tasks = {}
tasks_lock = threading.Lock()


def generate_task_id():
    return "task_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + \
           hashlib.md5(str(time.time()).encode()).hexdigest()[:6]


def safe_name(raw, fallback="unnamed"):
    """目录名净化：去掉路径分隔符，保留中英文/数字/下划线/连字符。"""
    text = (raw or "").strip()
    text = text.replace("/", "_").replace("\\", "_").replace("..", "_")
    text = "".join(ch for ch in text if ch.isalnum() or ch in "-_")
    return text or fallback


def collect_user_dirs(users=None):
    """
    收集给定用户的所有上传目录（兼容旧版 session 命名遗留的目录）。
    返回 list[Path]，其中每个 Path 都是 uploads/ 下的子目录。
    - users=None: 返回所有用户目录
    - users=[...]: 返回列表中每个 username 对应的所有目录（精确匹配 + 前缀匹配 session 目录）
    """
    if not UPLOAD_DIR.exists():
        return []

    all_dirs = [d for d in UPLOAD_DIR.iterdir() if d.is_dir()]

    if users is None:
        return sorted(all_dirs, key=lambda p: p.name)

    dirs = []
    for user in users:
        base = safe_name(user, "user")
        for d in all_dirs:
            if d.name == base:
                dirs.append(d)
            elif d.name.startswith(base + "_") or d.name.startswith(base + "-"):
                dirs.append(d)
    return sorted(dirs, key=lambda p: p.name)


def get_uniq_usernames():
    """
    从 uploads/ 目录推导出去重后的唯一 username 集合。
    规则：优先返回精确匹配"username"的目录名；若只有 session 目录（如 李鑫zw_session_016），
    则根据 CSV 内部的 # user= 元数据反推（读第一个 CSV）。
    """
    if not UPLOAD_DIR.exists():
        return []

    dir_names = sorted(d.name for d in UPLOAD_DIR.iterdir() if d.is_dir())

    exact = set()
    for name in dir_names:
        if "_session_" not in name and "-session-" not in name:
            exact.add(name)

    session_dirs = [name for name in dir_names if "_session_" in name or "-session-" in name]
    derived_from_csv = set()
    for session_dir_name in session_dirs:
        d = UPLOAD_DIR / session_dir_name
        csv_files = list(d.glob("*.csv"))
        if csv_files:
            try:
                _, csv_user, _, _, _, _ = parse_recording(csv_files[0])
                if csv_user:
                    derived_from_csv.add(safe_name(csv_user, "user"))
                    continue
            except Exception:
                pass
        base = session_dir_name.split("_session_")[0].split("-session-")[0]
        derived_from_csv.add(base)

    return sorted(exact | derived_from_csv)


def hash_password(password):
    """密码哈希"""
    return hashlib.sha256(password.encode()).hexdigest()


# ==================== API: 用户注册 ====================
@app.route("/api/register", methods=["POST"])
def register():
    """用户注册"""
    payload = request.get_json(silent=True) or {}
    username = payload.get("username", "").strip()
    password = payload.get("password", "").strip()
    email = payload.get("email", "").strip()
    gender = (payload.get("gender") or "").strip().lower()
    weight = payload.get("weight")
    hand = (payload.get("hand") or "").strip().lower()

    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400
    if len(username) < 3 or len(username) > 20:
        return jsonify({"error": "用户名长度 3-20 个字符"}), 400
    if len(password) < 6:
        return jsonify({"error": "密码长度至少 6 个字符"}), 400
    if gender and gender not in ("male", "female"):
        return jsonify({"error": "gender 必须为 male/female"}), 400
    if hand and hand not in ("left", "right"):
        return jsonify({"error": "hand 必须为 left/right"}), 400
    if weight is not None:
        try:
            weight = float(weight)
        except Exception:
            return jsonify({"error": "weight 必须为数字"}), 400

    conn = get_db()
    try:
        password_hash = hash_password(password)
        conn.execute(
            "INSERT INTO users (username, password_hash, email, gender, weight, hand) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (username, password_hash, email,
             gender or None, weight, hand or None)
        )
        conn.commit()

        # 获取用户 ID
        user = conn.execute("SELECT id, is_admin FROM users WHERE username = ?", (username,)).fetchone()
        user_id = user['id']

        # 生成 Token
        token = generate_token(user_id, username)

        return jsonify({
            "message": "注册成功",
            "token": token,
            "user_id": user_id,
            "username": username,
            "is_admin": bool(user['is_admin']),
            "gender": gender or None,
            "weight": weight,
            "hand": hand or None
        }), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "用户名已存在"}), 409
    finally:
        conn.close()


# ==================== API: 用户登录 ====================
@app.route("/api/login", methods=["POST"])
def login():
    """用户登录"""
    payload = request.get_json(silent=True) or {}
    username = payload.get("username", "").strip()
    password = payload.get("password", "").strip()
    
    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400
    
    conn = get_db()
    try:
        user = conn.execute(
            "SELECT id, username, password_hash, is_admin, gender, weight, hand "
            "FROM users WHERE username = ? AND status = 'active'",
            (username,)
        ).fetchone()

        if not user or user['password_hash'] != hash_password(password):
            return jsonify({"error": "用户名或密码错误"}), 401

        # 更新最后登录时间
        conn.execute(
            "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
            (user['id'],)
        )
        conn.commit()

        # 生成 Token
        token = generate_token(user['id'], user['username'])

        return jsonify({
            "message": "登录成功",
            "token": token,
            "user_id": user['id'],
            "username": user['username'],
            "is_admin": bool(user['is_admin']),
            "gender": user['gender'],
            "weight": user['weight'],
            "hand": user['hand']
        })
    finally:
        conn.close()


# ==================== API: 获取用户信息 ====================
@app.route("/api/profile", methods=["GET"])
@token_required
def get_profile():
    """获取当前用户信息"""
    conn = get_db()
    try:
        user = conn.execute(
            "SELECT id, username, email, created_at, last_login FROM users WHERE id = ?",
            (request.user_id,)
        ).fetchone()
        
        if not user:
            return jsonify({"error": "用户不存在"}), 404
        
        # 统计数据量
        file_count = conn.execute(
            "SELECT COUNT(*) as count FROM user_files WHERE user_id = ?",
            (request.user_id,)
        ).fetchone()['count']
        
        model_count = conn.execute(
            "SELECT COUNT(*) as count FROM user_models WHERE user_id = ?",
            (request.user_id,)
        ).fetchone()['count']
        
        return jsonify({
            "id": user['id'],
            "username": user['username'],
            "email": user['email'],
            "created_at": user['created_at'],
            "last_login": user['last_login'],
            "stats": {
                "file_count": file_count,
                "model_count": model_count
            }
        })
    finally:
        conn.close()


def parse_recording(path):
    """
    解析手机录制 CSV。
    返回 (label, user, session, protocol, hand, rows)；rows 为 list[list[float]]，每行 8 个通道的原始 ADC 值。
    """
    label = None
    user = None
    session = None
    protocol = None
    hand = None
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                body = line[1:].strip()
                if "=" in body:
                    key, value = body.split("=", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key == "label":
                        label = value
                    elif key == "user":
                        user = value
                    elif key == "session":
                        session = value
                    elif key == "protocol":
                        protocol = value
                    elif key == "hand":
                        hand = value.lower()
                continue

            parts = line.split(",")
            if len(parts) < CHANNELS + 1:
                continue
            try:
                values = [float(item) for item in parts]
            except ValueError:
                continue  # 跳过 "sample_index,ch1,..." 表头行
            rows.append(values[1:CHANNELS + 1])  # 丢掉 sample_index 列

    return label, user, session, protocol, hand, rows


def build_label_order(found_labels):
    """把出现过的标签排成稳定的类别顺序（规范标签优先）。"""
    ordered = [item for item in CANONICAL_LABELS if item in found_labels]
    ordered += sorted(item for item in found_labels if item not in CANONICAL_LABELS)
    return ordered[:MAX_CLASSES]


# ==================== API: 上传数据（增量模式） ====================
@app.route("/api/upload", methods=["POST"])
@token_required
def upload_csv():
    """
    上传采集 CSV（增量模式）。
    - 需要 Token 认证
    - 文件名冲突时自动追加序号，不覆盖已有文件
    - 标签从 CSV 的 # label= 头部解析，不依赖客户端参数
    - 支持 mode 参数：incremental（增量，默认）| full（全量覆盖）
    - 统一按用户名存放在 uploads/<username>/ 下，session 保留在 CSV 元数据中
    """
    files = request.files.getlist("files")
    files = [item for item in files if item and item.filename]
    if not files:
        return jsonify({"error": "未上传文件"}), 400

    user = request.username  # 从 Token 获取用户名
    mode = request.form.get("mode", "incremental").strip()  # 默认增量模式
    
    first_session = None
    for item in files:
        if item and item.filename:
            tmp_path = Path(item.filename)
            try:
                tmp_file = DATA_DIR / f"tmp_parse_{tmp_path.name}"
                tmp_file.parent.mkdir(parents=True, exist_ok=True)
                item.save(tmp_file)
                _, _, session, _, _, _ = parse_recording(tmp_file)
                first_session = session
                tmp_file.unlink()
            except Exception:
                pass
            break
    
    user_dir = UPLOAD_DIR / safe_name(user, "user")
    
    if mode == "full":
        if user_dir.exists():
            shutil.rmtree(user_dir)
        user_dir.mkdir(parents=True, exist_ok=True)
    else:
        user_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    skipped = 0
    conn = get_db()
    try:
        for item in files:
            name = safe_name(Path(item.filename).stem, "rec")
            target = user_dir / (name + ".csv")
            index = 1
            if target.exists():
                skipped += 1
                continue
            while target.exists():
                target = user_dir / f"{name}_{index}.csv"
                index += 1
            item.save(target)
            saved.append(target.name)
            
            label, _, session, _, _, rows = parse_recording(target)
            conn.execute(
                "INSERT INTO user_files (user_id, filename, label, file_size) VALUES (?, ?, ?, ?)",
                (request.user_id, target.name, label, target.stat().st_size)
            )
        conn.commit()

        # 写 user_meta.json：性别/体重为用户级；手别已下沉到记录级（各 CSV 的 # hand= 头部）
        try:
            user_row = conn.execute(
                "SELECT gender, weight, hand FROM users WHERE id = ?",
                (request.user_id,)
            ).fetchone()
            meta = {
                "gender": user_row['gender'] if user_row else None,
                "weight": user_row['weight'] if user_row else None,
                # 兜底手别：CSV 缺失 # hand= 时使用（注册时填写）
                "hand":   user_row['hand'] if user_row else None,
            }
            (user_dir / "user_meta.json").write_text(
                json.dumps(meta, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass
    finally:
        conn.close()

    return jsonify({
        "message": f"上传成功，新增 {len(saved)} 个文件，跳过 {skipped} 个已存在文件",
        "user": user_dir.name,
        "session": first_session or user,
        "files": saved,
        "skipped": skipped,
        "mode": mode,
    })


# ==================== API: 查询基础模型状态 ====================
@app.route("/api/base_model_status", methods=["GET"])
@token_required
def base_model_status():
    """查询 CNN-Transformer 基础模型是否存在"""
    exists = BASE_MODEL_PATH.exists()
    
    # 获取上次训练的用户列表
    last_users = []
    if BASE_MODEL_USERS_FILE.exists():
        with open(BASE_MODEL_USERS_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                last_users = content.split(",")
    
    # 获取当前所有用户（去重后的唯一 username，兼容旧 session 目录）
    current_users = get_uniq_usernames()
    
    needs_retrain = set(current_users) != set(last_users) if last_users else True
    
    return jsonify({
        "has_base_model": exists,
        "last_trained_users": last_users,
        "current_users": current_users,
        "needs_retrain": needs_retrain and exists,
        "message": "基础模型已就绪" if exists else "首次训练将创建基础模型"
    })


# ==================== API: 手动触发基础模型重新训练 ====================
@app.route("/api/retrain_base_model", methods=["POST"])
@token_required
def retrain_base_model_api():
    """手动触发基础模型重新训练（管理员功能）"""
    if not BASE_MODEL_RETRAIN_LOCK.acquire(blocking=False):
        return jsonify({"error": "基础模型正在重新训练中，请稍候"}), 409
    
    thread_started = False
    try:
        all_users = get_uniq_usernames()
        if not all_users:
            return jsonify({"error": "没有用户上传数据"}), 400
        
        task_id = generate_task_id()
        with tasks_lock:
            tasks[task_id] = {
                "status": "queued",
                "progress": "等待开始重新训练基础模型",
                "model_id": None,
                "labels": [],
                "accuracy": 0.0,
                "error": None,
                "created_at": datetime.now().isoformat(),
                "user_id": request.user_id,
                "model_type": "cnn_transformer",
                "is_base_model_retrain": True,
            }
        
        threading.Thread(
            target=run_base_model_retraining,
            args=(task_id, all_users),
            daemon=True
        ).start()
        thread_started = True
        
        return jsonify({
            "task_id": task_id,
            "status": "queued",
            "message": f"基础模型重新训练任务已提交，将使用 {len(all_users)} 个用户的数据"
        })
    finally:
        # 只有后台线程尚未开始（任务未生效 / 出错）时才由本接口释放锁；
        # 一旦线程启动，锁归后台线程持有，由 run_base_model_retraining 的 finally 释放，避免二次释放。
        if not thread_started:
            BASE_MODEL_RETRAIN_LOCK.release()


def run_base_model_retraining(task_id, users):
    """后台重新训练基础模型"""
    workspace = DATA_DIR / task_id
    try:
        set_progress(task_id, status="training", progress="开始重新训练基础模型...")
        
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True)
        
        user_dirs = collect_user_dirs(users)
        
        if not user_dirs:
            raise RuntimeError("未找到上传的数据")
        
        # 解析所有 CSV
        records = []
        found_labels = []
        for user_dir in user_dirs:
            for csv_file in sorted(user_dir.glob("*.csv")):
                label, _user, _session, file_protocol, hand, rows = parse_recording(csv_file)
                if not label or not rows:
                    continue
                if label not in found_labels:
                    found_labels.append(label)
                records.append((user_dir.name, label, rows, hand))
        
        if not records:
            raise RuntimeError("上传的数据为空或格式无法解析")
        
        label_order = build_label_order(found_labels)
        if len(label_order) < 2:
            raise RuntimeError(f"至少需要 2 种不同动作，当前只解析到 {len(label_order)} 种")
        
        set_progress(task_id, progress=f"解析到 {len(records)} 个文件，{len(label_order)} 类")
        
        # 转换成训练脚本要求的 txt 结构
        data_root = workspace / "Sep_EMG_Data"
        counters = {}
        record_hand = {}   # 记录级手别：相对路径 -> left/right
        for user_name, label, rows, hand in records:
            gesture_dir = data_root / user_name / f"gesture_{label_order.index(label)}"
            gesture_dir.mkdir(parents=True, exist_ok=True)
            key = (user_name, label)
            counters[key] = counters.get(key, 0) + 1
            target = gesture_dir / f"{counters[key]}.txt"
            with open(target, "w", encoding="utf-8") as out:
                for row in rows:
                    out.write(",".join(f"{value:.2f}" for value in row) + "\n")
            if hand:
                record_hand[f"{user_name}/gesture_{label_order.index(label)}/{target.name}"] = hand
        
        set_progress(task_id, progress="数据转换完成，开始训练基础模型...")

        # 合并元数据：users 为性别/体重（用户级），records 为手别（记录级）
        meta_all = {"users": {}, "records": record_hand}
        for ud in user_dirs:
            src_meta = ud / "user_meta.json"
            if src_meta.exists():
                try:
                    m = json.loads(src_meta.read_text(encoding="utf-8"))
                    meta_all["users"][ud.name] = {
                        "gender": m.get("gender"),
                        "weight": m.get("weight"),
                        "hand": m.get("hand"),
                    }
                except Exception:
                    pass
        (data_root / "meta.json").write_text(
            json.dumps(meta_all, ensure_ascii=False), encoding="utf-8"
        )

        # 调用训练脚本（base 模式）
        train_script = BASE_DIR / "train_cnn_transformer.py"
        if not train_script.exists():
            raise RuntimeError(f"训练脚本不存在: {train_script}")
        
        result = subprocess.run(
            [sys.executable, str(train_script.resolve()), "--mode", "base"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=1800,  # 基础模型训练可能需要更长时间
        )
        
        print(f"[{task_id}] 基础模型训练 stdout 尾部:\n{result.stdout[-2000:]}", flush=True)
        if result.returncode != 0:
            raise RuntimeError("训练失败：" + (result.stderr or result.stdout)[-500:])
        
        # 保存基础模型到服务器目录
        best_base = workspace / "best_lx_model_transformer.pth"
        if best_base.exists():
            shutil.copy2(best_base, BASE_MODEL_PATH)

            # 更新用户列表
            with open(BASE_MODEL_USERS_FILE, "w", encoding="utf-8") as f:
                f.write(",".join(users))

            # 复制 scripted .pt 和元数据供零样本下载
            best_pt = workspace / "best_lx_model_scripted.pt"
            if best_pt.exists():
                shutil.copy2(best_pt, BASE_MODEL_SCRIPTED_PATH)
            for src_name, dst_path in [
                ("norm_stats.json",    BASE_MODEL_NORM_STATS),
                ("filter_coeffs.json", BASE_MODEL_FILTER_COEFFS),
            ]:
                src = workspace / src_name
                if src.exists():
                    shutil.copy2(src, dst_path)

            set_progress(task_id, status="completed", progress="基础模型重新训练完成")
        else:
            raise RuntimeError("训练未产出基础模型文件")
        
    except Exception as e:
        set_progress(task_id, status="failed", error=str(e))
        print(f"[{task_id}] 基础模型重新训练失败: {e}", flush=True)
    finally:
        # 锁只释放一次；用 locked() 保证幂等，避免并发/版本问题导致 release unlocked lock
        if BASE_MODEL_RETRAIN_LOCK.locked():
            BASE_MODEL_RETRAIN_LOCK.release()


# ==================== API: 触发训练 ====================
@app.route("/api/train", methods=["POST"])
@token_required
def start_training():
    payload = request.get_json(silent=True) or {}
    users = [request.username]  # 只训练当前用户的数据
    protocol = (payload.get("protocol") or "").strip()
    model_type = (payload.get("model_type") or "cnn1d").strip().lower()  # cnn1d | cnn_transformer

    task_id = generate_task_id()
    with tasks_lock:
        tasks[task_id] = {
            "status": "queued",
            "progress": "等待开始",
            "model_id": None,
            "labels": [],
            "accuracy": 0.0,
            "error": None,
            "created_at": datetime.now().isoformat(),
            "user_id": request.user_id,
            "model_type": model_type,
        }

    threading.Thread(target=run_training, args=(task_id, users, protocol, model_type), daemon=True).start()
    return jsonify({"task_id": task_id, "status": "queued", "message": "训练任务已提交"})


def set_progress(task_id, **kwargs):
    with tasks_lock:
        if task_id in tasks:
            tasks[task_id].update(kwargs)


def check_and_schedule_base_model_retrain():
    """检查是否需要自动重新训练基础模型（方案 1：定期更新）"""
    if not BASE_MODEL_RETRAIN_LOCK.acquire(blocking=False):
        print("[基础模型] 正在重新训练中，跳过本次检查", flush=True)
        return
    
    try:
        current_users = get_uniq_usernames()
        if not current_users:
            return
        
        # 获取上次训练的用户列表
        last_users = []
        if BASE_MODEL_USERS_FILE.exists():
            with open(BASE_MODEL_USERS_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    last_users = content.split(",")
        
        # 检查用户列表是否变化
        if set(current_users) == set(last_users):
            print("[基础模型] 用户列表未变化，无需重新训练", flush=True)
            return
        
        # 用户列表变化了，需要重新训练
        print(f"[基础模型] 检测到用户变化：旧={last_users}, 新={current_users}，开始重新训练", flush=True)
        
        # 在后台线程中执行重新训练
        task_id = generate_task_id()
        with tasks_lock:
            tasks[task_id] = {
                "status": "queued",
                "progress": "自动检测用户变化，开始重新训练基础模型",
                "model_id": None,
                "labels": [],
                "accuracy": 0.0,
                "error": None,
                "created_at": datetime.now().isoformat(),
                "user_id": None,
                "model_type": "cnn_transformer",
                "is_base_model_retrain": True,
                "is_auto_retrain": True,
            }
        
        threading.Thread(
            target=run_base_model_retraining,
            args=(task_id, current_users),
            daemon=True
        ).start()
        
    finally:
        # 锁只释放一次；用 locked() 保证幂等，避免并发/版本问题导致 release unlocked lock
        if BASE_MODEL_RETRAIN_LOCK.locked():
            BASE_MODEL_RETRAIN_LOCK.release()


def run_training(task_id, users, protocol, model_type="cnn1d"):
    """后台训练：CSV → txt → 调用 train_model_sep.py → 收集 .pt"""
    workspace = DATA_DIR / task_id
    try:
        set_progress(task_id, status="training", progress="扫描上传数据...")
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True)

        # 1. 收集待训练的用户目录（兼容旧 session 目录）
        user_dirs = collect_user_dirs(users if users else None)

        if not user_dirs:
            raise RuntimeError("未找到上传的数据，请先上传采集 CSV")

        # 2. 解析所有 CSV，确定类别顺序
        records = []          # [(user_name, label, session, rows, hand)]
        found_labels = []
        for user_dir in user_dirs:
            for csv_file in sorted(user_dir.glob("*.csv")):
                label, _user, session, file_protocol, hand, rows = parse_recording(csv_file)
                if protocol and file_protocol and file_protocol != protocol:
                    continue
                if not label or not rows:
                    continue
                if label not in found_labels:
                    found_labels.append(label)
                records.append((user_dir.name, label, session or "default", rows, hand))

        if not records:
            raise RuntimeError("上传的数据为空或格式无法解析")

        label_order = build_label_order(found_labels)
        if len(label_order) < 2:
            raise RuntimeError(
                f"至少需要 2 种不同动作，当前只解析到 {len(label_order)} 种：{label_order}")
        if len(found_labels) > MAX_CLASSES:
            raise RuntimeError(
                f"最多支持 {MAX_CLASSES} 种动作，当前有 {len(found_labels)} 种：{found_labels}")

        set_progress(task_id, progress=f"解析到 {len(records)} 个文件，{len(label_order)} 类：{'/'.join(label_order)}",
                     labels=label_order)

        # 3. 转换成训练脚本要求的 txt 结构：Sep_EMG_Data/<user>/session_<n>/gesture_<N>/<m>.txt
        #    session_<n> 用 session 去重后的序号映射，同一用户同一 session 映射到同一个 n
        #    向后兼容：若 CSV 没有 session 元数据，统一归入 session_default
        data_root = workspace / "Sep_EMG_Data"
        session_id_map = {}
        counters = {}
        record_hand = {}   # 记录级手别：相对路径 -> left/right
        for user_name, label, session, rows, hand in records:
            sess_key = (user_name, session)
            if sess_key not in session_id_map:
                session_id_map[sess_key] = len(
                    [k for k in session_id_map if k[0] == user_name]
                )
            session_int = session_id_map[sess_key]
            gesture_dir = (
                data_root / user_name / f"session_{session_int}"
                / f"gesture_{label_order.index(label)}"
            )
            gesture_dir.mkdir(parents=True, exist_ok=True)
            key = (user_name, label, session_int)
            counters[key] = counters.get(key, 0) + 1
            target = gesture_dir / f"{counters[key]}.txt"
            with open(target, "w", encoding="utf-8") as out:
                for row in rows:
                    out.write(",".join(f"{value:.2f}" for value in row) + "\n")
            if hand:
                record_hand[
                    f"{user_name}/session_{session_int}/gesture_{label_order.index(label)}/{target.name}"
                ] = hand

        set_progress(task_id, progress="数据转换完成，开始训练...")

        # 合并元数据：users 为性别/体重（用户级），records 为手别（记录级）
        meta_all = {"users": {}, "records": record_hand}
        for ud in user_dirs:
            src_meta = ud / "user_meta.json"
            if src_meta.exists():
                try:
                    m = json.loads(src_meta.read_text(encoding="utf-8"))
                    meta_all["users"][ud.name] = {
                        "gender": m.get("gender"),
                        "weight": m.get("weight"),
                        "hand": m.get("hand"),
                    }
                except Exception:
                    pass
        (data_root / "meta.json").write_text(
            json.dumps(meta_all, ensure_ascii=False), encoding="utf-8"
        )

        # 4. 根据模型类型选择训练脚本
        if model_type == "cnn_transformer":
            train_script = BASE_DIR / "train_cnn_transformer.py"
        else:
            train_script = TRAIN_SCRIPT
        
        if not train_script.exists():
            raise RuntimeError(f"训练脚本不存在: {train_script}")
        
        # 调用训练脚本（必须用当前解释器，系统的 python 是 3.6 没装 torch）
        # 使用绝对路径，避免 cwd 切换后找不到脚本
        
        # CNN-Transformer 两阶段训练逻辑
        if model_type == "cnn_transformer":
            # 检查是否有基础模型
            base_model_path = BASE_DIR / "base_model_transformer.pth"
            
            if base_model_path.exists():
                # 阶段 1 已完成：有基础模型，直接微调
                set_progress(task_id, progress="检测到基础模型，开始用户微调...")
                result = subprocess.run(
                    [sys.executable, str(train_script.resolve()), "--mode", "finetune",
                     "--pretrained", str(base_model_path.resolve()),
                     "--subject", users[0] if users else ""],
                    cwd=str(workspace),
                    capture_output=True,
                    text=True,
                    timeout=900,
                )
                print(f"[{task_id}] 微调返回码: {result.returncode}", flush=True)
                print(f"[{task_id}] 微调 stdout 尾部:\n{result.stdout[-2000:]}", flush=True)
                if result.stderr:
                    print(f"[{task_id}] 微调 stderr:\n{result.stderr[-2000:]}", flush=True)
            else:
                # 首次训练：用当前用户数据训练初始基础模型（单用户模式，不启用GRL/对比学习）
                set_progress(task_id, progress="首次训练，创建初始基础模型...")
                print(f"[{task_id}] 开始执行训练脚本: {train_script}", flush=True)
                print(f"[{task_id}] 工作目录: {workspace}", flush=True)
                print(f"[{task_id}] Python 解释器: {sys.executable}", flush=True)
                result = subprocess.run(
                    [sys.executable, str(train_script.resolve()), "--mode", "base",
                     "--no_grl", "--no_contra"],
                    cwd=str(workspace),
                    capture_output=True,
                    text=True,
                    timeout=900,
                )
                print(f"[{task_id}] 训练脚本执行完成，返回码: {result.returncode}", flush=True)
                print(f"[{task_id}] stdout 尾部:\n{result.stdout[-2000:]}", flush=True)
                if result.stderr:
                    print(f"[{task_id}] stderr:\n{result.stderr[-1000:]}", flush=True)
                
                # 如果基础模型训练成功，保存基础模型到服务器目录
                if result.returncode == 0:
                    best_base = workspace / "best_lx_model_transformer.pth"
                    print(f"[{task_id}] 检查基础模型文件: {best_base}, exists={best_base.exists()}", flush=True)
                    if best_base.exists():
                        shutil.copy2(best_base, base_model_path)
                        print(f"[{task_id}] 基础模型已复制到: {base_model_path}", flush=True)

                        # 同步导出零样本下载所需的 scripted .pt 与元数据
                        best_scripted = workspace / "best_lx_model_scripted.pt"
                        if best_scripted.exists():
                            shutil.copy2(best_scripted, BASE_MODEL_SCRIPTED_PATH)
                        for src_name, dst_path in [
                            ("norm_stats.json",    BASE_MODEL_NORM_STATS),
                            ("filter_coeffs.json", BASE_MODEL_FILTER_COEFFS),
                        ]:
                            src = workspace / src_name
                            if src.exists():
                                shutil.copy2(src, dst_path)
                        print(f"[{task_id}] 零样本基础模型已导出: {BASE_MODEL_SCRIPTED_PATH}", flush=True)
                        
                        # 更新用户列表
                        all_users = get_uniq_usernames()
                        with open(BASE_MODEL_USERS_FILE, "w", encoding="utf-8") as f:
                            f.write(",".join(all_users))
                        print(f"[{task_id}] 用户列表已更新: {all_users}", flush=True)
                        
                        set_progress(task_id, progress="基础模型已保存，开始用户微调...")
                        print(f"[{task_id}] 开始用户微调...", flush=True)
                        
                        # 阶段 2：用户微调
                        result = subprocess.run(
                            [sys.executable, str(train_script.resolve()), "--mode", "finetune",
                             "--pretrained", str(base_model_path.resolve()),
                             "--subject", users[0] if users else ""],
                            cwd=str(workspace),
                            capture_output=True,
                            text=True,
                            timeout=900,
                        )
                        print(f"[{task_id}] 微调 stdout 尾部:\n{result.stdout[-2000:]}", flush=True)
                        if result.returncode != 0:
                            print(f"[{task_id}] 微调 stderr:\n{result.stderr[-1000:]}", flush=True)
                            raise RuntimeError("用户微调失败：" + (result.stderr or result.stdout)[-500:])
                        
                        print(f"[{task_id}] 用户微调完成", flush=True)
                    else:
                        raise RuntimeError(f"基础模型文件不存在: {best_base}")
                else:
                    raise RuntimeError("基础模型训练失败：" + (result.stderr or result.stdout)[-500:])
        else:
            # CNN1D 原始训练逻辑
            result = subprocess.run(
                [sys.executable, str(train_script.resolve())],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=900,
            )
        print(f"[{task_id}] 训练 stdout 尾部:\n{result.stdout[-2000:]}", flush=True)
        if result.returncode != 0:
            raise RuntimeError("训练失败：" + (result.stderr or result.stdout)[-500:])

        accuracy = 0.0
        match = re.search(r"最佳准确率[:：]\s*([\d.]+)", result.stdout)
        if match:
            accuracy = float(match.group(1))

        # 5. 收集导出的 TorchScript（优先 best）和归一化统计
        best_pt = workspace / "best_lx_model_scripted.pt"
        plain_pt = workspace / "lx_model_scripted.pt"
        pt_file = best_pt if best_pt.exists() else plain_pt
        if not pt_file.exists():
            raise RuntimeError("训练未产出 .pt 文件，请检查训练日志")

        # 读取归一化统计（训练脚本输出的 norm_stats.json）
        norm_stats = None
        norm_file = workspace / "norm_stats.json"
        if norm_file.exists():
            with open(norm_file, "r", encoding="utf-8") as nf:
                norm_stats = json.load(nf)

        # 读取滤波器系数（训练脚本输出的 filter_coeffs.json）
        # 手机端直接复用这组 b/a，避免自行重算导致幅度不一致
        filter_coeffs = None
        filter_file = workspace / "filter_coeffs.json"
        if filter_file.exists():
            with open(filter_file, "r", encoding="utf-8") as ff:
                filter_coeffs = json.load(ff)

        model_id = "model_" + task_id
        shutil.copy2(pt_file, MODEL_DIR / f"{model_id}.pt")

        metadata = {
            "model_id": model_id,
            "created_at": datetime.now().isoformat(),
            "protocol": protocol or "",
            "model_type": model_type,
            "labels": label_order,          # 类别下标顺序，手机端据此建立映射
            "accuracy": round(accuracy, 2),
            "users": [item.name for item in user_dirs],
            "source_file": pt_file.name,
            "filename": f"{model_id}.pt",
        }
        if norm_stats:
            metadata["normStats"] = norm_stats
        if filter_coeffs:
            metadata["filterCoeffs"] = filter_coeffs
        with open(MODEL_DIR / f"{model_id}.json", "w", encoding="utf-8") as out:
            json.dump(metadata, out, ensure_ascii=False, indent=2)

        set_progress(task_id, status="done", progress=f"训练完成，准确率 {accuracy:.2f}%",
                     model_id=model_id, accuracy=round(accuracy, 2), labels=label_order)

        # 6. 检查是否需要自动重新训练基础模型（方案 1）
        if model_type == "cnn_transformer":
            check_and_schedule_base_model_retrain()

    except subprocess.TimeoutExpired:
        set_progress(task_id, status="failed", error="训练超时（>15 分钟）")
    except Exception as error:
        set_progress(task_id, status="failed", error=str(error)[:400])
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


# ==================== API: 保存用户模型 ====================
@app.route("/api/save_user_model", methods=["POST"])
@token_required
def save_user_model():
    """
    训练完成后，用户选择是否保存模型到个人文件夹。
    保存格式：用户_时间戳.pt + 用户_时间戳.json
    """
    payload = request.get_json(silent=True) or {}
    model_id = payload.get("model_id", "").strip()
    
    if not model_id:
        return jsonify({"error": "缺少 model_id 参数"}), 400
    
    user = request.username
    user_dir = UPLOAD_DIR / safe_name(user, "user")
    user_dir.mkdir(parents=True, exist_ok=True)
    
    pt_file = MODEL_DIR / f"{model_id}.pt"
    json_file = MODEL_DIR / f"{model_id}.json"
    
    if not pt_file.exists():
        return jsonify({"error": "模型文件不存在"}), 404
    
    # 生成保存名称：用户_时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"{safe_name(user)}_{timestamp}"
    
    # 复制 .pt 文件
    save_pt = user_dir / f"{save_name}.pt"
    shutil.copy2(pt_file, save_pt)
    
    # 复制 .json 元数据
    save_json = user_dir / f"{save_name}.json"
    if json_file.exists():
        shutil.copy2(json_file, save_json)
    
    return jsonify({
        "message": f"模型已保存为 {save_name}",
        "save_name": save_name,
        "user": user,
        "timestamp": timestamp,
    })


# ==================== API: 列出用户保存的模型 ====================
@app.route("/api/user_models", methods=["GET"])
@token_required
def list_user_models():
    """
    列出当前用户保存的所有模型。
    返回：{"user": "xxx", "models": [{"name": "xxx_20240101_120000", "timestamp": "...", "accuracy": 0.98}, ...]}
    """
    user = request.username
    user_dir = UPLOAD_DIR / safe_name(user, "user")
    if not user_dir.exists():
        return jsonify({"user": user, "models": []})
    
    models = []
    for json_file in sorted(user_dir.glob("*_*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            models.append({
                "name": json_file.stem,
                "timestamp": meta.get("created_at", ""),
                "accuracy": meta.get("accuracy", 0.0),
                "labels": meta.get("labels", []),
            })
        except Exception:
            # 如果 JSON 解析失败，只从文件名提取信息
            name = json_file.stem
            # 从文件名提取时间戳部分
            parts = name.rsplit("_", 2)
            if len(parts) >= 2:
                ts = f"{parts[-2]}_{parts[-1]}"
            else:
                ts = ""
            models.append({
                "name": name,
                "timestamp": ts,
                "accuracy": 0.0,
                "labels": [],
            })
    
    return jsonify({"user": user, "models": models})


# ==================== API: 下载用户保存的模型 ====================
@app.route("/api/download_user_model/<model_name>", methods=["GET"])
@token_required
def download_user_model(model_name):
    """
    下载当前用户保存的特定模型。
    """
    user = request.username
    user_dir = UPLOAD_DIR / safe_name(user, "user")
    pt_file = user_dir / f"{model_name}.pt"
    json_file = user_dir / f"{model_name}.json"
    
    if not pt_file.exists():
        return jsonify({"error": "模型不存在"}), 404
    
    labels = []
    norm_stats = None
    filter_coeffs = None
    if json_file.exists():
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                labels = meta.get("labels", [])
                norm_stats = meta.get("normStats")
                filter_coeffs = meta.get("filterCoeffs")
        except Exception:
            pass
    
    response = send_file(
        pt_file,
        as_attachment=True,
        download_name=f"{model_name}.pt",
        mimetype="application/octet-stream",
    )
    response.headers["X-Model-Labels"] = quote(",".join(labels), safe="")
    if norm_stats:
        response.headers["X-Model-NormStats"] = quote(json.dumps(norm_stats), safe="")
    if filter_coeffs:
        response.headers["X-Model-FilterCoeffs"] = quote(json.dumps(filter_coeffs), safe="")
    return response


# ==================== API: 零样本下载基础模型 ====================
@app.route("/api/download_base_model", methods=["GET"])
@token_required
def download_base_model():
    """零样本下载：直接返回基础模型 scripted .pt（无需训练）"""
    if not BASE_MODEL_SCRIPTED_PATH.exists():
        return jsonify({"error": "基础模型尚未训练，请等待管理员重训"}), 404

    labels = CANONICAL_LABELS
    norm_stats = None
    if BASE_MODEL_NORM_STATS.exists():
        try:
            norm_stats = json.loads(BASE_MODEL_NORM_STATS.read_text(encoding="utf-8"))
        except Exception:
            pass
    filter_coeffs = None
    if BASE_MODEL_FILTER_COEFFS.exists():
        try:
            filter_coeffs = json.loads(BASE_MODEL_FILTER_COEFFS.read_text(encoding="utf-8"))
        except Exception:
            pass

    response = send_file(
        BASE_MODEL_SCRIPTED_PATH,
        as_attachment=True,
        download_name="base_model_scripted.pt",
        mimetype="application/octet-stream",
    )
    response.headers["X-Model-Labels"] = quote(",".join(labels), safe="")
    if norm_stats:
        response.headers["X-Model-NormStats"] = quote(json.dumps(norm_stats), safe="")
    if filter_coeffs:
        response.headers["X-Model-FilterCoeffs"] = quote(json.dumps(filter_coeffs), safe="")
    return response


# ==================== API: 查询状态 ====================
@app.route("/api/status/<task_id>", methods=["GET"])
@token_required
def get_status(task_id):
    with tasks_lock:
        task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "任务不存在"}), 404
    # 验证任务属于当前用户
    if task.get('user_id') != request.user_id:
        return jsonify({"error": "无权访问此任务"}), 403
    return jsonify(dict(task, task_id=task_id))


# ==================== API: 下载模型 ====================
@app.route("/api/model/<model_id>", methods=["GET"])
@app.route("/api/download/<model_id>", methods=["GET"])
@token_required
def download_model(model_id):
    pt_file = MODEL_DIR / f"{model_id}.pt"
    if not pt_file.exists():
        return jsonify({"error": "模型不存在"}), 404

    meta_file = MODEL_DIR / f"{model_id}.json"
    labels = []
    norm_stats = None
    filter_coeffs = None
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
            labels = meta.get("labels", [])
            norm_stats = meta.get("normStats")
            filter_coeffs = meta.get("filterCoeffs")

    response = send_file(
        pt_file,
        as_attachment=True,
        download_name=f"{model_id}.pt",
        mimetype="application/octet-stream",
    )
    # HTTP 头只能是 Latin-1，标签可能是中文，故做 URL 编码，手机端解码
    response.headers["X-Model-Labels"] = quote(",".join(labels), safe="")
    # 归一化统计（per-channel mean/std），JSON 编码后 URL 编码
    if norm_stats:
        response.headers["X-Model-NormStats"] = quote(json.dumps(norm_stats), safe="")
    # 滤波器系数（b/a），手机端直接复用，保证与训练端滤波完全一致
    if filter_coeffs:
        response.headers["X-Model-FilterCoeffs"] = quote(json.dumps(filter_coeffs), safe="")
    return response


# ==================== API: 列出模型 ====================
@app.route("/api/models", methods=["GET"])
def list_models():
    models = []
    for meta_file in sorted(MODEL_DIR.glob("*.json")):
        with open(meta_file, "r", encoding="utf-8") as handle:
            models.append(json.load(handle))
    return jsonify({"models": models})


# ==================== API: 查询用户已有文件列表 ====================
@app.route("/api/files", methods=["GET"])
@token_required
def list_user_files():
    """
    查询当前用户已上传的文件列表，用于增量上传时判断哪些文件需要上传。
    返回：{"user": "xxx", "files": ["file1.csv", "file2.csv"], "count": 2}
    """
    user = request.username
    user_dir = UPLOAD_DIR / safe_name(user, "user")
    if not user_dir.exists():
        return jsonify({"user": user, "files": [], "count": 0})
    
    files = sorted([f.name for f in user_dir.glob("*.csv")])
    return jsonify({"user": user, "files": files, "count": len(files)})


# ==================== API: 删除用户模型 ====================
@app.route("/api/user_models/<model_name>", methods=["DELETE"])
@token_required
def delete_user_model(model_name):
    """
    删除当前用户保存的特定模型。
    """
    user = request.username
    user_dir = UPLOAD_DIR / safe_name(user, "user")
    
    pt_file = user_dir / f"{model_name}.pt"
    json_file = user_dir / f"{model_name}.json"
    
    deleted = []
    if pt_file.exists():
        pt_file.unlink()
        deleted.append(f"{model_name}.pt")
    if json_file.exists():
        json_file.unlink()
        deleted.append(f"{model_name}.json")
    
    if not deleted:
        return jsonify({"error": "模型不存在"}), 404
    
    return jsonify({
        "message": f"已删除模型: {model_name}",
        "deleted_files": deleted,
        "user": user,
    })


# ==================== API: 删除用户上传数据 ====================
@app.route("/api/user_data", methods=["DELETE"])
@token_required
def delete_user_data():
    """
    删除当前用户的所有上传数据（CSV 文件）。
    可选参数：session（只删除特定 session 的数据）
    """
    payload = request.get_json(silent=True) or {}
    user = request.username
    session = payload.get("session", "").strip()
    
    if session:
        # 只删除特定 session 的数据
        session_dir = UPLOAD_DIR / safe_name(session)
        if not session_dir.exists():
            return jsonify({"error": f"Session 不存在: {session}"}), 404
        
        csv_files = list(session_dir.glob("*.csv"))
        for f in csv_files:
            f.unlink()
        session_dir.rmdir()
        
        return jsonify({
            "message": f"已删除 session 数据: {session}",
            "deleted_files": len(csv_files),
            "user": user,
        })
    else:
        # 删除用户所有 session 数据
        user_sessions = []
        for d in UPLOAD_DIR.iterdir():
            if d.is_dir() and d.name.startswith(safe_name(user)):
                user_sessions.append(d)
        
        total_files = 0
        for session_dir in user_sessions:
            csv_files = list(session_dir.glob("*.csv"))
            for f in csv_files:
                f.unlink()
            session_dir.rmdir()
            total_files += len(csv_files)
        
        return jsonify({
            "message": f"已删除用户所有数据",
            "deleted_sessions": len(user_sessions),
            "deleted_files": total_files,
            "user": user,
        })


# ==================== API: 删除用户所有数据（包括模型） ====================
@app.route("/api/user/reset", methods=["POST"])
@token_required
def reset_user():
    """
    重置当前用户：删除所有上传数据 + 所有保存的模型。
    需要确认参数：confirm=True
    """
    payload = request.get_json(silent=True) or {}
    if not payload.get("confirm"):
        return jsonify({
            "error": "需要确认操作",
            "message": "请设置 confirm=true 以确认删除所有数据",
        }), 400
    
    user = request.username
    user_dir = UPLOAD_DIR / safe_name(user, "user")
    
    # 1. 删除所有 session 数据
    deleted_sessions = 0
    deleted_files = 0
    for d in UPLOAD_DIR.iterdir():
        if d.is_dir() and d.name.startswith(safe_name(user)):
            csv_files = list(d.glob("*.csv"))
            for f in csv_files:
                f.unlink()
            d.rmdir()
            deleted_sessions += 1
            deleted_files += len(csv_files)
    
    # 2. 删除所有保存的模型
    deleted_models = 0
    if user_dir.exists():
        model_files = list(user_dir.glob("*"))
        for f in model_files:
            f.unlink()
        deleted_models = len(model_files)
    
    return jsonify({
        "message": "用户数据已重置",
        "user": user,
        "deleted_sessions": deleted_sessions,
        "deleted_files": deleted_files,
        "deleted_models": deleted_models,
    })


# ==================== API: 获取用户信息 ====================
@app.route("/api/user/info", methods=["GET"])
@token_required
def get_user_info():
    """
    获取当前用户的统计信息。
    返回：session 数量、文件数量、模型数量等
    """
    user = request.username
    user_dir = UPLOAD_DIR / safe_name(user, "user")
    
    # 统计 session 数据
    sessions = []
    total_files = 0
    for d in UPLOAD_DIR.iterdir():
        if d.is_dir() and d.name.startswith(safe_name(user)):
            csv_files = list(d.glob("*.csv"))
            sessions.append({
                "name": d.name,
                "files": len(csv_files),
            })
            total_files += len(csv_files)
    
    # 统计模型数据
    models = []
    if user_dir.exists():
        for json_file in user_dir.glob("*_.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                models.append({
                    "name": json_file.stem,
                    "accuracy": meta.get("accuracy", 0.0),
                    "labels": meta.get("labels", []),
                })
            except Exception:
                pass
    
    return jsonify({
        "user": user,
        "sessions": sessions,
        "total_files": total_files,
        "models": models,
        "total_models": len(models),
    })


# ==================== API: 管理员 - 查询所有用户 ====================
@app.route("/api/admin/users", methods=["GET"])
@admin_required
def admin_list_users():
    """
    管理员专用：查询所有用户及其统计信息。
    需要管理员密码：admin_password=zhiwei8888
    """
    conn = get_db()
    users = conn.execute("SELECT id, username, email, is_admin, created_at, last_login, status FROM users ORDER BY created_at").fetchall()
    conn.close()
    
    result = []
    for user in users:
        user_name = user['username']
        user_dir = UPLOAD_DIR / safe_name(user_name, "user")
        
        # 统计 session 数据
        session_count = 0
        total_files = 0
        for d in UPLOAD_DIR.iterdir():
            if d.is_dir() and d.name.startswith(safe_name(user_name)):
                csv_files = list(d.glob("*.csv"))
                session_count += 1
                total_files += len(csv_files)
        
        # 统计模型数据
        model_count = 0
        if user_dir.exists():
            model_count = len(list(user_dir.glob("*.pt")))
        
        result.append({
            "id": user['id'],
            "username": user['username'],
            "email": user['email'],
            "is_admin": bool(user['is_admin']),
            "created_at": user['created_at'],
            "last_login": user['last_login'],
            "status": user['status'],
            "stats": {
                "sessions": session_count,
                "files": total_files,
                "models": model_count,
            }
        })
    
    return jsonify({"users": result, "total": len(result)})


# ==================== API: 管理员 - 删除用户 ====================
@app.route("/api/admin/users/<int:user_id>", methods=["DELETE"])
@admin_required
def admin_delete_user(user_id):
    """
    管理员专用：删除指定用户及其所有数据。
    需要管理员密码：admin_password=zhiwei8888
    
    删除内容：
    1. 数据库中的用户记录
    2. 用户上传的所有 CSV 数据（所有 session）
    3. 用户保存的所有模型文件
    """
    conn = get_db()
    user = conn.execute("SELECT id, username, is_admin FROM users WHERE id = ?", (user_id,)).fetchone()
    
    if not user:
        conn.close()
        return jsonify({"error": "用户不存在"}), 404
    
    # 不允许删除管理员账号
    if user['is_admin']:
        conn.close()
        return jsonify({"error": "不能删除管理员账号"}), 400
    
    user_name = user['username']
    user_dir = UPLOAD_DIR / safe_name(user_name, "user")
    
    # 1. 删除所有 session 数据
    deleted_sessions = 0
    deleted_files = 0
    for d in UPLOAD_DIR.iterdir():
        if d.is_dir() and d.name.startswith(safe_name(user_name)):
            csv_files = list(d.glob("*.csv"))
            for f in csv_files:
                f.unlink()
            d.rmdir()
            deleted_sessions += 1
            deleted_files += len(csv_files)
    
    # 2. 删除所有模型文件
    deleted_models = 0
    if user_dir.exists():
        model_files = list(user_dir.glob("*"))
        for f in model_files:
            f.unlink()
        deleted_models = len(model_files)
    
    # 3. 删除数据库记录
    conn.execute("DELETE FROM user_files WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM user_models WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    
    return jsonify({
        "message": f"用户 {user_name} 已删除",
        "deleted_user": user_name,
        "deleted_sessions": deleted_sessions,
        "deleted_files": deleted_files,
        "deleted_models": deleted_models,
    })


# ==================== API: 管理员 - 设置管理员账号 ====================
@app.route("/api/admin/set_admin", methods=["POST"])
@admin_required
def admin_set_admin():
    """
    管理员专用：将指定用户设置为管理员或取消管理员权限。
    需要管理员密码：admin_password=zhiwei8888
    """
    payload = request.get_json(silent=True) or {}
    username = payload.get("username", "").strip()
    is_admin = payload.get("is_admin", False)
    
    if not username:
        return jsonify({"error": "缺少 username 参数"}), 400
    
    conn = get_db()
    user = conn.execute("SELECT id, username FROM users WHERE username = ?", (username,)).fetchone()
    
    if not user:
        conn.close()
        return jsonify({"error": "用户不存在"}), 404
    
    conn.execute("UPDATE users SET is_admin = ? WHERE username = ?", 
                (1 if is_admin else 0, username))
    conn.commit()
    conn.close()
    
    action = "设置为管理员" if is_admin else "取消管理员权限"
    return jsonify({
        "message": f"用户 {username} 已{action}",
        "username": username,
        "is_admin": is_admin,
    })


# ==================== 健康检查 ====================
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


# ==================== API: 管理员 - 全部模型管理 ====================

def _scan_all_models():
    """
    统一扫描所有存储位置的模型：
      1) MODEL_DIR/*.json           → 训练产出的原始模型
      2) UPLOAD_DIR/*/*.json        → 用户保存的模型副本
    返回统一结构列表，每项都带 storage 标识和完整路径。
    """
    all_models = []

    # --- 位置 A: 训练产出 (models/) ---
    if MODEL_DIR.exists():
        for meta_file in sorted(MODEL_DIR.glob("*.json")):
            try:
                with open(meta_file, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                pt_path = MODEL_DIR / f"{meta_file.stem}.pt"
                users = meta.get("users", [])
                if isinstance(users, list) and len(users) > 0:
                    owner = users[0] if len(users) == 1 else f"{users[0]} 等{len(users)}人"
                else:
                    owner = "未知用户"
                all_models.append({
                    "key": f"models/{meta_file.stem}",
                    "storage": "train",
                    "model_id": meta_file.stem,
                    "owner": owner,
                    "has_pt": pt_path.exists(),
                    "pt_size": pt_path.stat().st_size if pt_path.exists() else 0,
                    "accuracy": meta.get("accuracy", 0),
                    "model_type": meta.get("model_type", ""),
                    "labels": meta.get("labels", []),
                    "created_at": meta.get("created_at", ""),
                    "pt_path": str(pt_path),
                    "json_path": str(meta_file),
                })
            except Exception:
                pass

    # --- 位置 B: 用户保存副本 (uploads/<user>/*.json) ---
    if UPLOAD_DIR.exists():
        for user_dir in UPLOAD_DIR.iterdir():
            if not user_dir.is_dir():
                continue
            for meta_file in sorted(user_dir.glob("*.json")):
                # 跳过非模型的 json（比如 CSV 的索引或其它元数据）
                # 判断标准：同目录下存在同名 .pt，或 json 里有 labels 字段
                pt_path = user_dir / f"{meta_file.stem}.pt"
                try:
                    with open(meta_file, "r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                    is_model = pt_path.exists() or "labels" in meta or "accuracy" in meta
                    if not is_model:
                        continue
                    all_models.append({
                        "key": f"user/{user_dir.name}/{meta_file.stem}",
                        "storage": "user",
                        "model_id": meta_file.stem,
                        "owner": user_dir.name,
                        "has_pt": pt_path.exists(),
                        "pt_size": pt_path.stat().st_size if pt_path.exists() else 0,
                        "accuracy": meta.get("accuracy", 0),
                        "model_type": meta.get("model_type", ""),
                        "labels": meta.get("labels", []),
                        "created_at": meta.get("created_at", ""),
                        "pt_path": str(pt_path),
                        "json_path": str(meta_file),
                    })
                except Exception:
                    pass

    return all_models


@app.route("/api/admin/models", methods=["GET"])
@admin_required
def admin_list_all_models():
    """
    管理员：列出服务器上 **所有** 模型（训练产出 + 用户保存的副本）。
    """
    return jsonify({"models": _scan_all_models(), "total": len(_scan_all_models())})


def _delete_model_by_key(key):
    """
    根据 scan_all_models 返回的 key 精确删除模型文件。
    key 格式: 'models/<model_id>' 或 'user/<user_dir>/<model_stem>'
    返回 (bool success, list deleted_files)。
    """
    deleted_files = []
    parts = key.split("/", 2)
    if len(parts) < 2:
        return False, deleted_files

    if parts[0] == "models":
        target = MODEL_DIR / parts[1]
    elif parts[0] == "user" and len(parts) == 3:
        target = UPLOAD_DIR / parts[1] / parts[2]
    else:
        return False, deleted_files

    pt_file = target.with_suffix(".pt")
    json_file = target.with_suffix(".json")

    if pt_file.exists():
        pt_file.unlink()
        deleted_files.append(str(pt_file))
    if json_file.exists():
        json_file.unlink()
        deleted_files.append(str(json_file))

    if not deleted_files:
        return False, deleted_files
    return True, deleted_files


@app.route("/api/admin/models/batch", methods=["DELETE"])
@admin_required
def admin_delete_models_batch():
    """
    管理员：批量删除模型（支持训练模型 + 用户副本混合）。
    body: {"keys": ["models/model_task_xxx", "user/李鑫zw/李鑫ZW_20260917_111220"]}
    也兼容旧格式 {"model_ids": [...]} —— 仅删除 MODEL_DIR 下的训练模型。
    """
    payload = request.get_json(silent=True) or {}
    keys = payload.get("keys", [])

    # 兼容旧调用：只给了 model_id，默认删 MODEL_DIR 里的
    if not keys and payload.get("model_ids"):
        keys = [f"models/{mid}" for mid in payload["model_ids"]]

    if not keys:
        return jsonify({"error": "未指定要删除的模型"}), 400

    deleted = []
    not_found = []
    for k in keys:
        ok, files = _delete_model_by_key(k)
        if ok:
            deleted.append({"key": k, "files": files})
        else:
            not_found.append(k)

    return jsonify({
        "message": f"已删除 {len(deleted)} 个模型，{len(not_found)} 个未找到",
        "deleted": deleted,
        "not_found": not_found,
    })


@app.route("/api/admin/models/<path:key>", methods=["DELETE"])
@admin_required
def admin_delete_model(key):
    """
    管理员：删除单个模型。key 形如 'models/xxx' 或 'user/xxx/yyy'。
    """
    # URL 里 '/' 会被拆分，这里重新拼回去
    full_key = request.path.rsplit("/", 1)[-1]
    # 但上面 <path:key> 已经拿到了完整的后半段
    full_key = request.full_path.split("/api/admin/models/", 1)[1]
    full_key = full_key.split("?", 1)[0]  # 去掉 query

    ok, files = _delete_model_by_key(full_key)
    if not ok:
        return jsonify({"error": "模型不存在"}), 404

    return jsonify({"message": f"已删除 {full_key}", "deleted_files": files})


# ==================== API: 管理员 - 数据统计 ====================
@app.route("/api/admin/stats", methods=["GET"])
@admin_required
def admin_stats():
    """
    管理员专用：获取系统整体数据统计。
    """
    conn = get_db()
    user_count = conn.execute("SELECT COUNT(*) as cnt FROM users").fetchone()['cnt']
    admin_count = conn.execute("SELECT COUNT(*) as cnt FROM users WHERE is_admin=1").fetchone()['cnt']
    active_count = conn.execute("SELECT COUNT(*) as cnt FROM users WHERE status='active'").fetchone()['cnt']
    conn.close()
    
    total_files = 0
    user_upload_dirs = 0
    user_dirs = list(UPLOAD_DIR.iterdir()) if UPLOAD_DIR.exists() else []
    for d in user_dirs:
        if d.is_dir():
            user_upload_dirs += 1
            total_files += len(list(d.glob("*.csv")))
    
    model_files = list(MODEL_DIR.glob("*.pt")) if MODEL_DIR.exists() else []
    model_count = len(model_files)
    model_size_mb = sum(f.stat().st_size for f in model_files) / (1024 * 1024)
    
    upload_files = list(UPLOAD_DIR.rglob("*.csv")) if UPLOAD_DIR.exists() else []
    upload_size_mb = sum(f.stat().st_size for f in upload_files) / (1024 * 1024)
    
    return jsonify({
        "users": {
            "total": user_count,
            "active": active_count,
            "admins": admin_count,
        },
        "files": {
            "upload_dirs": user_upload_dirs,
            "total_csv": total_files,
            "upload_size_mb": round(upload_size_mb, 2),
        },
        "models": {
            "total": model_count,
            "size_mb": round(model_size_mb, 2),
        },
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"EMG 训练服务启动：http://0.0.0.0:{port}")
    print(f"解释器：{sys.executable}")
    print(f"上传目录：{UPLOAD_DIR}")
    print(f"模型目录：{MODEL_DIR}")
    print(f"数据库：{DB_FILE}")
    app.run(host="0.0.0.0", port=port, debug=False)