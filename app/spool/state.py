"""任务输入、不可变快照和 SQLite 状态。"""

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import tempfile
import time

from .worker import Fault, tree


ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}\Z")
IMAGE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.:/-]*@sha256:[a-f0-9]{64}\Z")
GPU = re.compile(r"GPU-[a-zA-Z0-9-]+\Z")
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELED"}
HELD = {"PREPARING", "STARTING", "RUNNING", "UNKNOWN"}
EXCLUDED = {".git", ".venv", "__pycache__", ".ssh", ".docker", ".swanlab", ".netrc", ".aws", ".spool", ".cache"}


def json_read(path):
    '''读取配置或任务 JSON，错误保留输入路径和解析诊断。'''
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise Fault("input", "INVALID_JSON", "无法读取 JSON", f"{path}: {exc}") from None


def identify(value, field):
    '''校验任务、资产或节点标识，返回可用于目录名的字符串。'''
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise Fault("input", "INVALID_ID", f"{field} 只能使用字母、数字、点、横线和下划线，且长度为 1–64")
    return value


def config(path):
    '''读取调度主机的配置及节点、仓库、运行目录设置。'''
    raw = json_read(path)
    if not isinstance(raw, dict) or not isinstance(raw.get("hosts"), dict) or not raw["hosts"]:
        raise Fault("config", "INVALID_CONFIG", "配置须包含至少一个节点")
    raw.setdefault("state", "~/.local/state/spool")
    raw.setdefault("source_roots", ["~/projects"])
    raw.setdefault("swanlab_netrc", "~/.swanlab/.netrc")
    raw.setdefault("poll_seconds", 15)  # 队列及容器状态检查间隔
    raw.setdefault("large_job_wait_seconds", 1800)  # 达到此等待后，为最老的多卡任务留出同机卡数
    raw.setdefault("worker_root", "~/.local/state/spool-worker")
    raw["state"] = Path(raw["state"]).expanduser().resolve()
    if not isinstance(raw["source_roots"], list) or not raw["source_roots"]:
        raise Fault("config", "INVALID_CONFIG", "source_roots 须列出允许提交的目录")
    raw["source_roots"] = [Path(root).expanduser().resolve() for root in raw["source_roots"]]
    raw["swanlab_netrc"] = Path(raw["swanlab_netrc"]).expanduser()
    if sum(isinstance(node, dict) and node.get("local") is True for node in raw["hosts"].values()) != 1:
        raise Fault("config", "INVALID_CONFIG", "须指定唯一的本机节点")
    for name, node in raw["hosts"].items():
        identify(name, "host")
        if (not isinstance(node, dict) or type(node.get("local", False)) is not bool
                or (not node.get("local") and not re.fullmatch(r"[\w@.-]+", node.get("ssh", "")))):
            raise Fault("config", "INVALID_CONFIG", f"节点 {name} 缺少合法 SSH 别名")
        if not node.get("local") and node["ssh"].startswith("-"):
            raise Fault("config", "INVALID_CONFIG", f"节点 {name} 的 SSH 别名无效")
        placement = node.get("placement", {})
        if (not isinstance(placement, dict) or set(placement) - {"priority", "max_spool_gpus", "gpu_order"}
                or ("priority" in placement and type(placement["priority"]) is not int)
                or ("max_spool_gpus" in placement and
                    (type(placement["max_spool_gpus"]) is not int or placement["max_spool_gpus"] < 1))
                or placement.get("gpu_order", "asc") not in ("asc", "desc")):
            raise Fault("config", "INVALID_PLACEMENT", f"节点 {name} 的选卡偏好无效")
    port = raw.get("registry_port")
    if port is not None and (type(port) is not int or not 1024 <= port <= 65535):
        raise Fault("config", "INVALID_CONFIG", "registry_port 须是 1024–65535 的整数")
    for key in ("poll_seconds", "large_job_wait_seconds"):
        if type(raw[key]) is not int or raw[key] < 1:
            raise Fault("config", "INVALID_CONFIG", f"{key} 须为正整数")
    return raw


def source_path(path, roots):
    '''将中心的源码或数据目录限制在操作者配置的根目录中。'''
    if not isinstance(path, str) or not path:
        raise Fault("input", "INVALID_SOURCE", "请提供调度主机上的目录")
    source = Path(path).expanduser().resolve()
    if not source.is_dir() or not any(source.is_relative_to(root) for root in roots):
        raise Fault("input", "INVALID_SOURCE", "源目录须位于调度主机的 source_roots 之内", str(source))
    return source


def result_root(path, cfg):
    '''解析原生结果根目录，拒绝写入 Spool 自己的状态目录。'''
    root = source_path(path, cfg["source_roots"])
    if root.is_relative_to(cfg["state"]) or cfg["state"].is_relative_to(root):
        raise Fault("input", "INVALID_RESULT_ROOT", "产物目录须与 Spool 状态目录分离", str(root))
    return root


def relative_path(value, field):
    '''校验容器输出中的相对路径，避免读取输出目录之外的文件。'''
    path = PurePosixPath(value) if isinstance(value, str) and value else None
    if not path or path.is_absolute() or not path.parts or ".." in path.parts or "," in value:
        raise Fault("input", "INVALID_PATH", f"{field} 须为 /outputs 下的相对路径")
    return path.as_posix()


def argv(value, field):
    '''校验训练入口的 argv。字符串中的 shell 语法由用户显式启动 shell 时解析。'''
    if not isinstance(value, list) or not value or any(not isinstance(s, str) or "\0" in s for s in value) or not value[0]:
        raise Fault("input", "INVALID_COMMAND", f"{field} 须为非空字符串参数数组")
    return value


def normalize(raw, cfg):
    '''将用户提交的 JSON 固化为调度规格，检查资源、数据和续训入口。'''
    fields = {"name", "image", "code", "command", "host", "allowed_hosts", "gpus", "diagnostic", "min_free_mib",
              "min_driver_major", "cpus", "memory_gib", "shm_gib", "assets", "swanlab", "env", "workdir", "checkpoint",
              "use_image_user", "swanlab_project", "gpu_uuids", "result_root"}
    if not isinstance(raw, dict) or set(raw) - fields:
        raise Fault("input", "INVALID_SPEC", "任务包含未知字段", str(sorted(set(raw) - fields)) if isinstance(raw, dict) else "")
    image = raw.get("image", "")
    if not isinstance(image, str) or not IMAGE.fullmatch(image):
        raise Fault("input", "IMAGE_NOT_PINNED", "任务镜像须用仓库名@sha256:digest")
    name = raw.get("name", "training")
    if not isinstance(name, str) or not name.strip() or len(name) > 128:
        raise Fault("input", "INVALID_SPEC", "name 须为 1–128 字符")
    spec = dict(name=name, image=image, command=argv(raw.get("command"), "command"))
    spec["code"] = str(source_path(raw["code"], cfg["source_roots"])) if raw.get("code") is not None else None
    spec["host"] = raw.get("host")
    if spec["host"] is not None:
        identify(spec["host"], "host")
    hosts = raw.get("allowed_hosts", list(cfg["hosts"]))
    if not isinstance(hosts, list) or not hosts or not all(isinstance(host, str) and host in cfg["hosts"] for host in hosts):
        raise Fault("input", "INVALID_HOST", "allowed_hosts 包含未知节点")
    if spec["host"] is not None and spec["host"] not in hosts:
        raise Fault("input", "INVALID_HOST", "指定的 host 须在 allowed_hosts 中")
    spec["allowed_hosts"] = list(dict.fromkeys(hosts))
    for key, default in (("diagnostic", False), ("swanlab", False)):
        spec[key] = raw.get(key, default)
        if type(spec[key]) is not bool:
            raise Fault("input", "INVALID_SPEC", f"{key} 须为 true 或 false")
    if "swanlab_project" in raw:
        project = raw["swanlab_project"]
        if (not spec["swanlab"] or not isinstance(project, str) or not re.fullmatch(r"[^/\x00-\x1f]+/[^/\x00-\x1f]+", project)
                or any(p in {"", ".", ".."} or p != p.strip() for p in project.split("/"))):
            raise Fault("input", "INVALID_SWANLAB_PROJECT", "swanlab_project 须随在线记录使用，格式为 workspace/project")
        spec["swanlab_project"] = project
    limits = {"gpus": (1, 0, 16), "min_free_mib": (0, 0, 2000000), "min_driver_major": (0, 0, 2000),
              "cpus": (2, 1, 1024), "memory_gib": (8, 0.125, 8192), "shm_gib": (2, 0.0625, 8192)}
    for key, (default, low, high) in limits.items():
        value = raw.get(key, default)
        expected = (int, float) if key in {"memory_gib", "shm_gib"} else (int,)
        if type(value) not in expected or not low <= value <= high:
            raise Fault("input", "INVALID_RESOURCE", f"{key} 须在 {low}–{high} 之间")
        spec[key] = value
    spec["memory_mib"] = math.ceil(spec.pop("memory_gib") * 1024)
    spec["shm_mib"] = math.ceil(spec.pop("shm_gib") * 1024)
    if (spec["gpus"] == 0) != spec["diagnostic"]:
        raise Fault("input", "INVALID_RESOURCE", "CPU 诊断须同时指定 gpus=0 和 diagnostic=true")
    if "gpu_uuids" in raw:
        uuids = raw["gpu_uuids"]
        if (not spec["host"] or not isinstance(uuids, list) or not uuids
                or not all(isinstance(gpu, str) and GPU.fullmatch(gpu) for gpu in uuids)
                or len(set(uuids)) != spec["gpus"] or len(uuids) != spec["gpus"]):
            raise Fault("input", "INVALID_GPU_ASSIGNMENT", "指定卡须提供 host 和与 gpus 等量的不同 GPU UUID")
        spec["gpu_uuids"] = uuids
    if "result_root" in raw:
        spec["result_root"] = str(result_root(raw["result_root"], cfg))
    raw_assets = raw.get("assets", [])
    if not isinstance(raw_assets, list):
        raise Fault("input", "INVALID_ASSETS", "assets 须列出已经登记的资产 ID")
    spec["assets"] = list(dict.fromkeys(identify(item, "asset") for item in raw_assets))
    spec["workdir"] = raw.get("workdir", "/outputs")
    if spec["workdir"] not in {"/workspace", "/outputs"}:
        raise Fault("input", "INVALID_WORKDIR", "workdir 只能是 /workspace 或 /outputs")
    spec["env"] = raw.get("env", {})
    if not isinstance(spec["env"], dict) or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
            or key.startswith(("SWANLAB_", "SPOOL_", "NVIDIA_", "CUDA_VISIBLE_")) or key in {"HOME", "XDG_CACHE_HOME"}
            or re.search(r"PASSWORD|TOKEN|SECRET|API_KEY", key, re.I) or not isinstance(value, str) or "\0" in value
            for key, value in spec["env"].items()):
        raise Fault("input", "INVALID_ENV", "env 仅接受普通环境变量，设备和凭据由 Spool 管理")
    checkpoint = raw.get("checkpoint")
    spec["checkpoint"] = None
    if checkpoint is not None:
        if not isinstance(checkpoint, dict) or set(checkpoint) - {"path", "resume_command", "every_seconds"}:
            raise Fault("input", "INVALID_CHECKPOINT", "checkpoint 须指定 path 和 resume_command")
        interval = checkpoint.get("every_seconds", 120)  # 断点文件回收周期，避免高频复制大模型
        if type(interval) is not int or not 15 <= interval <= 86400:
            raise Fault("input", "INVALID_CHECKPOINT", "every_seconds 须在 15–86400 之间")
        spec["checkpoint"] = dict(path=relative_path(checkpoint.get("path"), "checkpoint.path"),
                                  command=argv(checkpoint.get("resume_command"), "checkpoint.resume_command"),
                                  every_seconds=interval)
    use_image_user = raw.get("use_image_user", False)
    if type(use_image_user) is not bool:
        raise Fault("input", "INVALID_SPEC", "use_image_user 须为 true 或 false")
    if use_image_user:
        if any((spec["code"], spec["assets"], spec["swanlab"], spec["checkpoint"], spec["workdir"] != "/outputs")):
            raise Fault("input", "IMAGE_USER_REQUIRES_SELF_CONTAINED_JOB",
                        "镜像账号模式要求纯镜像作业，使用 /outputs 并关闭 SwanLab")
        spec["use_image_user"] = True  # 省略默认值，保留已有任务指纹。
    return spec


def snapshot(source, base, assets=False):
    '''复制并散列一次输入，原子发布到内容寻址目录。
    训练代码排除环境和凭据目录，资产遇到此类成员时返回错误。
    '''
    def ignored(parent, names):
        '''为 copytree 检查每个成员，返回应跳过的文件名。'''
        exclusions = {name for name in names if name in EXCLUDED or name.startswith((".env", ".spool-return-"))
                      or name.endswith((".pem", ".key", ".pyc"))}
        if assets and exclusions:
            raise Fault("snapshot", "ASSET_PRIVATE_FILES", "资产目录包含环境或私有文件", str(Path(parent) / sorted(exclusions)[0]))
        for name in set(names) - exclusions:
            path = Path(parent) / name
            mode = path.lstat().st_mode
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise Fault("snapshot", "UNSUPPORTED_FILE", "源快照只接受普通文件和目录", str(path))
        return exclusions

    with tempfile.TemporaryDirectory(dir=base, prefix=".staging-") as directory:
        stage = Path(directory) / "snapshot"
        if source is not None:
            if base.is_relative_to(source):
                raise Fault("snapshot", "RECURSIVE_SOURCE", "请提交一个不包含 Spool 状态目录的源目录")
            shutil.copytree(source, stage, ignore=ignored, symlinks=True)
        else:
            stage.mkdir()
        content = tree(stage)
        target = base / content
        try:
            stage.rename(target)
        except OSError:
            if not target.is_dir() or target.is_symlink() or tree(target) != content:
                raise
    return content


class Store:
    def __init__(self, state):
        '''打开用户私有的 SQLite 状态目录并初始化表。'''
        self.state = state
        state.mkdir(mode=0o700, parents=True, exist_ok=True)
        if state.stat().st_mode & 0o077 or state.stat().st_uid != os.getuid():
            raise Fault("config", "UNSAFE_STATE", "状态目录须属于当前账号且权限为 0700", str(state))
        for child in ("code", "assets", "results", "logs", "checkpoints", "result-locks"):
            (state / child).mkdir(mode=0o700, exist_ok=True)
        self.path = state / "spool.db"
        with self.db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS assets (id TEXT PRIMARY KEY, digest TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS accepted_images (reference TEXT PRIMARY KEY, accepted_at REAL NOT NULL)")

    @contextmanager
    def db(self, write=False):
        '''逐次建立 SQLite 连接，写入时用事务锁保护状态转换。'''
        db = sqlite3.connect(self.path, timeout=15)
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            with db:
                yield db
        finally:
            db.close()

    def get(self, job, db=None):
        '''按任务 ID 获取一个状态快照，可复用当前写事务。'''
        if db is None:
            with self.db() as connection:
                return self.get(job, connection)
        row = db.execute("SELECT body FROM jobs WHERE id=?", (job,)).fetchone()
        if row is None:
            raise Fault("queue", "JOB_NOT_FOUND", f"未找到任务 {job}")
        return json.loads(row[0])

    def jobs(self):
        '''按任务写入次序读取队列和历史记录。'''
        with self.db() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT body FROM jobs ORDER BY rowid")]

    def change(self, job, mutate):
        '''在事务中读取、修改并写回一个任务，返回更新后的记录。'''
        with self.db(write=True) as db:
            record = self.get(job, db)
            mutate(record)
            db.execute("UPDATE jobs SET body=? WHERE id=?", (json.dumps(record), job))
            return record

    def enqueue(self, job, spec, *, force_start=False):
        '''按指定 ID 幂等入队，预先冻结代码与资产版本。'''
        fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        with self.db() as db:
            row = db.execute("SELECT body FROM jobs WHERE id=?", (job,)).fetchone()
        if row:
            previous = json.loads(row[0])
            if previous["fingerprint"] != fingerprint:
                raise Fault("queue", "JOB_ID_CONFLICT", "同一任务 ID 对应了不同提交参数")
            return previous
        digests = self.assets()
        missing = set(spec["assets"]) - digests.keys()
        if missing:
            raise Fault("queue", "ASSET_NOT_FOUND", "未登记的数据资产", ", ".join(sorted(missing)))
        code = snapshot(Path(spec["code"]) if spec["code"] else None, self.state / "code")
        now = time.time()
        entry = dict(id=job, state="QUEUED", submitted_at=now, due=now, spec=spec, code=code,
                     assets={name: digests[name] for name in spec["assets"]}, fingerprint=fingerprint, attempts=[],
                     resume_retries=0, prepare_retries=0, cancel_requested=False, result_state="NONE",
                     cleanup_state="PENDING")
        if force_start:
            entry["force_start"] = True  # 调度选项独立于冻结的训练规格
        with self.db(write=True) as db:
            db.execute("INSERT OR IGNORE INTO jobs(id, body) VALUES (?, ?)", (job, json.dumps(entry)))
            previous = self.get(job, db)
            if previous["fingerprint"] != fingerprint:
                raise Fault("queue", "JOB_ID_CONFLICT", "同一任务 ID 对应了不同提交参数")
            return previous

    def assets(self):
        '''返回已登记的资产名及不可变内容摘要。'''
        with self.db() as db:
            return dict(db.execute("SELECT id, digest FROM assets ORDER BY id"))

    def accept_image(self, reference):
        '''登记已验收的镜像摘要，重复登记沿用原验收顺序。'''
        with self.db(write=True) as db:
            db.execute("INSERT OR IGNORE INTO accepted_images VALUES (?, ?)", (reference, time.time()))

    def accepted_images(self):
        '''按验收时间返回镜像摘要，供按仓库保留当前与上一份版本。'''
        with self.db() as db:
            return [row[0] for row in db.execute("SELECT reference FROM accepted_images ORDER BY accepted_at DESC")]

    def add_asset(self, name, source):
        '''登记一份只读挂载的数据版本，拒绝改写同名版本。'''
        content = snapshot(source, self.state / "assets", assets=True)
        with self.db(write=True) as db:
            db.execute("INSERT OR IGNORE INTO assets VALUES (?, ?)", (name, content))
            existing = db.execute("SELECT digest FROM assets WHERE id=?", (name,)).fetchone()[0]
            if existing != content:
                raise Fault("snapshot", "ASSET_VERSION_CONFLICT", "数据版本已有不同内容，请使用新版本名")
        return {"id": name, "digest": content}
