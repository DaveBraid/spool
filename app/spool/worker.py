"""算力节点协议，可由调度主机通过 SSH 直接交给 Python 运行。"""

import contextlib
import csv
from fnmatch import fnmatchcase
import hashlib
import json
import netrc
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit


RESULT_EXCLUDES = (".cache", ".netrc", ".env*", ".ssh", "*.pem", "*.key", ".swanlab", ".spool", ".rsync-partial")


class Fault(Exception):
    def __init__(self, stage, code, message, detail="", exit_code=None, transient=False):
        '''用阶段、稳定错误码和原始诊断构造可通过 JSON 返回的异常。
        transient 表示在确认操作未生效后，可以按退避策略重试。
        '''
        super().__init__(message)
        self.info = dict(stage=stage, code=code, message=message, detail=detail, exit_code=exit_code,
                         transient=transient)


def run(args, stage, code, *, timeout=15, input_text=None, check=True, transient=False):
    '''运行参数数组并保留输出和退出码。
    输入可经 stdin 传入凭据；诊断只收集进程输出。
    '''
    try:
        result = subprocess.run(args, input=input_text, capture_output=True, encoding="utf-8",
                                errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        partial = b"\n".join(value for value in (exc.stderr, exc.stdout) if value)
        raise Fault(stage, code + "_TIMEOUT", "操作超时", partial.decode(errors="replace")[-16000:],
                    transient=True) from None
    except OSError as exc:
        raise Fault(stage, code, "无法执行命令", str(exc)) from None
    if check and result.returncode:
        raise Fault(stage, code, "命令返回失败", (result.stderr + result.stdout)[-16000:], result.returncode,
                    transient)
    return result


def private_write(path, content):
    '''以当前账号独占权限原子写入凭据或运行标记。
    目标路径的父目录已由调用方限制在 Spool 目录内。
    '''
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def netrc_hosts(path):
    '''校验 SwanLab 登录文件权限，返回解析后的认证条目。'''
    if path.is_symlink():
        raise Fault("swanlab", "SWANLAB_CREDENTIAL_INVALID", "登录文件须为普通文件")
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise Fault("swanlab", "SWANLAB_CREDENTIAL_INVALID", "登录文件须属于服务账号，权限为 0600 或 0400")
        return netrc.netrc(str(path)).hosts
    except (OSError, netrc.NetrcParseError) as exc:
        raise Fault("swanlab", "SWANLAB_CREDENTIAL_UNAVAILABLE", "无法读取 SwanLab 登录文件", type(exc).__name__) from None


def secret_values(path):
    '''返回该文件的口令供脱敏使用，调用方不得记录其值。'''
    return [entry[2] for entry in netrc_hosts(path).values() if entry[2]]


def swanlab_file(path):
    '''只导出 SwanLab 云端认证条目，避免把其他服务的登录信息送到节点。'''
    entries = []
    for host, (login, account, password) in netrc_hosts(path).items():
        endpoint = urlsplit(host if "://" in host else "//" + host)
        name = endpoint.hostname or ""
        if endpoint.username or (name != "swanlab.cn" and not name.endswith(".swanlab.cn")):
            continue
        if password:
            entries.append(f"machine {host} login {json.dumps(login)} account {json.dumps(account)} password {json.dumps(password)}")
    if not entries:
        raise Fault("swanlab", "SWANLAB_CREDENTIAL_INVALID", "调度主机的登录文件未包含 SwanLab 云端凭据")
    return "\n".join(entries) + "\n"


def redact(value, secrets):
    '''返回可供 CLI 和 Agent 查看的一段诊断。
    优先遮蔽实际凭据，再遮蔽常见认证字段。
    '''
    for secret in sorted(set(secrets), key=len, reverse=True):
        value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._=/+-]+", r"\1 [REDACTED]", value)
    return re.sub(r"(?i)(\b(?:SWANLAB_API_KEY|API[_ -]?KEY|password|token)\s*[=:]\s*)[^\s,]+",
                  r"\1[REDACTED]", value)


def digest(path):
    '''读取普通文件的 SHA-256，按块读取以限制内存。
    文件应位于调用方确认的本机目录内。
    '''
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def tree(path, *, checksum=True, exclude=(), directories=False):
    '''生成目录内容清单与摘要，用于校验快照和缓存。
    输入目录只能有普通文件和子目录。
    '''
    if path.is_symlink() or not path.is_dir():
        raise Fault("transfer", "UNSUPPORTED_FILE", "需要可读的普通目录", str(path))
    files = {}
    for parent, dirs, names in os.walk(path, followlinks=False):
        dirs[:] = [name for name in dirs if not any(fnmatchcase(name, pattern) for pattern in exclude)]
        for name in dirs + [name for name in names if not any(fnmatchcase(name, pattern) for pattern in exclude)]:
            item = Path(parent) / name
            mode = item.lstat().st_mode
            if stat.S_ISREG(mode):
                if checksum:
                    files[item.relative_to(path).as_posix()] = [digest(item), mode & 0o111]
            elif stat.S_ISDIR(mode):
                if checksum and directories:
                    files[item.relative_to(path).as_posix() + "/"] = None
            else:
                raise Fault("transfer", "UNSUPPORTED_FILE", "快照只接受普通文件和目录", str(item))
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest() if checksum else None


def gpu_inventory():
    '''查询 GPU UUID、空闲显存、忙闲和驱动版本。
    计算进程查询失败或缺少必要指标时，将卡视为不可用。
    '''
    query = "index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version"
    args = ["nvidia-smi", "--format=csv,noheader,nounits"]
    devices = run([*args, "--query-gpu=" + query], "probe", "NVIDIA_SMI_FAILED", timeout=6)
    processes = run([*args, "--query-compute-apps=gpu_uuid,pid"], "probe", "GPU_PROCESS_QUERY_FAILED", timeout=6)
    active = {(row[0], row[1]) for row in csv.reader(processes.stdout.splitlines(), skipinitialspace=True)
              if len(row) >= 2 and row[1].isdigit()}
    rows = list(csv.reader(devices.stdout.splitlines(), skipinitialspace=True))
    max_memory = 1024  # 空闲显存占用允许至 1 GiB，包含桌面和后台图形任务。
    eligible = {row[1] for row in rows if row[4].isdigit() and int(row[4]) <= max_memory}
    mixed = set()
    if eligible & {uuid for uuid, _ in active}:
        try:
            monitor = run(["nvidia-smi", "pmon", "-c", "1", "-s", "u"], "probe", "GPU_TYPE_QUERY_FAILED", timeout=5)
            # 驱动也会将图形与计算混合进程列入计算查询，已确认的混合进程按占用阈值判定。
            mixed = {(row[0], row[1]) for line in monitor.stdout.splitlines() if len(row := line.split()) >= 3
                     and row[0].isdigit() and row[1].isdigit() and row[2] in {"G", "C+G"}}
        except Fault:
            pass  # 进程类型无法确认时保持忙卡判定。
    gpus = []
    for index, uuid, name, total, used, util, driver in rows:
        known = total.isdigit() and used.isdigit() and util.isdigit() and driver.split(".")[0].isdigit()
        computing = any(device == uuid and (index, pid) not in mixed for device, pid in active)
        gpus.append(dict(index=int(index), uuid=uuid, name=name, driver_major=int(driver.split(".")[0]) if known else 0,
                         total_mib=int(total) if known else 0, free_mib=max(0, int(total) - int(used)) if known else 0,
                         busy=not known or computing or int(used) > max_memory or int(util) > 10))
    return gpus


def boot_id():
    '''读取节点的启动标识，用于区分重启和普通网络故障。
    在没有 Linux procfs 的环境中返回空值。
    '''
    path = Path("/proc/sys/kernel/random/boot_id")
    return path.read_text().strip() if path.is_file() else None


def available_memory():
    '''读取宿主机当前可用内存，单位 MiB；缺少指标时返回 None。'''
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def probe():
    '''返回本节点 Docker、GPU、CPU 和内存的即时状态。
    Docker 权限失败时保留原始错误，同时仍展示 GPU 清单。
    '''
    result = dict(boot_id=boot_id(), cpus=os.cpu_count(), memory_mib=available_memory(), docker=False, gpus=[], errors=[])
    try:
        info = run(["docker", "info", "--format", "{{.ServerVersion}}"], "probe", "DOCKER_UNAVAILABLE", timeout=6)
        result.update(docker=True, docker_version=info.stdout.strip())
    except Fault as exc:
        result["errors"].append(exc.info)
    try:
        result["gpus"] = gpu_inventory()
    except Fault as exc:
        result["errors"].append(exc.info)
    return result


def inspect(attempt):
    '''按本次尝试的固定容器名读取状态，并核验所有权标签。
    仅在 Docker 明确报告容器不存在时返回 None。
    '''
    proc = run(["docker", "container", "inspect", "spool-" + attempt], "inspect", "INSPECT_FAILED", check=False)
    if proc.returncode:
        if "no such container" in proc.stderr.lower() or "no such object" in proc.stderr.lower():
            return None
        raise Fault("inspect", "INSPECT_FAILED", "容器状态不可确认", proc.stderr, proc.returncode, True)
    value = json.loads(proc.stdout)[0]
    labels = value["Config"].get("Labels") or {}
    if labels.get("io.spool.attempt") != attempt or labels.get("io.spool.uid") != str(os.getuid()):
        raise Fault("inspect", "CONTAINER_OWNERSHIP_MISMATCH", "容器标签与当前尝试不匹配")
    state = value["State"]
    return dict(container_id=value["Id"], status=state["Status"], exit_code=state["ExitCode"],
                oom=state.get("OOMKilled", False), detail=state.get("Error", ""), boot_id=boot_id(),
                started_at=state.get("StartedAt"), finished_at=state.get("FinishedAt"))


@contextlib.contextmanager
def lock(path, stage, code):
    '''获取节点本地文件锁，防止同一尝试或 GPU 被并行启动。
    锁已占用时立即返回可等待的状态。
    '''
    import fcntl

    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Fault(stage, code, "节点上已有相关操作正在执行", transient=True) from None
        yield


def context(payload):
    '''创建本次尝试的私有目录，返回执行根和尝试目录。
    路径由中心配置，尝试 ID 仅接受 UUID 十六进制表示。
    '''
    attempt = payload["attempt"]
    if not re.fullmatch(r"[0-9a-f]{32}", attempt):
        raise Fault("prepare", "INVALID_ATTEMPT", "无效的尝试 ID")
    root = Path(payload["root"]).expanduser().resolve()
    work = root / "runs" / attempt
    for path in (root, root / "runs", root / "assets", root / "locks", work, work / "outputs", work / "secrets"):
        if path.is_symlink():
            raise Fault("prepare", "UNSAFE_PATH", "运行目录不能是符号链接", str(path))
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        if path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077:
            raise Fault("prepare", "UNSAFE_PATH", "运行目录须属于当前账号且权限为 0700", str(path))
    return root, work


def fetch_image(payload, root):
    '''按 digest 解析本地镜像，缓存缺失时拉取同一版本。
    私有仓库凭据仅保存在 Spool 的 Docker 客户端目录内。
    '''
    image = payload["image"]
    env_config = root / "docker-client"
    docker = ["docker", "--config", str(env_config)] if env_config.is_dir() and payload.get("registry") else ["docker"]
    query = [*docker, "image", "inspect", image]
    result = run(query, "image", "IMAGE_INSPECT_FAILED", check=False)
    if result.returncode:
        try:
            run([*docker, "pull", image], "image", "IMAGE_PULL_FAILED", timeout=900, transient=True)
        except Fault as exc:
            detail = exc.info["detail"].lower()
            if any(word in detail for word in ("unauthorized", "denied", "manifest unknown", "not found", "no matching manifest")):
                exc.info["transient"] = False
            raise
        result = run(query, "image", "IMAGE_INSPECT_FAILED")
    value = json.loads(result.stdout)[0]
    labels = value["Config"].get("Labels") or {}
    minimum = labels.get("io.spool.min-driver") or labels.get("io.lab-platform.min-driver")
    if minimum is not None and not re.fullmatch(r"[0-9]{1,4}", minimum):
        raise Fault("image", "INVALID_DRIVER_LABEL", "镜像的最低驱动版本标签须使用主版本号")
    resolved = dict(image_id=value["Id"], min_driver=int(minimum) if minimum else 0)
    if payload.get("use_image_user"):
        user = value["Config"].get("User") or ""
        if not re.fullmatch(r"[1-9][0-9]*:[1-9][0-9]*", user):
            raise Fault("image", "NONROOT_IMAGE_USER_REQUIRED", "镜像需声明非 root 数字 UID:GID")
        resolved["container_user"] = user
    return resolved


def launch(payload, root, work):
    '''在节点最后查卡后按固定名称启动容器。
    写入启动标记后若响应丢失，中心必须核对状态再决定是否重投。
    '''
    attempt, spec = payload["attempt"], payload["spec"]
    gpus = payload["gpus"]
    force_start = payload.get("force_start", False)
    with contextlib.ExitStack() as stack:
        stack.enter_context(lock(work / "operation.lock", "launch", "ATTEMPT_BUSY"))
        if (work / "fenced").exists():
            raise Fault("launch", "ATTEMPT_FENCED", "尝试已停止，不能再次启动")
        existing = inspect(attempt)
        if existing:
            return existing
        if (work / "launch-requested").exists():
            raise Fault("launch", "LAUNCH_UNCERTAIN", "需要核对上次启动结果")
        memory = available_memory() if not force_start else None
        if (os.cpu_count() or 0) < spec["cpus"] or (not force_start and (memory is None or memory < spec["memory_mib"] + 1024)):
            raise Fault("launch", "RESOURCE_BUSY", "启动前节点已无足够 CPU 或内存", transient=True)
        for gpu in sorted(gpus):
            if not re.fullmatch(r"GPU-[a-zA-Z0-9-]+", gpu):
                raise Fault("launch", "INVALID_GPU", "GPU UUID 无效")
            stack.enter_context(lock(root / "locks" / gpu, "launch", "GPU_BUSY"))
        if gpus:
            cards = {gpu["uuid"]: gpu for gpu in gpu_inventory()}
            if any(gpu not in cards or (not force_start and (cards[gpu]["busy"] or cards[gpu]["free_mib"] < spec["min_free_mib"])) for gpu in gpus):
                raise Fault("launch", "GPU_BUSY", "启动前 GPU 已不可用", transient=True)
            minimum = max(spec["min_driver_major"], int(payload.get("min_driver") or 0))
            if any(cards[gpu]["driver_major"] < minimum for gpu in gpus):
                versions = ", ".join(f"{gpu}={cards[gpu]['driver_major']}" for gpu in gpus)
                raise Fault("launch", "DRIVER_INCOMPATIBLE", f"镜像或任务要求驱动主版本至少为 {minimum}", versions)
        user = payload["container_user"] if spec.get("use_image_user") else f"{os.getuid()}:{os.getgid()}"
        args = ["docker", "run", "-d", "--name", "spool-" + attempt, "--restart=no",
                "--label", "io.spool.attempt=" + attempt, "--label", "io.spool.uid=" + str(os.getuid()),
                "--user", user, "--cap-drop=ALL", "--security-opt=no-new-privileges",
                "--cpus", str(spec["cpus"]), "--memory", str(spec["memory_mib"]) + "m",
                "--shm-size", str(spec["shm_mib"]) + "m", "--log-opt", "max-size=20m", "--log-opt", "max-file=3",
                "--workdir", spec["workdir"], "--entrypoint", ""]
        if spec.get("use_image_user"):
            uid, gid = user.split(":")
            # 产物写在容器里，退出后以节点账号复制。临时缓存随容器退出释放。
            args += ["--tmpfs", f"/outputs/.cache:rw,nosuid,nodev,mode=0700,uid={uid},gid={gid}"]
        else:
            args += ["--mount", f"type=bind,src={work}/code,dst=/workspace,readonly",
                     "--mount", f"type=bind,src={work}/outputs,dst=/outputs"]
        env = {"HOME": "/outputs", "PYTHONUNBUFFERED": "1", "XDG_CACHE_HOME": "/outputs/.cache", **spec["env"]}
        if gpus:
            args += ["--gpus", '"device=' + ",".join(gpus) + '"']  # Docker CLI 以 CSV 解析多张 UUID
        else:
            env["NVIDIA_VISIBLE_DEVICES"] = "void"
        for name, content in payload["assets"].items():
            args += ["--mount", f"type=bind,src={root}/assets/{content},dst=/inputs/{name},readonly"]
        if payload.get("resume_path"):
            args += ["--mount", f"type=bind,src={work}/resume.pt,dst=/run/spool/checkpoint,readonly"]
            env["SPOOL_CHECKPOINT"] = "/run/spool/checkpoint"
        if spec["swanlab"]:
            credentials = work / "secrets/.netrc"
            if not secret_values(credentials):
                raise Fault("swanlab", "SWANLAB_CREDENTIAL_INVALID", "SwanLab 登录文件为空")
            args += ["--mount", f"type=bind,src={work}/secrets,dst=/run/secrets/swanlab,readonly"]
            env.update(SWANLAB_ROOT="/run/secrets/swanlab", SWANLAB_MODE="online", SWANLAB_ON_ONLINE_ERROR="raise",
                       SWANLAB_RUN_ID=payload["job"], SWANLAB_RESUME="allow")
            if not spec.get("result_root"):
                env["SWANLAB_LOGDIR"] = "/outputs/swanlog"  # 旧归档任务保持此前的记录路径。
        for key, value in env.items():
            args += ["--env", f"{key}={value}"]
        private_write(work / "launch-requested", boot_id() or "unknown")
        try:
            run([*args, payload["image_id"], *payload["command"]], "launch", "CONTAINER_START_FAILED", timeout=90)
        except Fault as exc:
            if exc.info["exit_code"] is not None:
                secret = work / "secrets/.netrc"
                exc.info["detail"] = redact(exc.info["detail"], secret_values(secret) if secret.exists() else [])
                private_write(work / "launch-error", json.dumps(exc.info))
            raise
        return inspect(attempt)


def status(payload, work, cancel=False):
    '''核对容器及启动标记，或给指定尝试加停止标记并取消。
    还可能迟到的启动请求必须先被隔离，才可释放 GPU。
    '''
    attempt = payload["attempt"]
    with lock(work / "operation.lock", "inspect", "ATTEMPT_BUSY"):
        stop_requested = work / "stop-requested"
        if cancel:
            private_write(work / "fenced", "cancel\n")
        value = inspect(attempt)
        marker = work / "launch-requested"
        error = work / "launch-error"
        if value is None:
            before_restart = marker.exists() and boot_id() and marker.read_text() not in {"unknown", boot_id()}
            if marker.exists() and not error.exists() and not before_restart:
                return {"status": "uncertain", "boot_id": boot_id()}
            private_write(work / "fenced", "absent\n")
            return {"status": "absent", "boot_id": boot_id(), "stop_requested": stop_requested.is_file(),
                    "error": json.loads(error.read_text()) if error.exists() else None}
        if cancel and value["status"] in {"running", "paused", "restarting", "created"}:
            if value["status"] == "created":
                run(["docker", "rm", "spool-" + attempt], "cancel", "CANCEL_FAILED", timeout=20)
                return {"status": "absent", "boot_id": boot_id()}
            # 先记录确实向尚在运行的容器发起了停止，SSH 回执丢失后仍可核对。
            private_write(stop_requested, "cancel\n")
            run(["docker", "stop", "--time=20", "spool-" + attempt], "cancel", "CANCEL_FAILED", timeout=30)
            value = inspect(attempt)
        if value and value["status"] == "created" and error.exists():
            value.update(status="start_failed", error=json.loads(error.read_text()))
        if value:
            value["stop_requested"] = stop_requested.is_file()
        return value or {"status": "uncertain", "boot_id": boot_id()}


def export_outputs(attempt, work):
    '''从已退出的容器复制产物，节点任务目录下的输出与重试保持一致。'''
    with lock(work / "operation.lock", "results", "ATTEMPT_BUSY"):
        output = work / "outputs"
        if any(output.iterdir()):
            return {}
        value = inspect(attempt)
        if not value or value["status"] != "exited":
            raise Fault("results", "CONTAINER_OUTPUTS_UNAVAILABLE", "容器未退出或已不可读取")
        with tempfile.TemporaryDirectory(dir=work) as temporary:
            staged = Path(temporary) / "outputs"
            run(["docker", "cp", "spool-" + attempt + ":/outputs", str(staged)],
                "results", "CONTAINER_COPY_FAILED", timeout=1200, transient=True)
            staged.chmod(0o700)
            staged.replace(output)  # 一次发布完整目录，收集失败可从同一份节点产物重试。
    return {}


def cleanup_attempt(attempt, work, archived_container):
    '''确认已结束并保全日志后删除该尝试的容器和工作副本，保留防重启标记。'''
    with lock(work / "operation.lock", "cleanup", "ATTEMPT_BUSY"):
        value = inspect(attempt)
        if value and value["status"] != "exited" and not (value["status"] == "created" and (work / "launch-error").is_file()):
            raise Fault("cleanup", "CONTAINER_NOT_STOPPED", "容器尚未确认退出", value["status"])
        if value and value["status"] == "exited" and value["container_id"] != archived_container:
            raise Fault("cleanup", "LOG_NOT_ARCHIVED", "请先回收此容器的终端日志")
        private_write(work / "fenced", "collected\n")
        if value:
            run(["docker", "rm", value["container_id"]], "cleanup", "CONTAINER_REMOVE_FAILED", timeout=30, transient=True)
        for path in work.iterdir():
            if path.name in {"fenced", "operation.lock"}:
                continue  # 上一次启动请求迟到时仍会被同一把锁和标记挡住。
            if not path.is_symlink() and path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    return {}


def image_cache(references, apply):
    '''只检查传入的私有仓库摘要。保留所有仍被本机容器引用的镜像。'''
    run(["docker", "info", "--format", "{{.ServerVersion}}"], "image", "DOCKER_UNAVAILABLE")
    entries = {}
    for reference in references:
        info = run(["docker", "image", "inspect", "--format", "{{.Id}} {{.Size}}", reference],
                   "image", "IMAGE_INSPECT_FAILED", check=False)
        if info.returncode:
            if "no such image" in info.stderr.lower() or "no such object" in info.stderr.lower():
                continue
            raise Fault("image", "IMAGE_INSPECT_FAILED", "无法核对节点镜像", info.stderr.strip())
        image_id, size = info.stdout.strip().split()
        containers = run(["docker", "ps", "-aq", "--filter", "ancestor=" + image_id], "image", "IMAGE_REFERENCES_FAILED")
        state = "IN_USE" if containers.stdout.strip() else "CANDIDATE"
        entries[reference] = {"state": state, "size": int(size)}
        if apply and state == "CANDIDATE":
            try:
                run(["docker", "image", "rm", "--no-prune", reference], "image", "IMAGE_CACHE_REMOVE_FAILED", timeout=180)
                entries[reference]["state"] = "REMOVED"
            except Fault as exc:
                entries[reference].update(state="FAILED", error=exc.info)
    return entries


def dispatch(payload):
    '''处理来自调度主机的单次请求，返回结构化状态或错误。'''
    action = payload["action"]
    if action == "probe":
        return probe()
    if action == "image_cache":
        return image_cache(payload["references"], payload["apply"])
    root, work = context(payload)
    if action == "init":
        return {"root": str(root), "work": str(work)}
    if action == "verify":
        if payload["kind"] == "checkpoint":
            path = work / "resume.pt"
            if path.is_symlink() or not path.is_file() or digest(path) != payload["digest"]:
                raise Fault("checkpoint", "CHECKPOINT_MISMATCH", "节点上的断点文件未通过校验")
            return {}
        path = work / "code" if payload["kind"] == "code" else root / "assets" / payload["digest"]
        if not path.is_dir() or path.is_symlink() or tree(path) != payload["digest"]:
            raise Fault("transfer", "CHECKSUM_MISMATCH", "节点文件与中心清单不一致", str(path))
        if payload["kind"] == "asset":
            private_write(root / "assets" / (payload["digest"] + ".ready"), "ok\n")
        return {}
    if action == "asset_ready":
        base = root / "assets"
        data = base / payload["digest"]
        return {"ready": data.is_dir() and not data.is_symlink() and (base / (payload["digest"] + ".ready")).is_file()}
    if action == "image":
        return fetch_image(payload, root)
    if action == "configure_registry":
        path = root / "docker-client"
        path.mkdir(mode=0o700, exist_ok=True)
        private_write(path / "config.json", json.dumps(payload["docker_config"]))
        return {}
    if action == "secret":
        path = work / "secrets/.netrc"
        private_write(path, payload["netrc"])
        secret_values(path)
        return {}
    if action == "start":
        return launch(payload, root, work)
    if action in {"status", "cancel"}:
        return status(payload, work, action == "cancel")
    if action == "export_outputs":
        return export_outputs(payload["attempt"], work)
    if action == "result_digest":
        return {"digest": tree(work / "outputs", exclude=RESULT_EXCLUDES, directories=True)}
    if action == "cleanup":
        return cleanup_attempt(payload["attempt"], work, payload.get("archived_container"))
    if action == "logs":
        if not inspect(payload["attempt"]):
            raise Fault("logs", "CONTAINER_NOT_FOUND", "还没有可读取的容器日志")
        result = run(["docker", "logs", "--tail", str(payload["tail"]), "spool-" + payload["attempt"]],
                     "logs", "LOG_READ_FAILED", check=False)
        # docker logs 把容器 stderr 保持在 stderr 通道。
        path = work / "secrets/.netrc"
        secrets = secret_values(path) if path.exists() else []
        if result.returncode:
            raise Fault("logs", "LOG_READ_FAILED", "无法读取容器日志", redact(result.stderr, secrets), result.returncode)
        return {"stdout": redact(result.stdout[-160000:], secrets), "stderr": redact(result.stderr[-160000:], secrets)}
    if action == "checkpoint":
        output = work / "outputs"
        path = output / payload["path"]
        if path.resolve() != path or not path.is_file():
            return None
        before = path.stat()
        checksum = digest(path)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            return None
        return {"path": str(path), "digest": checksum, "bytes": after.st_size}
    if action == "forget_secret":
        (work / "secrets/.netrc").unlink(missing_ok=True)
        return {}
    raise Fault("worker", "UNKNOWN_ACTION", "未知的节点操作")


def main():
    '''从 stdin 读取受控请求并向 stdout 返回单个 JSON 对象。
    stderr 和实际退出码保留在错误对象中。
    '''
    os.umask(0o077)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)  # 客户端断线后允许写完当前启动结果
    payload = json.load(sys.stdin)
    try:
        reply = {"ok": True, "data": dispatch(payload)}
    except Fault as exc:
        reply = {"ok": False, "error": exc.info}
    except (ValueError, OSError, KeyError) as exc:
        reply = {"ok": False, "error": Fault("worker", "WORKER_ERROR", type(exc).__name__, str(exc)).info}
    print(json.dumps(reply, ensure_ascii=False))


if __name__ == "__main__":
    main()
