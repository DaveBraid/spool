"""本机或 SSH 执行、文件传输、私有镜像仓库。"""

import base64
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import subprocess
import sys

from .worker import RESULT_EXCLUDES, Fault, private_write, redact, run


SSH = ["ssh", "-T", "-oBatchMode=yes", "-oStrictHostKeyChecking=yes", "-oUpdateHostKeys=no", "-oConnectTimeout=5",
       "-oServerAliveInterval=10", "-oServerAliveCountMax=2", "-oForwardAgent=no", "-oForwardX11=no"]


class Transport:
    def __init__(self, cfg):
        '''预载相同的节点执行程序，保证本机与远端的接口一致。'''
        self.cfg = cfg
        encoded = base64.b64encode(files("spool").joinpath("worker.py").read_bytes()).decode()
        script = "import base64;exec(compile(base64.b64decode(" + repr(encoded) + "), 'spool-worker', 'exec'))"
        self.command = ["python3", "-I", "-B", "-c", script]
        self.tunnels = {}

    def call(self, host, action, attempt=None, *, timeout=24, **data):
        '''在指定节点执行一次 JSON 请求，保留远端错误阶段和进程输出。'''
        node = self.cfg["hosts"][host]
        payload = dict(action=action, root=node.get("root", self.cfg["worker_root"]), attempt=attempt, **data)
        args = [sys.executable, *self.command[1:]]
        if not node.get("local"):
            args = [*SSH, "-oClearAllForwardings=yes", node["ssh"], shlex.join(self.command)]
        result = run(args, action, "NODE_TRANSPORT_FAILED", timeout=timeout, input_text=json.dumps(payload), transient=True)
        try:
            body = json.loads(result.stdout)
        except (ValueError, TypeError):
            raise Fault(action, "NODE_PROTOCOL_ERROR", "节点没有返回有效 JSON", redact((result.stderr + result.stdout)[-16000:], [])) from None
        if not body["ok"]:
            raise Fault(**body["error"])
        return body["data"]

    def copy(self, host, local, remote, *, pull=False, directory=True, result=False):
        '''在中心和节点之间传送普通文件，返回时保证复制命令成功。'''
        node = self.cfg["hosts"][host]
        stage = "results" if result else "checkpoint" if pull else "transfer"
        if node.get("local"):
            source, destination = (Path(remote), Path(local)) if pull else (Path(local), Path(remote))
            try:
                if directory:
                    shutil.copytree(source, destination, dirs_exist_ok=True, symlinks=True,
                                    ignore=shutil.ignore_patterns(*RESULT_EXCLUDES) if result else None)
                else:
                    shutil.copy2(source, destination, follow_symlinks=False)
            except OSError as exc:
                raise Fault(stage, "COPY_FAILED", "本机复制失败", str(exc)) from None
            return
        shell = shlex.join([*SSH, "-oClearAllForwardings=yes"])
        args = ["rsync", "-rltp", "--protect-args", "--partial-dir=.rsync-partial", "-e", shell]
        if result:
            args.extend("--exclude=" + pattern for pattern in RESULT_EXCLUDES)
        suffix = "/" if directory else ""
        pair = [str(local) + suffix, node["ssh"] + ":" + remote + suffix]
        try:
            run([*args, *(reversed(pair) if pull else pair)], stage, "RSYNC_FAILED", timeout=1800, transient=True)
        except Fault as exc:
            if exc.info["exit_code"] not in {10, 12, 24, 30, 35, 255, None}:
                exc.info["transient"] = False
            raise

    def inventory(self):
        '''并行采样已配置节点，失败的节点仍返回可定位的诊断。'''
        def observe(host):
            '''采样单节点，将连接失败记录为不可用。'''
            try:
                return self.call(host, "probe", timeout=22)
            except Fault as exc:
                return dict(docker=False, gpus=[], errors=[exc.info])

        hosts = list(self.cfg["hosts"])
        with ThreadPoolExecutor(max_workers=min(8, len(hosts))) as pool:
            return dict(zip(hosts, pool.map(observe, hosts)))

    def keep_tunnels(self):
        '''为配置的私有仓库维护节点回环转发，断开后由下一轮重连。'''
        port = self.cfg.get("registry_port")
        if not port or not registry_password(self.cfg, create=False):
            return
        for name, node in self.cfg["hosts"].items():
            if node.get("local") or (name in self.tunnels and self.tunnels[name].poll() is None):
                continue
            log = self.cfg["state"] / "logs" / f"registry-tunnel-{name}.log"
            with log.open("w") as stderr:
                self.tunnels[name] = subprocess.Popen([
                    *SSH, "-oExitOnForwardFailure=yes", "-N", "-R", f"127.0.0.1:{port}:127.0.0.1:{port}", node["ssh"],
                ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=stderr)

    def close(self):
        '''退出服务时关闭本进程发起的镜像转发连接。'''
        for proc in self.tunnels.values():
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def registry_password(cfg, *, create):
    '''读取 Spool 专用仓库口令，首次建仓时生成并私有保存。'''
    path = cfg["state"] / "registry-password"
    if path.exists():
        if path.is_symlink() or path.stat().st_mode & 0o077 or path.stat().st_uid != os.getuid():
            raise Fault("registry", "UNSAFE_REGISTRY_PASSWORD", "仓库凭据必须由 Spool 账号私有保存")
        return path.read_text().strip()
    if create:
        value = secrets.token_urlsafe(32)
        private_write(path, value + "\n")
        return value
    return None


def registry_config(cfg):
    '''构造隔离的 Docker 客户端认证，仅经本机或加密 SSH 使用。'''
    password = registry_password(cfg, create=False)
    if not password:
        raise Fault("registry", "REGISTRY_NOT_INITIALIZED", "请先在调度主机初始化 Spool 镜像仓库")
    auth = base64.b64encode(("spool:" + password).encode()).decode()
    return {"auths": {f"127.0.0.1:{cfg['registry_port']}": {"auth": auth}}}


def registry_init(cfg, image):
    '''在调度主机的回环地址启动带 bcrypt 认证的 Distribution 仓库。'''
    port = cfg.get("registry_port")
    if not port:
        raise Fault("registry", "REGISTRY_DISABLED", "配置缺少 registry_port")
    if not re.fullmatch(r"(?:docker\.io/library/)?registry(?::[a-zA-Z0-9_.-]+|@sha256:[a-f0-9]{64})", image):
        raise Fault("registry", "INVALID_REGISTRY_IMAGE", "仅接受 Docker 官方 registry 镜像引用")
    current = run(["docker", "container", "inspect", "spool-registry"], "registry", "REGISTRY_INSPECT_FAILED", check=False)
    if current.returncode == 0:
        value = json.loads(current.stdout)[0]
        if (value["Config"].get("Labels") or {}).get("io.spool.registry") != str(os.getuid()):
            raise Fault("registry", "REGISTRY_NAME_CONFLICT", "同名容器不属于 Spool")
        bindings = value["HostConfig"]["PortBindings"].get("5000/tcp", [])
        if not all(x["HostIp"] == "127.0.0.1" and x["HostPort"] == str(port) for x in bindings) or not bindings:
            raise Fault("registry", "REGISTRY_PORT_MISMATCH", "现有 Spool 仓库端口与配置不一致")
        registry_config(cfg)  # 已有仓库必须沿用现有认证，不自动重置
        if not value["State"]["Running"]:
            run(["docker", "start", "spool-registry"], "registry", "REGISTRY_START_FAILED")
        return {"registry": f"127.0.0.1:{port}"}
    password = registry_password(cfg, create=True)
    auth = cfg["state"] / "registry-auth"
    data = cfg["state"] / "registry-data"
    for path in (auth, data):
        path.mkdir(exist_ok=True, mode=0o700)
    if shutil.which("htpasswd"):
        hashed = run(["htpasswd", "-niB", "spool"], "registry", "HTPASSWD_FAILED", input_text=password + "\n").stdout.strip()
    else:
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                import crypt
            hashed = "spool:" + crypt.crypt(password, crypt.mksalt(crypt.METHOD_BLOWFISH))
        except (ImportError, ValueError, TypeError):
            raise Fault("registry", "BCRYPT_UNAVAILABLE", "需要系统 htpasswd 或支持 bcrypt 的 Python crypt") from None
    if not re.match(r"spool:\$2[aby]\$", hashed):
        raise Fault("registry", "BCRYPT_UNAVAILABLE", "无法生成 Distribution 支持的 bcrypt 哈希")
    private_write(auth / "htpasswd", hashed + "\n")
    run(["docker", "run", "-d", "--name", "spool-registry", "--restart=unless-stopped", "--cap-drop=ALL",
         "--security-opt=no-new-privileges", "--user", f"{os.getuid()}:{os.getgid()}",
         "--label", "io.spool.registry=" + str(os.getuid()), "-p", f"127.0.0.1:{port}:5000",
         "--mount", f"type=bind,src={auth},dst=/auth,readonly", "--mount", f"type=bind,src={data},dst=/var/lib/registry",
         "-e", "REGISTRY_AUTH=htpasswd", "-e", "REGISTRY_AUTH_HTPASSWD_REALM=Spool",
         "-e", "REGISTRY_AUTH_HTPASSWD_PATH=/auth/htpasswd",
         "-e", "REGISTRY_STORAGE_DELETE_ENABLED=true", image], "registry", "REGISTRY_START_FAILED", timeout=180)
    return {"registry": f"127.0.0.1:{port}"}


def publish(cfg, image, tag):
    '''推送本机已构建的 Docker 镜像，返回可直接入队的 digest 引用。'''
    if not re.fullmatch(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag):
        raise Fault("image", "INVALID_TAG", "目标须使用仓库名:标签，例如 project/train:v1")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.:/@-]*", image):
        raise Fault("image", "INVALID_IMAGE", "本地镜像引用无效")
    folder = cfg["state"] / "docker-client"
    folder.mkdir(exist_ok=True, mode=0o700)
    private_write(folder / "config.json", json.dumps(registry_config(cfg)))
    reference = f"127.0.0.1:{cfg['registry_port']}/{tag}"
    docker = ["docker", "--config", str(folder)]
    run([*docker, "tag", image, reference], "image", "IMAGE_TAG_FAILED")
    result = run([*docker, "push", reference], "image", "IMAGE_PUSH_FAILED", timeout=1800, transient=True)
    found = re.search(r"\bdigest: (sha256:[a-f0-9]{64})\b", result.stdout + result.stderr)
    if not found:
        raise Fault("image", "IMAGE_DIGEST_MISSING", "推送完成，但 Docker 未返回镜像 digest", result.stdout[-4000:])
    return {"image": reference.rsplit(":", 1)[0] + "@" + found[1]}
