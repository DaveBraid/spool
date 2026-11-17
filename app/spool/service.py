"""调度主机上的队列、准备、状态核对与结果回收。"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import threading
import time
import uuid

from . import results
from .state import HELD, TERMINAL, result_root
from .swanlab import sync_run, termination_state
from .transport import Transport, registry_config
from .worker import Fault, digest, lock, private_write, redact, secret_values, swanlab_file, tree


BACKOFF = (30, 120, 300)  # 提交前传输故障的三次重试间隔，单位秒


def select(job, reports, active, cfg, *, excluded=(), capacity=False, force=False):
    '''按本机优先、节点配置顺序选择同一主机上的卡。
    capacity 仅用于为久候多卡任务挑一台可容纳它的节点。
    force 允许共用已占用的资源，仍检查节点资格和驱动。
    '''
    spec = job["spec"]
    check_load = not (force or capacity)
    hosts = [spec["host"]] if spec["host"] else sorted(spec["allowed_hosts"], key=lambda name: not cfg["hosts"][name].get("local"))
    for host in hosts:
        info = reports.get(host, {})
        if host in excluded or not info.get("docker") or cfg["hosts"][host].get("maintenance"):
            continue
        other = [item for item in active if item["attempts"][-1]["host"] == host]
        held = {gpu for item in other for gpu in item["attempts"][-1]["gpus"]}
        cpus = sum(item["spec"]["cpus"] for item in other)
        pending = sum(item["spec"]["memory_mib"] for item in other if item["state"] != "RUNNING")
        if info.get("cpus") is None or spec["cpus"] + (cpus if check_load else 0) > info["cpus"]:
            continue
        if check_load and (info.get("memory_mib") is None or spec["memory_mib"] + pending + 1024 > info["memory_mib"]):
            continue  # 留 1 GiB 给宿主机和其他常驻进程
        models = set(job.get("resume_models") or ())
        minimum = max(spec["min_driver_major"], job.get("image_min_driver", 0))
        pinned = spec.get("gpu_uuids")
        cards = [g for g in info.get("gpus", []) if (not pinned or g["uuid"] in pinned)
                 and (not models or g["name"] in models)
                 and g["driver_major"] >= minimum
                 and (force or (g["total_mib"] if capacity else g["free_mib"]) >= spec["min_free_mib"])
                 and (not check_load or (not g["busy"] and g["uuid"] not in held))]
        if len(cards) >= spec["gpus"]:
            if pinned:
                cards.sort(key=lambda g: pinned.index(g["uuid"]))
            elif force:
                cards.sort(key=lambda g: (g["busy"] or g["uuid"] in held, -g["free_mib"]))
            return host, cards[:spec["gpus"]]
    return None


def placement_error(job, report, active, cfg, preparing):
    '''解释 Agent 指定的节点或 GPU 为什么本轮不能启动。'''
    spec = job["spec"]
    host = spec["host"]
    if cfg["hosts"][host].get("maintenance") or not report.get("docker"):
        return Fault("queue", "NODE_UNAVAILABLE", "指定节点维护中或 Docker 不可用", host).info
    cards = {gpu["uuid"]: gpu for gpu in report.get("gpus", [])}
    pinned = spec["gpu_uuids"]
    missing = [gpu for gpu in pinned if gpu not in cards]
    if missing:
        return Fault("queue", "GPU_NOT_FOUND", "节点未报告指定 GPU", ", ".join(missing)).info
    minimum = max(spec["min_driver_major"], job.get("image_min_driver", 0))
    incompatible = [gpu for gpu in pinned if cards[gpu]["driver_major"] < minimum
                    or job.get("resume_models") and cards[gpu]["name"] not in job["resume_models"]]
    if incompatible:
        return Fault("queue", "GPU_INCOMPATIBLE", "指定 GPU 的型号或驱动不满足要求", ", ".join(incompatible)).info
    held = {gpu for item in active if item["attempts"][-1]["host"] == host for gpu in item["attempts"][-1]["gpus"]}
    busy = [gpu for gpu in pinned if gpu in held or cards[gpu]["busy"] or cards[gpu]["free_mib"] < spec["min_free_mib"]]
    if busy and not job.get("force_start"):
        return Fault("queue", "GPU_BUSY", "指定 GPU 被占用、已预留或余量不足", ", ".join(busy), transient=True).info
    if host in preparing and not job.get("force_start"):
        return Fault("queue", "NODE_PREPARING", "指定节点正在准备其他任务", host, transient=True).info
    return Fault("queue", "RESOURCE_BUSY", "指定节点资源或队列条件暂不满足", host, transient=True).info


class Service:
    def __init__(self, cfg, store, transport=None):
        '''初始化单一调度进程，并共享各节点执行和中心状态。'''
        self.cfg, self.store = cfg, store
        self.transport = transport or Transport(cfg)
        self.stopping = threading.Event()
        self.inflight = {}
        self.swanlab_inflight = None

    def mark(self, job, *, attempt=None, **fields):
        '''原子更新任务和最后一次尝试，保留并发写入的取消请求。'''
        def update(entry):
            '''在同一事务内更新显示状态和尝试诊断。'''
            entry.update(fields)
            if attempt is not None:
                entry["attempts"][-1].update(attempt)

        return self.store.change(job, update)

    def claim(self, job, host, cards, report):
        '''登记选中的卡，再允许执行线程开始准备。'''
        attempt = dict(id=uuid.uuid4().hex, host=host, gpus=[g["uuid"] for g in cards], models=[g["name"] for g in cards],
                       gpu_indices=[g.get("index") for g in cards],
                       boot_id=report["boot_id"], state="PREPARING", created_at=time.time())

        def update(entry):
            '''仅领取仍在排队的任务，避免覆盖 CLI 刚发出的取消。'''
            if entry["state"] == "QUEUED" and not entry["cancel_requested"]:
                entry["attempts"].append(attempt)
                entry.update(state="PREPARING", stage="snapshot", error=None, queue_diagnostics={})

        entry = self.store.change(job["id"], update)
        return entry if entry["attempts"] and entry["attempts"][-1]["id"] == attempt["id"] else None

    def safe_fault(self, info):
        '''脱敏异常字段，同时保留命令的真实诊断和退出码。'''
        try:
            values = secret_values(self.cfg["swanlab_netrc"])
        except Fault:
            values = []
        return {key: redact(value, values) if isinstance(value, str) else value for key, value in info.items()}

    def terminal(self, job, state, error=None, **extra):
        '''标记训练终态并交给独立的结果回收流程。'''
        flags = dict(state=state, error=error, finish_time=time.time(), **extra)
        return self.mark(job["id"], attempt=flags, result_state="PENDING" if job["attempts"][-1].get("work") else "NONE", **flags)

    def requeue(self, job, info, **fields):
        '''结束旧尝试并重新排队，取消与重新排队在同一事务内判定。'''
        now = time.time()

        def update(entry):
            '''有并发取消请求时停在终态，使已准备的产物仍可回收。'''
            old = entry["attempts"][-1]
            state = "CANCELED" if entry["cancel_requested"] else "FAILED"
            old.update(state=state, error=info, finish_time=now)
            if state == "CANCELED":
                entry.update(state=state, error=info, finish_time=now, result_state="PENDING" if old.get("work") else "NONE")
            else:
                entry.update(state="QUEUED", error=info, **fields)

        self.store.change(job["id"], update)

    def forget_secret(self, job):
        '''在确认旧容器不再运行后移除该尝试特有的登录文件。'''
        if not job["spec"]["swanlab"] or not job["attempts"][-1].get("work"):
            return
        try:
            self.worker(job, "forget_secret")
        except Fault as exc:
            self.mark(job["id"], attempt=dict(secret_cleanup_error=self.safe_fault(exc.info)))

    def prelaunch_failure(self, job, info):
        '''仅对已确认未启动的尝试应用忙卡等待或限次退避。'''
        job = self.store.get(job["id"])
        info = self.safe_fault(info)
        if job["cancel_requested"]:
            self.terminal(job, "CANCELED", info)
            return
        if info["code"] in {"GPU_BUSY", "RESOURCE_BUSY"}:
            delay, retries = self.cfg["poll_seconds"], job["prepare_retries"]
        elif info["transient"] and job["prepare_retries"] < len(BACKOFF):
            retries = job["prepare_retries"] + 1
            delay = BACKOFF[retries - 1]
        else:
            self.terminal(job, "FAILED", info)
            return
        self.forget_secret(job)
        self.requeue(job, info, stage="waiting", due=time.time() + delay, prepare_retries=retries)

    def worker(self, entry, action, **data):
        '''用尝试中登记的节点发送一个操作，避免把失败查到其他节点。'''
        attempt = entry["attempts"][-1]
        return self.transport.call(attempt["host"], action, attempt["id"], **data)

    def prepare(self, job):
        '''分发代码、数据、镜像和凭据，然后持久化 STARTING 并启动容器。'''
        key, spec = job["id"], job["spec"]
        attempt = job["attempts"][-1]
        host = attempt["host"]
        try:
            node = self.worker(job, "init")
            self.mark(key, attempt=dict(work=node["work"]), stage="transfer")
            self.transport.copy(host, self.store.state / "code" / job["code"], node["work"] + "/code")
            self.worker(job, "verify", kind="code", digest=job["code"], timeout=180)
            for data in sorted(set(job["assets"].values())):
                if not self.worker(job, "asset_ready", digest=data)["ready"]:
                    self.transport.copy(host, self.store.state / "assets" / data, node["root"] + "/assets/" + data)
                    self.worker(job, "verify", kind="asset", digest=data, timeout=900)
            if job["resume_retries"]:
                self.transport.copy(host, self.store.state / "checkpoints" / (key + ".pt"), node["work"] + "/resume.pt", directory=False)
                self.worker(job, "verify", kind="checkpoint", digest=job["checkpoint_digest"], timeout=300)
            self.mark(key, stage="image")
            port = self.cfg.get("registry_port")
            own_registry = bool(port and spec["image"].startswith(f"127.0.0.1:{port}/"))
            if own_registry:
                self.worker(job, "configure_registry", docker_config=registry_config(self.cfg))
            resolved = self.worker(job, "image", image=spec["image"], registry=own_registry,
                                   use_image_user=spec.get("use_image_user", False), timeout=930)
            minimum = resolved.get("min_driver") or 0
            self.mark(key, image_min_driver=minimum)
            if spec["swanlab"]:
                self.mark(key, stage="swanlab")
                self.worker(job, "secret", netrc=swanlab_file(self.cfg["swanlab_netrc"]))

            def launch_state(entry):
                '''在启动前原子检查取消请求，并登记可能已经生效的边界。'''
                if not entry["cancel_requested"] and not self.stopping.is_set():
                    entry.update(state="STARTING", stage="launch")
                    entry["attempts"][-1].update(state="STARTING", image_id=resolved["image_id"])

            entry = self.store.change(key, launch_state)
            if entry["state"] != "STARTING":
                raise Fault("launch", "START_DEFERRED", "尚未执行启动，等待取消或服务恢复", transient=True)
            resume = bool(job["resume_retries"])
            value = self.worker(entry, "start", timeout=110, job=key, spec=spec, assets=job["assets"],
                                gpus=attempt["gpus"], resume_path=resume, **resolved,
                                force_start=entry.get("force_start", False),
                                command=spec["checkpoint"]["command"] if resume else spec["command"])
            self.apply_status(self.store.get(key), value or {"status": "uncertain"})
        except Fault as exc:
            current = self.store.get(key)
            if exc.info["code"] == "DRIVER_INCOMPATIBLE" and not spec["host"] and current.get("image_min_driver", 0) > spec["min_driver_major"]:
                self.forget_secret(current)
                self.requeue(current, self.safe_fault(exc.info), stage="waiting", due=time.time())
            elif current["state"] == "STARTING" and exc.info["code"] not in {"GPU_BUSY", "RESOURCE_BUSY", "DRIVER_INCOMPATIBLE", "ATTEMPT_FENCED"}:
                # 发送启动后的 SSH 超时和 Docker 返回码均需先核对实际容器。
                self.unknown(current, exc.info)
            else:
                self.prelaunch_failure(current, exc.info)

    def unknown(self, job, info):
        '''保留潜在运行任务的占用，随后按同一尝试 ID 再核对。'''
        error = self.safe_fault(info)
        self.mark(job["id"], attempt=dict(state="UNKNOWN", error=error), state="UNKNOWN", stage="reconcile", error=error)

    def checkpoint(self, job):
        '''对已经写完的断点做双端内容校验，保留最近一份可迁移的副本。'''
        spec = job["spec"]["checkpoint"]
        attempt = job["attempts"][-1]
        self.mark(job["id"], attempt=dict(checkpoint_probe=time.time()))
        try:
            source = self.worker(job, "checkpoint", path=spec["path"], timeout=300)
            if not source or source["bytes"] == 0 or source["digest"] == job.get("checkpoint_digest"):
                return
            base = self.store.state / "checkpoints"
            with tempfile.TemporaryDirectory(dir=base) as directory:
                target = Path(directory) / "checkpoint.pt"
                self.transport.copy(attempt["host"], target, source["path"], pull=True, directory=False)
                if target.is_symlink() or target.stat().st_size != source["bytes"] or digest(target) != source["digest"]:
                    raise Fault("checkpoint", "CHECKPOINT_CHANGED", "复制期间断点发生变化，下次再同步", transient=True)
                target.replace(base / (job["id"] + ".pt"))
            self.mark(job["id"], checkpoint_digest=source["digest"], checkpoint_at=time.time(), checkpoint_error=None)
        except Fault as exc:
            self.mark(job["id"], checkpoint_error=self.safe_fault(exc.info))

    def record_logs(self, job):
        '''在终态回收已脱敏的训练 stdout 与 stderr，供失联时回读。'''
        attempt = job["attempts"][-1]
        try:
            logs = self.worker(job, "logs", tail=2000)
            if not self.valid_logs(logs):
                raise Fault("logs", "LOG_PROTOCOL_ERROR", "节点没有返回完整的 stdout 和 stderr", transient=True)
        except Fault as exc:
            self.mark(job["id"], attempt=dict(log_error=self.safe_fault(exc.info)))
            return {}
        try:
            secrets = secret_values(self.cfg["swanlab_netrc"])
        except Fault:
            secrets = []
        logs = {key: redact(value, secrets) for key, value in logs.items()}
        path = self.store.state / "logs" / (attempt["id"] + ".json")
        private_write(path, json.dumps(logs, ensure_ascii=False))
        self.mark(job["id"], attempt=dict(log_path=str(path), log_error=None))
        return logs

    @staticmethod
    def valid_logs(value):
        '''确认 stdout 和 stderr 都已返回，空流仍是有效内容。'''
        return isinstance(value, dict) and all(isinstance(value.get(key), str) for key in ("stdout", "stderr"))

    def recover_training(self, job, info):
        '''仅对已确认的节点中断和已有可靠断点，自动续训一次。'''
        job = self.store.get(job["id"])
        if job["cancel_requested"]:
            self.terminal(job, "CANCELED", info)
        elif job["spec"]["checkpoint"] and job.get("checkpoint_digest") and job["resume_retries"] == 0:
            attempt = job["attempts"][-1]
            self.forget_secret(job)
            self.requeue(job, info, stage="resume", due=time.time(), resume_retries=1, prepare_retries=0,
                         resume_models=attempt["models"])
        else:
            self.terminal(job, "FAILED", info)

    def apply_status(self, job, result):
        '''根据 Docker 的已确认状态更新任务，必要时按断点重排。'''
        key, attempt = job["id"], job["attempts"][-1]
        status = result["status"]
        rebooted = bool(result.get("boot_id") and attempt.get("boot_id") and result["boot_id"] != attempt["boot_id"])
        if status == "running":
            self.mark(key, attempt=dict(state="RUNNING", container_id=result["container_id"], started_at=result["started_at"],
                                       boot_id=result["boot_id"]), state="RUNNING", stage="training", error=None)
            spec = job["spec"]["checkpoint"]
            if spec and time.time() - attempt.get("checkpoint_probe", 0) >= spec["every_seconds"]:
                self.checkpoint(self.store.get(key))
            return
        if status in {"uncertain", "created", "restarting", "paused", "removing", "dead"}:
            info = job.get("error") or Fault("reconcile", "LAUNCH_UNCONFIRMED", "当前无法确认容器已启动或已退出", status, transient=True).info
            self.unknown(job, info)
            return
        if status in {"absent", "start_failed"}:
            if status == "absent" and attempt.get("container_id") and (job["cancel_requested"] or rebooted):
                self.mark(key, attempt=dict(container_stopped=True, stop_requested=bool(result.get("stop_requested"))))
            if job["cancel_requested"]:
                self.terminal(job, "CANCELED")
            elif rebooted:
                self.recover_training(job, Fault("training", "NODE_REBOOTED", "执行节点已重启且旧容器不再运行").info)
            elif attempt.get("container_id"):
                self.unknown(job, Fault("reconcile", "CONTAINER_DISAPPEARED", "曾确认运行的容器已从 Docker 记录消失").info)
            else:
                self.prelaunch_failure(job, result.get("error") or Fault("launch", "START_NOT_ACCEPTED", "节点确认此前尝试未启动", transient=True).info)
            return
        if status != "exited":
            self.unknown(job, Fault("reconcile", "UNEXPECTED_CONTAINER_STATE", "Docker 返回了未预期的状态", status).info)
            return
        self.mark(key, attempt=dict(container_id=result["container_id"], exit_code=result["exit_code"], oom=result["oom"],
                                    container_stopped=True, stop_requested=bool(result.get("stop_requested"))))
        logs = self.record_logs(self.store.get(key))
        details = (logs.get("stderr", "") + "\n" + logs.get("stdout", "")).strip()[-16000:] or result.get("detail", "")
        job = self.store.get(key)
        if job["cancel_requested"]:
            self.terminal(job, "CANCELED")
        elif result["exit_code"] == 0 and not result["oom"]:
            self.terminal(job, "SUCCEEDED")
        else:
            interrupted = (rebooted or "device has been lost" in details.lower()) and not result["oom"]
            if result["oom"]:
                code = "TRAINING_OOM"
            elif interrupted:
                code = "NODE_REBOOTED" if rebooted else "GPU_DEVICE_LOST"
            else:
                code = "TRAINING_FAILED"
            stage = "training"
            if job["spec"]["swanlab"] and not interrupted and not result["oom"] and any(
                    marker in details.lower() for marker in ("swanlaberror", "swanlab.error", "swanlab.errors", "swanlab.exceptions")):
                stage, code = "swanlab", "SWANLAB_RUNTIME_FAILED"
            error = self.safe_fault(Fault(stage, code, "训练容器退出", details, result["exit_code"]).info)
            if interrupted:
                self.recover_training(job, error)
            else:
                self.terminal(job, "FAILED", error)

    def observe(self, job):
        '''查询正在运行或未确认的尝试，按请求取消当前容器。'''
        if job["state"] == "PREPARING":
            self.prelaunch_failure(job, Fault("prepare", "PREPARE_INTERRUPTED", "上次准备被服务退出中断", transient=True).info)
            return
        try:
            result = self.worker(job, "cancel" if job["cancel_requested"] else "status", timeout=40)
        except Fault as exc:
            self.unknown(job, exc.info)
            return
        self.apply_status(job, result)

    def collect(self, job):
        '''校验并回收终态产物，保存正式路径后再允许后台清理。'''
        key = job["id"]
        try:
            with lock(self.store.state / "result-locks" / key, "results", "RESULT_BUSY"):
                job = self.store.get(key)
                if job["result_state"] == "SYNCED":
                    return
                root = job.get("result_root_override") or job["spec"].get("result_root")
                if root:
                    self.collect_native(job, Path(root))
                else:
                    self.collect_archive(job)
        except (Fault, OSError) as exc:
            info = exc.info if isinstance(exc, Fault) else Fault("results", "RESULT_SYNC_FAILED", "结果回收失败", str(exc)).info
            retries = job.get("result_retries", 0)
            retry = info["transient"] and retries < len(BACKOFF)
            self.mark(key, result_state="PENDING" if retry else "FAILED", result_error=self.safe_fault(info),
                      result_retries=retries + 1 if retry else retries, result_due=time.time() + (BACKOFF[retries] if retry else 0))
        finally:
            self.forget_secret(job)  # 训练结束且已尝试保全日志后，删除临时凭据副本

    def result_digest(self, job):
        '''返回已结束节点的产物摘要，纯镜像任务先导出容器内容。'''
        if job["spec"].get("use_image_user") and job["attempts"][-1].get("container_id"):
            self.worker(job, "export_outputs", timeout=1230)
        return self.worker(job, "result_digest", timeout=1800)["digest"]

    def receive_results(self, job, folder, expected=None):
        '''传输产物并对比两端摘要。旧式归档可沿用历史的类型校验。'''
        attempt = job["attempts"][-1]
        folder.mkdir(mode=0o700)
        self.transport.copy(attempt["host"], folder, attempt["work"] + "/outputs", pull=True, result=True)
        actual = tree(folder, checksum=expected is not None, directories=True)
        if expected is not None and actual != expected:
            raise Fault("results", "RESULT_CHECKSUM_MISMATCH", "回传产物与执行节点内容不一致", transient=True)
        return actual

    def collect_archive(self, job):
        '''缺少原生目录的任务回收到独立的 Spool 归档。'''
        key, attempt = job["id"], job["attempts"][-1]
        destination = self.store.state / "results" / key
        destination.mkdir(mode=0o700, exist_ok=True)
        target = destination / attempt["id"]
        if job.get("cleanup_state"):
            expected = self.result_digest(job)
        else:
            expected = None
            if job["spec"].get("use_image_user") and attempt.get("container_id") and not target.is_dir():
                self.worker(job, "export_outputs", timeout=1230)
        if not target.is_dir():
            with tempfile.TemporaryDirectory(dir=destination) as temp:
                folder = Path(temp) / "outputs"
                self.receive_results(job, folder, expected)
                folder.rename(target)
        elif expected is not None and tree(target, directories=True) != expected:
            raise Fault("results", "RESULT_CHECKSUM_MISMATCH", "原归档与执行节点内容不一致", str(target))
        self.mark(key, result_state="SYNCED", result_path=str(target), result_error=None)

    def collect_native(self, job, root):
        '''在目的文件系统验证一次，再按原生相对路径发布。'''
        key, attempt = job["id"], job["attempts"][-1]
        if result_root(str(root), self.cfg) != root:
            raise Fault("results", "RESULT_ROOT_UNAVAILABLE", "原生结果目录已变更或不可用", str(root))
        stage = results.staging(root, attempt["id"])
        stages = list(dict.fromkeys(job.get("result_stages", []) + [str(stage)]))
        self.mark(key, result_stages=stages)
        folder, verified = stage / "outputs", stage / "verified"
        if not verified.is_file():
            if folder.exists():
                shutil.rmtree(folder)  # 仅移除本次未校验完成的传输副本。
            actual = self.receive_results(job, folder, self.result_digest(job))
            private_write(verified, actual + "\n")
        elif verified.is_symlink() or verified.read_text().strip() != tree(folder, directories=True):
            raise Fault("results", "RESULT_STAGING_CHANGED", "已校验的暂存文件被更改，保留源文件", str(folder))
        entries = results.layout(folder)
        with lock(self.store.state / "publish-results.lock", "results", "RESULT_BUSY"):
            results.conflicts(folder, root, entries)
            self.mark(key, result_publishing=str(root))
            results.publish_files(folder, root, entries)
            self.mark(key, result_state="SYNCED", result_path=str(root), result_entries=entries,
                      result_error=None, result_publishing=None)

    def cleanup(self, job):
        '''保全终态输出与终端日志后，回收这次尝试的工作副本。'''
        key, attempt = job["id"], job["attempts"][-1]
        try:
            with lock(self.store.state / "result-locks" / key, "cleanup", "RESULT_BUSY"):
                job = self.store.get(key)
                destination = Path(job["result_path"])
                if destination != destination.resolve() or not destination.is_dir() or any(
                        (destination / name).resolve() != destination / name or
                        not ((destination / name).is_dir() if directory else (destination / name).is_file())
                        for name, directory in job.get("result_entries", {}).items()):
                    raise Fault("cleanup", "RESULT_ARCHIVE_MISSING", "回传文件已移动或不可读", str(destination))
                container = attempt.get("container_id")
                log_path = self.store.state / "logs" / (attempt["id"] + ".json")
                try:
                    logs_saved = not log_path.is_symlink() and self.valid_logs(json.loads(log_path.read_text()))
                except (OSError, ValueError):
                    logs_saved = False
                if container and not logs_saved:
                    if not self.record_logs(job):
                        problem = self.store.get(key)["attempts"][-1].get("log_error") or {}
                        raise Fault("cleanup", "LOG_NOT_ARCHIVED", "终端日志尚未回传",
                                    problem.get("detail") or problem.get("message", ""), transient=True)
                self.worker(job, "cleanup", timeout=180, archived_container=container)
                for path in job.get("result_stages", ()):
                    if Path(path).exists():
                        shutil.rmtree(path)
                checkpoint = job["spec"]["checkpoint"]
                backup = self.store.state / "checkpoints" / (key + ".pt")
                if checkpoint and backup.is_file():
                    final = destination / checkpoint["path"]
                    if final.is_file() and digest(final) == job.get("checkpoint_digest"):
                        backup.unlink()
                self.mark(key, cleanup_state="DONE", cleanup_error=None, result_stages=[])
        except (Fault, OSError) as exc:
            info = exc.info if isinstance(exc, Fault) else Fault("cleanup", "CLEANUP_FAILED", "无法回收任务工作副本", str(exc), transient=True).info
            retries = job.get("cleanup_retries", 0)
            retry = info["transient"] and retries < len(BACKOFF)
            self.mark(key, cleanup_state="PENDING" if retry else "FAILED", cleanup_error=self.safe_fault(info),
                      cleanup_retries=retries + int(retry), cleanup_due=time.time() + (BACKOFF[retries] if retry else 0))

    def sync_swanlab(self, job, destination):
        '''为已经终止的任务补报云端状态。
        日志和产物使用原有回收流程，云端失败只写入 SwanLab 的诊断字段。
        '''
        logs = ""
        log_path = next((a["log_path"] for a in reversed(job["attempts"]) if a.get("log_path")), None)
        if log_path:
            try:
                saved = json.loads(Path(log_path).read_text())
                logs = saved["stdout"] + "\n" + saved["stderr"]
            except (OSError, KeyError, TypeError, ValueError):
                pass  # 日志不可读时仍可根据已登记的项目或任务 ID 查找
        try:
            state = sync_run(job, destination, self.cfg["swanlab_netrc"], logs)
            self.mark(job["id"], swanlab_sync_state="SYNCED" if state else "NOT_FOUND", swanlab_state=state, swanlab_error=None)
        except (Fault, KeyError, TypeError, ValueError) as exc:
            info = exc.info if isinstance(exc, Fault) else Fault("swanlab", "SWANLAB_INVALID_RESPONSE", "SwanLab 返回格式异常", type(exc).__name__).info
            retries = job.get("swanlab_sync_retries", 0)
            retry = info["transient"] and retries < len(BACKOFF)
            self.mark(job["id"], swanlab_sync_state="PENDING" if retry else "FAILED",
                      swanlab_error=self.safe_fault(info), swanlab_sync_retries=retries + int(retry),
                      swanlab_sync_due=time.time() + (BACKOFF[retries] if retry else 0))

    def sync_swanlab_batch(self, jobs):
        '''在后台依次处理本轮可补报的任务，避免历史任务拖慢后续轮次。'''
        for job, destination in jobs:
            if self.stopping.is_set():
                return
            self.mark(job["id"], swanlab_sync_state="PENDING")
            try:
                self.sync_swanlab(job, destination)
            except Exception as error:
                info = Fault("swanlab", "SWANLAB_SYNC_FAILED", "SwanLab 终态补报出现内部错误", type(error).__name__).info
                self.mark(job["id"], swanlab_sync_state="FAILED", swanlab_error=self.safe_fault(info))

    def tick(self, pool):
        '''处理已知尝试与结果，随后从队列按资源可用性领取任务。'''
        for key, future in list(self.inflight.items()):
            if not future.done():
                continue
            del self.inflight[key]
            error = future.exception()
            if error:
                job = self.store.get(key)
                info = self.safe_fault(Fault("service", "SERVICE_ERROR", "处理任务时出现内部错误", repr(error)).info)
                if job["state"] in {"STARTING", "RUNNING", "UNKNOWN"}:
                    self.unknown(job, info)
                elif job["state"] == "PREPARING":
                    self.prelaunch_failure(job, info)
                elif job["state"] in TERMINAL and job["result_state"] == "SYNCED":
                    self.mark(key, cleanup_state="FAILED", cleanup_error=info)
                elif job["state"] in TERMINAL:
                    self.mark(key, result_state="FAILED", result_error=info)
        self.transport.keep_tunnels()
        jobs = self.store.jobs()
        now = time.time()
        active = [job for job in jobs if job["state"] in HELD]
        for job in jobs:
            if job["id"] in self.inflight:
                continue
            if job["state"] in HELD:
                self.inflight[job["id"]] = pool.submit(self.observe, job)
            elif job["state"] in TERMINAL and job["result_state"] == "PENDING" and job.get("result_due", 0) <= now:
                self.inflight[job["id"]] = pool.submit(self.collect, job)
            elif (job["state"] in TERMINAL and job["result_state"] == "SYNCED"
                  and job.get("cleanup_state") == "PENDING" and job.get("cleanup_due", 0) <= now):
                self.inflight[job["id"]] = pool.submit(self.cleanup, job)
        # 云端状态回报独占一个后台名额，训练归档继续使用原来的任务。
        if self.swanlab_inflight is not None and self.swanlab_inflight.done():
            self.swanlab_inflight = None
        if self.swanlab_inflight is None and not self.stopping.is_set():
            pending = [(job, destination) for job in reversed(jobs)
                       if job.get("swanlab_sync_state", "PENDING") == "PENDING" and job.get("swanlab_sync_due", 0) <= now
                       and (destination := termination_state(job))]
            if pending:
                self.swanlab_inflight = pool.submit(self.sync_swanlab_batch, pending)
        waiting = [job for job in jobs if job["state"] == "QUEUED" and job["due"] <= now and job["id"] not in self.inflight]
        if not waiting or self.stopping.is_set():
            return
        reports = self.transport.inventory()
        preparing = {job["attempts"][-1]["host"] for job in active if job["state"] in {"PREPARING", "STARTING"}}
        priority = next((job for job in waiting if not job.get("force_start") and job["spec"]["gpus"] > 1
                         and now - job["submitted_at"] >= self.cfg["large_job_wait_seconds"]), None)
        drain = select(priority, reports, active, self.cfg, capacity=True) if priority else None
        waiting.sort(key=lambda job: (not job.get("force_start"), drain is not None and job["id"] != priority["id"]))
        for job in waiting:
            if job.get("force_start"):
                # 即使同节点正在准备其他任务，也优先尝试可用的空卡。
                choice = select(job, reports, active, self.cfg) or select(job, reports, active, self.cfg, force=True)
            else:
                blocked = preparing | ({drain[0]} if drain and job["id"] != priority["id"] else set())
                choice = select(job, reports, active, self.cfg, excluded=blocked)
            if choice and not self.stopping.is_set():
                host, cards = choice
                claimed = self.claim(job, host, cards, reports[host])
                if claimed:
                    active.append(claimed)
                    preparing.add(host)
                    self.inflight[job["id"]] = pool.submit(self.prepare, claimed)
            elif not choice:
                allowed = [job["spec"]["host"]] if job["spec"]["host"] else job["spec"]["allowed_hosts"]
                diagnostics = {name: reports[name]["errors"] for name in allowed if reports.get(name, {}).get("errors") and not reports[name].get("docker")}
                if job["spec"].get("gpu_uuids") and allowed[0] not in diagnostics:
                    host = allowed[0]
                    diagnostics[host] = [placement_error(job, reports.get(host, {}), active, self.cfg, preparing)]
                elif job.get("force_start"):
                    spec = job["spec"]
                    minimum = max(spec["min_driver_major"], job.get("image_min_driver", 0))
                    for name in allowed:
                        if name in diagnostics:
                            continue
                        report = reports.get(name, {})
                        cards = ", ".join(f"{g['name']} 驱动 {g['driver_major']}" for g in report.get("gpus", [])) or "无"
                        detail = (f"要求 CPU {spec['cpus']}、GPU {spec['gpus']} 张、驱动主版本 ≥ {minimum}。"
                                  f"检测到 CPU {report.get('cpus')}、GPU {cards}、Docker 就绪 {bool(report.get('docker'))}、"
                                  f"维护状态 {bool(self.cfg['hosts'][name].get('maintenance'))}。")
                        diagnostics[name] = [Fault("queue", "FORCE_TARGET_UNAVAILABLE", "当前节点无法启动任务", detail).info]
                if diagnostics != job.get("queue_diagnostics", {}):
                    self.mark(job["id"], queue_diagnostics=diagnostics)

    def serve(self):
        '''独占中心调度锁，按固定周期工作直至 systemd 请求停止。'''
        def stop(_signal, _frame):
            '''让后续轮次停止领取新任务。'''
            self.stopping.set()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        with lock(self.store.state / "service.lock", "service", "SERVICE_ALREADY_ACTIVE"):
            pool = ThreadPoolExecutor(max_workers=12)
            try:
                while not self.stopping.is_set():
                    private_write(self.store.state / "heartbeat", str(time.time()))
                    self.tick(pool)
                    self.stopping.wait(self.cfg["poll_seconds"])
            finally:
                self.transport.close()
                pool.shutdown(wait=False, cancel_futures=True)
