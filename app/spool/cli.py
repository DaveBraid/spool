"""Spool 命令行入口。"""

import argparse
from contextlib import suppress
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import time
import uuid

from . import __version__
from .images import Registry, clean as clean_images
from . import results as result_files
from .service import Service
from .state import HELD, Store, TERMINAL, config, identify, json_read, normalize, result_root, source_path
from .transport import Transport, publish, registry_init
from .worker import Fault, lock, redact, secret_values, swanlab_file, tree


class Parser(argparse.ArgumentParser):
    def error(self, message):
        '''让参数解析失败也走统一的错误返回。'''
        raise Fault("input", "INVALID_ARGUMENT", message)


def parser():
    '''定义面向用户和 Agent 的 CLI 入口。'''
    cli = Parser(prog="spool", description="GPU 训练调度与执行")
    cli.add_argument("--config", type=Path, default=Path(os.environ.get("SPOOL_CONFIG", "~/.config/spool/config.json")).expanduser())
    cli.add_argument("--json", action="store_true", help="以 JSON 形式输出，包括失败详情")
    cli.add_argument("--version", action="version", version=__version__)
    actions = cli.add_subparsers(dest="action", required=True)
    for name, help_text in (("serve", "运行调度服务"), ("health", "查看调度服务心跳"), ("nodes", "实时查看节点就绪状态"),
                            ("assets", "查看已登记的数据版本")):
        actions.add_parser(name, help=help_text)
    submit = actions.add_parser("submit", help="冻结输入并将任务入队")
    submit.add_argument("file", type=Path)
    submit.add_argument("--id", help="幂等提交 ID。提交结果未知时用相同 ID 查询或重发")
    submit.add_argument("--force", action="store_true", help="强制启动，允许共用忙卡")
    queue = actions.add_parser("queue", help="查看排队和执行中的任务")
    queue.add_argument("--all", action="store_true", help="同时查看终态任务")
    for name, help_text in (("status", "查看任务、尝试和真实错误"), ("cancel", "请求停止自己的任务"),
                            ("force-start", "强制启动已排队的任务"),
                            ("cleanup", "重试已回传任务的工作副本清理")):
        actions.add_parser(name, help=help_text).add_argument("job")
    collect = actions.add_parser("collect", help="重新安排结果回收")
    collect.add_argument("job")
    collect.add_argument("--output", help="为尚未发布的结果重新指定已有目标根目录")
    results = actions.add_parser("results", help="将已回传的产物导出到指定目录")
    results.add_argument("job")
    results.add_argument("--output", type=Path, required=True, help="执行 CLI 的机器上的新目录")
    results.add_argument("--move", action="store_true", help="将旧式归档迁入同盘的原生目录")
    logs = actions.add_parser("logs", help="读取训练的 stdout 和 stderr")
    logs.add_argument("job")
    logs.add_argument("--attempt")
    logs.add_argument("--tail", type=int, default=200)
    asset = actions.add_parser("asset-add", help="登记一份不可变的数据版本")
    asset.add_argument("asset")
    asset.add_argument("directory")
    registry = actions.add_parser("registry-init", help="启动调度主机的私有镜像仓库")
    registry.add_argument("--image", default="registry:3", help="官方 Distribution 镜像引用")
    push = actions.add_parser("publish", help="发布本机镜像并输出 digest 引用")
    push.add_argument("image")
    push.add_argument("tag", help="仓库名:标签，例如 training/isaaclab:v1")
    keep = actions.add_parser("image-keep", help="登记验收通过的镜像并回收过期节点缓存")
    keep.add_argument("reference", help="Spool 私有仓库的 repository@sha256:digest")
    cleanup = actions.add_parser("image-clean", help="预览或执行 Spool 私有镜像清理")
    cleanup.add_argument("--apply", action="store_true", help="执行清理，省略时只预览")
    cleanup.add_argument("--registry", action="store_true", help="同时检查或回收私有仓库空间")
    return cli


def export_results(entry, output):
    '''
    从任务记录读取已回传的目录，复制到调用方指定的新目录。
    返回原归档和导出路径。完整复制前保留临时目录，路径冲突时保护用户文件。
    '''
    if entry["result_state"] != "SYNCED" or not entry.get("result_path"):
        detail = (entry.get("result_error") or {}).get("detail") or entry["result_state"]
        raise Fault("results", "RESULTS_NOT_SYNCED", "任务产物尚未回传到调度主机", detail)
    source = Path(entry["result_path"])
    if source.is_symlink() or not source.is_dir():
        raise Fault("results", "RESULT_ARCHIVE_MISSING", "无法读取任务的原始归档", str(source))

    try:
        output = output.expanduser()
        if output.is_symlink():
            raise FileExistsError(str(output))
        output = output.resolve()
        if output.exists():
            raise FileExistsError(str(output))
        if output.is_relative_to(source.resolve()):
            raise Fault("results", "RESULT_EXPORT_OVERLAP", "导出目录不能位于原始归档内", str(output))
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent, prefix=".spool-export-") as temporary:
            staged = Path(temporary) / "results"
            if "result_entries" in entry:
                staged.mkdir(mode=0o700)
                result_files.copy_files(source, staged, entry["result_entries"])
            else:
                shutil.copytree(source, staged, symlinks=True)
            try:
                tree(staged, checksum=False)
            except Fault as exc:
                raise Fault("results", "RESULT_EXPORT_INVALID_FILES", exc.info["message"], exc.info["detail"]) from None
            output.mkdir(mode=0o700)  # 发布前独占目标名，防止并发导出覆盖已有目录。
            try:
                staged.rename(output)
            except BaseException:
                with suppress(OSError):
                    output.rmdir()
                raise
    except FileExistsError as exc:
        raise Fault("results", "RESULT_EXPORT_EXISTS", "导出目标已存在，请指定新目录", str(exc)) from None
    except (OSError, RuntimeError) as exc:
        raise Fault("results", "RESULT_EXPORT_FAILED", "复制任务产物失败", str(exc)) from None
    return {"job_id": entry["id"], "attempt_id": entry["attempts"][-1]["id"],
            "result_path": str(source), "output_path": str(output)}


def operate(args, cfg, store):
    '''执行一个命令，将队列与执行状态作为普通 JSON 数据返回。'''
    action = args.action
    if action == "serve":
        Service(cfg, store).serve()
        return {}
    if action == "health":
        path = store.state / "heartbeat"
        age = round(time.time() - float(path.read_text()), 1) if path.is_file() else None
        try:
            ready = bool(swanlab_file(cfg["swanlab_netrc"]))
        except Fault:
            ready = False
        return {"version": __version__, "active": age is not None and age <= max(60, cfg["poll_seconds"] * 3),
                "heartbeat_age_seconds": age, "swanlab_credential_present": ready, "state": str(store.state)}
    if action == "nodes":
        result = Transport(cfg).inventory()
        held = {name: set() for name in result}
        for job in store.jobs():
            if job["state"] in HELD:
                attempt = job["attempts"][-1]
                held.get(attempt["host"], set()).update(attempt["gpus"])
        for name, node in result.items():
            node["local"] = cfg["hosts"][name].get("local", False)
            node["maintenance"] = cfg["hosts"][name].get("maintenance", False)
            node["placement"] = cfg["hosts"][name].get("placement", {})
            node["spool_reserved_count"] = len(held[name])
            for gpu in node.get("gpus", ()):
                gpu["spool_reserved"] = gpu["uuid"] in held[name]
            path = store.state / "logs" / f"registry-tunnel-{name}.log"
            if path.is_file():
                diagnostic = path.read_text(errors="replace").strip()
                if diagnostic:
                    node["registry_tunnel_diagnostic"] = redact(diagnostic[-4000:], [])
        return result
    if action == "submit":
        spec = normalize(json_read(args.file), cfg)
        private = spec["image"].startswith(f"127.0.0.1:{cfg.get('registry_port')}/")
        guard = lock(store.state / "images.lock", "image", "IMAGE_MAINTENANCE_BUSY") if private else nullcontext()
        with guard:
            return store.enqueue(identify(args.id, "job") if args.id else uuid.uuid4().hex, spec, force_start=args.force)
    if action == "queue":
        return [job for job in store.jobs() if args.all or job["state"] not in TERMINAL]
    if action == "assets":
        return store.assets()
    if action == "asset-add":
        source = source_path(args.directory, cfg["source_roots"])
        return store.add_asset(identify(args.asset, "asset"), source)
    if action in {"registry-init", "publish", "image-keep", "image-clean"}:
        with lock(store.state / "images.lock", "image", "IMAGE_MAINTENANCE_BUSY"):
            if action == "registry-init":
                return registry_init(cfg, args.image)
            if action == "publish":
                return publish(cfg, args.image, args.tag)
            if action == "image-keep":
                Registry(cfg).accept(store, args.reference)
                return {"accepted": args.reference, **clean_images(cfg, store, apply=True)}
            return clean_images(cfg, store, apply=args.apply, registry_cleanup=args.registry)
    job = identify(args.job, "job")
    if action == "status":
        return store.get(job)
    if action == "force-start":
        def force_start(entry):
            '''切换排队任务的调度方式，保留原任务 ID、快照和重试记录。'''
            if entry["state"] == "QUEUED" and not entry["cancel_requested"]:
                entry.update(force_start=True, due=time.time(), queue_diagnostics={})
            elif not (entry.get("force_start") and entry["state"] in HELD and not entry["cancel_requested"]):
                raise Fault("queue", "JOB_NOT_QUEUED", "当前任务状态不允许强制启动", entry["state"])

        return store.change(job, force_start)
    if action == "cancel":
        def cancel(entry):
            '''取消排队项，或通知常驻服务核对并停止已领取容器。'''
            if entry["state"] not in TERMINAL:
                entry["cancel_requested"] = True
                if entry["state"] == "QUEUED":
                    entry.update(state="CANCELED", stage="cancel", finish_time=time.time())

        return store.change(job, cancel)
    if action == "collect":
        output = str(result_root(args.output, cfg)) if getattr(args, "output", None) else None

        def collect(entry):
            '''恢复终态任务的结果同步，保留训练结果状态。'''
            if entry["state"] not in TERMINAL or not entry["attempts"] or not entry["attempts"][-1].get("work"):
                raise Fault("results", "RESULTS_NOT_READY", "任务未结束或没有节点产物目录")
            if output:
                if entry.get("result_publishing") or entry["result_state"] == "SYNCED":
                    raise Fault("results", "RESULT_PATH_LOCKED", "已开始发布结果，当前回传目录已固定", entry.get("result_publishing") or entry["result_path"])
                entry.update(result_root_override=output, cleanup_state="PENDING")
            if entry["result_state"] != "SYNCED":
                entry.update(result_state="PENDING", result_retries=0, result_due=time.time(), result_error=None)

        with lock(store.state / "result-locks" / job, "results", "RESULT_BUSY"):
            return store.change(job, collect)
    if action == "cleanup":
        def cleanup(entry):
            '''给已回收结果的受管任务重试工作副本清理。'''
            if entry["result_state"] != "SYNCED" or not entry.get("cleanup_state"):
                raise Fault("cleanup", "CLEANUP_NOT_READY", "任务尚无可清理的已校验结果")
            if entry["cleanup_state"] != "DONE":
                entry.update(cleanup_state="PENDING", cleanup_retries=0, cleanup_due=time.time(), cleanup_error=None)

        with lock(store.state / "result-locks" / job, "cleanup", "RESULT_BUSY"):
            return store.change(job, cleanup)
    entry = store.get(job)
    if action == "results":
        with lock(store.state / "result-locks" / job, "results", "RESULT_BUSY"):
            entry = store.get(job)
            if getattr(args, "move", False):
                with lock(store.state / "publish-results.lock", "results", "RESULT_BUSY"):
                    return result_files.move_archive(store, entry, result_root(str(args.output), cfg))
            return export_results(entry, args.output)
    attempt = next((known for known in reversed(entry["attempts"]) if args.attempt is None or known["id"] == args.attempt), None)
    if attempt is None:
        raise Fault("logs", "ATTEMPT_NOT_FOUND", "尚无匹配的执行尝试")
    tail = max(1, min(args.tail, 2000))
    if attempt.get("log_path"):
        saved = json_read(Path(attempt["log_path"]))
        logs = {name: "\n".join(content.splitlines()[-tail:]) for name, content in saved.items()}
    else:
        logs = Transport(cfg).call(attempt["host"], "logs", attempt["id"], tail=tail)
    try:
        secrets = secret_values(cfg["swanlab_netrc"])
    except Fault:
        secrets = []
    return {"job_id": job, "attempt_id": attempt["id"], "host": attempt["host"],
            **{stream: redact(content, secrets) for stream, content in logs.items()}}


def display(action, data):
    '''以可读形式打印任务概况和错误详情。'''
    if action == "logs":
        if data["stdout"]:
            print(data["stdout"].rstrip())
        if data["stderr"]:
            print(data["stderr"].rstrip(), file=sys.stderr)
    elif action == "results":
        print("已导出", data["output_path"])
    elif action == "queue":
        for entry in data:
            print(entry["id"], entry["state"], entry["spec"]["name"])
        if not data:
            print("队列为空")
    elif action in {"submit", "status", "cancel", "collect", "cleanup", "force-start"}:
        print(data["id"], data["state"], data["spec"]["name"])
        if data.get("force_start"):
            print("  调度 强制启动")
        if data["attempts"]:
            attempt = data["attempts"][-1]
            print("  节点", attempt["host"], "GPU", ",".join(attempt["gpus"]) or "CPU", "尝试", attempt["id"])
        if data.get("result_path"):
            print("  产物", data["result_path"])
        if data.get("cleanup_state"):
            print("  清理", data["cleanup_state"])
        if data.get("swanlab_sync_state"):
            labels = {"ABORTED": "Ctrl+C", "CRASHED": "中断", "FINISHED": "完成"}
            print("  SwanLab", data["swanlab_sync_state"], labels.get(data.get("swanlab_state"), ""))
        for key in ("error", "checkpoint_error", "result_error", "cleanup_error", "swanlab_error"):
            if data.get(key):
                fault = data[key]
                print(f"  [{fault['stage']}/{fault['code']}] {fault['message']}")
                if fault["detail"]:
                    print(fault["detail"].rstrip())
        for host, diagnostics in data.get("queue_diagnostics", {}).items():
            for fault in diagnostics:
                print(f"  {host} [{fault['code']}] {fault['detail'].strip() or fault['message']}")
    else:
        print(json.dumps(data, ensure_ascii=False, indent=2))


def main():
    '''运行一次 CLI，参数和命令异常统一以非零退出和结构化错误返回。'''
    os.umask(0o077)
    raw = sys.argv[1:]
    machine = "--json" in raw
    try:
        args = parser().parse_args([flag for flag in raw if flag != "--json"])
        cfg = config(args.config.expanduser())
        data = operate(args, cfg, Store(cfg["state"]))
        if machine:
            print(json.dumps({"ok": True, "data": data}, ensure_ascii=False))
        else:
            display(args.action, data)
        return 0
    except (Fault, OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
        info = error.info if isinstance(error, Fault) else Fault("cli", "OPERATION_FAILED", type(error).__name__, str(error)).info
        if machine:
            print(json.dumps({"ok": False, "error": info}, ensure_ascii=False))
        else:
            print(f"[{info['stage']}/{info['code']}] {info['message']}", file=sys.stderr)
            if info["detail"]:
                print(info["detail"].rstrip(), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
