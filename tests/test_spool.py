"""验证首次实现中会影响训练和凭据的状态边界。"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr
from io import StringIO
import json
import os
from pathlib import Path
from subprocess import TimeoutExpired
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

from spool.cli import operate, parser
from spool.images import manifest_removals, retained_images
from spool.service import Service, placement_error, select
from spool.state import Store, normalize
from spool.swanlab import sync_run, termination_state
from spool.worker import COMMAND_TIMEOUT, Fault, cleanup_attempt, dispatch, export_outputs, gpu_inventory, launch, status, swanlab_file, private_write, tree


def card(uuid, model="4090"):
    '''为状态机测试构造一张空闲 GPU。'''
    return dict(uuid=uuid, name=model, driver_major=580, total_mib=24000, free_mib=23000, busy=False)


def report(gpus):
    '''返回符合选卡逻辑的节点资源样本。'''
    return dict(cpus=16, memory_mib=64000, docker=True, boot_id="boot-a", gpus=gpus)


class FakeTransport:
    def __init__(self):
        '''模拟 SSH 返回及晚到的容器状态。'''
        self.start_calls = 0
        self.container = None
        self.results = {}
        self.cleaned = False

    def call(self, host, action, attempt, **kwargs):
        '''模拟一次启动回执丢失，保留真实运行状态用于后续核对。'''
        if action == "init":
            return {"work": "/work/" + attempt, "root": "/work"}
        if action == "image":
            return {"image_id": "sha256:resolved", "min_driver": None}
        if action == "start":
            self.start_calls += 1
            self.container = dict(status="running", container_id="container", started_at="time", boot_id="boot-a")
            raise Fault("launch", "NODE_TRANSPORT_FAILED_TIMEOUT", "SSH 回执丢失", "Connection lost", transient=True)
        if action == "status":
            return self.container
        if action == "logs":
            return {"stdout": "training started", "stderr": "Traceback\nRuntimeError: real training error"}
        if action == "result_digest":
            with tempfile.TemporaryDirectory() as source:
                self.copy(host, Path(source), "outputs", result=True)
                return {"digest": tree(Path(source), directories=True)}
        if action == "cleanup":
            self.cleaned = True
        return {}

    def copy(self, host, local, remote, **kwargs):
        '''将测试节点的产物复制到调度端。'''
        if kwargs.get("result"):
            for name, body in self.results.items():
                path = local / name
                path.parent.mkdir(exist_ok=True, parents=True)
                path.write_text(body)

    def keep_tunnels(self):
        '''测试无需建立镜像仓库转发。'''


class Spool(unittest.TestCase):
    def setUp(self):
        '''为每个测试创建独立中心数据与两个节点。'''
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.cfg = dict(hosts={"remote": {"ssh": "remote"}, "dispatcher": {"local": True}}, source_roots=[self.root],
                        state=self.root / "state", poll_seconds=15, swanlab_netrc=self.root / ".netrc")
        self.store = Store(self.cfg["state"])
        self.transport = FakeTransport()
        self.service = Service(self.cfg, self.store, self.transport)
        self.inventory = {"remote": report([card("GPU-b")]), "dispatcher": report([card("GPU-a")])}

    def spec(self, **kwargs):
        '''补齐一份无需真实镜像拉取的任务规格。'''
        return normalize(dict(image="registry/train@sha256:" + "a" * 64, command=["python", "train.py"], **kwargs), self.cfg)

    def claim(self, key, **kwargs):
        '''创建并领取一个本地测试任务。'''
        job = self.store.enqueue(key, self.spec(**kwargs))
        return self.service.claim(job, "dispatcher", self.inventory["dispatcher"]["gpus"], self.inventory["dispatcher"])

    def test_snapshot_and_idempotency(self):
        '''保留入队时的代码，重复提交同一 ID 不会生成第二单。'''
        source = self.root / "project"
        source.mkdir()
        (source / "train.py").write_text("old")
        (source / ".env").write_text("private")
        spec = self.spec(code=str(source))
        job = self.store.enqueue("trial", spec)
        (source / "train.py").write_text("new")
        repeat = self.store.enqueue("trial", spec)
        self.assertEqual(job["code"], repeat["code"])
        frozen = self.store.state / "code" / job["code"]
        self.assertEqual((frozen / "train.py").read_text(), "old")
        self.assertFalse((frozen / ".env").exists())
        with self.assertRaises(Fault):
            self.store.enqueue("trial", self.spec(name="different"))

    def test_live_log_timeout_reports_progress_and_preserves_training(self):
        '''验证无节点回包和收到阶段后的超时，保留原尝试及训练状态。'''
        self.cfg["worker_root"] = str(self.root / "worker")
        job = self.claim("live-log")
        self.service.mark(job["id"], state="RUNNING", stage="training", attempt=dict(host="remote", state="RUNNING"))
        args = parser().parse_args(["logs", job["id"], "--tail", "180"])
        before = self.store.get(job["id"])
        phase = "节点已确认容器，正在读取 Docker 日志"
        for output, expected in ((None, "未收到进程输出"), (phase.encode(), phase)):
            with self.subTest(output=output):
                timeout = TimeoutExpired(["ssh"], 40, stderr=output)
                with patch("spool.worker.subprocess.run", side_effect=timeout) as ssh, self.assertRaises(Fault) as error:
                    operate(args, self.cfg, self.store)
                self.assertGreater(ssh.call_args.kwargs["timeout"], 2 * COMMAND_TIMEOUT)
                self.assertEqual(error.exception.info["code"], "NODE_TRANSPORT_FAILED_TIMEOUT")
                self.assertTrue(error.exception.info["transient"])
                self.assertIn("SSH 节点 remote\n已等待 40 秒", error.exception.info["detail"])
                self.assertIn(expected, error.exception.info["detail"])
        self.assertEqual(self.store.get(job["id"]), before)

    def test_docker_log_timeout_preserves_phase_and_redacts_partial_output(self):
        '''确认节点已找到容器后，返回 Docker 自己的超时错误和脱敏输出。'''
        root = self.root / "worker"
        attempt = "a" * 32
        dispatch(dict(action="init", root=str(root), attempt=attempt))
        private_write(root / "runs" / attempt / "secrets/.netrc", "machine swanlab.cn login user password mock-secret\n")
        timeout = TimeoutExpired(["docker", "logs"], COMMAND_TIMEOUT, stderr=b"partial training output mock-secret")
        with patch("spool.worker.inspect", return_value={"status": "running"}), \
             patch("spool.worker.subprocess.run", side_effect=timeout), \
             redirect_stderr(StringIO()) as progress, self.assertRaises(Fault) as error:
            dispatch(dict(action="logs", root=str(root), attempt=attempt, tail=180))
        self.assertIn("正在读取 Docker 日志", progress.getvalue())
        self.assertEqual(error.exception.info["code"], "LOG_READ_FAILED_TIMEOUT")
        self.assertIn("partial training output [REDACTED]", error.exception.info["detail"])

    def test_pinned_cards_keep_identity_and_order_when_busy(self):
        '''精确放置不静默换卡，允许强制时仍维持 UUID 顺序。'''
        self.inventory["remote"]["gpus"] = [card("GPU-b"), card("GPU-c"), card("GPU-d")]
        job = self.store.enqueue("pinned", self.spec(host="remote", gpus=2, gpu_uuids=["GPU-d", "GPU-b"]))
        chosen = select(job, self.inventory, [], self.cfg)
        self.assertEqual([gpu["uuid"] for gpu in chosen[1]], ["GPU-d", "GPU-b"])
        self.inventory["remote"]["gpus"][0]["busy"] = True
        self.assertIsNone(select(job, self.inventory, [], self.cfg))
        self.assertEqual(placement_error(job, self.inventory["remote"], [], self.cfg, set())["code"], "GPU_BUSY")
        self.assertEqual([gpu["uuid"] for gpu in select(job, self.inventory, [], self.cfg, force=True)[1]],
                         ["GPU-d", "GPU-b"])

    def test_native_collection_waits_for_logs_before_cleanup(self):
        '''训练结果、原生配置与原生记录归位后，日志仍是清理的前置条件。'''
        root = self.root / "project"
        root.mkdir()
        self.transport.results = {"logs/run/model.pt": "checkpoint", "logs/run/swanlab/events": "metric",
                                  "outputs/date/time/.hydra/config.yaml": "config"}
        job = self.claim("native", result_root=str(root))
        self.service.mark("native", state="SUCCEEDED", result_state="PENDING",
                          attempt={"work": "/work/attempt", "container_id": "container"})
        self.service.collect(self.store.get("native"))
        synced = self.store.get("native")
        self.assertEqual((synced["result_state"], synced["result_path"]), ("SYNCED", str(root)))
        for path, content in self.transport.results.items():
            self.assertEqual((root / path).read_text(), content)
        self.assertFalse((self.store.state / "results" / "native").exists())
        with patch.object(self.service, "record_logs", return_value={}):
            self.service.cleanup(synced)
        self.assertFalse(self.transport.cleaned)
        self.assertEqual(self.store.get("native")["cleanup_error"]["code"], "LOG_NOT_ARCHIVED")
        self.service.cleanup(self.store.get("native"))
        self.assertTrue(self.transport.cleaned)
        self.assertEqual(self.store.get("native")["cleanup_state"], "DONE")
        self.assertFalse((root / (".spool-return-" + job["attempts"][-1]["id"])).exists())

    def test_native_conflict_preserves_local_results_and_recovers(self):
        '''产物冲突不覆盖原文件，也不会触发节点清理。'''
        root = self.root / "project"
        root.mkdir()
        (root / "model.pt").write_text("local")
        self.transport.results = {"model.pt": "remote", "training.log": "run"}
        self.claim("conflict", result_root=str(root))
        self.service.mark("conflict", state="SUCCEEDED", result_state="PENDING", attempt={"work": "/work/attempt"})
        self.service.collect(self.store.get("conflict"))
        entry = self.store.get("conflict")
        self.assertEqual(entry["result_error"]["code"], "RESULT_PATH_CONFLICT")
        self.assertEqual((root / "model.pt").read_text(), "local")
        self.assertFalse((root / "training.log").exists())
        self.assertFalse(self.transport.cleaned)
        alternate = self.root / "alternate"
        alternate.mkdir()
        args = parser().parse_args(["collect", "conflict", "--output", str(alternate)])
        self.service.collect(operate(args, self.cfg, self.store))
        self.assertEqual((alternate / "model.pt").read_text(), "remote")

    def test_cleanup_rejects_running_or_unarchived_container(self):
        '''即使收集方误调用，节点也不会删除运行中或日志未归档的容器。'''
        with patch("spool.worker.inspect", return_value={"status": "running", "container_id": "actual"}), \
                patch("spool.worker.run") as docker:
            with self.assertRaisesRegex(Fault, "尚未确认退出"):
                cleanup_attempt("a" * 32, self.root, "actual")
            docker.assert_not_called()
        with patch("spool.worker.inspect", return_value={"status": "exited", "container_id": "actual"}), \
                patch("spool.worker.run") as docker:
            with self.assertRaisesRegex(Fault, "终端日志"):
                cleanup_attempt("a" * 32, self.root, "different")
            docker.assert_not_called()

    def test_registry_preserves_shared_child_and_queued_image(self):
        '''多架构索引共享子清单时，只回收无引用的旧版本。'''
        registry = SimpleNamespace(address="127.0.0.1:18747")
        prefix = registry.address + "/training/project@sha256:"
        old, prior, current, child, unique = [prefix + letter * 64 for letter in "abcde"]
        for ref in (old, prior, current):
            self.store.accept_image(ref)
        spec = self.spec()
        spec["image"] = old
        self.store.enqueue("pending", spec)
        stable, protected, _ = retained_images(self.store)
        self.assertEqual(stable[prefix.split("@")[0]], [current, prior])
        self.assertIn(old, protected)
        graph = {old: [child, unique], prior: [child], current: [], child: [], unique: []}
        registry.manifest = lambda ref: dict(manifests=[dict(digest=c.split("@")[1]) for c in graph[ref]])
        for ref in graph:
            link = self.root / "docker/registry/v2/repositories/training/project/_manifests/revisions/sha256" / ref[-64:] / "link"
            link.parent.mkdir(parents=True)
            link.touch()
        self.assertEqual(manifest_removals(registry, self.root, {old}, {prior, current}), [old, unique])

    def test_dispatcher_preferred_and_unknown_reserved(self):
        '''默认优先 调度主机，UNKNOWN 的卡仍不可再分。'''
        waiting = self.store.enqueue("waiting", self.spec())
        self.assertEqual(select(waiting, self.inventory, [], self.cfg)[0], "dispatcher")
        hold = self.claim("held")
        self.service.unknown(hold, Fault("launch", "UNKNOWN", "状态未知").info)
        self.assertEqual(select(waiting, self.inventory, [self.store.get("held")], self.cfg)[0], "remote")
        explicit = self.store.enqueue("explicit", self.spec(host="remote"))
        self.assertEqual(select(explicit, self.inventory, [], self.cfg)[0], "remote")
        self.inventory["remote"]["gpus"][0]["driver_major"] = 600
        waiting["image_min_driver"] = 600
        self.assertEqual(select(waiting, self.inventory, [], self.cfg)[0], "remote")

    def test_force_start_commands_keep_the_job_snapshot(self):
        '''新提交和已排队任务都可启用强制模式，每个 ID 仍只领取一次。'''
        source = self.root / "force-job.json"
        source.write_text(json.dumps({"image": "registry/train@sha256:" + "a" * 64, "command": ["python", "train.py"]}))
        submit = parser().parse_args(["submit", str(source), "--id", "new-force", "--force"])
        first = operate(submit, self.cfg, self.store)
        self.assertTrue(first["force_start"])
        self.assertEqual(operate(submit, self.cfg, self.store), first)

        original = self.store.enqueue("queued", self.spec())
        self.service.mark("queued", due=time.time() + 300, prepare_retries=2)
        force = parser().parse_args(["force-start", "queued"])
        promoted = operate(force, self.cfg, self.store)
        self.assertTrue(promoted["force_start"])
        self.assertLessEqual(promoted["due"], time.time())
        self.assertEqual((promoted["code"], promoted["fingerprint"], promoted["prepare_retries"]),
                         (original["code"], original["fingerprint"], 2))
        node = self.inventory["dispatcher"]
        claimed = self.service.claim(promoted, "dispatcher", node["gpus"], node)
        self.assertIsNone(self.service.claim(promoted, "dispatcher", node["gpus"], node))
        self.assertEqual(operate(force, self.cfg, self.store), claimed)
        self.service.mark("queued", state="SUCCEEDED")
        with self.assertRaises(Fault) as rejected:
            operate(force, self.cfg, self.store)
        self.assertEqual(rejected.exception.info["code"], "JOB_NOT_QUEUED")

    def test_force_uses_a_busy_node_before_normal_waiters(self):
        '''强制任务可跳过大任务预留和同机准备限制，普通任务继续等待。'''
        self.cfg["large_job_wait_seconds"] = 1
        self.claim("preparing-on-node", host="dispatcher", cpus=16)
        large = self.store.enqueue("large", self.spec(host="dispatcher", gpus=2))
        self.service.mark(large["id"], submitted_at=time.time() - 10)
        self.store.enqueue("ordinary", self.spec(host="dispatcher"))
        self.store.enqueue("forced", self.spec(host="dispatcher", min_free_mib=12000), force_start=True)
        self.inventory["dispatcher"].update(memory_mib=1024, gpus=[dict(card("GPU-a"), busy=True, free_mib=100),
                                                               dict(card("GPU-c"), busy=True, free_mib=200)])
        with patch.object(self.transport, "inventory", return_value=self.inventory, create=True), \
                patch.object(self.service, "observe"), patch.object(self.service, "prepare") as prepare:
            with ThreadPoolExecutor(max_workers=2) as pool:
                self.service.tick(pool)
            prepare.assert_called_once()
            chosen = prepare.call_args.args[0]
        self.assertEqual((chosen["id"], chosen["attempts"][-1]["gpus"]), ("forced", ["GPU-c"]))
        self.assertEqual(self.store.get("large")["state"], "QUEUED")
        self.assertEqual(self.store.get("ordinary")["state"], "QUEUED")

    def test_force_keeps_node_and_driver_constraints(self):
        '''选卡仍尊重固定主机、维护开关和驱动。'''
        job = self.store.enqueue("force-fixed", self.spec(host="remote", min_driver_major=600), force_start=True)
        self.inventory["dispatcher"]["gpus"][0]["driver_major"] = 610
        self.assertIsNone(select(job, self.inventory, [], self.cfg, force=True))
        self.inventory["remote"]["gpus"][0]["driver_major"] = 600
        self.assertEqual(select(job, self.inventory, [], self.cfg, force=True)[0], "remote")
        self.cfg["hosts"]["remote"]["maintenance"] = True
        self.assertIsNone(select(job, self.inventory, [], self.cfg, force=True))

    def test_worker_forced_start_uses_busy_gpu_and_keeps_limits(self):
        '''强制允许实际 Docker 启动，UUID 与驱动检查和资源限制依旧生效。'''
        work = self.root / "attempt"
        work.mkdir()
        (self.root / "locks").mkdir()
        spec = self.spec(cpus=4, min_free_mib=20000, min_driver_major=570)
        payload = dict(attempt="a" * 32, spec=spec, assets={}, gpus=["GPU-a"], image_id="sha256:resolved",
                       command=spec["command"])
        busy = dict(card("GPU-a"), busy=True, free_mib=20)
        with patch("spool.worker.inspect", return_value=None), patch("spool.worker.os.cpu_count", return_value=16), \
                patch("spool.worker.available_memory", return_value=64000) as memory, \
                patch("spool.worker.gpu_inventory", return_value=[busy]) as inventory, patch("spool.worker.run") as docker:
            with self.assertRaises(Fault) as blocked:
                launch(payload, self.root, work)
            self.assertEqual(blocked.exception.info["code"], "GPU_BUSY")
            memory.return_value = 1024
            payload["force_start"] = True
            for cards, code in (([], "GPU_BUSY"), ([dict(busy, driver_major=560)], "DRIVER_INCOMPATIBLE")):
                inventory.return_value = cards
                with self.assertRaises(Fault) as invalid:
                    launch(payload, self.root, work)
                self.assertEqual(invalid.exception.info["code"], code)
            docker.assert_not_called()
            inventory.return_value = [busy]
            launch(payload, self.root, work)
            docker.assert_called_once()
        args = docker.call_args.args[0]
        self.assertEqual(args[:2], ["docker", "run"])
        self.assertEqual(args[args.index("--cpus") + 1], "4")
        self.assertEqual(args[args.index("--memory") + 1], "8192m")
        self.assertEqual(args[args.index("--gpus") + 1], '"device=GPU-a"')

    def test_response_lost_and_real_training_error(self):
        '''启动超时不重投，后续可得到退出码与原始 traceback。'''
        job = self.claim("lost-ack")
        self.service.prepare(job)
        unknown = self.store.get(job["id"])
        self.assertEqual(unknown["state"], "UNKNOWN")
        self.assertIn("Connection lost", unknown["error"]["detail"])
        self.service.observe(unknown)
        self.assertEqual(self.store.get(job["id"])["state"], "RUNNING")
        self.transport.container.update(status="exited", exit_code=17, oom=False)
        self.service.observe(self.store.get(job["id"]))
        failed = self.store.get(job["id"])
        self.assertEqual(self.transport.start_calls, 1)
        self.assertEqual(failed["error"]["code"], "TRAINING_FAILED")
        self.assertEqual(failed["error"]["exit_code"], 17)
        self.assertIn("RuntimeError: real training error", failed["error"]["detail"])
        self.assertTrue(Path(failed["attempts"][0]["log_path"]).is_file())

    def test_busy_and_network_retries(self):
        '''资源竞争不计次数，临时传输失败有上限。'''
        job = self.claim("retry")
        self.service.prelaunch_failure(job, Fault("launch", "GPU_BUSY", "忙", transient=True).info)
        self.assertEqual(self.store.get("retry")["prepare_retries"], 0)
        for _ in range(4):
            current = self.store.get("retry")
            launched = self.service.claim(current, "dispatcher", self.inventory["dispatcher"]["gpus"], self.inventory["dispatcher"])
            self.service.prelaunch_failure(launched, Fault("transfer", "RSYNC_FAILED", "断线", transient=True).info)
        current = self.store.get("retry")
        self.assertEqual((current["state"], current["prepare_retries"]), ("FAILED", 3))

    def test_swanlab_termination_reasons(self):
        '''人工取消、异常退出和未确认退出使用各自的 SwanLab 状态。'''
        launched = dict(container_id="id", exit_code=137, oom=False, stop_requested=True)
        job = dict(spec=self.spec(swanlab=True), attempts=[launched], state="CANCELED", cancel_requested=True)
        self.assertEqual(termination_state(job), "ABORTED")
        self.assertEqual(termination_state(dict(job, attempts=[dict(launched, oom=True)])), "CRASHED")
        self.assertEqual(termination_state(dict(job, attempts=[dict(launched, exit_code=1, stop_requested=False)])), "CRASHED")
        self.assertEqual(termination_state(dict(job, state="FAILED", cancel_requested=False)), "CRASHED")
        self.assertEqual(termination_state(dict(job, attempts=[dict(launched, exit_code=0, stop_requested=False)])), None)
        for state in ("RUNNING", "UNKNOWN", "QUEUED", "SUCCEEDED"):
            self.assertIsNone(termination_state(dict(job, state=state)))
        self.assertIsNone(termination_state(dict(job, attempts=[dict(launched, exit_code=None)])))
        self.assertEqual(termination_state(dict(job, attempts=[launched, {"work": "/staging"}])), "ABORTED")

    def test_swanlab_sync_validates_run_and_reauthenticates(self):
        '''针对确切实验补报中断，并用新的云端会话确认。'''
        job = dict(id="trial", spec=self.spec(swanlab=True, swanlab_project="team/research"))
        running = dict(cuid="cloudid", slug="trial", user={"username": "owner"}, state="RUNNING")
        path = "/project/team/research/runs"
        first, second = {"Cookie": "sid=first"}, {"Cookie": "sid=second"}
        with patch("spool.swanlab.login", side_effect=[("owner", first), ("owner", second)]), \
                patch("spool.swanlab.request", side_effect=[running, {}, dict(running, state="CRASHED")]) as request:
            self.assertEqual(sync_run(job, "CRASHED", self.root / ".netrc", ""), "CRASHED")
            read, write, read_back = request.call_args_list
            self.assertEqual((read.args[0], read_back.args[0]), (path + "/trial", path + "/trial"))
            self.assertEqual(write.args[0], path + "/cloudid/state")
            self.assertEqual(write.kwargs["body"]["state"], "CRASHED")
            self.assertEqual(read_back.kwargs["headers"], second)
        with patch("spool.swanlab.login", return_value=("owner", first)), \
                patch("spool.swanlab.request", return_value=dict(running, state="FINISHED")) as request:
            self.assertEqual(sync_run(job, "CRASHED", self.root / ".netrc", ""), "FINISHED")
            request.assert_called_once()

    def test_swanlab_sync_failure_keeps_logs_and_results(self):
        '''云端超时和异常训练不影响 调度主机 上实际报错及回传。'''
        job = self.claim("sw-error", swanlab=True)
        self.service.mark(job["id"], state="RUNNING", attempt={"work": "/work/attempt"})
        self.service.apply_status(self.store.get(job["id"]), dict(status="exited", boot_id="boot-a", container_id="container",
                                                                  exit_code=1, oom=False))
        with patch("spool.service.sync_run", side_effect=Fault("swanlab", "SW_TIMEOUT", "网络超时", transient=True)) as request:
            with ThreadPoolExecutor(max_workers=2) as pool:
                self.service.tick(pool)
            request.assert_called_once()
            self.assertEqual(request.call_args.args[1], "CRASHED")
        done = self.store.get(job["id"])
        self.assertEqual((done["state"], done["result_state"], done["swanlab_sync_state"]), ("FAILED", "SYNCED", "PENDING"))
        self.assertEqual(done["swanlab_error"]["code"], "SW_TIMEOUT")
        self.assertIn("real training error", Path(done["attempts"][-1]["log_path"]).read_text())

    def test_resume_only_once(self):
        '''已有校验过的断点时，符合条件的中断仅自动恢复一次。'''
        job = self.claim("resume", checkpoint={"path": "latest.pt", "resume_command": ["python", "resume.py"]})
        self.service.mark(job["id"], checkpoint_digest="verified")
        reboot = Fault("training", "NODE_REBOOTED", "已核实节点重启").info
        self.service.recover_training(job, reboot)
        queued = self.store.get(job["id"])
        self.assertEqual((queued["state"], queued["resume_retries"], queued["resume_models"]), ("QUEUED", 1, ["4090"]))
        second = self.service.claim(queued, "dispatcher", self.inventory["dispatcher"]["gpus"], self.inventory["dispatcher"])
        self.service.recover_training(second, reboot)
        self.assertEqual(self.store.get(job["id"])["state"], "FAILED")

    def test_worker_fences_only_proven_absence(self):
        '''节点仅在旧请求不可能迟到时释放容器预留。'''
        work = self.root / "work"
        work.mkdir()
        payload = {"attempt": "a" * 32}
        with patch("spool.worker.inspect", return_value=None), patch("spool.worker.boot_id", return_value="boot-a"):
            private_write(work / "launch-requested", "boot-a")
            self.assertEqual(status(payload, work)["status"], "uncertain")
            self.assertFalse((work / "fenced").exists())
            (work / "launch-requested").unlink()
            self.assertEqual(status(payload, work)["status"], "absent")
            self.assertTrue((work / "fenced").is_file())

    def test_worker_records_stop_only_for_running_container(self):
        '''保留 Docker 停止请求的证据，已自然退出则无需发送停止。'''
        payload = {"attempt": "a" * 32}
        self.service.stopping.clear()
        exited = dict(status="exited", exit_code=1)
        with patch("spool.worker.inspect", return_value=exited), patch("spool.worker.run") as docker:
            self.assertFalse(status(payload, self.root, cancel=True)["stop_requested"])
            docker.assert_not_called()
        with patch("spool.worker.inspect", side_effect=[{"status": "running"}, exited, exited]), patch("spool.worker.run") as docker:
            self.assertTrue(status(payload, self.root, cancel=True)["stop_requested"])
            self.assertTrue(status(payload, self.root)["stop_requested"])
            docker.assert_called_once()

    def test_swanlab_scoped_to_its_cloud(self):
        '''只传 SwanLab 条目，其他服务使用的凭据不进入任务。'''
        path = self.cfg["swanlab_netrc"]
        private_write(path, "machine api.swanlab.cn login test password lab-value\nmachine other.invalid login test password other-value\n")
        output = swanlab_file(path)
        self.assertIn("lab-value", output)
        self.assertNotIn("other-value", output)
        os.chmod(path, 0o644)
        with self.assertRaises(Fault):
            swanlab_file(path)

    def test_image_user_scope_and_default(self):
        '''显式选择镜像账号，既有任务规格保持相同指纹。'''
        self.assertEqual(self.spec(use_image_user=False), self.spec())
        self.assertIs(self.spec(use_image_user=True)["use_image_user"], True)
        with self.assertRaises(Fault):
            self.spec(use_image_user=True, swanlab=True)

    def test_image_outputs_reused_after_collection_retry(self):
        '''已导出的私有产物仅复制一次，断网重试从同一目录取回。'''
        work = self.root / "work"
        output = work / "outputs"
        output.mkdir(parents=True)

        def copy_outputs(args, *_, **__):
            '''模拟 Docker 把文件复制到本地调用账号的目录。'''
            staged = Path(args[-1])
            staged.mkdir()
            private_write(staged / "checkpoint.pt", "completed")

        with patch("spool.worker.inspect", return_value={"status": "exited"}), \
             patch("spool.worker.run", side_effect=copy_outputs) as copy:
            export_outputs("a" * 32, work)
            export_outputs("a" * 32, work)
        self.assertEqual((output / "checkpoint.pt").read_text(), "completed")
        self.assertEqual(output.stat().st_mode & 0o777, 0o700)
        self.assertEqual(copy.call_count, 1)

    def test_results_export_keeps_archive_and_existing_destination(self):
        '''
        从已有归档导出整个目录，并验证重复导出和归档内路径的错误返回。
        '''
        job = self.claim("export")
        attempt = job["attempts"][-1]["id"]
        archive = self.store.state / "results" / "export" / attempt
        (archive / "logs").mkdir(parents=True)
        (archive / "logs/model_0.pt").write_bytes(b"trained model")
        self.service.mark("export", state="SUCCEEDED", result_state="SYNCED", result_path=str(archive))

        output = self.root / "project runs" / "training"
        args = parser().parse_args(["results", "export", "--output", str(output)])
        exported = operate(args, self.cfg, self.store)
        self.assertEqual(exported["attempt_id"], attempt)
        self.assertEqual(exported["result_path"], str(archive))
        self.assertEqual(exported["output_path"], str(output))
        self.assertEqual((output / "logs/model_0.pt").read_bytes(), b"trained model")
        with self.assertRaises(Fault) as conflict:
            operate(args, self.cfg, self.store)
        self.assertEqual(conflict.exception.info["code"], "RESULT_EXPORT_EXISTS")

        inside_archive = parser().parse_args(["results", "export", "--output", str(archive / "export")])
        with self.assertRaises(Fault) as overlap:
            operate(inside_archive, self.cfg, self.store)
        self.assertEqual(overlap.exception.info["code"], "RESULT_EXPORT_OVERLAP")
        self.assertEqual((archive / "logs/model_0.pt").read_bytes(), b"trained model")

    def test_results_export_waits_for_sync_and_cleans_incomplete_copy(self):
        '''
        回传未完成时不创建目标。复制失败时保留原始错误并清理暂存内容。
        '''
        self.claim("pending")
        output = self.root / "export"
        args = parser().parse_args(["results", "pending", "--output", str(output)])
        with self.assertRaises(Fault) as unsynced:
            operate(args, self.cfg, self.store)
        self.assertEqual(unsynced.exception.info["code"], "RESULTS_NOT_SYNCED")

        archive = self.root / "archive"
        archive.mkdir()
        self.service.mark("pending", state="SUCCEEDED", result_state="SYNCED", result_path=str(archive))
        with patch("spool.cli.shutil.copytree", side_effect=OSError("disk full")):
            with self.assertRaises(Fault) as copy_failed:
                operate(args, self.cfg, self.store)
        self.assertEqual(copy_failed.exception.info["code"], "RESULT_EXPORT_FAILED")
        self.assertIn("disk full", copy_failed.exception.info["detail"])
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".spool-export-*")), [])

    def test_desktop_baseline_does_not_ignore_training_processes(self):
        '''1 GiB 内按驱动类型区分图形混合进程，保留计算和未知类型占用。'''
        processes = SimpleNamespace(stdout="GPU-a, 3756\n")
        for memory, kind, busy in ((871, "C+G", False), (871, "C", True), (871, "-", True), (1025, "C+G", True)):
            card = SimpleNamespace(stdout=f"0, GPU-a, RTX 4090, 24564, {memory}, 4, 580.178.04\n")
            monitor = SimpleNamespace(stdout=f"0 3756 {kind} 0 0 desktop\n")
            with patch("spool.worker.run", side_effect=[card, processes, monitor]):
                self.assertIs(gpu_inventory()[0]["busy"], busy)


if __name__ == "__main__":
    unittest.main()
