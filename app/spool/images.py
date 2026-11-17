"""私有仓库与节点镜像缓存的保留规则。"""

import base64
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection, HTTPException
import json
import os
from pathlib import Path
import shutil
from urllib.parse import quote
from urllib.parse import urlsplit

from .state import IMAGE, TERMINAL
from .transport import Transport, registry_password
from .worker import Fault, run


ACCEPT = ", ".join((
    "application/vnd.oci.image.index.v1+json", "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json", "application/vnd.docker.distribution.manifest.v2+json",
))


class Registry:
    def __init__(self, cfg):
        '''连接调度主机上的私有仓库。认证只用于回环连接。'''
        self.port = cfg.get("registry_port")
        if not self.port:
            raise Fault("image", "REGISTRY_DISABLED", "未配置 Spool 私有镜像仓库")
        self.address = f"127.0.0.1:{self.port}"
        secret = registry_password(cfg, create=False)
        if not secret:
            raise Fault("image", "REGISTRY_NOT_INITIALIZED", "请先初始化 Spool 私有镜像仓库")
        auth = base64.b64encode(("spool:" + secret).encode()).decode()
        self.headers = {"Authorization": "Basic " + auth, "Accept": ACCEPT}

    def request(self, method, path):
        '''请求清单元数据，保留 HTTP 状态供 Agent 诊断。'''
        connection = HTTPConnection("127.0.0.1", self.port, timeout=12)
        try:
            connection.request(method, "/v2/" + path, headers=self.headers)
            response = connection.getresponse()
            body = response.read()
            if response.status not in {200, 202}:
                raise Fault("image", "REGISTRY_REQUEST_FAILED", "仓库请求失败", f"{method} {path} HTTP {response.status}")
            return response.getheader("Docker-Content-Digest"), json.loads(body) if body else {}, response.getheader("Link")
        except (OSError, HTTPException) as exc:
            raise Fault("image", "REGISTRY_UNAVAILABLE", "无法连接 Spool 私有仓库", str(exc), transient=True) from None
        finally:
            connection.close()

    def manifest(self, reference, method="GET"):
        '''按完整的仓库摘要访问清单。'''
        name, digest = reference.split("/", 1)[1].split("@", 1)
        return self.request(method, f"{quote(name)}/manifests/{digest}")[1]

    def accept(self, store, reference):
        '''只登记已经存在于 Spool 私有仓库的有效摘要。'''
        if not reference.startswith(self.address + "/") or not IMAGE.fullmatch(reference):
            raise Fault("image", "INVALID_MANAGED_IMAGE", "请指定 Spool 私有仓库中的完整镜像摘要")
        self.manifest(reference, "HEAD")
        store.accept_image(reference)

    def pages(self, path, key):
        '''读取完整的仓库或标签列表，分页地址必须仍在原回环仓库。'''
        visited = set()
        while path not in visited:
            visited.add(path)
            _, page, link = self.request("GET", path)
            yield from page.get(key) or ()
            if not link:
                return
            target = urlsplit(link.partition("<")[2].partition(">")[0])
            if target.netloc not in ("", self.address) or not target.path.startswith("/v2/"):
                break
            path = target.path[4:] + ("?" + target.query if target.query else "")
        raise Fault("image", "REGISTRY_LIST_INCOMPLETE", "无法读取完整的仓库清单")

    def tagged(self):
        '''返回受管仓库中所有带标签的不可变镜像引用。'''
        refs = set()
        for name in self.pages("_catalog", "repositories"):
            for tag in self.pages(quote(name) + "/tags/list", "tags"):
                digest, _, _ = self.request("HEAD", f"{quote(name)}/manifests/{quote(tag, safe='')}")
                ref = f"{self.address}/{name}@{digest}"
                if not IMAGE.fullmatch(ref):
                    raise Fault("image", "INVALID_MANIFEST", "仓库未返回有效的镜像摘要", f"{name}:{tag}")
                refs.add(ref)
        return refs


def retained_images(store):
    '''保留每个仓库近两份已验收镜像，以及尚能启动或续训的任务镜像。'''
    stable = {}
    for ref in store.accepted_images():
        group = stable.setdefault(ref.split("@")[0], [])
        if len(group) < 2:
            group.append(ref)
    jobs = store.jobs()
    protected = {ref for group in stable.values() for ref in group}
    protected.update(job["spec"]["image"] for job in jobs if job["state"] not in TERMINAL)
    return stable, protected, jobs


def cache_nodes(cfg, candidates, apply):
    '''在各节点核查精确引用。一个节点失联时其他节点仍可维护。'''
    transport = Transport(cfg)

    def inspect(host):
        '''读取节点候选或删除可重新拉取的缓存。'''
        try:
            return host, transport.call(host, "image_cache", references=candidates, apply=apply, timeout=600)
        except Fault as exc:
            return host, {"error": exc.info}

    with ThreadPoolExecutor(max_workers=min(8, len(cfg["hosts"]))) as pool:
        return dict(pool.map(inspect, cfg["hosts"]))


def manifest_removals(registry, data, candidates, protected):
    '''保留未纳入清理的清单及其子清单，包括无标签的历史摘要。'''
    base = data / "docker/registry/v2/repositories"
    if candidates and not base.is_dir():
        raise Fault("image", "REGISTRY_LAYOUT_UNAVAILABLE", "无法验证私有仓库的清单引用关系")
    graph = {}
    for link in base.glob("**/_manifests/revisions/sha256/*/link"):
        parts = link.relative_to(base).parts
        repo = "/".join(parts[:-5])
        reference = f"{registry.address}/{repo}@sha256:{parts[-2]}"
        if IMAGE.fullmatch(reference):
            content = registry.manifest(reference)
            children = [item["digest"] for item in content.get("manifests", ())]
            if content.get("subject"):
                children.append(content["subject"]["digest"])
            graph[reference] = [reference.split("@")[0] + "@" + digest for digest in children]
    if not set(candidates) <= graph.keys():
        raise Fault("image", "REGISTRY_LAYOUT_UNAVAILABLE", "候选镜像未出现在仓库清单索引中")

    def reachable(roots):
        '''遍历清单引用，结果包含所有根及可达的子清单。'''
        visited, waiting = set(), list(roots)
        while waiting:
            ref = waiting.pop()
            if ref not in visited:
                visited.add(ref)
                waiting.extend(graph.get(ref, ()))
        return visited

    removing = reachable(candidates)
    keeping = reachable((graph.keys() - removing) | protected)
    return sorted((removing & graph.keys()) - keeping, key=lambda ref: (ref not in candidates, ref))


def sweep_registry(store, metadata):
    '''暂停 Registry 写入并运行其官方回收工具，完成后恢复原实例。'''
    config = metadata["Config"]["Cmd"]
    if config not in (["/etc/distribution/config.yml"], ["/etc/docker/registry/config.yml"]):
        raise Fault("image", "REGISTRY_CONFIG_UNSUPPORTED", "请为自定义配置的仓库运行其原生回收工具")
    path = store.state / "registry-data"
    uid = f"{os.getuid()}:{os.getgid()}"
    container = run(["docker", "create", "--network", "none", "--user", uid,
                     "--mount", f"type=bind,src={path},dst=/var/lib/registry",
                     "--entrypoint", "/bin/registry", metadata["Image"], "garbage-collect", config[0]],
                    "image", "REGISTRY_GC_SETUP_FAILED").stdout.strip()
    before = shutil.disk_usage(path).free
    try:
        run(["docker", "stop", "spool-registry"], "image", "REGISTRY_STOP_FAILED", timeout=40)
        result = run(["docker", "start", "--attach", container], "image", "REGISTRY_GC_FAILED", timeout=1800)
    finally:
        # 即使命令超时也先终止本次 GC 容器，避免恢复写入时仍在回收。
        run(["docker", "rm", "-f", container], "image", "REGISTRY_GC_REMOVE_FAILED", timeout=30)
        run(["docker", "start", "spool-registry"], "image", "REGISTRY_RESTART_FAILED", timeout=30)
    return {"free_bytes_change": shutil.disk_usage(path).free - before, "detail": (result.stdout + result.stderr)[-4000:]}


def clean(cfg, store, *, apply=False, registry_cleanup=False):
    '''预览或清理私有镜像，调用方持有镜像维护锁。'''
    registry = Registry(cfg)
    stable, protected, jobs = retained_images(store)
    roots = sorted(ref for ref in registry.tagged() if ref.split("@")[0] in stable and ref not in protected)
    caches = cache_nodes(cfg, roots, apply) if roots else {}
    used = {ref for node in caches.values() for ref, info in node.items()
            if ref != "error" and info["state"] in {"IN_USE", "FAILED"}}
    unverified_hosts = {name for name, node in caches.items() if "error" in node}
    used.update(job["spec"]["image"] for job in jobs if any(a["host"] in unverified_hosts for a in job["attempts"]))
    data = {"protected": sorted(protected | used), "caches": caches, "registry_candidates": sorted(set(roots) - used)}
    if not registry_cleanup:
        return data
    candidates = set(roots) - used
    if not candidates:
        return data
    path = store.state / "registry-data"
    # 把仍保留的带标签子清单当作独立版本，保护它的内容。
    all_tags = registry.tagged()
    candidates = {ref for ref in candidates if ref in all_tags}
    data["registry_manifests"] = manifest_removals(registry, path, candidates, (all_tags - candidates) | protected | used)
    if not apply or not data["registry_manifests"]:
        return data
    if unverified_hosts:
        raise Fault("image", "CACHE_REFERENCES_UNVERIFIED", "有节点无法核对本地容器引用，暂不清理仓库", ", ".join(sorted(unverified_hosts)), transient=True)
    if any(job["state"] in {"QUEUED", "PREPARING", "STARTING"}
           and job["spec"]["image"].startswith(registry.address + "/") for job in jobs):
        raise Fault("image", "REGISTRY_IN_USE", "还有待拉取私有镜像的任务，稍后再回收仓库", transient=True)
    metadata = json.loads(run(["docker", "inspect", "spool-registry"], "image", "REGISTRY_INSPECT_FAILED").stdout)[0]
    if (metadata["Config"].get("Labels", {}).get("io.spool.registry") != str(os.getuid())
            or not any(m["Destination"] == "/var/lib/registry" and Path(m["Source"]) == path for m in metadata["Mounts"])):
        raise Fault("image", "REGISTRY_OWNERSHIP_MISMATCH", "仓库容器和数据目录与 Spool 配置不符")
    if "REGISTRY_STORAGE_DELETE_ENABLED=true" not in metadata["Config"]["Env"]:
        raise Fault("image", "REGISTRY_DELETE_DISABLED", "请先在私有仓库配置启用清单删除")
    if metadata["Config"]["Cmd"] not in (["/etc/distribution/config.yml"], ["/etc/docker/registry/config.yml"]):
        raise Fault("image", "REGISTRY_CONFIG_UNSUPPORTED", "请使用自定义仓库的原生回收工具")
    for ref in data["registry_manifests"]:
        registry.manifest(ref, "DELETE")
    data["registry_gc"] = sweep_registry(store, metadata)
    return data
