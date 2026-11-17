"""在容器确认退出后补报 SwanLab 实验终态。"""

from datetime import datetime, timezone
from http.client import HTTPSConnection, HTTPException
import json
import re
from urllib.parse import quote, unquote, urlencode, urlsplit

from .worker import Fault, netrc_hosts


API_HOST = "api.swanlab.cn"
STOP_EXIT_CODES = {0, 130, 137, 143}
CLOSED = {"FINISHED", "ABORTED", "CRASHED"}


def termination_state(job):
    '''按最新尝试的停止证据和退出原因选择云端终态。
    旧记录的退出码只会在 Docker 确认退出后写入，仍可用于补报。
    '''
    if not job["spec"]["swanlab"] or job["state"] not in {"CANCELED", "FAILED"} or not job["attempts"]:
        return None
    # 续训可能在再次启动前失败。仍需为最后一个确实运行过的容器补报。
    attempt = next((a for a in reversed(job["attempts"]) if a.get("container_id")), {})
    code = attempt.get("exit_code")
    if code is None and not attempt.get("container_stopped"):
        return None
    if attempt.get("oom"):
        return "CRASHED"
    # 旧作业只在有明确人工取消记录和信号退出码时判为人工停止。
    requested = attempt.get("stop_requested", code in (STOP_EXIT_CODES - {0}))
    if job.get("cancel_requested") and requested and (code is None or code in STOP_EXIT_CODES):
        return "ABORTED"
    return "CRASHED" if code != 0 else None


def request(path, *, method="GET", headers=None, body=None, missing=False):
    '''调用 SwanLab 云端接口并返回 JSON。连接固定到官方主机，错误不包含凭据或响应正文。'''
    conn = HTTPSConnection(API_HOST, timeout=8)
    headers = dict(headers or {})
    if body is not None:
        headers["Content-Type"] = "application/json"
    try:
        conn.request(method, "/api" + path, body=json.dumps(body) if body is not None else None, headers=headers)
        response = conn.getresponse()
        code = response.status
        if missing and code == 404:
            return None
        if not 200 <= code < 300:
            raise Fault("swanlab", "SWANLAB_HTTP_FAILED", "云端拒绝状态查询或更新", f"{method} HTTP {code}",
                        transient=code in {408, 429} or code >= 500)
        data = response.read(4 * 1024 * 1024 + 1)
        if len(data) > 4 * 1024 * 1024:
            raise Fault("swanlab", "SWANLAB_RESPONSE_TOO_LARGE", "云端响应超过读取上限")
        decoded = json.loads(data) if data else {}
        if not isinstance(decoded, dict):
            raise ValueError("expected a JSON object")
        return decoded
    except (OSError, HTTPException) as exc:
        raise Fault("swanlab", "SWANLAB_CONNECTION_FAILED", "连接 SwanLab 失败", type(exc).__name__, transient=True) from None
    except ValueError as exc:
        raise Fault("swanlab", "SWANLAB_INVALID_RESPONSE", "SwanLab 返回格式异常", type(exc).__name__, transient=True) from None
    finally:
        conn.close()


def login(netrc_path):
    '''复用调度主机的云端凭据，返回认证账号和一次会话的请求头。'''
    secrets = [entry[2] for machine, entry in netrc_hosts(netrc_path).items()
               if urlsplit(machine if "://" in machine else "//" + machine).hostname == API_HOST and entry[2]]
    if len(set(secrets)) != 1:
        raise Fault("swanlab", "SWANLAB_CREDENTIAL_INVALID", "需要唯一的 SwanLab 云端凭据来补报状态")
    data = request("/login/api_key", method="POST", headers={"Authorization": secrets[0]})
    owner, sid = (data.get("userInfo") or {}).get("username"), data.get("sid")
    if not isinstance(owner, str) or not owner or not isinstance(sid, str) or not re.fullmatch(r"[^;\s]+", sid):
        raise Fault("swanlab", "SWANLAB_AUTH_INVALID", "SwanLab 登录未返回有效会话")
    return owner, {"Cookie": "sid=" + sid}


def project_paths(job, logs, owner, headers):
    '''优先采用任务配置和日志中的准确项目，其次查询账号已有项目。'''
    spec = job["spec"]
    if spec.get("swanlab_project"):
        return [tuple(spec["swanlab_project"].split("/", 1))]
    link = rf"https://(?:www\.)?swanlab\.cn/@([^/\s]+)/([^/\s]+)/runs/{re.escape(job['id'])}(?=[/\s?\x1b]|$)"
    linked = sorted({(unquote(workspace), unquote(project)) for workspace, project in re.findall(link, logs)})
    if linked:
        return linked
    command = spec["command"]
    for flag, value in zip(command, command[1:]):
        if flag == "--swanlab_project_name" and value and not value.startswith("--"):
            return [(owner, value)]  # 兼容旧训练入口的项目参数
    for arg in command:
        if arg.startswith("--swanlab_project_name=") and arg.split("=", 1)[1]:
            return [(owner, arg.split("=", 1)[1])]
    projects = []
    page = 1
    while True:
        data = request("/project/" + quote(owner, safe="") + "?" + urlencode({"page": page, "size": 100}), headers=headers)
        projects.extend((owner, row["name"]) for row in data["list"])
        if page >= data["pages"]:
            return projects
        page += 1


def sync_run(job, destination, netrc_path, logs):
    '''用唯一的 Spool 任务 ID 核对现有实验，补报一次并重新登录验证云端状态。'''
    owner, headers = login(netrc_path)
    found = []
    for workspace, project in project_paths(job, logs, owner, headers):
        prefix = "/project/" + quote(workspace, safe="") + "/" + quote(project, safe="") + "/runs/"
        path = prefix + quote(job["id"], safe="")
        run = request(path, headers=headers, missing=True)
        if run is not None:
            found.append((prefix, path, run))
    if not found:
        return None  # 训练可能在创建 SwanLab 实验前就已经失败
    if len(found) > 1:
        raise Fault("swanlab", "SWANLAB_RUN_AMBIGUOUS", "多个 SwanLab 项目包含同一任务 ID，请指定任务的 swanlab_project")
    prefix, path, run = found[0]
    cuid = run.get("cuid", "")
    if run.get("slug") != job["id"] or (run.get("user") or {}).get("username") != owner or not re.fullmatch(r"[a-zA-Z0-9]+", cuid):
        raise Fault("swanlab", "SWANLAB_RUN_MISMATCH", "云端实验 ID 或所有者与 Spool 任务不符")
    if run.get("state") in CLOSED:
        return run["state"]  # 保留 SDK 已经上传的终态
    if run.get("state") != "RUNNING":
        raise Fault("swanlab", "SWANLAB_UNEXPECTED_STATE", "云端实验状态不支持补报")
    # SwanLab 用上报时刻保留此前已入库的指标，实际退出时间仍由 Spool 记录。
    finished = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    request(prefix + cuid + "/state", method="PUT", headers=headers,
            body={"state": destination, "finishedAt": finished, "from": "sdk"})
    refreshed_owner, headers = login(netrc_path)  # 状态更新后旧会话可能失效
    if refreshed_owner != owner:
        raise Fault("swanlab", "SWANLAB_ACCOUNT_CHANGED", "状态更新前后的 SwanLab 登录账号不一致")
    updated = request(path, headers=headers)
    if updated.get("cuid") != cuid or updated.get("state") != destination:
        raise Fault("swanlab", "SWANLAB_STATE_UNCONFIRMED", "未读回预期的 SwanLab 终态", transient=True)
    return destination
