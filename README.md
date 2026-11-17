# Spool

Spool 在后台分发和管理容器训练。从操作电脑（System 0）通过 SSH 调用调度主机（System 1）的 `spool` CLI。调度主机管理队列、镜像、输入快照和回传，也能自己跑训练。远程节点（System 2）通过 SSH 接入，拉取固定摘要执行。提交后无需保持终端连接。

## 部署

System 1 需要 Linux、Python 3.10+、Docker、rsync、OpenSSH 和 systemd 用户服务。远程节点需要 Python 3.10+、rsync、Docker 权限及适用的 NVIDIA 驱动与容器运行时。

```bash
bash deploy/install.sh
# 编辑 ~/.config/spool/config.json，填写源码范围及节点 SSH 别名。
~/.local/bin/spool registry-init  # 可选：使用 Spool 的私有镜像仓库
systemctl --user daemon-reload
systemctl --user enable --now spool.service
~/.local/bin/spool nodes --json
```

参考 [配置样例](deploy/config.example.json)。把远程 SSH 别名加到 `hosts`，例如 `"worker-1": {"ssh": "worker-1"}`；为信任的节点准备 SSH 主机密钥。状态默认位于 `~/.local/state/spool`。需要用户服务在注销和重启后继续运行时，按系统策略开启 user linger。私有仓库绑定回环地址，通过 SSH 反向转发接入远程节点；初始化需要 `htpasswd` 或系统提供 bcrypt。

## 训练与结果

1. `spool publish <本机镜像> <仓库:标签> --json` 发布训练环境并返回不可变摘要。仅改源码时沿用已验收的摘要，提交新的代码快照。
2. 准备任务 JSON，`image` 填镜像摘要，`command` 填容器内的执行参数数组。用 `nodes --json` 与 `queue --json` 选择节点和 GPU UUID；操作者在私有配置的 `placement` 里设置 `priority`（越小越优先）、`max_spool_gpus` 与 `gpu_order`（`asc` 或 `desc`）。服务预留指定 UUID，启动前复查使用情况；指定卡不可用时返回原因。
3. `spool submit job.json --id <唯一任务ID> --json` 冻结输入并排队。明确允许共享忙卡时，可用 `submit --force` 或 `force-start <ID>`。用 `status` 看阶段和真实错误，`logs` 查 stdout/stderr。`--json` 的 `ok` 表示命令成功，训练状态看 `data.state`。
4. 容器的 `/outputs` 按相对路径回传到 `result_root`，保留训练配置、检查点和原生日志。省略 `result_root` 时使用独立任务归档。Spool 在记录 `SYNCED` 并保全终端日志后自动清理这次尝试的工作副本与停止的容器，保留任务索引和终端日志尾部。目标有同名文件时保留源文件并报告冲突。用 `collect <ID>` 重试回传，用 `cleanup <ID>` 重试清理。

SwanLab 可选。在线训练复用 System 1 的 `~/.swanlab/.netrc`，训练镜像需有相应 SDK 且能访问云端。确认容器退出后，Spool 给云端仍显示运行的实验补报终态：人工停止为 `Ctrl+C`，异常退出为「中断」。云端状态和文件回传独立处理。

## 镜像维护

验证私有镜像后运行 `spool image-keep <完整摘要> --json`。每个仓库保留最近两个已验收的摘要、非终态任务的镜像以及容器仍引用的镜像。登记新版后自动尝试回收可重新拉取的过期节点缓存。

`spool image-clean --json` 预览节点缓存，`--apply` 执行。加 `--registry` 可预览私有仓库候选，在确认节点引用和没有待拉取任务的维护窗口再加 `--apply`。仓库必须启用清单删除；回收期间暂时停止仓库，已运行的容器继续训练。镜像层由多个版本共用时，仅回收无人引用的层。

## 验证与维护

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=app python3 -m unittest discover -s tests -v
```

每次公开更新同步维护服务、CLI 和发布说明，参见 [AGENTS.md](AGENTS.md)。
