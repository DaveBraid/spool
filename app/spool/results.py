"""训练结果校验后按原生相对路径归位。"""

import os
from pathlib import Path
import shutil

from .worker import Fault, tree


def staging(root, attempt):
    '''在结果根目录下建立本次尝试专用的同盘暂存目录。'''
    if root != root.resolve() or not root.is_dir():
        raise Fault("results", "RESULT_ROOT_UNAVAILABLE", "原生结果目录已变更或不可用", str(root))
    stage = root / (".spool-return-" + attempt)
    if stage.is_symlink():
        raise Fault("results", "RESULT_ROOT_UNAVAILABLE", "暂存目录不能是符号链接", str(stage))
    stage.mkdir(mode=0o700, exist_ok=True)
    info = stage.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise Fault("results", "RESULT_ROOT_UNAVAILABLE", "暂存目录须属于当前账号且权限为 0700", str(stage))
    return stage


def layout(source):
    '''列出训练产物的相对路径与类型，保留隐藏文件和空目录。'''
    tree(source, checksum=False)
    return {str(path.relative_to(source)): path.is_dir() for path in sorted(source.rglob("*"))}


def conflicts(source, root, entries):
    '''检查所有目标路径，允许目录共用和同一次发布产生的硬链接。'''
    if root != root.resolve() or not root.is_dir():
        raise Fault("results", "RESULT_ROOT_UNAVAILABLE", "原生结果目录已变更或不可用", str(root))
    for name, directory in entries.items():
        target = root / name
        if target.is_symlink() or (target.exists() and not (
                target.is_dir() if directory else target.is_file() and target.samefile(source / name))):
            raise Fault("results", "RESULT_PATH_CONFLICT", "原生路径已有其他文件", str(target))


def publish_files(source, root, entries):
    '''以不可覆盖的硬链接发布暂存文件。重试时识别已经发布的同一文件。'''
    for name, directory in entries.items():
        target = root / name
        try:
            if target.parent != target.parent.resolve():
                raise Fault("results", "RESULT_PATH_CONFLICT", "产物的目标路径包含符号链接", str(target.parent))
            if directory:
                target.mkdir(mode=0o700, exist_ok=True)
            elif not target.exists():
                os.link(source / name, target)
        except FileExistsError:
            conflicts(source, root, {name: directory})
    conflicts(source, root, entries)  # 发布完成前再核对，防止并发创建同名文件。


def copy_files(source, destination, entries):
    '''只导出这次任务的成果，按记录的相对路径复制文件。'''
    for name, directory in entries.items():
        path = source / name
        if path != path.resolve() or (not path.is_dir() if directory else not path.is_file()):
            raise Fault("results", "RESULT_ARCHIVE_MISSING", "原生结果文件已移动或不可读", str(path))
        if directory:
            (destination / name).mkdir(mode=0o700, parents=True, exist_ok=True)
        else:
            shutil.copy2(path, destination / name)


def move_archive(store, entry, root):
    '''将明确选中的旧归档移到原生位置，更新索引后回收旧目录。'''
    key = entry["id"]
    if not entry["attempts"]:
        raise Fault("results", "RESULT_MOVE_UNAVAILABLE", "任务没有可移动的旧式 Spool 归档")
    attempt = entry["attempts"][-1]["id"]
    archive = store.state / "results" / key / attempt
    if entry.get("result_move_source") == str(archive) and entry["result_path"] == str(root):
        if archive.exists():
            shutil.rmtree(archive)
    else:
        if entry["result_state"] != "SYNCED" or entry.get("result_path") != str(archive) or not archive.is_dir():
            raise Fault("results", "RESULT_MOVE_UNAVAILABLE", "任务没有可移动的旧式 Spool 归档")
        if root.stat().st_dev != archive.stat().st_dev:
            raise Fault("results", "RESULT_MOVE_CROSS_DEVICE", "移动目标和旧归档须在同一文件系统")
        entries = layout(archive)
        conflicts(archive, root, entries)
        publish_files(archive, root, entries)
        store.change(key, lambda job: job.update(result_path=str(root), result_entries=entries, result_move_source=str(archive)))
        shutil.rmtree(archive)
    try:
        archive.parent.rmdir()
    except OSError:
        pass  # 同一个任务可能还有其他旧归档。
    return {"job_id": key, "attempt_id": attempt, "result_path": str(root), "output_path": str(root)}
