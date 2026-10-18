# -*- coding: utf-8 -*-
"""写入重试层。

背景：在部分受管环境下，结果目录的写入失败是**逐文件、随机**的——
同一批文件的写入测试中，少数路径抛 `PermissionError [Errno 13]`，而同一路径在
另一次尝试中又能正常写入（实测连写 10 次全部成功，另有少量文件随机失败）。

本模块在进程内为常见写操作加一层"失败重试"，把这种随机拦截对长跑实验的影响
降到可忽略；重试次数用尽后仍会抛出原始异常，不会吞掉真实错误。

用法：在实验脚本的**最前面** `import _robust_io`（早于 numpy/pandas/torch 等）。
"""
import builtins
import io
import logging
import os
import pathlib
import shutil
import time

logger = logging.getLogger(__name__)

_ATTEMPTS = int(os.environ.get("ROBUST_IO_ATTEMPTS", "15"))
_BASE_DELAY = 0.2
_MAX_DELAY = 3.0

STATS = {"retries": 0, "recovered": 0, "raised": 0}


def _retry(fn, *args, **kwargs):
    delay = _BASE_DELAY
    last = None
    for attempt in range(_ATTEMPTS):
        try:
            out = fn(*args, **kwargs)
            if attempt:
                STATS["recovered"] += 1
            return out
        except PermissionError as e:      # 受限环境下的写入拦截（以及 Windows 文件锁）
            last = e
            STATS["retries"] += 1
            if attempt == 0:
                logger.warning(f"robust_io: 写入被拒，重试中 -> {e}")
            time.sleep(delay)
            delay = min(delay * 1.7, _MAX_DELAY)
    STATS["raised"] += 1
    raise last


def install():
    """幂等安装：重复调用不会二次包裹。"""
    if getattr(builtins.open, "_robust_io", False):
        return

    _open = builtins.open

    def open_retry(*args, **kwargs):
        return _retry(_open, *args, **kwargs)

    open_retry._robust_io = True
    open_retry._original = _open
    builtins.open = open_retry
    io.open = open_retry

    for name in ("unlink", "mkdir", "rmdir", "rename", "replace", "touch"):
        orig = getattr(pathlib.Path, name)

        def make(orig_fn):
            def wrapper(self, *a, **kw):
                return _retry(orig_fn, self, *a, **kw)
            wrapper._robust_io = True
            return wrapper

        setattr(pathlib.Path, name, make(orig))

    for mod, names in ((os, ("remove", "unlink", "mkdir", "makedirs",
                             "rename", "replace", "rmdir")),
                       (shutil, ("move", "copy", "copy2", "copyfile"))):
        for name in names:
            if not hasattr(mod, name):
                continue
            orig = getattr(mod, name)

            def make(orig_fn):
                def wrapper(*a, **kw):
                    return _retry(orig_fn, *a, **kw)
                wrapper._robust_io = True
                return wrapper

            setattr(mod, name, make(orig))


install()