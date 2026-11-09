"""通过项目 .venv 查询 ROCm SDK 路径, 无需先激活 shell 环境."""

import subprocess
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def sdk_root() -> Path:
    """调用项目 rocm-sdk 获取 SDK 根目录, 返回路径并在进程内缓存结果."""
    result = subprocess.run(
        [str(REPO_ROOT / ".venv/bin/rocm-sdk"), "path", "--root"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())
