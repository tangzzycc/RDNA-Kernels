"""定位 HIP 算子源文件, 编译并加载共享库."""

import ctypes
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path

from .sdk import REPO_ROOT, sdk_root


def kernel_source(operator: str) -> Path:
    """按 family/operator 查找主机入口, 优先使用 C ABI launch 的定义.

    支持 launch.hip 包含同目录的核函数文件, 同一入口可分发多个版本.
    没有 launch 定义时查找 name_kernel, 跳过 profiler.hip 与仅有声明的入口.
    未找到入口或存在多个定义时抛出异常.
    """
    if not re.fullmatch(r"[a-z][a-z0-9_]*/[a-z][a-z0-9_]*", operator):
        raise ValueError("Expected family/operator, such as elementwise/add")
    family, name = operator.split("/")
    launch_entry = re.compile(
        rf'^\s*extern\s+"C"\s+hipError_t\s+launch_{name}\s*\([^)]*\)\s*\{{',
        re.MULTILINE,
    )
    kernel_entry = re.compile(
        rf'^\s*__global__\s+void\s+{name}_kernel\s*\(', re.MULTILINE,
    )
    candidates = [
        path for path in sorted((REPO_ROOT / "kernels" / family).glob("*.hip"))
        if path.name != "profiler.hip"
    ]
    sources = [path for path in candidates if launch_entry.search(path.read_text())]
    if not sources:
        sources = [path for path in candidates if kernel_entry.search(path.read_text())]
    if not sources:
        raise FileNotFoundError(f"No HIP kernel entry point found for {operator}")
    if len(sources) != 1:
        raise ValueError(f"Multiple HIP source files define {operator}: {sources}")
    return sources[0]


def load_kernel(operator: str) -> ctypes.CDLL:
    """编译并加载算子所属的源文件, 返回 ctypes.CDLL 共享库对象.

    GPU_ARCH 指定目标架构, 默认 gfx1151; 同一源文件和架构在进程内复用缓存.
    """
    source = kernel_source(operator)
    arch = os.environ.get("GPU_ARCH", "gfx1151")
    if not re.fullmatch(r"gfx[0-9a-f]+", arch):
        raise ValueError(f"Invalid GPU_ARCH: {arch}")
    return _load_source(source, arch)


@lru_cache(maxsize=None)
def _load_source(source: Path, arch: str) -> ctypes.CDLL:
    """调用项目 hipcc 编译指定源文件和架构, 加载并缓存共享库.

    编译失败时抛出包含编译器输出的 RuntimeError.
    """
    relative = source.relative_to(REPO_ROOT / "kernels").with_suffix(".so")
    library = REPO_ROOT / "build/hip" / relative
    library.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(REPO_ROOT / ".venv/bin/hipcc"),
            "-O3", "-std=c++17", f"--offload-arch={arch}",
            "-shared", "-fPIC",
            str(source), "-o", str(library),
        ],
        env={**os.environ, "ROCM_PATH": str(sdk_root())},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"Failed to compile {source}:\n{result.stdout}")
    return ctypes.CDLL(str(library))
