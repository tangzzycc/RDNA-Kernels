"""Build and load operator shared libraries."""

import ctypes
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path

from .sdk import REPO_ROOT, sdk_root


def kernel_source(operator: str) -> Path:
    """Locate an operator by its kernel entry point within its family."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*/[a-z][a-z0-9_]*", operator):
        raise ValueError("Expected family/operator, such as elementwise/add")
    family, name = operator.split("/")
    entry = re.compile(rf"^\s*__global__\s+void\s+{name}_kernel\s*\(", re.MULTILINE)
    sources = [
        path for path in sorted((REPO_ROOT / "kernels" / family).glob("*.hip"))
        if path.name != "profiler.hip" and entry.search(path.read_text())
    ]
    if not sources:
        raise FileNotFoundError(f"No HIP kernel entry point found for {operator}")
    if len(sources) != 1:
        raise ValueError(f"Multiple HIP source files define {operator}: {sources}")
    return sources[0]


def load_kernel(operator: str) -> ctypes.CDLL:
    """Build each shared source once per process, even for different operators."""
    source = kernel_source(operator)
    arch = os.environ.get("GPU_ARCH", "gfx1151")
    if not re.fullmatch(r"gfx[0-9a-f]+", arch):
        raise ValueError(f"Invalid GPU_ARCH: {arch}")
    return _load_source(source, arch)


@lru_cache(maxsize=None)
def _load_source(source: Path, arch: str) -> ctypes.CDLL:
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
