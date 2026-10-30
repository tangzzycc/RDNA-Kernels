"""Build and load operator shared libraries."""

import ctypes
import os
import re
import subprocess
from functools import lru_cache

from .sdk import REPO_ROOT, sdk_root


@lru_cache(maxsize=None)
def load_kernel(operator: str) -> ctypes.CDLL:
    """Build an operator once per process; family profiler sources are separate."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*/[a-z][a-z0-9_]*", operator):
        raise ValueError("Expected family/operator, such as elementwise/add")
    source = REPO_ROOT / "kernels" / f"{operator}.hip"
    if not source.is_file():
        raise FileNotFoundError(source)
    arch = os.environ.get("GPU_ARCH", "gfx1151")
    if not re.fullmatch(r"gfx[0-9a-f]+", arch):
        raise ValueError(f"Invalid GPU_ARCH: {arch}")
    library = REPO_ROOT / "build/hip" / f"{operator}.so"
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
