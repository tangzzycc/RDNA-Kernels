"""Resolve the project's ROCm SDK independently of shell activation."""

import subprocess
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def sdk_root() -> Path:
    result = subprocess.run(
        [str(REPO_ROOT / ".venv/bin/rocm-sdk"), "path", "--root"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())
