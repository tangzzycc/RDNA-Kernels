"""为 clangd 生成使用项目 ROCm SDK 的 HIP 编译数据库。"""

import json
import os
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    arch = os.environ.get("GPU_ARCH", "gfx1151")
    if not re.fullmatch(r"gfx[0-9a-f]+", arch):
        raise ValueError(f"Invalid GPU_ARCH: {arch}")

    result = subprocess.run(
        [str(REPO_ROOT / ".venv/bin/rocm-sdk"), "path", "--root"],
        check=True,
        capture_output=True,
        text=True,
    )
    sdk_root = Path(result.stdout.strip())
    # 使用实际 Clang 驱动，方便 clangd 查询系统头文件；hipcc 是 Python 包装器。
    compiler = sdk_root / "lib/llvm/bin/clang++"
    if not compiler.is_file():
        raise FileNotFoundError(compiler)

    result = subprocess.run(
        [
            "git", "ls-files", "--cached", "--others", "--exclude-standard",
            "-z", "--", "kernels", "common/hip",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    # 头文件也指定 HIP 上下文，避免单独打开 common.h 时被当成普通 C++。
    suffixes = {
        ".hip", ".cu", ".cpp", ".cc", ".cxx",
        ".h", ".hpp", ".hh", ".hxx", ".cuh",
    }
    sources = sorted({
        REPO_ROOT / name
        for name in result.stdout.split("\0")
        if name and Path(name).suffix in suffixes
        and (REPO_ROOT / name).is_file()
    })

    entries = []
    for source in sources:
        relative = source.relative_to(REPO_ROOT)
        output = REPO_ROOT / "build/hip/objects" / relative.with_suffix(".o")
        entries.append({
            "directory": str(REPO_ROOT),
            "file": str(source),
            "arguments": [
                str(compiler), "-x", "hip", "-std=c++17", "-O3",
                f"--rocm-path={sdk_root}", f"--offload-arch={arch}",
                "-I", str(REPO_ROOT),
                "-c", str(source), "-o", str(output),
            ],
        })

    database = REPO_ROOT / "build/hip/compile_commands.json"
    database.parent.mkdir(parents=True, exist_ok=True)
    database.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"Updated {database.relative_to(REPO_ROOT)}: {len(entries)} sources")


if __name__ == "__main__":
    main()
