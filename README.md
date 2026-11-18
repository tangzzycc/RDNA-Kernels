# RDNA-Kernels

RDNA-Kernels helps beginners get started with implementing GPU operators on AMD RDNA hardware
using HIP and FlyDSL. Kernels from common models serve as examples for understanding
GPU programming, checking correctness, and exploring optimization step by step.

## Environment

The development environment uses a Radeon 8060S (RDNA 3.5, `gfx1151`), Python 3.12,
and ROCm SDK 10.1.0 with matching profiler tools.

From the repository root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/rocm-sdk init
.venv/bin/python scripts/configure_rocm_env.py
source .venv/bin/activate
```

The activated environment provides `hipcc`, `rocprofv3`, and `rocprof-compute`.

## Testing and Profiling

Implementations live in `hip/`, shared helpers in `common/`, and correctness tests in `tests/`.
Select an operator with `family/operator`:

```bash
python -m pytest tests/test_elementwise.py -q
./scripts/analyze_kernel.sh --run-only elementwise/add
./scripts/analyze_kernel.sh elementwise/add
```

Family READMEs contain test commands and measured results:
[norm](hip/norm/README.md), [elementwise](hip/elementwise/README.md),
and [matmul](hip/matmul/README.md).

## Formatting

C/C++, HIP, and CUDA code follows [vLLM&#39;s clang-format style](https://github.com/vllm-project/vllm/blob/main/.clang-format).

```bash
./scripts/format_code.sh
./scripts/format_code.sh --check
```
