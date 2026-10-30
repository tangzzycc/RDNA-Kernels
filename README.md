# RDNA-Kernels

RDNA-Kernels is a project for practicing GPU kernel development on AMD RDNA architectures using HIP, Triton, and FlyDSL. It covers operators and fused kernels found in common models, with a focus on implementation, correctness validation, performance analysis, and optimization for RDNA hardware.

## ROCm Environment

The project uses a dedicated `.venv` environment for HIP compilation and performance analysis on the Radeon 8060S (RDNA 3.5, `gfx1151`).

| Item                   | Version / Configuration          |
| ---------------------- | -------------------------------- |
| Python                 | 3.12.3                           |
| ROCm SDK               | 10.1.0 stable                    |
| ROCm Profiler package  | 10.1.0, matching the SDK release |
| GPU compilation target | `gfx1151`                      |

The SDK is installed from [AMD&#39;s official stable wheel repository](https://stable.repo.amd.com/rocm/whl-next/) and includes development packages, runtime libraries, and the `gfx1151` device package.
The profiler's Python analysis dependencies follow the requirements bundled with this release.

### Tools

| Command             | Component Version         | Purpose                                           |
| ------------------- | ------------------------- | ------------------------------------------------- |
| `hipcc`           | HIP 7.16.26385 / Clang 24 | Compile HIP programs                              |
| `rocprofv3`       | 1.4.1                     | Trace APIs and kernels, collect hardware counters |
| `rocprof-compute` | 3.9.0                     | Collect and analyze kernel metrics                |
| `rocprof-sys-run` | 1.9.0                     | Trace system and GPU activity                     |

Component versions are numbered independently of the ROCm release package version.

### Installation

From the repository root, use `uv` to create the environment and install dependencies:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-rocm.txt
uv pip install --python .venv/bin/python --index-url https://pypi.org/simple -r requirements-profiler.txt
uv pip install --python .venv/bin/python --index-url https://pypi.org/simple -r requirements-test.txt
uv pip install --python .venv/bin/python --index-url https://pypi.org/simple -r requirements-format.txt
.venv/bin/rocm-sdk init
.venv/bin/python scripts/configure_rocm_env.py
```

### Activation and Usage

Activate the environment in Bash and check the SDK version and path:

```bash
source .venv/bin/activate
rocm-sdk version
rocm-sdk path --root
```

Once activated, HIP and profiler commands are resolved from `.venv/bin`, and `ROCM_PATH` is automatically set to this environment's SDK.
Run `deactivate` to exit and restore the previous environment variables.

## Validation and Analysis

Correctness tests are grouped by operator family in `tests/test_<family>.py`.
HIP tests compile the operator into a shared library and compare GPU outputs with NumPy references.
Shared HIP support lives in `common/python/`, with reusable test utilities in `tests/helpers.py`.
Family-level `profiler.hip` entry points provide smoke checks and benchmarks;
`--profile` launches the selected workload once. Operator sources contain the kernel and launch function.

```bash
.venv/bin/python -m pytest tests/test_elementwise.py -v
./scripts/analyze_kernel.sh elementwise/add
```

## Code Formatting

HIP, CUDA, and C/C++ sources and headers use [vLLM&#39;s clang-format configuration](https://github.com/vllm-project/vllm/blob/main/.clang-format),
with clang-format 23.1.3 matching its official pre-commit hook.
The script includes tracked and untracked source files throughout the repository, skipping Git-ignored files and symlinks.

```bash
./scripts/format_code.sh
./scripts/format_code.sh --check
```
