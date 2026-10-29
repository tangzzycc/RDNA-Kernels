# RDNA-Kernels

RDNA-Kernels is a project for practicing GPU kernel development on AMD RDNA architectures using HIP, Triton, and FlyDSL. It covers operators and fused kernels found in common models, with a focus on implementation, correctness validation, performance analysis, and optimization for RDNA hardware.

## ROCm Environment

The project uses a dedicated `.venv` environment for HIP compilation and performance analysis on the Radeon 8060S (RDNA 3.5, `gfx1151`).

| Item | Version / Configuration |
| --- | --- |
| Python | 3.12.3 |
| ROCm SDK | 10.1.0 stable |
| ROCm Profiler package | 10.1.0, matching the SDK release |
| GPU compilation target | `gfx1151` |

The SDK is installed from [AMD's official stable wheel repository](https://stable.repo.amd.com/rocm/whl-next/) and includes development packages, runtime libraries, and the `gfx1151` device package.
The profiler's Python analysis dependencies follow the requirements bundled with this release.

### Tools

| Command | Component Version | Purpose |
| --- | --- | --- |
| `hipcc` | HIP 7.16.26385 / Clang 24 | Compile HIP programs |
| `rocprofv3` | 1.4.1 | Trace APIs and kernels, collect hardware counters |
| `rocprof-compute` | 3.9.0 | Collect and analyze kernel metrics |
| `rocprof-sys-run` | 1.9.0 | Trace system and GPU activity |

Component versions are numbered independently of the ROCm release package version.

### Installation

From the repository root, use `uv` to create the environment and install dependencies:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-rocm.txt
uv pip install --python .venv/bin/python --index-url https://pypi.org/simple -r requirements-profiler.txt
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
