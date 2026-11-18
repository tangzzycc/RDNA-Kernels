# matmul

矩阵乘法统一计算 `C = A * B`, 基准 shape 为 `M=N=K=8192`.
各版本的默认分块配置集中在 [launch.hip](launch.hip).

## 测试命令

在仓库根目录执行. 脚本默认使用 WGP 模式, 添加 `--cu-mode` 切换为 CU 模式.
`--` 后的参数依次为 `M N K [warmup iterations batches]`.

### 正确性

```bash
GPU_MODE=wgp .venv/bin/python -m pytest tests/test_matmul.py -q
GPU_MODE=cu .venv/bin/python -m pytest tests/test_matmul.py -q
```

### SGEMM 与 Opt

默认预热 1 次, 每批计时 1 次, 共 3 批.

```bash
for version in 1 2 3 4 5 6 9 10; do
  ./scripts/analyze_kernel.sh --run-only "matmul/sgemm_v$version"
done

./scripts/analyze_kernel.sh --run-only matmul/sgemm_opt
./scripts/analyze_kernel.sh --run-only matmul/sgemm_opt_fp16

# CU 模式.
./scripts/analyze_kernel.sh --cu-mode --run-only matmul/sgemm_v10

# 自定义预热次数, 每批计时次数和批数.
./scripts/analyze_kernel.sh --run-only matmul/sgemm_v10 -- 8192 8192 8192 8 4 4
```

### HGEMM

CU 模式, 预热 10 次, 每批计时 1 次, 共 30 批.

```bash
for version in {1..8}; do
  ./scripts/analyze_kernel.sh --cu-mode --run-only "matmul/hgemm_v$version" -- 8192 8192 8192 10 1 30
done

# V4-V8 支持非整齐 shape.
./scripts/analyze_kernel.sh --cu-mode --run-only matmul/hgemm_v8 -- 129 97 65
```

### Profiling

```bash
# 采集并打印 rocprof-compute 报告.
./scripts/analyze_kernel.sh --cu-mode matmul/hgemm_v8

# 采集并打印 DRAM 接口吞吐.
./scripts/analyze_kernel.sh --bandwidth --samples 3 matmul/sgemm_v10
./scripts/analyze_kernel.sh --bandwidth --samples 3 matmul/sgemm_opt
./scripts/analyze_kernel.sh --bandwidth --samples 3 matmul/sgemm_opt_fp16
```

## 测试结果

本机环境: Radeon 8060S (`gfx1151`), ROCm 10.1.0.
以下结果均为 `M=N=K=8192`, 耗时取计时批次的平均值, FMA 计为 2 FLOPs.
DRAM 接口吞吐为读写合计.

峰值算力参考:

- FP32: 理论 **29.696 TFLOP/s**, ROCm Roofline 实测 **18.841 TFLOP/s**.
- FP16 WMMA: 理论 **59.4 TFLOP/s**.

### SGEMM

FP32 输入, 累加与输出, A/B/C 均为行主序.
WGP 模式, 预热 1 次, 每批计时 1 次, 共 3 批.

| 版本 | 配置 | 正确性 | 平均耗时 (ms) | TFLOP/s | DRAM 接口 (GB/s) |
| --- | --- | :---: | ---: | ---: | ---: |
| V1 | `block(32, 32)` | 通过 | 11795.296 | 0.093 | 6.11 |
| V2 | `<32>` | 通过 | 7432.055 | 0.148 | 10.58 |
| V3 | `<32>` | 通过 | 1156.955 | 0.950 | 97.80 |
| V4 | `<64,64,8,8>` | 通过 | 385.424 | 2.853 | 127.89 |
| V5 | `<128,128,8,8,8>` | 通过 | 175.822 | 6.254 | 142.07 |
| V6 | `<128,128,8,8,8>` | 通过 | 133.092 | 8.261 | 210.45 |
| V9 | `<128,256,8,4,4,256>` | 通过 | 112.116 | 9.807 | 205.89 |
| V10 | `<128,128,16,32,128,2,4,4,128>` | 通过 | 100.212 | 10.972 | 261.98 |

### Opt

FP32 与 FP16 共用 `sgemm_opt_kernel<T>`, A/B/C 均为行主序, 使用 FP32 LDS 与 FMA 累加.
FP32 使用 `BLOCK_K=8`, FP16 使用 `BLOCK_K=16` 并预取下一轮数据.
WGP 模式, 预热 1 次, 每批计时 1 次, 共 3 批.

| 输入/输出精度 | 配置 | 正确性 | 平均耗时 (ms) | TFLOP/s | DRAM 接口 (GB/s) |
| --- | --- | :---: | ---: | ---: | ---: |
| FP32 | `<128,128,8,32,128,2,4,4,128>` | 通过 | 89.143 | 12.334 | 208.81 |
| FP16 | `<128,128,16,32,128,2,4,4,128>` | 通过 | 102.886 | 10.687 | 92.03 |

流水线配置对照, 预热 8 次, 每批计时 8 次, 共 3 批.

| 输入/输出精度 | BLOCK_K=8 平均耗时 (ms) | BLOCK_K=16, 软件预取平均耗时 (ms) |
| --- | ---: | ---: |
| FP32 | 84.806 | 102.742 |
| FP16 | 120.703 | 103.590 |

### SGEMM / Opt 编译模式对比

分块配置相同, 预热 1 次, 每批计时 1 次, 共 3 批.
V1-V4 的 WGP 数据为先前记录, 其余行使用 WGP/CU 配对测试.
各版本均通过完整输出核对.

| 版本 | WGP 平均耗时 (ms) | CU 平均耗时 (ms) | CU TFLOP/s |
| --- | ---: | ---: | ---: |
| V1 | 11795.296 | 9907.854 | 0.111 |
| V2 | 7432.055 | 8276.065 | 0.133 |
| V3 | 1156.955 | 964.112 | 1.140 |
| V4 | 385.424 | 313.816 | 3.504 |
| V5 | 177.294 | 580.587 | 1.894 |
| V6 | 137.512 | 134.887 | 8.151 |
| V9 | 113.354 | 116.987 | 9.399 |
| V10 | 102.832 | 104.096 | 10.562 |
| Opt FP32 | 89.701 | 109.295 | 10.060 |
| Opt FP16 | 107.516 | 110.275 | 9.971 |

V5 在 CU 模式下的 scratch 为 152 bytes/thread, WGP 模式下为 0.

### HGEMM

参考实现: [AMD RDNA FP16 GEMM](https://rocm.blogs.amd.com/software-tools-optimization/rdna-fp16-gemm/README.html).
FP16 输入, WMMA 累加与输出. `A[M,K]` 行主序, `B[K,N]` 与 `C[M,N]` 列主序,
地址分别为 `A[row*K+k]`, `B[col*K+k]`, `C[col*M+row]`.
CU 模式, 预热 10 次, 每批计时 1 次, 共 30 批.
八个版本均通过完整输出核对; CU 与 WGP 模式各通过 104 项 HGEMM 测试.

| 版本 | 实现 | 平均耗时 (ms) | TFLOP/s | LDS (KiB) | Scratch (bytes/thread) |
| --- | --- | ---: | ---: | ---: | ---: |
| V1 | 单 wave WMMA | 1556.312 | 0.706 | 2 | 0 |
| V2 | 多 wave, 向量搬运, 双缓冲 | 103.065 | 10.668 | 32 | 0 |
| V3 | LDS padding | 101.282 | 10.856 | 48 | 0 |
| V4 | Raw buffer, 边界处理 | 110.747 | 9.928 | 48 | 8 |
| V5 | 加深 K tile | 77.536 | 14.181 | 63 | 0 |
| V6 | 分组 block 映射 | 46.067 | 23.868 | 63 | 0 |
| V7 | 交错预取与 WMMA | 37.876 | 29.029 | 63 | 0 |
| V8 | 单缓冲 LDS | 32.672 | 33.653 | 31.5 | 0 |

默认配置: V1 为 `<32,32,16>`, V2 为 `<256,256,16,64,64>`.
V3-V4 在 V2 上使用 `LDS_PAD=8`; V5-V8 为 `<128,96,64,64,48,8>`.
V6-V8 使用 `GROUP_SIZE=8`, V8 使用 `NUM_BUFFERS=1`.
