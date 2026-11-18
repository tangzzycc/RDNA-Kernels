# elementwise

## 测试命令

在仓库根目录执行.

```bash
.venv/bin/python -m pytest tests/test_elementwise.py -q

for op in add mul silu silu_mul; do
  ./scripts/analyze_kernel.sh --run-only "elementwise/$op" -- 134217728
done
./scripts/analyze_kernel.sh --run-only elementwise/softmax -- 32768 5120

# CU 模式, 使用相同的输入规模.
./scripts/analyze_kernel.sh --cu-mode --run-only elementwise/add -- 134217728
./scripts/analyze_kernel.sh --cu-mode --run-only elementwise/softmax -- 32768 5120

# 采集并打印 DRAM 接口吞吐.
./scripts/analyze_kernel.sh --bandwidth elementwise/add -- 134217728
```

## 测试结果

Radeon 8060S (`gfx1151`), ROCm 10.1.0, WGP 模式.
BF16, 逐元素缓冲区各为 256 MiB, Softmax 缓冲区各为 320 MiB. 耗时为预热 20 次后计时 100 次的平均值.

| 算子 | 输入规模 | 正确性 | 耗时 (us) | DRAM 接口 (GB/s) |
| --- | --- | ---: | ---: | ---: |
| `add` | `n=134217728` | 通过 | 3498.428 | 227.17 |
| `mul` | `n=134217728` | 通过 | 3488.709 | 227.15 |
| `silu` | `n=134217728` | 通过 | 2302.150 | 226.09 |
| `silu_mul` | `n=134217728` | 通过 | 3483.355 | 227.28 |
| `softmax` | `[32768, 5120]` | 通过 | 4893.179 | 180.06 |

## CU 模式对比

两种模式均为本次测试, 输入规模和计时方式同上. CU 正确性测试及全部基础检查通过.
DRAM 吞吐为读写合计.

| 算子 | WGP 耗时 (us) | CU 耗时 (us) | WGP DRAM (GB/s) | CU DRAM (GB/s) |
| --- | ---: | ---: | ---: | ---: |
| `add` | 3536.937 | 3628.836 | 226.22 | 221.51 |
| `mul` | 3535.332 | 3633.483 | 225.28 | 220.98 |
| `silu` | 2329.241 | 2325.793 | 228.00 | 224.65 |
| `silu_mul` | 3511.390 | 3511.435 | 227.04 | 226.36 |
| `softmax` | 4720.885 | 4515.236 | 174.15 | 179.30 |

本次 Softmax 在 CU 模式下耗时减少约 4.4%, 其他算子的耗时差异在 3% 以内.
