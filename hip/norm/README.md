# norm

## 测试命令

在仓库根目录执行.

```bash
.venv/bin/python -m pytest tests/test_norm.py -q

./scripts/analyze_kernel.sh --run-only norm/rmsnorm -- 32768 5120
./scripts/analyze_kernel.sh --run-only norm/add_rmsnorm -- 32768 5120

# CU 模式, 使用相同的输入规模.
./scripts/analyze_kernel.sh --cu-mode --run-only norm/rmsnorm -- 32768 5120
./scripts/analyze_kernel.sh --cu-mode --run-only norm/add_rmsnorm -- 32768 5120

# 采集并打印 DRAM 接口吞吐.
./scripts/analyze_kernel.sh --bandwidth norm/rmsnorm -- 32768 5120
```

## 测试结果

Radeon 8060S (`gfx1151`), ROCm 10.1.0, WGP 模式.
BF16, 每个矩阵缓冲区 320 MiB. 耗时取 7 批中位数, 每批预热 100 次并计时 500 次.

| 算子 | 输入规模 | 正确性 | 耗时 (us) | DRAM 接口 (GB/s) |
| --- | --- | ---: | ---: | ---: |
| `rmsnorm` | `[32768, 5120]` | 通过 | 3098.228 | 233.88 |
| `add_rmsnorm` | `[32768, 5120]` | 通过 | 6232.884 | 253.65 |

`add_rmsnorm` 重复计时使用 `x=0, gamma=0` 保持原地数据稳定; 随机输入由 pytest 检查.

## CU 模式对比

两种模式均为本次测试, 输入规模和计时方式同上. CU 正确性测试及全部基础检查通过.
DRAM 吞吐为读写合计.

| 算子 | WGP 耗时 (us) | CU 耗时 (us) | WGP DRAM (GB/s) | CU DRAM (GB/s) |
| --- | ---: | ---: | ---: | ---: |
| `rmsnorm` | 3125.197 | 3149.664 | 227.87 | 234.85 |
| `add_rmsnorm` | 6313.520 | 6372.502 | 252.92 | 249.65 |

本次两种模式的耗时差异均在 1% 以内.
