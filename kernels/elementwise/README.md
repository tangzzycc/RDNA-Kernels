# elementwise

## 测试命令

在仓库根目录执行.

```bash
.venv/bin/python -m pytest tests/test_elementwise.py -q

for op in add mul silu silu_mul; do
  ./scripts/analyze_kernel.sh --run-only "elementwise/$op" -- 134217728
done
./scripts/analyze_kernel.sh --run-only elementwise/softmax -- 32768 5120

# 采集并打印 DRAM 接口吞吐.
./scripts/analyze_kernel.sh --bandwidth elementwise/add -- 134217728
```

## 测试结果

Radeon 8060S (`gfx1151`), ROCm 10.1.0.
BF16, 逐元素缓冲区各为 256 MiB, Softmax 缓冲区各为 320 MiB. 耗时为预热 20 次后计时 100 次的平均值.

| 算子 | 输入规模 | 正确性 | 耗时 (us) | DRAM 接口 (GB/s) |
| --- | --- | ---: | ---: | ---: |
| `add` | `n=134217728` | 通过 | 3498.428 | 227.17 |
| `mul` | `n=134217728` | 通过 | 3488.709 | 227.15 |
| `silu` | `n=134217728` | 通过 | 2302.150 | 226.09 |
| `silu_mul` | `n=134217728` | 通过 | 3483.355 | 227.28 |
| `softmax` | `[32768, 5120]` | 通过 | 4893.179 | 180.06 |
