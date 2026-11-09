# matmul

## 测试命令

在仓库根目录执行.

```bash
.venv/bin/python -m pytest tests/test_matmul.py -q

# 各版本使用 launch.hip 中的默认配置.
for version in 1 2 3 4 5 6 9 10; do
  ./scripts/analyze_kernel.sh --run-only "matmul/sgemm_v$version" -- 5120 5120 5120 3 3 3
done

# 指定预热次数, 每批 launch 次数和计时批数.
./scripts/analyze_kernel.sh --run-only matmul/sgemm_v10 -- 5120 5120 5120 8 4 4

# 采集并打印 DRAM 接口吞吐.
./scripts/analyze_kernel.sh --bandwidth matmul/sgemm_v10
```

## 测试结果

Radeon 8060S (`gfx1151`), ROCm 10.1.0.
FP32, `M=N=K=5120`, `C=A*B`. 统一预热 3 次, 每批计时 3 次, 共 3 批取平均值. FMA 计为 2 FLOPs.
FP32 峰值参考: 理论 **29.696 TFLOP/s** (40 CU, 2.9 GHz, VOPD 双发射 FMA), ROCm Roofline 实测 **18.841 TFLOP/s**.

| 版本 | 配置                              | 正确性 | 平均耗时 (ms) | TFLOP/s | DRAM 接口 (GB/s) |
| ---- | --------------------------------- | -----: | ------------: | ------: | ---------------: |
| V1   | `block(32, 32)`                 |   通过 |      2309.391 |   0.116 |             7.33 |
| V2   | `<32>`                          |   通过 |       860.421 |   0.312 |            21.91 |
| V3   | `<32>`                          |   通过 |       211.765 |   1.268 |           127.23 |
| V4   | `<64,64,8,8>`                   |   通过 |        92.477 |   2.903 |           114.51 |
| V5   | `<128,128,8,8,8>`               |   通过 |        42.013 |   6.389 |           110.95 |
| V6   | `<128,128,8,8,8>`               |   通过 |        32.389 |   8.288 |           118.90 |
| V9   | `<128,256,8,4,4,256>`           |   通过 |        26.191 |  10.249 |           157.97 |
| V10  | `<128,128,16,32,128,2,4,4,128>` |   通过 |        21.444 |  12.518 |           181.43 |
