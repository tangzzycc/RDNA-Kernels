# Kernels

记录已实现的 HIP 算子的定义, 实现参考, 流程与注意事项.

## norm

源码: [rmsnorm.hip](../kernels/norm/rmsnorm.hip).

沿最后一维独立归一化. 隐藏状态按连续的 `[M, N]` 处理, `gamma` 为所有行共享的 `[N]` 权重. `rows = M`, `hidden_size = N`; `[B, S, N]` 可展平为 `[B * S, N]`.

输入, 权重与输出使用 BF16, 平方和及归一化计算使用 FP32. `eps` 为数值稳定项, 常用值为 `1e-6`.

### rmsnorm

**定义**

对每行计算均方根, 归一化后乘以权重:

$$
r = \sqrt{\frac{1}{N}\sum_{i=0}^{N-1}x_i^2 + \epsilon},
\qquad
y_i = \frac{x_i}{r}\gamma_i
$$

`x` 与 `y` 均为 `[M, N]`, 使用独立缓冲区. 归一化和权重乘法完成后, 结果转换为 BF16 写回 `y`.

```cpp
extern "C" hipError_t launch_rmsnorm(
    const hip_bfloat16* x, const hip_bfloat16* gamma, hip_bfloat16* y,
    int rows, int hidden_size, float eps, hipStream_t stream);
```

**实现参考**

[vLLM 的 layernorm_kernels.cu](https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu) 中的 `vllm::rms_norm_kernel` 和 `rms_norm` 启动逻辑. 使用 HIP 和 hipCUB 实现向量访存与 block 归约.

**实现流程**

1. 每行分配一个 block, 线程按 `blockDim.x` 跨步处理向量任务.
2. 使用 `vec_n_t` 读取 `x`, 展开向量内循环, 累加各线程的局部平方和.
3. 使用 `hipcub::BlockReduce` 归约平方和, 线程 `0` 计算 `inv_rms` 并写入共享内存.
4. 同步后重新读取 `x` 和 `gamma`, 计算归一化及缩放, 按向量写回 `y`.
5. launch 检查参数, 选择向量宽度与线程数, 分发模板并在指定 stream 上启动.

**Tips**

- `VEC_SIZE` 表示一次处理的 BF16 元素数. launch 从 `gcd(8, N)` 开始选择, 再检查所有指针的对齐; 长度不能整除或指针对齐不足时减小宽度, 保证每一行都能完整向量读写.
- `BlockReduce<float, 1024>` 中的 `1024` 是归约类型的线程容量, 实际参与线程数通过 `blockDim.x` 传入. launch 根据行数选择 `1024` 或 `256` 的线程上限, 再受向量任务数限制.
- 两次读取 `x` 避免为整行数据保留大量寄存器. 寄存器缓存可以减少访存, 但特征维度较大时可能产生 spill, 需要结合 shape 测量.
- profiler 打印的 effective bandwidth 按逻辑输入输出字节数计算. 缓存命中和重复读取会影响实际访存量, 该值不能直接当作 DRAM 带宽.

### add_rmsnorm

**定义**

融合残差加法与 RMSNorm:

$$
s = x + r,
\qquad
y = \mathrm{RMSNorm}(s, \gamma, \epsilon)
$$

`r` 表示输入残差. `x` 和 `residual` 均为 `[M, N]`, 使用独立缓冲区; 原地写回 `x` 为归一化结果 `y`, `residual` 为相加结果 `s`.

相加结果先舍入为 BF16. 平方和与后续归一化均使用这个舍入后的 `s`, 最终输出转换为 BF16.

```cpp
extern "C" hipError_t launch_add_rmsnorm(
    hip_bfloat16* x, hip_bfloat16* residual, const hip_bfloat16* gamma,
    int rows, int hidden_size, float eps, hipStream_t stream);
```

**实现参考**

[vLLM 的 layernorm_kernels.cu](https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu) 中的 `vllm::fused_add_rms_norm_kernel`. 沿用先更新残差并归约, 再归一化写回隐藏状态的两次遍历结构.

**实现流程**

1. 每行分配一个 block, 定位 `x` 和 `residual` 的行指针, 转换为向量指针.
2. 向量读取两路输入, 相加后写回 `residual`, 同时累加更新后残差的平方和.
3. 使用 hipCUB 归约, 线程 `0` 计算并共享 `inv_rms`, 随后同步整个 block.
4. 重新读取 `residual` 和 `gamma`, 计算归一化及缩放, 按向量写回 `x`.
5. launch 沿用 RMSNorm 的分发方式, 对齐检查同时覆盖 `x`, `residual` 和 `gamma`.

**Tips**

- 平方和使用 BF16 相加结果转回 FP32 后计算, 与第二次遍历读到的 `residual` 保持一致.
- 第二次遍历读取更新后的 `residual`. 原始 `x` 只参与加法, 最后被归一化结果覆盖; `residual` 保留给后续残差连接.
- `__syncthreads()` 保证共享的归一化因子可见, 并完成当前 block 的残差写回. 两次遍历之间所有线程都要经过同步.
- 融合减少一次独立的 kernel 启动, 并复用残差写回时的遍历计算平方和. 正确性检查需要同时验证 `x` 和 `residual` 两路结果; launch 只返回启动状态, 在指定 stream 上异步执行.
