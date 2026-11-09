#include "rmsnorm.hip"

#include <algorithm>
#include <numeric>

// x, y: [M, N], 按行连续存储; gamma: [N], 所有行共享.
// rows = M, hidden_size = N; 输入 [B, S, N] 可展平为 [M, N], M = B * S.
// 提供 C ABI, 供 Python 测试和独立程序调用; 算子在指定流上异步执行.
extern "C" hipError_t launch_rmsnorm(const hip_bfloat16* x, const hip_bfloat16* gamma,
                                     hip_bfloat16* y, int rows, int hidden_size, float eps,
                                     hipStream_t stream) {
  // step 1: 检查 shape 与指针，空输入直接返回
  if (rows < 0 || hidden_size <= 0) {
    return hipErrorInvalidValue;
  }
  if (rows == 0) {
    return hipSuccess;
  }
  if (x == nullptr || gamma == nullptr || y == nullptr) {
    return hipErrorInvalidValue;
  }
  // step 2: 根据长度整除与地址对齐选择向量宽度
  constexpr int max_vec_size = 16 / sizeof(hip_bfloat16);
  int vec_size = std::gcd(max_vec_size, hidden_size);
  // 支持带偏移的输入指针
  while (vec_size > 1) {
    const size_t alignment = vec_size * sizeof(hip_bfloat16);
    if (is_aligned(x, alignment) && is_aligned(gamma, alignment) && is_aligned(y, alignment)) {
      break;
    }
    vec_size /= 2;
  }
  // step 3: 每行分配一个 block，根据行数与向量任务数选择线程数
  const int max_block_size = rows < 256 ? 1024 : 256;
  const int block_size = std::min(hidden_size / vec_size, max_block_size);
  dim3 grid(rows);
  dim3 block(block_size);

  // step 4: 将运行时向量宽度分发到对应模板实例并启动 kernel
  HIP_DISPATCH_VEC_SIZE(vec_size, rmsnorm_kernel<VEC_SIZE>
                        <<<grid, block, 0, stream>>>(x, gamma, y, hidden_size, eps));

  // step 5: 返回启动状态，保持 stream 中的异步执行
  return hipGetLastError();
}

// x, residual: [M, N], 按行连续存储; gamma: [N], 所有行共享.
// x 写回归一化结果, residual 写回相加结果; 三个参数使用独立缓冲区.
// rows = M, hidden_size = N; 输入 [B, S, N] 可展平为 [M, N], M = B * S.
// eps 为 FP32 的数值稳定项; 提供 C ABI, 在指定 stream 上异步执行.
extern "C" hipError_t launch_add_rmsnorm(hip_bfloat16* x, hip_bfloat16* residual,
                                         const hip_bfloat16* gamma, int rows, int hidden_size,
                                         float eps, hipStream_t stream) {
  // step 1: 检查 shape 与指针, 空输入直接返回.
  if (rows < 0 || hidden_size <= 0) {
    return hipErrorInvalidValue;
  }
  if (rows == 0) {
    return hipSuccess;
  }
  if (x == nullptr || residual == nullptr || gamma == nullptr) {
    return hipErrorInvalidValue;
  }

  // step 2: 用长度的最大公约数选择向量宽度, 再检查三个指针的对齐.
  constexpr int max_vec_size = 16 / sizeof(hip_bfloat16);
  int vec_size = std::gcd(max_vec_size, hidden_size);
  while (vec_size > 1) {
    const size_t alignment = vec_size * sizeof(hip_bfloat16);
    if (is_aligned(x, alignment) && is_aligned(residual, alignment) &&
        is_aligned(gamma, alignment)) {
      break;
    }
    vec_size /= 2;
  }

  // step 3: 每行分配一个 block, 根据行数和向量任务数选择线程数.
  const int max_block_size = rows < 256 ? 1024 : 256;
  const int block_size = std::min(hidden_size / vec_size, max_block_size);
  const dim3 grid(rows);
  const dim3 block(block_size);

  // step 4: 将运行时向量宽度分发到模板实例, 在指定 stream 上启动 kernel.
  HIP_DISPATCH_VEC_SIZE(vec_size, add_rmsnorm_kernel<VEC_SIZE>
                        <<<grid, block, 0, stream>>>(x, residual, gamma, hidden_size, eps));

  // step 5: 返回启动状态, 保持 stream 中的异步执行.
  return hipGetLastError();
}
