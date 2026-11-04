#include <__clang_hip_runtime_wrapper.h>
#include <cmath>
#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <hipcub/backend/rocprim/block/block_reduce.hpp>
#include <hipcub/backend/rocprim/thread/thread_operators.hpp>
#include <limits>

#include "../../common/hip/common.h"

// 单元素规则使用 FP32 计算, 公共遍历负责 BF16 读写和标量尾部处理.

// Add: y = a + b.
struct AddOp {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return a + b;
  }
};

// Mul: y = a * b.
struct MulOp {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return a * b;
  }
};

// SiLU: y = x * sigmoid(x) = x / (1 + exp(-x)).
// 参考 PyTorch 的 silu_kernel:
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ActivationSiluKernel.cu
struct SiluOp {
  __device__ __forceinline__ float operator()(float x) const {
    return x / (1.0f + expf(-x));
  }
};

// SiLU + Mul: act = BF16(silu(gate)), y = BF16(float(act) * float(up)).
// 参考 vLLM 的 silu_and_mul 与 act_and_mul_kernel:
// https://github.com/vllm-project/vllm/blob/main/csrc/activation_kernels.cu
struct SiluMulOp {
  __device__ __forceinline__ float operator()(float gate, float up) const {
    const hip_bfloat16 act = static_cast<hip_bfloat16>(SiluOp{}(gate));
    return static_cast<float>(act) * up;
  }
};

// 以下逐元素 kernel 复用公共遍历, VEC_SIZE 为 launch 选择的向量元素数.
template <int VEC_SIZE>
__global__ void add_kernel(const hip_bfloat16* __restrict__ a,
                           const hip_bfloat16* __restrict__ b,
                           hip_bfloat16* __restrict__ y, size_t n) {
  binary_elementwise<VEC_SIZE>(a, b, y, n, AddOp{});
}

template <int VEC_SIZE>
__global__ void mul_kernel(const hip_bfloat16* __restrict__ a,
                           const hip_bfloat16* __restrict__ b,
                           hip_bfloat16* __restrict__ y, size_t n) {
  binary_elementwise<VEC_SIZE>(a, b, y, n, MulOp{});
}

template <int VEC_SIZE>
__global__ void silu_kernel(const hip_bfloat16* __restrict__ x,
                            hip_bfloat16* __restrict__ y, size_t n) {
  unary_elementwise<VEC_SIZE>(x, y, n, SiluOp{});
}

template <int VEC_SIZE>
__global__ void silu_mul_kernel(const hip_bfloat16* __restrict__ gate,
                                const hip_bfloat16* __restrict__ up,
                                hip_bfloat16* __restrict__ y, size_t n) {
  binary_elementwise<VEC_SIZE>(gate, up, y, n, SiluMulOp{});
}

/*
 * Softmax: 沿每行最后一维归一化, 将输入分数转换为概率.
 *
 *   m         = max_i x[row, i]
 *   s         = sum_i exp(x[row, i] - m)
 *   y[row, i] = exp(x[row, i] - m) / s
 *
 * 参考 PyTorch SoftMax.cu 的最大值平移与归约结构:
 * aten/src/ATen/native/cuda/SoftMax.cu
 * https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/SoftMax.cu
 *
 * 实现步骤:
 * 1. 每行分配一个 block, 定位行指针, 各线程跨步处理行内元素.
 * 2. 使用 FP32 归约行内最大值, 共享归约结果并同步整个 block.
 * 3. 计算 exp(x - m), 使用 FP32 归约指数和, 共享结果后同步.
 * 4. 用指数和归一化, 将结果转换为 BF16 写回 y.
 */
template <int BLOCK_SIZE>
__global__ void softmax_kernel(const hip_bfloat16* __restrict__ x,
                               hip_bfloat16* __restrict__ y, int cols) {
  // step 1: 定位当前行, 准备局部归约变量和共享内存.
  const size_t offset = static_cast<size_t>(blockIdx.x) * cols;
  const hip_bfloat16* input_row = x + offset;
  hip_bfloat16* output_row = y + offset;

  float local_max = -INFINITY;
  float local_sum = 0.0f;

  __shared__ float row_max;
  __shared__ float row_sum;

  // step 2: 归约行最大值 m, 由线程 0 共享结果并同步整个 block.
  for (size_t i = threadIdx.x; i < static_cast<size_t>(cols); i += BLOCK_SIZE) {
    local_max = fmax(local_max, static_cast<float>(input_row[i]));
  }

  using BlockReduce = hipcub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage reduce_store;
  const float block_max =
      BlockReduce(reduce_store).Reduce(local_max, hipcub::Max{});
  if (threadIdx.x == 0) {
    row_max = block_max;
  }
  __syncthreads();

  // step 3: 计算 exp(x - m) 并归约指数和 s, 共享 s 后同步整个 block.
  for (size_t i = threadIdx.x; i < static_cast<size_t>(cols); i += BLOCK_SIZE) {
    const float value = static_cast<float>(input_row[i]);
    local_sum += expf(value - row_max);
  }

  const float block_sum =
      BlockReduce(reduce_store).Reduce(local_sum, hipcub::Sum{});
  if (threadIdx.x == 0) {
    row_sum = block_sum;
  }
  __syncthreads();

  // step 4: 用 s 归一化每个指数值, 转换为 BF16 写回 y.
  const float inv_sum = 1.0f / row_sum;
  for (size_t i = threadIdx.x; i < static_cast<size_t>(cols); i += BLOCK_SIZE) {
    const float value = static_cast<float>(input_row[i]);
    const float probability = expf(value - row_max) * inv_sum;
    output_row[i] = static_cast<hip_bfloat16>(probability);
  }
}

// a, b, y: [n], BF16, 连续存储, 形状相同且使用独立缓冲区.
// 输入 [M, N] 可展平为 [n], n = M * N; 输入 [B, S, N] 时 M = B * S.
// n 为元素总数, 无需整除向量宽度; n == 0 时允许空指针并直接返回成功.
// 提供 C ABI, 供 Python 测试和独立程序调用; 在指定 stream 上异步执行.
extern "C" hipError_t launch_add(const hip_bfloat16* a, const hip_bfloat16* b,
                                 hip_bfloat16* y, size_t n,
                                 hipStream_t stream) {
  // step 1: 处理 n == 0, 检查非空输入的 a, b 和 y 指针.
  if (n == 0) {
    return hipSuccess;
  }
  if (a == nullptr || b == nullptr || y == nullptr) {
    return hipErrorInvalidValue;
  }
  // step 2: 从 BF16 的 8 元素向量开始, 根据长度和三个指针的对齐逐级缩小.
  int vec_size = 16 / sizeof(hip_bfloat16);
  while (vec_size > 1 && (static_cast<size_t>(vec_size) > n ||
                          !is_aligned(a, vec_size * sizeof(hip_bfloat16)) ||
                          !is_aligned(b, vec_size * sizeof(hip_bfloat16)) ||
                          !is_aligned(y, vec_size * sizeof(hip_bfloat16)))) {
    vec_size /= 2;
  }
  // step 3: 用完整向量数加尾部任务计算 grid, 每个 block 使用 256 个线程.
  constexpr unsigned int threads = 256;
  const size_t tasks = n / vec_size + (n % vec_size != 0);
  const size_t blocks = (tasks - 1) / threads + 1;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    return hipErrorInvalidValue;
  }
  // step 4: 将 vec_size 分发为模板 VEC_SIZE, 异步启动 kernel 并返回启动状态.
  const dim3 grid(static_cast<unsigned int>(blocks)), block(threads);
  HIP_DISPATCH_VEC_SIZE(
      vec_size, add_kernel<VEC_SIZE><<<grid, block, 0, stream>>>(a, b, y, n));
  return hipGetLastError();
}

// a, b, y: [n], BF16, 连续存储, 形状相同且使用独立缓冲区.
// 输入 [M, N] 可展平为 [n], n = M * N; 输入 [B, S, N] 时 M = B * S.
// n 为元素总数, 无需整除向量宽度; n == 0 时允许空指针并直接返回成功.
// 提供 C ABI, 供 Python 测试和独立程序调用; 在指定 stream 上异步执行.
extern "C" hipError_t launch_mul(const hip_bfloat16* a, const hip_bfloat16* b,
                                 hip_bfloat16* y, size_t n,
                                 hipStream_t stream) {
  // step 1: 处理 n == 0, 检查非空输入的 a, b 和 y 指针.
  if (n == 0) {
    return hipSuccess;
  }
  if (a == nullptr || b == nullptr || y == nullptr) {
    return hipErrorInvalidValue;
  }
  // step 2: 从 BF16 的 8 元素向量开始, 根据长度和三个指针的对齐逐级缩小.
  int vec_size = 16 / sizeof(hip_bfloat16);
  while (vec_size > 1 && (static_cast<size_t>(vec_size) > n ||
                          !is_aligned(a, vec_size * sizeof(hip_bfloat16)) ||
                          !is_aligned(b, vec_size * sizeof(hip_bfloat16)) ||
                          !is_aligned(y, vec_size * sizeof(hip_bfloat16)))) {
    vec_size /= 2;
  }
  // step 3: 用完整向量数加尾部任务计算 grid, 每个 block 使用 256 个线程.
  constexpr unsigned int threads = 256;
  const size_t tasks = n / vec_size + (n % vec_size != 0);
  const size_t blocks = (tasks - 1) / threads + 1;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    return hipErrorInvalidValue;
  }
  // step 4: 将 vec_size 分发为模板 VEC_SIZE, 异步启动 kernel 并返回启动状态.
  const dim3 grid(static_cast<unsigned int>(blocks)), block(threads);
  HIP_DISPATCH_VEC_SIZE(
      vec_size, mul_kernel<VEC_SIZE><<<grid, block, 0, stream>>>(a, b, y, n));
  return hipGetLastError();
}

// x, y: [n], BF16, 连续存储且使用独立缓冲区.
// 输入 [M, N] 可展平为 [n], n = M * N; 输入 [B, S, N] 时 M = B * S.
// n 为元素总数, 无需整除向量宽度; n == 0 时允许空指针并直接返回成功.
// 提供 C ABI, 供 Python 测试和独立程序调用; 在指定 stream 上异步执行.
extern "C" hipError_t launch_silu(const hip_bfloat16* x, hip_bfloat16* y,
                                  size_t n, hipStream_t stream) {
  // step 1: 处理 n == 0, 检查非空输入的 x 和 y 指针.
  if (n == 0) {
    return hipSuccess;
  }
  if (x == nullptr || y == nullptr) {
    return hipErrorInvalidValue;
  }
  // step 2: 从 BF16 的 8 元素向量开始, 根据长度和两个指针的对齐逐级缩小.
  int vec_size = 16 / sizeof(hip_bfloat16);
  while (vec_size > 1 && (static_cast<size_t>(vec_size) > n ||
                          !is_aligned(x, vec_size * sizeof(hip_bfloat16)) ||
                          !is_aligned(y, vec_size * sizeof(hip_bfloat16)))) {
    vec_size /= 2;
  }
  // step 3: 用完整向量数加尾部任务计算 grid, 每个 block 使用 256 个线程.
  constexpr unsigned int threads = 256;
  const size_t tasks = n / vec_size + (n % vec_size != 0);
  const size_t blocks = (tasks - 1) / threads + 1;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    return hipErrorInvalidValue;
  }
  // step 4: 将 vec_size 分发为模板 VEC_SIZE, 异步启动 kernel 并返回启动状态.
  const dim3 grid(static_cast<unsigned int>(blocks)), block(threads);
  HIP_DISPATCH_VEC_SIZE(
      vec_size, silu_kernel<VEC_SIZE><<<grid, block, 0, stream>>>(x, y, n));
  return hipGetLastError();
}

// gate, up, y: [n], BF16, 连续存储且使用独立缓冲区.
// 输入 [M, N] 可展平为 [n], n = M * N; 两路输入和输出形状相同.
// vLLM 使用拼接输入 [M, 2 * N], 本接口使用独立的 gate 和 up 缓冲区.
// n 为元素总数, 无需整除向量宽度; n == 0 时允许空指针并直接返回成功.
// 提供 C ABI, 供 Python 测试和独立程序调用; 在指定 stream 上异步执行.
extern "C" hipError_t launch_silu_mul(const hip_bfloat16* gate,
                                      const hip_bfloat16* up, hip_bfloat16* y,
                                      size_t n, hipStream_t stream) {
  // step 1: 处理 n == 0, 检查非空输入的 gate, up 和 y 指针.
  if (n == 0) {
    return hipSuccess;
  }
  if (gate == nullptr || up == nullptr || y == nullptr) {
    return hipErrorInvalidValue;
  }
  // step 2: 从 BF16 的 8 元素向量开始, 根据长度和三个指针的对齐逐级缩小.
  int vec_size = 16 / sizeof(hip_bfloat16);
  while (vec_size > 1 && (static_cast<size_t>(vec_size) > n ||
                          !is_aligned(gate, vec_size * sizeof(hip_bfloat16)) ||
                          !is_aligned(up, vec_size * sizeof(hip_bfloat16)) ||
                          !is_aligned(y, vec_size * sizeof(hip_bfloat16)))) {
    vec_size /= 2;
  }
  // step 3: 用完整向量数加尾部任务计算 grid, 每个 block 使用 256 个线程.
  constexpr unsigned int threads = 256;
  const size_t tasks = n / vec_size + (n % vec_size != 0);
  const size_t blocks = (tasks - 1) / threads + 1;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    return hipErrorInvalidValue;
  }
  // step 4: 将 vec_size 分发为模板 VEC_SIZE, 异步启动 kernel 并返回启动状态.
  const dim3 grid(static_cast<unsigned int>(blocks)), block(threads);
  HIP_DISPATCH_VEC_SIZE(vec_size, silu_mul_kernel<VEC_SIZE>
                        <<<grid, block, 0, stream>>>(gate, up, y, n));
  return hipGetLastError();
}

// x, y: [M, N], BF16, 按行连续存储且使用独立缓冲区.
// rows = M, cols = N, 每个 block 处理一行, 沿最后一维归一化.
// 使用 FP32 计算最大值和指数和, 最终转换为 BF16 输出.
// 每行至少一个有限值; 非有限输入仅支持 -inf 掩码.
// 提供 C ABI, 供 Python 测试和独立程序调用; 在指定 stream 上异步执行.
extern "C" hipError_t launch_softmax(const hip_bfloat16* x, hip_bfloat16* y,
                                     int rows, int cols, hipStream_t stream) {
  // step 1: 检查 shape, 空输入直接返回, 非空输入检查指针.
  if (rows < 0 || cols <= 0) {
    return hipErrorInvalidValue;
  }
  if (rows == 0) {
    return hipSuccess;
  }
  if (x == nullptr || y == nullptr) {
    return hipErrorInvalidValue;
  }

  // step 2: 选择线程数, 长行由各线程跨步处理.
  int block_size = 32;
  while (block_size < cols && block_size < 256) {
    block_size *= 2;
  }
  const dim3 grid(static_cast<unsigned int>(rows));
  const dim3 block(static_cast<unsigned int>(block_size));

  // step 3: 将运行时 block_size 分发为模板 BLOCK_SIZE.
  HIP_DISPATCH_BLOCK_SIZE(block_size, softmax_kernel<BLOCK_SIZE>
                          <<<grid, block, 0, stream>>>(x, y, cols));
  return hipGetLastError();
}
