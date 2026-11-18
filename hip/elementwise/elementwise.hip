#include <__clang_hip_runtime_wrapper.h>
#include <cmath>
#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <hipcub/backend/rocprim/block/block_reduce.hpp>
#include <hipcub/backend/rocprim/thread/thread_operators.hpp>

#include "../../common/hip/common.h"

// 单元素规则使用 FP32 计算, 公共遍历负责 BF16 读写和标量尾部处理.

// Add: y = a + b.
struct AddOp {
  __device__ __forceinline__ float operator()(float a, float b) const { return a + b; }
};

// Mul: y = a * b.
struct MulOp {
  __device__ __forceinline__ float operator()(float a, float b) const { return a * b; }
};

// SiLU: y = x * sigmoid(x) = x / (1 + exp(-x)).
struct SiluOp {
  __device__ __forceinline__ float operator()(float x) const { return x / (1.0f + expf(-x)); }
};

// SiLU + Mul: act = BF16(silu(gate)), y = BF16(float(act) * float(up)).
struct SiluMulOp {
  __device__ __forceinline__ float operator()(float gate, float up) const {
    const hip_bfloat16 act = static_cast<hip_bfloat16>(SiluOp{}(gate));
    return static_cast<float>(act) * up;
  }
};

// 以下逐元素 kernel 复用公共遍历, VEC_SIZE 为 launch 选择的向量元素数.
template <int VEC_SIZE>
__global__ void add_kernel(const hip_bfloat16* __restrict__ a, const hip_bfloat16* __restrict__ b,
                           hip_bfloat16* __restrict__ y, size_t n) {
  binary_elementwise<VEC_SIZE>(a, b, y, n, AddOp{});
}

template <int VEC_SIZE>
__global__ void mul_kernel(const hip_bfloat16* __restrict__ a, const hip_bfloat16* __restrict__ b,
                           hip_bfloat16* __restrict__ y, size_t n) {
  binary_elementwise<VEC_SIZE>(a, b, y, n, MulOp{});
}

template <int VEC_SIZE>
__global__ void silu_kernel(const hip_bfloat16* __restrict__ x, hip_bfloat16* __restrict__ y,
                            size_t n) {
  unary_elementwise<VEC_SIZE>(x, y, n, SiluOp{});
}

template <int VEC_SIZE>
__global__ void silu_mul_kernel(const hip_bfloat16* __restrict__ gate,
                                const hip_bfloat16* __restrict__ up, hip_bfloat16* __restrict__ y,
                                size_t n) {
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
__global__ void softmax_kernel(const hip_bfloat16* __restrict__ x, hip_bfloat16* __restrict__ y,
                               int cols) {
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
  const float block_max = BlockReduce(reduce_store).Reduce(local_max, hipcub::Max{});
  if (threadIdx.x == 0) {
    row_max = block_max;
  }
  __syncthreads();

  // step 3: 计算 exp(x - m) 并归约指数和 s, 共享 s 后同步整个 block.
  for (size_t i = threadIdx.x; i < static_cast<size_t>(cols); i += BLOCK_SIZE) {
    const float value = static_cast<float>(input_row[i]);
    local_sum += expf(value - row_max);
  }

  const float block_sum = BlockReduce(reduce_store).Reduce(local_sum, hipcub::Sum{});
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
