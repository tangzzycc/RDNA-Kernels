#include <__clang_hip_math.h>
#include <__clang_hip_runtime_wrapper.h>
#include <algorithm>
#include <cstddef>
#include <hip/amd_detail/amd_hip_bfloat16.h>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipcub/backend/rocprim/block/block_reduce.hpp>
#include <hipcub/backend/rocprim/thread/thread_operators.hpp>
#include <hipcub/hipcub.hpp>
#include <numeric>
#include <numeric>
#include "../../common/hip/common.h"

/*
 * RMSNorm: 沿每行长度为 N 的特征维度归一化, 再乘以共享权重 gamma.
 *
 *   mean_square = (1 / N) * sum_{i=0}^{N-1} x[row, i]^2
 *   inv_rms     = 1 / sqrt(mean_square + eps)
 *   y[row, i]   = BF16(float(x[row, i]) * inv_rms * float(gamma[i]))
 *
 * 参考 vLLM 的 vllm::rms_norm_kernel 及 rms_norm 启动逻辑:
 * csrc/layernorm_kernels.cu
 * https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu
 *
 * 实现步骤:
 * 1. 每行分配一个 block, 各线程按 blockDim.x 跨步分担向量任务.
 * 2. 使用 vec_n_t 读取 x, 展开向量内的计算循环并累加局部平方和.
 * 3. 使用 hipCUB BlockReduce<float, 1024> 归约, 有效线程数为 blockDim.x.
 * 4. 线程 0 计算 inv_rms 并写入共享内存, 同步后供所有线程使用.
 * 5. 重新读取 x 和 gamma, 融合归一化与缩放, 按向量写回输出.
 */
template <int VEC_SIZE>
__global__ void rmsnorm_kernel(const hip_bfloat16* __restrict__ x,
                               const hip_bfloat16* __restrict__ gamma,
                               hip_bfloat16* __restrict__ y, int hidden_size,
                               float eps) {
  // step 1: 准备行指针和累加变量
  __shared__ float s_inv_rms;
  float variance = 0.0f;

  size_t offset = static_cast<size_t>(blockIdx.x) * hidden_size;
  const hip_bfloat16* input_row = x + offset;
  const hip_bfloat16* weight_row = gamma;
  hip_bfloat16* output_row = y + offset;

  // step 2: 定义向量与标量的平方和操作
  auto vec_op = [&variance](const vec_n_t<hip_bfloat16, VEC_SIZE>& value) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float x = static_cast<float>(value.val[i]);
      variance += x * x;
    }
  };
  auto scalar_op = [&variance](const hip_bfloat16& value) {
    float x = static_cast<float>(value);
    variance += x * x;
  };

  vectorize_read_with_alignment<VEC_SIZE>(input_row, hidden_size, threadIdx.x,
                                          blockDim.x, vec_op, scalar_op);

  // step 3: 使用 hipCUB 完成 block 归约
  using BlockReduce = hipcub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduce_store;
  variance =
      BlockReduce(reduce_store).Reduce(variance, hipcub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_inv_rms = rsqrtf(variance / hidden_size + eps);
  }
  __syncthreads();

  // step 4: 将行指针转换为向量指针
  auto* v_in =
      reinterpret_cast<const vec_n_t<hip_bfloat16, VEC_SIZE>*>(input_row);
  auto* v_w =
      reinterpret_cast<const vec_n_t<hip_bfloat16, VEC_SIZE>*>(weight_row);
  auto* v_out = reinterpret_cast<vec_n_t<hip_bfloat16, VEC_SIZE>*>(output_row);
  for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += blockDim.x) {
    vec_n_t<hip_bfloat16, VEC_SIZE> dst;
    // 读取一个输入向量和一个权重向量
    vec_n_t<hip_bfloat16, VEC_SIZE> src1 = v_in[i];
    vec_n_t<hip_bfloat16, VEC_SIZE> src2 = v_w[i];
// 展开计算其中的 VEC_SIZE 个元素
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      float x = static_cast<float>(src1.val[j]);
      float w = static_cast<float>(src2.val[j]);
      dst.val[j] = static_cast<hip_bfloat16>(x * s_inv_rms * w);
    }
    v_out[i] = dst;
  }
}

// x, y: [M, N], 按行连续存储; gamma: [N], 所有行共享.
// rows = M, hidden_size = N; 输入 [B, S, N] 可展平为 [M, N], M = B * S.
// 提供 C ABI, 供 Python 测试和独立程序调用; 算子在指定流上异步执行.
extern "C" hipError_t launch_rmsnorm(const hip_bfloat16* x,
                                     const hip_bfloat16* gamma, hip_bfloat16* y,
                                     int rows, int hidden_size, float eps,
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
    if (is_aligned(x, alignment) && is_aligned(gamma, alignment) &&
        is_aligned(y, alignment)) {
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
  HIP_DISPATCH_VEC_SIZE(
      vec_size, rmsnorm_kernel<VEC_SIZE>
      <<<grid, block, 0, stream>>>(x, gamma, y, hidden_size, eps));

  // step 5: 返回启动状态，保持 stream 中的异步执行
  return hipGetLastError();
}

/*
 * Add + RMSNorm: 融合残差加法与归一化, 原地更新 x 和 residual.
 *
 *   s[i]     = BF16(float(x[row, i]) + float(residual[row, i]))
 *   inv_rms  = 1 / sqrt((1 / N) * sum_{i=0}^{N-1} float(s[i])^2 + eps)
 *   x[row, i]        = BF16(float(s[i]) * inv_rms * float(gamma[i]))
 *   residual[row, i] = s[i]
 *
 * 实现思路参考 vLLM 的 vllm::fused_add_rms_norm_kernel:
 * csrc/layernorm_kernels.cu
 * https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu
 *
 * 实现步骤:
 * 1. 每行分配一个 block, 各线程按 blockDim.x 跨步分担向量任务.
 * 2. 向量读取 x 和 residual, 将相加结果写回 residual, 累加局部平方和.
 * 3. 使用 hipCUB BlockReduce 归约平方和, 计算并共享 inv_rms.
 * 4. 同步后重新读取 residual 和 gamma, 融合归一化与缩放, 向量写回 x.
 */
template <int VEC_SIZE>
__global__ void add_rmsnorm_kernel(hip_bfloat16* __restrict__ x,
                                   hip_bfloat16* __restrict__ residual,
                                   const hip_bfloat16* __restrict__ gamma,
                                   int hidden_size, float eps) {
  // step 1: 定位当前行, 准备向量指针和归约变量
  using Vec = vec_n_t<hip_bfloat16, VEC_SIZE>;
  const size_t offset = static_cast<size_t>(blockIdx.x) * hidden_size;
  const int num_vec = hidden_size / VEC_SIZE;

  auto* v_x = reinterpret_cast<Vec*>(x + offset);
  auto* v_residual = reinterpret_cast<Vec*>(residual + offset);
  auto* v_gamma = reinterpret_cast<const Vec*>(gamma);

  float variance = 0.0f;
  __shared__ float s_inv_rms;

  // step 2: 向量读取并相加, 写回 residual, 累加更新后残差的平方和
  for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
    const Vec input = v_x[i];
    const Vec residual = v_residual[i];
    Vec sum;
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      const float a = static_cast<float>(input.val[j]);
      const float b = static_cast<float>(residual.val[j]);

      sum.val[j] = static_cast<hip_bfloat16>(a + b);

      const float s = static_cast<float>(sum.val[j]);
      variance += s * s;
    }
    v_residual[i] = sum;
  }

  // step 3: 归约整行平方和, 由线程 0 计算 inv_rms 并同步共享结果
  using BlockReduce = hipcub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduce_store;

  variance =
      BlockReduce(reduce_store).Reduce(variance, hipcub::Sum{}, blockDim.x);
  if (threadIdx.x == 0) {
    s_inv_rms = rsqrtf(variance / hidden_size + eps);
  }
  __syncthreads();

  // step 4: 读取 residual 和 gamma, 融合归一化与缩放, 向量写回 x
  for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
    const Vec residual = v_residual[i];
    const Vec weight = v_gamma[i];
    Vec output;

#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      const float s = static_cast<float>(residual.val[j]);
      const float w = static_cast<float>(weight.val[j]);

      output.val[j] = static_cast<hip_bfloat16>(s * s_inv_rms * w);
    }

    v_x[i] = output;
  }
}

// x, residual: [M, N], 按行连续存储; gamma: [N], 所有行共享.
// x 写回归一化结果, residual 写回相加结果; 三个参数使用独立缓冲区.
// rows = M, hidden_size = N; 输入 [B, S, N] 可展平为 [M, N], M = B * S.
// eps 为 FP32 的数值稳定项; 提供 C ABI, 在指定 stream 上异步执行.
extern "C" hipError_t launch_add_rmsnorm(hip_bfloat16* x,
                                         hip_bfloat16* residual,
                                         const hip_bfloat16* gamma, int rows,
                                         int hidden_size, float eps,
                                         hipStream_t stream) {
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
  HIP_DISPATCH_VEC_SIZE(
      vec_size, add_rmsnorm_kernel<VEC_SIZE>
      <<<grid, block, 0, stream>>>(x, residual, gamma, hidden_size, eps));

  // step 5: 返回启动状态, 保持 stream 中的异步执行.
  return hipGetLastError();
}
