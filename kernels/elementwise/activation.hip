#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstddef>

// x、y：[M,25600]，BF16，连续存储，使用独立缓冲区。
// n = M * 25600，展平后的元素总数。
__global__ void silu_kernel(const hip_bfloat16* x, hip_bfloat16* y, size_t n) {}

// gate、up、y：[M,25600]，BF16，各自使用独立的连续缓冲区。
// n = M * 25600，展平后的元素总数。
__global__ void silu_mul_kernel(const hip_bfloat16* gate,
                                const hip_bfloat16* up, hip_bfloat16* y,
                                size_t n) {}

// x、y：[M,N]，BF16，按行连续存储，使用独立缓冲区。
// rows = M = B×64×S；cols = N = L，沿最后一维归一化。
__global__ void softmax_kernel(const hip_bfloat16* x, hip_bfloat16* y, int rows,
                               int cols) {}

// FP32 路径用于 sampler；x、y：[rows,cols]，连续存储，独立缓冲区。
// rows 为采样请求数；cols 为词表大小 151936 或筛选后的候选数量。
__global__ void softmax_kernel(const float* x, float* y, int rows, int cols) {}
