#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstddef>

// x、y：[M,25600]，BF16，连续存储，使用独立缓冲区。
// n = M * 25600，展平后的元素总数。
__global__ void silu_kernel(const hip_bfloat16* x, hip_bfloat16* y, size_t n) {}
