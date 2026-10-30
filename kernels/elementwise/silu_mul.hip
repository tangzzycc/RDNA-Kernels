#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstddef>

// gate、up、y：[M,25600]，BF16，各自使用独立的连续缓冲区。
// n = M * 25600，展平后的元素总数。
__global__ void silu_mul_kernel(const hip_bfloat16* gate,
                                const hip_bfloat16* up, hip_bfloat16* y,
                                size_t n) {}
