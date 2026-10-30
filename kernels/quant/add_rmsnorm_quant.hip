#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

// x、residual：[M,N]，BF16；residual 原地更新。
// gamma：[N]，BF16；q：[M,N]，INT8；scales：[M]，FP32。
// 按行连续存储，各输入与输出使用独立缓冲区。
// rows = M；hidden_size = N = 5120；eps = 1e-6。
__global__ void add_rmsnorm_quant_kernel(const hip_bfloat16* x,
                                         hip_bfloat16* residual,
                                         const hip_bfloat16* gamma, int8_t* q,
                                         float* scales, int rows,
                                         int hidden_size, float eps) {}
