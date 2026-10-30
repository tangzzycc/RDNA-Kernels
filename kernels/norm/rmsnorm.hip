#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

// x、y：[M,N]，按行连续存储，使用独立缓冲区。
// gamma：[N]，所有行共享。
// rows = M = B×S；hidden_size = N = 5120；eps = 1e-6。
__global__ void rmsnorm_kernel(const hip_bfloat16* x, const hip_bfloat16* gamma,
                               hip_bfloat16* y, int rows, int hidden_size,
                               float eps) {}

// x：[M,N]，输入隐藏状态，写回归一化结果 y。
// residual：[M,N]，输入残差，写回更新后的残差 s。
// x 与 residual 使用独立缓冲区；gamma：[N]，所有行共享。
// rows = M = B×S；hidden_size = N = 5120；eps = 1e-6。
__global__ void add_rmsnorm_kernel(hip_bfloat16* x, hip_bfloat16* residual,
                                   const hip_bfloat16* gamma, int rows,
                                   int hidden_size, float eps) {}
