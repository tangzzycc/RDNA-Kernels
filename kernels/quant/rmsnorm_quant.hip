#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

// x：[M,N]；gamma：[N]，BF16。
// q：[M,N]，INT8；scales：[M]，FP32，保存每行量化尺度。
// 所有矩阵按行连续存储，输入输出使用独立缓冲区。
// rows = M；hidden_size = N，隐藏状态为 5120；eps = 1e-6。
__global__ void rmsnorm_quant_kernel(const hip_bfloat16* x,
                                     const hip_bfloat16* gamma, int8_t* q,
                                     float* scales, int rows, int hidden_size,
                                     float eps) {}
