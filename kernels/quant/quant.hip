#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

// x：[rows,cols]，BF16；q：[rows,cols]，INT8，按行连续存储。
// channel_scales：[cols]，FP32，正数，对输入逐列除以 d。
// channel_scales 为 nullptr 时跳过通道缩放。
// scales：[rows]，FP32；各输入输出使用独立缓冲区。
// 激活量化时 rows = M、cols = K；权重量化时 rows = N、cols = K。
__global__ void quant_int8_kernel(const hip_bfloat16* x,
                                  const float* channel_scales, int8_t* q,
                                  float* scales, int rows, int cols) {}

// weight：[N,K]，FP32，离线处理后的编码输入；packed：[N,K/2]，uint8_t。
// scales：[N,K/group_size]，FP32，正数，离线给定。
// zero_points：同 shape，uint8_t，实际零点，取值 0..15。
// rows = N；cols = K；group_size = Gs，默认 128。
// cols 为偶数，且可被正数 group_size 整除；输入输出使用独立缓冲区。
__global__ void quant_int4_kernel(const float* weight, const float* scales,
                                  const uint8_t* zero_points, uint8_t* packed,
                                  int rows, int cols, int group_size) {}

// x：[M,N]；gamma：[N]，BF16。
// q：[M,N]，INT8；scales：[M]，FP32，保存每行量化尺度。
// 所有矩阵按行连续存储，输入输出使用独立缓冲区。
// rows = M；hidden_size = N，隐藏状态为 5120；eps = 1e-6。
__global__ void rmsnorm_quant_kernel(const hip_bfloat16* x,
                                     const hip_bfloat16* gamma, int8_t* q,
                                     float* scales, int rows, int hidden_size,
                                     float eps) {}

// x、residual：[M,N]，BF16；residual 原地更新。
// gamma：[N]，BF16；q：[M,N]，INT8；scales：[M]，FP32。
// 按行连续存储，各输入与输出使用独立缓冲区。
// rows = M；hidden_size = N = 5120；eps = 1e-6。
__global__ void add_rmsnorm_quant_kernel(const hip_bfloat16* x,
                                         hip_bfloat16* residual,
                                         const hip_bfloat16* gamma, int8_t* q,
                                         float* scales, int rows,
                                         int hidden_size, float eps) {}
