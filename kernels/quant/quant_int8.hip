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
