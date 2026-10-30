#include <hip/hip_runtime.h>

#include <cstdint>

// weight：[N,K]，FP32，离线处理后的编码输入；packed：[N,K/2]，uint8_t。
// scales：[N,K/group_size]，FP32，正数，离线给定。
// zero_points：同 shape，uint8_t，实际零点，取值 0..15。
// rows = N；cols = K；group_size = Gs，默认 128。
// cols 为偶数，且可被正数 group_size 整除；输入输出使用独立缓冲区。
__global__ void quant_int4_kernel(const float* weight, const float* scales,
                                  const uint8_t* zero_points, uint8_t* packed,
                                  int rows, int cols, int group_size) {}
