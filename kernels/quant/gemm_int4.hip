#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

// a：[M,K]，BF16；packed_weight：[N,K/2]，uint8_t，行主序。
// scales：[N,K/group_size]，FP32；zero_points：同 shape，uint8_t。
// out：[M,N]，BF16，行主序；输入输出使用独立缓冲区。
// m、n、k：矩阵维度 M、N、K；group_size 默认 128。
// k 为偶数且可被正数 group_size 整除。
__global__ void gemm_int4_kernel(const hip_bfloat16* a,
                                 const uint8_t* packed_weight,
                                 const float* scales,
                                 const uint8_t* zero_points, hip_bfloat16* out,
                                 int m, int n, int k, int group_size) {}
