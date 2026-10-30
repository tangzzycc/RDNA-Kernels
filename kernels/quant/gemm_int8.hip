#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

// a：[M,K]；weight：[N,K]，INT8，行主序。
// a_scales：[M]；weight_scales：[N]，FP32。
// out：[M,N]，BF16，行主序；输入输出使用独立缓冲区。
// m、n、k：矩阵维度 M、N、K，线性投影尺寸沿用 gemm。
__global__ void gemm_int8_kernel(const int8_t* a, const int8_t* weight,
                                 const float* a_scales,
                                 const float* weight_scales, hip_bfloat16* out,
                                 int m, int n, int k) {}
