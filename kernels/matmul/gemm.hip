#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

// a：[M,K]；b：[K,N]；c：[M,N]，BF16，行主序，独立缓冲区。
// m、n、k：矩阵维度 M、N、K；线性投影时 m = B×S。
__global__ void gemm_kernel(const hip_bfloat16* a, const hip_bfloat16* b,
                            hip_bfloat16* c, int m, int n, int k) {}
