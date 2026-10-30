#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

// weight：[N,K]，BF16，行主序。
// x：[K]；y：[N]，BF16，连续存储，各输入输出使用独立缓冲区。
// n、k：输出维度 N、输入维度 K。
__global__ void gemv_kernel(const hip_bfloat16* weight, const hip_bfloat16* x,
                            hip_bfloat16* y, int n, int k) {}
