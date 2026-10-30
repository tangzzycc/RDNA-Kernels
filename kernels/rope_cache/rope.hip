#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

// x：[T,H,D]，BF16，token/head/维度依次连续，原地写回。
// positions：[T]，int64_t，取值对应 cos/sin 的位置行。
// cos、sin：[P,D/2]，BF16，按行连续存储。
// tokens = T = B×S；heads = H = 64 或 8；head_size = D = 128。
__global__ void rope_kernel(hip_bfloat16* x, const int64_t* positions,
                            const hip_bfloat16* cos, const hip_bfloat16* sin,
                            int tokens, int heads, int head_size) {}
