#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

// q：[T,Hq,D]；k：[T,Hkv,D]，BF16，连续存储，独立缓冲区，原地写回。
// positions：[T]，int64_t；cos、sin：[P,D/2]，BF16。
// tokens = T = B×S；q_heads = Hq = 64；kv_heads = Hkv = 8。
// head_size = D = 128。
__global__ void qk_rope_kernel(hip_bfloat16* q, hip_bfloat16* k,
                               const int64_t* positions,
                               const hip_bfloat16* cos, const hip_bfloat16* sin,
                               int tokens, int q_heads, int kv_heads,
                               int head_size) {}
