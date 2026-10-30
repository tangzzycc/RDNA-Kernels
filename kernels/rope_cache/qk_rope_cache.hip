#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

// q：[T,Hq,D]，BF16，连续存储，原地写回旋转结果。
// k、v：[T,Hkv,D]，BF16，连续存储。
// k_cache、v_cache：[C,Hkv,D]，BF16，按槽位/head/维度连续存储。
// positions：[T]，int64_t，用于查找 RoPE 系数。
// slot_mapping：[T]，int64_t，写入槽位位于 [0,C)，本次写入互不重复。
// cos、sin：[P,D/2]，BF16；各输入与缓存使用独立缓冲区。
// tokens = T = B×S；q_heads = Hq = 64；kv_heads = Hkv = 8。
// head_size = D = 128。
__global__ void qk_rope_cache_kernel(
    hip_bfloat16* q, const hip_bfloat16* k, const hip_bfloat16* v,
    const int64_t* positions, const int64_t* slot_mapping,
    const hip_bfloat16* cos, const hip_bfloat16* sin, hip_bfloat16* k_cache,
    hip_bfloat16* v_cache, int tokens, int q_heads, int kv_heads,
    int head_size) {}
