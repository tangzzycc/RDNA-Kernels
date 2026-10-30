#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

// k、v：[T,Hkv,D]，BF16；k 已完成 RoPE。
// k_cache、v_cache：[C,Hkv,D]，BF16，按槽位/head/维度连续存储。
// slot_mapping：[T]，int64_t，槽位位于 [0,C)，本次写入互不重复。
// tokens = T；kv_heads = Hkv = 8；head_size = D = 128。
__global__ void cache_write_kernel(const hip_bfloat16* k, const hip_bfloat16* v,
                                   const int64_t* slot_mapping,
                                   hip_bfloat16* k_cache, hip_bfloat16* v_cache,
                                   int tokens, int kv_heads, int head_size) {}
