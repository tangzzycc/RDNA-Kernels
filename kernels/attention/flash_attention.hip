#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

// q：[B,Hq,S,D]；k、v：[B,Hkv,L,D]；out：[B,Hq,S,D]，BF16，连续存储。
// context_lens、query_start：[B]，int32_t；有效范围沿用 attention。
// batch_size = B；q_heads = Hq = 64；kv_heads = Hkv = 8。
// query_len = S；kv_len = L；head_size = D = 128；scale = 1 / sqrt(D)。
// 输入输出使用独立缓冲区。
__global__ void flash_attention_kernel(
    const hip_bfloat16* q, const hip_bfloat16* k, const hip_bfloat16* v,
    hip_bfloat16* out, const int32_t* context_lens, const int32_t* query_start,
    int batch_size, int q_heads, int kv_heads, int query_len, int kv_len,
    int head_size, float scale) {}
