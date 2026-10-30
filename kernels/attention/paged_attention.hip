#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

// q、out：[B,Hq,S,D]，BF16，连续存储，独立缓冲区。
// k_cache、v_cache：[num_blocks,block_size,Hkv,D]，BF16，连续存储。
// block_table：[B,max_blocks]，int32_t，映射逻辑页号到有效物理页号。
// context_lens、query_start：[B]，int32_t，有效范围沿用 attention。
// 每行 block_table 至少包含 ceil(context_lens / block_size) 个有效页。
// batch_size = B；q_heads = Hq = 64；kv_heads = Hkv = 8。
// query_len = S；head_size = D = 128；scale = 1 / sqrt(D)。
__global__ void paged_attention_kernel(
    const hip_bfloat16* q, const hip_bfloat16* k_cache,
    const hip_bfloat16* v_cache, const int32_t* block_table,
    const int32_t* context_lens, const int32_t* query_start, hip_bfloat16* out,
    int batch_size, int q_heads, int kv_heads, int query_len, int head_size,
    int block_size, int max_blocks, float scale) {}
