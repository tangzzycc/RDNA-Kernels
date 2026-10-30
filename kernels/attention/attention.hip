#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

// q：[B,Hq,S,D]；k、v：[B,Hkv,L,D]；out：[B,Hq,S,D]，BF16。
// 张量各自连续存储，输入输出使用独立缓冲区。
// context_lens、query_start：[B]，int32_t，分别为有效 KV 长度和查询起点。
// 每个请求满足 0 <= query_start，query_start + query_len <= context_lens。
// context_lens <= kv_len；所有长度均为正数。
// batch_size = B；q_heads = Hq = 64；kv_heads = Hkv = 8。
// query_len = S；kv_len = L（容量）；head_size = D = 128。
// scale = 1 / sqrt(D)；causal mask 由长度与查询起点确定。
__global__ void attention_kernel(const hip_bfloat16* q, const hip_bfloat16* k,
                                 const hip_bfloat16* v, hip_bfloat16* out,
                                 const int32_t* context_lens,
                                 const int32_t* query_start, int batch_size,
                                 int q_heads, int kv_heads, int query_len,
                                 int kv_len, int head_size, float scale) {}
