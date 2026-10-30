#include <hip/hip_runtime.h>

#include <cstdint>

// logits：[R,V]，FP32，已应用所需惩罚，按行连续存储。
// temperatures、top_p、uniforms：[R]，FP32；temperatures >= 0。
// top_p 位于 (0,1]；uniforms 位于 [0,1)；top_k：[R]，int32_t，取值 0..V。
// token_ids：[R]，int64_t；rows = R；vocab_size = V = 151936。
__global__ void topk_topp_sample_kernel(
    const float* logits, const float* temperatures, const int32_t* top_k,
    const float* top_p, const float* uniforms, int64_t* token_ids, int rows,
    int vocab_size) {}
