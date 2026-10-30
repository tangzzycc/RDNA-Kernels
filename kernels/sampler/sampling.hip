#include <hip/hip_runtime.h>

#include <cstdint>

// logits：[R,V]，FP32，按行连续存储。
// token_ids：[R]，int64_t；并列最大值选择较小的 ID。
// rows = R；vocab_size = V = 151936。
__global__ void greedy_kernel(const float* logits, int64_t* token_ids, int rows,
                              int vocab_size) {}

// logits、filtered_logits：[R,V]，FP32，连续存储，独立缓冲区。
// top_k：[R]，int32_t，取值 0..V；0 表示关闭筛选。
// rows = R；vocab_size = V = 151936。
__global__ void top_k_kernel(const float* logits, const int32_t* top_k,
                             float* filtered_logits, int rows, int vocab_size) {
}

// probs、filtered_probs：[R,V]，FP32，连续存储，独立缓冲区。
// probs 每行非负且归一化；top_p：[R]，FP32，取值 (0,1]。
// rows = R；vocab_size = V = 151936。
__global__ void top_p_kernel(const float* probs, const float* top_p,
                             float* filtered_probs, int rows, int vocab_size) {}

// probs：[R,V]，FP32，按原词表顺序连续存储，每行总质量 > 0。
// uniforms：[R]，FP32，取值 [0,1)；token_ids：[R]，int64_t。
// rows = R；vocab_size = V = 151936。
__global__ void sample_kernel(const float* probs, const float* uniforms,
                              int64_t* token_ids, int rows, int vocab_size) {}

// logits：[R,V]，FP32，已应用所需惩罚，按行连续存储。
// temperatures、top_p、uniforms：[R]，FP32；temperatures >= 0。
// top_p 位于 (0,1]；uniforms 位于 [0,1)；top_k：[R]，int32_t，取值 0..V。
// token_ids：[R]，int64_t；rows = R；vocab_size = V = 151936。
__global__ void topk_topp_sample_kernel(
    const float* logits, const float* temperatures, const int32_t* top_k,
    const float* top_p, const float* uniforms, int64_t* token_ids, int rows,
    int vocab_size) {}
