#include <hip/hip_runtime.h>

#include <cstdint>

// logits、filtered_logits：[R,V]，FP32，连续存储，独立缓冲区。
// top_k：[R]，int32_t，取值 0..V；0 表示关闭筛选。
// rows = R；vocab_size = V = 151936。
__global__ void top_k_kernel(const float* logits, const int32_t* top_k,
                             float* filtered_logits, int rows, int vocab_size) {
}
