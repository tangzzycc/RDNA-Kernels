#include <hip/hip_runtime.h>

// probs、filtered_probs：[R,V]，FP32，连续存储，独立缓冲区。
// probs 每行非负且归一化；top_p：[R]，FP32，取值 (0,1]。
// rows = R；vocab_size = V = 151936。
__global__ void top_p_kernel(const float* probs, const float* top_p,
                             float* filtered_probs, int rows, int vocab_size) {}
