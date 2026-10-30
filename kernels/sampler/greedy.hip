#include <hip/hip_runtime.h>

#include <cstdint>

// logits：[R,V]，FP32，按行连续存储。
// token_ids：[R]，int64_t；并列最大值选择较小的 ID。
// rows = R；vocab_size = V = 151936。
__global__ void greedy_kernel(const float* logits, int64_t* token_ids, int rows,
                              int vocab_size) {}
