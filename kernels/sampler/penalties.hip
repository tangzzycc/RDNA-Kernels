#include <hip/hip_runtime.h>

#include <cstdint>

// logits：[R,V]，FP32，原地更新。
// prompt_seen：[R,V]，uint8_t；output_counts：[R,V]，int32_t，非负次数。
// repetition、frequency、presence：[R]，FP32；repetition > 0。
// 各矩阵按行连续存储；rows = R；vocab_size = V = 151936。
__global__ void penalties_kernel(float* logits, const uint8_t* prompt_seen,
                                 const int32_t* output_counts,
                                 const float* repetition,
                                 const float* frequency, const float* presence,
                                 int rows, int vocab_size) {}
