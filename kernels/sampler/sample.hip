#include <hip/hip_runtime.h>

#include <cstdint>

// probs：[R,V]，FP32，按原词表顺序连续存储，每行总质量 > 0。
// uniforms：[R]，FP32，取值 [0,1)；token_ids：[R]，int64_t。
// rows = R；vocab_size = V = 151936。
__global__ void sample_kernel(const float* probs, const float* uniforms,
                              int64_t* token_ids, int rows, int vocab_size) {}
