#include <hip/hip_runtime.h>

// logits：[R,V]，FP32，按行连续存储，原地更新。
// temperatures：[R]，FP32，每行温度 > 0。
// rows = R；vocab_size = V = 151936。
__global__ void temperature_kernel(float* logits, const float* temperatures,
                                   int rows, int vocab_size) {}
