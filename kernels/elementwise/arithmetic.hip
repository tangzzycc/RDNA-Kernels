#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <limits>
#include <cstddef>

// a、b、y：[M,5120]，BF16，连续存储，各自使用独立缓冲区。
// n = M * 5120，展平后的元素总数；M = B×S，也支持任意连续数组。
__global__ void add_kernel(const hip_bfloat16* a, const hip_bfloat16* b,
                           hip_bfloat16* y, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a[i] + b[i];
  }
}

// 提供 C ABI，供 Python 测试和独立程序调用；算子在指定流上异步执行。
extern "C" hipError_t launch_add(const hip_bfloat16* a, const hip_bfloat16* b,
                                 hip_bfloat16* y, size_t n,
                                 hipStream_t stream) {
  if (n == 0) {
    return hipSuccess;
  }
  constexpr unsigned int threads = 256;
  const size_t blocks = (n - 1) / threads + 1;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    return hipErrorInvalidValue;
  }
  add_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(a, b, y,
                                                                        n);
  return hipGetLastError();
}

// a、b、y：[M,25600]，BF16，各自使用独立的连续缓冲区。
// n = M * 25600，展平后的元素总数。
__global__ void mul_kernel(const hip_bfloat16* a, const hip_bfloat16* b,
                           hip_bfloat16* y, size_t n) {}
