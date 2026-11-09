#include "elementwise.hip"

#include <limits>

// a, b, y: [n], BF16, 连续存储, 形状相同且使用独立缓冲区.
// 输入 [M, N] 可展平为 [n], n = M * N; 输入 [B, S, N] 时 M = B * S.
// n 为元素总数, 无需整除向量宽度; n == 0 时允许空指针并直接返回成功.
// 提供 C ABI, 供 Python 测试和独立程序调用; 在指定 stream 上异步执行.
extern "C" hipError_t launch_add(const hip_bfloat16* a, const hip_bfloat16* b, hip_bfloat16* y,
                                 size_t n, hipStream_t stream) {
  // step 1: 处理 n == 0, 检查非空输入的 a, b 和 y 指针.
  if (n == 0) {
    return hipSuccess;
  }
  if (a == nullptr || b == nullptr || y == nullptr) {
    return hipErrorInvalidValue;
  }
  // step 2: 从 BF16 的 8 元素向量开始, 根据长度和三个指针的对齐逐级缩小.
  int vec_size = 16 / sizeof(hip_bfloat16);
  while (vec_size > 1 &&
         (static_cast<size_t>(vec_size) > n || !is_aligned(a, vec_size * sizeof(hip_bfloat16)) ||
          !is_aligned(b, vec_size * sizeof(hip_bfloat16)) ||
          !is_aligned(y, vec_size * sizeof(hip_bfloat16)))) {
    vec_size /= 2;
  }
  // step 3: 用完整向量数加尾部任务计算 grid, 每个 block 使用 256 个线程.
  constexpr unsigned int threads = 256;
  const size_t tasks = n / vec_size + (n % vec_size != 0);
  const size_t blocks = (tasks - 1) / threads + 1;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    return hipErrorInvalidValue;
  }
  // step 4: 将 vec_size 分发为模板 VEC_SIZE, 异步启动 kernel 并返回启动状态.
  const dim3 grid(static_cast<unsigned int>(blocks)), block(threads);
  HIP_DISPATCH_VEC_SIZE(vec_size, add_kernel<VEC_SIZE><<<grid, block, 0, stream>>>(a, b, y, n));
  return hipGetLastError();
}

// a, b, y: [n], BF16, 连续存储, 形状相同且使用独立缓冲区.
// 输入 [M, N] 可展平为 [n], n = M * N; 输入 [B, S, N] 时 M = B * S.
// n 为元素总数, 无需整除向量宽度; n == 0 时允许空指针并直接返回成功.
// 提供 C ABI, 供 Python 测试和独立程序调用; 在指定 stream 上异步执行.
extern "C" hipError_t launch_mul(const hip_bfloat16* a, const hip_bfloat16* b, hip_bfloat16* y,
                                 size_t n, hipStream_t stream) {
  // step 1: 处理 n == 0, 检查非空输入的 a, b 和 y 指针.
  if (n == 0) {
    return hipSuccess;
  }
  if (a == nullptr || b == nullptr || y == nullptr) {
    return hipErrorInvalidValue;
  }
  // step 2: 从 BF16 的 8 元素向量开始, 根据长度和三个指针的对齐逐级缩小.
  int vec_size = 16 / sizeof(hip_bfloat16);
  while (vec_size > 1 &&
         (static_cast<size_t>(vec_size) > n || !is_aligned(a, vec_size * sizeof(hip_bfloat16)) ||
          !is_aligned(b, vec_size * sizeof(hip_bfloat16)) ||
          !is_aligned(y, vec_size * sizeof(hip_bfloat16)))) {
    vec_size /= 2;
  }
  // step 3: 用完整向量数加尾部任务计算 grid, 每个 block 使用 256 个线程.
  constexpr unsigned int threads = 256;
  const size_t tasks = n / vec_size + (n % vec_size != 0);
  const size_t blocks = (tasks - 1) / threads + 1;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    return hipErrorInvalidValue;
  }
  // step 4: 将 vec_size 分发为模板 VEC_SIZE, 异步启动 kernel 并返回启动状态.
  const dim3 grid(static_cast<unsigned int>(blocks)), block(threads);
  HIP_DISPATCH_VEC_SIZE(vec_size, mul_kernel<VEC_SIZE><<<grid, block, 0, stream>>>(a, b, y, n));
  return hipGetLastError();
}

// x, y: [n], BF16, 连续存储且使用独立缓冲区.
// 输入 [M, N] 可展平为 [n], n = M * N; 输入 [B, S, N] 时 M = B * S.
// n 为元素总数, 无需整除向量宽度; n == 0 时允许空指针并直接返回成功.
// 提供 C ABI, 供 Python 测试和独立程序调用; 在指定 stream 上异步执行.
extern "C" hipError_t launch_silu(const hip_bfloat16* x, hip_bfloat16* y, size_t n,
                                  hipStream_t stream) {
  // step 1: 处理 n == 0, 检查非空输入的 x 和 y 指针.
  if (n == 0) {
    return hipSuccess;
  }
  if (x == nullptr || y == nullptr) {
    return hipErrorInvalidValue;
  }
  // step 2: 从 BF16 的 8 元素向量开始, 根据长度和两个指针的对齐逐级缩小.
  int vec_size = 16 / sizeof(hip_bfloat16);
  while (vec_size > 1 &&
         (static_cast<size_t>(vec_size) > n || !is_aligned(x, vec_size * sizeof(hip_bfloat16)) ||
          !is_aligned(y, vec_size * sizeof(hip_bfloat16)))) {
    vec_size /= 2;
  }
  // step 3: 用完整向量数加尾部任务计算 grid, 每个 block 使用 256 个线程.
  constexpr unsigned int threads = 256;
  const size_t tasks = n / vec_size + (n % vec_size != 0);
  const size_t blocks = (tasks - 1) / threads + 1;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    return hipErrorInvalidValue;
  }
  // step 4: 将 vec_size 分发为模板 VEC_SIZE, 异步启动 kernel 并返回启动状态.
  const dim3 grid(static_cast<unsigned int>(blocks)), block(threads);
  HIP_DISPATCH_VEC_SIZE(vec_size, silu_kernel<VEC_SIZE><<<grid, block, 0, stream>>>(x, y, n));
  return hipGetLastError();
}

// gate, up, y: [n], BF16, 连续存储且使用独立缓冲区.
// 输入 [M, N] 可展平为 [n], n = M * N; 两路输入和输出形状相同.
// vLLM 使用拼接输入 [M, 2 * N], 本接口使用独立的 gate 和 up 缓冲区.
// n 为元素总数, 无需整除向量宽度; n == 0 时允许空指针并直接返回成功.
// 提供 C ABI, 供 Python 测试和独立程序调用; 在指定 stream 上异步执行.
extern "C" hipError_t launch_silu_mul(const hip_bfloat16* gate, const hip_bfloat16* up,
                                      hip_bfloat16* y, size_t n, hipStream_t stream) {
  // step 1: 处理 n == 0, 检查非空输入的 gate, up 和 y 指针.
  if (n == 0) {
    return hipSuccess;
  }
  if (gate == nullptr || up == nullptr || y == nullptr) {
    return hipErrorInvalidValue;
  }
  // step 2: 从 BF16 的 8 元素向量开始, 根据长度和三个指针的对齐逐级缩小.
  int vec_size = 16 / sizeof(hip_bfloat16);
  while (vec_size > 1 &&
         (static_cast<size_t>(vec_size) > n || !is_aligned(gate, vec_size * sizeof(hip_bfloat16)) ||
          !is_aligned(up, vec_size * sizeof(hip_bfloat16)) ||
          !is_aligned(y, vec_size * sizeof(hip_bfloat16)))) {
    vec_size /= 2;
  }
  // step 3: 用完整向量数加尾部任务计算 grid, 每个 block 使用 256 个线程.
  constexpr unsigned int threads = 256;
  const size_t tasks = n / vec_size + (n % vec_size != 0);
  const size_t blocks = (tasks - 1) / threads + 1;
  if (blocks > std::numeric_limits<unsigned int>::max()) {
    return hipErrorInvalidValue;
  }
  // step 4: 将 vec_size 分发为模板 VEC_SIZE, 异步启动 kernel 并返回启动状态.
  const dim3 grid(static_cast<unsigned int>(blocks)), block(threads);
  HIP_DISPATCH_VEC_SIZE(vec_size,
                        silu_mul_kernel<VEC_SIZE><<<grid, block, 0, stream>>>(gate, up, y, n));
  return hipGetLastError();
}

// x, y: [M, N], BF16, 按行连续存储且使用独立缓冲区.
// rows = M, cols = N, 每个 block 处理一行, 沿最后一维归一化.
// 使用 FP32 计算最大值和指数和, 最终转换为 BF16 输出.
// 每行至少一个有限值; 非有限输入仅支持 -inf 掩码.
// 提供 C ABI, 供 Python 测试和独立程序调用; 在指定 stream 上异步执行.
extern "C" hipError_t launch_softmax(const hip_bfloat16* x, hip_bfloat16* y, int rows, int cols,
                                     hipStream_t stream) {
  // step 1: 检查 shape, 空输入直接返回, 非空输入检查指针.
  if (rows < 0 || cols <= 0) {
    return hipErrorInvalidValue;
  }
  if (rows == 0) {
    return hipSuccess;
  }
  if (x == nullptr || y == nullptr) {
    return hipErrorInvalidValue;
  }

  // step 2: 选择线程数, 长行由各线程跨步处理.
  int block_size = 32;
  while (block_size < cols && block_size < 256) {
    block_size *= 2;
  }
  const dim3 grid(static_cast<unsigned int>(rows));
  const dim3 block(static_cast<unsigned int>(block_size));

  // step 3: 将运行时 block_size 分发为模板 BLOCK_SIZE.
  HIP_DISPATCH_BLOCK_SIZE(block_size,
                          softmax_kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(x, y, cols));
  return hipGetLastError();
}
