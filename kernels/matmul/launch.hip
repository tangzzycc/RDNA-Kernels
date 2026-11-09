#include "sgemm.hip"

#include <cstdint>
#include <limits>

// V9 默认参数为 <BM, BN, BK, TM, TN, NUM_THREADS> = <128, 256, 8, 4, 4, 256>.
// gfx1151, FP32, M=N=K=5120 复测约 26.23 ms, V6 对照约 33.07 ms.
// 调优脚本通过 -D 覆盖参数, V6 固定使用 <128, 128, 8, 8, 8> 作为对照.
#ifndef SGEMM_BM
  #define SGEMM_BM 128
#endif
#ifndef SGEMM_BN
  #define SGEMM_BN 256
#endif
#ifndef SGEMM_BK
  #define SGEMM_BK 8
#endif
#ifndef SGEMM_TM
  #define SGEMM_TM 4
#endif
#ifndef SGEMM_TN
  #define SGEMM_TN 4
#endif
#ifndef SGEMM_NUM_THREADS
  #define SGEMM_NUM_THREADS 256
#endif

// V10 默认参数为 <BM, BN, BK, WM, WN, WN_ITERS, TM, TN, NUM_THREADS>.
// gfx1151, FP32, M=N=K=5120 参数搜索采用 <128, 128, 16, 32, 128, 2, 4, 4, 128>.
// 复测约 21.81 ms, 调优脚本通过 -D 覆盖参数, 初始配置由脚本显式固定作为对照.
#ifndef SGEMM_V10_BM
  #define SGEMM_V10_BM 128
#endif
#ifndef SGEMM_V10_BN
  #define SGEMM_V10_BN 128
#endif
#ifndef SGEMM_V10_BK
  #define SGEMM_V10_BK 16
#endif
#ifndef SGEMM_V10_WM
  #define SGEMM_V10_WM 32
#endif
#ifndef SGEMM_V10_WN
  #define SGEMM_V10_WN 128
#endif
#ifndef SGEMM_V10_WN_ITERS
  #define SGEMM_V10_WN_ITERS 2
#endif
#ifndef SGEMM_V10_TM
  #define SGEMM_V10_TM 4
#endif
#ifndef SGEMM_V10_TN
  #define SGEMM_V10_TN 4
#endif
#ifndef SGEMM_V10_NUM_THREADS
  #define SGEMM_V10_NUM_THREADS 128
#endif

namespace sgemm_launch {

constexpr int BM = SGEMM_BM, BN = SGEMM_BN, BK = SGEMM_BK;
constexpr int TM = SGEMM_TM, TN = SGEMM_TN, NUM_THREADS = SGEMM_NUM_THREADS;

constexpr int V10_BM = SGEMM_V10_BM, V10_BN = SGEMM_V10_BN, V10_BK = SGEMM_V10_BK;
constexpr int V10_WM = SGEMM_V10_WM, V10_WN = SGEMM_V10_WN, V10_WN_ITERS = SGEMM_V10_WN_ITERS;
constexpr int V10_TM = SGEMM_V10_TM, V10_TN = SGEMM_V10_TN, V10_NUM_THREADS = SGEMM_V10_NUM_THREADS;

// 汇总启动尺寸和核函数地址, 供统一 launch 与 profiler 共用.
struct Config {
  int block_m, block_n, block_k;
  int thread_m, thread_n, threads;
  const void* kernel;
};

inline Config configuration(int version) {
  switch (version) {
    case 1:
      return {32, 32, 1, 1, 1, 1024, reinterpret_cast<const void*>(sgemm_v1_kernel)};
    case 2:
      return {32, 32, 1, 1, 1, 1024, reinterpret_cast<const void*>(sgemm_v2_kernel<32>)};
    case 3:
      return {32, 32, 32, 1, 1, 1024, reinterpret_cast<const void*>(sgemm_v3_kernel<32>)};
    case 4:
      return {64, 64, 8, 8, 1, 512, reinterpret_cast<const void*>(sgemm_v4_kernel<64, 64, 8, 8>)};
    case 5:
      return {128,
              128,
              8,
              8,
              8,
              256,
              reinterpret_cast<const void*>(sgemm_v5_kernel<128, 128, 8, 8, 8>)};
    case 6:
      return {128,
              128,
              8,
              8,
              8,
              256,
              reinterpret_cast<const void*>(sgemm_v6_kernel<128, 128, 8, 8, 8>)};
    case 9:
      return {BM,
              BN,
              BK,
              TM,
              TN,
              NUM_THREADS,
              reinterpret_cast<const void*>(sgemm_v9_kernel<BM, BN, BK, TM, TN, NUM_THREADS>)};
    case 10:
      return {V10_BM,
              V10_BN,
              V10_BK,
              V10_TM,
              V10_TN,
              V10_NUM_THREADS,
              reinterpret_cast<const void*>(
                  sgemm_v10_kernel<V10_BM, V10_BN, V10_BK, V10_WM, V10_WN, V10_WN_ITERS, V10_TM,
                                   V10_TN, V10_NUM_THREADS>)};
    default:
      return {};
  }
}

}  // namespace sgemm_launch

// A: [M, K], B: [K, N], C: [M, N], FP32, 行主序且使用独立缓冲区.
// M, N 非负, K 为正数, 空输出直接返回; 矩阵元素数须在 int 索引范围内.
// version: 1, 2, 3, 4, 5, 6, 9, 10, 对应 sgemm_vN_kernel.
// V1-V2 处理 M, N 边界; V3 起要求完整 tile; V6, V9, V10 使用 16 字节对齐的 float4.
// 提供 C ABI, 在指定 stream 上异步启动, C 直接写回 A * B.
extern "C" hipError_t launch_sgemm(const float* A, const float* B, float* C, int M, int N, int K,
                                   int version, hipStream_t stream) {
  // 检查 shape, 版本与指针, 避免把不支持的分块交给 kernel.
  using namespace sgemm_launch;
  const Config config = configuration(version);
  if (M < 0 || N < 0 || K <= 0 || config.kernel == nullptr) {
    return hipErrorInvalidValue;
  }
  if (M == 0 || N == 0) {
    return hipSuccess;
  }
  if (A == nullptr || B == nullptr || C == nullptr) {
    return hipErrorInvalidValue;
  }
  if (int64_t(M) * K > std::numeric_limits<int>::max() ||
      int64_t(K) * N > std::numeric_limits<int>::max() ||
      int64_t(M) * N > std::numeric_limits<int>::max()) {
    return hipErrorInvalidValue;
  }
  if (version >= 3 && (M % config.block_m || N % config.block_n || K % config.block_k)) {
    return hipErrorInvalidValue;
  }
  if (version >= 6 && (!is_aligned(A, 16) || !is_aligned(B, 16) || !is_aligned(C, 16))) {
    return hipErrorInvalidValue;
  }

  // V1-V3 的 grid.x 覆盖 row, V4 起 grid.x 覆盖 col, 与各版本的 block 映射保持一致.
  const dim3 grid = version <= 3 ? dim3((M - 1) / config.block_m + 1, (N - 1) / config.block_n + 1)
                                 : dim3(N / config.block_n, M / config.block_m);
  const dim3 block = version == 1 ? dim3(32, 32) : dim3(config.threads);

  // 各版本使用同一组矩阵参数, 模板配置集中在本文件中.
  switch (version) {
    case 1:
      sgemm_v1_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
      break;
    case 2:
      sgemm_v2_kernel<32><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
      break;
    case 3:
      sgemm_v3_kernel<32><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
      break;
    case 4:
      sgemm_v4_kernel<64, 64, 8, 8><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
      break;
    case 5:
      sgemm_v5_kernel<128, 128, 8, 8, 8><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
      break;
    case 6:
      sgemm_v6_kernel<128, 128, 8, 8, 8><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
      break;
    case 9:
      sgemm_v9_kernel<BM, BN, BK, TM, TN, NUM_THREADS>
          <<<grid, block, 0, stream>>>(A, B, C, M, N, K);
      break;
    case 10:
      sgemm_v10_kernel<V10_BM, V10_BN, V10_BK, V10_WM, V10_WN, V10_WN_ITERS, V10_TM, V10_TN,
                       V10_NUM_THREADS><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
      break;
  }
  return hipGetLastError();
}

