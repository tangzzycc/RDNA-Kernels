#include "../../common/hip/common.h"
#include "launch.hip"

#include <algorithm>
#include <cmath>
#include <random>
#include <type_traits>
#include <vector>

namespace {

using namespace sgemm_launch;

template <typename T, bool OPT = false, bool WMMA = false>
int run_gemm(int version, int argc, char** argv) {
  constexpr bool is_half = std::is_same_v<T, __half>;
  bool profile_only = argc > 0 && std::strcmp(argv[0], "--profile") == 0;
  if (profile_only) {
    // 只启动一次目标 kernel, 供 rocprof 采集, 跳过参考计算和计时循环.
    --argc;
    ++argv;
  }
  if (argc != 0 && argc != 3 && argc != 6) {
    std::fprintf(stderr, "Expected [M N K [warmup iterations batches]].\n");
    return 1;
  }
  const int M = argc >= 3 ? std::atoi(argv[0]) : 8192;
  const int N = argc >= 3 ? std::atoi(argv[1]) : 8192;
  const int K = argc >= 3 ? std::atoi(argv[2]) : 8192;
  const int warmup = argc == 6 ? std::atoi(argv[3]) : 1;
  const int iterations = argc == 6 ? std::atoi(argv[4]) : 1;
  const int batches = argc == 6 ? std::atoi(argv[5]) : 3;
  const Config config = [&] {
    if constexpr (WMMA) {
      const auto h = hgemm_launch::configuration(version);
      // WMMA 的计算分块按 wave 记录, SGEMM 则按单线程记录.
      return Config{h.block_m, h.block_n, h.block_k, h.wave_m, h.wave_n, h.threads, h.kernel};
    } else {
      return OPT ? sgemm_opt_launch::configuration<T>() : configuration(version);
    }
  }();
  const int tile_m = config.block_m, tile_n = config.block_n, tile_k = config.block_k;
  const int compute_m = config.thread_m, compute_n = config.thread_n, num_threads = config.threads;
  const bool needs_tiles = !WMMA || version <= 3;
  if (M <= 0 || N <= 0 || K <= 0 || (needs_tiles && (M % tile_m || N % tile_n || K % tile_k)) ||
      (!WMMA && (M % 64 || N % 64 || K % 8)) || int64_t(M) * K > std::numeric_limits<int>::max() ||
      int64_t(K) * N > std::numeric_limits<int>::max() ||
      int64_t(M) * N > std::numeric_limits<int>::max() || warmup < 0 || iterations <= 0 ||
      batches <= 0) {
    std::fprintf(stderr, "Unsupported shape or timing counts.\n");
    return 1;
  }

  std::vector<T> A(M * K), B(K * N);
  std::mt19937 generator(2026);
  std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
  // WMMA 用随机正负一核对完整输出, 避免将 FP16 累加误差误判为索引错误.
  const auto sample = [&] {
    if constexpr (WMMA)
      return T(generator() % 2 ? 1.0f : -1.0f);
    else
      return T(distribution(generator));
  };
  for (auto& value : A) value = sample();
  // HGEMM 的 B 按 col*K+k 存储, SGEMM 的 B 按 k*N+col 存储.
  for (auto& value : B) value = sample();
  T *device_A, *device_B, *device_C;
  HIP_CHECK(hipMalloc(&device_A, A.size() * sizeof(T)));
  HIP_CHECK(hipMalloc(&device_B, B.size() * sizeof(T)));
  HIP_CHECK(hipMalloc(&device_C, M * N * sizeof(T)));
  HIP_CHECK(hipMemcpy(device_A, A.data(), A.size() * sizeof(T), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device_B, B.data(), B.size() * sizeof(T), hipMemcpyHostToDevice));
  const auto launch = [&] {
    if constexpr (WMMA) {
      return launch_hgemm(device_A, device_B, device_C, M, N, K, version, nullptr);
    } else if constexpr (OPT && is_half) {
      return launch_sgemm_opt_fp16(device_A, device_B, device_C, M, N, K, nullptr);
    } else if constexpr (OPT) {
      return launch_sgemm_opt(device_A, device_B, device_C, M, N, K, nullptr);
    } else {
      return launch_sgemm(device_A, device_B, device_C, M, N, K, version, nullptr);
    }
  };

  if (profile_only) {
    HIP_CHECK(launch());
    HIP_CHECK(hipDeviceSynchronize());
  } else {
    // 用已验证的 SGEMM V4 核对每个输出, 计时只包含目标 kernel.
    std::vector<T> expected(M * N), actual(M * N);
    if constexpr (is_half) {
      // 先把实际 FP16 输入转回 FP32, 用 SGEMM V4 计算参考, 再舍入为 FP16.
      // V4 参考 kernel 使用行主序, HGEMM 的 B 转成行主序参考, C 再按列主序核对.
      // 非整齐 shape 给参考矩阵补零后提取有效区域.
      const int ref_m = (M + 63) / 64 * 64, ref_n = (N + 63) / 64 * 64;
      const int ref_k = (K + 7) / 8 * 8;
      std::vector<float> reference_A(ref_m * ref_k), reference_B(ref_k * ref_n);
      std::vector<float> reference_C(ref_m * ref_n);
      for (int row = 0; row < M; ++row)
        for (int k = 0; k < K; ++k) reference_A[row * ref_k + k] = float(A[row * K + k]);
      for (int k = 0; k < K; ++k)
        for (int col = 0; col < N; ++col) {
          const int input = WMMA ? col * K + k : k * N + col;
          reference_B[k * ref_n + col] = float(B[input]);
        }
      float *ref_A, *ref_B, *ref_C;
      HIP_CHECK(hipMalloc(&ref_A, reference_A.size() * sizeof(float)));
      HIP_CHECK(hipMalloc(&ref_B, reference_B.size() * sizeof(float)));
      HIP_CHECK(hipMalloc(&ref_C, reference_C.size() * sizeof(float)));
      HIP_CHECK(hipMemcpy(ref_A, reference_A.data(), reference_A.size() * sizeof(float),
                          hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(ref_B, reference_B.data(), reference_B.size() * sizeof(float),
                          hipMemcpyHostToDevice));
      HIP_CHECK(launch_sgemm(ref_A, ref_B, ref_C, ref_m, ref_n, ref_k, 4, nullptr));
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemcpy(reference_C.data(), ref_C, reference_C.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
      for (int row = 0; row < M; ++row)
        for (int col = 0; col < N; ++col)
          expected[WMMA ? col * M + row : row * N + col] =
              __float2half_rn(reference_C[row * ref_n + col]);
      HIP_CHECK(hipFree(ref_A));
      HIP_CHECK(hipFree(ref_B));
      HIP_CHECK(hipFree(ref_C));
    } else {
      HIP_CHECK(launch_sgemm(device_A, device_B, device_C, M, N, K, 4, nullptr));
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(
          hipMemcpy(expected.data(), device_C, expected.size() * sizeof(T), hipMemcpyDeviceToHost));
    }
    HIP_CHECK(hipMemset(device_C, 0xff, actual.size() * sizeof(T)));
    HIP_CHECK(launch());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(actual.data(), device_C, actual.size() * sizeof(T), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < actual.size(); ++i) {
      if (!std::isfinite(float(actual[i])) ||
          std::abs(float(actual[i]) - float(expected[i])) > (WMMA ? 1e-3f : 0.0f)) {
        std::fprintf(stderr, "Correctness failed at %zu: actual=%g, reference=%g\n", i,
                     float(actual[i]), float(expected[i]));
        return 1;
      }
    }

    hipFuncAttributes attributes;
    HIP_CHECK(hipFuncGetAttributes(&attributes, config.kernel));
    for (int i = 0; i < warmup; ++i) {
      HIP_CHECK(launch());
    }
    HIP_CHECK(hipDeviceSynchronize());
    double mean_ms = 0.0;
    for (int i = 0; i < batches; ++i) {
      mean_ms += benchmark_kernel(launch, 0, iterations) / (1000.0 * batches);
    }
    if constexpr (WMMA) {
      std::printf("HGEMM V%d: correctness passed (%zu outputs), mean=%.6f ms, %.6f TFLOP/s\n",
                  version, actual.size(), mean_ms, 2.0 * M * N * K / (mean_ms * 1e9));
      std::printf(
          "accumulation=FP16 WMMA, wave tile=[%d,%d], threads=%d, VGPR=%d, "
          "LDS=%zu bytes, scratch=%zu bytes\n",
          compute_m, compute_n, num_threads, attributes.numRegs, attributes.sharedSizeBytes,
          attributes.localSizeBytes);
    } else if constexpr (OPT) {
      std::printf("SGEMM Opt (%s): correctness passed (%zu outputs), mean=%.6f ms, %.6f TFLOP/s\n",
                  is_half ? "FP16" : "FP32", actual.size(), mean_ms,
                  2.0 * M * N * K / (mean_ms * 1e9));
    } else {
      std::printf("V%d: correctness passed (%zu outputs), mean=%.6f ms, %.6f TFLOP/s\n", version,
                  actual.size(), mean_ms, 2.0 * M * N * K / (mean_ms * 1e9));
    }
    if constexpr (OPT) {
      std::printf(
          "accumulation=FP32 FMA, wave tile=[%d,%d], WAVE_N_ITERS=%d, "
          "threads=%d, VGPR=%d, LDS=%zu bytes, scratch=%zu bytes\n",
          sgemm_opt_launch::WM, sgemm_opt_launch::WN, sgemm_opt_launch::WN_ITERS, num_threads,
          attributes.numRegs, attributes.sharedSizeBytes, attributes.localSizeBytes);
    }
    if (!WMMA && version == 10) {
      std::printf(
          "Wave tile=[%d,%d], WAVE_N_ITERS=%d, threads=%d, VGPR=%d, LDS=%zu bytes, "
          "scratch=%zu bytes\n",
          V10_WM, V10_WN, V10_WN_ITERS, num_threads, attributes.numRegs, attributes.sharedSizeBytes,
          attributes.localSizeBytes);
      // 单独保留完整 V10 参数记录, 原 RESULT 格式继续供 V9 脚本使用.
      std::printf("V10_RESULT\tV10\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.6f\t%.6f\t%d\t%zu\t%zu\n",
                  V10_BM, V10_BN, V10_BK, V10_WM, V10_WN, V10_WN_ITERS, V10_TM, V10_TN, num_threads,
                  mean_ms, 2.0 * M * N * K / (mean_ms * 1e9), attributes.numRegs,
                  attributes.sharedSizeBytes, attributes.localSizeBytes);
    }
    // RESULT 为制表符分隔记录, 供脚本排序与汇总.
    // 字段: version, BM, BN, BK, TM, TN, threads, ms, TFLOP/s, VGPR, LDS bytes, scratch bytes.
    // HGEMM_RESULT 的 TM/TN 位置记录 WAVE_M/WAVE_N, 表示整个 wave 的输出分块.
    std::printf("%s\tV%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.6f\t%.6f\t%d\t%zu\t%zu\n",
                WMMA  ? "HGEMM_RESULT"
                : OPT ? (is_half ? "OPT_RESULT_FP16" : "OPT_RESULT_FP32")
                      : "RESULT",
                version, tile_m, tile_n, tile_k, compute_m, compute_n, num_threads, mean_ms,
                2.0 * M * N * K / (mean_ms * 1e9), attributes.numRegs, attributes.sharedSizeBytes,
                attributes.localSizeBytes);
  }
  HIP_CHECK(hipFree(device_A));
  HIP_CHECK(hipFree(device_B));
  HIP_CHECK(hipFree(device_C));
  return 0;
}

template <int VERSION, typename T = float, bool OPT = false, bool WMMA = false>
int run_version(int argc, char** argv) {
  return run_gemm<T, OPT, WMMA>(VERSION, argc, argv);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "--device-info") == 0) {
    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    std::printf("%zu\t%d\t%s\n", properties.sharedMemPerBlock, properties.maxThreadsPerBlock,
                properties.gcnArchName);
    return 0;
  }
  const ProfileWorkload workloads[] = {
      {"sgemm_v1", run_version<1>},
      {"sgemm_v2", run_version<2>},
      {"sgemm_v3", run_version<3>},
      {"sgemm_v4", run_version<4>},
      {"sgemm_v5", run_version<5>},
      {"sgemm_v6", run_version<6>},
      {"sgemm_v9", run_version<9>},
      {"sgemm_v10", run_version<10>},
      {"sgemm_opt", run_version<11, float, true>},
      {"sgemm_opt_fp16", run_version<12, __half, true>},
      {"hgemm_v1", run_version<1, __half, false, true>},
      {"hgemm_v2", run_version<2, __half, false, true>},
      {"hgemm_v3", run_version<3, __half, false, true>},
      {"hgemm_v4", run_version<4, __half, false, true>},
      {"hgemm_v5", run_version<5, __half, false, true>},
      {"hgemm_v6", run_version<6, __half, false, true>},
      {"hgemm_v7", run_version<7, __half, false, true>},
      {"hgemm_v8", run_version<8, __half, false, true>},

  };
  return run_profiler(argc, argv, workloads);
}
