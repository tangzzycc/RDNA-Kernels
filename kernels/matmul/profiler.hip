#include "../../common/hip/common.h"
#include "launch.hip"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace {

using namespace sgemm_launch;

int run_sgemm(int version, int argc, char** argv) {
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
  const int M = argc >= 3 ? std::atoi(argv[0]) : 5120;
  const int N = argc >= 3 ? std::atoi(argv[1]) : 5120;
  const int K = argc >= 3 ? std::atoi(argv[2]) : 5120;
  const int warmup = argc == 6 ? std::atoi(argv[3]) : 3;
  const int iterations = argc == 6 ? std::atoi(argv[4]) : 3;
  const int batches = argc == 6 ? std::atoi(argv[5]) : 3;
  const Config config = configuration(version);
  const int tile_m = config.block_m, tile_n = config.block_n, tile_k = config.block_k;
  const int thread_m = config.thread_m, thread_n = config.thread_n, num_threads = config.threads;
  if (M <= 0 || N <= 0 || K <= 0 || M % tile_m || N % tile_n || K % tile_k || M % 64 || N % 64 ||
      K % 8 || warmup < 0 || iterations <= 0 || batches <= 0) {
    std::fprintf(stderr, "Shape must cover complete tiles; timing counts must be positive.\n");
    return 1;
  }

  std::vector<float> A(M * K), B(K * N);
  std::mt19937 generator(2026);
  std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
  for (auto& value : A) value = distribution(generator);
  for (auto& value : B) value = distribution(generator);
  float *device_A, *device_B, *device_C;
  HIP_CHECK(hipMalloc(&device_A, A.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&device_B, B.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&device_C, M * N * sizeof(float)));
  HIP_CHECK(hipMemcpy(device_A, A.data(), A.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device_B, B.data(), B.size() * sizeof(float), hipMemcpyHostToDevice));

  if (profile_only) {
    HIP_CHECK(launch_sgemm(device_A, device_B, device_C, M, N, K, version, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
  } else {
    // 与已验证的 V4 逐元素比较, 两者均按 K 顺序累加, 当前实现要求结果完全一致.
    std::vector<float> expected(M * N), actual(M * N);
    HIP_CHECK(launch_sgemm(device_A, device_B, device_C, M, N, K, 4, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(expected.data(), device_C, expected.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemset(device_C, 0xff, actual.size() * sizeof(float)));
    HIP_CHECK(launch_sgemm(device_A, device_B, device_C, M, N, K, version, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(
        hipMemcpy(actual.data(), device_C, actual.size() * sizeof(float), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < actual.size(); ++i) {
      if (!std::isfinite(actual[i]) || actual[i] != expected[i]) {
        std::fprintf(stderr, "Correctness failed at %zu: actual=%g, reference=%g\n", i, actual[i],
                     expected[i]);
        return 1;
      }
    }

    hipFuncAttributes attributes;
    HIP_CHECK(hipFuncGetAttributes(&attributes, config.kernel));
    for (int i = 0; i < warmup; ++i) {
      HIP_CHECK(launch_sgemm(device_A, device_B, device_C, M, N, K, version, nullptr));
    }
    HIP_CHECK(hipDeviceSynchronize());
    double mean_ms = 0.0;
    for (int i = 0; i < batches; ++i) {
      mean_ms +=
          benchmark_kernel(
              [&] { return launch_sgemm(device_A, device_B, device_C, M, N, K, version, nullptr); },
              0, iterations) /
          (1000.0 * batches);
    }
    std::printf("V%d: correctness passed (%zu outputs), mean=%.6f ms, %.6f TFLOP/s\n", version,
                actual.size(), mean_ms, 2.0 * M * N * K / (mean_ms * 1e9));
    if (version == 10) {
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
    std::printf("RESULT\tV%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.6f\t%.6f\t%d\t%zu\t%zu\n", version, tile_m,
                tile_n, tile_k, thread_m, thread_n, num_threads, mean_ms,
                2.0 * M * N * K / (mean_ms * 1e9), attributes.numRegs, attributes.sharedSizeBytes,
                attributes.localSizeBytes);
  }
  HIP_CHECK(hipFree(device_A));
  HIP_CHECK(hipFree(device_B));
  HIP_CHECK(hipFree(device_C));
  return 0;
}

template <int VERSION>
int run_version(int argc, char** argv) {
  return run_sgemm(VERSION, argc, argv);
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
      {"sgemm_v1", run_version<1>}, {"sgemm_v2", run_version<2>},   {"sgemm_v3", run_version<3>},
      {"sgemm_v4", run_version<4>}, {"sgemm_v5", run_version<5>},   {"sgemm_v6", run_version<6>},
      {"sgemm_v9", run_version<9>}, {"sgemm_v10", run_version<10>},
  };
  return run_profiler(argc, argv, workloads);
}
