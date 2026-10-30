#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(call)                                                  \
  do {                                                                   \
    const hipError_t hip_check_status = (call);                          \
    if (hip_check_status != hipSuccess) {                                \
      std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, \
                   #call, hipGetErrorString(hip_check_status));          \
      std::exit(EXIT_FAILURE);                                           \
    }                                                                    \
  } while (false)

// launch() 返回 hipError_t，并将工作提交到默认流；返回平均耗时，单位为微秒。
template <typename Launch>
float benchmark_kernel(Launch launch, int warmup = 20, int iterations = 100) {
  for (int i = 0; i < warmup; ++i) {
    HIP_CHECK(launch());
  }
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    HIP_CHECK(launch());
  }
  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));
  float elapsed_ms;
  HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return elapsed_ms * 1000.0f / iterations;
}

struct ProfileWorkload {
  const char* name;
  int (*run)(int argc, char** argv);
};

// 各算子族的 profiler 注册工作负载，共用命令行选择与调用逻辑。
template <size_t N>
int run_profiler(int argc, char** argv, const ProfileWorkload (&workloads)[N]) {
  if (argc == 2 && std::strcmp(argv[1], "--list") == 0) {
    for (const auto& workload : workloads) {
      std::printf("%s\n", workload.name);
    }
    return 0;
  }
  if (argc >= 2) {
    for (const auto& workload : workloads) {
      if (std::strcmp(argv[1], workload.name) == 0) {
        return workload.run(argc - 2, argv + 2);
      }
    }
  }
  std::fprintf(stderr, "Usage: %s operator [--profile] [operator_args...]\n",
               argv[0]);
  std::fprintf(stderr, "Available operators:");
  for (const auto& workload : workloads) {
    std::fprintf(stderr, " %s", workload.name);
  }
  std::fprintf(stderr, "\n");
  return 1;
}
