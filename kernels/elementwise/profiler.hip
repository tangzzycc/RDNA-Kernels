#include "add.hip"
#include "../../common/hip/common.h"

#include <algorithm>
#include <charconv>
#include <vector>

// 执行基础检查与性能测试；--profile 模式仅执行一次采集工作负载。
static int run_add(int argc, char** argv) {
  const bool profile = argc > 0 && std::strcmp(argv[0], "--profile") == 0;
  const int first_arg = profile ? 1 : 0;
  size_t n = 1 << 24;
  if (argc > first_arg) {
    const char* text = argv[first_arg];
    const char* end = text + std::strlen(text);
    const auto result = std::from_chars(text, end, n);
    if (result.ec != std::errc{} || result.ptr != end || n == 0) {
      std::fprintf(stderr, "Expected a positive element count.\n");
      return 1;
    }
  }
  if (argc > first_arg + 1 ||
      n > std::numeric_limits<size_t>::max() / sizeof(hip_bfloat16) ||
      (n - 1) / 256 + 1 > std::numeric_limits<unsigned int>::max()) {
    std::fprintf(stderr,
                 "Usage: profiler add [--profile] [positive_element_count]\n");
    return 1;
  }

  const size_t bytes = n * sizeof(hip_bfloat16);
  std::vector<hip_bfloat16> a(n, hip_bfloat16(1.25f)), b(n, hip_bfloat16(2.0f));
  hip_bfloat16 *device_a, *device_b, *device_c;
  HIP_CHECK(hipMalloc(&device_a, bytes));
  HIP_CHECK(hipMalloc(&device_b, bytes));
  HIP_CHECK(hipMalloc(&device_c, bytes));
  HIP_CHECK(hipMemcpy(device_a, a.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device_b, b.data(), bytes, hipMemcpyHostToDevice));
  const auto launch = [&] {
    return launch_add(device_a, device_b, device_c, n, nullptr);
  };
  HIP_CHECK(launch());
  HIP_CHECK(hipDeviceSynchronize());

  if (profile) {
    std::printf("Profile add: dtype=BF16, n=%zu, block=256\n", n);
  } else {
    std::vector<hip_bfloat16> c(n);
    HIP_CHECK(hipMemcpy(c.data(), device_c, bytes, hipMemcpyDeviceToHost));
    if (!std::all_of(c.begin(), c.end(), [](hip_bfloat16 value) {
          return float(value) == 3.25f;
        })) {
      std::fprintf(stderr, "add: smoke check failed\n");
      return 1;
    }
    const float mean_us = benchmark_kernel(launch);
    std::printf(
        "add: smoke check passed, dtype=BF16, n=%zu, block=256, mean=%.3f us, "
        "effective bandwidth=%.2f GB/s\n",
        n, mean_us, 3.0 * bytes / (mean_us * 1e3));
  }

  HIP_CHECK(hipFree(device_a));
  HIP_CHECK(hipFree(device_b));
  HIP_CHECK(hipFree(device_c));
  return 0;
}

int main(int argc, char** argv) {
  const ProfileWorkload workloads[] = {
      {"add", run_add},
  };
  return run_profiler(argc, argv, workloads);
}
