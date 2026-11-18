#include "launch.hip"
#include "../../common/hip/common.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <vector>

static int run_norm(const char* name, bool fused, int argc, char** argv) {
  // step 1: 解析执行模式与 shape.
  const bool profile = argc > 0 && std::strcmp(argv[0], "--profile") == 0;
  const int first_arg = profile ? 1 : 0;
  int rows = 1024;
  int hidden_size = 5120;
  constexpr float eps = 1e-6f;
  int* shape[] = {&rows, &hidden_size};

  if (argc > first_arg + 2) {
    std::fprintf(stderr, "Usage: profiler %s [--profile] [rows [hidden_size]]\n", name);
    return 1;
  }
  for (int i = first_arg; i < argc; ++i) {
    const char* text = argv[i];
    const char* end = text + std::strlen(text);
    const auto result = std::from_chars(text, end, *shape[i - first_arg]);
    if (result.ec != std::errc{} || result.ptr != end || *shape[i - first_arg] <= 0) {
      std::fprintf(stderr, "Expected positive rows and hidden_size.\n");
      return 1;
    }
  }
  if (static_cast<size_t>(rows) >
      std::numeric_limits<size_t>::max() / sizeof(hip_bfloat16) / hidden_size) {
    std::fprintf(stderr, "Tensor byte size exceeds size_t range.\n");
    return 1;
  }

  // step 2: 准备确定性输入和权重，分配独立的 GPU 缓冲区
  const size_t elements = static_cast<size_t>(rows) * hidden_size;
  const size_t bytes = elements * sizeof(hip_bfloat16);
  const size_t weight_bytes = static_cast<size_t>(hidden_size) * sizeof(hip_bfloat16);
  // 融合算子原地更新 x 和 residual, 用 x=0, gamma=0 保持多次计时的数据一致.
  std::vector<hip_bfloat16> x(elements, hip_bfloat16(fused ? 0.0f : 1.25f));
  std::vector<hip_bfloat16> gamma(hidden_size, hip_bfloat16(fused ? 0.0f : 2.0f));
  hip_bfloat16 *device_x, *device_gamma, *device_y;
  HIP_CHECK(hipMalloc(&device_x, bytes));
  HIP_CHECK(hipMalloc(&device_gamma, weight_bytes));
  HIP_CHECK(hipMalloc(&device_y, bytes));
  HIP_CHECK(hipMemcpy(device_x, x.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device_gamma, gamma.data(), weight_bytes, hipMemcpyHostToDevice));
  if (fused) {
    std::vector<hip_bfloat16> residual(elements, hip_bfloat16(1.25f));
    HIP_CHECK(hipMemcpy(device_y, residual.data(), bytes, hipMemcpyHostToDevice));
  }

  // step 3: 执行一次 kernel，采集模式仅保留这一次工作负载
  const auto launch = [&] {
    if (fused) {
      return launch_add_rmsnorm(device_x, device_y, device_gamma, rows, hidden_size, eps, nullptr);
    }
    return launch_rmsnorm(device_x, device_gamma, device_y, rows, hidden_size, eps, nullptr);
  };
  HIP_CHECK(launch());
  HIP_CHECK(hipDeviceSynchronize());

  int status = 0;
  if (profile) {
    std::printf("Profile %s: dtype=BF16, rows=%d, hidden_size=%d, eps=%g\n", name, rows,
                hidden_size, eps);
  } else {
    // step 4: 用常量输入完成基础检查，再多批次预热计时并取中位数
    std::vector<hip_bfloat16> y(elements);
    HIP_CHECK(hipMemcpy(y.data(), fused ? device_x : device_y, bytes, hipMemcpyDeviceToHost));
    const float expected =
        fused ? 0.0f : float(hip_bfloat16(1.25f / std::sqrt(1.25f * 1.25f + eps) * 2.0f));
    bool passed = std::all_of(y.begin(), y.end(),
                              [expected](hip_bfloat16 value) { return float(value) == expected; });
    if (fused) {
      HIP_CHECK(hipMemcpy(y.data(), device_y, bytes, hipMemcpyDeviceToHost));
      passed &=
          std::all_of(y.begin(), y.end(), [](hip_bfloat16 value) { return float(value) == 1.25f; });
    }
    if (!passed) {
      std::fprintf(stderr, "%s: smoke check failed\n", name);
      status = 1;
    } else {
      std::vector<float> timings(7);
      for (float& timing : timings) {
        timing = benchmark_kernel(launch, 100, 500);
      }
      std::sort(timings.begin(), timings.end());
      const float median_us = timings[timings.size() / 2];
      std::printf(
          "%s: smoke check passed, dtype=BF16, rows=%d, hidden_size=%d, "
          "eps=%g, median=%.3f us\n",
          name, rows, hidden_size, eps, median_us);
    }
  }

  // step 5: 释放缓冲区并返回执行状态
  HIP_CHECK(hipFree(device_x));
  HIP_CHECK(hipFree(device_gamma));
  HIP_CHECK(hipFree(device_y));
  return status;
}

static int run_rmsnorm(int argc, char** argv) { return run_norm("rmsnorm", false, argc, argv); }
static int run_add_rmsnorm(int argc, char** argv) {
  return run_norm("add_rmsnorm", true, argc, argv);
}

int main(int argc, char** argv) {
  const ProfileWorkload workloads[] = {
      {"rmsnorm", run_rmsnorm},
      {"add_rmsnorm", run_add_rmsnorm},
  };
  return run_profiler(argc, argv, workloads);
}
