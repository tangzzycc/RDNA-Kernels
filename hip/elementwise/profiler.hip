#include "launch.hip"
#include "../../common/hip/common.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <vector>

// 复用参数解析, BF16 缓冲区, 基础检查与计时; --profile 仅执行一次采集负载.
template <size_t INPUTS, typename Launch>
static int run_elementwise(const char* name, int argc, char** argv,
                           const std::array<float, INPUTS>& values, float expected,
                           Launch launch_op) {
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
  if (argc > first_arg + 1 || n > std::numeric_limits<size_t>::max() / sizeof(hip_bfloat16) ||
      (n - 1) / 256 + 1 > std::numeric_limits<unsigned int>::max()) {
    std::fprintf(stderr, "Usage: profiler %s [--profile] [positive_element_count]\n", name);
    return 1;
  }

  const size_t bytes = n * sizeof(hip_bfloat16);
  std::array<hip_bfloat16*, INPUTS> inputs{};
  std::vector<hip_bfloat16> host(n);
  for (size_t i = 0; i < INPUTS; ++i) {
    std::fill(host.begin(), host.end(), hip_bfloat16(values[i]));
    HIP_CHECK(hipMalloc(&inputs[i], bytes));
    HIP_CHECK(hipMemcpy(inputs[i], host.data(), bytes, hipMemcpyHostToDevice));
  }
  hip_bfloat16* output;
  HIP_CHECK(hipMalloc(&output, bytes));
  const auto launch = [&] { return launch_op(inputs, output, n); };
  HIP_CHECK(launch());
  HIP_CHECK(hipDeviceSynchronize());

  if (profile) {
    std::printf("Profile %s: dtype=BF16, n=%zu, block=256\n", name, n);
  } else {
    HIP_CHECK(hipMemcpy(host.data(), output, bytes, hipMemcpyDeviceToHost));
    const float expected_bf16 = float(hip_bfloat16(expected));
    if (!std::all_of(host.begin(), host.end(),
                     [&](hip_bfloat16 value) { return float(value) == expected_bf16; })) {
      std::fprintf(stderr, "%s: smoke check failed\n", name);
      return 1;
    }
    const float mean_us = benchmark_kernel(launch);
    std::printf("%s: smoke check passed, dtype=BF16, n=%zu, block=256, mean=%.3f us\n", name, n,
                mean_us);
  }

  for (auto* input : inputs) {
    HIP_CHECK(hipFree(input));
  }
  HIP_CHECK(hipFree(output));
  return 0;
}

static int run_add(int argc, char** argv) {
  return run_elementwise("add", argc, argv, std::array<float, 2>{1.25f, 2.0f}, 3.25f,
                         [](const auto& inputs, auto* output, size_t n) {
                           return launch_add(inputs[0], inputs[1], output, n, nullptr);
                         });
}

static int run_mul(int argc, char** argv) {
  return run_elementwise("mul", argc, argv, std::array<float, 2>{1.25f, 2.0f}, 2.5f,
                         [](const auto& inputs, auto* output, size_t n) {
                           return launch_mul(inputs[0], inputs[1], output, n, nullptr);
                         });
}

static int run_silu(int argc, char** argv) {
  const float expected = float(1.25 / (1.0 + std::exp(-1.25)));
  return run_elementwise("silu", argc, argv, std::array<float, 1>{1.25f}, expected,
                         [](const auto& inputs, auto* output, size_t n) {
                           return launch_silu(inputs[0], output, n, nullptr);
                         });
}

static int run_silu_mul(int argc, char** argv) {
  const hip_bfloat16 act(float(1.25 / (1.0 + std::exp(-1.25))));
  return run_elementwise("silu_mul", argc, argv, std::array<float, 2>{1.25f, 1.25f},
                         float(act) * 1.25f, [](const auto& inputs, auto* output, size_t n) {
                           return launch_silu_mul(inputs[0], inputs[1], output, n, nullptr);
                         });
}

static int run_softmax(int argc, char** argv) {
  const bool profile = argc > 0 && std::strcmp(argv[0], "--profile") == 0;
  const int first_arg = profile ? 1 : 0;
  int rows = 1024, cols = 5120;
  int* shape[] = {&rows, &cols};
  if (argc > first_arg + 2) {
    std::fprintf(stderr, "Usage: profiler softmax [--profile] [rows [cols]]\n");
    return 1;
  }
  for (int i = first_arg; i < argc; ++i) {
    const char* text = argv[i];
    const char* end = text + std::strlen(text);
    const auto result = std::from_chars(text, end, *shape[i - first_arg]);
    if (result.ec != std::errc{} || result.ptr != end || *shape[i - first_arg] <= 0) {
      std::fprintf(stderr, "Expected positive rows and cols.\n");
      return 1;
    }
  }
  const size_t elements = size_t(rows) * cols;
  const size_t bytes = elements * sizeof(hip_bfloat16);
  std::vector<hip_bfloat16> host(elements, hip_bfloat16(1.25f));
  hip_bfloat16 *device_x, *device_y;
  HIP_CHECK(hipMalloc(&device_x, bytes));
  HIP_CHECK(hipMalloc(&device_y, bytes));
  HIP_CHECK(hipMemcpy(device_x, host.data(), bytes, hipMemcpyHostToDevice));
  const auto launch = [&] { return launch_softmax(device_x, device_y, rows, cols, nullptr); };
  HIP_CHECK(launch());
  HIP_CHECK(hipDeviceSynchronize());
  if (profile) {
    std::printf("Profile softmax: dtype=BF16, rows=%d, cols=%d\n", rows, cols);
  } else {
    HIP_CHECK(hipMemcpy(host.data(), device_y, bytes, hipMemcpyDeviceToHost));
    const float expected = float(hip_bfloat16(1.0f / cols));
    if (!std::all_of(host.begin(), host.end(),
                     [expected](hip_bfloat16 value) { return float(value) == expected; })) {
      std::fprintf(stderr, "softmax: smoke check failed\n");
      HIP_CHECK(hipFree(device_x));
      HIP_CHECK(hipFree(device_y));
      return 1;
    }
    const float mean_us = benchmark_kernel(launch);
    std::printf("softmax: smoke check passed, dtype=BF16, rows=%d, cols=%d, mean=%.3f us\n", rows,
                cols, mean_us);
  }
  HIP_CHECK(hipFree(device_x));
  HIP_CHECK(hipFree(device_y));
  return 0;
}

int main(int argc, char** argv) {
  const ProfileWorkload workloads[] = {
      {"add", run_add},           {"mul", run_mul},         {"silu", run_silu},
      {"silu_mul", run_silu_mul}, {"softmax", run_softmax},
  };
  return run_profiler(argc, argv, workloads);
}
