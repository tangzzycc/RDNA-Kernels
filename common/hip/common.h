#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

// 判断 ptr 的地址是否满足 alignment 字节对齐, 可在主机和设备端调用.
// alignment 为 0 时返回 false.
__host__ __device__ inline bool is_aligned(const void* ptr, size_t alignment) {
  return alignment != 0 &&
         reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}

// 在主机端检查 HIP 调用的返回值, 出错时打印调用位置和错误信息并退出进程.
#define HIP_CHECK(call)                                                  \
  do {                                                                   \
    const hipError_t hip_check_status = (call);                          \
    if (hip_check_status != hipSuccess) {                                \
      std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, \
                   #call, hipGetErrorString(hip_check_status));          \
      std::exit(EXIT_FAILURE);                                           \
    }                                                                    \
  } while (false)

// 将运行时向量宽度分发为编译期 VEC_SIZE, 并执行对应分支中的代码.
// 支持 1, 2, 4, 8; 其他值使外层函数返回 hipErrorInvalidValue.
// 用于返回 hipError_t 的 launch 函数, 分支代码可使用常量 VEC_SIZE.
#define HIP_DISPATCH_VEC_SIZE(value, ...) \
  do {                                    \
    switch (value) {                      \
      case 8: {                           \
        constexpr int VEC_SIZE = 8;       \
        __VA_ARGS__;                      \
        break;                            \
      }                                   \
      case 4: {                           \
        constexpr int VEC_SIZE = 4;       \
        __VA_ARGS__;                      \
        break;                            \
      }                                   \
      case 2: {                           \
        constexpr int VEC_SIZE = 2;       \
        __VA_ARGS__;                      \
        break;                            \
      }                                   \
      case 1: {                           \
        constexpr int VEC_SIZE = 1;       \
        __VA_ARGS__;                      \
        break;                            \
      }                                   \
      default:                            \
        return hipErrorInvalidValue;      \
    }                                     \
  } while (false)

// 预热后使用 HIP event 计时, 返回一次 launch 调用的平均设备耗时, 单位为 us.
// launch 须向默认流提交工作并返回 hipError_t.
// warmup 为预热次数, iterations 为计时次数且必须大于 0.
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

// 注册一个 profiler 工作负载, name 用于命令行选择, run 执行该负载.
struct ProfileWorkload {
  const char* name;
  int (*run)(int argc, char** argv);
};

// 根据 argv[1] 选择已注册的工作负载, 将余下参数传给其 run 函数.
// --list 打印工作负载名称并返回 0; 参数无效时打印用法并返回 1.
// 执行工作负载时返回 run 的退出码, 供各算子族的 profiler main 复用.
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

// 连续存放 VEC_SIZE 个 T 元素, 并按 sizeof(T) * VEC_SIZE 字节对齐.
// VEC_SIZE 必须是正的二次幂, 该类型用于向量化读取和写入.
template <typename T, int VEC_SIZE>
struct alignas(sizeof(T) * VEC_SIZE) vec_n_t {
  static_assert(VEC_SIZE > 0 && (VEC_SIZE & (VEC_SIZE - 1)) == 0,
                "VEC_SIZE must be a positive power of two");
  T val[VEC_SIZE];
};

// 将 input 的 len 个元素划分为标量前缀, 对齐的向量主体和标量尾部.
// tid 为参与线程编号, stride 为参与线程数; 各线程跨步处理各自的元素.
// vec_op 接收 vec_n_t<T, VEC_SIZE>, scalar_op 接收单个 T 元素.
// 各线程须使用相同的 input, len 和 stride, 且满足 0 <= tid < stride.
template <int VEC_SIZE, typename T, typename VecOp, typename ScalarOp>
__device__ inline void vectorize_read_with_alignment(const T* input, int len,
                                                     int tid, int stride,
                                                     VecOp&& vec_op,
                                                     ScalarOp&& scalar_op) {
  using Vec = vec_n_t<T, VEC_SIZE>;
  constexpr int WIDTH = sizeof(T) * VEC_SIZE;

  std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(input);

  // 地址和长度都满足向量要求时, 直接按向量遍历整个区域.
  if (addr % WIDTH == 0 && len % VEC_SIZE == 0) {
    const auto* vectors = reinterpret_cast<const Vec*>(input);
    int num_vec = len / VEC_SIZE;

    for (int i = tid; i < num_vec; i += stride) {
      Vec value = vectors[i];
      vec_op(value);
    }
    return;
  }

  // 计算到下一个向量对齐地址之间的元素数.
  int misalignment = static_cast<int>(addr % WIDTH);
  int prefix = 0;
  if (misalignment != 0) {
    prefix = (WIDTH - misalignment) / sizeof(T);
  }
  prefix = prefix < len ? prefix : len;

  // 处理标量前缀, 使剩余区域的起始地址满足向量对齐.
  for (int i = tid; i < prefix; i += stride) {
    scalar_op(input[i]);
  }

  input += prefix;
  len -= prefix;

  // 按向量遍历主体, 只读取完整的向量.
  int num_vec = len / VEC_SIZE;
  const auto* vectors = reinterpret_cast<const Vec*>(input);

  for (int i = tid; i < num_vec; i += stride) {
    Vec value = vectors[i];
    vec_op(value);
  }

  // 处理不足一个向量的标量尾部.
  int tail_start = num_vec * VEC_SIZE;
  for (int i = tail_start + tid; i < len; i += stride) {
    scalar_op(input[i]);
  }
}
