#include "../../common/hip/common.h"

#include <__clang_hip_runtime_wrapper.h>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/amd_detail/amd_hip_vector_types.h>
#include <hip/hip_runtime.h>

/*
 * SGEMM: 使用 FP32 计算矩阵乘法 C = A * B.
 *
 *   C[row, col] = sum_k A[row, k] * B[k, col]
 *
 * 参考 Simon Boehm 的 CUDA Matmul Worklog 与配套代码:
 * https://siboehm.com/articles/22/CUDA-MMM
 * https://github.com/siboehm/SGEMM_CUDA/tree/master/src/kernels
 *
 * 沿用文章编号, 演进路线:
 * V1 朴素内积 -> V2 合并访存 -> V3 LDS 分块 -> V4 一维寄存器分块.
 * V5 二维寄存器分块 -> V6 向量化与 A 转置 -> V9 参数调优 -> V10 wave 分块.
 * 原仓库 V7 的 B 布局重排与 V8 的 padding 留作 LDS bank 冲突的可选实验.
 *
 * BLOCK_M, BLOCK_N: 一个 block 覆盖的输出行数和列数.
 * BLOCK_K: 每轮沿 K 加载的长度, LDS 缓存 A[BLOCK_M, BLOCK_K] 与 B[BLOCK_K, BLOCK_N].
 * THREAD_M, THREAD_N: 每线程输出小块的行数和列数.
 * 加载任务由线性 tid 与跨步循环分配, 每线程加载量由分块大小和线程数推导.
 *
 * gfx1151 使用 wave32, CUDA SMEM 对应 HIP LDS.
 * 初始配置用于理解和对照, 最优参数需在本机测量, 库性能对照使用 rocBLAS FP32.
 * 全程使用标量 FP32 FMA, 不把 CUDA 的硬件吞吐直接套到 RDNA.
 */

// V1: 朴素内积, 二维 block, 每线程计算 C[row, col], 相邻线程的 row 递增.
// 启动 grid(ceil(M / block.x), ceil(N / block.y)), 可先使用 block(32, 32).
__global__ void sgemm_v1_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                float* __restrict__ C, int M, int N, int K) {
  // block(32, 32) 时, 一个 wave 内 row 递增, col 固定.
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N) {
    // 固定 row, col, 沿 k 计算 A 第 row 行与 B 第 col 列的内积.
    float tmp = 0.0f;
    for (int k = 0; k < K; k++) {
      // A[row * K + k] → A[row, k], B[k * N + col] → B[k, col].
      // 相邻线程读取 A 的地址相隔 K 个元素, 读取 B 的地址相同.
      tmp += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = tmp;
  }
}

// V2: 合并访存, 一维 block, 每线程计算 C[row, col], 相邻线程的 col 递增.
// 启动 grid(ceil(M / BLOCK_SIZE), ceil(N / BLOCK_SIZE)), block(BLOCK_SIZE * BLOCK_SIZE),
// 可先使用 BLOCK_SIZE=32.
template <int BLOCK_SIZE>
__global__ void sgemm_v2_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                float* __restrict__ C, int M, int N, int K) {
  // blockIdx.x 定位 row 分块, blockIdx.y 定位 col 分块.
  // tid / BLOCK_SIZE 对应局部行, tid % BLOCK_SIZE 对应局部列.
  const int row = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
  const int col = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

  if (row < M && col < N) {
    // 固定 row, col, 沿 k 计算 A 第 row 行与 B 第 col 列的内积.
    float tmp = 0.0f;
    for (int k = 0; k < K; k++) {
      // A[row * K + k] → A[row, k], B[k * N + col] → B[k, col].
      // BLOCK_SIZE=32 时, 一个 wave 内读取 A 的地址相同, B 的地址连续, C 也连续写回.
      tmp += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = tmp;
  }
}

// V3: LDS 分块, 缓存 A, B 的当前 K 分块, 每线程计算 C[row, col].
// 启动 grid(M / BLOCK_SIZE, N / BLOCK_SIZE), block(BLOCK_SIZE * BLOCK_SIZE).
// 可先使用 BLOCK_SIZE=32, M, N, K 须为 BLOCK_SIZE 的整数倍, 相邻线程的 col 递增.
template <int BLOCK_SIZE>
__global__ void sgemm_v3_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                float* __restrict__ C, int M, int N, int K) {
  // LDS 分别缓存 A, B 的 [BLOCK_SIZE, BLOCK_SIZE] 分块.
  // block 内每个 A 值复用于 BLOCK_SIZE 个 col, 每个 B 值复用于 BLOCK_SIZE 个 row.
  __shared__ float A_shared[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ float B_shared[BLOCK_SIZE * BLOCK_SIZE];

  // 一维线程编号映射为分块内的 row, col, 每线程负责一个输出和两个加载元素.
  const int thread_col = threadIdx.x % BLOCK_SIZE;
  const int thread_row = threadIdx.x / BLOCK_SIZE;

  // blockIdx.x 定位 row 分块, blockIdx.y 定位 col 分块, 三个指针指向各自分块起点.
  const float* A_ptr = A + blockIdx.x * BLOCK_SIZE * K;
  const float* B_ptr = B + blockIdx.y * BLOCK_SIZE;
  float* C_ptr = C + blockIdx.x * BLOCK_SIZE * N + blockIdx.y * BLOCK_SIZE;

  // tmp 累加当前输出, 跨所有 K 分块保留结果.
  float tmp = 0.0f;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_SIZE) {
    // A_ptr[thread_row * K + thread_col] → A[实际 row, bkIdx + thread_col].
    A_shared[thread_row * BLOCK_SIZE + thread_col] = A_ptr[thread_row * K + thread_col];
    // B_ptr[thread_row * N + thread_col] → B[bkIdx + thread_row, 实际 col].
    B_shared[thread_row * BLOCK_SIZE + thread_col] = B_ptr[thread_row * N + thread_col];
    __syncthreads();

    // A 沿 K 跨列, B 沿 K 跨行, 准备下一轮的全局加载地址.
    A_ptr += BLOCK_SIZE;
    B_ptr += BLOCK_SIZE * N;

    // As 的 thread_row 行与 Bs 的 thread_col 列做内积, i 遍历当前 K 分块.
    for (int i = 0; i < BLOCK_SIZE; i++) {
      // A_shared[thread_row * BLOCK_SIZE + i] → As[thread_row, i].
      // B_shared[i * BLOCK_SIZE + thread_col] → Bs[i, thread_col].
      tmp += A_shared[thread_row * BLOCK_SIZE + i] * B_shared[i * BLOCK_SIZE + thread_col];
    }
    __syncthreads();
  }
  // C_ptr[thread_row * N + thread_col] → C[实际 row, 实际 col].
  C_ptr[thread_row * N + thread_col] = tmp;
}

// V4: 一维寄存器分块, 每线程计算固定 col 的 THREAD_M 个 row, 复用 LDS 中的 B 值.
// 启动 grid(N / BLOCK_N, M / BLOCK_M), block(BLOCK_M * BLOCK_N / THREAD_M).
// 可先使用 <64, 64, 8, 8>, 初版每线程加载一个 A 和一个 B.
// 两个 LDS 分块的元素数均须等于线程数, BLOCK_M 须为 THREAD_M 的整数倍.
// M, N, K 分别按 BLOCK_M, BLOCK_N, BLOCK_K 整除.
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M>
__global__ void sgemm_v4_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                float* __restrict__ C, int M, int N, int K) {
  // LDS 缓存 A[BLOCK_M, BLOCK_K] 和 B[BLOCK_K, BLOCK_N].
  __shared__ float A_shared[BLOCK_M * BLOCK_K];
  __shared__ float B_shared[BLOCK_K * BLOCK_N];

  // thread_col 定位输出 col, thread_row 定位 THREAD_M 个 row 组成的行组.
  const int thread_col = threadIdx.x % BLOCK_N;
  const int thread_row = threadIdx.x / BLOCK_N;

  /**
   * blockIdx.y 定位 row 分块, blockIdx.x 定位 col 分块, 沿用参考实现的映射.
   * 两种映射都存在共享 A tile, 读取 B 相邻列分块的任务, 区别是这些任务的编号排列.
   * 原映射固定 x, 增加 y 时共享 A; 当前映射固定 y, 增加 x 时共享 A.
   *
   * 例如 FP32, M=N=K=5120, BLOCK_M=BLOCK_N=64, BLOCK_K=8, grid(80, 80).
   * 固定 bkIdx=0, 两个任务加载的 tile 为:
   *   A: 都加载 A[0:64, 0:8].
   *   B: 分别加载 B[0:8, 0:64] 与 B[0:8, 64:128].
   * 两种映射下 B 的每行加载区域都相邻, B_ptr 起点相差 256 bytes.
   *
   * 这两个任务对应的 block 坐标与编号为:
   *   原映射: blockIdx=(0, 0), (0, 1), 线性编号为 0, 80.
   *   当前映射: blockIdx=(0, 0), (1, 0), 线性编号为 0, 1.
   * block 的线性编号为 blockIdx.y * gridDim.x + blockIdx.x, x 维变化最快.
   *
   * 用 Ar 表示第 r 个 A 行分块, Bc 表示第 c 个 B 列分块, 按线性编号排列:
   *   原映射: (A0, B0), (A1, B0), ..., (A79, B0), (A0, B1), ...
   *   当前映射: (A0, B0), (A0, B1), ..., (A0, B79), (A1, B0), ...
   * 总共需要加载的分块相同.
   * 当前映射把共享 A, 读取 B 相邻列分块的任务放在相邻编号上, 增加缓存复用的机会.
   * GPU 不保证严格按编号执行, 收益依赖 shape 与硬件.
   *
   * A, B 均为行主序, A tile [64, 8] 与 B tile [8, 64] 的布局不对称.
   * M == N 也不会使两种任务排列的缓存行为等价, 上述下标区间为左闭右开.
   * 交换映射改变任务排列与缓存复用的时机.
   * threadIdx 的映射与 wave 内合并访存保持原样.
   *
   * 本机 V4 实测该映射提高 L1, L2 命中率并减少 DRAM 读取量, 收益依赖 shape 与硬件.
   * 启动须同步使用 grid(N / BLOCK_N, M / BLOCK_M), 非方形 shape 尤其不能写反.
   */
  // A_ptr -> A[blockIdx.y * BLOCK_M, 0].
  // B_ptr -> B[0, blockIdx.x * BLOCK_N].
  // C_ptr -> C[blockIdx.y * BLOCK_M, blockIdx.x * BLOCK_N].
  const float* A_ptr = A + blockIdx.y * BLOCK_M * K;
  const float* B_ptr = B + blockIdx.x * BLOCK_N;
  float* C_ptr = C + blockIdx.y * BLOCK_M * N + blockIdx.x * BLOCK_N;

  // thread_res[j] 累加行组内第 j 个输出, 跨所有 K 分块保留结果.
  float thread_res[THREAD_M] = {0.0f};

  // 加载坐标按 A 的 BLOCK_K 列宽和 B 的 BLOCK_N 列宽展开, 每线程各搬运一个元素.
  const int load_rowA = threadIdx.x / BLOCK_K;
  const int load_colA = threadIdx.x % BLOCK_K;
  const int load_rowB = threadIdx.x / BLOCK_N;
  const int load_colB = threadIdx.x % BLOCK_N;

  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_K) {
    // A_ptr[load_rowA * K + load_colA] → A[实际 row, bkIdx + load_colA].
    A_shared[load_rowA * BLOCK_K + load_colA] = A_ptr[load_rowA * K + load_colA];
    // B_ptr[load_rowB * N + load_colB] → B[bkIdx + load_rowB, 实际 col].
    B_shared[load_rowB * BLOCK_N + load_colB] = B_ptr[load_rowB * N + load_colB];
    __syncthreads();

    A_ptr += BLOCK_K;
    B_ptr += BLOCK_K * N;

    for (int i = 0; i < BLOCK_K; i++) {
      // 固定 i, A 第 i 列的 THREAD_M 个 row 分别乘同一个 Bs[i, thread_col].
      for (int j = 0; j < THREAD_M; j++) {
        // A_shared[(thread_row * THREAD_M + j) * BLOCK_K + i]
        // → As[thread_row * THREAD_M + j, i].
        thread_res[j] += A_shared[(thread_row * THREAD_M + j) * BLOCK_K + i] *
                         B_shared[i * BLOCK_N + thread_col];
      }
    }
    __syncthreads();
  }
  for (int i = 0; i < THREAD_M; ++i) {
    // C_ptr[(thread_row * THREAD_M + i) * N + thread_col]
    // → C[当前线程的起始 row + i, 实际 col].
    C_ptr[(thread_row * THREAD_M + i) * N + thread_col] = thread_res[i];
  }
}

// V5: 二维寄存器分块, 每线程计算 THREAD_M 个 row 与 THREAD_N 个 col 组成的输出小块.
// 启动 grid(N / BLOCK_N, M / BLOCK_M), block(BLOCK_M * BLOCK_N / (THREAD_M * THREAD_N)).
// 可先使用 <128, 128, 8, 8, 8>, 分别沿 row, col 划分输出小块, 用跨步循环协作加载 LDS.
// THREAD_M, THREAD_N 须分别整除 BLOCK_M, BLOCK_N, M, N, K 按各自分块整除.
// 初版加载要求线程数分别按 BLOCK_K, BLOCK_N 整除.
// 两个加载 stride 分别整除 BLOCK_M, BLOCK_K.
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N>
__global__ void sgemm_v5_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                float* __restrict__ C, int M, int N, int K) {
  // 输出 tile [BLOCK_M, BLOCK_N] 划分为多个 [THREAD_M, THREAD_N] 小块, 每线程负责一个.
  // <128, 128, 8, 8, 8> 对应 256 个线程, 每线程计算 8 * 8 个输出.
  const int num_thread_block_tile = BLOCK_M * BLOCK_N / (THREAD_M * THREAD_N);

  // thread_row, thread_col 定位输出小块.
  // 小块起点为 [thread_row * THREAD_M, thread_col * THREAD_N].
  // 例中逻辑线程网格为 [16, 16], 相邻线程沿 col 方向负责相邻的 8 列输出.
  const int thread_col = threadIdx.x % (BLOCK_N / THREAD_N);
  const int thread_row = threadIdx.x / (BLOCK_N / THREAD_N);

  // 每轮缓存 A tile [BLOCK_M, BLOCK_K] 与 B tile [BLOCK_K, BLOCK_N].
  // 两者相乘得到当前 K 分块对 C tile [BLOCK_M, BLOCK_N] 的贡献.
  __shared__ float A_shared[BLOCK_M * BLOCK_K];
  __shared__ float B_shared[BLOCK_K * BLOCK_N];

  // 沿用 V4 的 block 映射, blockIdx.y 对应 row, blockIdx.x 对应 col, 详细解释见 V4.
  const float* A_ptr = A + blockIdx.y * BLOCK_M * K;
  const float* B_ptr = B + blockIdx.x * BLOCK_N;
  float* C_ptr = C + blockIdx.y * BLOCK_M * N + blockIdx.x * BLOCK_N;

  // thread_res[m * THREAD_N + n] 保存线程输出小块的第 [m, n] 项, 跨所有 K 分块累加.
  // 固定 k 时, reg_M 缓存 A 的 THREAD_M 个 row, reg_N 缓存 B 的 THREAD_N 个 col.
  float thread_res[THREAD_M * THREAD_N] = {0.0f};
  float reg_M[THREAD_M] = {0.0f};
  float reg_N[THREAD_N] = {0.0f};

  // 加载坐标与计算坐标独立, 按 tile 的列宽展开 tid, 让相邻线程搬运同一行的相邻元素.
  // A tile 的行宽为 BLOCK_K, B tile 的行宽为 BLOCK_N.
  const int load_rowA = threadIdx.x / BLOCK_K;
  const int load_colA = threadIdx.x % BLOCK_K;
  const int load_rowB = threadIdx.x / BLOCK_N;
  const int load_colB = threadIdx.x % BLOCK_N;
  // 每轮协作搬运覆盖的行数. 例中 A 每轮搬运 32 行, B 每轮搬运 2 行.
  const int stride_A = num_thread_block_tile / BLOCK_K;
  const int stride_B = num_thread_block_tile / BLOCK_N;

  // bkIdx 遍历整个 K 维, 每轮计算 A[:, bkIdx:bkIdx+BLOCK_K] * B[bkIdx:bkIdx+BLOCK_K, :].
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_K) {
    // o 沿 A tile 的 row 方向跨步, 填满 BLOCK_M 行, 例中 o=0, 32, 64, 96.
    for (int o = 0; o < BLOCK_M; o += stride_A) {
      // As[load_rowA + o, load_colA]
      // <- A[blockIdx.y * BLOCK_M + load_rowA + o, bkIdx + load_colA].
      A_shared[(load_rowA + o) * BLOCK_K + load_colA] = A_ptr[(load_rowA + o) * K + load_colA];
    }
    // o 沿 B tile 的 row 方向跨步, 填满 BLOCK_K 行, 例中 o=0, 2, 4, 6.
    for (int o = 0; o < BLOCK_K; o += stride_B) {
      // Bs[load_rowB + o, load_colB]
      // <- B[bkIdx + load_rowB + o, blockIdx.x * BLOCK_N + load_colB].
      B_shared[(load_rowB + o) * BLOCK_N + load_colB] = B_ptr[(load_rowB + o) * N + load_colB];
    }
    __syncthreads();

    A_ptr += BLOCK_K;
    B_ptr += BLOCK_K * N;

    for (int k = 0; k < BLOCK_K; k++) {
      // 固定 k, 从 A tile 的第 k 列取 THREAD_M 个 row.
      // 从 B tile 的第 k 行取 THREAD_N 个 col.
      for (int i = 0; i < THREAD_M; i++) {
        // A_shared[(thread_row * THREAD_M + i) * BLOCK_K + k]
        // -> As[thread_row * THREAD_M + i, k].
        reg_M[i] = A_shared[(thread_row * THREAD_M + i) * BLOCK_K + k];
      }
      for (int i = 0; i < THREAD_N; i++) {
        // B_shared[k * BLOCK_N + thread_col * THREAD_N + i]
        // -> Bs[k, thread_col * THREAD_N + i].
        reg_N[i] = B_shared[k * BLOCK_N + thread_col * THREAD_N + i];
      }

      // reg_M [THREAD_M] 与 reg_N [THREAD_N] 做外积, 累加到输出小块 [THREAD_M, THREAD_N].
      // 每个 A 值复用于 THREAD_N 个 col, 每个 B 值复用于 THREAD_M 个 row.
      for (int m = 0; m < THREAD_M; m++) {
        for (int n = 0; n < THREAD_N; n++) {
          thread_res[m * THREAD_N + n] += reg_M[m] * reg_N[n];
        }
      }
    }
    __syncthreads();
  }

  // 完成整个 K 维累加后, 将线程输出小块写回 C, n 方向对应同一行内的连续列.
  for (int m = 0; m < THREAD_M; m++) {
    for (int n = 0; n < THREAD_N; n++) {
      // C_ptr[(thread_row * THREAD_M + m) * N + thread_col * THREAD_N + n]
      // -> C[row, col], 全局坐标为:
      //    row = blockIdx.y * BLOCK_M + thread_row * THREAD_M + m.
      //    col = blockIdx.x * BLOCK_N + thread_col * THREAD_N + n.
      C_ptr[(thread_row * THREAD_M + m) * N + thread_col * THREAD_N + n] =
          thread_res[m * THREAD_N + n];
    }
  }
}

// V6: 向量化访存, A 转置存入 LDS, 用 float4 加载输入与写回输出, 每线程仍计算二维输出小块.
// 启动 grid(N / BLOCK_N, M / BLOCK_M), block(BLOCK_M * BLOCK_N / (THREAD_M * THREAD_N)).
// 可先使用 <128, 128, 8, 8, 8>, 初版每线程搬运一个 A float4 和一个 B float4.
// 两个 LDS 分块的元素数均须等于 4 * 线程数, 因此一次加载即可填满 tile, 无需 stride 循环.
// THREAD_M, THREAD_N 分别整除 BLOCK_M, BLOCK_N, M, N, K 按各自分块整除.
// BLOCK_K, BLOCK_N, N, THREAD_N 须为 4 的整数倍, A, B, C 的起始地址须为 16 字节对齐.
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N>
__global__ void sgemm_v6_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                float* __restrict__ C, int M, int N, int K) {
  // thread_row, thread_col 定位线程输出小块, 起点为 [thread_row * THREAD_M, thread_col * THREAD_N].
  // 例中 256 个线程组成 [16, 16] 的逻辑网格, 每线程计算 [8, 8] 个输出.
  const int thread_col = threadIdx.x % (BLOCK_N / THREAD_N);
  const int thread_row = threadIdx.x / (BLOCK_N / THREAD_N);

  // 全局 A tile 为 [BLOCK_M, BLOCK_K], 转置后按 As[BLOCK_K, BLOCK_M] 存入 LDS.
  // B tile 按 Bs[BLOCK_K, BLOCK_N] 存储, 与全局 B 的行列顺序一致.
  // 固定 k 时, A 的不同 row 在 As 的同一行内连续, 便于读取线程所需的 THREAD_M 个值.
  __shared__ float A_shared[BLOCK_M * BLOCK_K];
  __shared__ float B_shared[BLOCK_K * BLOCK_N];

  // 沿用 V4 的 block 映射, blockIdx.y 对应 row, blockIdx.x 对应 col, 详细解释见 V4.
  const float* A_ptr = A + blockIdx.y * BLOCK_M * K;
  const float* B_ptr = B + blockIdx.x * BLOCK_N;
  float* C_ptr = C + blockIdx.y * BLOCK_M * N + blockIdx.x * BLOCK_N;

  // thread_res[m * THREAD_N + n] 保存线程输出小块的第 [m, n] 项, 跨整个 K 维累加.
  // 固定 k 时, reg_M 缓存 A 的 THREAD_M 个 row, reg_N 缓存 B 的 THREAD_N 个 col.
  float thread_res[THREAD_M * THREAD_N] = {0.0f};
  float reg_M[THREAD_M] = {0.0f};
  float reg_N[THREAD_N] = {0.0f};

  // load_rowA/B 是元素行坐标, load_vec_colA/B 是 float4 列编号.
  // 每个 float4 含 4 个元素, 实际起始列为 load_vec_colA/B * 4.
  // 例中 A tile 每行有 2 个 float4, 256 个线程覆盖 128 行.
  // B tile 每行有 32 个 float4, 256 个线程覆盖 8 行, 各自恰好搬运 1024 个元素.
  const int load_rowA = threadIdx.x / (BLOCK_K / 4);
  const int load_vec_colA = threadIdx.x % (BLOCK_K / 4);
  const int load_rowB = threadIdx.x / (BLOCK_N / 4);
  const int load_vec_colB = threadIdx.x % (BLOCK_N / 4);

  // bkIdx 沿 K 推进, 每轮加载完整的 A, B tile, 计算当前 K 分块的贡献.
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_K) {
    // 沿全局 A 的同一行读取 4 个连续元素, tmp 对应以下起始坐标:
    // row = blockIdx.y * BLOCK_M + load_rowA.
    // col = bkIdx + load_vec_colA * 4.
    float4 tmp = FETCH_FLOAT4(A_ptr[load_rowA * K + load_vec_colA * 4]);
    // tmp 的四个元素分散写入 As 的四行, 固定列为 load_rowA.
    // A tile [load_rowA, load_vec_colA * 4 + i] -> As[load_vec_colA * 4 + i, load_rowA].
    // As 的行宽为 BLOCK_M, 因此索引为 k * BLOCK_M + row.
    A_shared[(load_vec_colA * 4 + 0) * BLOCK_M + load_rowA] = tmp.x;
    A_shared[(load_vec_colA * 4 + 1) * BLOCK_M + load_rowA] = tmp.y;
    A_shared[(load_vec_colA * 4 + 2) * BLOCK_M + load_rowA] = tmp.z;
    A_shared[(load_vec_colA * 4 + 3) * BLOCK_M + load_rowA] = tmp.w;

    // B 保持行主序, 从全局内存读取 float4, 连续写入 Bs 的同一行.
    // Bs[load_rowB, load_vec_colB * 4 : load_vec_colB * 4 + 4] 对应全局 B 的以下起始坐标:
    // row = bkIdx + load_rowB.
    // col = blockIdx.x * BLOCK_N + load_vec_colB * 4.
    FETCH_FLOAT4(B_shared[load_rowB * BLOCK_N + load_vec_colB * 4]) =
        FETCH_FLOAT4(B_ptr[load_rowB * N + load_vec_colB * 4]);
    __syncthreads();

    A_ptr += BLOCK_K;
    B_ptr += BLOCK_K * N;

    for (int k = 0; k < BLOCK_K; k++) {
      // A 转置后, 原 tile 的第 k 列对应 As 的第 k 行.
      for (int i = 0; i < THREAD_M; i++) {
        // reg_M[i] -> As[k, thread_row * THREAD_M + i].
        reg_M[i] = A_shared[k * BLOCK_M + thread_row * THREAD_M + i];
      }
      for (int i = 0; i < THREAD_N; i++) {
        // reg_N[i] -> Bs[k, thread_col * THREAD_N + i].
        reg_N[i] = B_shared[k * BLOCK_N + thread_col * THREAD_N + i];
      }

      // reg_M [THREAD_M] 与 reg_N [THREAD_N] 做外积, 累加到输出小块 [THREAD_M, THREAD_N].
      // 每个 A 值复用于 THREAD_N 个 col, 每个 B 值复用于 THREAD_M 个 row.
      for (int m = 0; m < THREAD_M; m++) {
        for (int n = 0; n < THREAD_N; n++) {
          thread_res[m * THREAD_N + n] += reg_M[m] * reg_N[n];
        }
      }
    }
    __syncthreads();
  }

  // 每次将同一输出行的 4 个结果打包为 float4, 沿 col 方向写回 C.
  // 例中 THREAD_N=8, n 取 0, 4, 每线程每行执行两次向量写入.
  for (int m = 0; m < THREAD_M; m++) {
    for (int n = 0; n < THREAD_N; n += 4) {
      // tmp 对应 thread_res[m, n:n+4], 结果直接覆盖 C 的以下连续 4 列:
      // row = blockIdx.y * BLOCK_M + thread_row * THREAD_M + m.
      // col = blockIdx.x * BLOCK_N + thread_col * THREAD_N + n.
      float4 tmp;
      tmp.x = thread_res[m * THREAD_N + n + 0];
      tmp.y = thread_res[m * THREAD_N + n + 1];
      tmp.z = thread_res[m * THREAD_N + n + 2];
      tmp.w = thread_res[m * THREAD_N + n + 3];
      FETCH_FLOAT4(C_ptr[(thread_row * THREAD_M + m) * N + thread_col * THREAD_N + n]) = tmp;
    }
  }
}

// V9: 参数调优, 泛化 V6 的向量搬运循环, 每线程沿 row, col 计算多个寄存器输出小块.
// 参考 9_kernel_autotuned.cuh, 沿用 V6 的 float4 加载, A 转置布局与外积累加.
// 启动 grid(N / BLOCK_N, M / BLOCK_M), block(256), K9_NUM_THREADS=256.
// 16 * 16 表示逻辑线程网格, gfx1151 上一个 block 包含 8 个 wave32.
// scripts/sgemm_v9_autotune.sh 扫描五个 tile 参数, 检查整除条件与设备 LDS 上限.
// FP32, M=N=K=5120 时, <128, 256, 8, 4, 4, 256> 复测约 26.23 ms.
// <256, 128, 8, 4, 4, 256> 基本持平, 两组均比本次 V6 对照快约 1.26 倍.
// 以上参数顺序为 <BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N, K9_NUM_THREADS>.
// M, N, K 须覆盖完整 tile, 向量地址须对齐, 其他 shape 的参数需重新测量.
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N, int K9_NUM_THREADS>
__global__ void __launch_bounds__(K9_NUM_THREADS)
    sgemm_v9_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                    int M, int N, int K) {
  constexpr int TILE_M = THREAD_M * 16;
  constexpr int TILE_N = THREAD_N * 16;

  // 全部线程共同覆盖基础 tile [TILE_M, TILE_N], 沿 row, col 重复覆盖整个 block tile.
  // <128, 256, 8, 4, 4, 256> 中基础 tile 为 [64, 64], M_ITERS=2, N_ITERS=4.
  constexpr int M_ITERS = BLOCK_M / TILE_M;
  constexpr int N_ITERS = BLOCK_N / TILE_N;
  const int thread_col = threadIdx.x % (TILE_N / THREAD_N);
  const int thread_row = threadIdx.x / (TILE_N / THREAD_N);

  // A 转置后按 As[BLOCK_K, BLOCK_M] 存储, B 保持 Bs[BLOCK_K, BLOCK_N] 布局.
  __shared__ float A_shared[BLOCK_M * BLOCK_K];
  __shared__ float B_shared[BLOCK_K * BLOCK_N];

  // 沿用 V4 的 block 映射, blockIdx.y 对应 row, blockIdx.x 对应 col, 详细解释见 V4.
  const float* A_ptr = A + blockIdx.y * BLOCK_M * K;
  const float* B_ptr = B + blockIdx.x * BLOCK_N;
  float* C_ptr = C + blockIdx.y * BLOCK_M * N + blockIdx.x * BLOCK_N;

  // 每线程负责 M_ITERS * N_ITERS 个 [THREAD_M, THREAD_N] 输出小块.
  // 结果按 [M_ITERS * THREAD_M, N_ITERS * THREAD_N] 展开, 跨整个 K 维累加.
  // 例中每线程保存 8 个 [4, 4] 小块, 共 128 个结果, 更大 tile 会增加寄存器压力.
  float thread_res[M_ITERS * N_ITERS * THREAD_M * THREAD_N] = {0.0f};
  float reg_M[THREAD_M] = {0.0f};
  float reg_N[THREAD_N] = {0.0f};

  // load_rowA/B 为加载行坐标, load_vec_colA/B 为 float4 列编号, 元素起始列为编号 * 4.
  // stride 表示全部线程一轮搬运覆盖的行数, 沿 row 跨步填满 A, B tile.
  // 例中 stride_A=128, stride_B=4, A 加载一轮, B 加载两轮, 不再要求两者加载量相同.
  const int load_rowA = threadIdx.x / (BLOCK_K / 4);
  const int load_vec_colA = threadIdx.x % (BLOCK_K / 4);
  const int load_rowB = threadIdx.x / (BLOCK_N / 4);
  const int load_vec_colB = threadIdx.x % (BLOCK_N / 4);
  constexpr int stride_A = (K9_NUM_THREADS * 4) / BLOCK_K;
  constexpr int stride_B = (K9_NUM_THREADS * 4) / BLOCK_N;

  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_K) {
    // offset 沿 A tile 的 row 方向跨步, 向量读取后转置写入 As.
    // 原 A tile [load_rowA + offset, load_vec_colA * 4 + i]
    // -> As[load_vec_colA * 4 + i, load_rowA + offset].
    for (int offset = 0; offset + stride_A <= BLOCK_M; offset += stride_A) {
      const float4 tmp = FETCH_FLOAT4(A_ptr[(load_rowA + offset) * K + load_vec_colA * 4]);
      A_shared[(load_vec_colA * 4 + 0) * BLOCK_M + load_rowA + offset] = tmp.x;
      A_shared[(load_vec_colA * 4 + 1) * BLOCK_M + load_rowA + offset] = tmp.y;
      A_shared[(load_vec_colA * 4 + 2) * BLOCK_M + load_rowA + offset] = tmp.z;
      A_shared[(load_vec_colA * 4 + 3) * BLOCK_M + load_rowA + offset] = tmp.w;
    }

    // offset 沿 B tile 的 row 方向跨步, 保持行主序, 连续读写 float4.
    for (int offset = 0; offset + stride_B <= BLOCK_K; offset += stride_B) {
      FETCH_FLOAT4(B_shared[(load_rowB + offset) * BLOCK_N + load_vec_colB * 4]) =
          FETCH_FLOAT4(B_ptr[(load_rowB + offset) * N + load_vec_colB * 4]);
    }
    __syncthreads();

    // m_iter, n_iter 定位基础 tile, 每线程在其中计算自己的 [THREAD_M, THREAD_N] 小块.
    // 例中同一线程沿 row 间隔 64 行, 沿 col 间隔 64 列, 覆盖 2 * 4 个小块.
    for (int m_iter = 0; m_iter < M_ITERS; m_iter++) {
      for (int n_iter = 0; n_iter < N_ITERS; n_iter++) {
        for (int k = 0; k < BLOCK_K; k++) {
          for (int i = 0; i < THREAD_M; i++) {
            // reg_M[i] -> As[k, m_iter * TILE_M + thread_row * THREAD_M + i].
            reg_M[i] = A_shared[k * BLOCK_M + m_iter * TILE_M + thread_row * THREAD_M + i];
          }
          for (int i = 0; i < THREAD_N; i++) {
            // reg_N[i] -> Bs[k, n_iter * TILE_N + thread_col * THREAD_N + i].
            reg_N[i] = B_shared[k * BLOCK_N + n_iter * TILE_N + thread_col * THREAD_N + i];
          }

          // 固定 k, A 的 THREAD_M 个 row 与 B 的 THREAD_N 个 col 做外积.
          // 结果累加到当前小块, thread_res 的行宽为 N_ITERS * THREAD_N.
          for (int m = 0; m < THREAD_M; m++) {
            for (int n = 0; n < THREAD_N; n++) {
              const int index =
                  (m_iter * THREAD_M + m) * (N_ITERS * THREAD_N) + n_iter * THREAD_N + n;
              thread_res[index] += reg_M[m] * reg_N[n];
            }
          }
        }
      }
    }
    __syncthreads();

    A_ptr += BLOCK_K;
    B_ptr += BLOCK_K * N;
  }

  // 先移动到当前基础 tile 的 C 起点, 再沿每个线程输出小块的 col 方向写回 float4.
  for (int m_iter = 0; m_iter < M_ITERS; m_iter++) {
    for (int n_iter = 0; n_iter < N_ITERS; n_iter++) {
      float* C_tile = C_ptr + m_iter * TILE_M * N + n_iter * TILE_N;
      for (int m = 0; m < THREAD_M; m++) {
        for (int n = 0; n < THREAD_N; n += 4) {
          // 输出全局 row = blockIdx.y * BLOCK_M + m_iter * TILE_M + thread_row * THREAD_M + m.
          // 输出全局 col = blockIdx.x * BLOCK_N + n_iter * TILE_N + thread_col * THREAD_N + n.
          const int index = (m_iter * THREAD_M + m) * (N_ITERS * THREAD_N) + n_iter * THREAD_N + n;
          float4 tmp;
          tmp.x = thread_res[index + 0];
          tmp.y = thread_res[index + 1];
          tmp.z = thread_res[index + 2];
          tmp.w = thread_res[index + 3];
          FETCH_FLOAT4(C_tile[(thread_row * THREAD_M + m) * N + thread_col * THREAD_N + n]) = tmp;
        }
      }
    }
  }
}

// V10: wave 分块, block tile 先划分为 wave tile, 再分配各线程的寄存器输出小块.
// 参考 10_kernel_warptiling.cuh, 按 gfx1151 的 wave32 分配线程, 沿用 V6 的向量加载与 A 转置.
// 启动 grid(N / BLOCK_N, M / BLOCK_M), block(BLOCK_SIZE).
// gfx1151, FP32, M=N=K=5120 采用 <128, 128, 16, 32, 128, 2, 4, 4, 128>, 复测约 21.81 ms.
// WAVE_N_ITERS 指定 wave tile 沿 col 划分的子块数, WAVE_M_ITERS 由覆盖面积推导.
// BLOCK_SIZE = (BLOCK_M / WAVE_M) * (BLOCK_N / WAVE_N) * 32.
// 分块, 子块与加载 stride 须整除并完整覆盖, M, N, K 覆盖完整 tile, 向量地址须对齐.
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WAVE_M, int WAVE_N, int WAVE_N_ITERS,
          int THREAD_M, int THREAD_N, int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE)
    sgemm_v10_kernel(const float* __restrict__ A, const float* __restrict__ B,
                     float* __restrict__ C, int M, int N, int K) {
  constexpr int WAVE_SIZE = 32;

  // wave_row, wave_col 定位 block 内的 wave tile, 相邻 wave 沿 col 方向排列.
  // 例中 block tile [128, 128] 划分为 4 * 1 个 wave tile [32, 128].
  const int wave_id = threadIdx.x / WAVE_SIZE;
  const int wave_col = wave_id % (BLOCK_N / WAVE_N);
  const int wave_row = wave_id / (BLOCK_N / WAVE_N);

  // 一个子块由 32 个线程各计算 [THREAD_M, THREAD_N] 覆盖, 面积为 32 * THREAD_M * THREAD_N.
  // 例中 WAVE_M_ITERS=4, WAVE_N_ITERS=2, wave tile 分成 4 * 2 个 [8, 64] 子块.
  constexpr int WAVE_M_ITERS = WAVE_M * WAVE_N / (WAVE_SIZE * THREAD_M * THREAD_N * WAVE_N_ITERS);
  constexpr int SUB_M = WAVE_M / WAVE_M_ITERS;
  constexpr int SUB_N = WAVE_N / WAVE_N_ITERS;

  // lane_id 定位 wave 内线程, thread_row, thread_col 定位子块内的线程输出小块.
  // 例中 32 个线程组成 [2, 16] 的逻辑网格, 每线程在一个子块中计算 [4, 4] 个输出.
  const int lane_id = threadIdx.x % WAVE_SIZE;
  const int thread_col = lane_id % (SUB_N / THREAD_N);
  const int thread_row = lane_id / (SUB_N / THREAD_N);

  // 全 block 协作加载, A 转置为 As[BLOCK_K, BLOCK_M], B 保持 Bs[BLOCK_K, BLOCK_N] 布局.
  __shared__ float A_shared[BLOCK_M * BLOCK_K];
  __shared__ float B_shared[BLOCK_K * BLOCK_N];

  // 沿用 V4 的 block 映射, A_ptr, B_ptr 定位 block tile, C_ptr 直接定位本 wave 的输出起点.
  const float* A_ptr = A + blockIdx.y * BLOCK_M * K;
  const float* B_ptr = B + blockIdx.x * BLOCK_N;
  float* C_ptr =
      C + (blockIdx.y * BLOCK_M + wave_row * WAVE_M) * N + blockIdx.x * BLOCK_N + wave_col * WAVE_N;

  // 加载坐标按 float4 列宽展开全部线程, 沿 row 跨步填满两个 LDS tile.
  // 例中 stride_A=32, stride_B=4, A, B 各执行四轮加载.
  const int load_rowA = threadIdx.x / (BLOCK_K / 4);
  const int load_vec_colA = threadIdx.x % (BLOCK_K / 4);
  const int load_rowB = threadIdx.x / (BLOCK_N / 4);
  const int load_vec_colB = threadIdx.x % (BLOCK_N / 4);
  constexpr int stride_A = BLOCK_SIZE * 4 / BLOCK_K;
  constexpr int stride_B = BLOCK_SIZE * 4 / BLOCK_N;

  // 结果按 [WAVE_M_ITERS * THREAD_M, WAVE_N_ITERS * THREAD_N] 展开, 跨整个 K 维累加.
  // 固定 k, reg_M 与 reg_N 缓存本线程在所有 wave 子块中需要的 A, B 值.
  // 例中每线程保存 [16, 8] 共 128 个结果, reg_M 含 16 个值, reg_N 含 8 个值.
  float thread_res[WAVE_M_ITERS * THREAD_M * WAVE_N_ITERS * THREAD_N] = {0.0f};
  float reg_M[WAVE_M_ITERS * THREAD_M] = {0.0f};
  float reg_N[WAVE_N_ITERS * THREAD_N] = {0.0f};

  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_K) {
    // 原 A tile [load_rowA + offset, load_vec_colA * 4 + i]
    // -> As[load_vec_colA * 4 + i, load_rowA + offset].
    for (int offset = 0; offset + stride_A <= BLOCK_M; offset += stride_A) {
      const float4 tmp = FETCH_FLOAT4(A_ptr[(load_rowA + offset) * K + load_vec_colA * 4]);
      A_shared[(load_vec_colA * 4 + 0) * BLOCK_M + load_rowA + offset] = tmp.x;
      A_shared[(load_vec_colA * 4 + 1) * BLOCK_M + load_rowA + offset] = tmp.y;
      A_shared[(load_vec_colA * 4 + 2) * BLOCK_M + load_rowA + offset] = tmp.z;
      A_shared[(load_vec_colA * 4 + 3) * BLOCK_M + load_rowA + offset] = tmp.w;
    }
    // Bs[load_rowB + offset, load_vec_colB * 4 : load_vec_colB * 4 + 4], 连续读写同一行.
    for (int offset = 0; offset + stride_B <= BLOCK_K; offset += stride_B) {
      FETCH_FLOAT4(B_shared[(load_rowB + offset) * BLOCK_N + load_vec_colB * 4]) =
          FETCH_FLOAT4(B_ptr[(load_rowB + offset) * N + load_vec_colB * 4]);
    }
    __syncthreads();

    for (int k = 0; k < BLOCK_K; k++) {
      // 固定 k, 先读取所有 row 子块的 A 值与所有 col 子块的 B 值, 再复用于各个外积.
      for (int m_iter = 0; m_iter < WAVE_M_ITERS; m_iter++) {
        for (int i = 0; i < THREAD_M; i++) {
          // reg_M[m_iter * THREAD_M + i]
          // -> As[k, wave_row * WAVE_M + m_iter * SUB_M + thread_row * THREAD_M + i].
          reg_M[m_iter * THREAD_M + i] = A_shared[k * BLOCK_M + wave_row * WAVE_M + m_iter * SUB_M +
                                                  thread_row * THREAD_M + i];
        }
      }
      for (int n_iter = 0; n_iter < WAVE_N_ITERS; n_iter++) {
        for (int i = 0; i < THREAD_N; i++) {
          // reg_N[n_iter * THREAD_N + i]
          // -> Bs[k, wave_col * WAVE_N + n_iter * SUB_N + thread_col * THREAD_N + i].
          reg_N[n_iter * THREAD_N + i] = B_shared[k * BLOCK_N + wave_col * WAVE_N + n_iter * SUB_N +
                                                  thread_col * THREAD_N + i];
        }
      }

      // 每个 row 子块的 A 向量与每个 col 子块的 B 向量做外积, 累加对应输出小块.
      // thread_res 的行宽为 WAVE_N_ITERS * THREAD_N, reg_M 跨 col 子块复用, reg_N 跨 row 子块复用.
      for (int m_iter = 0; m_iter < WAVE_M_ITERS; m_iter++) {
        for (int n_iter = 0; n_iter < WAVE_N_ITERS; n_iter++) {
          for (int m = 0; m < THREAD_M; m++) {
            for (int n = 0; n < THREAD_N; n++) {
              const int index =
                  (m_iter * THREAD_M + m) * (WAVE_N_ITERS * THREAD_N) + n_iter * THREAD_N + n;
              thread_res[index] += reg_M[m_iter * THREAD_M + m] * reg_N[n_iter * THREAD_N + n];
            }
          }
        }
      }
    }
    __syncthreads();

    A_ptr += BLOCK_K;
    B_ptr += BLOCK_K * N;
  }

  // C_ptr 已含 block 与 wave 偏移, 再叠加子块和线程偏移, 沿 col 方向写回 float4.
  for (int m_iter = 0; m_iter < WAVE_M_ITERS; m_iter++) {
    for (int n_iter = 0; n_iter < WAVE_N_ITERS; n_iter++) {
      float* C_tile = C_ptr + m_iter * SUB_M * N + n_iter * SUB_N;
      for (int m = 0; m < THREAD_M; m++) {
        for (int n = 0; n < THREAD_N; n += 4) {
          // C_tile[(thread_row * THREAD_M + m) * N + thread_col * THREAD_N + n] -> C[row, col].
          // row = blockIdx.y * BLOCK_M + wave_row * WAVE_M
          //       + m_iter * SUB_M + thread_row * THREAD_M + m.
          // col = blockIdx.x * BLOCK_N + wave_col * WAVE_N
          //       + n_iter * SUB_N + thread_col * THREAD_N + n.
          const int index =
              (m_iter * THREAD_M + m) * (WAVE_N_ITERS * THREAD_N) + n_iter * THREAD_N + n;
          float4 tmp;
          tmp.x = thread_res[index + 0];
          tmp.y = thread_res[index + 1];
          tmp.z = thread_res[index + 2];
          tmp.w = thread_res[index + 3];
          FETCH_FLOAT4(C_tile[(thread_row * THREAD_M + m) * N + thread_col * THREAD_N + n]) = tmp;
        }
      }
    }
  }
}
