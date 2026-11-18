// Portions adapted from AMD's RDNA FP16 GEMM examples.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

/*
 * HGEMM: C[row,col] = sum_k A[row,k] * B[k,col], FP16 输入, 累加和输出.
 * A[M,K] 行主序, B[K,N] 和 C[M,N] 列主序, 地址分别为:
 * A[row*K+k], B[col*K+k], C[col*M+row].
 *
 * 参考 AMD 的 RDNA FP16 GEMM 优化博客及 stage01-stage07 配套源码:
 * https://rocm.blogs.amd.com/software-tools-optimization/rdna-fp16-gemm/README.html
 * https://github.com/ROCm/rocm-blogs/tree/release/blogs/software-tools-optimization/rdna-fp16-gemm/src
 *
 * BLOCK_M/BLOCK_N: block 输出 tile 的行数/列数; BLOCK_K: 一轮 K tile 的长度.
 * WAVE_M/WAVE_N: 一个 wave 输出 tile 的行数/列数, 均为 16 的整数倍.
 * block 线程数为 (BLOCK_M / WAVE_M) * (BLOCK_N / WAVE_N) * 32.
 * LDS_PAD: LDS 行末的 FP16 填充量; GROUP_SIZE: 一组输出 tile 的行数.
 * LDS 将 A tile 存为 As[row,k], B tile 存为 Bs[col,k], 两者均沿 K 连续.
 * gfx1151 使用 wave32, 每条 WMMA 计算一个 16x16x16 的矩阵乘加.
 * V1-V3 要求完整 tile; V4-V8 对输入边缘补零, 仅写回有效输出.
 *
 * V1: 标量搬运 -> LDS -> fragment -> WMMA -> C.
 * V2-V6: TileCopy 预取寄存器 -> pipeline 提交 LDS -> multiply_tile -> store_wave.
 * V7-V8: interleaved 用 fold_step 在 WMMA 间穿插下一轮预取和 fragment 读取.
 */

// half16 是一个 lane 持有的 16 个 FP16 寄存器元素, 并不是整个 16x16 矩阵.
using half16 = _Float16 __attribute__((ext_vector_type(16)));

/*
 * 从 LDS 的 [row, k] 布局准备一个 16x16 操作数的当前 lane 部分.
 * 对 A, row 是输入行; 对 Bs, row 是原始 B 的列, 两者都沿 K 连续.
 * lane 0-15 分别读 16 行, lane 16-31 重复这 16 行, 每个 lane 沿 K 读 16 个值.
 * 例如 lane=3 与 lane=19 都读 tile[row+3, k:k+16].
 * stride 是 LDS 一行占用的元素数, 包含 padding; 全局 A/B 的 leading dimension 为 K.
 */
__device__ __forceinline__ half16 load_wmma_fragment(const __half* tile, int row, int k, int stride,
                                                     int lane) {
  half16 fragment;
  __builtin_memcpy(&fragment, tile + (row + lane % 16) * stride + k, sizeof(fragment));
  return fragment;
}

// 32 个 lane 共同完成 D[16,16] = A[16,16] * B[16,16] + acc[16,16].
// a, b, acc 都只是当前 lane 的部分, 不能用单个线程独立完成这条矩阵指令.
// FP16 累加, opsel=false 使用每个累加器 VGPR 的低 16 bits, 即 half16 的偶数槽位.
__device__ __forceinline__ half16 wmma_mma(half16 a, half16 b, half16 acc) {
  return __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, acc, false);
}

// 将当前 lane 的 8 个结果写入列主序 C, 有效结果位于 acc 的偶数槽位.
// lane=3 写 col0+3 的 row0+0,2,...,14; lane=19 写同一列的 row0+1,3,...,15.
// CHECK_BOUNDS 仅限制写回范围, 所有 lane 已共同完成当前 fragment 的 WMMA.
template <bool CHECK_BOUNDS>
__device__ __forceinline__ void store_wmma_fragment(__half* C, half16 acc, int row0, int col0,
                                                    int M, int N, int lane) {
  const int col = col0 + lane % 16;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int row = row0 + 2 * e + lane / 16;
    if constexpr (CHECK_BOUNDS) {
      if (row < M && col < N) C[col * M + row] = acc[2 * e];
    } else {
      C[col * M + row] = acc[2 * e];
    }
  }
}

namespace hgemm {
using half8 = _Float16 __attribute__((ext_vector_type(8)));
using raw128 = unsigned int __attribute__((ext_vector_type(4)));

// A, B 的 LDS 都沿 K 连续, B 的 row 对应输出 col.
// XOR 以完整的 16 元素 fragment 分组, BLOCK_K / 16 须为二次幂.
// padding: 逻辑 As[row,k] 的物理地址是 row * (BLOCK_K + LDS_PAD) + k.
// 例如 BLOCK_K=64, LDS_PAD=8, 每行保留 64 个有效值, 接着跳过 8 个填充值.
// XOR: BLOCK_K=64 时每行有 4 个 fragment 组, row=1 将组 0,1,2,3 放到 1,0,3,2.
// store 与 fragment 使用相同映射, 逻辑矩阵不变, 单个 lane 的 16 个 K 值仍连续.
template <int BLOCK_K, int LDS_PAD, bool SWIZZLE = false>
struct LdsLayout {
  static constexpr int STRIDE = BLOCK_K + LDS_PAD;

  __device__ static __forceinline__ int offset(int row, int k) {
    if constexpr (SWIZZLE) {
      return row * STRIDE + ((k / 16) ^ (row % (BLOCK_K / 16))) * 16 + k % 16;
    } else {
      return row * STRIDE + k;
    }
  }

  __device__ static __forceinline__ half16 fragment(const __half* tile, int row, int k, int lane) {
    if constexpr (!SWIZZLE) {
      return load_wmma_fragment(tile, row, k, STRIDE, lane);
    } else {
      half16 value;
      __builtin_memcpy(&value, tile + offset(row + lane % 16, k), sizeof(value));
      return value;
    }
  }
};

// descriptor 包含整个分配范围, 硬件对范围外的 raw buffer 加载返回零.
// bytes 为整个矩阵的字节数; 0x31004000 是参考代码用于 gfx11 的描述符配置字.
// 这个范围只能判断是否越过分配末尾, 不能判断是否跨过 A 的一行或 B 的一列.
__device__ __forceinline__ __amdgpu_buffer_rsrc_t make_buffer(const void* ptr, unsigned bytes) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(ptr), 0, bytes, 0x31004000);
}

/*
 * 为一个 block 分配 A[BLOCK_M,BLOCK_K] 和 B[BLOCK_K,BLOCK_N] 的搬运任务.
 * 一次搬运 8 个 FP16 = 16 bytes, chunk 表示同一线程第几次搬运.
 * 将 tile 按 8 元素向量展开, 当前线程的向量编号为 thread + chunk * THREADS.
 * A 的每行和 B 的每列都沿 K 连续, 直接向量写入 As[BLOCK_M,BLOCK_K], Bs[BLOCK_N,BLOCK_K].
 *
 * V5 默认 BM=128, BN=96, BK=64, THREADS=128:
 * A 有 128*64/8=1024 个向量, 每线程搬运 8 次; B 有 64*96/8=768 个, 搬运 6 次.
 * K_VECTORS=8, 表示 LDS 一行中沿 K 排列的向量数, 不包含 padding.
 * As/Bs 使用相同的搬运坐标, 但 LDS row 分别对应 A 的 row 和 B 的 col.
 */
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREADS, int LDS_PAD, bool BUFFER_LOAD,
          bool SWIZZLE = false>
struct TileCopy {
  using Layout = LdsLayout<BLOCK_K, LDS_PAD, SWIZZLE>;
  static constexpr int K_VECTORS = BLOCK_K / 8;
  static constexpr int CHUNKS_A = (BLOCK_M * BLOCK_K / 8 + THREADS - 1) / THREADS;
  static constexpr int CHUNKS_B = (BLOCK_K * BLOCK_N / 8 + THREADS - 1) / THREADS;
  const __half* A;
  const __half* B;
  int M, N, K;
  int row0, col0;
  // 首个 chunk 的 LDS 行号与 K 向量编号, vec_k 乘 8 后才是元素坐标.
  int load_row, vec_k;
  bool aligned_a, aligned_b;
  __amdgpu_buffer_rsrc_t resource_a, resource_b;

  __device__ TileCopy(const __half* A, const __half* B, int M, int N, int K, int row0, int col0,
                      int thread)
      : A(A),
        B(B),
        M(M),
        N(N),
        K(K),
        row0(row0),
        col0(col0),
        load_row(thread / K_VECTORS),
        vec_k(thread % K_VECTORS),
        aligned_a(reinterpret_cast<unsigned long long>(A) % 16 == 0),
        aligned_b(reinterpret_cast<unsigned long long>(B) % 16 == 0) {
    if constexpr (BUFFER_LOAD) {
      // A 和 B 分别持有覆盖整个矩阵的 descriptor, 加载时传入相对起点的字节偏移.
      resource_a = make_buffer(A, unsigned(M) * unsigned(K) * sizeof(__half));
      resource_b = make_buffer(B, unsigned(K) * unsigned(N) * sizeof(__half));
    }
  }

  __device__ __forceinline__ void position(int chunk, int& row, int& k) const {
    // 等价于 vector=thread+chunk*THREADS, row=vector/K_VECTORS, k=vector%K_VECTORS*8.
    // 构造时保存起始商和余数, 展开的 chunk 只增加固定步长.
    // V5 的 thread=17: chunk=0 -> As/Bs[2,8:16], chunk=1 -> As/Bs[18,8:16].
    const int vector_col = vec_k + chunk * THREADS % K_VECTORS;
    row = load_row + chunk * THREADS / K_VECTORS + vector_col / K_VECTORS;
    k = vector_col % K_VECTORS * 8;
  }

  __device__ __forceinline__ half8 load_a(int chunk, int k0) const {
    int row, k;
    position(chunk, row, k);
    half8 value{};
    if (row >= BLOCK_M) return value;
    if constexpr (BUFFER_LOAD) {
      if (row0 + row >= M) return value;
      // A[(row0+row), k0+k:k0+k+8], 一行的全局步长为 K.
      const int offset = (row0 + row) * K + k0 + k;
      if (k0 + k + 8 <= K && aligned_a && offset % 8 == 0) {
        // 完整且对齐的向量使用一次 128-bit raw load, 偏移单位为 bytes.
        const raw128 raw = __builtin_amdgcn_raw_buffer_load_b128(
            resource_a, unsigned(offset) * sizeof(__half), 0, 0);
        __builtin_memcpy(&value, &raw, sizeof(value));
      } else {
        // K 尾部或地址未对齐时逐元素读取, value 中未覆盖的位置保持零.
        // 只依赖 descriptor 会把下一行的值误当成当前行的 K 尾部.
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          if (k0 + k + e < K) value[e] = A[offset + e];
        }
      }
    } else {
      // V2/V3 的 launch 已要求完整 tile 和向量对齐, 直接复制 16 bytes 到寄存器.
      __builtin_memcpy(&value, A + (row0 + row) * K + k0 + k, sizeof(value));
    }
    return value;
  }

  __device__ __forceinline__ half8 load_b(int chunk, int k0) const {
    int col, k;
    position(chunk, col, k);
    half8 value{};
    if (col >= BLOCK_N) return value;
    if constexpr (BUFFER_LOAD) {
      if (col0 + col >= N) return value;
      // 列主序 B[k0+k:k0+k+8, col0+col], 一列的全局步长为 K.
      const int offset = (col0 + col) * K + k0 + k;
      if (k0 + k + 8 <= K && aligned_b && offset % 8 == 0) {
        const raw128 raw = __builtin_amdgcn_raw_buffer_load_b128(
            resource_b, unsigned(offset) * sizeof(__half), 0, 0);
        __builtin_memcpy(&value, &raw, sizeof(value));
      } else {
        // K 尾部逐元素填零, 避免完整向量跨到 B 的下一列.
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          if (k0 + k + e < K) value[e] = B[offset + e];
        }
      }
    } else {
      __builtin_memcpy(&value, B + (col0 + col) * K + k0 + k, sizeof(value));
    }
    return value;
  }

  __device__ __forceinline__ void store_a(__half* tile, half8 value, int chunk) const {
    // A 沿 K 读出的 8 个值也沿 LDS 的 K 写入, 保持整段向量复制.
    int row, k;
    position(chunk, row, k);
    if (row < BLOCK_M) __builtin_memcpy(tile + Layout::offset(row, k), &value, sizeof(value));
  }

  __device__ __forceinline__ void store_b(__half* tile, half8 value, int chunk) const {
    // B 的列与 Bs 的 row 都沿 K 连续, 无需转置, 一次写入 8 个 FP16.
    // Bs[col,k:k+8] <- B[k0+k:k0+k+8,col0+col].
    int col, k;
    position(chunk, col, k);
    if (col < BLOCK_N) __builtin_memcpy(tile + Layout::offset(col, k), &value, sizeof(value));
  }

  __device__ __forceinline__ void prefetch(int k0, half8* staged_a, half8* staged_b) const {
    // 只读 Global -> 寄存器, 当前 LDS 不变, staged_* 随后交给 commit.
#pragma unroll
    for (int chunk = 0; chunk < CHUNKS_A; ++chunk) {
      staged_a[chunk] = load_a(chunk, k0);
    }
#pragma unroll
    for (int chunk = 0; chunk < CHUNKS_B; ++chunk) {
      staged_b[chunk] = load_b(chunk, k0);
    }
  }

  __device__ __forceinline__ void commit(__half* tile_a, __half* tile_b, const half8* staged_a,
                                         const half8* staged_b) const {
    // 只写寄存器 -> LDS, A/B 都应用相同的 padding/swizzle 映射.
#pragma unroll
    for (int chunk = 0; chunk < CHUNKS_A; ++chunk) {
      store_a(tile_a, staged_a[chunk], chunk);
    }
#pragma unroll
    for (int chunk = 0; chunk < CHUNKS_B; ++chunk) {
      store_b(tile_b, staged_b[chunk], chunk);
    }
  }
};

// 线性 block 在 GROUP_SIZE 行内向下推进, 再切换列, 末尾短组使用实际行数.
// 例如一组有 2 行, 3 列 tile: block 0..5 -> (row,col)=(0,0),(1,0),(0,1),
// (1,1),(0,2),(1,2). 邻近 block 访问同一个 B 列分块, 再在小组内推进 col.
// 这里只改变 block 对应的矩阵区域, 不改变 wave/lane 内部的计算或搬运坐标.
template <int BLOCK_M, int BLOCK_N, int GROUP_SIZE>
__device__ __forceinline__ void map_block(int M, int N, int& row0, int& col0) {
  const int rows = (M - 1) / BLOCK_M + 1, cols = (N - 1) / BLOCK_N + 1;
  const int block = blockIdx.x + blockIdx.y * gridDim.x;
  // 每组覆盖 GROUP_SIZE * cols 个 tile, 最后一组可能不足 GROUP_SIZE 行.
  const int group = block / (GROUP_SIZE * cols), first_row = group * GROUP_SIZE;
  const int group_rows = rows - first_row < GROUP_SIZE ? rows - first_row : GROUP_SIZE;
  const int local = block - first_row * cols;
  // local % group_rows 在组内向下移动, local / group_rows 在组内向右移动.
  row0 = (first_row + local % group_rows) * BLOCK_M;
  col0 = (local / group_rows) * BLOCK_N;
}

/*
 * 用当前 LDS 中一个 K tile 更新一个 wave 的输出累加器.
 * wave_row/wave_col 是 wave 在 block tile 中的起点, 全局 block 起点由调用方处理.
 * 以 WAVE_M=64, WAVE_N=48, BLOCK_K=64 为例:
 * 一个 wave 有 4x3 个输出 fragment, 每轮 K=16 读 4 个 A, 3 个 B fragment.
 * A 的每个 fragment 与 3 个 B 相乘, B 的每个 fragment 被 4 个 A 复用.
 * 每轮执行 12 条 WMMA, 一个 K tile 共 4 轮, acc[4][3] 跨所有 K tile 持续累加.
 */
template <int WAVE_M, int WAVE_N, int BLOCK_K, int LDS_PAD, bool SWIZZLE = false>
__device__ __forceinline__ void multiply_tile(const __half* tile_a, const __half* tile_b,
                                              int wave_row, int wave_col, int lane,
                                              half16 (&acc)[WAVE_M / 16][WAVE_N / 16]) {
  using Layout = LdsLayout<BLOCK_K, LDS_PAD, SWIZZLE>;
  half16 a[WAVE_M / 16], b[WAVE_N / 16];
#pragma unroll
  for (int k = 0; k < BLOCK_K; k += 16) {
    // a[m] 对应 As[wave_row+16*m:wave_row+16*m+16, k:k+16] 的当前 lane 部分.
#pragma unroll
    for (int m = 0; m < WAVE_M / 16; ++m) {
      a[m] = Layout::fragment(tile_a, wave_row + m * 16, k, lane);
    }
    // b[n] 从 Bs 的 16 行读取, 对应原 B 的 16 列, 每列有 16 个 K 值.
#pragma unroll
    for (int n = 0; n < WAVE_N / 16; ++n) {
      b[n] = Layout::fragment(tile_b, wave_col + n * 16, k, lane);
    }
    // acc[m][n] += A 的第 m 个 16 行分块 * B 的第 n 个 16 列分块.
#pragma unroll
    for (int m = 0; m < WAVE_M / 16; ++m) {
#pragma unroll
      for (int n = 0; n < WAVE_N / 16; ++n) {
        acc[m][n] = wmma_mma(a[m], b[n], acc[m][n]);
      }
    }
  }
}

// 输入边缘的 lane 仍参加所有 WMMA 和同步, 仅在写回时跳过无效坐标.
// row0/col0 在这里已经是 wave 的全局起点, m/n 再定位 wave 内的 16x16 fragment.
template <int WAVE_M, int WAVE_N, bool CHECK_BOUNDS>
__device__ __forceinline__ void store_wave(__half* C, half16 (&acc)[WAVE_M / 16][WAVE_N / 16],
                                           int row0, int col0, int M, int N, int lane) {
#pragma unroll
  for (int m = 0; m < WAVE_M / 16; ++m) {
#pragma unroll
    for (int n = 0; n < WAVE_N / 16; ++n) {
      store_wmma_fragment<CHECK_BOUNDS>(C, acc[m][n], row0 + m * 16, col0 + n * 16, M, N, lane);
    }
  }
}

/*
 * V2-V6 共用的双缓冲主循环, 差别由模板参数和传入的 row0/col0 决定.
 * BUFFER_LOAD=false: V2/V3 的完整 tile 加载; true: V4-V6 的 raw load 和边界处理.
 * LDS_PAD/SWIZZLE 选择物理布局, wave 的逻辑计算区域不变.
 *
 * tile_a/b 是连续 LDS 缓冲区的基址, buffer 选择当前 K tile, 另一槽用于下一轮提交.
 * staged_a/b 只保存下一轮的全局输入, acc 保存从 K=0 开始的部分和.
 * 例如 K 有 3 个 tile: 提交 0 -> 预取 1/计算 0/提交 1 ->
 * 预取 2/计算 1/提交 2 -> 计算 2 -> 写回 C.
 * 这里是软件流水线, 普通全局加载先到寄存器, 并非异步 Global -> LDS 指令.
 */
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WAVE_M, int WAVE_N, int LDS_PAD,
          bool BUFFER_LOAD, bool SWIZZLE = false>
__device__ __forceinline__ void pipeline(const __half* A, const __half* B, __half* C, int M, int N,
                                         int K, int row0, int col0, __half* tile_a,
                                         __half* tile_b) {
  constexpr int THREADS = BLOCK_M / WAVE_M * (BLOCK_N / WAVE_N) * 32;
  constexpr int TILE_A = BLOCK_M * (BLOCK_K + LDS_PAD);
  constexpr int TILE_B = BLOCK_N * (BLOCK_K + LDS_PAD);
  using Copy = TileCopy<BLOCK_M, BLOCK_N, BLOCK_K, THREADS, LDS_PAD, BUFFER_LOAD, SWIZZLE>;
  const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
  // wave 按输出 col 方向优先展开, 每个 wave 再负责 WAVE_M x WAVE_N 区域.
  // V5 的四个 wave 起点为 (0,0), (0,48), (64,0), (64,48).
  const int wave_row = wave / (BLOCK_N / WAVE_N) * WAVE_M;
  const int wave_col = wave % (BLOCK_N / WAVE_N) * WAVE_N;
  const Copy copy(A, B, M, N, K, row0, col0, threadIdx.x);
  // acc 的一个 half16 只保存当前 lane 的 8 个有效输出, 并非一个完整输出矩阵.
  half16 acc[WAVE_M / 16][WAVE_N / 16] = {};
  half8 staged_a[Copy::CHUNKS_A], staged_b[Copy::CHUNKS_B];
  // 固定 LDS 基址加槽偏移, 避免动态指针数组让 fragment 读取退化为 FLAT 指令.
  int buffer = 0;
  const int tiles_k = (K - 1) / BLOCK_K + 1;
  // 首个 K tile 直接提交 LDS, 后续 tile 先暂存寄存器再与当前计算交错.
  copy.prefetch(0, staged_a, staged_b);
  copy.commit(tile_a, tile_b, staged_a, staged_b);
  __syncthreads();
  for (int tile = 0; tile < tiles_k - 1; ++tile) {
    // 读取下一轮输入到 staged_*, 当前 LDS 保持可读, 不会提前覆盖当前数据.
    copy.prefetch((tile + 1) * BLOCK_K, staged_a, staged_b);
    const __half* current_a = tile_a + buffer * TILE_A;
    const __half* current_b = tile_b + buffer * TILE_B;
    multiply_tile<WAVE_M, WAVE_N, BLOCK_K, LDS_PAD, SWIZZLE>(current_a, current_b, wave_row,
                                                             wave_col, lane, acc);
    // 当前计算结束后, 把下一轮输入写到另一份 LDS, 再统一切换 buffer.
    const int next_buffer = buffer ^ 1;
    copy.commit(tile_a + next_buffer * TILE_A, tile_b + next_buffer * TILE_B, staged_a, staged_b);
    __syncthreads();
    buffer = next_buffer;
  }
  // 最后一个 tile 没有下一轮可预取, 只计算并将累加器按全局坐标写回.
  multiply_tile<WAVE_M, WAVE_N, BLOCK_K, LDS_PAD, SWIZZLE>(
      tile_a + buffer * TILE_A, tile_b + buffer * TILE_B, wave_row, wave_col, lane, acc);
  store_wave<WAVE_M, WAVE_N, BUFFER_LOAD>(C, acc, row0 + wave_row, col0 + wave_col, M, N, lane);
}

/*
 * V7/V8 的一轮计算上下文, 将地址和寄存器数组交给每个编译期 STEP.
 * 默认 wave tile=64x48, K tile=64: TILES_M=4, TILES_N=3, K_SLICES=4, STEPS=48.
 * a[Slice*TILES_M+Row] 和 b[Slice*TILES_N+Col] 保存当前 lane 的操作数 fragment.
 * acc[Row][Col] 在所有 slice 之间复用, 保存同一个输出 fragment 的累加结果.
 * lane_a/b 已包含 wave 起点和 lane%16, read_a/b 只需增加固定的 fragment 偏移.
 */
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WAVE_M, int WAVE_N, int LDS_PAD>
struct FoldContext {
  static constexpr int TILES_M = WAVE_M / 16, TILES_N = WAVE_N / 16;
  static constexpr int K_SLICES = BLOCK_K / 16, STEPS = K_SLICES * TILES_M * TILES_N;
  static constexpr int THREADS = BLOCK_M / WAVE_M * (BLOCK_N / WAVE_N) * 32;
  using Copy = TileCopy<BLOCK_M, BLOCK_N, BLOCK_K, THREADS, LDS_PAD, true>;
  const Copy& copy;
  // next_k 是下一轮 K tile 的全局起点, lane_a/b 指向当前 LDS tile 的 lane 起点.
  int next_k;
  const __half* lane_a;
  const __half* lane_b;
  half16* a;
  half16* b;
  half16 (*acc)[TILES_N];
  half8* staged_a;
  half8* staged_b;

  template <int Row, int Slice>
  __device__ __forceinline__ void read_a() const {
    // 跳过 Row 个 16 行分块和 Slice 个 16 列 K 分块, 一个 lane 连续读 16 个 K 值.
    constexpr int OFFSET = Row * 16 * (BLOCK_K + LDS_PAD) + Slice * 16;
    __builtin_memcpy(&a[Slice * TILES_M + Row], lane_a + OFFSET, sizeof(half16));
  }

  template <int Col, int Slice>
  __device__ __forceinline__ void read_b() const {
    // Bs 的逻辑 row 是原 B 的 col, Col 每增加 1 就跳过 16 个输出列.
    constexpr int OFFSET = Col * 16 * (BLOCK_K + LDS_PAD) + Slice * 16;
    __builtin_memcpy(&b[Slice * TILES_N + Col], lane_b + OFFSET, sizeof(half16));
  }
};

template <int STEP, int TILES_M, int TILES_N>
struct FoldPosition {
  // 默认每个 slice 有 4*3=12 条 WMMA, N_TILE 变化最快, 然后 M_TILE, 最后 SLICE.
  // STEP=0,1,2 -> (slice,m,n)=(0,0,0),(0,0,1),(0,0,2).
  // STEP=3 -> (0,1,0), STEP=12 -> (1,0,0), STEP=47 -> (3,3,2).
  static constexpr int M_TILE = STEP / TILES_N % TILES_M;
  static constexpr int N_TILE = STEP % TILES_N;
  static constexpr int SLICE = STEP / (TILES_M * TILES_N);
};

/*
 * 一个 STEP 对应一条 WMMA, 递归在编译期展开, 不产生运行时递归调用.
 * 可以先把它理解为按 slice -> m -> n 遍历的三重循环, 再看提前加载的安排.
 * PREFETCH=true 时顺便把下一轮的 Global 数据读到 staged_*, 当前 LDS 不变.
 * 函数先准备下一条 WMMA 要用的 fragment, 再用 Here 对应的 fragment 更新 acc.
 * 第一次调用前, interleaved 已经读好 a[0], b[0], 因此当前指令的输入总是可用.
 */
template <int STEP, bool PREFETCH, typename Context>
__device__ __forceinline__ void fold_step(const Context& ctx) {
  using Here = FoldPosition<STEP, Context::TILES_M, Context::TILES_N>;
  if constexpr (PREFETCH) {
    // STEP 0,2,... 预取 A chunk 0,1,...; STEP 1,3,... 预取 B chunk 0,1,... .
    // 默认 A 有 8 个 chunk, B 有 6 个, 全部预取完成后仍继续当前 tile 的 WMMA.
    constexpr int CHUNK = STEP / 2;
    if constexpr (STEP % 2 == 0 && CHUNK < Context::Copy::CHUNKS_A) {
      ctx.staged_a[CHUNK] = ctx.copy.load_a(CHUNK, ctx.next_k);
    } else if constexpr (STEP % 2 == 1 && CHUNK < Context::Copy::CHUNKS_B) {
      ctx.staged_b[CHUNK] = ctx.copy.load_b(CHUNK, ctx.next_k);
    }
  }
  if constexpr (STEP + 1 < Context::STEPS) {
    using Next = FoldPosition<STEP + 1, Context::TILES_M, Context::TILES_N>;
    if constexpr (Next::M_TILE != Here::M_TILE || Next::SLICE != Here::SLICE) {
      // 切换输出行或 K slice 时才换 A, 同一行的三个输出 fragment 复用一个 A.
      ctx.template read_a<Next::M_TILE, Next::SLICE>();
    }
    // 只在 slice 的第一行依次准备 b[0],b[1],b[2], 后续三行复用它们.
    if constexpr (Next::M_TILE == 0) {
      ctx.template read_b<Next::N_TILE, Next::SLICE>();
    }
  }
  // 按参考实现提高当前 wave 的 WMMA 调度优先级, 执行后恢复为 0.
  __builtin_amdgcn_s_setprio(1);
  ctx.acc[Here::M_TILE][Here::N_TILE] = wmma_mma(
      ctx.a[Here::SLICE * Context::TILES_M + Here::M_TILE],
      ctx.b[Here::SLICE * Context::TILES_N + Here::N_TILE], ctx.acc[Here::M_TILE][Here::N_TILE]);
  __builtin_amdgcn_s_setprio(0);
  if constexpr (STEP + 1 < Context::STEPS) {
    fold_step<STEP + 1, PREFETCH>(ctx);
  }
}

/*
 * V7/V8 的主循环: 每个 wave 在当前 tile 的 48 条 WMMA 内, 穿插下一轮输入的寄存器预取.
 * 与 pipeline 的区别是预取放进 fold_step, 而非在 multiply_tile 前集中完成.
 * NUM_BUFFERS=2 为 V7, NUM_BUFFERS=1 为 V8, 两者使用相同的 WMMA 计算顺序.
 * 单缓冲只节省 LDS, staged_a/b 仍存在, 下一轮数据仍先放在寄存器中.
 */
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WAVE_M, int WAVE_N, int LDS_PAD,
          int GROUP_SIZE, int NUM_BUFFERS>
__device__ __forceinline__ void interleaved(const __half* A, const __half* B, __half* C, int M,
                                            int N, int K, __half* tile_a, __half* tile_b) {
  using Context = FoldContext<BLOCK_M, BLOCK_N, BLOCK_K, WAVE_M, WAVE_N, LDS_PAD>;
  using Copy = typename Context::Copy;
  constexpr int STRIDE = BLOCK_K + LDS_PAD;
  constexpr int TILE_A = BLOCK_M * STRIDE, TILE_B = BLOCK_N * STRIDE;
  int row0, col0;
  map_block<BLOCK_M, BLOCK_N, GROUP_SIZE>(M, N, row0, col0);
  const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
  // 与 pipeline 相同的 wave/lane 映射, 仅 block 的全局区域由 map_block 分组选择.
  const int wave_row = wave / (BLOCK_N / WAVE_N) * WAVE_M;
  const int wave_col = wave % (BLOCK_N / WAVE_N) * WAVE_N;
  // 固定 lane 的 LDS 行起点, 后续只增加 buffer 和编译期 fragment 的偏移.
  const __half* lane_a = tile_a + (wave_row + lane % 16) * STRIDE;
  const __half* lane_b = tile_b + (wave_col + lane % 16) * STRIDE;
  const Copy copy(A, B, M, N, K, row0, col0, threadIdx.x);
  half16 acc[Context::TILES_M][Context::TILES_N] = {};
  half16 a[Context::K_SLICES * Context::TILES_M], b[Context::K_SLICES * Context::TILES_N];
  half8 staged_a[Copy::CHUNKS_A], staged_b[Copy::CHUNKS_B];
  // 与 pipeline 相同, 用一个 LDS 基址和整数槽偏移选择缓冲区.
  int buffer = 0;
  const int tiles_k = (K - 1) / BLOCK_K + 1;
  // 首个 K tile 直接提交 LDS, 后续 tile 先暂存寄存器再与当前计算交错.
  copy.prefetch(0, staged_a, staged_b);
  copy.commit(tile_a, tile_b, staged_a, staged_b);
  __syncthreads();
  for (int tile = 0; tile < tiles_k - 1; ++tile) {
    // Context 将当前槽的 lane 起点交给 fold_step, 下一轮 Global 起点为 (tile+1)*BLOCK_K.
    const Context ctx{copy,
                      (tile + 1) * BLOCK_K,
                      lane_a + buffer * TILE_A,
                      lane_b + buffer * TILE_B,
                      a,
                      b,
                      acc,
                      staged_a,
                      staged_b};
    ctx.template read_a<0, 0>();
    ctx.template read_b<0, 0>();
    // 计算当前 tile, 同时在寄存器中准备下一 tile 的全部搬运数据.
    fold_step<0, true>(ctx);
    // 单缓冲将覆盖刚读过的 LDS, 须等所有 wave 读完; 双缓冲写另一份 LDS.
    if constexpr (NUM_BUFFERS == 1) __syncthreads();
    const int next_buffer = NUM_BUFFERS == 1 ? 0 : buffer ^ 1;
    copy.commit(tile_a + next_buffer * TILE_A, tile_b + next_buffer * TILE_B, staged_a, staged_b);
    __syncthreads();
    buffer = next_buffer;
  }
  // 尾轮禁用 Global 预取, 沿用相同的 fragment 加载和 WMMA 顺序计算最后的 K tile.
  const Context ctx{
      copy, 0, lane_a + buffer * TILE_A, lane_b + buffer * TILE_B, a, b, acc, staged_a, staged_b};
  ctx.template read_a<0, 0>();
  ctx.template read_b<0, 0>();
  fold_step<0, false>(ctx);
  store_wave<WAVE_M, WAVE_N, true>(C, acc, row0 + wave_row, col0 + wave_col, M, N, lane);
}
}  // namespace hgemm

/*
 * V1: 单 wave 基线, 默认 <32,32,16>, block(32), grid(N / BLOCK_N, M / BLOCK_M).
 * 一个 block 算 C 的 32x32 区域, 每轮将 A[32,16], B[16,32] 搬入 As[32,16], Bs[32,16].
 * 搬运坐标按 tile 展平分配, fragment 则按 WMMA 的 lane 布局重新读取.
 * 32x32 输出拆成 2x2 个 16x16 fragment, 每个 lane 保存 4*8=32 个有效输出.
 * 一轮 K tile 执行四条 WMMA, 全部 K tile 累加结束后才写 C.
 */
template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void __launch_bounds__(32)
    hgemm_v1_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                    __half* __restrict__ C, int M, int N, int K) {
  const int lane = threadIdx.x;
  // row0/col0 是当前 block 在 C[M,N] 中的左上角, 也是 A 行区域和 B 列区域的起点.
  const int row0 = blockIdx.y * BLOCK_M, col0 = blockIdx.x * BLOCK_N;
  __shared__ __half As[BLOCK_M * BLOCK_K], Bs[BLOCK_N * BLOCK_K];
  half16 acc[BLOCK_M / 16][BLOCK_N / 16] = {};
  for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
    // 将 As[BLOCK_M,BLOCK_K] 展平分给 32 个 lane, index 每轮增加一个 wave 的线程数.
    // 默认 lane=3 的 index 为 3,35,..., 对应 tile 的 (row,k)=(0,3),(2,3),... .
    for (int index = lane; index < BLOCK_M * BLOCK_K; index += 32) {
      const int row = index / BLOCK_K, k = index % BLOCK_K;
      // As[row, k] <- A[row0 + row, k0 + k].
      As[row * BLOCK_K + k] = A[(row0 + row) * K + k0 + k];
    }
    // 列主序 B 沿 K 连续, 按 Bs[col,k] 展平分给 lane, 搬运方式与 A 对称.
    for (int index = lane; index < BLOCK_K * BLOCK_N; index += 32) {
      const int col = index / BLOCK_K, k = index % BLOCK_K;
      // Bs[col,k] <- B[k0+k,col0+col], 全局和 LDS 都沿 K 连续.
      Bs[col * BLOCK_K + k] = B[(col0 + col) * K + k0 + k];
    }
    __syncthreads();
    // LDS 已包含整块输入, lane 按 WMMA 布局重新读取, acc[2][2] 更新四个输出 fragment.
    hgemm::multiply_tile<BLOCK_M, BLOCK_N, BLOCK_K, 0>(As, Bs, 0, 0, lane, acc);
    __syncthreads();
  }
  hgemm::store_wave<BLOCK_M, BLOCK_N, false>(C, acc, row0, col0, M, N, lane);
}

/*
 * V2: 多 wave 分块与软件双缓冲, 默认 <256,256,16,64,64>, block(512).
 * block 输出 256x256, 按 4x4 个 wave 划分, 每个 wave 输出 64x64.
 * 每个 wave 的 acc 是 4x4 个 fragment, 一轮 K=16 执行 16 条 WMMA.
 * TileCopy 用 128-bit 读取下一轮 A/B 到寄存器, pipeline 计算当前 LDS 后提交另一份.
 * 相比 V1, 每次搬运 8 个 FP16, 输入在更大的输出分块内复用, 每轮一次 block 同步.
 */
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WAVE_M, int WAVE_N>
__global__ void __launch_bounds__(BLOCK_M / WAVE_M * (BLOCK_N / WAVE_N) * 32)
    hgemm_v2_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                    __half* __restrict__ C, int M, int N, int K) {
  // 连续分配两个 K tile 的缓冲槽, 每槽保存 As[256,16] 与 Bs[256,16], 两者均沿 K 连续.
  __shared__ __half As[2 * BLOCK_M * BLOCK_K];
  __shared__ __half Bs[2 * BLOCK_N * BLOCK_K];
  hgemm::pipeline<BLOCK_M, BLOCK_N, BLOCK_K, WAVE_M, WAVE_N, 0, false>(
      A, B, C, M, N, K, blockIdx.y * BLOCK_M, blockIdx.x * BLOCK_N, As, Bs);
}

/*
 * V3: V2 的搬运和计算不变, 只通过 LdsLayout 改变 LDS 物理地址.
 * 默认 LDS_PAD=8, LDS 行宽从 16 变成 24 个 FP16, 相邻行的 bank 起点随之改变.
 * USE_SWIZZLE=true 时以完整的 16 元素 fragment 组做 XOR, 通常搭配 LDS_PAD=0.
 * gfx1151 的 BLOCK_K=16 只有一组, XOR 是恒等映射, 更深的 K 才有可交换的组.
 * 行宽改变后, TileCopy 的写入和 multiply_tile 的读取都应用同一个 Layout.
 */
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WAVE_M, int WAVE_N, int LDS_PAD,
          bool USE_SWIZZLE = false>
__global__ void __launch_bounds__(BLOCK_M / WAVE_M * (BLOCK_N / WAVE_N) * 32)
    hgemm_v3_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                    __half* __restrict__ C, int M, int N, int K) {
  __shared__ __half As[2 * BLOCK_M * (BLOCK_K + LDS_PAD)];
  __shared__ __half Bs[2 * BLOCK_N * (BLOCK_K + LDS_PAD)];
  hgemm::pipeline<BLOCK_M, BLOCK_N, BLOCK_K, WAVE_M, WAVE_N, LDS_PAD, false, USE_SWIZZLE>(
      A, B, C, M, N, K, blockIdx.y * BLOCK_M, blockIdx.x * BLOCK_N, As, Bs);
}

/*
 * V4: 默认 <256,256,16,64,64,8>, 将 pipeline 的 BUFFER_LOAD 从 false 改为 true.
 * TileCopy::load_a/load_b 的完整向量走 raw buffer load, 尾部和未对齐地址走标量加载.
 * M/N/K 边缘写入 LDS 的无效元素均为零, 所有 lane 仍执行相同的 WMMA.
 * store_wave 最后只写 row<M, col<N 的输出, 因而 launch 可以使用向上取整的 grid.
 */
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WAVE_M, int WAVE_N, int LDS_PAD>
__global__ void __launch_bounds__(BLOCK_M / WAVE_M * (BLOCK_N / WAVE_N) * 32)
    hgemm_v4_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                    __half* __restrict__ C, int M, int N, int K) {
  __shared__ __half As[2 * BLOCK_M * (BLOCK_K + LDS_PAD)];
  __shared__ __half Bs[2 * BLOCK_N * (BLOCK_K + LDS_PAD)];
  hgemm::pipeline<BLOCK_M, BLOCK_N, BLOCK_K, WAVE_M, WAVE_N, LDS_PAD, true>(
      A, B, C, M, N, K, blockIdx.y * BLOCK_M, blockIdx.x * BLOCK_N, As, Bs);
}

/*
 * V5: 默认 <128,96,64,64,48,8>, block(128), 复用 V4 的流水线和边界处理.
 * 输出 tile 从 256x256 缩小到 128x96, 释放 LDS 空间将 K tile 从 16 加深到 64.
 * 2x2 个 wave 各算 64x48, acc 为 4x3 个 fragment, 每个 wave 每 K tile 执行 48 条 WMMA.
 * 一个 LDS 槽为 (128+96)*(64+8)*sizeof(__half)=32256 bytes, 两槽共 63 KiB.
 * 例如 N=8192 时最后一个列 tile 只有 32 列有效, 仍按 96 列计算, 输出边缘由 V4 处理.
 */
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WAVE_M, int WAVE_N, int LDS_PAD>
__global__ void __launch_bounds__(BLOCK_M / WAVE_M * (BLOCK_N / WAVE_N) * 32)
    hgemm_v5_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                    __half* __restrict__ C, int M, int N, int K) {
  __shared__ __half As[2 * BLOCK_M * (BLOCK_K + LDS_PAD)];
  __shared__ __half Bs[2 * BLOCK_N * (BLOCK_K + LDS_PAD)];
  hgemm::pipeline<BLOCK_M, BLOCK_N, BLOCK_K, WAVE_M, WAVE_N, LDS_PAD, true>(
      A, B, C, M, N, K, blockIdx.y * BLOCK_M, blockIdx.x * BLOCK_N, As, Bs);
}

/*
 * V6: 分块与计算沿用 V5, map_block 按 GROUP_SIZE 个 tile 行分组, 改善跨 block 的复用.
 * 默认 GROUP_SIZE=8, 邻近 block 先共享 B 的列分块, 再切换到下一列分块.
 * 全局起点 row0/col0 改变, wave 与 lane 的局部坐标不变.
 */
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WAVE_M, int WAVE_N, int LDS_PAD,
          int GROUP_SIZE>
__global__ void __launch_bounds__(BLOCK_M / WAVE_M * (BLOCK_N / WAVE_N) * 32)
    hgemm_v6_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                    __half* __restrict__ C, int M, int N, int K) {
  __shared__ __half As[2 * BLOCK_M * (BLOCK_K + LDS_PAD)];
  __shared__ __half Bs[2 * BLOCK_N * (BLOCK_K + LDS_PAD)];
  int row0, col0;
  hgemm::map_block<BLOCK_M, BLOCK_N, GROUP_SIZE>(M, N, row0, col0);
  hgemm::pipeline<BLOCK_M, BLOCK_N, BLOCK_K, WAVE_M, WAVE_N, LDS_PAD, true>(A, B, C, M, N, K, row0,
                                                                            col0, As, Bs);
}

/*
 * V7: V6 的分块和分组映射不变, 将 pipeline 换成 interleaved, 继续使用两份 LDS.
 * 把 multiply_tile 的 k/m/n 循环展成 fold_step 的 48 个编译期 STEP.
 * 在这些 STEP 中安排下一 tile 的 8 个 A chunk, 6 个 B chunk 预取, 并提前读 fragment.
 * FoldPosition 给出每条 WMMA 的 slice/m/n, FoldContext 保存 lane 的 LDS 基址与寄存器数组.
 * 固定 lane 的 LDS 行起点后, read_a/read_b 只增加编译期 fragment 偏移.
 */
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WAVE_M, int WAVE_N, int LDS_PAD,
          int GROUP_SIZE>
__global__ void __launch_bounds__(BLOCK_M / WAVE_M * (BLOCK_N / WAVE_N) * 32)
    hgemm_v7_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                    __half* __restrict__ C, int M, int N, int K) {
  __shared__ __half As[2 * BLOCK_M * (BLOCK_K + LDS_PAD)];
  __shared__ __half Bs[2 * BLOCK_N * (BLOCK_K + LDS_PAD)];
  hgemm::interleaved<BLOCK_M, BLOCK_N, BLOCK_K, WAVE_M, WAVE_N, LDS_PAD, GROUP_SIZE, 2>(
      A, B, C, M, N, K, As, Bs);
}

/*
 * V8: 计算和寄存器预取沿用 V7, 只改变 LDS 缓冲槽的数量.
 * 默认 NUM_BUFFERS=1, LDS 从 63 KiB 减至 31.5 KiB, 可能容纳更多驻留 block.
 * 下一 tile 的 staged_a/b 要覆盖当前 LDS, 因此 interleaved 在提交前增加一次同步.
 * NUM_BUFFERS=2 时恢复双缓冲行为, 用于和单缓冲比较.
 */
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WAVE_M, int WAVE_N, int LDS_PAD,
          int GROUP_SIZE, int NUM_BUFFERS>
__global__ void __launch_bounds__(BLOCK_M / WAVE_M * (BLOCK_N / WAVE_N) * 32)
    hgemm_v8_kernel(const __half* __restrict__ A, const __half* __restrict__ B,
                    __half* __restrict__ C, int M, int N, int K) {
  __shared__ __half As[NUM_BUFFERS * BLOCK_M * (BLOCK_K + LDS_PAD)];
  __shared__ __half Bs[NUM_BUFFERS * BLOCK_N * (BLOCK_K + LDS_PAD)];
  // 单缓冲只分配槽 0, interleaved 始终使用 buffer=0.
  hgemm::interleaved<BLOCK_M, BLOCK_N, BLOCK_K, WAVE_M, WAVE_N, LDS_PAD, GROUP_SIZE, NUM_BUFFERS>(
      A, B, C, M, N, K, As, Bs);
}
