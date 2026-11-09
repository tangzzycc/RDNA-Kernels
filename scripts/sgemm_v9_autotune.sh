#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
gpu_arch="${GPU_ARCH:-gfx1151}"
device="${HIP_VISIBLE_DEVICES:-0}"
M=5120 N=5120 K=5120
warmup=3 iterations=3 batches=3
top=5 run_timeout=30
output=""

# 沿用原版 tile 搜索范围, 当前 V9 固定使用 16x16 逻辑线程网格.
# gfx1151 使用 wave32, 256 个线程对应 8 个 wave, 此处的 16 表示逻辑网格宽度.
BK_VALUES=(8 16 32 64)
TM_VALUES=(4 8 16 32)
TN_VALUES=(4 8 16 32)
BM_VALUES=(64 128 256)
BN_VALUES=(64 128 256)
NUM_THREADS_VALUES=(256)

usage() {
    cat <<'EOF'
Usage: scripts/sgemm_v9_autotune.sh [options]

Search SGEMM V9 template parameters on gfx1151. Compile each valid configuration,
check every output against V4, and rank HIP event timings. Repeat the fastest
configurations in rotating order with V6 as a baseline.

Options:
  --shape M N K     Matrix dimensions (default: 5120 5120 5120)
  --device ID       HIP GPU index (default: HIP_VISIBLE_DEVICES, or 0)
  --output DIR      Result directory (default: timestamped tests/results directory)
  --top N           Number of finalists to remeasure (default: 5)
  --warmup N        Search warmup launches (default: 3)
  --iterations N    Launches per search batch (default: 3)
  --batches N       Search timing batches (default: 3)
  --timeout SEC     Per-configuration runtime limit (default: 30)
  -h, --help        Show this help
EOF
}

while (($#)); do
    case "$1" in
        --shape) M="$2"; N="$3"; K="$4"; shift 4 ;;
        --device) device="$2"; shift 2 ;;
        --output) output="$2"; shift 2 ;;
        --top) top="$2"; shift 2 ;;
        --warmup) warmup="$2"; shift 2 ;;
        --iterations) iterations="$2"; shift 2 ;;
        --batches) batches="$2"; shift 2 ;;
        --timeout) run_timeout="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 1 ;;
    esac
done

for value in "$M" "$N" "$K" "$iterations" "$batches" "$top" "$run_timeout"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { printf 'Expected positive integers.\n' >&2; exit 1; }
done
[[ "$warmup" =~ ^[0-9]+$ ]] || { printf 'Invalid warmup count.\n' >&2; exit 1; }
((M % 64 == 0 && N % 64 == 0 && K % 8 == 0)) || {
    printf 'The reference kernel requires M, N divisible by 64 and K divisible by 8.\n' >&2
    exit 1
}

cd -- "$repo_root"
# shellcheck source=/dev/null
source "$repo_root/.venv/bin/activate"
export ROCM_PATH
ROCM_PATH="$(rocm-sdk path --root)"
export HIP_VISIBLE_DEVICES="$device"

run_id="$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"
output="${output:-$repo_root/tests/results/matmul/sgemm_v9_autotune/$run_id}"
mkdir -p -- "$output"
output="$(cd -- "$output" && pwd)"
build_dir="$repo_root/build/hip/matmul/sgemm_v9_autotune/$run_id"
mkdir -p -- "$build_dir" "$output/configs" "$output/confirm"
# 一轮搜索使用同一份源码, 避免期间的编辑混入不同配置的结果.
for source in kernels/matmul/*.hip common/hip/common.h; do
    mkdir -p -- "$build_dir/source/$(dirname -- "$source")"
    cp -- "$repo_root/$source" "$build_dir/source/$source"
done
driver="$build_dir/source/kernels/matmul/profiler.hip"
header=$'version\tBM\tBN\tBK\tTM\tTN\tthreads\tmean_ms\tTFLOP_s\tVGPR\tLDS_bytes\tscratch_bytes'
printf '%s\n' "$header" > "$output/results.tsv"
printf '%s\n' "$header" > "$output/confirm_runs.tsv"

compile_config() {
    # 用 -D 实例化模板参数, 各配置独立编译, 保持源码快照不变.
    local binary="$1" bm="$2" bn="$3" bk="$4" tm="$5" tn="$6" nt="$7"
    hipcc -O3 -std=c++17 "--offload-arch=$gpu_arch" \
        "-DSGEMM_BM=$bm" "-DSGEMM_BN=$bn" "-DSGEMM_BK=$bk" \
        "-DSGEMM_TM=$tm" "-DSGEMM_TN=$tn" "-DSGEMM_NUM_THREADS=$nt" \
        "$driver" -o "$binary"
}

probe="$build_dir/probe"
# 从 HIP 查询实际线程数与 LDS 上限, gfx1151 本机每 block 可用 64 KiB LDS.
compile_config "$probe" 128 128 16 8 8 256 > "$output/probe_compile.log" 2>&1
IFS=$'\t' read -r max_lds max_threads actual_arch < <("$probe" --device-info)
printf 'GPU=%s, max LDS/block=%s bytes, max threads/block=%s\n' \
    "$actual_arch" "$max_lds" "$max_threads"
printf 'Shape=(%s,%s,%s), target=%s, HIP_VISIBLE_DEVICES=%s\n' "$M" "$N" "$K" "$gpu_arch" "$device"
{
    printf 'shape=%s,%s,%s\narch=%s\ndevice=%s\nmax_lds=%s\n' "$M" "$N" "$K" "$gpu_arch" "$device" "$max_lds"
    sha256sum "$build_dir/source/kernels/matmul/"*.hip "$build_dir/source/common/hip/common.h"
} > "$output/config.txt"

total=$((${#BK_VALUES[@]} * ${#TM_VALUES[@]} * ${#TN_VALUES[@]} * ${#BM_VALUES[@]} * ${#BN_VALUES[@]} * ${#NUM_THREADS_VALUES[@]}))
visited=0 tested=0 skipped=0 failed=0
for bk in "${BK_VALUES[@]}"; do
    for tm in "${TM_VALUES[@]}"; do
        for tn in "${TN_VALUES[@]}"; do
            for bm in "${BM_VALUES[@]}"; do
                for bn in "${BN_VALUES[@]}"; do
                    for nt in "${NUM_THREADS_VALUES[@]}"; do
                        visited=$((visited + 1))
                        # nt * 4 是一轮协作加载的元素数, 须覆盖整数行且能整轮填满 A, B tile.
                        # BM, BN 须覆盖整数个 [16 * TM, 16 * TN] 基础 tile.
                        # 本版不处理边界, M, N, K 须按对应 block tile 整除.
                        # LDS 用量为 (BM * BK + BK * BN) * sizeof(float), 按设备上限过滤.
                        if ((nt * 4 % bk || nt * 4 % bn || bm % (16 * tm) || bn % (16 * tn) ||
                             bm * bk % (4 * nt) || bn * bk % (4 * nt) ||
                             M % bm || N % bn || K % bk || nt > max_threads ||
                             (bm + bn) * bk * 4 > max_lds)); then
                            skipped=$((skipped + 1))
                            continue
                        fi
                        id="${bm}_${bn}_${bk}_${tm}_${tn}_${nt}"
                        binary="$build_dir/$id"
                        printf '[%d/%d] <%s,%s,%s,%s,%s,%s> ' "$visited" "$total" "$bm" "$bn" "$bk" "$tm" "$tn" "$nt"
                        if ! compile_config "$binary" "$bm" "$bn" "$bk" "$tm" "$tn" "$nt" \
                            > "$output/configs/$id.compile.log" 2>&1; then
                            printf 'compile failed\n'
                            failed=$((failed + 1))
                            continue
                        fi
                        if ! timeout "$run_timeout" "$binary" sgemm_v9 "$M" "$N" "$K" "$warmup" "$iterations" "$batches" \
                            > "$output/configs/$id.run.log" 2>&1; then
                            printf 'run/check failed; see %s.run.log\n' "$id"
                            failed=$((failed + 1))
                            continue
                        fi
                        awk '/^RESULT\t/ { sub(/^RESULT\t/, ""); print }' "$output/configs/$id.run.log" >> "$output/results.tsv"
                        awk -F '\t' '/^RESULT\t/ { printf "%.3f ms, %.3f TFLOP/s, VGPR=%s, scratch=%s bytes\n", $9, $10, $11, $13 }' "$output/configs/$id.run.log"
                        tested=$((tested + 1))
                    done
                done
            done
        done
    done
done

((tested > 0)) || { printf 'No configuration passed; see %s\n' "$output" >&2; exit 1; }
{
    printf '%s\n' "$header"
    tail -n +2 "$output/results.tsv" | sort -t $'\t' -k8,8g
} > "$output/ranking.tsv"
printf 'Search: %d passed, %d skipped, %d failed.\n' "$tested" "$skipped" "$failed"

# 最快的 top 组与固定 V6 对照轮换复测, 减少运行顺序与 GPU 状态变化的影响.
# 每组复测 3 轮, 每轮预热 8 次, 计时 4 批 * 4 次, 合计 48 次计时启动.
mapfile -t finalists < <(awk -v top="$top" 'NR > 1 && NR <= top + 1' "$output/ranking.tsv")
if ((M % 128 == 0 && N % 128 == 0)); then
    finalists+=($'V6\t128\t128\t8\t8\t8\t256')
fi
for round in 0 1 2; do
    for ((position=0; position<${#finalists[@]}; position++)); do
        index=$(((round + position) % ${#finalists[@]}))
        IFS=$'\t' read -r version bm bn bk tm tn nt _ <<< "${finalists[index]}"
        id="${bm}_${bn}_${bk}_${tm}_${tn}_${nt}"
        binary="$build_dir/$id"
        [[ "$version" != V6 ]] || binary="$probe"
        log="$output/confirm/${version}_${id}_${round}.log"
        printf 'Confirm %d: %s <%s,%s,%s,%s,%s,%s> ' "$((round + 1))" "$version" "$bm" "$bn" "$bk" "$tm" "$tn" "$nt"
        if timeout "$run_timeout" "$binary" "sgemm_${version,,}" "$M" "$N" "$K" 8 4 4 > "$log" 2>&1; then
            awk '/^RESULT\t/ { sub(/^RESULT\t/, ""); print }' "$log" >> "$output/confirm_runs.tsv"
            awk -F '\t' '/^RESULT\t/ { printf "%.3f ms\n", $9 }' "$log"
        else
            printf 'failed; see %s\n' "$log"
        fi
    done
done

# 仅汇总完整通过 3 轮复测的配置, 按平均耗时排名, 保留 VGPR, LDS 与 scratch 用量.
{
    printf '%s\n' "$header"
    awk -F '\t' -v OFS='\t' -v flops="$((2 * M * N * K))" '
        NR > 1 { key = $1 OFS $2 OFS $3 OFS $4 OFS $5 OFS $6 OFS $7;
                 sum[key] += $8; count[key]++; resources[key] = $10 OFS $11 OFS $12 }
        END { for (key in sum) if (count[key] == 3) {
            mean = sum[key] / count[key];
            print key, sprintf("%.6f", mean), sprintf("%.6f", flops / (mean * 1e9)), resources[key]
        } }
    ' "$output/confirm_runs.tsv" | sort -t $'\t' -k8,8g
} > "$output/confirmed.tsv"
awk -F '\t' 'NR == 1 || $1 == "V9"' "$output/confirmed.tsv" > "$output/confirmed_v9.tsv"
[[ $(wc -l < "$output/confirmed_v9.tsv") -gt 1 ]] || { printf 'No finalist passed all confirmation rounds.\n' >&2; exit 1; }
sed -n '2p' "$output/confirmed_v9.tsv" > "$output/best.tsv"

{
    printf 'SGEMM V9 autotune: gfx1151, FP32, shape=(%s,%s,%s)\n\n' "$M" "$N" "$K"
    printf 'Search: %d passed, %d skipped, %d failed. Finalists: 3 rotating rounds, 48 measured launches each.\n\n' "$tested" "$skipped" "$failed"
    printf '| Version | BM | BN | BK | TM | TN | Threads | Mean (ms) | TFLOP/s | VGPR | LDS (KiB) | Scratch (bytes) |\n'
    printf '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    awk -F '\t' 'NR > 1 { printf "| %s | %s | %s | %s | %s | %s | %s | %.3f | %.3f | %s | %.0f | %s |\n", $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11 / 1024, $12 }' "$output/confirmed.tsv"
} > "$output/summary.md"
cat "$output/summary.md"
printf '\nBest V9 (version BM BN BK TM TN threads ms TFLOP/s resources):\n'
cat "$output/best.tsv"
printf '\nResults: %s\n' "$output"
