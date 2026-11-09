#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
gpu_arch="${GPU_ARCH:-gfx1151}"
device="${HIP_VISIBLE_DEVICES:-0}"
M=5120 N=5120 K=5120
warmup=3 iterations=3 batches=3
top=5 run_timeout=30 compile_jobs=4
output="" config_file=""

# 先搜索常用 block, wave 与线程分块, 线程数由 block 内的 wave 数推导.
BM_VALUES=(64 128 256)
BN_VALUES=(64 128 256)
BK_VALUES=(8 16 32 64)
WM_VALUES=(32 64)
WN_VALUES=(32 64 128)
WN_ITERS_VALUES=(1 2 4)
TM_VALUES=(4 8)
TN_VALUES=(4 8)
NUM_THREADS_VALUES=(128 256)

usage() {
    cat <<'EOF'
Usage: scripts/sgemm_v10_autotune.sh [options]

Search SGEMM V10 parameters on gfx1151. Check every output against V4 before
timing. Remeasure the fastest configurations alongside tuned V9, V6 and the
initial V10 configuration. Compile in CPU batches, then measure on the GPU.

Options:
  --shape M N K      Matrix dimensions (default: 5120 5120 5120)
  --device ID        HIP GPU index (default: HIP_VISIBLE_DEVICES, or 0)
  --output DIR       Result directory (default: timestamped tests/results directory)
  --configs FILE     Search a whitespace-separated list of nine template parameters:
                     BM BN BK WM WN WN_ITERS TM TN NUM_THREADS
  --jobs N           Parallel compilations per batch (default: 4)
  --top N            Number of finalists to remeasure (default: 5)
  --warmup N         Search warmup launches (default: 3)
  --iterations N     Launches per search batch (default: 3)
  --batches N        Search timing batches (default: 3)
  --timeout SEC      Per-configuration runtime limit (default: 30)
  -h, --help         Show this help
EOF
}

while (($#)); do
    case "$1" in
        --shape) M="$2"; N="$3"; K="$4"; shift 4 ;;
        --device) device="$2"; shift 2 ;;
        --output) output="$2"; shift 2 ;;
        --configs) config_file="$2"; shift 2 ;;
        --jobs) compile_jobs="$2"; shift 2 ;;
        --top) top="$2"; shift 2 ;;
        --warmup) warmup="$2"; shift 2 ;;
        --iterations) iterations="$2"; shift 2 ;;
        --batches) batches="$2"; shift 2 ;;
        --timeout) run_timeout="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 1 ;;
    esac
done

for value in "$M" "$N" "$K" "$iterations" "$batches" "$top" "$run_timeout" "$compile_jobs"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { printf 'Expected positive integers.\n' >&2; exit 1; }
done
[[ "$warmup" =~ ^[0-9]+$ ]] || { printf 'Invalid warmup count.\n' >&2; exit 1; }
((M % 64 == 0 && N % 64 == 0 && K % 8 == 0)) || {
    printf 'V4 reference requires M, N divisible by 64 and K divisible by 8.\n' >&2
    exit 1
}
if [[ -n "$config_file" ]]; then
    config_file="$(realpath -- "$config_file")"
fi

cd -- "$repo_root"
# shellcheck source=/dev/null
source "$repo_root/.venv/bin/activate"
export ROCM_PATH
ROCM_PATH="$(rocm-sdk path --root)"
export HIP_VISIBLE_DEVICES="$device"
run_id="$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)"
output="${output:-$repo_root/tests/results/matmul/sgemm_v10_autotune/$run_id}"
mkdir -p -- "$output"
output="$(cd -- "$output" && pwd)"
build_dir="$repo_root/build/hip/matmul/sgemm_v10_autotune/$run_id"
mkdir -p -- "$build_dir" "$output/configs" "$output/confirm"

# 同一轮使用源码快照, 各配置独立编译, 不通过 sed 修改 kernel.
for source in kernels/matmul/*.hip common/hip/common.h; do
    mkdir -p -- "$build_dir/source/$(dirname -- "$source")"
    cp -- "$repo_root/$source" "$build_dir/source/$source"
done
driver="$build_dir/source/kernels/matmul/profiler.hip"
header=$'version\tBM\tBN\tBK\tWM\tWN\tWN_ITERS\tTM\tTN\tthreads\tmean_ms\tTFLOP_s\tVGPR\tLDS_bytes\tscratch_bytes'
printf '%s\n' "$header" > "$output/results.tsv"
printf '%s\n' "$header" > "$output/confirm_runs.tsv"

compile_config() {
    local binary="$1" bm="$2" bn="$3" bk="$4" wm="$5" wn="$6" ni="$7" tm="$8" tn="$9" nt="${10}"
    hipcc -O3 -std=c++17 "--offload-arch=$gpu_arch" \
        -DSGEMM_BM=128 -DSGEMM_BN=256 -DSGEMM_BK=8 \
        -DSGEMM_TM=4 -DSGEMM_TN=4 -DSGEMM_NUM_THREADS=256 \
        "-DSGEMM_V10_BM=$bm" "-DSGEMM_V10_BN=$bn" "-DSGEMM_V10_BK=$bk" \
        "-DSGEMM_V10_WM=$wm" "-DSGEMM_V10_WN=$wn" "-DSGEMM_V10_WN_ITERS=$ni" \
        "-DSGEMM_V10_TM=$tm" "-DSGEMM_V10_TN=$tn" "-DSGEMM_V10_NUM_THREADS=$nt" \
        "$driver" -o "$binary"
}

# 显式固定三个对照配置, 后续修改 profiler 默认值也不改变对照的含义.
probe="$build_dir/probe"
compile_config "$probe" 64 64 16 32 32 1 4 4 128 > "$output/probe_compile.log" 2>&1
IFS=$'\t' read -r max_lds max_threads actual_arch < <("$probe" --device-info)
printf 'GPU=%s, max LDS/block=%s bytes, max threads/block=%s\n' "$actual_arch" "$max_lds" "$max_threads"
printf 'Shape=(%s,%s,%s), target=%s, HIP_VISIBLE_DEVICES=%s\n' "$M" "$N" "$K" "$gpu_arch" "$device"
{
    printf 'shape=%s,%s,%s\narch=%s\ndevice=%s\nmax_lds=%s\n' "$M" "$N" "$K" "$gpu_arch" "$device" "$max_lds"
    sha256sum "$build_dir/source/kernels/matmul/"*.hip "$build_dir/source/common/hip/common.h"
} > "$output/config.txt"

valid_config() {
    local bm="$1" bn="$2" bk="$3" wm="$4" wn="$5" ni="$6" tm="$7" tn="$8" nt="$9"
    # block 由整数个 wave tile 覆盖, 每个 wave 为 32 个线程.
    ((bm % wm == 0 && bn % wn == 0 && nt == (bm / wm) * (bn / wn) * 32)) || return 1
    ((nt <= max_threads && (bm + bn) * bk * 4 <= max_lds)) || return 1
    ((M % bm == 0 && N % bn == 0 && K % bk == 0)) || return 1
    # float4 加载覆盖整数行, stride 循环完整填满 A, B tile, 输出沿连续 4 列写回.
    ((bk % 4 == 0 && bn % 4 == 0 && tn % 4 == 0 && nt * 4 % bk == 0 && nt * 4 % bn == 0)) || return 1
    ((bm * bk % (4 * nt) == 0 && bn * bk % (4 * nt) == 0)) || return 1
    # 每个 wave 子块恰由 32 个 [TM,TN] 小块覆盖, M_ITERS 与两个子块尺寸均须为整数.
    ((wm * wn % (32 * tm * tn * ni) == 0)) || return 1
    local mi=$((wm * wn / (32 * tm * tn * ni)))
    ((mi > 0 && wm % mi == 0 && wn % ni == 0)) || return 1
    local sm=$((wm / mi)) sn=$((wn / ni))
    ((sm % tm == 0 && sn % tn == 0 && (sm / tm) * (sn / tn) == 32))
}

configs=()
declare -A seen=()
total=0 skipped=0 tested=0 failed=0
add_config() {
    total=$((total + 1))
    if ! valid_config "$@"; then
        skipped=$((skipped + 1))
        return
    fi
    local config="$*"
    [[ -z "${seen[$config]:-}" ]] || return 0
    seen["$config"]=1
    configs+=("$config")
}

if [[ -n "$config_file" ]]; then
    while read -r bm bn bk wm wn ni tm tn nt extra; do
        [[ -n "$bm" && "$bm" != \#* ]] || continue
        [[ -z "$extra" ]] || { printf 'Expected nine parameters per config.\n' >&2; exit 1; }
        for value in "$bm" "$bn" "$bk" "$wm" "$wn" "$ni" "$tm" "$tn" "$nt"; do
            [[ "$value" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid config value: %s\n' "$value" >&2; exit 1; }
        done
        add_config "$bm" "$bn" "$bk" "$wm" "$wn" "$ni" "$tm" "$tn" "$nt"
    done < "$config_file"
else
    for bm in "${BM_VALUES[@]}"; do
        for bn in "${BN_VALUES[@]}"; do
            for bk in "${BK_VALUES[@]}"; do
                for wm in "${WM_VALUES[@]}"; do
                    for wn in "${WN_VALUES[@]}"; do
                        for ni in "${WN_ITERS_VALUES[@]}"; do
                            for tm in "${TM_VALUES[@]}"; do
                                for tn in "${TN_VALUES[@]}"; do
                                    for nt in "${NUM_THREADS_VALUES[@]}"; do
                                        add_config "$bm" "$bn" "$bk" "$wm" "$wn" "$ni" "$tm" "$tn" "$nt"
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
fi
printf '%s\n' "${configs[@]}" > "$output/candidates.txt"
printf 'Candidates: %d valid, %d skipped from %d combinations.\n' "${#configs[@]}" "$skipped" "$total"

record_result() {
    awk -F '\t' -v OFS='\t' '
        /^V10_RESULT\t/ { sub(/^V10_RESULT\t/, ""); print }
        /^RESULT\t/ && $2 != "V10" { print $2,$3,$4,$5,0,0,0,$6,$7,$8,$9,$10,$11,$12,$13 }
    ' "$1" >> "$2"
}

# 一批编译全部结束后才测 GPU, 避免 CPU 编译争用共享内存带宽影响排名.
for ((start=0; start<${#configs[@]}; start+=compile_jobs)); do
    pids=() compiled=()
    end=$((start + compile_jobs))
    ((end <= ${#configs[@]})) || end=${#configs[@]}
    for ((index=start; index<end; index++)); do
        read -r bm bn bk wm wn ni tm tn nt <<< "${configs[index]}"
        id="${configs[index]// /_}"
        compile_config "$build_dir/$id" "$bm" "$bn" "$bk" "$wm" "$wn" "$ni" "$tm" "$tn" "$nt" \
            > "$output/configs/$id.compile.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if wait "$pid"; then compiled+=(1); else compiled+=(0); fi
    done
    for ((index=start; index<end; index++)); do
        id="${configs[index]// /_}"
        printf '[%d/%d] <%s> ' "$((index + 1))" "${#configs[@]}" "${configs[index]// /,}"
        if [[ "${compiled[index-start]}" == 0 ]]; then
            printf 'compile failed\n'; failed=$((failed + 1)); continue
        fi
        log="$output/configs/$id.run.log"
        if ! timeout "$run_timeout" "$build_dir/$id" sgemm_v10 "$M" "$N" "$K" "$warmup" "$iterations" "$batches" > "$log" 2>&1; then
            printf 'run/check failed; see %s\n' "$log"; failed=$((failed + 1)); continue
        fi
        record_result "$log" "$output/results.tsv"
        awk -F '\t' '/^V10_RESULT\t/ { printf "%.3f ms, %.3f TFLOP/s, VGPR=%s, scratch=%s bytes\n", $12,$13,$14,$16 }' "$log"
        tested=$((tested + 1))
    done
done
((tested > 0)) || { printf 'No configuration passed; see %s\n' "$output" >&2; exit 1; }
{ printf '%s\n' "$header"; tail -n +2 "$output/results.tsv" | sort -t $'\t' -k11,11g; } > "$output/ranking.tsv"
printf 'Search: %d passed, %d skipped, %d failed.\n' "$tested" "$skipped" "$failed"

# 最快的 top 组与固定对照轮换复测, 每组共 3 轮 * 4 批 * 4 次 = 48 次计时启动.
mapfile -t finalists < <(awk -v top="$top" 'NR > 1 && NR <= top + 1' "$output/ranking.tsv")
if ((M % 128 == 0 && N % 256 == 0)); then
    finalists+=($'V9\t128\t256\t8\t0\t0\t0\t4\t4\t256')
fi
if ((M % 128 == 0 && N % 128 == 0)); then
    finalists+=($'V6\t128\t128\t8\t0\t0\t0\t8\t8\t256')
fi
if ((K % 16 == 0)); then
    finalists+=($'V10_initial\t64\t64\t16\t32\t32\t1\t4\t4\t128')
fi
for round in 0 1 2; do
    for ((position=0; position<${#finalists[@]}; position++)); do
        index=$(((round + position) % ${#finalists[@]}))
        IFS=$'\t' read -r version bm bn bk wm wn ni tm tn nt _ <<< "${finalists[index]}"
        id="${bm}_${bn}_${bk}_${wm}_${wn}_${ni}_${tm}_${tn}_${nt}"
        binary="$build_dir/$id"
        operator="sgemm_${version,,}"
        if [[ "$version" != V10 ]]; then binary="$probe"; fi
        if [[ "$version" == V10_initial ]]; then operator=sgemm_v10; fi
        log="$output/confirm/${version}_${id}_${round}.log"
        printf 'Confirm %d: %s <%s> ' "$((round + 1))" "$version" "$id"
        if timeout "$run_timeout" "$binary" "$operator" "$M" "$N" "$K" 8 4 4 > "$log" 2>&1; then
            if [[ "$version" == V10_initial ]]; then
                sed 's/^V10_RESULT\tV10\t/V10_RESULT\tV10_initial\t/' "$log" > "$log.initial"
                record_result "$log.initial" "$output/confirm_runs.tsv"
            else record_result "$log" "$output/confirm_runs.tsv"; fi
            awk -F '\t' '/^RESULT\t/ { printf "%.3f ms\n", $9 }' "$log"
        else printf 'failed; see %s\n' "$log"; fi
    done
done

# 仅汇总完整通过 3 轮复测的配置, 按平均耗时排名, 同时保留编译资源用量.
{
    printf '%s\n' "$header"
    awk -F '\t' -v OFS='\t' -v flops="$((2 * M * N * K))" '
        NR > 1 { key=$1; for(i=2;i<=10;i++)key=key OFS $i;
                 sum[key]+=$11;count[key]++;resources[key]=$13 OFS $14 OFS $15 }
        END { for(key in sum)if(count[key]==3){ mean=sum[key]/count[key];
              print key,sprintf("%.6f",mean),sprintf("%.6f",flops/(mean*1e9)),resources[key] } }
    ' "$output/confirm_runs.tsv" | sort -t $'\t' -k11,11g
} > "$output/confirmed.tsv"
awk -F '\t' 'NR == 1 || $1 == "V10"' "$output/confirmed.tsv" > "$output/confirmed_v10.tsv"
[[ $(wc -l < "$output/confirmed_v10.tsv") -gt 1 ]] || { printf 'No finalist passed all confirmation rounds.\n' >&2; exit 1; }
sed -n '2p' "$output/confirmed_v10.tsv" > "$output/best.tsv"
{
    printf 'SGEMM V10 autotune: %s, FP32, shape=(%s,%s,%s)\n\n' "$actual_arch" "$M" "$N" "$K"
    printf 'Search: %d passed, %d skipped, %d failed. Finalists: 3 rotating rounds, 48 measured launches each.\n\n' "$tested" "$skipped" "$failed"
    printf '| Version | BM | BN | BK | WM | WN | WN_ITERS | TM | TN | Threads | Mean ms | TFLOP/s | VGPR | LDS KiB | Scratch bytes |\n'
    printf '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    awk -F '\t' 'NR>1{printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %.3f | %.3f | %s | %.0f | %s |\n",$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14/1024,$15}' "$output/confirmed.tsv"
} > "$output/summary.md"
cat "$output/summary.md"
printf '\nBest V10: '; cat "$output/best.tsv"
printf '\nResults: %s\n' "$output"
