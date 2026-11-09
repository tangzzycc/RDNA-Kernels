#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
kernel_key=""
gpu_arch="${GPU_ARCH:-gfx1151}"
list_only=0
run_only=0
bandwidth_only=0
samples=5
refresh_roof=0
kernel_args=()

usage() {
    cat <<'EOF'
Usage: scripts/analyze_kernel.sh [options] family/operator [-- operator_args...]

Compile kernels/<family>/profiler.hip, run the selected operator, collect all
counters, and print the complete rocprof-compute report. The family profiler
runs a smoke check and benchmark; --profile runs one deterministic workload.
Full correctness tests are in tests/test_<family>.py and run separately via pytest.
Arguments after -- are forwarded to both modes.

Options:
  --list          List operators included in family profilers
  --run-only      Compile and execute without profiling
  --bandwidth     Collect GCEA counters and print total DRAM interface throughput
  --samples N     Bandwidth captures (default: 5, report the median sample)
  --arch ARCH     GPU target (default: gfx1151, or GPU_ARCH)
  --refresh-roof  Recompute cached hardware Roofline peaks
  -h, --help      Show this help

Examples:
  scripts/analyze_kernel.sh elementwise/add
  scripts/analyze_kernel.sh elementwise/add -- 1048576
  scripts/analyze_kernel.sh --run-only elementwise/add
  scripts/analyze_kernel.sh --bandwidth matmul/sgemm_v10
EOF
}

die() {
    printf '%s\n' "$*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        --list) list_only=1; shift ;;
        --run-only) run_only=1; shift ;;
        --bandwidth) bandwidth_only=1; shift ;;
        --samples)
            (($# >= 2)) || die "--samples requires a positive integer."
            samples="$2"; shift 2 ;;
        --refresh-roof) refresh_roof=1; shift ;;
        --arch)
            (($# >= 2)) || die "--arch requires a GPU target."
            gpu_arch="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; kernel_args=("$@"); break ;;
        -*) die "Unknown option: $1" ;;
        *)
            [[ -z "$kernel_key" ]] || die "Specify one operator; put its arguments after --."
            kernel_key="$1"; shift ;;
    esac
done

[[ "$samples" =~ ^[1-9][0-9]*$ ]] || die "--samples requires a positive integer."
((run_only && bandwidth_only)) && die "Choose either --run-only or --bandwidth."

has_main() {
    grep -Eq '(^|[^[:alnum:]_])int[[:space:]]+main[[:space:]]*\(' "$1"
}

list_workloads() {
    awk '
        /ProfileWorkload[[:space:]]+workloads[[:space:]]*\[/ { table = 1 }
        table {
            line = $0
            while (match(line, /"[a-z][a-z0-9_]*"/)) {
                print substr(line, RSTART + 1, RLENGTH - 2)
                line = substr(line, RSTART + RLENGTH)
            }
            if ($0 ~ /};/) table = 0
        }
    ' "$1"
}

has_workload() {
    local driver="$1" operator="$2"
    [[ -f "$driver" ]] && has_main "$driver" &&
        list_workloads "$driver" | grep -Fx -- "$operator" >/dev/null
}

if ((list_only)); then
    for driver in "$repo_root"/kernels/*/profiler.hip; do
        if [[ -f "$driver" ]] && has_main "$driver"; then
            family="$(basename -- "$(dirname -- "$driver")")"
            while IFS= read -r operator; do
                printf '%s/%s\n' "$family" "$operator"
            done < <(list_workloads "$driver")
        fi
    done
    exit 0
fi

[[ "$kernel_key" =~ ^[a-z][a-z0-9_]*/[a-z][a-z0-9_]*$ ]] ||
    die "Specify family/operator, e.g. elementwise/add. Use --list to see runnable operators."
[[ "$gpu_arch" =~ ^gfx[0-9a-f]+$ ]] || die "Expected a GPU target such as gfx1151."
family="${kernel_key%/*}"
operator="${kernel_key#*/}"
driver="$repo_root/kernels/$family/profiler.hip"
has_workload "$driver" "$operator" ||
    die "$kernel_key needs a workload in kernels/$family/profiler.hip before it can be analyzed."
[[ -f "$repo_root/.venv/bin/activate" ]] || die "Project .venv is missing; follow the README installation steps."

cd -- "$repo_root"
# shellcheck source=/dev/null
source "$repo_root/.venv/bin/activate"

run_logged() {
    local log="$1" display="$2"
    shift 2
    if [[ "$display" != quiet ]]; then
        printf '\n$'
        printf ' %q' "$@"
        printf '\n'
    fi
    if [[ "$display" == all ]]; then
        "$@" 2>&1 | tee "$log"
    elif "$@" 2>&1 | tee "$log" | awk '/\[Run |\[roofline\]|ERROR/ { print; fflush() }'; then
        return 0
    else
        local status=$?
        tail -n 25 "$log" >&2
        return "$status"
    fi
}

valid_roofline() {
    [[ -s "$1" ]] || return 1
    awk -F, '
        NR == 1 { for (i = 1; i <= NF; i++) {
            if ($i == "FP32Flops") fp32 = i
            if ($i == "L2Bw") l2 = i
        } }
        NR == 2 { valid = fp32 && l2 && $fp32 + 0 > 0 && $l2 + 0 > 0; exit }
        END { exit !valid }
    ' "$1"
}

executable="$repo_root/build/hip/$family/profiler"
results="$repo_root/tests/results/$kernel_key"
mkdir -p -- "$(dirname -- "$executable")" "$results"
run_logged "$results/compile.log" all \
    hipcc -O3 -std=c++17 "--offload-arch=$gpu_arch" "$driver" -o "$executable"
run_logged "$results/benchmark.log" all "$executable" "$operator" "${kernel_args[@]}"
if ((run_only)); then
    printf '\nResults: tests/results/%s\n' "$kernel_key"
    exit 0
fi

if ((bandwidth_only)); then
    # 每次创建独立目录, 保留之前的采样记录.
    mkdir -p -- "$results/gcea"
    gcea_results="$(mktemp -d "$results/gcea/$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)-XXXXXX")"
    for ((sample = 1; sample <= samples; sample++)); do
        printf '\nGCEA capture %d/%d\n' "$sample" "$samples"
        run_logged "$gcea_results/$sample.log" quiet \
            rocprofv3 --pmc GCEA_RDRAM_SIZE_REQ_sum GCEA_WDRAM_SIZE_REQ_sum \
            --kernel-include-regex "(^| )${operator}_kernel[<(]" --output-format csv \
            -d "$gcea_results/$sample" \
            -- "$executable" "$operator" --profile "${kernel_args[@]}"
    done

    # 按官方 32-byte 单位换算; 时间戳为 ns, bytes/ns 即十进制 GB/s.
    # 选取合计吞吐中位数对应的记录, 读写计数使用同一 dispatch 的时间.
    python - "$gcea_results" "$kernel_key" "${kernel_args[@]}" <<'PY'
import csv
import json
from pathlib import Path
import sys

directory = Path(sys.argv[1])
counter_names = ("GCEA_RDRAM_SIZE_REQ_sum", "GCEA_WDRAM_SIZE_REQ_sum")
records = []
for capture in sorted(p for p in directory.iterdir() if p.is_dir()):
    counters = {}
    for path in capture.rglob("*counter_collection.csv"):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                name = row["Counter_Name"]
                if name not in counter_names:
                    continue
                if name in counters:
                    raise SystemExit(f"Multiple target dispatches in {capture}")
                counters[name] = row
    if set(counters) != set(counter_names):
        raise SystemExit(f"Missing GCEA read/write counters in {capture}")
    read, write = (counters[name] for name in counter_names)
    for key in ("Dispatch_Id", "Agent_Id", "Kernel_Name", "Start_Timestamp", "End_Timestamp"):
        if read[key] != write[key]:
            raise SystemExit(f"Read/write counters belong to different dispatches: {capture}")
    duration_ns = int(read["End_Timestamp"]) - int(read["Start_Timestamp"])
    if duration_ns <= 0:
        raise SystemExit(f"Invalid dispatch duration in {capture}")
    read_bytes = float(read["Counter_Value"]) * 32
    write_bytes = float(write["Counter_Value"]) * 32
    records.append({
        "sample": int(capture.name), "duration_ns": duration_ns,
        "read_bytes": read_bytes, "write_bytes": write_bytes,
        "total_GB_s": (read_bytes + write_bytes) / duration_ns,
    })
if not records:
    raise SystemExit("No GCEA counter records were generated")
median = sorted(records, key=lambda item: item["total_GB_s"])[len(records) // 2]
summary = {"operator": sys.argv[2], "args": sys.argv[3:],
           "representative": median, "samples": records}
(directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
line = (f"{sys.argv[2]}: GCEA DRAM interface={median['total_GB_s']:.2f} GB/s, "
        f"profile={median['duration_ns'] / 1e6:.3f} ms, samples={len(records)}")
(directory / "summary.txt").write_text(line + "\n")
print(line)
PY
    printf '\nResults: %s\n' "${gcea_results#"$repo_root/"}"
    exit 0
fi

profile="$results/profile"
run_logged "$results/profile.log" progress \
    rocprof-compute profile --output-directory "$profile" --overwrite --no-roof \
    -- "$executable" "$operator" --profile "${kernel_args[@]}"

cache="$repo_root/.cache/rocprof-compute/$gpu_arch"
roofline="$cache/roofline.csv"
mkdir -p -- "$(dirname -- "$cache")"
if ((refresh_roof)) || ! valid_roofline "$roofline"; then
    run_logged "$cache.log" progress \
        rocprof-compute profile --bench-only --output-directory "$cache" --overwrite
    valid_roofline "$roofline" || die "Roofline benchmark did not produce valid peaks; see $cache.log"
else
    printf '\nReusing hardware Roofline peaks: %s\n' "$roofline"
fi
cp -- "$roofline" "$profile/roofline.csv"
run_logged "$results/analysis.txt" all \
    rocprof-compute analyze --path "$profile" --time-unit us \
    --include-cols Description --output-format stdout
printf '\nResults: tests/results/%s\n' "$kernel_key"
