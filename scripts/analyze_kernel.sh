#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
kernel_key=""
gpu_arch="${GPU_ARCH:-gfx1151}"
list_only=0
run_only=0
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
  --arch ARCH     GPU target (default: gfx1151, or GPU_ARCH)
  --refresh-roof  Recompute cached hardware Roofline peaks
  -h, --help      Show this help

Examples:
  scripts/analyze_kernel.sh elementwise/add
  scripts/analyze_kernel.sh elementwise/add -- 1048576
  scripts/analyze_kernel.sh --run-only elementwise/add
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

has_main() {
    grep -Eq '(^|[^[:alnum:]_])int[[:space:]]+main[[:space:]]*\(' "$1"
}

list_workloads() {
    awk '
        /ProfileWorkload[[:space:]]+workloads[[:space:]]*\[/ { table = 1; next }
        table && /};/ { table = 0 }
        table && match($0, /"[a-z][a-z0-9_]*"/) {
            print substr($0, RSTART + 1, RLENGTH - 2)
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
    printf '\n$'
    printf ' %q' "$@"
    printf '\n'
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
