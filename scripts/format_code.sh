#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
formatter="$repo_root/.venv/bin/clang-format"
check_only=0

usage() {
    cat <<'EOF'
Usage: scripts/format_code.sh [--check]

Format all repository C/C++/HIP/CUDA sources and headers using .clang-format.
Include tracked and untracked files; skip Git-ignored files and symlinks.
Supported extensions: c, cc, cpp, cxx, hip, cu, h, hh, hpp, hxx, cuh, inl.

Options:
  --check     Report formatting errors without modifying files
  -h, --help  Show this help
EOF
}

while (($#)); do
    case "$1" in
        --check) check_only=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 1 ;;
    esac
done

[[ -x "$formatter" ]] || {
    printf '%s\n' 'clang-format is missing. Install requirements-format.txt into .venv.' >&2
    exit 1
}

cd -- "$repo_root"
git rev-parse --is-inside-work-tree >/dev/null
mapfile -d '' -t candidates < <(
    git ls-files -z --cached --others --exclude-standard -- \
        '*.c' '*.cc' '*.cpp' '*.cxx' '*.hip' '*.cu' \
        '*.h' '*.hh' '*.hpp' '*.hxx' '*.cuh' '*.inl'
)
sources=()
for file in "${candidates[@]}"; do
    if [[ -f "$file" && ! -L "$file" ]]; then
        sources+=("$file")
    fi
done

if ((${#sources[@]} == 0)); then
    printf 'No source files to format.\n'
    exit 0
fi

if ((check_only)); then
    "$formatter" --style=file --dry-run --Werror -- "${sources[@]}"
    printf 'Formatting check passed (%d files).\n' "${#sources[@]}"
else
    "$formatter" --style=file -i -- "${sources[@]}"
    printf 'Formatted %d files.\n' "${#sources[@]}"
fi
