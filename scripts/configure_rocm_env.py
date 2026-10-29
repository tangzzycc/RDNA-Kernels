"""Bind the project's Bash virtualenv activation to its installed ROCm SDK."""

from pathlib import Path


DEACTIVATE_HOOK = '''    # RDNA ROCm deactivate hook
    if [ "${_RDNA_ROCM_PATH_WAS_SET+x}" = x ]; then
        if [ "$_RDNA_ROCM_PATH_WAS_SET" = x ]; then
            ROCM_PATH="${_RDNA_OLD_ROCM_PATH-}"
            export ROCM_PATH
        else
            unset ROCM_PATH
        fi
        unset _RDNA_ROCM_PATH_WAS_SET _RDNA_OLD_ROCM_PATH
    fi

'''

ACTIVATE_HOOK = '''
# RDNA ROCm activate hook
# rocprof-compute otherwise defaults to the system SDK under /opt/rocm.
_RDNA_ROCM_ROOT="$("$VIRTUAL_ENV/bin/rocm-sdk" path --root)" || return 1
_RDNA_OLD_ROCM_PATH="${ROCM_PATH-}"
_RDNA_ROCM_PATH_WAS_SET="${ROCM_PATH+x}"
ROCM_PATH="$_RDNA_ROCM_ROOT"
export ROCM_PATH
unset _RDNA_ROCM_ROOT
'''


def main() -> None:
    activation = Path(__file__).resolve().parents[1] / ".venv/bin/activate"
    text = activation.read_text()
    if "# RDNA ROCm activate hook" in text:
        print("ROCm activation hook is already configured.")
        return

    anchor = "deactivate () {\n"
    if text.count(anchor) != 1:
        raise RuntimeError("Unexpected Bash activation script; no changes made.")

    activation.with_name("activate.before-rocm").write_text(text)
    text = text.replace(anchor, anchor + DEACTIVATE_HOOK, 1)
    activation.write_text(text.rstrip() + "\n" + ACTIVATE_HOOK)
    print(f"Configured ROCm activation in {activation}")


if __name__ == "__main__":
    main()
