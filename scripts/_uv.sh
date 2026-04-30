#!/usr/bin/env bash

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found on PATH." >&2
    echo "Install it from https://docs.astral.sh/uv/ and rerun this script." >&2
    return 1 2>/dev/null || exit 1
fi

_uv_helper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_uv_helper_dir}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/uv-cache-${USER:-user}}"
mkdir -p "${UV_CACHE_DIR}"

export TORCH_CUDA_EXTRA="${TORCH_CUDA_EXTRA:-cu128}"
export UV_OPTIONAL_EXTRAS="${UV_OPTIONAL_EXTRAS:-}"

declare -ag UV_EXTRA_ARGS=("--extra" "${TORCH_CUDA_EXTRA}")
for _extra in ${UV_OPTIONAL_EXTRAS}; do
    UV_EXTRA_ARGS+=("--extra" "${_extra}")
done
unset _extra

uv_sync_project() {
    uv sync --no-dev "${UV_EXTRA_ARGS[@]}" "$@"
}

uv_run() {
    uv run "${UV_EXTRA_ARGS[@]}" "$@"
}
