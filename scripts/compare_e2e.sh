#!/usr/bin/env bash
# compare_e2e.sh — Task 22, Step 5/6: run the C++ reference CLI (`da3-cli`)
# and the Rust `da bench` subcommand against the same model/image, on the
# same --repeat/--warmup protocol, and print both medians side-by-side plus
# the speedup/slowdown factor (rust_median_ms / cpp_median_ms).
#
# Usage:
#   scripts/compare_e2e.sh --model <path.gguf> --image <path.png> [--repeat N] [--warmup W] [--threads T]
#
# Model/image may also come from env vars DA_MODEL / DA_IMAGE (flags win).
#
# Graceful skip: if the C++ CLI binary (`da3-cli`) isn't found at either of
# its two plausible build-output locations (`../build/examples/cli/da3-cli`
# or `../build/da3-cli`, relative to this script's repo root — see
# `examples/cli/CMakeLists.txt`'s `add_executable(da3-cli main.cpp)` and the
# top-level `CMakeLists.txt` for how `add_subdirectory(examples/cli)` is
# wired), this script prints a clear message and exits 0 rather than
# failing with a confusing error — matching the honesty discipline the rest
# of this task's report follows: no C++ binary has ever been built in this
# development environment (`../build/` doesn't exist), so this path is
# expected to be taken here. Exit code 2 is used for real usage errors
# (missing --model/--image), so callers can tell "nothing to compare" (0)
# apart from "you used this wrong" (2).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"          # depth-anything-rs/
REPO_ROOT="$(cd "${RS_ROOT}/.." && pwd)"           # top-level C++ project

MODEL="${DA_MODEL:-}"
IMAGE="${DA_IMAGE:-}"
REPEAT=10
WARMUP=1
THREADS=""

usage() {
    echo "usage: $(basename "$0") --model <path.gguf> --image <path.png> [--repeat N] [--warmup W] [--threads T]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --repeat) REPEAT="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${MODEL}" || -z "${IMAGE}" ]]; then
    echo "error: --model and --image (or DA_MODEL/DA_IMAGE) are required" >&2
    usage
    exit 2
fi

# ---------------------------------------------------------------------
# Locate the C++ reference CLI binary.
# ---------------------------------------------------------------------
CPP_BIN=""
for candidate in "${REPO_ROOT}/build/examples/cli/da3-cli" "${REPO_ROOT}/build/da3-cli"; do
    if [[ -x "${candidate}" ]]; then
        CPP_BIN="${candidate}"
        break
    fi
done

if [[ -z "${CPP_BIN}" ]]; then
    echo "C++ CLI binary not found at either:"
    echo "  ${REPO_ROOT}/build/examples/cli/da3-cli"
    echo "  ${REPO_ROOT}/build/da3-cli"
    echo "run 'cmake --build ../build' first (from ${RS_ROOT}), or 'cmake -B build && cmake --build build' from ${REPO_ROOT} if the build directory doesn't exist yet."
    echo "skipping E2E comparison; nothing more to do."
    exit 0
fi

# ---------------------------------------------------------------------
# Locate (build if needed) the Rust `da` binary.
# ---------------------------------------------------------------------
DA_BIN="${RS_ROOT}/target/release/da"
if [[ ! -x "${DA_BIN}" ]]; then
    echo "Rust 'da' binary not found at ${DA_BIN}; building it now (cargo build --release -p da-cli)..." >&2
    (cd "${RS_ROOT}" && cargo build --release -p da-cli)
fi
if [[ ! -x "${DA_BIN}" ]]; then
    echo "error: cargo build succeeded but ${DA_BIN} still doesn't exist" >&2
    exit 1
fi

if [[ ! -f "${MODEL}" ]]; then
    echo "error: model not found: ${MODEL}" >&2
    exit 2
fi
if [[ ! -f "${IMAGE}" ]]; then
    echo "error: image not found: ${IMAGE}" >&2
    exit 2
fi

echo "== compare_e2e: model=${MODEL} image=${IMAGE} repeat=${REPEAT} warmup=${WARMUP} threads=${THREADS:-<default>} =="

# ---------------------------------------------------------------------
# Run the C++ CLI's built-in bench hook: `da3-cli depth --model M --input I
# --repeat N [--threads T]` (src/cli.hpp's Parsed.repeat > 0 with a single
# --input routes to examples/cli/main.cpp's cmd_depth_bench). Its stdout
# line looks like:
#   bench: out=WxH threads=T load=Xms infer=Yms/iter (median over N, min=A max=B p90=C)
# The `infer=Y` field (median ms/iter, excluding model load) is what's
# comparable to `da bench`'s median_ms.
# ---------------------------------------------------------------------
CPP_ARGS=(depth --model "${MODEL}" --input "${IMAGE}" --repeat "${REPEAT}")
if [[ -n "${THREADS}" ]]; then
    CPP_ARGS+=(--threads "${THREADS}")
fi

echo "-- C++ (da3-cli) --"
CPP_OUT="$("${CPP_BIN}" "${CPP_ARGS[@]}" 2>&1)"
echo "${CPP_OUT}"
CPP_MEDIAN="$(echo "${CPP_OUT}" | grep -oE 'infer=[0-9.]+ms/iter' | head -1 | grep -oE '[0-9.]+' || true)"

# ---------------------------------------------------------------------
# Run `da bench` with the matching protocol.
# ---------------------------------------------------------------------
echo "-- Rust (da bench) --"
RUST_OUT="$("${DA_BIN}" bench --model "${MODEL}" --image "${IMAGE}" --repeat "${REPEAT}" --warmup "${WARMUP}" 2>&1)"
echo "${RUST_OUT}"
RUST_MEDIAN="$(echo "${RUST_OUT}" | grep -oE '^median_ms=[0-9.]+' | head -1 | cut -d= -f2 || true)"

echo "== summary =="
if [[ -z "${CPP_MEDIAN}" ]]; then
    echo "error: could not parse C++ median from da3-cli output above" >&2
    exit 1
fi
if [[ -z "${RUST_MEDIAN}" ]]; then
    echo "error: could not parse Rust median_ms from da bench output above" >&2
    exit 1
fi

echo "cpp_median_ms=${CPP_MEDIAN}"
echo "rust_median_ms=${RUST_MEDIAN}"
FACTOR="$(awk -v r="${RUST_MEDIAN}" -v c="${CPP_MEDIAN}" 'BEGIN { if (c == 0) { print "inf" } else { printf "%.4f", r / c } }')"
echo "factor_rust_over_cpp=${FACTOR}  # >1.0 = Rust is slower than C++, <1.0 = Rust is faster"
