#!/usr/bin/env bash
# gen_baseline.sh — Task 23, Step 3: build the additive C++ `bench_components`
# target (tests/bench_components.cpp), run it, and write its JSON output to
# depth-anything-rs/baseline.json. This file is what criterion benchmarks on
# the Rust side (e.g. Task 6/9's GEMM benchmarks) compare against.
#
# Graceful degradation, matching compare_e2e.sh's honesty discipline:
#   - no cmake/C++ toolchain            -> write a "skipped" placeholder, exit 0
#   - cmake configure/build fails       -> write a "skipped" placeholder, exit 0
#   - bench_components itself prints a  -> pass that JSON straight through
#     {"skipped": true, ...} marker        (e.g. no ../models/*.gguf present)
#   - real measurement                  -> write it verbatim as baseline.json
#
# Usage: scripts/gen_baseline.sh [gguf_path]
#   Model path may also come from DA_BENCH_GGUF (arg wins). Defaults to
#   ../models/depth-anything-base-f32.gguf (relative to the C++ repo root),
#   matching the convention the rest of tests/CMakeLists.txt uses.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"          # depth-anything-rs/
REPO_ROOT="$(cd "${RS_ROOT}/.." && pwd)"           # top-level C++ project
OUT_JSON="${RS_ROOT}/baseline.json"

GGUF="${DA_BENCH_GGUF:-${REPO_ROOT}/models/depth-anything-base-f32.gguf}"
if [[ $# -gt 0 ]]; then
    GGUF="$1"
fi

write_skip() {
    local reason="$1"
    echo "{\"skipped\": true, \"reason\": \"${reason}\"}" > "${OUT_JSON}"
    echo "wrote placeholder ${OUT_JSON}: ${reason}"
}

# ---------------------------------------------------------------------
# 1. Toolchain check.
# ---------------------------------------------------------------------
if ! command -v cmake >/dev/null 2>&1; then
    write_skip "no cmake found on PATH; cannot build bench_components in this environment"
    exit 0
fi

# ---------------------------------------------------------------------
# 2. Configure + build the additive bench_components target.
# ---------------------------------------------------------------------
BUILD_DIR="${REPO_ROOT}/build"
if ! cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release >/tmp/gen_baseline_cmake_configure.log 2>&1; then
    write_skip "cmake configure failed; see /tmp/gen_baseline_cmake_configure.log (e.g. third_party/ggml submodule not populated)"
    exit 0
fi
if ! cmake --build "${BUILD_DIR}" --target bench_components -j >/tmp/gen_baseline_cmake_build.log 2>&1; then
    write_skip "cmake build of bench_components failed; see /tmp/gen_baseline_cmake_build.log"
    exit 0
fi

# ---------------------------------------------------------------------
# 3. Locate the built binary.
# ---------------------------------------------------------------------
BIN=""
for candidate in "${BUILD_DIR}/tests/bench_components" "${BUILD_DIR}/bench_components"; do
    if [[ -x "${candidate}" ]]; then
        BIN="${candidate}"
        break
    fi
done
if [[ -z "${BIN}" ]]; then
    write_skip "bench_components built but binary not found under ${BUILD_DIR}"
    exit 0
fi

# ---------------------------------------------------------------------
# 4. Run it (from the repo root, so its default relative model path
#    resolves the same way tests/CMakeLists.txt's WORKING_DIRECTORY does)
#    and capture stdout JSON verbatim -- whether it's a real measurement
#    or the binary's own {"skipped": true, ...} marker.
# ---------------------------------------------------------------------
OUTPUT="$(cd "${REPO_ROOT}" && DA_BENCH_GGUF="${GGUF}" "${BIN}")" || {
    write_skip "bench_components exited non-zero; no baseline produced"
    exit 0
}

if [[ -z "${OUTPUT}" ]]; then
    write_skip "bench_components produced no output"
    exit 0
fi

echo "${OUTPUT}" > "${OUT_JSON}"
echo "wrote ${OUT_JSON}:"
echo "${OUTPUT}"
