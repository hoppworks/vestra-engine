#!/usr/bin/env python3
"""Compare pinned C++ PR #2 and Vestra Engine multi-view artifacts.

Both CLIs write `<prefix>_view<N>.pfm` depth maps and matching camera JSON.
This script records per-view depth, W2C extrinsics, and pixel-intrinsics
metrics separately. It exits non-zero if the locked F32 contract is violated.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


def read_pfm(path: Path) -> tuple[int, int, list[float]]:
    with path.open("rb") as handle:
        magic = handle.readline().strip()
        if magic != b"Pf":
            raise ValueError(f"{path}: expected grayscale PFM")
        dimensions = handle.readline().split()
        if len(dimensions) != 2:
            raise ValueError(f"{path}: invalid PFM dimensions")
        width, height = map(int, dimensions)
        scale = float(handle.readline())
        if scale >= 0:
            raise ValueError(f"{path}: expected little-endian PFM")
        values = list(struct.unpack(f"<{width * height}f", handle.read(width * height * 4)))
    return width, height, [
        value
        for row in range(height - 1, -1, -1)
        for value in values[row * width : (row + 1) * width]
    ]


def metrics(reference: list[float], candidate: list[float]) -> dict[str, float]:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("tensors must have equal, non-zero length")
    n = len(reference)
    mean_ref = sum(reference) / n
    mean_candidate = sum(candidate) / n
    covariance = sum((left - mean_ref) * (right - mean_candidate) for left, right in zip(reference, candidate))
    reference_variance = sum((value - mean_ref) ** 2 for value in reference)
    candidate_variance = sum((value - mean_candidate) ** 2 for value in candidate)
    pearson = covariance / math.sqrt(reference_variance * candidate_variance)
    differences = [abs(left - right) for left, right in zip(reference, candidate)]
    return {"pearson_r": pearson, "mae": sum(differences) / n, "max_abs": max(differences)}


def camera_values(path: Path) -> tuple[list[float], list[float]]:
    document = json.loads(path.read_text())
    extrinsics = [value for row in document["extrinsics"] for value in row]
    intrinsics = [value for row in document["intrinsics"] for value in row]
    return extrinsics, intrinsics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp-prefix", type=Path, required=True)
    parser.add_argument("--rust-prefix", type=Path, required=True)
    parser.add_argument("--views", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-pearson", type=float, default=0.9999)
    parser.add_argument("--max-mae", type=float, default=0.005)
    parser.add_argument("--max-extrinsics-mae", type=float, default=0.005)
    parser.add_argument("--max-intrinsics-abs", type=float, default=1.5)
    arguments = parser.parse_args()

    reports = []
    passed = True
    for view in range(arguments.views):
        cpp_width, cpp_height, cpp_depth = read_pfm(Path(f"{arguments.cpp_prefix}_view{view}.pfm"))
        rust_width, rust_height, rust_depth = read_pfm(Path(f"{arguments.rust_prefix}_view{view}.pfm"))
        if (cpp_width, cpp_height) != (rust_width, rust_height):
            raise ValueError(f"view {view}: depth dimensions differ")
        depth = metrics(cpp_depth, rust_depth)
        cpp_extrinsics, cpp_intrinsics = camera_values(Path(f"{arguments.cpp_prefix}_view{view}.json"))
        rust_extrinsics, rust_intrinsics = camera_values(Path(f"{arguments.rust_prefix}_view{view}.json"))
        extrinsics = metrics(cpp_extrinsics, rust_extrinsics)
        intrinsics = metrics(cpp_intrinsics, rust_intrinsics)
        view_passed = (
            depth["pearson_r"] >= arguments.min_pearson
            and depth["mae"] <= arguments.max_mae
            and extrinsics["mae"] <= arguments.max_extrinsics_mae
            and intrinsics["max_abs"] <= arguments.max_intrinsics_abs
        )
        passed = passed and view_passed
        reports.append(
            {
                "view": view,
                "depth": depth,
                "extrinsics": extrinsics,
                "intrinsics": intrinsics,
                "passed": view_passed,
            }
        )

    result = {
        "contract": {
            "min_pearson": arguments.min_pearson,
            "max_mae": arguments.max_mae,
            "max_extrinsics_mae": arguments.max_extrinsics_mae,
            "max_intrinsics_abs": arguments.max_intrinsics_abs,
        },
        "views": reports,
        "passed": passed,
    }
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
