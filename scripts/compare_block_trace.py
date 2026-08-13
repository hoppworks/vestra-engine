#!/usr/bin/env python3
"""Locate the first divergent multi-view transformer block.

`VESTRA_TRACE_DIR` makes the temporary C++ oracle probe and Vestra Engine emit
contiguous little-endian F32 tensors in `[view][token][channel]` order. This
tool reports one deterministic MAE/max pair per block; it intentionally makes
no pass/fail claim because its job is to select the next operator-level seam.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


def read_values(path: Path) -> list[float]:
    data = path.read_bytes()
    if not data or len(data) % 4:
        raise ValueError(f"{path}: expected a non-empty F32 binary tensor")
    return list(struct.unpack(f"<{len(data) // 4}f", data))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp-dir", type=Path, required=True)
    parser.add_argument("--rust-dir", type=Path, required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = []
    for block in range(arguments.blocks):
        reference = read_values(arguments.cpp_dir / f"cpp-block-{block}.f32")
        candidate = read_values(arguments.rust_dir / f"rust-block-{block}.f32")
        if len(reference) != len(candidate):
            raise ValueError(f"block {block}: tensor lengths differ")
        differences = [abs(left - right) for left, right in zip(reference, candidate)]
        report.append(
            {
                "block": block,
                "values": len(reference),
                "mae": math.fsum(differences) / len(differences),
                "max_abs": max(differences),
            }
        )
    result = {"blocks": report}
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
