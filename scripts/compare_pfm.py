#!/usr/bin/env python3
"""Compare two float PFM depth maps and emit machine-readable parity metrics."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


def read_pfm(path: Path) -> tuple[int, int, list[float]]:
    with path.open("rb") as stream:
        if stream.readline().strip() != b"Pf":
            raise ValueError(f"{path}: expected single-channel PFM")
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        endian = "<" if scale < 0 else ">"
        payload = stream.read()
    expected = width * height * 4
    if len(payload) != expected:
        raise ValueError(f"{path}: expected {expected} data bytes, found {len(payload)}")
    values = list(struct.unpack(f"{endian}{width * height}f", payload))
    return width, height, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()

    ref_w, ref_h, reference = read_pfm(args.reference)
    got_w, got_h, candidate = read_pfm(args.candidate)
    if (ref_w, ref_h) != (got_w, got_h):
        raise SystemExit(
            f"shape mismatch: reference={ref_w}x{ref_h}, candidate={got_w}x{got_h}"
        )

    count = len(reference)
    ref_mean = math.fsum(reference) / count
    got_mean = math.fsum(candidate) / count
    deltas = [got - ref for ref, got in zip(reference, candidate)]
    covariance = math.fsum(
        (ref - ref_mean) * (got - got_mean)
        for ref, got in zip(reference, candidate)
    )
    ref_variance = math.fsum((value - ref_mean) ** 2 for value in reference)
    got_variance = math.fsum((value - got_mean) ** 2 for value in candidate)
    denominator = math.sqrt(ref_variance * got_variance)
    correlation = covariance / denominator if denominator else float("nan")

    result = {
        "shape": [ref_h, ref_w],
        "count": count,
        "reference_min": min(reference),
        "reference_max": max(reference),
        "candidate_min": min(candidate),
        "candidate_max": max(candidate),
        "pearson_r": correlation,
        "mae": math.fsum(abs(delta) for delta in deltas) / count,
        "rmse": math.sqrt(math.fsum(delta * delta for delta in deltas) / count),
        "max_abs_error": max(abs(delta) for delta in deltas),
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
