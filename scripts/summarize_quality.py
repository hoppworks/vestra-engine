#!/usr/bin/env python3
"""Summarize four-image implementation fidelity and quantization drift."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from compare_pfm import read_pfm

IMAGES = ("canyon", "desk", "mountains", "street")
COMPARISONS = {
    "cpp_f32_vs_official_pytorch_f32": ("torch", "cpp"),
    "rust_f32_vs_cpp_f32": ("cpp", "rust"),
    "cpp_cuda_f32_vs_cpp_cpu_f32": ("cpp", "cuda_f32"),
    "cpp_q8_0_vs_cpp_f32": ("cpp", "q8"),
    "cpp_q4_k_vs_cpp_f32": ("cpp", "q4"),
}


def metrics(reference_path: Path, candidate_path: Path) -> dict:
    rw, rh, reference = read_pfm(reference_path)
    cw, ch, candidate = read_pfm(candidate_path)
    if (rw, rh) != (cw, ch):
        raise ValueError(f"shape mismatch: {reference_path} vs {candidate_path}")
    count = len(reference)
    ref_mean = math.fsum(reference) / count
    got_mean = math.fsum(candidate) / count
    deltas = [got - ref for ref, got in zip(reference, candidate)]
    covariance = math.fsum((ref - ref_mean) * (got - got_mean) for ref, got in zip(reference, candidate))
    ref_variance = math.fsum((value - ref_mean) ** 2 for value in reference)
    got_variance = math.fsum((value - got_mean) ** 2 for value in candidate)
    return {
        "shape": [rh, rw],
        "pearson_r": covariance / math.sqrt(ref_variance * got_variance),
        "mae": math.fsum(abs(delta) for delta in deltas) / count,
        "rmse": math.sqrt(math.fsum(delta * delta for delta in deltas) / count),
        "max_abs_error": max(abs(delta) for delta in deltas),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("/tmp"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "corpus": list(IMAGES),
        "acceptance": {
            "f32_implementation_fidelity": "Pearson r >= 0.9999 and MAE <= 0.005 on every image",
            "q8_fidelity": "Pearson r >= 0.9999 and MAE <= 0.005 on every image",
            "q4": "reported as an unthresholded compression trade-off",
        },
        "comparisons": {},
    }
    for label, (reference, candidate) in COMPARISONS.items():
        rows = {
            image: metrics(args.directory / f"{reference}_{image}.pfm", args.directory / f"{candidate}_{image}.pfm")
            for image in IMAGES
        }
        thresholded = label != "cpp_q4_k_vs_cpp_f32"
        result["comparisons"][label] = {
            "images": rows,
            "mean_pearson_r": math.fsum(row["pearson_r"] for row in rows.values()) / len(rows),
            "mean_mae": math.fsum(row["mae"] for row in rows.values()) / len(rows),
            "passes_declared_threshold": (
                all(row["pearson_r"] >= 0.9999 and row["mae"] <= 0.005 for row in rows.values())
                if thresholded else None
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
