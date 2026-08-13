#!/usr/bin/env python3
"""Steady-state F32 benchmark for the pinned official Depth Anything 3 code.

The image is decoded once and the model is loaded once, before timing. Each
timed iteration then performs official DA3 preprocessing, the DA3-BASE model
forward pass, and official output conversion. The high-level API's automatic
mixed precision wrapper is intentionally not used: this runner is a CPU F32
comparison against the F32 C++/ggml and Rust arms.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="depth-anything/DA3-BASE")
    parser.add_argument("--image", required=True)
    parser.add_argument("--res", type=int, default=504)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--output-pfm", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repeat < 1 or args.threads < 1:
        raise SystemExit("--warmup must be non-negative; --repeat and --threads must be positive")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")

    # Decode once, outside both warm-up and measured inference.
    image = Image.open(args.image).convert("RGB")
    model = DepthAnything3.from_pretrained(args.model).to(device).eval()
    parameter_dtype = next(model.parameters()).dtype
    if parameter_dtype != torch.float32:
        raise RuntimeError(f"expected F32 model parameters, found {parameter_dtype}")

    @torch.inference_mode()
    def infer_once():
        images_cpu, extrinsics, intrinsics = model._preprocess_inputs(
            [image], process_res=args.res, process_res_method="upper_bound_resize"
        )
        images, extrinsics, intrinsics = model._prepare_model_inputs(
            images_cpu, extrinsics, intrinsics
        )
        if images.dtype != torch.float32:
            raise RuntimeError(f"expected F32 model input, found {images.dtype}")
        raw = model.model(images, extrinsics, intrinsics, [], False, False, "saddle_balanced")
        return model.output_processor(raw)

    for _ in range(args.warmup):
        prediction = infer_once()

    samples = []
    for index in range(args.repeat):
        started = time.perf_counter()
        prediction = infer_once()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        samples.append(elapsed_ms)
        print(f"iter[{index}]_ms={elapsed_ms:.6f}", flush=True)

    print(f"output_depth_shape={tuple(prediction.depth.shape)}")
    print(f"mean_ms={statistics.fmean(samples):.6f}")
    print(f"median_ms={statistics.median(samples):.6f}")
    if args.output_pfm is not None:
        depth = np.asarray(prediction.depth[0], dtype="<f4")
        args.output_pfm.parent.mkdir(parents=True, exist_ok=True)
        with args.output_pfm.open("wb") as handle:
            handle.write(f"Pf\n{depth.shape[1]} {depth.shape[0]}\n-1.0\n".encode())
            np.flipud(depth).tofile(handle)
        print(f"wrote_pfm={args.output_pfm}")


if __name__ == "__main__":
    main()
