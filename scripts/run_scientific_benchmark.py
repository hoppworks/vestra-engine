#!/usr/bin/env python3
"""Reproducible DA3 runtime benchmark with randomized independent trials.

Run inside the pinned ``depth-bench`` container.  Each arm loads its model in a
fresh process, performs one untimed warm-up, then emits raw steady-state samples.
CPU and GPU suites are randomized separately with a fixed seed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(os.environ.get("DA3_BENCH_ROOT", "/benchroot"))
RS_ROOT = ROOT / "depth-anything-rs"
IMAGE = Path(
    os.environ.get("DA3_BENCH_IMAGE", str(ROOT / "assets/samples/mountains.jpg"))
)
MODELS = {
    "f32": ROOT / "models/depth-anything-base-f32.gguf",
    "q8_0": ROOT / "models/depth-anything-base-q8_0.gguf",
    "q4_k": ROOT / "models/depth-anything-base-q4_k.gguf",
}
ITER_RE = re.compile(r"^iter\[\d+\]_ms=([0-9.]+)$", re.MULTILINE)
RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
T95 = {
    9: 2.262,
    19: 2.093,
}  # two-sided Student-t, 95%, for the supported 10- and 20-trial protocols
OFFICIAL_DA3_CONTAINER = os.environ.get("DA3_OFFICIAL_CONTAINER", "da3-pytorch-bench")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()


def capture_or_unavailable(command: list[str]) -> str:
    """Preserve a failed provenance probe without preventing a timed run."""
    proc = subprocess.run(command, text=True, capture_output=True)
    if proc.returncode == 0:
        return proc.stdout.strip()
    detail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit {proc.returncode}"
    return f"unavailable: {detail}"


def arms(repeat: int, threads: int) -> dict[str, dict]:
    common_cpp = ["depth", "--input", str(IMAGE), "--threads", str(threads), "--repeat", str(repeat)]
    torch_base = [
        "podman", "exec",
        "-e", "PYTHONPATH=/opt/da3/src",
        "-e", f"OMP_NUM_THREADS={threads}",
        "-e", f"MKL_NUM_THREADS={threads}",
        OFFICIAL_DA3_CONTAINER,
        "python", "/benchroot/depth-anything-rs/scripts/bench_official_da3.py",
        "--model", "depth-anything/DA3-BASE",
        "--image", "/benchroot/src/depth-anything.cpp/assets/samples/mountains.jpg",
        "--res", "504", "--threads", str(threads), "--warmup", "1", "--repeat", str(repeat),
    ]
    result: dict[str, dict] = {}
    for device, binary in (("cpu", ROOT / "build/examples/cli/da3-cli"),
                           ("gpu", ROOT / "build-cuda/examples/cli/da3-cli")):
        for quant in MODELS:
            result[f"cpp-{quant}-{device}"] = {
                "suite": device,
                "precision": quant,
                "command": [str(binary), *common_cpp, "--model", str(MODELS[quant])],
                "env": {},
            }
    result["pytorch-f32-cpu"] = {
        "suite": "cpu", "precision": "f32", "command": torch_base,
        "env": {"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)},
        # /usr/bin/time observes the Podman client, not the persistent Python
        # container process. Do not publish that wrapper RSS as model memory.
        "rss_available": False,
    }
    result["rust-f32-cpu"] = {
        "suite": "cpu", "precision": "f32",
        "command": [str(RS_ROOT / "target/release/da"), "bench", "--model", str(MODELS["f32"]),
                    "--image", str(IMAGE), "--warmup", "1", "--repeat", str(repeat)],
        "env": {"RAYON_NUM_THREADS": str(threads)},
    }
    return result


def run_arm(spec: dict) -> dict:
    env = os.environ.copy()
    env.update(spec["env"])
    command = ["/usr/bin/time", "-v", *spec["command"]]
    started = time.time()
    proc = subprocess.run(command, text=True, capture_output=True, env=env)
    if proc.returncode:
        raise RuntimeError(f"arm failed ({proc.returncode}): {' '.join(spec['command'])}\n{proc.stdout}\n{proc.stderr}")
    samples = [float(value) for value in ITER_RE.findall(proc.stdout)]
    if not samples:
        raise RuntimeError(f"no raw samples in output:\n{proc.stdout}")
    rss = RSS_RE.search(proc.stderr)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p95_ms": sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
        "rss_mb": int(rss.group(1)) / 1024 if rss and spec.get("rss_available", True) else None,
        "wall_s": time.time() - started,
    }


def summarize(records: list[dict]) -> dict:
    medians = [row["median_ms"] for row in records]
    mean = statistics.fmean(medians)
    sd = statistics.stdev(medians)
    sem = sd / math.sqrt(len(medians))
    critical = T95.get(len(medians) - 1, 1.96)
    all_samples = [sample for row in records for sample in row["samples_ms"]]
    rss_values = [row["rss_mb"] for row in records if row["rss_mb"] is not None]
    return {
        "trials": len(records),
        "samples": len(all_samples),
        "trial_median_mean_ms": mean,
        "trial_median_median_ms": statistics.median(medians),
        "trial_median_sd_ms": sd,
        "mean_95ci_ms": [mean - critical * sem, mean + critical * sem],
        "all_samples_p95_ms": sorted(all_samples)[math.ceil(0.95 * len(all_samples)) - 1],
        "rss_mean_mb": statistics.fmean(rss_values) if rss_values else None,
    }


def markdown(data: dict) -> str:
    lines = [
        "# Scientific benchmark results", "",
        f"Generated: `{data['created_utc']}`", "",
        f"The primary estimator is the arithmetic mean of {data['protocol']['trials']} independent trial medians;",
        "the interval is a two-sided 95% Student-t confidence interval across trials.", "",
        "| Runtime | Device | Precision | Mean median (ms) | 95% CI (ms) | SD (ms) | p95 pooled (ms) | RSS (MiB) |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in sorted(data["summary"].items(), key=lambda item: (data["arms"][item[0]]["suite"], item[1]["trial_median_mean_ms"])):
        lo, hi = row["mean_95ci_ms"]
        runtime, precision, device = name.split("-")
        rss = "n/a" if row["rss_mean_mb"] is None else f"{row['rss_mean_mb']:.1f}"
        lines.append(
            f"| {runtime} | {device} | {precision} | {row['trial_median_mean_ms']:.3f} | "
            f"[{lo:.3f}, {hi:.3f}] | {row['trial_median_sd_ms']:.3f} | "
            f"{row['all_samples_p95_ms']:.3f} | {rss} |"
        )
    lines += [
        "", "## Interpretation rules", "",
        "- Direct implementation comparisons use only F32 arms on the same device.",
        "- Q8_0 and Q4_K are compression/accuracy trade-offs, not same-precision speed claims.",
        "- CPU and GPU results are separate populations and must not be presented as language-only effects.",
        "- Timing excludes model load and image decode; it includes preprocessing, backbone, depth/confidence head and host postprocessing.",
        "- Raw commands, trial order, per-iteration samples, hashes and hardware metadata are preserved in `raw-results.json`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--cooldown", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--output", type=Path, default=RS_ROOT / "docs/benchmarks/2026-08-workhorse")
    parser.add_argument(
        "--cpu-f32-direct",
        action="store_true",
        help="run only the locked C++/ggml-F32 and Rust-F32 CPU comparison",
    )
    parser.add_argument(
        "--cpu-f32-three-way",
        action="store_true",
        help="run the locked C++/ggml, Rust, and official PyTorch F32 CPU comparison",
    )
    args = parser.parse_args()
    if args.trials < 2 or args.repeat < 2:
        parser.error("--trials and --repeat must both be >= 2")

    arm_specs = arms(args.repeat, args.threads)
    if args.cpu_f32_direct and args.cpu_f32_three_way:
        parser.error("--cpu-f32-direct and --cpu-f32-three-way are mutually exclusive")
    if args.cpu_f32_direct:
        arm_specs = {
            name: spec
            for name, spec in arm_specs.items()
            if name in {"cpp-f32-cpu", "rust-f32-cpu"}
        }
    elif args.cpu_f32_three_way:
        arm_specs = {
            name: spec
            for name, spec in arm_specs.items()
            if name in {"cpp-f32-cpu", "rust-f32-cpu", "pytorch-f32-cpu"}
        }
    selected_models = {"f32": MODELS["f32"]} if (args.cpu_f32_direct or args.cpu_f32_three_way) else MODELS
    software = {
        "cpp_commit": capture_or_unavailable(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "ggml_commit": capture_or_unavailable(["git", "-C", str(ROOT / "third_party/ggml"), "rev-parse", "HEAD"]),
        "rust_commit": capture_or_unavailable(["git", "-C", str(RS_ROOT), "rev-parse", "HEAD"]),
    }
    hardware = {
        "platform": platform.platform(),
        "cpu": capture(["bash", "-lc", "lscpu | sed -n 's/^Model name:[[:space:]]*//p'"]),
        "logical_cpus": os.cpu_count(),
        "memory": capture(["bash", "-lc", "free -h | sed -n '2p'"]),
    }
    if args.cpu_f32_three_way:
        software.update({
            "official_pytorch_da3_commit": capture_or_unavailable([
                "git", "-C", str(ROOT / "third_party/Depth-Anything-3"), "rev-parse", "HEAD"
            ]),
            "official_pytorch": capture_or_unavailable([
                "podman", "exec", OFFICIAL_DA3_CONTAINER, "python", "-c", "import torch; print(torch.__version__)"
            ]),
        })
    elif not args.cpu_f32_direct:
        software.update({
            "torch": capture([str(ROOT / ".venv/bin/python"), "-c", "import torch; print(torch.__version__)"]),
            "cuda_compiler": capture(["/usr/local/cuda-13.0/bin/nvcc", "--version"]),
        })
        hardware["gpu"] = capture([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"
        ])
    data = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": vars(args) | {"output": str(args.output)},
        "input": {"path": str(IMAGE), "sha256": sha256(IMAGE)},
        "models": {name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                   for name, path in selected_models.items()},
        "software": software,
        "hardware": hardware,
        "arms": {name: {k: v for k, v in spec.items() if k != "command"} | {"command": spec["command"]}
                 for name, spec in arm_specs.items()},
        "runs": [],
    }

    rng = random.Random(args.seed)
    for suite in sorted({spec["suite"] for spec in arm_specs.values()}):
        names = [name for name, spec in arm_specs.items() if spec["suite"] == suite]
        for trial in range(1, args.trials + 1):
            order = names[:]
            rng.shuffle(order)
            for position, name in enumerate(order, 1):
                if data["runs"]:
                    time.sleep(args.cooldown)
                print(f"[{suite} trial {trial}/{args.trials} arm {position}/{len(order)}] {name}", flush=True)
                measurement = run_arm(arm_specs[name])
                data["runs"].append({"suite": suite, "trial": trial, "position": position,
                                     "arm": name, **measurement})
                args.output.mkdir(parents=True, exist_ok=True)
                (args.output / "raw-results.partial.json").write_text(json.dumps(data, indent=2) + "\n")

    data["summary"] = {
        name: summarize([row for row in data["runs"] if row["arm"] == name])
        for name in arm_specs
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "raw-results.json").write_text(json.dumps(data, indent=2) + "\n")
    (args.output / "RESULTS.md").write_text(markdown(data))
    partial = args.output / "raw-results.partial.json"
    if partial.exists():
        partial.unlink()
    print(markdown(data))


if __name__ == "__main__":
    main()
