#!/usr/bin/env python3
"""Capture an attributable CPU profile for the locked DA3 F32 benchmark.

This is deliberately a *profiling* harness, not the final scientific timing
study.  It records the exact source and binary hashes that produced every
counter sample, refuses an obviously busy host, and alternates Rust/C++ runs.
Use ``run_scientific_benchmark.py`` for the final 10× timing claim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_EVENTS = ",".join(
    (
        "cycles",
        "instructions",
        "branches",
        "branch-misses",
        "cache-references",
        "cache-misses",
        "context-switches",
        "cpu-migrations",
        "page-faults",
    )
)
SOURCE_FILES = (
    "crates/da-engine/src/vit_block.rs",
    "crates/da-engine/src/dpt_head.rs",
    "crates/da-kernels/src/conv.rs",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()


def optional(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode == 0:
        return result.stdout.strip()
    message = result.stderr.strip().splitlines()
    return f"unavailable: {message[-1] if message else f'exit {result.returncode}'}"


def load_average() -> tuple[float, float, float]:
    return tuple(float(value) for value in Path("/proc/loadavg").read_text().split()[:3])  # type: ignore[return-value]


def busy_processes(max_process_cpu: float) -> list[dict[str, object]]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,pcpu=,comm="], text=True, capture_output=True, check=True
    )
    offenders = []
    own_pid = os.getpid()
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) != 3:
            continue
        pid, cpu, command = fields
        if int(pid) != own_pid and float(cpu) > max_process_cpu:
            offenders.append({"pid": int(pid), "cpu_percent": float(cpu), "command": command})
    return offenders


def assert_quiet(max_load_per_cpu: float, max_process_cpu: float) -> dict[str, object]:
    load = load_average()
    cpus = os.cpu_count() or 1
    permitted = cpus * max_load_per_cpu
    # ``loadavg`` decays over minutes.  It is useful provenance but cannot
    # gate the second arm immediately after the first 16-thread arm, which
    # would reject our own completed work.  A large currently-running process
    # is the meaningful safety condition (e.g. a game, compiler, or another
    # benchmark) and has no such lag.
    offenders = busy_processes(max_process_cpu)
    if offenders:
        raise RuntimeError(
            f"host has unrelated CPU-heavy process(es) above {max_process_cpu:.1f}%: {offenders}. "
            "Stop them before profiling."
        )
    return {
        "loadavg": load,
        "logical_cpus": cpus,
        "load_reference_limit": permitted,
        "max_process_cpu": max_process_cpu,
        "busy_processes": offenders,
    }


def source_manifest(root: Path, kernel_root: Path, rust_binary: Path, cpp_binary: Path, image: Path, model: Path) -> dict[str, object]:
    paths = {str(root / relative): sha256(root / relative) for relative in SOURCE_FILES}
    paths[str(kernel_root / "src/lib.rs")] = sha256(kernel_root / "src/lib.rs")
    return {
        "source_sha256": paths,
        "rust_binary_sha256": sha256(rust_binary),
        "cpp_binary_sha256": sha256(cpp_binary),
        "model_sha256": sha256(model),
        "image_sha256": sha256(image),
    }


def parse_perf_json(stderr: str) -> list[dict[str, object]] | None:
    lines = [line for line in stderr.splitlines() if line.lstrip().startswith("{")]
    if not lines:
        return None
    try:
        return [json.loads(line) for line in lines]
    except json.JSONDecodeError:
        return None


def run_profile(
    perf: str, events: str, delay_ms: int, command: list[str], env: dict[str, str]
) -> dict[str, object]:
    # ``perf stat`` inherits events into all worker threads.  Delaying event
    # enablement until after model load + the unmeasured warm-up keeps the
    # hardware sample aligned with the benchmark's measured work.  The bench
    # process keeps running many iterations after the delay, so no timed
    # inference is modified or excluded from its own report.
    invocation = [perf, "stat", "--json-output", "--delay", str(delay_ms), "-e", events, "--", *command]
    started = time.monotonic()
    result = subprocess.run(invocation, text=True, capture_output=True, env=env)
    if result.returncode:
        raise RuntimeError(
            f"profile arm failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return {
        "command": command,
        "wall_seconds": time.monotonic() - started,
        "stdout": result.stdout,
        "perf_json": parse_perf_json(result.stderr),
        "perf_stderr": result.stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/var/roothome/da3-bench"))
    parser.add_argument("--kernel-root", type=Path, default=Path("/var/roothome/da3-kernels"))
    parser.add_argument("--perf", type=str, default="perf")
    parser.add_argument("--events", default=DEFAULT_EVENTS)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50, help="Measured inference iterations per profile process")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--perf-delay-ms",
        type=int,
        default=3_000,
        help="Enable hardware counters after this many process milliseconds (must cover load + warm-up)",
    )
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--max-load-per-cpu", type=float, default=0.15)
    parser.add_argument(
        "--max-process-cpu",
        type=float,
        default=50.0,
        help="Refuse profiling when a non-profile process currently exceeds this CPU percentage",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1 or args.repeat < 2 or args.perf_delay_ms < 0 or args.max_process_cpu <= 0:
        parser.error("--trials must be >= 1, --repeat >= 2, --perf-delay-ms >= 0, and --max-process-cpu > 0")
    perf = shutil.which(args.perf) if os.path.sep not in args.perf else args.perf
    if not perf:
        parser.error(f"perf executable not found: {args.perf}")

    root = args.root.resolve()
    rs_root = root / "depth-anything-rs"
    image = root / "src/depth-anything.cpp/assets/samples/mountains.jpg"
    model = root / "models/depth-anything-base-f32.gguf"
    rust_binary = rs_root / "target/release/da"
    cpp_binary = root / "build/examples/cli/da3-cli"
    for path in (image, model, rust_binary, cpp_binary, args.kernel_root / "src/lib.rs"):
        if not path.is_file():
            parser.error(f"required artifact is missing: {path}")

    args.output.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {
        "schema_version": 1,
        "kind": "hardware-profile",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": {
            "threads": args.threads,
            "warmup": 1,
            "repeat": args.repeat,
            "trials": args.trials,
            "events": args.events,
            "seed": args.seed,
            "cooldown_seconds": args.cooldown,
            "perf_delay_ms": args.perf_delay_ms,
        },
        "host": {
            "platform": platform.platform(),
            "uname": optional(["uname", "-a"]),
            "lscpu": optional(["lscpu"]),
            "governor": optional(["bash", "-lc", "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"]),
            "perf_version": optional([perf, "--version"]),
        },
        "artifacts": source_manifest(rs_root, args.kernel_root.resolve(), rust_binary, cpp_binary, image, model),
        "runs": [],
    }
    rust = [str(rust_binary), "bench", "--model", str(model), "--image", str(image), "--warmup", "1", "--repeat", str(args.repeat)]
    cpp = [str(cpp_binary), "depth", "--input", str(image), "--threads", str(args.threads), "--repeat", str(args.repeat), "--model", str(model)]
    arms = {
        "rust": (rust, {**os.environ, "RAYON_NUM_THREADS": str(args.threads)}),
        "cpp": (cpp, os.environ.copy()),
    }
    rng = random.Random(args.seed)
    for trial in range(1, args.trials + 1):
        order = list(arms)
        rng.shuffle(order)
        for arm in order:
            quiet = assert_quiet(args.max_load_per_cpu, args.max_process_cpu)
            command, env = arms[arm]
            print(f"[trial {trial}/{args.trials}] {arm}", flush=True)
            run = run_profile(perf, args.events, args.perf_delay_ms, command, env)
            data["runs"].append({"trial": trial, "arm": arm, "host_before": quiet, **run})  # type: ignore[index]
            (args.output / "hardware-profile.partial.json").write_text(json.dumps(data, indent=2) + "\n")
            time.sleep(args.cooldown)
    (args.output / "hardware-profile.json").write_text(json.dumps(data, indent=2) + "\n")
    partial = args.output / "hardware-profile.partial.json"
    if partial.exists():
        partial.unlink()
    print(json.dumps({"output": str(args.output / "hardware-profile.json"), "runs": len(data["runs"])}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
