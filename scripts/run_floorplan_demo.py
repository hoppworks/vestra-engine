#!/usr/bin/env python3
"""One-command local video-to-3D/unscaled-floorplan demonstration pipeline."""

import argparse
import os
import subprocess
from pathlib import Path


def run(args, cwd, env=None):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(args, cwd=cwd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Create an unscaled 3D preview and SVG floorplan demo from a walkthrough video."
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run-id", default="demo")
    parser.add_argument("--process-res", type=int, default=168)
    parser.add_argument("--expect-open-passages", type=int,
                        help="Fail after export unless this many open passages are independently recovered.")
    parser.add_argument("--expect-closed-doors", type=int,
                        help="Fail after export unless this many closed doors are independently recovered.")
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"video does not exist: {args.video}")
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        parser.error("run-id may contain only letters, digits, '-' and '_'")

    workspace = Path(__file__).resolve().parents[1]
    repo = workspace.parent
    da3 = repo / "third_party/depth-anything-3/.venv/bin/da3"
    model = repo / "third_party/models/DA3-BASE"
    if not da3.is_file() or not model.is_dir():
        parser.error("DA3 runtime or DA3-BASE weights are not installed locally")

    run(["cargo", "run", "-p", "da-cli", "--", "scan", "--video", str(args.video.resolve()),
         "--out", str(args.out.resolve()), "--run-id", args.run_id, "--unscaled"], workspace)
    run_root = args.out.resolve() / "runs" / args.run_id
    colmap_text = run_root / "sidecars/colmap/text"
    run([str(repo / "third_party/depth-anything-3/.venv/bin/python"),
         str(workspace / "scripts/colmap_binary_to_text.py"),
         "--input", str(run_root / "sidecars/colmap/sparse/0"), "--output", str(colmap_text)], workspace)

    da3_input = run_root / "sidecars/depth/colmap-input"
    da3_input.mkdir(parents=True, exist_ok=False)
    os.symlink(run_root / "input/frames", da3_input / "images")
    os.symlink(run_root / "sidecars/colmap/sparse", da3_input / "sparse")
    preview = run_root / "sidecars/depth/preview"
    env = os.environ | {"KMP_DUPLICATE_LIB_OK": "TRUE"}
    run([str(da3), "colmap", str(da3_input), "--sparse-subdir", "0", "--model-dir", str(model),
         "--device", "cpu", "--process-res", str(args.process_res), "--export-format", "glb",
         "--export-dir", str(preview), "--auto-cleanup", "--num-max-points", "300000"], workspace, env)
    final = run_root / "final-demo"
    topology_command = [str(repo / "third_party/depth-anything-3/.venv/bin/python"),
         str(workspace / "scripts/floorplan_topology.py"), "--scene", str(preview / "scene.glb"),
         "--images", str(colmap_text / "images.txt"),
         "--frames", str(run_root / "input/frames"),
         "--out", str(final)]
    if args.expect_open_passages is not None:
        topology_command += ["--expect-open-passages", str(args.expect_open_passages)]
    if args.expect_closed_doors is not None:
        topology_command += ["--expect-closed-doors", str(args.expect_closed_doors)]
    run(topology_command, workspace)
    print(f"3D preview: {preview / 'scene.glb'}")
    print(f"Evidence-driven floorplan: {final / 'floorplan.svg'}")
    print(f"Topology evidence: {final / 'topology-evidence.json'}")


if __name__ == "__main__":
    main()
