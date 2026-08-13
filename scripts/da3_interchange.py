#!/usr/bin/env python3
"""Validate and index pose-conditioned DA3 per-frame output.

This program deliberately does *not* import, install, download, or execute
Depth Anything 3.  It is the durable boundary after a compatible DA3 runner:
one ``frames/<frame-id>.npz`` for every registered frame in a normalized Rust
``da-video/frame-manifest/v1`` manifest.  The final ``depth.manifest.json`` is
therefore safe for a Rust fusion stage to consume without guessing filenames,
coordinate conventions, or image sizes.

Required arrays in every archive:

* ``depth``: finite ``float32``-compatible ``(H, W)`` depth map
* ``confidence``: finite ``float32``-compatible ``(H, W)`` reliability map
* ``intrinsics``: finite ``(3, 3)`` matrix in the saved depth-map pixel space
* ``world_to_camera``: finite homogeneous ``(4, 4)`` matrix
* ``source_width`` and ``source_height``: scalar dimensions of the input image

``world_to_camera`` and source dimensions are checked against the canonical
COLMAP-normalized frame manifest.  ``intrinsics`` is intentionally not
compared byte-for-byte: DA3 may resize depth maps, so it describes ``depth``'s
pixel space while source dimensions preserve the original camera geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


FRAME_MANIFEST_SCHEMA = "da-video/frame-manifest/v1"
DEPTH_MANIFEST_SCHEMA = "da-floorplan/da3-depth-manifest/v1"
REPORT_SCHEMA = "da-floorplan/da3-interchange-report/v1"
REQUIRED_NPZ_KEYS = frozenset(
    {
        "depth",
        "confidence",
        "intrinsics",
        "world_to_camera",
        "source_width",
        "source_height",
    }
)


class ContractError(RuntimeError):
    """A user-actionable violation at the DA3-to-fusion boundary."""


@dataclass(frozen=True)
class RegisteredFrame:
    frame_id: str
    image: str
    source_width: int
    source_height: int
    world_to_camera: tuple[tuple[float, ...], ...]


def fail(message: str) -> ContractError:
    return ContractError(f"DA3 interchange contract error: {message}")


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise fail(f"{field} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise fail(f"{field} must be a finite number")
    return value


def _safe_relative_image(value: Any, frame_id: str) -> str:
    if not isinstance(value, str) or not value:
        raise fail(f"frame `{frame_id}` has no image path")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise fail(f"frame `{frame_id}` image must be a safe relative path: {value!r}")
    return path.as_posix()


def _matrix(value: Any, rows: int, columns: int, field: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or len(value) != rows:
        raise fail(f"{field} must be a {rows}x{columns} matrix")
    parsed: list[tuple[float, ...]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != columns:
            raise fail(f"{field} must be a {rows}x{columns} matrix")
        parsed.append(tuple(_finite_number(cell, field) for cell in row))
    return tuple(parsed)


def load_registered_frames(manifest_path: Path, frames_dir: Path) -> tuple[str, list[RegisteredFrame]]:
    """Read only the registered frames accepted by pose-conditioned DA3.

    The Rust crate keeps unregistered frames as diagnostic evidence.  They are
    deliberately not sent to pose-conditioned inference because no canonical
    world-to-camera transform exists for them.
    """
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise fail(f"frame manifest does not exist: {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise fail(f"frame manifest is not valid JSON: {error}") from error
    if not isinstance(document, dict):
        raise fail("frame manifest root must be an object")
    if document.get("schema") != FRAME_MANIFEST_SCHEMA:
        raise fail(
            f"expected manifest schema `{FRAME_MANIFEST_SCHEMA}`, got {document.get('schema')!r}"
        )
    run_id = document.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise fail("frame manifest run_id must be a non-empty string")
    coordinates = document.get("coordinate_system")
    if not isinstance(coordinates, dict) or coordinates.get("world_axis") != "z_up" or coordinates.get(
        "pose_convention"
    ) != "world_to_camera":
        raise fail("only z_up/world_to_camera frame manifests are supported")
    raw_frames = document.get("frames")
    if not isinstance(raw_frames, list):
        raise fail("frame manifest frames must be an array")

    resolved_root = frames_dir.resolve()
    seen: set[str] = set()
    frames: list[RegisteredFrame] = []
    for raw in raw_frames:
        if not isinstance(raw, dict):
            raise fail("every frame manifest entry must be an object")
        frame_id = raw.get("id")
        if not isinstance(frame_id, str) or not frame_id:
            raise fail("every frame must have a non-empty id")
        if frame_id in seen:
            raise fail(f"duplicate frame id `{frame_id}`")
        seen.add(frame_id)
        image = _safe_relative_image(raw.get("image"), frame_id)
        image_path = (resolved_root / image).resolve()
        if image_path != resolved_root and resolved_root not in image_path.parents:
            raise fail(f"frame `{frame_id}` image escapes --frames-dir")
        if not image_path.is_file():
            raise fail(f"frame `{frame_id}` image does not exist under --frames-dir: {image}")

        registration = raw.get("registration")
        if not isinstance(registration, dict):
            raise fail(f"frame `{frame_id}` registration must be an object")
        if registration.get("state") != "registered":
            continue
        pose = registration.get("pose")
        if not isinstance(pose, dict):
            raise fail(f"registered frame `{frame_id}` has no pose")
        world_to_camera_3x4 = _matrix(pose.get("world_to_camera"), 3, 4, f"frame `{frame_id}` world_to_camera")
        camera = raw.get("camera")
        intrinsics = camera.get("intrinsics") if isinstance(camera, dict) else None
        if not isinstance(intrinsics, dict):
            raise fail(f"frame `{frame_id}` has no camera intrinsics")
        width, height = intrinsics.get("width"), intrinsics.get("height")
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise fail(f"frame `{frame_id}` source width must be a positive integer")
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise fail(f"frame `{frame_id}` source height must be a positive integer")
        frames.append(
            RegisteredFrame(
                frame_id=frame_id,
                image=image,
                source_width=width,
                source_height=height,
                world_to_camera=world_to_camera_3x4 + ((0.0, 0.0, 0.0, 1.0),),
            )
        )
    if not frames:
        raise fail("manifest has no registered frames; run COLMAP normalization before DA3")
    return run_id, frames


def _np() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise fail("numpy is required to validate NPZ outputs; install it in the DA3 environment") from error
    return np


def _scalar_int(array: Any, field: str, frame_id: str) -> int:
    if getattr(array, "size", None) != 1:
        raise fail(f"frame `{frame_id}` {field} must be a scalar")
    value = array.reshape(()).item()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or int(value) <= 0:
        raise fail(f"frame `{frame_id}` {field} must be a positive integer")
    return int(value)


def _finite_array(array: Any, shape: tuple[int, ...], field: str, frame_id: str, np: Any) -> None:
    if array.shape != shape:
        raise fail(f"frame `{frame_id}` {field} must have shape {shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise fail(f"frame `{frame_id}` {field} must contain only finite numeric values")


def validate_frame_npz(path: Path, frame: RegisteredFrame, *, pose_tolerance: float = 1e-5) -> dict[str, Any]:
    """Validate one archive and return only JSON-safe index metadata."""
    np = _np()
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = REQUIRED_NPZ_KEYS.difference(archive.files)
            if missing:
                raise fail(f"frame `{frame.frame_id}` archive misses required keys: {', '.join(sorted(missing))}")
            depth = archive["depth"]
            confidence = archive["confidence"]
            if depth.ndim != 2 or depth.shape[0] == 0 or depth.shape[1] == 0:
                raise fail(f"frame `{frame.frame_id}` depth must be a non-empty (H, W) array")
            _finite_array(depth, depth.shape, "depth", frame.frame_id, np)
            _finite_array(confidence, depth.shape, "confidence", frame.frame_id, np)
            intrinsics = archive["intrinsics"]
            _finite_array(intrinsics, (3, 3), "intrinsics", frame.frame_id, np)
            if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
                raise fail(f"frame `{frame.frame_id}` intrinsics fx and fy must be positive")
            world_to_camera = archive["world_to_camera"]
            _finite_array(world_to_camera, (4, 4), "world_to_camera", frame.frame_id, np)
            if not np.allclose(world_to_camera[3], [0.0, 0.0, 0.0, 1.0], atol=pose_tolerance, rtol=0):
                raise fail(f"frame `{frame.frame_id}` world_to_camera must be homogeneous (last row [0,0,0,1])")
            expected_pose = np.asarray(frame.world_to_camera, dtype=world_to_camera.dtype)
            if not np.allclose(world_to_camera, expected_pose, atol=pose_tolerance, rtol=0):
                raise fail(
                    f"frame `{frame.frame_id}` world_to_camera differs from the canonical frame manifest; "
                    "DA3 must preserve supplied COLMAP poses"
                )
            source_width = _scalar_int(archive["source_width"], "source_width", frame.frame_id)
            source_height = _scalar_int(archive["source_height"], "source_height", frame.frame_id)
    except OSError as error:
        raise fail(f"cannot open NPZ for frame `{frame.frame_id}`: {path}: {error}") from error
    if source_width != frame.source_width or source_height != frame.source_height:
        raise fail(
            f"frame `{frame.frame_id}` source dimensions {source_width}x{source_height} differ from "
            f"the frame manifest's {frame.source_width}x{frame.source_height}"
        )
    return {
        "id": frame.frame_id,
        "image": frame.image,
        "npz": f"frames/{frame.frame_id}.npz",
        "depth_width": int(depth.shape[1]),
        "depth_height": int(depth.shape[0]),
        "source_width": source_width,
        "source_height": source_height,
    }


def atomic_json_write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        json.dump(document, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def validate_directory(manifest_path: Path, frames_dir: Path, output_dir: Path, *, write_manifest: bool) -> dict[str, Any]:
    run_id, frames = load_registered_frames(manifest_path, frames_dir)
    npz_dir = output_dir / "frames"
    indexed: list[dict[str, Any]] = []
    for frame in frames:
        path = npz_dir / f"{frame.frame_id}.npz"
        if not path.is_file():
            raise fail(
                f"missing pose-conditioned output for registered frame `{frame.frame_id}`: {path}. "
                "This tool does not run DA3; invoke a compatible DA3 sidecar first."
            )
        indexed.append(validate_frame_npz(path, frame))
    document = {
        "schema": DEPTH_MANIFEST_SCHEMA,
        "state": "validated",
        "run_id": run_id,
        "coordinate_system": {"world_axis": "z_up", "pose_convention": "world_to_camera"},
        "source_frame_manifest": str(manifest_path.resolve()),
        "frames": indexed,
    }
    if write_manifest:
        target = output_dir / "depth.manifest.json"
        if target.exists():
            raise fail(f"refusing to overwrite existing durable depth manifest: {target}")
        atomic_json_write(target, document)
    return document


def command_validate(args: argparse.Namespace) -> int:
    try:
        document = validate_directory(args.frame_manifest, args.frames_dir, args.output_dir, write_manifest=args.write_manifest)
        report = {
            "schema": REPORT_SCHEMA,
            "state": "validated",
            "registered_frames": len(document["frames"]),
            "depth_manifest": str((args.output_dir / "depth.manifest.json").resolve()) if args.write_manifest else None,
        }
        if args.report:
            if args.report.exists():
                raise fail(f"refusing to overwrite existing report: {args.report}")
            atomic_json_write(args.report, report)
        print(json.dumps(report, sort_keys=True))
        return 0
    except ContractError as error:
        print(error, file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate DA3 pose-conditioned per-frame NPZ interchange outputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate NPZ files and optionally write depth.manifest.json")
    validate.add_argument("--frame-manifest", type=Path, required=True, help="normalized da-video/frame-manifest/v1 JSON")
    validate.add_argument("--frames-dir", type=Path, required=True, help="root for image paths in the frame manifest")
    validate.add_argument("--output-dir", type=Path, required=True, help="DA3 output root containing frames/<id>.npz")
    validate.add_argument("--write-manifest", action="store_true", help="write output-dir/depth.manifest.json after validation")
    validate.add_argument("--report", type=Path, help="optional new JSON validation report")
    validate.set_defaults(handler=command_validate)
    self_test = subparsers.add_parser("self-test", help="run pure contract tests (does not require DA3)")
    self_test.set_defaults(handler=command_self_test)
    return parser


class ContractTests(unittest.TestCase):
    def test_rejects_unsafe_image_paths(self) -> None:
        with self.assertRaisesRegex(ContractError, "safe relative path"):
            _safe_relative_image("../outside.png", "frame-1")

    def test_parses_homogeneous_pose(self) -> None:
        pose = _matrix([[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 4]], 3, 4, "pose")
        self.assertEqual(pose[2][3], 4.0)

    def test_rejects_nonfinite_numbers(self) -> None:
        with self.assertRaisesRegex(ContractError, "finite"):
            _finite_number(float("nan"), "pose")


def command_self_test(_: argparse.Namespace) -> int:
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    return 0 if result.wasSuccessful() else 1


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
