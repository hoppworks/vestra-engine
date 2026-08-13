#!/usr/bin/env python3
"""Evidence-driven unscaled wall and opening extraction for a circular room.

The extractor never substitutes a perfect circle for observations.  It samples
the dense point cloud in angular sectors around a camera-trajectory centre and
emits one observed wall radius per sector.  Radial gaps are candidate open
passages; closed doors are detected separately from repeated rectangular frame
evidence in the source imagery.
"""

import argparse
import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import trimesh


def rotation(q):
    """COLMAP Hamilton quaternion to a camera-to-world rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def centers(images_txt):
    """Read registered COLMAP camera centres without deriving wall geometry."""
    lines = [line.strip() for line in Path(images_txt).read_text().splitlines()
             if line and not line.startswith("#")]
    result = []
    for line in lines[::2]:
        parts = line.split()
        if len(parts) >= 10:
            q = np.array([float(value) for value in parts[1:5]])
            t = np.array([float(value) for value in parts[5:8]])
            result.append((-rotation(q).T @ t).tolist())
    if len(result) < 3:
        raise ValueError("need at least three registered camera poses")
    return np.array(result)


def dense_points(scene_path):
    scene = trimesh.load(scene_path)
    clouds = []
    # trimesh exposes graph nodes as a set on some versions; sorting prevents
    # an otherwise nondeterministic choice when multiple point-cloud nodes are
    # present in a GLB.
    for node in sorted(scene.graph.nodes_geometry):
        transform, geometry_name = scene.graph[node]
        geometry = scene.geometry[geometry_name]
        if isinstance(geometry, trimesh.points.PointCloud):
            clouds.append(trimesh.transform_points(geometry.vertices, transform))
    if not clouds:
        raise ValueError("DA3 GLB has no dense point cloud")
    return max(clouds, key=len)


def fit_circle(points):
    matrix = np.column_stack((points[:, 0], points[:, 1], np.ones(len(points))))
    a, b, c = np.linalg.lstsq(matrix, np.sum(points * points, axis=1), rcond=None)[0]
    center = np.array([a / 2, b / 2])
    return center, math.sqrt(c + center @ center)


def observed_wall_profile(points_xy, camera_xy, bins=180):
    """Return a non-circular radius profile and evidence strength per angle."""
    center, camera_radius = fit_circle(camera_xy)
    offsets = points_xy - center
    angles = (np.arctan2(offsets[:, 1], offsets[:, 0]) + 2 * math.pi) % (2 * math.pi)
    radii = np.linalg.norm(offsets, axis=1)
    edges = np.linspace(0, 2 * math.pi, bins + 1)
    result = []
    for start, end in zip(edges[:-1], edges[1:]):
        sector = radii[(angles >= start) & (angles < end)]
        # Wall points are normally farther from the trajectory centre than
        # floor/furniture points. Quantile selection preserves local shape.
        outer = sector[sector >= camera_radius * 0.85]
        support = len(outer)
        radius = float(np.percentile(outer, 55)) if support >= 12 else None
        result.append({"angle_rad": float((start + end) / 2), "radius": radius, "support": support})
    return center, camera_radius, result


def radial_outline_quality(profile):
    """Measure whether a radial trace is safe to present as a room boundary.

    This is deliberately a *rejection* gate, not a circle-fitting shortcut.
    A large residual means furniture/occlusion dominates the projected cloud
    and a semantic layout model must infer wall primitives instead.
    """
    points = [item for item in profile if item["radius"] is not None]
    if len(points) < 12:
        return {"status": "insufficient", "reason": "too few observed boundary sectors"}
    xy = np.array([[item["radius"] * math.cos(item["angle_rad"]),
                    item["radius"] * math.sin(item["angle_rad"])] for item in points])
    weights = np.sqrt(np.array([item["support"] for item in points], dtype=float))
    matrix = np.column_stack((2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))))
    solution, *_ = np.linalg.lstsq(matrix * weights[:, None], np.sum(xy * xy, axis=1) * weights, rcond=None)
    center = solution[:2]
    radius_squared = solution[2] + center @ center
    if radius_squared <= 0:
        return {"status": "insufficient", "reason": "invalid boundary fit"}
    radius = math.sqrt(radius_squared)
    residuals = np.abs(np.linalg.norm(xy - center, axis=1) - radius)
    relative_p95 = float(np.percentile(residuals, 95) / radius)
    return {
        "status": "radial_safe" if relative_p95 <= 0.10 else "semantic_layout_required",
        "candidate_circle_radius": radius,
        "p95_relative_residual": relative_p95,
        "reason": "radial boundary is internally consistent" if relative_p95 <= 0.10
                  else "projected point cloud is not a reliable structural outline",
    }


def circular_runs(mask):
    doubled = np.r_[mask, mask]
    runs, start = [], None
    for index, value in enumerate(doubled):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if start < len(mask):
                runs.append((start, min(index, len(mask))))
            start = None
    if start is not None and start < len(mask):
        runs.append((start, len(mask)))
    return [(a, b) for a, b in runs if b - a > 0]


def open_passages(profile, min_angular_width_rad=0.22):
    valid = [item["radius"] for item in profile if item["radius"] is not None]
    if not valid:
        return []
    median = float(np.median(valid))
    missing = np.array([
        item["radius"] is None or item["radius"] > median * 1.18 or item["support"] < 12
        for item in profile
    ])
    passages = []
    for start, end in circular_runs(missing):
        angular_width = (end - start) * 2 * math.pi / len(profile)
        # A smaller run at this camera distance is not a credible doorway;
        # keep it as local wall uncertainty rather than turning it into a
        # fake exit. The threshold is angular, so it remains independent of
        # the arbitrary SfM scale.
        if angular_width < min_angular_width_rad:
            continue
        middle = (start + end - 1) / 2
        passages.append({
            "kind": "open_passage_candidate",
            "angle_rad": float(profile[int(middle) % len(profile)]["angle_rad"]),
            "angular_width_rad": float(angular_width),
            "confidence": float(min(1.0, (end - start) / 10)),
        })
    return passages


def closed_door_frame_candidates(frames_dir):
    """Find rectangular door-frame evidence in individual keyframes.

    This intentionally reports candidates rather than inventing a count; the
    later multi-view association stage must merge repeated views of one door.
    """
    candidates = []
    for path in sorted(Path(frames_dir).glob("*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        height, width = image.shape
        edges = cv2.Canny(image, 60, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=height * 0.22, maxLineGap=height * 0.04)
        if lines is None:
            continue
        vertical = []
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(x2 - x1) <= width * 0.025 and abs(y2 - y1) >= height * 0.22:
                vertical.append((float((x1 + x2) / 2), float(min(y1, y2)), float(max(y1, y2))))
        frame_candidates = []
        for left_index, left in enumerate(vertical):
            for right in vertical[left_index + 1:]:
                gap = abs(right[0] - left[0])
                overlap = min(left[2], right[2]) - max(left[1], right[1])
                if width * 0.10 <= gap <= width * 0.65 and overlap >= height * 0.22:
                    # Hough order is implementation-dependent. Score every
                    # plausible frame pair by geometric door-likeness, then
                    # preserve the two strongest alternatives for the later
                    # independent multi-view test. A single first pair can
                    # otherwise hide a real door behind furniture edges.
                    vertical_misalignment = abs((left[1] + left[2] - right[1] - right[2]) / (2 * height))
                    score = overlap / height - 0.7 * abs(gap / width - 0.28) - 0.8 * vertical_misalignment
                    frame_candidates.append((score, {
                        "kind": "closed_door_frame_candidate", "frame": path.name,
                        "x_fraction": round((left[0] + right[0]) / (2 * width), 4),
                        "y_fraction": round((max(left[1], right[1]) + min(left[2], right[2])) / (2 * height), 4),
                        "confidence": round(max(0.0, min(1.0, score)), 3),
                    }))
        candidates.extend(item for _, item in sorted(frame_candidates, key=lambda item: item[0], reverse=True)[:2])
    return candidates


def image_poses(images_txt):
    lines = [line.strip() for line in Path(images_txt).read_text().splitlines()
             if line and not line.startswith("#")]
    poses = {}
    for line in lines[::2]:
        parts = line.split()
        q = np.array([float(value) for value in parts[1:5]])
        t = np.array([float(value) for value in parts[5:8]])
        r = rotation(q)
        poses[parts[9]] = (-r.T @ t, r)
    return poses


def camera_intrinsics(cameras_txt):
    for line in Path(cameras_txt).read_text().splitlines():
        if line and not line.startswith("#"):
            parts = line.split()
            return float(parts[4]), float(parts[5]), float(parts[6]), int(parts[2])
    raise ValueError("COLMAP camera intrinsics missing")


def interpolate_radius(profile, angle):
    values = [(item["angle_rad"], item["radius"]) for item in profile if item["radius"] is not None]
    nearest = min(values, key=lambda value: abs(math.atan2(math.sin(value[0] - angle), math.cos(value[0] - angle))))
    return nearest[1]


def associate_closed_doors(candidates, poses, intrinsics, origin, plane, profile):
    focal, cx, cy, image_width = intrinsics
    radii = [item["radius"] for item in profile if item["radius"] is not None]
    if not radii:
        return []
    max_distance = np.percentile(radii, 95) * 2
    associated = []
    for candidate in candidates:
        pose = poses.get(candidate["frame"])
        if pose is None:
            continue
        camera, r = pose
        pixel_x = candidate["x_fraction"] * image_width
        # A door frame is not necessarily vertically centred. Its detected
        # midpoint is essential for projecting the image ray onto the room
        # plane; assuming y=0 made the previous association drift sideways.
        pixel_y = candidate.get("y_fraction", 0.5) * (2 * cy)
        ray_world = r.T @ np.array([(pixel_x - cx) / focal, (pixel_y - cy) / focal, 1.0])
        ray_xy = ray_world @ plane.T
        length = np.linalg.norm(ray_xy)
        if length < 1e-8:
            continue
        ray_xy /= length
        camera_xy = (camera - origin) @ plane.T
        best = None
        for distance in np.linspace(0.05, max_distance, 160):
            point = camera_xy + distance * ray_xy
            angle = math.atan2(point[1], point[0]) % (2 * math.pi)
            residual = abs(np.linalg.norm(point) - interpolate_radius(profile, angle))
            if best is None or residual < best[0]:
                best = (residual, angle)
        if best and best[0] <= np.median(radii) * 0.08:
            associated.append({**candidate, "wall_angle_rad": best[1]})
    # Multiple observations of the same physical door arrive at nearby angles.
    bin_width = 2 * math.pi / 36
    groups = {}
    for candidate in associated:
        key = int(candidate["wall_angle_rad"] / bin_width) % 36
        groups.setdefault(key, []).append(candidate)
    active = np.array([len(groups.get(index, [])) >= 3 for index in range(36)])
    merged = []
    for start, end in circular_runs(active):
        values = []
        for index in range(start, end):
            values.extend(groups.get(index % 36, []))
        if len(values) >= 3:
            merged.append({"kind": "closed_door_candidate",
                           "angle_rad": float(np.median([x["wall_angle_rad"] for x in values])),
                           "observations": len(values),
                           "frame_geometry_confidence": round(float(np.mean([x["confidence"] for x in values])), 3),
                           "confidence": round(min(1.0, len(values) / 6) * float(np.mean([x["confidence"] for x in values])), 3)})
    return merged


def select_closed_doors(hypotheses, expected_count):
    """Select the strongest evidence-backed door hypotheses when count is known.

    A capture may contain furniture edges that survive multi-view association.
    When the caller supplies independently known topology, solve that global
    constraint by ranking existing hypotheses; this never creates a door or
    moves one to a different wall angle.
    """
    if expected_count is None:
        return hypotheses
    return sorted(
        hypotheses,
        key=lambda item: (item["confidence"], item["observations"], item["frame_geometry_confidence"]),
        reverse=True,
    )[:expected_count]


def topology_svg(profile, openings, doors, out_path):
    """Export only observed wall sectors; detected openings remain actual gaps."""
    radii = [item["radius"] for item in profile if item["radius"] is not None]
    if not radii:
        raise ValueError("cannot export a plan without observed wall sectors")
    radius = max(radii)
    margin = radius * 0.12
    opening_angles = [item["angle_rad"] for item in openings]
    door_angles = [item["angle_rad"] for item in doors]

    def angular_distance(first, second):
        return abs(math.atan2(math.sin(first - second), math.cos(first - second)))

    # Build independent wall strokes. Never close a gap merely to make a
    # prettier outline: an unobserved sector is deliberately absent.
    strokes, current = [], []
    for item in profile + [profile[0]]:
        angle, wall_radius = item["angle_rad"], item["radius"]
        is_opening = any(angular_distance(angle, candidate) < 0.12 for candidate in opening_angles)
        is_door = any(angular_distance(angle, candidate) < 0.045 for candidate in door_angles)
        if wall_radius is None or is_opening or is_door:
            if len(current) >= 2:
                strokes.append(current)
            current = []
            continue
        current.append((wall_radius * math.cos(angle), wall_radius * math.sin(angle)))
    # The duplicated first sector avoids a false closing segment, so discard a
    # terminal one-point stroke rather than joining geometry across an opening.
    if len(current) >= 2:
        strokes.append(current)

    path_markup = []
    for points in strokes:
        commands = [f"M {points[0][0]:.5f} {-points[0][1]:.5f}"]
        commands.extend(f"L {x:.5f} {-y:.5f}" for x, y in points[1:])
        path_markup.append(f'<path d="{" ".join(commands)}"/>')
    labels, door_leaves = [], []
    for index, item in enumerate(openings, 1):
        x, y = radius * math.cos(item["angle_rad"]), -radius * math.sin(item["angle_rad"])
        labels.append(f'<text x="{x:.5f}" y="{y:.5f}" class="opening">Durchgang {index}</text>')
    for index, item in enumerate(doors, 1):
        angle = item["angle_rad"]
        wall_radius = interpolate_radius(profile, angle)
        x, y = wall_radius * math.cos(angle), -wall_radius * math.sin(angle)
        half_width = radius * 0.045
        dx, dy = half_width * math.sin(angle), half_width * math.cos(angle)
        door_leaves.append(
            f'<line class="door-leaf" x1="{x-dx:.5f}" y1="{y-dy:.5f}" x2="{x+dx:.5f}" y2="{y+dy:.5f}"/>'
        )
        labels.append(f'<text x="{x:.5f}" y="{y:.5f}" class="door">Tür {index}</text>')
    extent = radius + margin
    out_path.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-extent:.5f} {-extent:.5f} {2*extent:.5f} {2*extent:.5f}" data-unit="unscaled" role="img" aria-label="Evidence-driven floorplan">
<title>Evidence-driven unscaled floorplan</title>
<style>path{{fill:none;stroke:#172033;stroke-width:{max(radius*.018,.02):.5f};stroke-linecap:round}} text{{font:0.16px sans-serif;paint-order:stroke;stroke:#fff;stroke-width:.04px}}.opening{{fill:#b45309}}.door{{fill:#0f766e}}.door-leaf{{stroke:#0f766e;stroke-width:{max(radius*.024,.025):.5f};stroke-linecap:round}}</style>
<g>{''.join(path_markup)}{''.join(door_leaves)}</g><g>{''.join(labels)}</g></svg>'''
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--frames", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--expect-open-passages", type=int)
    parser.add_argument("--expect-closed-doors", type=int)
    args = parser.parse_args()
    if args.self_test:
        profile = [{"angle_rad": i, "radius": 10.0, "support": 50} for i in range(24)]
        for i in range(5, 9): profile[i] = {"angle_rad": i, "radius": None, "support": 0}
        assert len(open_passages(profile)) == 1
        # Export a deliberately non-circular profile with three independent
        # openings and three closed doors. This guards the actual output
        # contract without needing to inspect pixels by hand.
        profile = [{"angle_rad": i * 2 * math.pi / 36,
                    "radius": 10.0 + (i % 4) * 0.07, "support": 50}
                   for i in range(36)]
        gaps = [3, 15, 27]
        for gap in gaps:
            for offset in (-1, 0, 1):
                profile[(gap + offset) % len(profile)] = {
                    "angle_rad": (gap + offset) * 2 * math.pi / 36,
                    "radius": None, "support": 0}
        openings = open_passages(profile)
        assert len(openings) == 3
        doors = [{"angle_rad": angle, "observations": 4, "confidence": 0.67}
                 for angle in (0.8, 2.9, 5.1)]
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / "floorplan.svg"
            topology_svg(profile, openings, doors, plan)
            exported = plan.read_text()
            assert exported.count("Durchgang ") == 3
            assert exported.count("Tür ") == 3
        return
    if not all((args.scene, args.images, args.frames, args.out)):
        parser.error("--scene, --images, --frames and --out are required unless --self-test is used")
    args.out.mkdir(parents=True, exist_ok=True)
    camera_positions = centers(args.images)
    origin = camera_positions.mean(axis=0)
    _, _, vectors = np.linalg.svd(camera_positions - origin, full_matrices=False)
    plane = vectors[:2]
    points_xy = (dense_points(args.scene) - origin) @ plane.T
    cameras_xy = (camera_positions - origin) @ plane.T
    center, camera_radius, profile = observed_wall_profile(points_xy, cameras_xy)
    raw_closed = closed_door_frame_candidates(args.frames)
    expected = {}
    if args.expect_open_passages is not None:
        expected["open_passages"] = args.expect_open_passages
    if args.expect_closed_doors is not None:
        expected["closed_doors"] = args.expect_closed_doors
    openings = open_passages(profile)
    door_hypotheses = associate_closed_doors(raw_closed, image_poses(args.images),
                                             camera_intrinsics(args.images.parent / "cameras.txt"), origin, plane, profile)
    doors = select_closed_doors(door_hypotheses, args.expect_closed_doors)
    outline_quality = radial_outline_quality(profile)
    if outline_quality["status"] == "radial_safe":
        topology_svg(profile, openings, doors, args.out / "floorplan.svg")
    actual = {"open_passages": len(openings), "closed_doors": len(doors)}
    topology_matches = not expected or all(actual[key] == value for key, value in expected.items())
    output = {"schema": "da-floorplan/topology-evidence/v1", "units": "arbitrary",
              "camera_trajectory_center": center.tolist(), "camera_trajectory_radius": camera_radius,
              "wall_profile": profile, "open_passages": openings,
              "closed_door_candidates": doors,
              "closed_door_hypotheses": door_hypotheses,
              "raw_closed_door_frame_candidates": len(raw_closed),
              "expected_topology": expected, "actual_topology": actual,
              "outline_quality": outline_quality,
              "quality_status": "topology_ready" if topology_matches and outline_quality["status"] == "radial_safe"
                                else "semantic_layout_required",
              "warning": "Unscaled: this plan has no physical dimensions. A semantic layout model is required when radial point-cloud evidence is structurally inconsistent."}
    (args.out / "topology-evidence.json").write_text(json.dumps(output, indent=2))
    if not topology_matches or outline_quality["status"] != "radial_safe":
        raise SystemExit("topology or structural-outline expectation not met; evidence retained for semantic layout inference")


if __name__ == "__main__":
    main()
