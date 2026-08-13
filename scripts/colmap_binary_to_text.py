#!/usr/bin/env python3
"""Minimal, dependency-free COLMAP sparse-model reader for the floorplan path.

Only calibrated camera parameters and registered image poses are required by
the DA3/topology hand-off. Keeping this reader local avoids invoking COLMAP's
GUI-linked `model_converter`, which can abort on headless macOS installations.
"""

import argparse
import struct
from pathlib import Path


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}


def read_exact(handle, size):
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("truncated COLMAP binary model")
    return data


def unpack(handle, format_string):
    return struct.unpack(format_string, read_exact(handle, struct.calcsize(format_string)))


def read_cameras(path):
    cameras = []
    with path.open("rb") as handle:
        (count,) = unpack(handle, "<Q")
        for _ in range(count):
            camera_id, model_id, width, height = unpack(handle, "<IiQQ")
            try:
                model, parameter_count = CAMERA_MODELS[model_id]
            except KeyError as error:
                raise ValueError(f"unsupported COLMAP camera model id {model_id}") from error
            params = unpack(handle, f"<{parameter_count}d")
            cameras.append((camera_id, model, width, height, params))
    return cameras


def read_images(path):
    images = []
    with path.open("rb") as handle:
        (count,) = unpack(handle, "<Q")
        for _ in range(count):
            image_id, = unpack(handle, "<I")
            quaternion = unpack(handle, "<4d")
            translation = unpack(handle, "<3d")
            camera_id, = unpack(handle, "<I")
            name = bytearray()
            while (byte := read_exact(handle, 1)) != b"\0":
                name.extend(byte)
            point_count, = unpack(handle, "<Q")
            # x/y/dense-point-id is 24 bytes for each observation.
            handle.seek(point_count * 24, 1)
            images.append((image_id, quaternion, translation, camera_id, name.decode("utf-8")))
    return images


def write_text_model(input_dir, output_dir):
    cameras = read_cameras(input_dir / "cameras.bin")
    images = read_images(input_dir / "images.bin")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cameras.txt").write_text("\n".join(
        f"{identifier} {model} {width} {height} " + " ".join(format(value, ".17g") for value in params)
        for identifier, model, width, height, params in cameras
    ) + "\n")
    image_lines = []
    for identifier, quaternion, translation, camera_id, name in images:
        fields = (identifier, *quaternion, *translation, camera_id, name)
        image_lines.extend((" ".join(map(str, fields)), "0 0 -1"))
    (output_dir / "images.txt").write_text("\n".join(image_lines) + "\n")
    return len(cameras), len(images)


def main():
    parser = argparse.ArgumentParser(description="Convert required COLMAP binary pose metadata to text.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cameras, images = write_text_model(args.input, args.output)
    print(f"converted {cameras} cameras and {images} registered images")


if __name__ == "__main__":
    main()
