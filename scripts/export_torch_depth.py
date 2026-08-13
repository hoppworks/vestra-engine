#!/usr/bin/env python3
"""Export the pinned official DA3 PyTorch F32 depth output as PFM."""

import argparse
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch


def install_torchvision_transforms_stub() -> None:
    tv = types.ModuleType("torchvision")
    transforms = types.ModuleType("torchvision.transforms")

    class ToTensor:
        def __call__(self, image):
            array = np.array(image, copy=True)
            if array.ndim == 2:
                array = array[:, :, None]
            return torch.from_numpy(array).float().div(255).permute(2, 0, 1).contiguous()

    class Normalize:
        def __init__(self, mean, std):
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def __call__(self, tensor):
            return (tensor - self.mean) / self.std

    class CenterCrop:
        def __init__(self, size):
            self.size = size

        def __call__(self, tensor):
            height, width = tensor.shape[-2:]
            target_h, target_w = self.size
            top = max(0, (height - target_h) // 2)
            left = max(0, (width - target_w) // 2)
            return tensor[..., top : top + target_h, left : left + target_w]

    transforms.ToTensor = ToTensor
    transforms.Normalize = Normalize
    transforms.CenterCrop = CenterCrop
    tv.transforms = transforms
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = transforms


def write_pfm(path: Path, depth: np.ndarray) -> None:
    depth = np.asarray(depth, dtype="<f4")
    with path.open("wb") as handle:
        handle.write(f"Pf\n{depth.shape[1]} {depth.shape[0]}\n-1.0\n".encode())
        np.flipud(depth).tofile(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/tmp/da3-src"))
    parser.add_argument("--model", type=Path, default=Path("/benchroot/models/DA3-BASE"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    install_torchvision_transforms_stub()
    sys.path.insert(0, "/benchroot/scripts")
    sys.path.insert(0, str(args.source / "src"))
    from da3_reference import load_model
    from depth_anything_3.utils.io.input_processor import InputProcessor

    device = torch.device(args.device)
    _, net = load_model(str(args.model))
    net = net.to(device)
    tensor, _, _ = InputProcessor()(
        [str(args.image)], process_res=504, process_res_method="upper_bound_resize"
    )
    tensor = tensor.reshape(-1, 3, tensor.shape[-2], tensor.shape[-1])[0][None, None].to(device)
    _, _, _, height, width = tensor.shape
    with torch.inference_mode():
        features, _ = net.backbone.pretrained.get_intermediate_layers(
            tensor, n=[5, 7, 9, 11], export_feat_layers=[], ref_view_strategy="saddle_balanced"
        )
        output = net.head(list(features), height, width, patch_start_idx=0)["depth"]
    depth = output.detach().float().cpu().numpy().squeeze()
    write_pfm(args.output, depth)
    print(f"wrote {args.output} shape={depth.shape[0]}x{depth.shape[1]}")


if __name__ == "__main__":
    main()
