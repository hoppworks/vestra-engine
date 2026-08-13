# Ubiquitous Language

- **Point-cloud session**: One local video processed into per-frame depth, camera poses, and a renderable 3D point cloud.
- **Frame manifest**: The immutable list of decoded frames and their source metadata used by every benchmark runtime.
- **Runtime benchmark**: A speed comparison of implementations that execute the identical, pinned V2-Small model and preprocessing contract.
- **Model benchmark**: A quality comparison of different depth-model architectures on DA-2K; it is not a runtime speed claim.
- **Retired floorplan path**: The preserved DA3/COLMAP/floorplan code. It is not part of the active product, CLI, or benchmark contract.
