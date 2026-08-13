# Build the video-to-geometry vertical slice

Label: `wayfinder:task`

Blocked by: [Select the reconstruction stack](select-reconstruction-stack.md), [Set the capture and quality contract](set-capture-quality-contract.md)

Assignee: Codex

## Question

Implement the smallest end-to-end local command that turns an accepted walkthrough into a scaled, inspectable structural point cloud or mesh, preserving intermediate diagnostics.

## Resolution

In progress.

Completed in the owned local pipeline:

- `da-video`: deterministic, shell-free FFprobe/FFmpeg/COLMAP contracts;
  normalized frame/camera manifest; protected run workspaces.
- `da scan`: 1080p/24fps and scale-anchor validation, frame extraction, and
  COLMAP sequential matching with loop detection.
- Aspect-preserving, calibration-aware letterbox preprocessing in `da-engine`.
- `da-quality` and `da finish`: final GLB/SVG artifacts are blocked when
  reconstruction quality demands recapture.

Still required before this ticket closes: the official pose-conditioned DA3
sidecar must convert the COLMAP-normalized keyframes into dense depth and
confidence, and the owned fusion stage must derive the structural outer ring
and quality metrics automatically rather than receiving them as durable inputs.
