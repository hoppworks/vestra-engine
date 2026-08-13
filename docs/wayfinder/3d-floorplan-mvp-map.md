# Map: 3D Floorplan MVP from a Smartphone Walkthrough

Label: `wayfinder:map`

## Destination

Implement a strong local MVP that converts a normal smartphone walkthrough video plus one scale anchor into a metrically scaled 3D floorplan (GLB) and editable 2D floorplan (SVG), using this repository's depth engine where viable.

## Notes

Domain terms live in `../../CONTEXT.md`. This map intentionally carries execution as well as decisions: the user explicitly authorized full MVP implementation. Prefer robust, measurable geometry over UI polish. The connected Linear tracker requires reauthentication, so this repository-local Markdown tracker is the canonical artifact for this effort.

## Decisions so far

- Local-first delivery — no service, upload, or account system is required for the MVP.
- Capture contract — regular RGB smartphone walkthrough video plus one user-entered scale anchor.
- Output contract — export both GLB (3D floorplan) and SVG (2D floorplan).
- [Select the reconstruction stack](tickets/select-reconstruction-stack.md) — FFmpeg and COLMAP are replaceable local sidecars; DA3 contributes pose-conditioned depth/confidence; Rust owns validation, fusion, topology, and exports.
- [Set the capture and quality contract](tickets/set-capture-quality-contract.md) — only a fully connected, quality-gated walkthrough earns final exports; one anchor means scale-anchored, while an independent second anchor earns verified status.

## Not yet specified

- The most reliable practical reconstruction stack and exact dependency boundary.
- Quantitative quality gates that determine when a walkthrough is accepted, rejected, or needs recapture.
- The data layout and viewer UX after the CLI geometry pipeline is proven.

## Out of scope

- Cloud processing, collaboration, accounts, and hosted storage.
- Automatic furniture modeling, photorealistic texturing, and BIM-grade semantic annotation.

## Child tickets

- [Build the video-to-geometry vertical slice](tickets/build-video-to-geometry-slice.md) — task; claimed
- [Extract and export floorplans](tickets/extract-and-export-floorplans.md) — task; blocked by the video-to-geometry slice
- [Prove the MVP against real walkthroughs](tickets/prove-mvp-real-walkthroughs.md) — task; blocked by the exporter
