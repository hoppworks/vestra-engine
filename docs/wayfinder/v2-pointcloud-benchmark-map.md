# Map: Local V2 Point-Cloud Demo and Reproducible Benchmark

Label: `wayfinder:map`

## Destination

Specify the next product increment: a local browser application that turns a
selected video into an inspectable point cloud, plus a reproducible Depth
Anything V2 benchmark suite on the fixed Ryzen 9 + RTX 5080 machine.

## Notes

The UI is local-first: selecting a video never uploads it. The active product
does not create a floorplan. Existing DA3/floorplan code remains preserved as a
retired future path and must not be invoked by the new UI or benchmark.

Use `V2-Small` as the commercial-safe, exact-weight runtime benchmark. Report
`V2-Base` separately as a non-commercial research class. CPU and CUDA results
are distinct tables, never mixed. Domain terms live in `../../CONTEXT.md`.

This repository-local Markdown tracker is canonical because the connected
issue tracker is not available.

## Decisions so far

- Local delivery — the browser UI talks only to a local process; videos stay on
  the user's machine.
- Product output — the active use case ends in an interactive, polished point
  cloud viewer with downloadable artifacts, not a floorplan.
- Benchmark hardware — the canonical machine is Ryzen 9 + RTX 5080; CPU and
  CUDA are reported independently.
- Model boundary — V2-Small is the exact-weight speed-comparison target;
  V2-Base is a separately labelled research target.
- [Separate model quality from runtime speed](tickets/separate-model-quality-from-runtime-speed.md) — DA-2K compares different depth models; only same-weight V2-Small runtimes may support a speedup claim.

## Not yet specified

- The user's visual direction for the point-cloud viewer and its progress
  animation.
- The least-complex local browser/runtime boundary that handles large videos
  without copying them unnecessarily.
- The numerical parity thresholds for each V2 runtime conversion.
- Which CUDA runtimes are installed and support RTX 5080 on the target host.

## Out of scope

- 3D floorplan extraction, SVG plans, room topology, scale anchors, and door
  detection. They remain in the repository as a retired future path.
- Hosted processing, accounts, collaboration, or cloud video storage.
- A single synthetic “overall winner” score across models with different
  architectures and accuracy/quality settings.

## Child tickets

- [Retire the floorplan from the active product surface](tickets/retire-floorplan-product-surface.md) — task; unblocked.
- [Specify the local point-cloud viewer experience](tickets/specify-local-pointcloud-viewer.md) — grilling/prototype; unblocked after the user supplies the visual direction.
- [Lock the V2-Small runtime benchmark contract](tickets/lock-v2-runtime-benchmark-contract.md) — research/task; unblocked.
- [Provision and fingerprint the Ryzen 9 + RTX 5080 benchmark host](tickets/provision-ryzen-rtx-benchmark-host.md) — task; blocked by the runtime contract.
- [Specify the cross-model DA-2K comparison](tickets/specify-da2k-model-comparison.md) — research/task; blocked by the runtime contract.
