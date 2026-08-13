# Local Point-Cloud Runtime Modes

## Decision

The viewer must never silently run a full multi-view reconstruction on a development laptop and leave the user with an unexplained multi-minute wait.

| Mode | Purpose | Local behaviour | Claim allowed |
|---|---|---|---|
| Quick preview · this Mac | Confirm that a video can create a small, inspectable point cloud. | DA3-Base CPU, 0.15 fps, 144 px processing resolution, maximum 75,000 points. | Visual preview only; no performance or final-quality claim. |
| Full quality · RTX workhorse | Produce the high-quality artifact and canonical benchmark evidence. | Stores the video locally and waits for explicit SSH hand-off to the Ryzen 9 + RTX 5080 host. No laptop fallback starts. | Full-quality artifact and, once the benchmark contract is met, benchmark evidence. |

## Why

The first 42-frame DA3-Base CPU job used about 2.6 GB RSS and four CPU cores for more than five minutes without reaching export. It was working, but it was not an acceptable interactive product path. The RTX workhorse is therefore the authority for full reconstruction.

## Operator contract

Before an RTX benchmark or full-quality run, the operator confirms that other GPU-intensive workloads have been stopped. The resulting artifact records host fingerprint, checkpoint hash, CUDA/runtime versions, source-video hash, mode, and raw timings.
