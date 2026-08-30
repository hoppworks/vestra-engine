# Security policy

## Supported versions

Security fixes target the latest release and the `main` branch.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for the Vestra Engine
repository. Include the affected revision, reproduction steps, impact, and any
suggested mitigation. Do not open a public issue before a fix is available.

## Security boundaries

The project treats model files, images, GGUF metadata, and benchmark fixtures
as untrusted inputs. The sensitive boundaries are binary parsing, tensor shape
validation, CPU and CUDA dispatch, external benchmark commands, and release
provenance. A successful inference result does not establish that an input or
model is safe to redistribute.
