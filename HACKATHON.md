# Prior Labs TabPFN-3.5 Hackathon checklist

Official challenge page: <https://platform.priorlabs.ai/hackathon-3.5>

The official terms require participants to build with TabPFN-3.5 and submit a runnable **public** source-code repository under **Apache License 2.0**. A demonstration video is optional. The announced submission deadline is **6 October 2026**.

## Repository readiness

- [x] Uses the standard TabPFN-3.5 regressor explicitly.
- [x] Contains a runnable benchmark entry point and fixed test protocol.
- [x] Contains setup, checkpoint, validation, and execution instructions.
- [x] Records dataset provenance and architecture citations.
- [x] Excludes TabPFN weights and explains their separate non-commercial license.
- [x] Discloses which project components predate the hackathon.
- [x] Uses the required Apache License 2.0 and includes citation metadata.
- [ ] Repository visibility is public (change this in GitHub before submission).
- [x] Excludes generated artifacts, local paths, caches, logs, and credentials.

## Actions that must be completed in the Prior Labs account

1. Sign in or create a Prior Labs account on the challenge page.
2. Read and accept the current Terms & Conditions shown in the portal.
3. Submit the public repository URL before the portal deadline.
4. Add the project webpage/demo URL when available.
5. Optionally attach a short video showing the benchmark, dataset explorer, and TabPFN-3.5 comparison.

Portal account acceptance and the final submission cannot be represented by repository files; verify their status while signed in.

## Suggested submission summary

> NIRBENCH-DL tests whether a general tabular foundation model can compete with specialist neural networks and optimized chemometrics on high-dimensional near-infrared spectra. The harness evaluates TabPFN-3.5, 13 spectroscopy deep learning architectures, and preprocessing-aware PLS on 30 fixed regression tasks. It uses training-only model selection, ten seeded final evaluations, shared held-out tests, complete dataset provenance, and reproducible outputs that feed an interactive results website.
