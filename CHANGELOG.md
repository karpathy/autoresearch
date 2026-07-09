# Changelog

All notable changes to ResearchFS are documented here. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-07-09

### Added

- A lightweight `researchfs-core` distribution with SQLite-backed ingestion,
  retrieval, provenance, experiment, and agent-session APIs.
- A durable numbered migration ledger plus schema status, check, and upgrade
  commands.
- Multi-version core CI, package-boundary verification, tag-driven GitHub
  releases, an MIT license, and a private vulnerability-reporting policy.

### Changed

- NanoChat, Torch, CUDA kernels, Modal integration, benchmark images, and
  training code now live in the independently managed `benchmarks/nanochat/`
  fixture and are excluded from the core environment and distributions.
- The public repository identity and default branch now present ResearchFS as
  the product instead of the upstream autoresearch fixture.

[Unreleased]: https://github.com/mattmccarthy11/researchfs/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/mattmccarthy11/researchfs/releases/tag/v0.2.0
