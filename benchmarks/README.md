# Optional benchmarks

Benchmark projects are intentionally outside the `researchfs-core` package and
root dependency environment. Each benchmark owns its runtime, assets, fixture
data, and tests so product installs do not pull in training frameworks or
provider orchestration SDKs.

- [`nanochat/`](nanochat/) — NanoChat/autoresearch training and comparison
  fixture, with Torch in the benchmark environment and Modal/analysis tools as
  optional benchmark extras.

The directories here are structured so they can be extracted into independent
repositories without changing ResearchFS core APIs.
