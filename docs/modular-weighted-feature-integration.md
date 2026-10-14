# Weighted-feature integration qualification

`feat/modular` passed integration qualification on 2026-10-13 at
`cfbd7ada020ae4d96b81f143bad4f15779a10bab`. Fresh CPU-only and Metal builds passed
the selected Modular/entropy checks, and all three full-resolution hybrid
encodes preserved complete bytes, candidate decisions and decoded samples.

## Integrated source

The branch fast-forwarded from `cbc2d8e` to
`cfbd7ad`, bringing in 38 existing commits without
conflicts or source edits. This includes the preceding adaptive GPU, entropy,
training-budget and shared-feature work needed by weighted preparation.
The final preparation commits are:

- `fdeee05`: scalar operation, storage contract and attribution.
- `3847b52`: bounded CPU preparation and exact Metal wavefront preparation.
- `cfbd7ad`: feature correctness/performance qualification and evidence.

The integrated source tree is `0fb01ff7f3d34e29ee5dc4fe56bfec98cfa1d88d`.
The separate `research/modular-learning-cost` branch is not part of this
integration. The original learner and its policies remain the comparison basis.
Concurrent population-fusion edits appeared in the worktree after these builds.
They were left untouched and are outside this qualification of the recorded
integrated revision.

## Fresh correctness checks

Independent Ninja builds used Release `-O3 -g -DNDEBUG`, tests and benchmarks
enabled, CUDA disabled, and Metal respectively disabled/enabled. The pinned
libjxl decoding oracle is `e8ff09762481785938d8e4e01333ed3917571161`.

| Build | Passed checks |
| --- | ---: |
| CPU-only | 38 |
| Metal | 52 |

The fresh Metal weighted operation compared 2,831,887 features and 3,778,462
tokens across 203 cases, with 34 identical complete files and 6,198,994
independently decoded samples. The weighted workflow checked 55 complete
CPU/hybrid matches, 26,119,113 decoded samples and 7,711 failure-atomic cases.
The CPU wavefront proof matched 4,187,195 scalar/full/rolling features and 67
unsupported-range results across 312 cases.

Coverage includes entropy models/costs, training budgets/checkpoints/coverage,
weighted scalar/batch/wavefront parity, storage admission and failure behavior,
GPU features/tokens/populations, adaptive workflows, conformance, prescribed
policies/transforms, public API/CLI and frozen references. These are selected
Modular and entropy tests, not the full repository suite.

The general `modular_workflow` exhaustive allocation sweep was selected too
broadly for this integration. It repeats complete candidate search at every
allocation-failure position. After an initial unsharded run and an attempt to
parallelize it into eight shards per build, the remaining processes were
intentionally stopped because of the excessive runtime. Its original CTest
logs record one terminated test per build; partial shard logs are also retained.
This general sweep is **not qualified or counted as a pass**. The dedicated
weighted-workflow failure checks above completed successfully. No unexpected
test failure or skipped test was observed.

The two builds and correctness suites ran concurrently. Their elapsed times
are not performance measurements. No additional sanitizer or Metal shader
validation run was made here; the
[feature report](parallel-weighted-feature-preparation.md#final-qualification)
retains the earlier instrumented-pipeline eligibility limitation.

## Full-resolution regression check

After all test processes had stopped, one fresh encode ran for each original
4000×3000 RGB8 photograph, serially, using the newly built Metal profiler.
The options match the saved qualification: 262,144 training samples, 256 leaves,
four CPU participants, shared features, GPU learning statistics/populations,
estimated ANS selection and measure-all evaluation. No feature-selection flag
was supplied, so this exercises the default Auto route.

| Image | Encoded bytes | Fresh complete call (battery) | Fresh feature preparation | Previous Auto/4 median (AC) |
| --- | ---: | ---: | ---: | ---: |
| 1029 | 10,821,744 | 48.940 s | 1.177 s | 34.570 s |
| 1039 | 7,230,909 | 46.301 s | 1.210 s | 32.935 s |
| 1049 | 7,336,102 | 46.208 s | 1.181 s | 33.218 s |

Each complete file has the same SHA-256 as the frozen pre-integration output
and independently decodes to the source samples with pinned libjxl. Every
candidate-ledger field except elapsed time, the selected policy and training
work counters match. Each call dispatches all 25 feature preparations to Metal,
covering 900 million features with no CPU feature fallback. Memory stays within
admission, CPU participation stays within four, and detailed preparation timers
remain nested inside their parent interval.

The complete-call timer excludes backend setup, input/output file access and
independent decoding; it includes candidate-ledger serialization. Feature
preparation is a subset of that time. These are single-run integration checks
on an active Apple M4 Pro desktop (14 CPU cores, 48 GiB). They ran on battery,
with background indexing visible in the process snapshots, while the earlier
qualification used AC power. No thermal warning was reported. The fresh elapsed
times are higher, but these different conditions do not establish a performance
regression or a speedup; this is a correctness qualification, not a new matched
performance experiment. The earlier repeated
[CPU/Metal comparison](parallel-weighted-feature-preparation.md#complete-encoder-calls)
remains the performance qualification: about 3.8× faster feature preparation
and 7.8–8.4% less complete-call time versus four CPU preparation participants
on these images. This integration does not establish a new libjxl comparison.

## Evidence and reproduction

The [committed results](modular-weighted-feature-integration-results.json)
record revisions, individual test outcomes, hashes, exact output sizes, timings
and storage/routing counters. Local raw evidence is retained under
`build/modular-integration-20261013/`: the root manifest, separate CMake caches,
build logs and command records, CTest JUnit/logs, partial workflow shard logs
and the explicit stop record, and full encode outputs with power/thermal/process
snapshots. The manifests also identify and hash the frozen comparison data.

The accepted CTest scope is
`-R '^(modular_|ans_|entropy_|metal_modular_|metal_ans_token_cost$)'`
with `-E '^modular_workflow$'`. The retained `run_workflow_shards.py` documents
the broader attempted sweep, which remains incomplete.

```sh
build/modular-integration-20261013/metal/gjxl_modular_search_profile \
  IMAGE.pam OUTPUT.jxl 262144 256 4 \
  --metal-learn --estimated-ans --measure-all --metal-populations \
  --candidate-ledger --shared-features
```
