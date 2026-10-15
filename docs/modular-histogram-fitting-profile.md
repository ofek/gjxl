# Native histogram fitting attribution

2026-10-15. **Repeated histogram construction and property scans dominate the
current fitter.** Joint histograms plus property occupancy take **55–67%** of
fitting time across the three photographs. Candidate evaluation takes
**17–31%**, and sample partitioning **8–10%**. Workspace allocation/setup is
below 0.1%. Fitting remains **69–72% of complete hybrid encoder time**.

The subsequent [fitter work reduction](modular-fitter-work-reduction.md) combines
occupancy and histogram passes and reuses constant-property information across
descendants. The serial measurements below describe the earlier implementation.

This profiles the serial CPU fitter inside the qualified automatic hybrid
pipeline, with Metal feature/token work retained. Property and residual columns
are already prepared when the fitter begins; their preparation is outside the
exclusive table. The three source images are the original **4000×3000 RGB8**
1029/1039/1049 PAMs, 12 MP each.

The retained timing dataset contains **24 complete encodes**: one warmup and
three measured repetitions per image/mode, all observed on **AC power** at
both boundaries of every call. The initial and replacement captures contain
40 complete, exact encodes in total; superseded observations remain in the
[raw results](modular-histogram-fitting-profile-results.json). The power change
and replacement procedure are documented below. The earlier 52/38/35-second
measurements were on battery; this run establishes a fresh AC attribution
baseline.

Apple M4 Pro, 14 CPU cores (10 performance, 4 efficiency), 48 GiB, macOS 27.0.1,
Apple Clang 21, Release `-O3 -g -DNDEBUG`, Metal enabled. Pinned libjxl
`e8ff09762481785938d8e4e01333ed3917571161` performs exact decoding after the
encoder timer. No libjxl performance comparison was rerun.

Auto hybrid, auto/local buffering, Best predictors, four CPU participants ceiling. Identical binary with histogram attribution disabled/enabled. Fresh serial processes, rotating on/off order, one warmup per mode/input. Native encoder-call wall and process CPU exclude IO/backend creation/independent decoding. Fitting phases are exclusive wall intervals nested within the native fitter envelope and workflow fitting time. Node size means distinct weighted rows. Candidate evaluation includes prefix histogram updates/clears, entropy estimates and per-side predictor comparison. No per-threshold clock calls. Uninstrumented control/cleanup and instrumentation overhead remain in the envelope residual. Caller-owned diagnostic memory is outside codec admission.

## Instrumentation overhead

Same frozen binary and hybrid policy, with fitting attribution disabled/enabled. Values are medians of complete calls; repetitions and ranges remain in the JSON.

| Image | Power | Encoder off / on (s) | Workflow fitting off / on (s) | Fitting change |
| --- | --- | ---: | ---: | ---: |
| 1029 | AC Power | 35.770 / 36.530 | 25.739 / 26.411 | +2.61% |
| 1039 | AC Power | 26.191 / 26.186 | 18.029 / 18.126 | +0.54% |
| 1049 | AC Power | 23.745 / 23.836 | 16.304 / 16.475 | +1.05% |

## Exclusive fitting phases

Each column uses the individual profiled call with median fitter time. The rows add to its fitter envelope; they are not independently computed stage medians. These are new instrumented measurements, not an exact subdivision of the earlier 52/38/35 s calls.

| Phase | 1029 | 1039 | 1049 |
| --- | ---: | ---: | ---: |
| Input validation | 0.008 s (0.0%) | 0.006 s (0.0%) | 0.006 s (0.0%) |
| Workspace/result allocation and setup | 0.003 s (0.0%) | 0.003 s (0.0%) | 0.004 s (0.0%) |
| Find residual symbol range | 0.075 s (0.3%) | 0.059 s (0.3%) | 0.056 s (0.3%) |
| Parent histograms + base cost | 0.655 s (2.5%) | 0.586 s (3.2%) | 0.555 s (3.4%) |
| Property occupancy + cost-array initialization | 4.718 s (17.9%) | 3.998 s (22.1%) | 3.644 s (22.1%) |
| Joint property/residual histograms | 9.728 s (36.8%) | 8.200 s (45.2%) | 7.346 s (44.6%) |
| Prefix updates + entropy/predictor evaluation | 8.302 s (31.4%) | 3.368 s (18.6%) | 2.868 s (17.4%) |
| Choose threshold within each property | 0.388 s (1.5%) | 0.171 s (0.9%) | 0.163 s (1.0%) |
| Partition sample rows | 2.015 s (7.6%) | 1.474 s (8.1%) | 1.601 s (9.7%) |
| Validate/publish tree | 0.005 s (0.0%) | 0.003 s (0.0%) | 0.002 s (0.0%) |
| Other control, cleanup, probe overhead | 0.513 s (1.9%) | 0.256 s (1.4%) | 0.227 s (1.4%) |
| **Fitter envelope** | **26.409 s** | **18.124 s** | **16.473 s** |

## Node-size distribution

Rows are distinct weighted training rows, not image pixels. Time is inclusive node processing time as a share of each image's fitter envelope. Node counts combine the selected images. Each node is assigned once; ancestor and descendant row visits repeat work on the same samples.

| Rows per node | Nodes | 1029 | 1039 | 1049 |
| --- | ---: | ---: | ---: | ---: |
| 1–64 | 104,822 | 6.84% | 2.60% | 1.33% |
| 65–256 | 149,316 | 13.66% | 7.28% | 5.96% |
| 257–1,024 | 121,971 | 14.97% | 11.83% | 12.98% |
| 1,025–4,096 | 64,870 | 15.05% | 15.57% | 18.51% |
| 4,097–16,384 | 17,467 | 11.82% | 15.29% | 18.36% |
| 16,385–65,536 | 6,565 | 21.90% | 29.31% | 27.46% |
| >65,536 | 1,497 | 15.67% | 18.03% | 15.30% |

## Workload

| Image | Fits | Visited nodes | Joint-histogram row visits | Entropy evaluations | Max depth | Mean occupied bins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1029 | 192 | 240,142 | 8,776,216,832 | 474,605,692 | 29 | 31.96 |
| 1039 | 192 | 120,546 | 7,091,026,976 | 240,301,144 | 21 | 32.33 |
| 1049 | 192 | 105,820 | 6,704,891,872 | 245,722,952 | 19 | 37.48 |

The JSON also retains exact depth counts/times, padded residual-symbol spans, occupied-bin/span distributions, row swaps, and individual fit times/shapes. Symbol span is the padded maximum token plus one used by the cost loop, not nonzero-symbol count.

## Implications for the next implementation step

There are **192 independent local fits per image**, each evaluating Weighted
and Gradient predictors against 16 property columns. The per-fit measurements
make these substantial CPU work units:

| Image | Median / maximum individual fit | Largest observed padded symbol span |
| --- | ---: | ---: |
| 1029 | 139.1 / 202.6 ms | 56 |
| 1039 | 101.4 / 159.6 ms | 72 |
| 1049 | 103.1 / 125.2 ms | 68 |

**Parallel local CPU fits remain the next practical step.** They can reuse the
existing numerical algorithm and deterministic tree decisions. Scheduling must
respect existing admission and feature-batch lifetimes, bound concurrent sample
storage, and publish results in canonical stream order. This study supplies
work sizes and serial durations; it does not measure multicore scaling.

For GPU work, jointly target histogram construction, property occupancy and
candidate evaluation. Sending only the entropy arithmetic to the GPU would
leave most fitting work on the CPU. Nodes with at most 1,024 distinct rows
account for **20–35%** of fitting time; all nodes larger than 65,536 rows account
for only **15–18%**. Batch ready nodes across trees and measure a CPU cutoff for
small tasks. A root-only implementation would cover little of the measured
work. The JSON preserves depth and bin/symbol distributions for kernel sizing.

The counters explain the cost: each image makes **6.7–8.8 billion** joint-
histogram row visits and **240–475 million** candidate entropy evaluations.
Rows are revisited for each property, predictor and deeper node. The fit is
performing extensive repeated integer aggregation even though prediction
features have already been cached. These timings do not establish a hardware
memory-bandwidth bottleneck; that would need separate hardware-counter evidence.

Partitioning is a meaningful secondary target at **8–10%**. Workspace setup and
tree publication have little headroom. The current result supports the planned
order: parallel local fits, then batched GPU histogram/scoring, then reconsider
partitioning and histogram reuse using complete-call measurements and exact
codestream parity.

## Measurement boundaries and reproduction

Instrumentation is opt-in and allocates no fitting storage. Disabled hot loops are a separate compile-time instantiation. Timers surround node/property/predictor blocks, never individual thresholds. All phase times are wall time; the fit stays serial. Candidate evaluation deliberately combines prefix updates/clears, entropy estimation, and per-side predictor comparison. It does not isolate the logarithm polynomial. The remainder includes final category preference, tree bookkeeping, destruction, and unassigned probe overhead. The workflow's fitting timer additionally includes storage planning and prepared-column release/compaction outside LearnHistogramTree.

All **40 encodes** decoded exactly and preserved complete output bytes, routing, and learning work across on/off modes and repetitions. Enabled runs also preserved all workload counters. Timing uses 24 calls including warmups; 16 superseded calls remain in the JSON and raw archives. Each retained image cohort has a single observed power source at every call boundary. Reference-byte verification: matched the earlier qualified hybrid files.

Build and input identities, every run, before/after power/load observations, and dependencies are retained. Source was archived and hashed including uncommitted changes; the recorded base revision alone does not identify the instrumented source. Desktop activity was not isolated. This attribution is not a CPU/GPU speedup claim.

Artifacts: `/Users/yunhocho/GitHub/gjxl-histogram-fitting-attribution/build/histogram-fitting-attribution-20261015`. Manifest SHA-256: `245db818b11c054e484168110b98338281c24138e9ad9a86127c94abb9884344`.

Regenerate the measured tables and primary JSON with:

```sh
python3 tools/modular_reference_learning/analyze_histogram_attribution.py \
  /Users/yunhocho/GitHub/gjxl-histogram-fitting-attribution/build/histogram-fitting-attribution-20261015 \
  --replacement /Users/yunhocho/GitHub/gjxl-histogram-fitting-attribution/build/histogram-fitting-attribution-20261015-ac-replacement \
  --json docs/modular-histogram-fitting-profile-results.json \
  --markdown docs/modular-histogram-fitting-profile.md
```

## Sampling diagnostic: what is inside candidate evaluation?

A separate full-resolution 1029 encode was captured with Samply at 1 kHz,
main thread only, using the same frozen binary with fitting attribution off.
This run decoded exactly and matched the historical qualified output hash.
It is excluded from the paired timing medians.

Resolving inline frames against the matching Mach-O binary and dSYM attributes
**23.95% of sampled fitting CPU** to `EstimateBits` / `FastLog2`, including
histogram summation and entropy arithmetic. Assigning CPU deltas to the
preceding rather than current sample gives **23.66%**. There were 25,297
positive fitting samples, representing 24.924 seconds of sampled CPU;
99.97% of fitting leaf CPU was symbolized, with source/inline information for
97.97% of native leaf CPU. These are sampled CPU estimates, not additional
wall-clock phases to add to the table above.

These are the learner's approximate split-cost estimates, distinct from final
entropy encoding and the previously studied exact ANS token-cost simulation.

The two `CollectExtraBitsIncrease` variants account for **35.36%** of
disjoint leaf CPU samples. This is the joint-histogram accumulation loop,
corroborating its large wall-time share. Accessor/helper samples can fall
outside this function bucket, so it is not an exact phase measurement.

The disjoint inline leaf attribution is 15.15% in `EstimateBits` and 8.80% in
`FastLog2`. Candidate evaluation also updates and clears prefix histograms and
chooses predictors, so its full phase must not be described as logarithm cost.
The entropy estimator already uses four-lane ARM NEON and an explicit FMA
polynomial on this build. This cost is not evidence of a missing SIMD path.

[Sampling analysis and capture provenance](modular-histogram-fitting-samply-results.json)
retain both CPU-delta assignments, symbols, top source lines, tool versions,
commands and file hashes. The raw profile, full inline resolution, frozen
binary and matching dSYM remain in
`build/histogram-samply-20261015`. View the capture with:

```sh
samply load build/histogram-samply-20261015/1029.json.gz
```

The capture was observed on AC power. It did not have the per-call power/load
snapshot pair used by the primary study; its provenance labels the later
snapshot separately.

## Power-condition handling

The first capture began on AC, switched to battery during 1039, and
finished 1049 on battery. The entire original 1039 and 1049 cohorts are
excluded from the reported timing medians. After AC was restored, both
images were repeated in full with the same frozen binary, one warmup and
three measured calls per mode. The retained 1029 cohort and both replacement
cohorts are all AC at every recorded call boundary. Every original call
remains in the results, marked with its capture and whether timing used it.

A battery retry was stopped when AC was requested. Its two completed
warmups and interrupted next call are retained separately in
`build/histogram-fitting-attribution-20261015-1039-battery`; this incomplete
capture supplies no reported timings or passing qualification evidence.
The analyzer now rejects mixed-power cohorts and supports complete-cohort
replacements, checking binary/input identities, bytes and work counters
across captures. Its rejection was exercised against the original record.

## Validation and implementation

The private profile records exclusive phase wall time, inclusive per-node
processing time, node size/depth, property-bin and residual-symbol shapes,
work counters and individual fit envelopes. Node time includes that node's
phases, not its descendants' execution time. The current iterative fitter
processes each node once. Repeated row visits arise because the same training
rows participate at several depths and in several property/predictor passes.
Entropy-evaluation and symbol-term counters count candidate-side estimates;
base-parent estimates belong to the parent-histogram phase instead.

The instrumented and disabled hot loops are separate compile-time
instantiations. Clock calls surround node/property/predictor blocks and are
absent from the per-threshold loop. Sessions are thread-local and nestable;
the current fitter is serial. A future parallel implementation must explicitly
install a separate sink per worker and merge counters, rather than expecting
the caller's thread-local session to propagate. Caller-owned diagnostic
storage is allocated before encoder timing and excluded from codec admission.

The histogram-learning, histogram-preparation, shared-feature-consumer and
CPU/Metal reference-hybrid CTests passed (4/4). New checks compare exact trees
and partition order with profiling on/off, weighted samples, predictor ties,
heterogeneous symbols/bins, forced splits, failed fits and record exhaustion.
They also check hand-derived work counts and time conservation. A targeted
AddressSanitizer/UndefinedBehaviorSanitizer build of the fitter and its test
passed with leak checking disabled; linked archive code and GPU shaders were
not sanitizer-instrumented. The paired runner and both analyzers passed Python
syntax checks, and the paired runner/analyzer completed a two-call 1024-square
smoke study before the full-image study.

Implementation and commands are documented in the
[profiling tool README](../tools/modular_reference_learning/README.md#histogram-fitting-attribution).
Changes are isolated on `perf/histogram-fitting-attribution`, based on
`e5378fbb04e671e25f24512b9fb0fc83732f427d`. The final capture's code/tool
snapshot is committed as `3076658f9a7156f77e4dd012a3c5c1463f23b956`. The archived
source hashes identify each measured snapshot. This step
adds diagnostics and analysis only; it does not change fitting policy or
parallelize the fitter.
