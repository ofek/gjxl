# Modular entropy profiling

The next implementation target is reusable exact value populations per context.
On the six 12 MP photographic prefix cases, initial histogram construction,
clustered raw-value collection and raw-value aggregation account for
**99.86–99.97% of model time** across the two detailed-profile cohorts.
Configuration search and clustering are small once aggregation is separated.
These are diagnostic phase measurements, not measured optimization gains.

The GPU-token route's additional prefix-model time is concentrated in the
initial histogram scan. Warmed token-backing replays reproduce only a small part
of the full-workflow difference. ANS also exhibits a process-sensitive raw-value
collection penalty. Neither observation establishes a cache, allocator or
hardware cause, and this study does not change public dispatch.

## Capture and validation

Captured October 10, 2026 on Apple M4 Pro, 14 CPU cores, 48 GiB, macOS 27.0.1
(26A434). Frozen Release C++20, `-O3 -DNDEBUG`, Metal enabled, Metal stage
profiling and sanitizers disabled. Source is `faf0a7b` plus the eight compiled
overlay files identified by the retained source manifest. Benchmark SHA-256:
`b9652f7b1c4e4513e7f2d997349d475f0f7ffab8cbe7f5cfa68ba4a11624b1e0`.

The 14 cases reuse qualified inputs and expected files from the
[prescribed-token qualification](modular-gpu-token-performance.md): the three
4000×3000 RGB8 photos with gradient prefix, split7 prefix and gradient ANS;
1 MP/4K RGB8; 4K RGBA16; 1 MP RGB16 noise/ANS; and tiny RGB8. All calls have one
image, with the original one/four CPU participant settings. Modeling remains
explicitly serial. CPU, GPU preparation plus CPU tokens, and composed GPU tokens
use the same policy, input, domain capacity and CPU entropy implementation.

- One ordinary process: six measured triplets plus two warmups per case.
- Two detailed-profile processes: six measured triplets plus two warmups each.
- Two isolated model-replay processes: six measured triplets plus two warmups
  each, with interleaved host, split host and split Metal token backings.
- All six route orders are balanced per case/process; case order is shuffled.
- **1,008 complete encodes** match previously qualified bytes, including warmups.
  This is byte identity to independently decoded artifacts, not a fresh decode.
- **672 timed/warmup model replays** match exact reference `EntropyCode` objects;
  28 additional untimed CPU models establish those references. Every replay
  fixture first checks token values, contexts and stream boundaries exactly.
- Five focused Release CTests pass: entropy, entropy storage planning, Modular
  token handoff, GPU workflow, and the new Modular profile test. The latter
  checks prefix/ANS byte parity, nested/disabled sinks, accumulation and timing
  containment. Python analysis validates the full matrix, route/resource
  counters, candidate-count equality, artifact hashes and ordinary zero counters.

The implementation wires the shared entropy counters into `ModularProfile` and
adds one nested prefix aggregation counter. Existing configuration-search timing
retains its inclusive meaning. This report subtracts aggregation from that
parent before presenting disjoint stages. Entropy counters remain nested inside
the model phase and are never added to complete-call wall stages.

## Prefix attribution

GPU-token route, four CPU participants, milliseconds. Each entry is a median of
12 instrumented calls from two processes. Component medians need not sum to the
model median. The percentage above is a range of per-cohort medians of per-call
fractions, not a sum of independently computed medians.

| Photo / tree | Model | Initial histogram | Collect values | Aggregate values | Config search, excluding aggregation |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1029 / gradient | 150.46 | 81.03 | 24.12 | 44.74 | 0.03 |
| 1029 / split7 | 243.11 | 82.27 | 118.64 | 42.27 | 0.07 |
| 1039 / gradient | 157.39 | 81.15 | 24.80 | 51.48 | 0.04 |
| 1039 / split7 | 261.34 | 84.14 | 126.72 | 49.60 | 0.08 |
| 1049 / gradient | 155.39 | 81.14 | 24.28 | 50.13 | 0.04 |
| 1049 / split7 | 255.76 | 83.89 | 122.89 | 48.51 | 0.09 |

The 4K RGB8 control has 93.08 ms of model time, including 55.83 ms initial
histograms, 16.71 ms collection and 20.09 ms aggregation. The 4K RGBA16 control
has 140.36 ms modeling, including 74.76/22.50/42.80 ms for those same stages.
The tiny case is dominated by fixed work and does not support the same ranking.

Source explains these three passes:

1. `OptimizeEntropyCodeImpl` traverses every token to validate contexts, convert
   values with the default HybridUint configuration and build source histograms.
2. `BuildEntropyCodeForPartition` allocates a raw-value arena and scatters every
   token into the selected context partition's clusters.
3. `AggregateEntropyValues` counts those raw values, using a bounded dense prefix
   plus sparse counts for larger values. Only then does configuration search
   operate on weighted distinct values.

Per-context raw-value populations can supply the initial HybridUint histograms
and be merged according to the selected partition, addressing all three passes.
A default-configuration symbol histogram alone addresses the first pass.
Clustering, configuration choices and exact serialized-cost comparisons should
remain unchanged when introducing that consumer boundary.

## Token-backing comparison

Signed medians of paired differences, milliseconds. Positive means the second
route/backing took longer. Full-workflow columns compare GPU tokens against CPU
tokens; replay columns use the same precomputed values and contexts.

| Photo / tree | Full model delta | Initial histogram delta | Replay host split minus interleaved | Replay Metal split minus host split |
| --- | ---: | ---: | ---: | ---: |
| 1029 / gradient | +9.90 | +9.92 | +0.98 | +1.09 |
| 1029 / split7 | +8.92 | +10.14 | +1.13 | -0.34 |
| 1039 / gradient | +10.56 | +9.61 | +0.22 | +0.85 |
| 1039 / split7 | +9.64 | +9.35 | +1.02 | +1.37 |
| 1049 / gradient | +9.03 | +9.51 | +1.64 | +1.42 |
| 1049 / split7 | +8.68 | +9.39 | +1.33 | +1.57 |

Stage deltas are separately computed medians and need not add to the model
delta. The initial histogram scan accounts for the observed full-workflow
prefix difference. The warmed replays show small layout/backing differences,
but do not reproduce most of that gap. They hold all three backings resident
and exclude preparation/copies, so they do not isolate the production handoff's
cache state, scheduling, allocation placement or first-touch behavior. These
numbers do not justify copying GPU tokens into interleaved storage.

## ANS attribution and variability

The three photo cases use the same gradient policy and ANS candidate counts in
both profile processes. Keep their cohorts separate: pooling produces a median
between two distinctly different raw-value-collection timings.

GPU-token route, milliseconds:

| Photo | Cohort | Model | ANS raw-value collection | Exact ANS token cost |
| --- | --- | ---: | ---: | ---: |
| 1029 | 137 | 1071.52 | 382.93 | 493.14 |
| 1029 | 239 | 738.34 | 41.33 | 499.13 |
| 1039 | 137 | 1059.73 | 384.10 | 470.00 |
| 1039 | 239 | 716.86 | 41.43 | 469.23 |
| 1049 | 137 | 1059.88 | 380.98 | 474.01 |
| 1049 | 239 | 720.36 | 41.08 | 473.81 |

CPU-route collection is about 41–42 ms in both cohorts. The extra GPU-route
penalty is localized to collection; exact token-cost time remains similar.
That collection loop appends into growing per-cluster raw-value vectors. The
timer includes allocation/growth/copying and traversal, and does not separate
their contributions. Warmed replays do not consistently reproduce the penalty.

In cohort 239, exact ANS token cost accounts for 65.4–67.5% of photo model time.
`MeasureAnsCodes`/`MeasureAnsCode` simulate ordered rANS states to compare exact
candidate costs. This work is distinct from histogram/model construction and
cannot be replaced by unordered histograms without changing the decision rule.

There is a useful existing CPU boundary to evaluate next:
`OptimizeEntropyCodeAndPrepareClusters` retains prefix value populations, and
`OptimizeAnsEntropyCodeWithPreparedClusters` consumes them. Modular currently
calls the unprepared variants and repeats value collection/aggregation. A
follow-up should account for the retained populations in admission and establish
exact model/file parity before making a performance claim.

## Ordinary binary control

A separate bounded control runs the previously qualified frozen binary and the
new instrumentable binary in baseline/new/new/baseline process order. Only photo
1029 gradient prefix and gradient ANS are included, with two warmups and six
balanced measured triplets per process. All **192 additional complete encodes**
match the same expected files. These calls have all profiling counters disabled.

Median complete-call GPU-token time, milliseconds:

| Binary / seed | Prefix | ANS |
| --- | ---: | ---: |
| Previous qualified / 311 | 261.42 | 828.46 |
| New instrumentable / 311 | 259.96 | 822.78 |
| New instrumentable / 419 | 260.81 | 828.41 |
| Previous qualified / 419 | 260.67 | 823.64 |

This two-case control does not show a persistent large regression from the new
binary. Its allocation/workload history differs from the 14-case captures; it
does not explain or invalidate the intermittent collection penalty. Small
differences here are not optimization gains or a portable overhead bound.

## Recommended next slice

1. Define bounded, exact per-context raw-value populations and a CPU reference
   consumer that preserves current partition/configuration decisions. Include
   arbitrary supported residual values and checked count/overflow handling.
2. Reuse those populations through prefix modeling and the ANS prepared-value
   boundary. Compare against an efficient CPU producer before judging GPU gains.
3. Produce the populations beside GPU tokens, with deterministic reductions,
   explicit scratch/retained storage and the existing single completion fence.
   Ordered tokens remain necessary for final emission and exact ANS costing.
4. Treat exact ANS cost simulation as a separate later target; per-stream CPU
   parallelism is a control worth evaluating before GPU state-machine work.

No model optimization, representation replacement, GPU dispatch threshold or
automatic-route promotion is implemented by this profiling change.

## Reproduction and retained evidence

The [profiling protocol](../tools/modular_oracle/ENTROPY_PROFILING.md) defines
timer boundaries, the matrix, warmed replay limitations and commands. The
[structured summary](modular-entropy-profile-results.json) retains all 14 cases.
Raw evidence is under `build/modular-entropy-profile-20261010/`:

- `source/`, `source-manifest.json`, `source.patch`, configure/build logs and
  `release/CMakeCache.txt` bind the captured binary to its source and flags.
- `cases.tsv`, `expected/`, `entropy-run-provenance.json`, ordinary/profile/replay
  CSVs and logs retain inputs, commands, output identity and raw measurements.
- `entropy-results.json` retains complete per-stage and per-cohort analysis;
  `entropy-host-activity.jsonl` retains active-desktop/thermal observations.
- `binary-controls/` and `run_binary_controls.py` retain the bounded ordinary
  baseline/new comparison and its commands, logs, hashes and host activity.
- `tests-final.log` and `analysis-validation.json` retain correctness checks;
  the analyzer rejects missing routes, nonzero ordinary counters, invalid nested
  timings, wrong submission counts and inconsistent ANS candidate counts.
- `artifacts-sha256.json` binds the retained evidence and source dependencies.

All 144 main/control host snapshots report no recorded thermal or performance
warning. The desktop was active: main-cohort nonbenchmark CPU medians range from
189.7% to 395.8%, with a 917.8% maximum (process CPU can exceed 100% across cores).
Compiler processes also appear during profile cohort 239 and the first baseline
control. Those observations are retained in `environment-summary.json`; they
preclude a quiet-machine claim and do not establish GPU idleness.

These are selected inputs on one active desktop in one session. They establish
where work occurs and expose variability; they do not replace the independent
quiet-session repeat required by the earlier dispatch qualification.
