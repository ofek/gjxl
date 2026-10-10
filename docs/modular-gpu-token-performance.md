# Modular prediction/token performance qualification

**Decision: the aggregate pilot gate passes; retain the private route pending an
independent quiet-session repeat and a scoped dispatch design.** Primary paired
speedups are 1.068–1.231× versus CPU and 1.081–1.251× versus Prep. All six primary
cases and all five concurrency cases pass their predefined criteria. This does
not establish a universal size threshold or a repeatable gain on every session.

This is the 2026-10-10 prescribed-token checkpoint on Apple M4 Pro. It compares
three complete encoders from one frozen source snapshot: scalar CPU, GPU
unpack/RCT followed by CPU tokenization ("Prep"), and combined GPU unpack/RCT
plus prediction/token generation ("Tokens"). All three share the CPU entropy
encoder and produce identical complete bytes. Public routing remains unchanged.

## Results

The [machine-readable results](modular-gpu-token-performance-results.json) retain
all 56 cases, paired intervals, process/permutation medians, throughput, memory
and separate stage profiles. Each primary case below is RGB8 at 4000×3000,
prefix coding, one call and four CPU participants. Times are ordinary complete
call medians; ratios use paired observations.

| Photograph / policy | CPU ms | Prep ms | Tokens ms | CPU/Tokens [95% CI] | Prep/Tokens [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1029 / gradient | 283.798 | 289.756 | 267.506 | 1.068 [1.041, 1.086] | 1.081 [1.057, 1.096] |
| 1029 / split7 | 451.865 | 458.410 | 367.612 | 1.231 [1.214, 1.242] | 1.251 [1.228, 1.261] |
| 1039 / gradient | 278.536 | 284.366 | 255.678 | 1.093 [1.085, 1.097] | 1.108 [1.098, 1.119] |
| 1039 / split7 | 459.909 | 456.400 | 394.544 | 1.152 [1.139, 1.196] | 1.157 [1.149, 1.206] |
| 1049 / gradient | 278.440 | 291.176 | 259.507 | 1.089 [1.083, 1.099] | 1.104 [1.090, 1.120] |
| 1049 / split7 | 449.305 | 457.983 | 377.058 | 1.199 [1.192, 1.211] | 1.212 [1.202, 1.221] |

Four simultaneous calls also pass. These are whole-batch wall times, with one
CPU participant per call and an aggregate cap of four.

| Four-call workload | CPU ms | Prep ms | Tokens ms | CPU/Tokens [95% CI] | Prep/Tokens [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| mp1-rgb8-b4 | 63.824 | 65.550 | 55.015 | 1.157 [1.143, 1.171] | 1.199 [1.189, 1.203] |
| hd-rgb8-b4 | 126.475 | 130.142 | 107.550 | 1.179 [1.168, 1.192] | 1.210 [1.197, 1.217] |
| photo-large-1029-gradient-b4 | 662.105 | 690.318 | 549.506 | 1.218 [1.205, 1.229] | 1.260 [1.242, 1.283] |
| photo-large-1039-gradient-b4 | 631.554 | 664.860 | 540.763 | 1.174 [1.079, 1.226] | 1.213 [1.128, 1.271] |
| photo-large-1049-gradient-b4 | 632.186 | 666.847 | 569.070 | 1.105 [1.046, 1.168] | 1.159 [1.101, 1.212] |

For the three 12 MP four-call workloads, Tokens delivers 84.35–88.76 MP/s,
versus 72.50–76.00 MP/s on CPU, using median whole-batch wall times.

The gain is not uniform over the matrix. The 1×1 case takes 0.056 ms on CPU
versus 0.223 ms for Tokens (paired ratio 0.248). Large-photo ANS ratios versus
CPU are only 1.001/1.010/1.023 for 1029/1039/1049; the first two intervals
include parity. In contrast, serial 4K RGB8 reaches 1.204× and four-participant
4K RGB8 reaches 1.074×. These examples do not replace the predefined gate.
Fifty-three of 56 cases have intervals above parity against both controls;
this is a description of the selected pilot, not a general dispatch rule.

Repeatability remains a limitation even though the aggregate gate passes.
For example, 1049 gradient versus CPU has cohort medians 1.113/1.031/1.090;
one cohort falls below the 5% target. No separate-session repeat was run.
The evidence supports continued private integration and dispatch investigation,
with an independent quiet repeat before promotion, especially for modest gains.

Backend initialization took 34.12–44.45 ms in the three ordinary processes,
outside complete-call timings.

## Stage and storage observations

The separate profiled calls show the replacement of CPU preparation and
prediction work. The CPU column below is the median per-call sum of input,
RCT and token stages; the other columns are medians from the Tokens route.
They are diagnostics from a separate cohort, not additive explanations of
the ordinary latency medians.

| Case | CPU prep + tokens ms | GPU combined interval ms | Remaining CPU model ms | Remaining CPU emission ms |
| --- | ---: | ---: | ---: | ---: |
| 4K RGB8 / one participant | 115.83 | 21.88 | 92.39 | 275.59 |
| 4K RGB8 / four participants | 39.33 | 18.42 | 92.59 | 72.73 |
| 1029 gradient / four participants | 58.57 | 32.89 | 150.55 | 80.97 |
| 1029 split7 / four participants | 126.82 | 35.37 | 243.16 | 84.66 |

Scalar input/RCT/token stage counters are zero for the Tokens route. CPU
modeling and entropy emission remain substantial. Modeling itself is slightly
slower in these token-route diagnostics (for example, 150.55 versus 141.40 ms
for 1029 gradient); moving prediction does not remove this CPU continuation.
The measurements do not isolate the cause of that modeling difference.

| Single-call case | CPU peak MiB | Prep peak MiB | Tokens peak MiB | Common admission MiB |
| --- | ---: | ---: | ---: | ---: |
| 4K RGB8 / one participant | 335.55 | 335.55 | 288.09 | 3133.02 |
| 4K RGBA16 / one participant | 447.36 | 447.36 | 384.08 | 4177.16 |
| 1029 gradient / four participants | 482.00 | 482.00 | 413.34 | 4532.47 |

Compact token backing lowers the observed managed peak in these cases.
The common conservative admission capacity is much larger than actual backing;
neither figure is process RSS.

## Host activity and limits

All 337 process/thermal snapshots reported no thermal or performance warning.
Four allocation-failure tests from the separate palette/context worktree
overlapped the first 14 snapshots of the first cohort (about its first 66
seconds). They finished naturally; none were stopped by this qualification.
No competing codec/build jobs were sampled in the later ordinary or profile
cohorts. The first cohort remains in the analysis.

| Cohort | Snapshots | Median other CPU % | Maximum other CPU % | Codec/build snapshots |
| --- | ---: | ---: | ---: | ---: |
| measure-17 | 87 | 149.4 | 664.9 | 14 |
| measure-43 | 87 | 188.8 | 363.9 | 0 |
| measure-91 | 87 | 163.9 | 376.3 | 0 |
| profile-137 | 76 | 111.3 | 419.9 | 0 |

These are sums of sampled non-benchmark process CPU use: 100% is about one
core equivalent. Normal desktop activity remained present throughout.
Process samples do not establish GPU idleness or capture every short spike.
The bootstrap intervals do not remove this limitation or replace an
independent quiet-session repeat.

## Scope and acceptance

The [protocol](../tools/modular_oracle/GPU_TOKENS_PERFORMANCE.md) fixes six primary
cases: three original 4000×3000 RGB8 photographs, each using gradient and a
seven-node split tree, prefix coding and four CPU participants. Every primary
case must reach a median paired control/token ratio of at least 1.05 and a 95%
interval strictly above one, against both CPU and Prep. All five four-call
workloads must have a median and lower interval of at least 0.98 against both
controls. `acceptance.json`, including the input-manifest hash, was frozen before
the first ordinary timing. Serial gains cannot substitute for the primary gate.

The full 56-case matrix retains the earlier 32 cases and adds six synthetic
split-tree cases and six cases for each large photograph. It covers all six
packed formats, RCT 0/6/41, prefix/ANS, odd offsets, row padding, both 16-bit byte
orders, one/four CPU participants and four simultaneous calls. The split tree
uses properties 8/4/9 and Gradient/Select/Average4/Left leaves, with zero offsets
and unit multipliers. It exercises the general tree kernel; it is a prescribed
policy, not a learned or quality-optimized tree.

The 19 inputs include three existing 500px wesaturate photographs and three
original full-resolution comparison-corpus photographs: 1029 (planter), 1039
(sunforest) and 1049 (rivercity). Their RGB integer samples are preserved without
resizing or color conversion. Source and PAM hashes are retained. The larger
structured/alpha/noise cases are deterministic synthetic scalability probes.
This selected matrix is not a full photographic corpus or a factorial test of
all eligible predictors and trees.

## Correctness and measurement boundary

All 56 cases passed byte equality among the three routes and independent
pinned-libjxl decoding of 1,029,608,480 sample values under Metal API/GPU
validation. The count includes repeated policies on the same inputs. The decoder
revision is `e8ff09762481785938d8e4e01333ed3917571161`. Every subsequent warmup,
ordinary and diagnostic output must equal its saved qualified file.

The broader [correctness qualification](gpu-prediction-and-token-generation.md)
also checks 1,004,674 intermediate sample traces and, for each of the standalone
and composed producers, 5,195 cases and 49,630,564 exact tokens. The private
complete workflow passes 795 scalar-file comparisons, 6,732,696 independently
decoded samples and 786 injected failures. CPU Release, Metal validation and
scoped ASan/UBSan gates passed. Apple leak detection was disabled; native Metal
sanitizer runs suppress only the existing metal-cpp Foundation nil-call idiom.
The external decoder remains a separate Release gate, with all 931 native
sanitizer files matching the independently decoded Release files by SHA-256.
These results do not claim an unsuppressed whole-program sanitizer pass.

Ordinary measurements use three fresh processes (seeds 17/43/91), two warmup
triplets and six measured triplets per case/process. All six route permutations
occur once per measured case/cohort, with shuffled case order. A separate process
records six diagnostic triplets after two warmups for the 51 single-call cases.
Its stage timings are not pooled with ordinary timings.

All 1,008 ordinary triplets, 306 diagnostic triplets, 438 warmup triplets and
56 qualification triplets passed. Including every image in four-call batches,
this is 6,549 complete encodes.

Each sample includes admission, allocation, upload, GPU dependency and completion,
token handoff, CPU continuation and vector publication. Input loading, backend
setup, domain/output construction, comparisons, decoding, hashing, I/O and output
destruction are excluded. Earlier route outputs stay alive through the triplet.
Each route uses the same four-participant domain capacity: the largest complete
storage bound among the three routes, multiplied by concurrent calls. Four-call
samples measure batch wall time, including worker creation/join, with one CPU
participant per call and a shared cap of four.

Every call checks charge/participant release and peak limits. CPU uses zero
device submissions/allocations, Prep uses one/two, and Tokens uses one/four per
image. The Tokens route records unpack/RCT, a dependency barrier and prediction
in one command buffer with one wait. It releases upload, planes and dispatch
scratch before handing the compact token arena to CPU. No planar readback,
scalar tokenization, diagnostic traces or guard arenas run in timed calls.

Ratios are medians of paired control-time/token-time observations, so they need
not equal ratios of marginal latency medians. Values above one favor Tokens.
Intervals use 10,000 bootstrap replicates, resampling within process cohorts,
with seed 20261010. They describe this session's sampling noise, not uncertainty
across devices/sessions or multiplicity-adjusted simultaneous significance.
Per-cohort and per-permutation medians are retained with the full results.

## Source and retained evidence

The frozen source is committed base `fa4d9f305b837c9e652cd255e448066b7d071608`
plus a SHA-256 manifest of the step-3–6 overlay. It uses the same library and test
sources as the step-5 qualification snapshot. The Release C++20 build embeds
Metal 3.1 kernels, disables Metal profiling and sanitizers, and links the pinned
oracle, with `-O3 -DNDEBUG` host flags. Measurements use macOS 27.0.1 (26A434),
14 CPU cores and 48 GiB RAM, on AC power. Ordinary measurement explicitly
disables Metal API/GPU validation. Pipeline and OS caches are not cleared;
fresh-process setup observations are not cold-cache latency claims.
Documentation and analysis validation were updated afterward without changing
the measured library or benchmark; the exact analysis script is retained and
hashed separately. The previous unpack/RCT report remains evidence for its own
revision and is not used as this run's numerical control.

Local artifacts are retained in:

- `build/modular-token-workflow-qualification/`: isolated source, build/test logs,
  qualification metadata, sanitizer/Release output hashes and binaries.
- `build/modular-token-performance-build/`: frozen source/overlay manifest,
  configure/build logs, measured executable and Metal library.
- `build/modular-token-performance-20261010/`: acceptance rules, input/source
  hashes and samples, expected files, qualification metadata, ordinary/warmup/
  diagnostic CSVs, setup records, environment/commands/process/thermal samples,
  analysis script, full results/table and artifact hashes.

Managed backing excludes caller input, published vectors, backend/driver memory
and small control objects; it is not RSS. The `gpu_input_tokens` stage includes
metadata, allocation, upload, both kernels, completion and handoff. It does not
isolate kernel time, GPU utilization or transfer bandwidth. A second independent
session is required before using a small gain to support public dispatch.
