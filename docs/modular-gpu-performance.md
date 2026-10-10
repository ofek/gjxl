# Modular unpack/RCT performance qualification

**Decision: retain the private path for correctness and future composition;
do not enable automatic GPU dispatch for unpack/RCT plus host readback.**
No case demonstrates a complete-call GPU speedup. Of 32 selected cases, 30 favor
CPU under the paired intervals and two noise cases are inconclusive. This is a
bounded pilot on an active desktop, not a universal performance threshold.

This is the 2026-10-10 first-slice checkpoint on Apple M4 Pro, using committed
revision `155e60e` (private GPU preparation/readback plus the shared CPU encoder
continuation). It measures prescribed gradient/RCT decisions. Public automatic
dispatch and the installed API are outside the experiment.

## Results

Representative ordinary complete-call medians are below. Unless specified, these
use RCT 6, prefix coding, one CPU participant and one call. The ratio is computed
from paired observations, so it need not equal the ratio of marginal medians.
The [machine-readable results](modular-gpu-performance-results.json) include all
32 cases, interval endpoints, per-cohort/order results, memory and stage data.

| Case | CPU ms | GPU-assisted ms | Paired CPU/GPU ratio [95% interval] |
| --- | ---: | ---: | ---: |
| 1×1 RGB8 | 0.057 | 0.241 | 0.235 [0.221, 0.241] |
| 257×255 RGB8 | 3.832 | 4.368 | 0.873 [0.867, 0.879] |
| 1024×1024 RGB8 | 59.658 | 60.776 | 0.983 [0.976, 0.990] |
| 1024×1024 RGBA16, big endian | 72.279 | 73.226 | 0.990 [0.980, 0.998] |
| 3840×2160 RGB8 | 471.848 | 476.801 | 0.992 [0.989, 0.993] |
| 3840×2160 RGB8, ANS | 772.582 | 780.505 | 0.989 [0.987, 0.994] |
| 3840×2160 RGB8, four CPU participants | 198.473 | 201.592 | 0.984 [0.979, 0.986] |
| 3840×2160 RGBA16 | 570.521 | 581.270 | 0.983 [0.976, 0.990] |
| Four simultaneous 1024×1024 RGB8 calls | 63.799 | 65.631 | 0.974 [0.965, 0.981] |
| Four simultaneous 1920×1080 RGB8 calls | 126.717 | 130.456 | 0.971 [0.963, 0.985] |

The three photographs give paired ratios of 0.933–0.943 with prefix coding and
0.965–0.967 with ANS. The two random RGB16 cases have intervals spanning one;
they do not establish a gain. The two four-call workloads achieve 65.74/65.46
MP/s on CPU versus 63.91/63.58 MP/s on GPU-assisted encoding. They therefore do
not provide a concurrency-based reason to enable dispatch.

Backend initialization in the three ordinary measurement processes took
26.84–31.73 ms, separate from the table. Pipeline/OS caches were not cleared;
these are fresh-process setup observations, not cold-cache latency claims.

## Where the time goes

Separate profiled calls show that GPU preparation including upload, allocation,
wait, readback and adoption costs more than scalar preparation in these cases.
CPU preparation includes both unpack and prescribed RCT. These are stage-wall
diagnostics from a different cohort, not terms to subtract from the ordinary
timings above.

| Case | CPU preparation ms | GPU preparation ms | CPU preparation / profiled full call |
| --- | ---: | ---: | ---: |
| 1024×1024 RGB8 | 1.537 | 2.376 | 2.6% |
| 3840×2160 RGB8 | 12.343 | 21.056 | 2.6% |
| 3840×2160 RGBA16 | 16.580 | 25.788 | 2.9% |
| 3840×2160 Gray16, RCT 0 | 2.058 | 6.951 | 1.5% |
| 3840×2160 RGB8, four CPU participants | 12.260 | 15.101 | 6.2% |

For serial 4K RGB8, CPU tokenization/modeling/emission take approximately
100.2/85.6/276.7 ms out of a 477.1 ms profiled full call. Even eliminating its
12.3 ms preparation cost entirely would yield only about 1.027× speedup if all
other work stayed unchanged. With four CPU participants, preparation's share
grows to 6.2%, but the measured GPU route still costs more. Random RGB16 with
ANS is instead dominated by modeling (about 122.1 of 160.8 profiled ms).

The evidence supports extending the private experiment to prescribed prediction
and token generation that consume resident planes before readback, then
repeating complete-call qualification. It does not prove that extension will
win: CPU entropy emission/modeling remain major costs. Kernel-only optimization
or a size threshold for this slice is not justified by these results.

The single-call managed peak is equal on both paths for representative cases:
41.49 MiB at 1 MP RGB8, 335.55 MiB at 4K RGB8 and 447.36 MiB at 4K RGBA16.
Later CPU buffers dominate the peak after GPU planes are released. The respective
conservative admission capacities are 396.57/3133.02/4177.16 MiB; actual backing
and reservation are different measurements. Concurrent peaks vary with overlap.

## Scope and evidence

The source was archived from the committed revision and overlaid only with the
benchmark target and driver. Concurrent context-learning edits in the working
tree were excluded. The native Release build uses `-O3 -DNDEBUG`, embedded Metal
kernels, no Metal timing instrumentation, and no sanitizer instrumentation.
Measurements ran on macOS 27.0.1 (26A434), 14 CPU cores and 48 GiB RAM, on AC
power with an initialized backend. Backend setup is reported separately from
repeated-call latency.

The selected 32-case matrix covers all six packed formats, one pixel through
3840×2160, RCT 0/6/41, prefix and ANS, odd offsets, row padding, both 16-bit byte
orders, one/four CPU participants and two four-call workloads. Three existing
500px wesaturate photographs preserve their decoded RGB8 samples. Larger
structured/alpha/noise images are deterministic synthetic scalability probes.
This is not a full photographic corpus or a factorial comparison of all options.

All 32 cases passed complete CPU/GPU byte equality and pinned-libjxl decoding of
259,195,424 source samples, including invisible RGB. The decoder is revision
`e8ff09762481785938d8e4e01333ed3917571161`. Qualification enabled Metal debug and
shader validation; ordinary measurements disabled both. Each subsequent pair
must match the saved qualified output. Every GPU call must make exactly one
submission/two sample allocations, and every call must return its resource
charges and CPU participants within its declared limits.

The [benchmark protocol](../tools/modular_oracle/GPU_PERFORMANCE.md) documents
the complete timing boundary and reproduction commands. Each ordinary sample
includes admission, upload, synchronization, readback, device destruction, CPU
encoding and vector publication. It excludes input loading/conversion, backend
creation, domain/output construction, validation, I/O and output destruction.
There are 18 paired ordinary observations per case across three fresh processes,
with shuffled cases, balanced CPU/GPU order and two warmup pairs per process/case.
Four additional profiled pairs per single-call case are diagnostic only.

All 576 ordinary pairs and 120 diagnostic pairs passed output/resource checks;
warmups and the 32 initial qualification pairs also passed. This amounts to
2,260 complete encodes when counting all four images in each concurrent batch.

CPU/GPU speedup denotes the median paired CPU-time/GPU-time ratio; values above
one favor GPU. The 95% intervals bootstrap pairs within each process cohort.
They describe this session's sampling noise, not uncertainty across machines,
sessions or a corrected simultaneous inference over 32 cases. Four-call timings
are whole-batch wall times with thread creation/join included and an aggregate
CPU cap of four participants.

The other Modular qualification jobs had completed before measurement. Sixty
process/thermal snapshots recorded no competing codec/build jobs and no reported
thermal/performance warnings. Normal desktop work remained active: the sampled
sum of non-benchmark CPU use had a median of 121% and maximum of 286% (about
1.2 and 2.9 core equivalents). Process samples do not prove GPU idleness or
capture every short spike. Cohort and CPU-first/GPU-first ratios agree on the
representative regressions, but the smallest differences should not be treated
as portable constants. An otherwise-idle repeat and larger natural-image corpus
would be required before making a positive deployment/performance claim.

## Retained artifacts

Local source, build, correctness and measurement evidence is retained under
`build/modular-performance-qualification/`:

- `source-manifest.json`, `source/`, `configure.log`, `build.log`, and `release/`:
  committed source plus the isolated benchmark overlay and runnable binary.
- `qualification.json`, `qualify.log`, and `binary-manifest.json`: decoder,
  validation configuration, sample counts and binary/library hashes.
- `run/inputs.json`, `cases.tsv`, `inputs/`, and `expected/`: exact inputs,
  prescribed decisions, hashes, and independently decoded expected files.
- `run/measure-*.csv`, `profile-*.csv`, and `*-setup.txt`: raw paired ordinary,
  warmup, diagnostic and backend-creation measurements.
- `run/run-provenance.json`, `host-activity.jsonl`: executed commands,
  environment, machine/power state, process samples and thermal observations.
- `run/results.json`, `results-table.md`, and `artifacts-sha256.json`: analysis,
  full table and retained artifact hashes.

Managed backing peaks exclude caller input, published vectors, backend/driver
memory and small control objects. They must not be presented as RSS. Profiled
GPU preparation includes upload/kernel/wait/readback/allocation/adoption; it
does not isolate kernel time, GPU utilization or transfer bandwidth.
