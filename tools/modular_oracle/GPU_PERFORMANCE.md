# Prescribed unpack/RCT performance pilot

This measures the private CPU and GPU-assisted complete Modular encoders with
identical prescribed gradient/RCT/entropy decisions. It does not select public
backend routing or measure policy search. The GPU path performs unpack/RCT,
reads back the planes, and runs the shared CPU continuation.

Build `gjxl_modular_gpu_benchmark` in Release with `GJXL_ENABLE_METAL=ON`,
`GJXL_ENABLE_METAL_PROFILING=OFF`, `GJXL_BUILD_BENCHMARKS=ON`,
`GJXL_BUILD_TESTS=ON`, and `GJXL_MODULAR_ORACLE_BUILD` pointing to the clean pinned
libjxl build. Use an isolated source snapshot when another effort is editing the
checkout. The target is private and not installed. Without the pinned reference
target it can compare CPU/GPU bytes, but it cannot establish independent sample
correctness; require `reference=1` in the qualification setup record.

The Python scripts require numpy; input preparation also requires Pillow. Run
from the repository root, replacing `BUILD`, `RUN`, and `WESATURATE_500PX` with
local paths. Retain exact versions, source overlays and build flags with results.

```sh
python3 tools/modular_oracle/prepare_gpu_performance.py RUN --wesaturate WESATURATE_500PX
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 BUILD/gjxl_modular_gpu_benchmark qualify RUN/cases.tsv RUN 1 0 0
python3 tools/modular_oracle/run_gpu_performance.py BUILD/gjxl_modular_gpu_benchmark RUN
python3 tools/modular_oracle/analyze_gpu_performance.py RUN
```

The optional photograph directory is libjxl testdata's `wesaturate/500px`. Three
existing sRGB8 PNGs are copied as decoded RGB samples without resizing or color
conversion; originals are identified by path and SHA-256. Generated PAMs and
their hashes are retained. Large generated ramps/discontinuities/alpha/noise
inputs are scalability probes, not representative photography. No performance
assertion uses regenerated input in place of the retained hashed input.

The TSV columns are name, absolute input path, prescribed RCT, `prefix`/`ans`,
CPU participants (1 or 4), simultaneous calls (1 or 4), byte order and row
padding. Four simultaneous calls each use one CPU participant and share a
four-participant domain and a common initialized backend. Their measured batch
wall time includes thread construction/join. Every input starts at an odd byte
offset, and the final row has exactly the required extent. The default matrix
has 32 cases with photographs, 26 without them. It is a selected matrix, not a
factorial sweep of every combination.

## Measurement boundary and invariants

- The steady clock encloses the complete synchronous API call, including
  admission, GPU allocations, upload, submission/wait, host readback, device
  release, CPU continuation and atomic vector publication. Fresh output is
  used on every call. Both paths have the same prescribed decisions, CPU cap
  and domain capacity. No automatic dispatch/fallback or search is involved.
- Input loading/layout conversion, backend initialization, domain and empty
  vector construction, comparisons/decoding/hashing, file I/O, and returned
  output destruction are outside timing. The first result remains live during
  the second call in a pair. CPU-first/GPU-first order is balanced. Input and
  allocator caches are not flushed; this is steady repeated-call behavior.
- Each fresh process initializes one backend; setup wall time is saved
  separately. This is not a cold-cache/process-start latency measurement.
  There are two warmup pairs followed by six measured pairs per case in each
  of three processes (seeds 17/43/91). Case order is shuffled in each process;
  within each case, CPU/GPU order alternates. Warmup rows remain in the CSV but
  are excluded from analysis.
- Qualification independently decodes the CPU result to every original sample
  and requires GPU bytes to match it. Every warmup, ordinary and profiled pair
  must match both each other and that saved expected file. Each full GPU call
  must have exactly one submission/two sample allocations; CPU must have zero.
  Every call must release all domain charges/participants within its limits.
- Actual managed backing peaks and conservative domain capacity are separate
  columns. These are not process RSS or total physical memory: caller inputs,
  published vectors, backend/driver allocations and small control objects lie
  outside this accounting boundary. Concurrent calls can have different peak
  overlaps even with identical per-call plans.
- An additional fresh process collects four profiled pairs per single-call
  case after warmup. These timings are not pooled with ordinary timings.
  CPU preparation is the sum of exclusive `input` and `transforms`; GPU
  `input` includes allocation/upload/kernel/wait/readback/adoption and device
  destruction. It is not kernel time. The uninstrumented remainder includes
  planning/admission/publication and other gaps; it is not a utilization metric.

## Evidence and interpretation

`run_gpu_performance.py` strips inherited validation/sanitizer injection
environment variables, explicitly disables Metal validation, and records setup,
commands, binary hash, power settings, thermal status, OS/hardware and linked
libraries. It samples process CPU activity and thermal warnings every five
seconds without killing or pausing other work. Stop competing encodes/builds
before running. Inspect `host-activity.jsonl` and record any background load;
this sampling cannot prove GPU idleness or absence of short scheduling spikes.
The runner refuses to overwrite an existing timing cohort.

The report gives medians of complete-call CPU/GPU latency and the median of
paired CPU-time/GPU-time ratios; these two ratios need not be identical. A value
above one favors GPU. Confidence intervals resample pairs within each process
cohort (10,000 bootstrap replicates, fixed seed). They describe sampling noise
in this run, not cross-device/session uncertainty or corrected simultaneous
confidence over the full matrix. Inspect cohort and order-specific medians
before treating small differences as actionable. Do not pool unlike policies,
duplicate format variants or concurrent batches into a purported corpus score.

`results.json` retains per-case intervals, order/cohort summaries, throughput,
memory peaks and diagnostic stages. `results-table.md` is the full ordinary
timing table; `artifacts-sha256.json` covers inputs, expected outputs, raw CSVs,
logs and analysis. Preserve the build, complete source/hash manifest, scripts,
oracle revision and library hashes alongside the run directory. Describe the
source/device/corpus limitations before using this pilot for dispatch decisions.
