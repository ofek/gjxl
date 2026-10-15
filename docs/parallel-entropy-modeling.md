# Parallel Modular entropy modeling

Branch `perf/parallel-entropy-modeling` starts at `654e517` from
`perf/parallel-tree-fitting`. It retains the tree-fitting scheduler and encoding
decisions, and enables CPU parallelism in the subsequent prepared-stream encoder.

## Execution

For frames with separate group sections, independent stream jobs build their
local histogram/code models, write the local tree and model, and emit the token
payload. Each job has its own model and section writer. A shared model is built
before the group jobs and then read immutably. The global section is emitted
before launching those jobs; final TOC and file assembly stay ordered.

A shared model, or a local model running without outer stream workers, may use
the existing direct-rate ANS configuration search across clusters (up to eight
participants). Local stream workers install a one-participant scope to suppress
nested ANS teams. Histogram collection and clustering inside an individual model
remain serial; local stream parallelism runs these operations concurrently for
different models. Prefix and prefix-derived ANS also benefit from stream jobs.

Single-group frames retain ordered writing because their stream payloads share
section zero. Their ANS cluster search can still use the available CPU allowance.
Tree learning and transform/predictor decisions are unchanged.

`PreparedModularEntropyOptions::cpu_participants` is a private upper bound:
zero inherits the admitted caller's allowance, one selects the serial control,
and 2–256 requests bounded concurrency. Public learned encoding uses automatic
entropy participation under `ModularEncodingOptions::cpu_thread_count` and the
shared execution domain. Unadmitted private calls and explicit outer parallel
scopes stay serial. Worker grants respect both per-image and domain limits.

## Memory and failure behavior

The allocation-free planner includes simultaneous stream scratch, worker/status
arrays and optional worker profiles, in addition to token and section backing.
The existing ANS storage bound already covers cluster-search workers. Plan and
encode must use the same entropy options and CPU scope. A plan's participant
count is an allowance; a busy CPU domain may grant fewer workers.

The complete learned workflow tries to reserve the planned entropy envelope
before encoding. On memory admission failure, it halves the allowance and
replans, down to the original serial envelope. It retries admission only; an
encoding or worker-launch failure is returned after joining workers. No partial
file is published, and prior output remains unchanged. Allocation accounting
stays in the original shared reservation/domain throughout each joined stage.

## Profiling

Parallel jobs overlap model construction with emission in other jobs. The new
`stream_encoding` top-level stage records the caller's joined wall interval and
process CPU. `model` and `emission` retain serial/global intervals. Sum these
three stages to compare the complete model-plus-emission phase between modes.

Worker profiles are private and reduced after joining. The
`parallel_entropy.model_work_seconds` and `emission_work_seconds` fields sum
worker intervals; they must not be added to top-level wall times. `entropy_work`
also retains work-time semantics. Diagnostics include actual stream participant
grants and jobs, plus the workflow's admitted allowance and memory reductions.

`gjxl_modular_hybrid_profile` accepts an optional final `entropy-participants`
argument. Supplying it selects schema 3 with nine stage entries and the entropy
diagnostics. Omission keeps serial entropy and the original eight entries for
the older schema 1/2 study runners.

## Validation

The dedicated `modular_parallel_entropy` test compares serial, two-participant
and four-participant complete files across prefix, prefix-derived ANS, and
direct ANS with exact/estimated selection and 32/64-cluster limits. It covers
shared, local and mixed trees; empty slots; single-section frames; 8-bit RGB and
16-bit RGBA with Squeeze/DC streams; and borrowed completed tokens. With the
pinned oracle enabled, each parallel result is independently decoded and
compared with the original integer samples.

Additional checks cover memory-pressure fallback, per-image and shared-domain
limits, nested execution, concurrent callers, allocation faults and partial
worker-launch failures in both the stream and ANS-cluster schedulers. Existing
public C/C++ API tests exercise the default route and its allocation/publication
failure handling. The global-fit assertion checks the learning launch site,
because entropy work may now use workers even with a global tree.

All nine selected Release suites passed: `modular_parallel_entropy`,
`modular_parallel_learning`, `modular_api`, `modular_prepared_stream`,
`modular_profile`, `entropy_storage_plan`, `modular_reference_workflow`,
`modular_prepared_stream_reference`, and `ans_selection`. The new parity matrix
independently decoded 108 parallel files; the existing reference suites checked
648 prepared-stream files and 98 workflow files. The complete dedicated
concurrency suite also passed ThreadSanitizer (`-O1 -g`, CPU-only, without the
external decoder oracle), with no reported races. No exhaustive legacy
`modular_workflow` allocation sweep was run.

Builds use a fresh small CPU-only target set and reuse the existing pinned
libjxl oracle build. The new worktree does not duplicate dependency checkouts or
the parent worktree's build artifacts.

## Measured pilot

On the Apple M4 Pro, AC power throughout recorded snapshots, the retained
4000x3000 RGB8 `photo-large-1029` input was encoded on the CPU route. Learning
participation and the overall CPU ceiling were both fixed at 14. Only entropy
participation changed. Each mode had one unmeasured warmup and three measured
complete calls, with alternating order and independent exact decoding.

| Entropy participants | Median complete seconds | Median model + emission seconds | Maximum managed backing MiB | Maximum committed capacity MiB |
| --- | ---: | ---: | ---: | ---: |
| 1 | 12.374 | 2.679 | 785.440 | 1608.376 |
| 14 | 10.867 | 1.119 | 807.989 | 1804.068 |

The model-plus-emission phase improved **2.394x**, and complete encoding
improved **1.139x** on this input. Complete-call ranges were 12.323–12.407 s
and 10.823–10.916 s respectively. The parallel calls used 14 stream participants
for 192 jobs, with no memory-driven reduction. Actual managed backing increased
22.549 MiB; the conservative capacity reservation increased 195.692 MiB.
Committed capacity includes reserved space and is not resident memory.

All eight files were byte-identical, including warmups, and also matched the
parent branch's retained qualification output for this photograph:
`2cfd2402384ff6670fd8357e1e5e626aded66c7a91f7293f8ba54ec945a367ad`.
This is one active-desktop input and a CPU-only complete-call comparison; it
does not establish corpus-wide, Metal, or libjxl speedups.

[Structured results](parallel-entropy-modeling-results.json) retain the
observations and provenance. Local raw evidence and the frozen profiler are in
`build/parallel-entropy-comparison`; the executable was retained with an APFS
clone, and duplicate compressed outputs were removed after hash equality and
independent decoding.

## Reproduction

```sh
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGJXL_ENABLE_METAL=OFF -DGJXL_MODULAR_ORACLE_BUILD=/path/to/pinned-libjxl
cmake --build build/release --target gjxl_modular_parallel_entropy_test \
  gjxl_modular_hybrid_profile -j 6
ctest --test-dir build/release -R '^modular_parallel_entropy$' --output-on-failure
python3 tools/modular_reference_learning/run_parallel_entropy.py \
  --binary build/release/gjxl_modular_hybrid_profile \
  --input /path/to/photo.pam --threads 14 \
  --out build/parallel-entropy-comparison
```

The comparison holds learning participation and the overall CPU allowance
fixed, changes entropy participation only, uses an unmeasured warmup per mode,
and alternates three measured repetitions. It retains every observation,
environment snapshots, source diff and input/binary/output hashes. Each output
is independently decoded; identical output copies are replaced by a reference
to one retained compressed artifact. These are active-desktop observations,
not isolated hardware peak-throughput measurements.
