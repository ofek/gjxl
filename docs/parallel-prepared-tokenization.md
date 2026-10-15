# Parallel prepared-stream CPU tokenization

This branch starts at `c24f812`, the merged selection, entropy and tree-fitting
work on `perf/parallel-tree-fitting`. CPU sample tokenization now runs across
independent prepared streams. Channel order, pixel raster order, predictor
recurrence, context IDs, tree headers and final codestream order are unchanged.

## Execution and ownership

A validation pass counts generated and borrowed tokens. The result allocates
its token, tree-token and stream-view backing once. Parallel calls compute
canonical offsets before scheduling work; each job writes only its assigned
ranges and view slots. Each channel retains its own scalar weighted-predictor
history, or consumes the same immutable completed feature map as before.
Workers join before the result is published. No input samples or completed
token backing are copied.

`PreparedModularTokenizationOptions::cpu_participants` is an independent private
limit: zero inherits admitted CPU capacity, one is the serial control, and
2–256 caps concurrency. `ReferenceWorkflowOptions::tokenization` forwards it
through the learned workflow. Public learned encoding uses its existing
per-image and shared execution-domain CPU limits. Explicitly nested and
unadmitted calls stay serial, as do inputs with fewer than two CPU token jobs.
A fully GPU-tokenized input launches no CPU tokenization team; local tree
headers still use the existing CPU producer. Entropy limits apply after the
tokenization team has joined and released its grant.

## Storage and failures

The allocation-free prepared-stream planner exposes separate entropy and
tokenization allowances. Its preparation bound includes the owned output,
one maximum weighted-predictor scratch set per admitted token worker, canonical
range metadata, job statuses and thread handles. Metadata and predictor scratch
are released before entropy modeling. Borrowed feature/token owners stay
caller-accounted. The serial token path allocates no parallel metadata.

The complete learned workflow retries memory admission at smaller concurrency,
before any token or model work begins. It reduces the larger parallel stage
first and can reach the serial envelope without changing compression decisions.
Worker errors and partial launch failures join started workers and preserve
previous token and codestream outputs. Work is not retried after launch.

## Profiling and reproduction

The `tokens` stage measures joined caller wall and process CPU. Diagnostics
report CPU stream jobs, generated versus borrowed tokens, parallel calls,
actual peak participation, planned participation and memory-driven reductions.
Workers do not share a profiling sink or contribute overlapping wall timers.

The hybrid profiler accepts a trailing tokenization participant limit:

```text
gjxl_modular_hybrid_profile INPUT OUT ROUTE auto best fit-off LEARNING CPU_CEILING ENTROPY SELECTION TOKENIZATION
```

This selects schema 5. Older invocations retain serial CPU tokenization and
their earlier schemas and stage-array lengths. For the paired pilot, learning,
selection, entropy and the CPU ceiling remain fixed at 14:

```sh
python3 tools/modular_reference_learning/run_parallel_entropy.py \
  --phase tokenization --binary build/tokenization-provenance/hybrid_profile \
  --input INPUT --out build/parallel-prepared-tokenization-cpu \
  --route cpu --threads 14 --repetitions 3
```

Use `--route auto` for the hybrid control. Each mode has one warmup and three
measured complete calls in alternating order. Every output must independently
decode exactly and share the same compressed hash. Input loading, backend
setup, decoder validation and file I/O are outside the timing boundary.

## Validation

Eleven focused Release suites passed: `modular_parallel_prepared_tokenization`,
`modular_parallel_learning`, `modular_parallel_entropy`,
`modular_parallel_selection`, `modular_api`, `modular_prepared_stream`,
`modular_prepared_stream_reference`, `modular_shared_feature_consumers`,
`modular_reference_workflow`, `modular_reference_hybrid` and
`modular_reference_hybrid_palette`.

The dedicated test compares every sample token, owned token order, tree token
and canonical stream slot at 1/2/4/automatic participation. Cases include
Gray/RGB/RGBA, 8/16-bit input, padded rows, tiny single-group frames,
global/local/mixed tree scopes and Squeeze with multiple DC groups. All 54
parallel codestreams across prefix, prefix-derived ANS and exact/estimated
direct ANS independently decoded through pinned libjxl. Entropy is deliberately
limited to one participant in those comparisons to check independent controls.
Partial/all cached weighted features and mixed/all completed interleaved/split
tokens preserve token content and borrowed backing identities.

The same test checks allocation-free planning, the standalone preparation
storage bound, per-image and shared-domain CPU limits, unadmitted/nested calls,
concurrent encodes, managed allocation faults, partial worker launches and
in-worker residual errors with previous outputs preserved. The complete
suite passed ThreadSanitizer with `-O1 -g -DNDEBUG`,
`-fsanitize=thread -fno-omit-frame-pointer` and
`TSAN_OPTIONS=halt_on_error=1`. That build excludes Metal and the external
decoder; the Release run covers both. It reported no races.

Existing tests additionally checked 648 prepared-stream reference files,
112 public C/C++ cases, 98 learned-workflow files and 58 hybrid files, including
GPU feature/token batches and GPU failure paths. Older profiler schemas 1–4
retained serial CPU tokenization, their original stage-array lengths, identical
bytes and exact decoding on a small input. The runner's existing default
entropy and explicit selection modes also passed paired compatibility checks.

Validation logs are in `build/release/tokenization-focused.log`,
`build/release/tokenization-regression.log`,
`build/release/Testing/Temporary/LastTest.log` and
`build/tsan/parallel-prepared-tokenization.log`. Compatibility results are in
`build/tokenization-profiler-compatibility`.

## Measured pilot

Apple M4 Pro, battery power in every recorded snapshot, active desktop,
Release `-O3 -DNDEBUG`. Input is the retained 4000×3000 RGB8
`photo-large-1029.pam`. Both routes use automatic scope, Best predictors and
estimated direct ANS. Learning, selection, entropy and the overall CPU ceiling
are fixed at 14; only CPU tokenization changes between one and 14 participants.
Each mode has one warmup and three measured complete calls in alternating order.

| Route | Tokens stage seconds, 1 → 14 | Complete seconds, 1 → 14 | Complete speedup |
| --- | ---: | ---: | ---: |
| CPU | 2.984 → 0.289 | 11.127 → 8.066 | 1.380× |
| Automatic hybrid control | 0.119 → 0.128 | 8.454 → 8.443 | 1.001× |

Values are medians. CPU tokenization improved 10.321×. Every parallel CPU run
used 14 participants for 192 streams and 36,000,000 generated tokens, without
memory-driven reduction. CPU complete-call ranges were 10.709–11.227 s for
serial tokenization and 7.847–8.669 s for parallel tokenization. The broader
complete-call variation includes other stages and active-desktop scheduling.

The hybrid control borrowed all 36,000,000 tokens, used no CPU sample jobs and
launched no CPU tokenization team in either mode. Its tokens stage includes
existing GPU production, so its small timing difference does not measure CPU
tokenization scaling. Hybrid complete-call ranges were 8.252–10.774 s and
8.161–9.468 s. No hybrid speedup is claimed.

Maximum complete-call committed managed capacity was unchanged between modes
within each route: 1,891,702,456 bytes for CPU and 1,902,505,608 bytes for hybrid.
This measures reserved managed capacity, not resident memory. Token workers
need additional predictor scratch, but later stages set the overall peak for
this input.

All 16 files, including warmups, independently decoded exactly and matched the
merged parent's complete output hash:
`2cfd2402384ff6670fd8357e1e5e626aded66c7a91f7293f8ba54ec945a367ad`.
This is a one-image paired pilot on battery power, not a corpus-wide or libjxl
speed comparison. Use the current paired controls; earlier selection results
were measured on AC power.

[Structured results](parallel-prepared-tokenization-results.json) bind input,
frozen executable, source diffs, output and validation hashes. Raw manifests
and per-run diagnostics are in `build/parallel-prepared-tokenization-cpu` and
`build/parallel-prepared-tokenization-auto`. The APFS-cloned profiler and saved
runner are in `build/tokenization-provenance`. Source diffs include the new
test and pre-results documentation. Only comments, documentation and results
were added after the benchmark executable was frozen. Identical retained
outputs use APFS clones of the archived parent artifact to avoid duplicate
physical storage.

## Retired selection worktree

Both old tips were verified at `c24f812` before cleanup. The old selection
worktree was clean and idle with no inbound CMake dependency. Its 109 validation
files and 58 study/provenance files were hash-verified and preserved under
`../gjxl-parallel-tree-fitting/build/parallel-selection-validation`,
`parallel-selection-cpu`, `parallel-selection-auto`, `selection-provenance`
and `selection-profiler-compatibility`. The validation directory contains an
archive manifest; historical manifests retain their original paths. The merged
local `perf/parallel-selection` branch, worktree and regenerable build products
were removed. Remote refs were unchanged. The new worktree reuses the existing
pinned libjxl build and matching Metal headers without duplicating dependencies.
