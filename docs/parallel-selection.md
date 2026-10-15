# Parallel Modular transform and predictor selection

Branch `perf/parallel-selection` starts at `25b88da`, the merged entropy and
tree-fitting work on `perf/parallel-tree-fitting`. This change parallelizes the
local selection stage before sample preparation and tree fitting, in both CPU
and hybrid workflows. It changes scheduling, not the compression policy.

## Execution

Global selection remains one job before stream placement. The workflow then
creates all local records borrowing immutable channel slices. Nonempty records
are selected in bounded waves; each job owns its palette candidates, RCT trial
planes, weighted-predictor histories, selected image and counters. Empty stream
slots and IDs remain in their original positions. Final channel/transform/WP
views are installed only after all selection jobs join.

Within a stream, palette acceptance, the 19 eligible RCT trials, the five
weighted-predictor presets, floating-point arithmetic and tie order are
unchanged. Each stream's chosen tree is still fit by the existing learner.
The later CPU tokenizer is unchanged.

`ReferenceWorkflowOptions::selection_cpu_participants` is a private upper bound:
zero inherits admitted CPU capacity, one selects the serial control, and 2–256
requests bounded concurrency. Public learned encoding inherits its existing
`cpu_thread_count` and shared execution-domain limits. Unadmitted private calls
and calls inside explicit parallel scopes stay serial. Selection workers do
not launch nested teams.

## Memory and failure behavior

Scheduler metadata is admitted separately. Each job reserves the existing
selection planner's complete scratch-plus-result bound before workers launch.
Selected images remain managed backing after their job reservation closes.
Admission failure shortens a wave; if even one job cannot fit alongside the
scheduler metadata, metadata is released and remaining records use the original
serial admission path. Candidate sets and sampling are never reduced.

Worker allocation and launch failures return errors after joining every
started worker. They are not retried. Records are unpublished during selection,
so the previous prepared frame, encoded output and public summary remain intact
on failure. All reservations and CPU grants use the caller's existing domain.

## Profiling and reproduction

The existing `transforms` stage records the caller's joined wall/process-CPU
interval. `selection_seconds` records the selection scheduler's caller wall,
while `selection_work_seconds` sums job wall intervals. Work sums overlap and
must not be added to encoder wall stages. Stats also expose actual peak
participation, waves, memory-limited waves and completed selection calls.

The hybrid profiler accepts an optional last `selection-participants` argument:

```text
gjxl_modular_hybrid_profile INPUT OUT ROUTE auto best fit-off LEARNING CPU_CEILING ENTROPY SELECTION
```

Supplying it selects schema 4. Older invocations retain serial selection and
their existing schema and stage-array lengths. For paired complete-call timing,
hold learning and entropy participation fixed and vary selection only:

```sh
python3 tools/modular_reference_learning/run_parallel_entropy.py \
  --phase selection --binary build/release/gjxl_modular_hybrid_profile \
  --input INPUT --out build/parallel-selection-cpu \
  --route cpu --threads 14 --repetitions 3
```

Use `--route auto` for the hybrid route. Each mode has one warmup and alternating
measured runs. The runner requires exact independent decoding and identical
compressed hashes, retains one output, and records source, binary, input,
power and thermal provenance. Timing excludes input loading, backend setup,
decoder validation and file I/O.

## Validation

Eight focused Release suites passed: `modular_parallel_selection`,
`modular_parallel_learning`, `modular_parallel_entropy`, `modular_api`,
`modular_reference_workflow`, `modular_reference_selection`,
`modular_reference_hybrid` and `modular_reference_hybrid_palette`.

The dedicated suite checks exact selection traces, transforms, predictors,
selected samples, tree layout and complete prefix/ANS bytes at 1/2/4/automatic
participants. Its 36 parallel files independently decode through pinned libjxl.
Cases include Gray/RGB/RGBA, 8/16-bit samples, padded rows, little endian,
palette/constant/gradient/random content, global/local/buffered scopes and
empty canonical slots. It also checks per-image/shared-domain CPU limits,
nested/unadmitted execution, tight-memory serial fallback, concurrent callers,
allocation faults and partial worker-launch unwinding with preserved prior
frames. Existing hybrid suites checked 58 files, including multiple GPU feature
batches, All predictors, wide palettes and GPU failure handling. The public API
suite checked 112 C/C++ cases and its allocation/publication failure paths.

The complete dedicated suite passed ThreadSanitizer with `-O1 -g -DNDEBUG`,
`-fsanitize=thread -fno-omit-frame-pointer` and `TSAN_OPTIONS=halt_on_error=1`.
This sanitizer build disables Metal and the external decoder oracle; the
Release suites provide independent decoding and Metal coverage. Logs are in
`build/release/selection-ctest.log`, `build/release/Testing/Temporary/LastTest.log`
and `build/tsan/parallel-selection.log`.

The Release build enables Metal and reuses the existing pinned libjxl build
and matching `metal-cpp` headers; no dependency checkout is duplicated:

```sh
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGJXL_ENABLE_METAL=ON \
  -DCMAKE_CXX_FLAGS='-isystem /Users/yunhocho/GitHub/gjxl/third_party/metal-cpp' \
  -DGJXL_MODULAR_ORACLE_BUILD=/Users/yunhocho/GitHub/gjxl/build/pinned-libjxl
```

The older profiler schemas 1/2/3 and the runner's default entropy-comparison
mode also passed a small exact-decode compatibility check. New selection
measurements use schema 4; existing entropy measurement keys remain unchanged.

## Measured pilot

Apple M4 Pro, AC power in every recorded snapshot, active desktop, Release
`-O3 -DNDEBUG`. Input is the retained 4000×3000 RGB8 `photo-large-1029.pam`.
Both routes use automatic scope, Best predictors and estimated direct ANS.
Learning, entropy and the overall CPU ceiling are fixed at 14; only selection
changes between one and 14 participants. Each mode has one warmup and three
measured complete calls in alternating order.

| Route | Selection seconds, 1 → 14 | Complete seconds, 1 → 14 | Selection speedup | Complete speedup |
| --- | ---: | ---: | ---: | ---: |
| CPU | 3.751 → 0.461 | 10.866 → 7.588 | 8.136× | 1.432× |
| Automatic hybrid | 3.751 → 0.491 | 9.404 → 6.108 | 7.642× | 1.540× |

Values are medians. CPU complete-call ranges were 10.769–10.885 s at one
selection participant and 7.482–7.782 s at 14. Hybrid ranges were 9.328–9.531 s
and 6.038–6.110 s. Every parallel run used 14 selection participants in 14 waves
for 192 selections, without memory-driven reduction. Automatic hybrid used
four GPU feature batches and five token batches in each run.

Maximum complete-call committed managed capacity was unchanged between modes
within each route: 1,891,701,896 bytes for CPU and 1,902,505,608 bytes for hybrid.
This is reserved managed capacity, not resident memory. Selection still needs
per-job scratch; the larger subsequent phases set the overall peak here.

All 16 files, including warmups, independently decoded exactly and shared the
same complete output hash, also matching the parent entropy qualification:
`2cfd2402384ff6670fd8357e1e5e626aded66c7a91f7293f8ba54ec945a367ad`.
This is a one-image pilot, not a corpus-wide or libjxl speed comparison.

[Structured results](parallel-selection-results.json) bind input, executable,
source, output and validation hashes. Raw results and manifests are in
`build/parallel-selection-cpu` and `build/parallel-selection-auto`; the APFS-cloned
profiler and saved runner are in `build/selection-provenance`. Benchmark source
diffs include the new source/test files and pre-results documentation. Only
documentation/results were added afterward.

## Retired entropy worktree

Both old branch tips were verified at `25b88da` before local cleanup. The old
worktree was clean, idle, and had no inbound CMake dependency from other builds.
Its 21 benchmark files and 765 validation files were hash-verified and preserved
under `../gjxl-parallel-tree-fitting/build/parallel-entropy-comparison` and
`../gjxl-parallel-tree-fitting/build/parallel-entropy-validation`. The latter
contains an archive manifest. Historical manifests retain their original paths.
The merged local entropy branch/worktree and regenerable builds were removed;
remote refs were unchanged.
