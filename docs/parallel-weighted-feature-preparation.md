# Parallel weighted-prediction feature preparation

Status: Steps 1–5 are complete. The hybrid workflow now uses bounded CPU slice
parallelism or an exact Metal wavefront producer, selected before execution.
The operation and complete encoder qualification below preserve every feature,
candidate decision and encoded byte.
These results cover the legacy candidate-search workflow. The improved
one-fit histogram learner now shares the same feature producers through
[the stream integration](gpu-histogram-learner-integration.md), with its own
[matched CPU/hybrid qualification](gpu-histogram-learner-qualification.md).
This work starts from shared-feature commit `23cedb1` on
`feat/parallel-weighted-feature-prep`, independently of the ongoing learner
improvements on `research/modular-learning-cost`.

The objective is to accelerate construction of the immutable
`{prediction, property15}` map already shared by CPU learning and GPU
tokenization. Preserve exact features, learned policies, candidate decisions and
complete encoded bytes. Tree fitting, sampling policy, cache replacement policy
and entropy modeling are outside this effort.

## Motivation and measured boundary

The [shared-feature pilot](gpu-modular-shared-features.md) and its
[saved results](gpu-modular-shared-features-results.json) cover 4000×3000 RGB8
images 1029, 1039 and 1049 with one CPU participant. Across the six large shared
calls, mean end-to-end time was 63.95 seconds and feature preparation was
16.85 seconds (26.35%). Each large call prepared 25 layouts, producing 900
million weighted features. These are local pilot measurements with concurrent
system activity, not a fresh benchmark of this branch.

The 16.85-second counter includes allocation, initialization, copying transformed
planes, scalar predictor scans and completed-host-view publication. It does
**not** isolate the recurrence. Step 1 adds that missing attribution. A 4×
improvement to the *entire* preparation interval would project about 51.31
seconds end to end (1.25×); removing that interval entirely projects 47.10
seconds (1.36×). These are arithmetic projections for the saved workload, not
speedup claims. Learner improvements will change the relative proportions.

## Operation and storage contract

The reusable CPU operation lives in
[`weighted_features.h`](../src/codec/modular/weighted_features.h) and
[`weighted_features.cpp`](../src/codec/modular/weighted_features.cpp):

```cpp
Status PrepareWeightedPredictorFeatures(
    ModularChannelView slice,
    const WeightedPredictorParameters& parameters,
    std::span<WeightedPredictorFeature> output,
    FeaturePreparationProfile* profile = nullptr);
```

One call produces one independent canonical channel slice. Its input is a
borrowed transformed plane view starting at the slice rectangle's origin, with
the original plane stride and the slice's width/height. Input remains immutable
and must not alias output. Weighted parameters are explicit and part of cache
identity; no tree or learning configuration is needed.

The output span has exactly `width * height` entries. Each entry is two signed
64-bit integers, `prediction` followed by `property15`, preserving the existing
16-byte CPU/Metal ABI. Entries are slice raster order; all neighborhood edges
and predictor state resets are slice-local. No sample reduction, quantization,
floating-point arithmetic or history approximation is allowed.

The enclosing adapter continues to use `ValidateAdaptiveModularSlices` and
`VisitAdaptiveModularSlices` to validate and enumerate canonical
stream/channel/slice order. It supplies disjoint output subspans at the existing
`token_offset`s. The scalar primitive intentionally does not know frame geometry,
stream IDs, transformed-channel ordering, the learner or Metal. Keeping the
unit at a slice also permits a future executor to schedule the same jobs without
materializing another full-image map or changing the consumers.

`ComputeWeightedPredictorScratchBound(maximum_width, participants, out)` is an
allocation-free, checked, failure-atomic bound shared with adaptive GPU workflow
admission. Each participant needs four `uint32_t` and one `int32_t` two-row
arrays, each with `2 * (maximum_width + 2)` entries: a total of
`40 * (maximum_width + 2)` bytes per participant. The bound excludes borrowed
input, the caller-owned 16-byte-per-sample output, scheduler storage and object
headers. It accepts 1–256 participants and does not itself launch workers.
The batch operation uses the enclosing workflow's admitted ceiling, currently
at most four participants.

The enclosing operation still owns the plane arena, weighted arena and canonical
descriptor copy. It writes weighted features directly into the shared buffer;
there is no extra full-image staging allocation. The existing conservative
bridge bound retains its conservative full-map staging allowance and
widest-transformed-channel scratch bound, with the additional batch execution
and GPU dispatch storage described below. A completed owner is retained by the single-entry cache and
borrowed read-only by learning and GPU tokenization.

Validation of shape, parameters and output count precedes scratch allocation
and stores. Scratch is charged to `kPreparation` under the ambient reservation
and released before return. Unsupported numerical state may leave a partial
slice; the enclosing unpublished owner must be discarded. Publish only after
every slice and completed-host-view acquisition succeeds. Preserve the previous
published output on failure. There is no retry after execution begins.

## Implementation sequence

### 1. Extract the serial operation and expose its costs

Completed in commit `fdeee05`:

- Extract the serial feature scan into the CPU-neutral operation above, using
  the existing `WeightedPredictor` arithmetic and slice-local neighbors.
- Route the Metal shared-buffer producer through it without changing canonical
  layout, ownership, feature reuse or token-generation dispatch.
- Share the checked scratch bound with the existing adaptive storage planner.
- Add disjoint preparation timers, scalar contract tests and integration checks
  that features, encoded bytes and resource/failure behavior are preserved.

No CPU concurrency, wavefront arithmetic or Metal feature kernel is introduced
in Step 1. This keeps its contract usable by the independently changing learner.

### 2. Parallelize independent slices on the CPU

Implemented after the Step 1 commit `fdeee05`. `WeightedFeatureSlice` describes
a borrowed view and its canonical output offset. The batch operation validates
all views and exact contiguous output coverage before allocating or writing.
The original single-slice entry point remains the serial reference.

Whole-workflow admission passes its resolved participant ceiling (at most four)
through the feature cache into the backend. The operation can use fewer
participants if there are fewer slices, the CPU domain grants fewer workers,
an explicit parallel scope is already active, or no CPU execution scope exists.
That last case stays serial. Predictor backing is allocated before worker launch;
`ResetForWidth` resets the actual slice width without allocation. The serial
batch also reuses one predictor across slices.

`ComputeWeightedFeatureBatchStoragePlan` separates caller-owned job descriptors
from execution scratch. Execution covers five row arrays per participant,
predictor records, indexed statuses and thread records. Both operation and
whole-workflow bridge bounds include these allocations, retaining the previous
conservative feature staging allowance. A managed allocation failure before
launch cannot write features. Partial launch and numerical failures join all
workers and return without retry; the completed owner remains unpublished.

The following requirements guided the implementation:

Use the existing worker budget and caller-participating worker group. Resolve
the participant count before admission, allocate worker scratch before launch,
propagate resource context and assign each job a disjoint output range. Reuse
each worker's row state with an exact reset and the slice's actual width; simply
allocating a wider predictor changes boundary indexing and is insufficient.

Update both whole-workflow and operation storage plans to cover every active
worker, job/status storage and launch lifetime. Join all work before publication
or failure return; choose errors in canonical job order and do not retry after
work starts. Workers must not write a shared profiling sink. Keep joined caller
wall time distinct from any optional summed worker time.

Qualify one participant against the Step 1 reference, then two and four. Small
layouts may stay serial based on pre-execution eligibility. An ordinary
4000×3000 RGB layout has 16×12 groups and three channels, or 576 independent
slice jobs; palette metadata, Squeeze and DC/global slices require enumeration
from the actual transformed layout rather than this formula.

### 3. Prove a wavefront form against the scalar recurrence

Implemented in [`weighted_feature_math.h`](../src/codec/modular/weighted_feature_math.h),
with a CPU proof in
[`modular_weighted_wavefront_test.cpp`](../tests/modular_weighted_wavefront_test.cpp).
The scalar predictor mutates two row buffers while scanning left to right.
The wavefront uses immutable per-pixel state; it does not reorder calls to the
mutable predictor.

For each pixel retain its combined prediction error `E` and four subpredictor
errors `A[i]`. The scalar `Update` both stores `A[i](x,y)` in the current row and
adds it to the previous row at `x+1`, with unsigned 32-bit wrapping. Consequently,
the north error accumulator observed by the next pixel contains a west-pixel
contribution, and the northwest accumulator contains a two-pixels-west
contribution. Reconstruct these sums explicitly, including boundary duplication.

The dependency set is:

| State needed | Earlier pixels |
| --- | --- |
| Combined error | W, N, NW, NE |
| Subpredictor errors, including row accumulation | W, WW, N, NW, NE |
| Neighborhood pixel values | Already present in the immutable transformed plane |

Use wavefront index `t = x + 2*y`. W/NE are one step earlier, WW/N two steps
earlier and NW three steps earlier. The ordinary `x+y` diagonal would leave NE
on the same step. Example indices are:

```text
0 1 2 3 4 5
2 3 4 5 6 7
4 5 6 7 8 9
```

The tests check exact 64-bit intermediate arithmetic, signed shifts/rounding, unsigned
32-bit additions, reciprocal lookup, clamping and property-15 tie order
(W, N, NW, NE, replacing only for strictly larger magnitude). They include first/last
rows and columns, width/height one, odd and narrow shapes, custom parameters,
supported extremes and Unsupported range failures. Across 312 cases, both a
full-image error-state implementation and the four-bank rolling implementation
matched 4,187,195 scalar features and 67 unsupported-range results. Four parameter
sets include all-zero and maximum coefficients; another 32 deterministic
parameter sets exercise mixed headers. Separate checks cover unsigned wrapping,
reciprocal entries and property-15 ties.

### 4. Implement Metal wavefront preparation for common slices

Implemented in the `gjxl_modular_weighted_features` kernel with slices at most
256×256, one 128-thread group per slice. For a full 256×256
slice the schedule has 766 steps, up to 128 simultaneously active pixels and
about 86 pixels per step on average. Four rolling step banks, indexed by row,
retain one `int32_t` combined error and four `uint32_t` suberrors each:
`4 * 256 * 20 = 20,480` bytes of threadgroup state. Backend setup checks the device's
threadgroup-memory limit, thread dimensions, pipeline thread ceiling and static
threadgroup allocation. If those checks fail, feature preparation uses the CPU.
Output remains the same 16-byte map; no full-image error-state allocation is needed.

All threads, including inactive edge lanes, execute every wavefront barrier.
Numerical failure sets an atomic status while keeping the barrier schedule;
the caller waits or drains submitted work before returning without publication.
Unsupported shapes select CPU before dispatch. GPU failure after dispatch is
terminal, with no CPU retry. A host-validated descriptor/status buffer is the
only additional GPU dispatch allocation, included in operation/workflow admission.

The completed shared-buffer owner lets CPU learning and GPU tokenization consume
the same results without a staging copy or explicit GPU readback. CPU learning
and entropy modeling remain on their existing paths.

Private workflow selection defaults to `WeightedFeatureBackend::kAuto`. After
shape and device checks, it chooses Metal for at least 4,194,304 feature samples
with an average of at least 16,384 samples per slice. Other layouts use the
admitted CPU participants. This conservative crossover was qualified against
four CPU participants on the M4 Pro; it is not a universal device performance
guarantee. Direct operation calls still default to the scalar CPU route, keeping
an explicit reference producer. `kGpu` bypasses only the size crossover, not
shape/device eligibility or failure handling.

### 5. Qualify correctness and performance

Compare serial CPU, bounded parallel CPU and Metal on the same prepared input,
then compare complete encodes with matching policies. Cover all six packed
formats, transformed/global/DC/metadata layouts, palette/Squeeze, odd and narrow
edges, custom weighted parameters and fault injection. Require exact feature
maps, token values/contexts, candidate/learned policies and complete-file bytes,
with decode verification.

Report allocations, copies, dispatch and completion waits as part of the
operation; retain recurrence detail separately. Use balanced repeated captures,
record device/revision/configuration and preserve raw results. GPU preparation
must be compared with the improved parallel CPU producer, not only the old
serial loop. Change automatic routing only after this qualification supports it.

## Profiling and Step 1 validation

`gjxl_modular_search_profile` now emits
`adaptive_gpu.feature_preparation` with five accumulated caller-wall counters:

| Counter | Measured interval |
| --- | --- |
| `allocation_seconds` | Completed owner/descriptors and plane/weighted buffers |
| `copy_seconds` | Plane arena zeroing and transformed-plane copies |
| `scratch_seconds` | Worker predictor records, row allocation and zero initialization |
| `prediction_seconds` | Joined batch interval: state resets, scheduler setup, launch, predictor work and join |
| `publication_seconds` | Completed host view and final owner publication |

These disjoint counters are subsets of the existing `feature_seconds` total.
Validation, slice traversal, scratch destruction and instrumentation overhead
remain in the parent remainder. `learning_feature_seconds` remains the subset
requested during training; the rest belongs to token preparation. Do not add
the five counters or learning subset to the enclosing total. Failed attempts
can also accumulate elapsed work. With no profile sink there are no clock reads.
CPU-only encoding leaves these adaptive preparation counters at zero because
that route still computes history inside its existing learner/tokenizer.

Step 2 also reports `slice_jobs`, `parallel_calls` and `peak_participants` in
`feature_preparation`. These are work counts, not seconds. Only the caller
updates the sink; no summed worker time is mixed into the wall intervals.

The Metal producer also reports `gpu_calls`, `gpu_samples` and
`cpu_fallback_calls`. For Metal, `prediction_seconds` includes command encoding,
submission and completion wait, and `scratch_seconds` is zero. Dispatch-buffer
allocation/copy is included in the corresponding counters. A fallback count
means that Auto/GPU selected the CPU before dispatch, including small layouts;
it never means retrying failed GPU work.

New scalar tests cover strided subviews, guarded output spans, state reset,
custom parameters, exact output-size validation, scratch overflow and all five
allocation-failure boundaries. Existing adaptive Metal tests provide canonical
layout, feature/token comparison, cache, completed-file, decode and publication
failure coverage. Workflow profiling checks require nonoverlapping detail
inside the enclosing preparation interval. Saved pilot JSON is unchanged; it
cannot supply these newly introduced counters retroactively.

Validated on the Apple M4 Pro with Release (`-O3 -g -DNDEBUG`) Metal-enabled
and CPU-only builds, using the pinned libjxl oracle at
`e8ff09762481785938d8e4e01333ed3917571161`:

- All seven selected Metal-build tests passed: `modular_weighted_features`,
  `modular_adaptive_token_plan`, `modular_profile`, `modular_candidate_search`,
  `metal_modular_adaptive`, `metal_modular_adaptive_workflow` and
  `metal_modular_shared_features`.
- All four corresponding CPU-only tests passed: the four `modular_` tests above.
  The new contract test is also included in the native CI Modular gate.
- The Metal operation test checked 2,831,887 weighted features and 3,778,462
  tokens, plus 34 identical complete files. The workflow test checked 55
  complete CPU/hybrid matches, 26,119,113 independently decoded samples and
  7,732 failure-atomic cases. Shared-cache sample fingerprints, learned policies,
  checkpoints, cache keys and borrow failures also matched.
- Six profiler smoke calls on a 33×19 RGB fixture matched complete files and
  every candidate decision: all three Metal routes, profiling disabled, CPU in
  the Metal build and CPU-only. The five new counters were positive and nested
  for profiled Metal calls, and zero for the other calls. Local outputs and JSON
  are retained under `build/weighted-feature-step1/`.

These are correctness and instrumentation checks. No new large-image
performance qualification has been performed for Step 1.

Step 2 adds `modular_weighted_feature_batch` to native CI. Its cases compare
one/two/four participants against the single-slice reference, exercise mixed
widths and strided views, restrict domain capacity and nested execution, reject
malformed late jobs before any writes, sweep all 23 four-participant managed
allocations, and inject partial thread-launch failures. The Metal workflow
also injects those launch failures and checks unchanged public output and
released memory/CPU resources. A 257×33 RGB profiler fixture produces identical
complete bytes and candidate ledgers with one/two/four participants, with the
expected participant counters. Local checks are under
`build/weighted-feature-step2/` and the `step2-*.log` build logs. They are
correctness checks, not a speedup qualification.

## Final qualification

The final source passed all 11 selected Metal-build tests and all 10 CPU-only
tests. The CPU-only set also exercises the foundation, training budgets,
checkpoints and sampling coverage, since predictor scratch reuse changes a
shared helper. Both new CPU tests are in the native Modular CI gate.

The Metal operation matrix runs with both producers: 203 cases, all six packed
formats, odd/narrow edges, custom parameters, transformed/global/DC layouts,
palette and Squeeze. Each run compares 2,831,887 weighted features, 3,778,462
tokens and 34 identical complete files, with independent sample-exact decoding.
The whole-workflow matrix runs with both producers too: each has 55 complete
CPU/hybrid matches, 26,119,113 decoded samples and 7,711 failure-atomic checks.
Additional tests cover one/two/four CPU participants, nested/domain limits,
partial thread launch, GPU plane/feature/dispatch allocation boundaries,
submission/completion failure, unsupported numerical state, and Auto routing
below and at its large-layout crossover. Failed work preserves the previous
owner and releases admitted resources.

Metal shader validation was attempted separately. With instrumentation enabled,
the optional feature pipeline failed its threadgroup resource eligibility checks,
so that additional test returned skip (77). The normal GPU suite passed. This
report does not claim validation of the instrumented kernel or measured hardware
occupancy. Logs are under `build/release/final-tests.log`,
`build/cpu/final-tests.log` and `build/release/shader-validation.log`.

### Prepared-input operation

[`gjxl_modular_feature_profile`](../tools/modular_training_study/weighted_features.cpp)
compares CPU budgets of one, two and four participants with forced GPU production
on one prepared frame. Input reading/transforms, backend creation and a retained
serial reference are outside the timer. Allocation, plane copies, CPU scheduling
or GPU dispatch/wait, and publication are inside. Every returned feature is
compared with the reference after timing; local reservation and CPU-domain
accounting are checked after each trial. The reference and source frame consume
additional memory outside the per-operation reservation.

The 27 input/layout cases cover RGB size/aspect fixtures, a gray threshold fixture,
the three original 4000×3000 photographs with raw/RCT/Squeeze layouts, a 500×500
photograph, palette metadata and RGBA16 transforms. Each case has one warm-up per
producer and five measured rounds, reversing producer order between rounds.
That is 540 measured trials plus 108 warm-ups, all exact. RCT operation medians:

| Image | CPU budget 1 | CPU budget 2 | CPU budget 4 | Metal | Metal vs CPU 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1029 | 469.16 ms | 243.05 ms | 166.03 ms | 50.79 ms | 3.27× |
| 1039 | 458.70 ms | 241.51 ms | 163.81 ms | 48.97 ms | 3.34× |
| 1049 | 471.26 ms | 246.10 ms | 169.26 ms | 51.12 ms | 3.31× |

Across all nine large raw/RCT/Squeeze cases, Metal is 3.14–3.39× faster than CPU
budget 4. Small layouts lose: for example, 512×512 RGB takes 3.75 ms on CPU 4
versus 6.35 ms on Metal. At the accepted crossover, 2048×2048 gray improves from
17.15 to 10.20 ms, and the large 64-pixel-wide/tall cases improve by 2.54–2.71×.
All 14 measured cases accepted by Auto's size/average-slice gate favor Metal;
some rejected cases also favor Metal. The threshold deliberately leaves those
marginal or narrow cases on CPU. It uses transformed feature samples, not
original image pixels.

### Complete encoder calls

These are hybrid encoder calls using the same original learner, 262,144-sample
budget, 256-leaf cap, checkpoints, shared features, estimated ANS selection,
measure-all candidate evaluation and GPU statistics/populations. Only the CPU
budget and feature producer change. CPU 1 is one reference call per image;
CPU 4 and Auto 4 are medians of two alternating paired calls per image.
Auto 4 uses the new workflow default without an explicit feature-selection flag.
Backend setup, input/output file access and pinned-libjxl decoding are outside
the complete encoder-call timer; candidate-ledger serialization is inside it.

| Image | Total, CPU 1 | Total, CPU 4 | Total, Auto 4 | Preparation, CPU 1 / CPU 4 / Auto 4 |
| --- | ---: | ---: | ---: | ---: |
| 1029 | 45.67 s | 37.49 s | 34.57 s | 12.002 / 4.199 / 1.092 s |
| 1039 | 43.65 s | 35.91 s | 32.94 s | 11.740 / 4.142 / 1.081 s |
| 1049 | 43.72 s | 36.27 s | 33.22 s | 11.776 / 4.162 / 1.083 s |

Holding the four-participant budget fixed, automatic GPU preparation reduces
the preparation interval by 3.83–3.85× and complete encoder time by 7.78–8.41%.
Each call prepares 25 feature layouts, totaling 900 million samples. Auto uses
the GPU for all 25 and performs no feature fallback on these images. Preparation
is now roughly 3.2–3.3% of the complete call. Tree learning remains the largest
stage; its implementation was not changed by this work.

All 15 complete calls decode exactly, stay within memory/CPU admission, and
produce identical files, full candidate ledgers (excluding elapsed time) and
training work counts for each image. File sizes remain 10,821,744 bytes for 1029,
7,230,909 for 1039 and 7,336,102 for 1049. Retained feature-map storage is unchanged
at 16 bytes per transformed sample. The GPU adds descriptor/status dispatch
storage and 20 KiB of threadgroup state, without another full-image feature or
error map. Existing conservative workflow admission is retained.

### Evidence and reproduction

The [machine-readable results](parallel-weighted-feature-preparation-results.json)
contain per-case medians/ranges, nested timings, actual/planned storage, input
hashes, encoded-file hashes and provenance. Frozen binaries, exact base-`fdeee05`
source patches including new files, CMake caches, commands, raw outputs and
power/thermal/process snapshots are retained under
`build/weighted-feature-qualification/{micro,crossover,encodes}/`. Each directory
contains its manifest and capture script. The scripts audit complete outputs
and fail on feature, policy, byte or admission differences.

Measurements used the Apple M4 Pro (14 CPU cores, 48 GiB), Release
`-O3 -g -DNDEBUG`, macOS 27.0.1 and AC power. No thermal warning was reported.
The desktop was active, including App Store/indexing work, so these are local
qualification results rather than isolated-machine throughput claims. They do
not establish performance on other devices or a new comparison with libjxl.
The older shared-feature pilot used different conditions; compare the fresh
paired modes above instead of subtracting from its saved times.

```sh
cmake --build build/release --parallel 4 --target \
  gjxl_modular_feature_profile gjxl_modular_search_profile
build/release/gjxl_modular_feature_profile IMAGE.pam 5 rct
build/release/gjxl_modular_search_profile IMAGE.pam cpu.jxl 262144 256 4 \
  --metal-learn --estimated-ans --measure-all --metal-populations \
  --candidate-ledger --shared-features --cpu-features
build/release/gjxl_modular_search_profile IMAGE.pam auto.jxl 262144 256 4 \
  --metal-learn --estimated-ans --measure-all --metal-populations \
  --candidate-ledger --shared-features
```
