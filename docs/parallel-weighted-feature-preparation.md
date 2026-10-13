# Parallel weighted-prediction feature preparation

Status: Step 1 implements the reusable scalar operation and preparation
attribution. Steps 2–5 are planned; production remains serial within feature
preparation. This work starts from shared-feature commit `23cedb1` on
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
headers. It accepts 1–256 participants; this does not launch workers or change
the current encoder's one-participant preparation.

The enclosing operation still owns the plane arena, weighted arena and canonical
descriptor copy. It writes weighted features directly into the shared buffer;
there is no extra full-image staging allocation. The existing conservative
bridge bound retains its full-map staging allowance and widest-transformed-
channel scratch bound. Step 1 changes neither admission totals nor allocation
counts. A completed owner is still retained by the single-entry cache and
borrowed read-only by learning and GPU tokenization.

Validation of shape, parameters and output count precedes scratch allocation
and stores. Scratch is charged to `kPreparation` under the ambient reservation
and released before return. Unsupported numerical state may leave a partial
slice; the enclosing unpublished owner must be discarded. Publish only after
every slice and completed-host-view acquisition succeeds. Preserve the previous
published output on failure. There is no retry after execution begins.

## Implementation sequence

### 1. Extract the serial operation and expose its costs

Completed in this slice:

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

The current predictor mutates two row buffers while scanning left to right.
Reordering calls to that class is incorrect. First express it as immutable
per-pixel state and validate an experimental CPU implementation against Step 1.

For each pixel retain its combined prediction error `E` and four subpredictor
errors `A[i]`. The scalar `Update` both stores `A[i](x,y)` in the current row and
adds it to the previous row at `x+1`, with unsigned 32-bit wrapping. Consequently,
the north error accumulator observed by the next pixel contains a west-pixel
contribution, and the northwest accumulator contains a two-pixels-west
contribution. Reconstruct these sums explicitly, including boundary duplication.

The proposed dependency set is:

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

Prove exact 64-bit intermediate arithmetic, signed shifts/rounding, unsigned
32-bit additions, reciprocal lookup, clamping and property-15 tie order
(W, N, NW, NE, replacing only for strictly larger magnitude). Include first/last
rows and columns, width/height one, odd and narrow shapes, custom parameters,
supported extremes and Unsupported range failures. This is a proposed
reformulation; Step 1 does not yet establish its equivalence.

### 4. Implement Metal wavefront preparation for common slices

Begin with slices at most 256×256, one threadgroup per slice. For a full 256×256
slice the schedule has 766 steps, up to 128 simultaneously active pixels and
about 86 pixels per step on average. Four rolling step banks, indexed by row,
would retain one `int32_t` combined error and four `uint32_t` suberrors each:
`4 * 256 * 20 = 20,480` bytes of threadgroup state. This is a preliminary storage
design; verify complete scratch use, device limits, register pressure and
occupancy before admission or dispatch. Output remains the same 16-byte map.

All participating threads, including inactive edge lanes, must execute every
required barrier. Record range failure without divergent early barrier exits.
Complete or drain all submitted work before returning. Select a CPU route for
unsupported shapes before dispatch; GPU failure after dispatch is terminal.

Preserve the completed shared-buffer owner so CPU learning and GPU tokenization
consume exactly the same results without a staging copy or GPU readback.

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
| `scratch_seconds` | Scalar predictor row allocation and zero initialization |
| `prediction_seconds` | Neighborhood loads, Predict/Update and feature stores |
| `publication_seconds` | Completed host view and final owner publication |

These disjoint counters are subsets of the existing `feature_seconds` total.
Validation, slice traversal, scratch destruction and instrumentation overhead
remain in the parent remainder. `learning_feature_seconds` remains the subset
requested during training; the rest belongs to token preparation. Do not add
the five counters or learning subset to the enclosing total. Failed attempts
can also accumulate elapsed work. With no profile sink there are no clock reads.
CPU-only encoding leaves these adaptive preparation counters at zero because
that route still computes history inside its existing learner/tokenizer.

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
