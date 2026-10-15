# Shared weighted features for the histogram learner

Status: Steps 1–4 complete. Targeted correctness and repeated matched
performance qualification passed on 2026-10-14; see
[the qualification report](gpu-histogram-learner-qualification.md).
Work is on `feat/parallel-weighted-feature-prep`,
starting from `81467ca` (the public learner integration). The earlier
[weighted-feature effort](parallel-weighted-feature-preparation.md) is complete
for legacy search. Its scalar oracle, bounded CPU producer, Metal wavefront
math and qualification are reusable; its performance results do not qualify
the histogram workflow.

## Required outcome

The improved one-fit histogram workflow must be able to prepare weighted
features on Metal, share the completed features with native sample preparation,
and reuse them for GPU tokenization. Preserve its selected transforms and WP
headers, sampled rows, learned trees, local/shared entropy scopes, direct ANS
policy, and complete encoded bytes. The existing CPU workflow is the exact
control. Sampling, quantization, deduplication, histogram fitting, transform/WP
selection and entropy coding stay on the CPU in this effort.

This is a stream integration, not a return to the legacy candidate portfolio.
The new workflow chooses transforms and WP parameters per stream, uses a shared
tree through eight pixel groups, and local trees above that threshold. Its
memory admission is stagewise and includes backing retained by earlier stages.

## Implementation sequence

1. Define completed channel-feature borrows and teach native sample preparation
   and CPU tokenization to consume them. Prove exact samples, trees, tokens and
   bytes, along with validation, storage bounds and atomic publication.
2. Add a checked batch operation for already transformed, independently owned
   stream channels. Support per-stream WP parameters and trees, preserving the
   existing slice-local wavefront and token arithmetic. Provide bounded CPU
   preparation and Metal preparation/generation through the same contract.
3. Integrate batches into the histogram workflow after selection. Reuse each
   completed feature map through learning and tokenization, then release it.
   Preserve canonical stream IDs, channel ordinals and local/global learning
   boundaries. Add explicit routing/counters and a reproducible comparison
   driver; do not infer GPU execution merely from backend availability.
4. Qualify the complete CPU/hybrid workflow with targeted correctness and
   failure checks, independent decoding, and matched repeated timings. Record
   feature preparation, sample preparation/fitting, tokenization, complete-call
   time, output size, route counts and managed memory. Revisit automatic
   crossover using the new workload before claiming a speedup or enabling it
   by default.

## Feature borrow contract

`WeightedFeatureView` borrows one completed channel-local raster map of
`{int64 prediction, int64 property15}`: 16 bytes per sample. It records the
original channel view and selected WP parameters. Input identity includes the
first pixel address, descriptor and stride. Both sample backing and completed
feature backing remain immutable and live through the last consumer. Address
identity cannot detect mutation or address reuse; the owner must enforce that
lifetime. Views cannot be persisted across encodes.

An absent channel-map array selects scalar WP. A provided array has exactly
one entry per channel, including metadata/empty channels; an empty entry
selects scalar WP for that channel. Every nonempty entry must match the source,
parameters and exact channel area before allocating or publishing anything.
Views contain trusted results from a successfully completed producer; consumers
do not recompute the recurrence to authenticate feature contents.

Predictor state resets at each channel boundary. Sampling still visits pixels
in the original order, draws the same RNG values and builds identical columns,
quantization cuts and weighted-row deduplication. Cached features replace only
the recurrence. Global learning's excluded channels and canonical stream-ID
holes retain their existing semantics. A local learning scope still has its
original local property numbering; tokenization uses the actual coding IDs.

Consumers own neither map nor source and exclude both from their fresh-storage
bounds. They reserve WP row scratch only for uncovered channels. The enclosing
workflow charges feature owners and all retained source/tree/token backing
exactly once. Failed validation leaves planners and outputs unchanged; failed
execution discards unpublished output. There is no CPU retry after a failed
GPU submission.

## Batch and ownership requirements

Selected parameters are available only after stream selection. Batch independent
selected channels; never extend a predictor recurrence across stream boundaries.
The current legacy automatic crossover is 4,194,304 samples and an average
slice area of at least 16,384, and its wavefront supports at most 256 by 256.
Dispatching once per small group would miss that crossover. Batch admission
must bound feature/plane/descriptor backing and kernel limits before dispatch.
Larger metadata channels use an explicitly selected CPU route.

Feature lifetime spans both learning and tokenization. Local trees allow batches
to finish these consumers before releasing maps. A shared tree requires all
its training samples before fitting; define that scope's lifetime explicitly.
Avoid retaining a full large-image map merely because the old whole-frame GPU
owner did so. Retained tokens may outlive their feature maps and retain stream
boundaries for the unchanged CPU entropy continuation.

`GpuModularStreamProcessing` now accepts up to 1,024 nonempty independent
channel slices, with explicit original stream/channel IDs and per-slice WP
parameters. Its pixel-read-free storage planner validates backing, headers,
checked sizes and backend limits. One packed plane arena, one immutable feature
arena and the original-source feature views survive preparation. Temporary
dispatch/CPU worker backing is released before publication. The CPU producer
resets both width and parameters on reused worker scratch.

Token generation takes a complete ordered partition of the slices into tree
bindings. The Metal dispatch packs those trees once per binding and keeps child
indices and entropy contexts local to each tree. The caller still validates
image-dependent decoder bounds and local/shared model scopes. The resulting
split value/context arena owns its charge independently of feature maps.

Both preparation and generation expose fresh-storage plans that exclude old
output and borrowed source/tree backing. Callers reserve each stage alongside
all live retained backing. Mixed or oversized wavefront shapes choose the CPU
feature producer before work; the workflow should separate those small streams
from ordinary GPU batches. Errors after execution begins remain terminal.

## Integrated workflow

`ReferenceWorkflowOptions::gpu` enables the private hybrid route. Null keeps
the ordinary CPU control. The same selection and fitting functions serve both
routes. Public `ModularBackend::kMetal` is still unsupported; this change does
not promote automatic GPU use in the public encoder.

After selection, the workflow batches whole streams, up to 8,388,608 transformed
channel samples and 1,024 nonempty slices. It separates oversized metadata from
ordinary wavefront shapes. Each local batch prepares maps, fits its local trees
once, generates GPU tokens using those same maps, and releases its feature owner
before preparing the next batch. Shared/global learning retains all batches in
its scope through the one fit, then tokenizes/releases them. Explicit buffered
mode can therefore retain more than one batch; every live batch is charged.

`PreparedModularStreamTokenView` binds completed token backing to its immutable
source channels, selected WP parameters, coding ID and tree identity. The
prepared-stream planner checks those identities and token shape before
allocation. The unchanged CPU entropy continuation borrows tokens directly;
it still generates tree-header tokens and fresh tokens for uncovered streams.
The frame owns GPU tokens and stable source/tree allocations. Moves preserve
their addresses, and no feature map survives frame preparation.

Counters report actual CPU/GPU producers, generated tokens, reused/scalar
learning features and the live feature peak. Nested wall intervals separate
feature preparation, native sample preparation, fitting and GPU tokens. The
enclosing profile retains complete training/token stage attribution.

## Qualification gates

- Compare scalar and shared-map preparation with mixed WP headers, strided
  views, metadata, empty channels/group holes, partial coverage, all predictors,
  and exact learned trees; reject stale/malformed borrows before allocation.
- Compare CPU and Metal features/tokens for mixed streams and trees, edge
  dimensions and arithmetic failures; keep legacy operation checks passing.
- Compare complete files and independent decoded samples for integer formats,
  local/shared scopes (especially eight versus nine groups), palette/RCT/WP
  decisions, and prefix/direct-ANS modes.
- Exercise new allocation/admission/publication boundaries with small fixtures
  and bounded failure injection. Do not rerun the exhaustive legacy candidate
  allocation sweep as a substitute for testing this integration.
- Benchmark the same improved policy on the same inputs/revision/build/device,
  with explicit CPU and hybrid routes. Save raw repeated measurements and
  environmental conditions. A kernel win alone is not an end-to-end win.

## Evidence

Step 1, Release CPU-only build with the pinned decoder oracle:

- `modular_shared_feature_consumers`: 64 exact sample/tree comparisons,
  cached channel tokenization without WP allocation, six complete local/shared
  prefix/ANS file comparisons, metadata/exclusion cases, malformed-view
  rejection, allocation failure publication and admitted storage checks.
- `modular_histogram_preparation`, `modular_weighted_features`, and
  `modular_reference_workflow`: passed, including the existing complete workflow
  reference decoding and its small-fixture allocation checks.

Commands: `cmake --build build/cpu --parallel 6 --target
gjxl_modular_shared_feature_consumers_test gjxl_modular_histogram_preparation_test
gjxl_modular_weighted_features_test gjxl_modular_reference_workflow_test`, followed
by CTest restricted to the four named checks. The exhaustive legacy
`modular_workflow` allocation sweep was not run.

Step 2, Release CPU-only and Metal builds:

- CPU: `modular_stream_batch`, `modular_shared_feature_consumers`, and
  `modular_weighted_feature_batch` passed.
- Metal: `modular_stream_batch`, `metal_modular_stream_batch`,
  `modular_weighted_feature_batch`, `metal_modular_adaptive`, and
  `metal_modular_weighted_features` passed.
- New coverage includes mixed headers with 1/2/4 CPU participants; six Metal
  batch cases comparing 999,744 feature/token entries in total; a confirmed GPU
  wavefront batch; larger metadata and small automatic CPU routes; independent
  trees using all predictors and coding IDs; allocation/submission/completion
  failure publication; undersized admission; arithmetic failures; and rejection
  of an owner from another backend instance. Existing adaptive/wavefront tests
  cover legacy ABI compatibility after the shared kernel changes.
- CTest logs are retained locally under
  `build/histogram-gpu-integration/step2-{cpu,metal}-ctest.log`.

Step 3, Release CPU-only and Metal builds with the pinned decoder oracle:

- `modular_reference_hybrid`: 54 CPU/hybrid identical files, 21,686,132
  independently decoded samples, 25 actual GPU feature batches and 159 injected
  failures. Covers gray/RGB/RGBA, 8/16-bit inputs, both byte orders, invisible
  alpha, shared/local trees at eight/nine groups, all predictors, forced and
  automatic producer routes, multiple local batches, frame moves, detached
  packed input, direct ANS and prefix coding. Completed in 87.51 seconds.
- `modular_reference_hybrid_palette`: two additional identical files and
  1,580,550 decoded samples. A selected 600-color palette exercises CPU feature
  preparation for metadata wider than 256 alongside GPU pixel-group preparation.
- `modular_shared_feature_consumers`: additional completed-token borrowing
  checks prove no copy or double charge, exact complete bytes, malformed
  identity rejection before allocation, and unchanged prior outputs on failure.
- CPU `modular_prepared_stream` and `modular_reference_workflow` and Metal
  `metal_modular_stream_batch` passed. The CPU workflow retains its 98 exact
  decoded files and small-fixture allocation coverage. The exhaustive legacy
  `modular_workflow` sweep was not run.
- Logs are retained under `build/histogram-gpu-integration/step3-*`.
- AddressSanitizer/UndefinedBehaviorSanitizer CPU build: the shared-feature and
  completed-token consumer checks, stream batch checks, and reference workflow
  passed (43.35 seconds total; `UBSAN_OPTIONS=halt_on_error=1`, ASan leak
  detection disabled). This supplements the Metal runtime failure/lifetime
  tests; it does not instrument GPU shader execution.

The matched complete-call driver and freezing runner are documented in
[the tool README](../tools/modular_reference_learning/README.md#cpu-hybrid-histogram-qualification).
Step 4, matched Release CPU/hybrid qualification on Apple M4 Pro:

- Three original 4000×3000 RGB8 photographs, one warmup and three measured
  encodes per route/image: 24 identical files with exact independent decoding.
  Median complete-call runtime improves by 5.6–6.7% using the same learned
  policy and output bytes.
- Two center-crop controls compare ordinary CPU, shared CPU features with
  Metal tokens, forced GPU features with Metal tokens, and automatic hybrid
  routing. All 32 additional files are identical and decode exactly. The
  existing conservative feature crossover is retained.
- Feature preparation is 0.11–0.15 seconds on the full-resolution images;
  most of the observed gain comes from GPU token generation. CPU fitting
  remains 69–73% of hybrid runtime. Public default routing is unchanged.
- [Raw measurements and provenance](gpu-histogram-learner-integration-results.json)
  preserve actual routing, timing boundaries, repeats, exact outputs and
  memory admission. Development pilots are excluded. These are local battery
  measurements, separate from earlier AC-powered legacy-search results.
