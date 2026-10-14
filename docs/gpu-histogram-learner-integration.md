# Shared weighted features for the histogram learner

Status: Step 1 complete; batch/workflow integration and qualification remain.
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

No new GPU workflow or performance result is claimed at this stage.
