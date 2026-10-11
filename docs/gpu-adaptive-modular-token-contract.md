# Adaptive Modular token operation and storage contract

Status: private operation interfaces, ABI records, canonical descriptors, checked
storage planning, CPU feature production, Metal dispatch, and completed-token
handoff are implemented and device-qualified. Complete-search integration and
bounded reuse across candidates are also implemented and correctness-qualified;
the [repeated full-resolution qualification](gpu-adaptive-modular-ans.md) now
measures this path with exact GPU ANS costing. The existing
prescribed producer is unchanged; this contract is separate from its 31-node,
four-equal-plane ABI.

## Measured motivation

The first full-resolution control, 1029 at 262144 training samples and 256 leaves
with checkpoint selection, encoded 70 complete candidates. Its diagnostic caller
times were approximately 87 seconds for tokens, 68 for modeling, and 46 for
training. The final file used a 511-node tree and RCT 12. Optimizing the original
learner alone would leave most of that complete-call work on the CPU.

The next GPU operation therefore evaluates prescribed **learned** trees over
already transformed channels, including weighted leaves and property 15. The
CPU continues to choose candidates and serialize complete files. CPU transform
and weighted-feature generation form the initial bridge. The full search must
include those costs and every fallback.

## Operation boundary

There are two synchronous, failure-atomic operations:

1. Prepare immutable transformed features from a validated CPU frame, canonical
   stream plan, and weighted-predictor parameters. Copy channel samples into a
   resident plane arena. When weighted features are requested, evaluate the
   existing scalar predictor in canonical slice order and retain its prediction
   and property 15 for every sample. Publish a completed feature owner.
2. Generate compact tokens from that owner and one validated prescribed tree.
   Compile leaf contexts in scalar breadth-first order, then dispatch independent
   samples. Publish the existing completed-token abstraction after completion
   and device-status validation.

Neither operation learns a tree, selects a transform, or writes an entropy model.
The enclosing search retains the original candidate order, strict size comparison,
checkpoint tie behavior, complete entropy modeling, and output publication.

The feature owner is reusable across trees and checkpoint candidates with the
same transformed samples, stream layout, and weighted parameters. Its key excludes
the tree, entropy mode, and learning budget. The first cache holds one layout at a
time; eviction releases the old owner before constructing its replacement. It is
scoped to one encoder call and cannot identify inputs solely by an address across
calls. A failed replacement aborts the attempt without changing the caller's
previous compressed output.

## Eligibility and exactness

- All six existing packed formats remain accepted by the enclosing workflow.
  GPU evaluation receives transformed signed `int32` planes, rather than assuming
  unpack/RCT sample bounds or a fixed number of equal-sized channels.
- The CPU transform implementation remains authoritative initially, including
  palette metadata, Squeeze residual channels, shifts, and channel ordering.
- Trees use the CPU representation and image-dependent decoder limits: up to
  8192 leaves/16383 nodes, valid child indices, bounded depth, properties 0..15,
  and all 14 predictors. The existing prescribed token route keeps its own limit.
- Property 0 is the **stream-local channel ordinal**, not the transformed plane
  index. Properties 2/3 and every neighborhood boundary are local to the slice.
  Previous gradient resets each row; property 8 uses its unclamped value.
- Weighted state starts fresh for every slice and consumes every sample, even
  if the tree selects another predictor. A tree using property 15 requires the
  weighted maps even when none of its leaves is weighted.
- Feature generation must retain scalar overflow/error behavior. It cannot use
  only the selected training pixels or reset state at arbitrary tile boundaries.
- GPU neighborhood/property/predictor calculations use signed 64-bit
  intermediates for general transformed `int32` samples. Division truncates
  toward zero. Weighted features are signed 64-bit pairs; a narrower optional
  representation requires separate range validation and qualification.
- Leaf residuals apply the signed offset and positive multiplier exactly, check
  zero remainder and an `int32` quotient, then use unsigned signed-value packing.
  No floating-point cost or arithmetic substitutes are introduced.

Unsupported device limits or capability select a CPU path **before submission**,
with an explicit fallback count. Allocation, device, invalid-policy, and
representability failures are terminal; they are not silently retried on CPU.

## Device records

All host/device fields have fixed widths, explicit alignment, checked offsets,
and compile-time layout assertions. No host pointers or `size_t` enter the ABI.

- **Slice, 48 bytes:** `uint64 token_offset`, `uint64 plane_offset_elements`,
  `uint64 plane_stride_elements`; `uint32 stream_id`, `uint32 property_channel`,
  `uint32 width`, `uint32 height`; two reserved `uint32` fields initialized to zero.
  Plane offsets already include the slice origin. The source arena is immutable.
- **Weighted feature, 16 bytes:** `int64 prediction`, `int64 property15`, indexed
  by canonical token offset. The owner records whether this arena exists.
- **Node, 32 bytes:** reuse the existing fixed-width compiled node layout, with
  an active node count and dynamically sized backing. Leaf context fits `uint16`.
- **Output:** `uint32` packed residuals followed by an aligned `uint16` context
  array. Exactly one entry per sample, in canonical stream/slice/row-major order.

The canonical planner proves coverage and range arithmetic before allocation.
The backend validates native ownership, matching feature/layout identity, active
node count, descriptors, addressed ranges, and nonaliasing. Descriptor backing and
borrowed input outlive submission; no token view is exposed before the status word
is checked.

## Admission and lifetime

Let `T` be total transformed samples, `S` canonical slices, `C` transformed
channels, and `N` active tree nodes. Every multiplication, alignment, sum, and
pointer-range check is allocation-free and failure-atomic.

- Plane storage is the sum of each channel's `4 * area` bytes, with channel starts
  aligned to 256 bytes. It cannot be estimated from original width times height
  alone after transforms.
- Weighted storage is `16 * T` bytes when required, otherwise zero.
- Token storage is `Align256(4 * T) + 2 * T` bytes.
- Dispatch storage holds `48 * S` slice bytes, then aligned `32 * N` node bytes,
  then an aligned four-byte status word. Unused/reserved bytes are initialized.
- Initial CPU bridge peak includes the transformed CPU frame, weighted-feature
  staging, five two-row weighted-state arrays per active CPU participant,
  canonical layout, and overlapping resident plane/feature uploads. Shared
  direct initialization eliminates full-image staging in the Metal implementation.
  The original conservative allowance remains in the plan; no unplanned staging
  is permitted. Weighted state is computed serially per slice in this first bridge.
- Token generation includes retained feature arenas, output, dispatch and any
  dispatch staging. CPU modeling/emission retains the compact token arena once,
  along with any reusable feature owner. Views and resource tickets do not
  duplicate the backing charge.
- The complete search also retains the best encoded candidate, learning
  scratch/checkpoints where applicable, policy/header copies, and final
  publication capacity. A conservative sum with the existing CPU search bound
  is acceptable initially; reducing that bound requires a lifetime proof.

Input owned by the application, old output, immutable backend initialization,
driver memory, stacks, and small inline controls follow existing exclusions.
Replacement APIs charge old owners separately. All error paths drain pending work
before releasing backing or tickets. No resource reservation survives solely
because of an unjoined worker.

## Qualification gates

First compare features, all properties/predictors, selected leaves, contexts, and
complete tokens against the scalar path. Cover signed transformed extrema,
slice/row boundaries, nondefault weighted parameters, metadata/channel ordinals,
large/deep trees, and leaf representability failures. Compare pinned libjxl where
its direct oracle exposes the same prescribed policy.

Then compare full codestreams and selected policies for photographic, palette,
Squeeze, grayscale, alpha, 8-bit, and 16-bit cases. Test admission at the exact
bound, one byte below it, failed allocation/submission/wait, owner lifetimes, and
unchanged previous output on failure. Qualification must record which operations
ran on the GPU and include feature preparation/cache creation and CPU fallbacks
in complete-search timing. A winning-policy replay is a separate diagnostic.

The `modular_adaptive_token_plan` test passes in Release builds with Metal enabled
and disabled. It covers a 511-node weighted/property-15 tree, allocation-free
validation and failure publication, independently sized metadata and shifted
planes, dynamic dispatch sizing, retained feature lifetime accounting, exact
device-capacity admission and one-byte-short rejection.

## Implemented device qualification

`metal_modular_adaptive` runs both file-loaded and embedded Metal libraries on
Apple M4 Pro. The standalone producer/continuation suite checks:

- 203 production cases and 3,778,462 exact token comparisons with the scalar
  implementation; all six packed formats, padded/misaligned source rows, both
  16-bit byte orders, one-pixel and group edges, all 14 predictors and 16 properties.
- 2,831,887 exact weighted-feature pairs and 2,831,912 guarded diagnostic samples.
  These compare every property, every prediction, selected leaf/context, and
  packed residual, including nondefault weighted parameters and slice resets.
- Palette metadata, both Squeeze insertion orders, combined palette/Squeeze,
  unequal plane shapes, and DC-group placement. The independent CPU stream planner
  checks physical plane addressing and compact stream-local channel ordinals.
- A 511-node tree through complete serialization, the maximum 16383-node tree
  through token generation, and a 2047-node/depth-1024 weighted-property chain.
- General signed `int32` plane arithmetic using `int64` intermediates. Synthetic
  extreme planes test overflow and division semantics only; they are explicitly
  excluded from the valid-source and independent-decoder claims.
- 34 byte-identical complete files across prefix and ANS coding; the pinned
  libjxl decoder reconstructs 6,198,994 source samples exactly. This is prescribed
  candidate continuation, not adaptive search or its performance qualification.
- Exact implementation allocation peaks and one-byte-short failures for feature
  preparation and token dispatch; retained-arena accounting; feature lifetime after
  releasing CPU input; token lifetime after releasing features; CPU participation
  release/reacquisition at successful and failed waits.
- Malformed descriptors, invalid/image-ineligible trees, missing weighted maps,
  foreign or nonnative feature owners, unrepresentable residuals, weighted-state
  overflow, managed-host allocation sweeps, device allocation/submission/completion
  faults, and diagnostic/handoff allocation failures. Existing outputs and live
  resource charges are preserved on failure; pending work is drained.

Logs and complete-file artifacts are retained under `build/adaptive-integration/`
and `build/release/metal-modular-adaptive/`. The Release build and all 29 focused
Modular regression tests passed (`ctest -R modular -E '^modular_workflow$'`),
including the existing prescribed Metal paths and pinned/frozen oracles. The
Metal-disabled adaptive-plan and token-handoff tests also passed. The exhaustive
`modular_workflow` allocation sweep was not repeated for this producer milestone;
its earlier shard evidence is recorded in the integration document.

The producer has no silent fallback and does not alter encoder defaults.

## Complete-search integration

`ModularEncodingOptions::adaptive_gpu` privately opts the existing encoder into
the hybrid path. The same search implementation still enumerates, learns, encodes
and compares every candidate in the original order, with unchanged complete-size
and checkpoint tie rules. Both public-vector and managed-output publication remain
failure-atomic. Public API defaults and the original prescribed GPU route are
unchanged.

`AdaptiveModularTokenCache` borrows one fixed source view for the duration of one
encoder call. Its key contains RCT, the transform sequence and weighted parameters;
trees, entropy mode and learning budgets do not enter the key. A miss destroys the
old completed features and CPU layout before constructing a replacement. The CPU
frame and temporary device descriptors are released after preparation, while the
feature owner and canonical stream plan remain available to later candidates.
Search prepares weighted history even for an initially unweighted candidate;
standalone prescribed calls prepare it only when their tree needs it.

The allocation-free workflow planner conservatively adds the maximum adaptive
operation and canonical metadata bound to the complete CPU search bound. It
enumerates the same transform shapes, including worst-case palette metadata, and
includes cached features during learning, CPU fallback and entropy processing.
It intentionally does not discount unused GPU capacity for a capability-limited
backend. Existing outputs stay separately charged during replacement.

Missing capability or device-capacity rejection at preflight selects the CPU
candidate path. Subsequent feature, dispatch, representability, allocation or
completion failures are terminal, including an `Unsupported` returned after
preflight; none silently starts another encode on CPU.

`ModularProfile::adaptive_gpu` reports feature preparations/reuses/evictions,
successful GPU candidates, successful CPU fallback candidates, and nested feature
and token-operation intervals within the existing `tokens` stage. The feature
interval includes CPU weighted-history production and shared-memory initialization.
The token interval includes host dispatch setup, GPU wait, and completed-token
handoff; it is not a device-only timestamp. Input/transform stages remain separately
attributed, and all cache creation is inside the complete encoder-call timer.

The `metal_modular_adaptive_workflow` qualification passes 25 complete CPU/hybrid
comparisons across all six formats, prefix/ANS, palette/photographic patterns,
expanded learning with checkpoints, and one/four CPU participants. Winners,
candidate counts, training work counters and complete bytes match; the pinned
decoder verifies 53,380 samples. Additional direct cache cases vary tree, entropy,
weighted parameters and Squeeze layout; a new call at the same input address with
changed pixels verifies that features cannot escape a call.

The suite also passes 1,770 failure-atomic checks, including an exhaustive managed
host-allocation sweep for a tiny complete search, feature replacement after earlier
successful candidates, later token failure, device faults, explicit CPU fallback,
exact-bound admission and one-byte-short rejection. All domains are empty after
completed or failed calls. The other 29 focused Modular tests pass, along with three
Metal-disabled profile/plan/handoff tests. The expanded workflow test was rerun
after correcting its expected 8-bit palette-alpha samples to match packed bytes.
This correctness evidence does not establish full-resolution throughput; retained
logs are named `adaptive-search-*` in `build/adaptive-integration/`.
