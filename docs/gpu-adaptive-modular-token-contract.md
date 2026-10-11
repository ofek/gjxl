# Adaptive Modular token operation and storage contract

Status: private operation interfaces, ABI records, tree eligibility, and checked
storage planning are implemented. CPU feature production and Metal dispatch are
the next milestone. The existing prescribed producer is unchanged; this contract
is separate from its 31-node, four-equal-plane ABI.

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
  direct initialization may later eliminate a copy, but no unplanned staging is
  permitted.
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

The initial `modular_adaptive_token_plan` test passes in Release builds with
Metal enabled and disabled. It covers a 511-node weighted/property-15 tree,
allocation-free validation and failure publication, independently sized metadata
and shifted planes, dynamic dispatch sizing, retained feature lifetime accounting,
exact device-capacity admission and one-byte-short rejection. Device execution
and the remaining qualification gates above are not yet implemented.
