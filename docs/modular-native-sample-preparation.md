# Native Modular sample preparation

The reference-style quantizer, sample collection and deduplication now run
natively, feeding the [native histogram fitter](modular-native-histogram-learning.md).
There is no libjxl or Highway dependency in either native component. **All 32
Best/all-predictor controls on the 16-input corpus reproduce the reference
samples, raw trees and complete bridge bytes exactly**, with independent
lossless reconstruction checks. The default encoder still uses its existing
learner and search policy.

This completes the native preparation portion of the
[reference-first plan](modular-tree-learning-cost.md#recommended-implementation-order).
It establishes behavior and resource bounds. It does not establish a new
learning-stage speedup or qualify automatic candidate selection.

## Implementation contract

`histogram_sample_preparation.h/.cpp` borrows channel views for a synchronous
call. An image normally corresponds to one canonical Modular stream; empty
image entries preserve group-ID holes, and channel numbering resets for each
image. Metadata is a leading channel prefix. The original qualification below
uses the default weighted-predictor header and properties 0–15. A subsequent
[per-stream WP extension](modular-native-stream-preparation.md) now qualifies
actual e9 stream preparation; previous-channel properties remain unsupported.

The implementation preserves pinned libjxl
`e8ff09762481785938d8e4e01333ed3917571161` behavior:

- Geometric presampling with group-dependent seeds; property order; signed,
  absolute, difference, coordinate and fixed WP cuts; lower-bin equality and
  the reference's distinct quantization/map clamping boundaries.
- Empty-as-uninitialized property-cut caches, including their order-dependent
  behavior on degenerate distributions. The optimized implementation derives
  the same quantiles from small fixed histograms, without retaining and
  rescanning vectors of presampled pixels.
- Per-channel Bernoulli seeds, the minimum 1,024-pixel sampling fraction,
  default WP state updates on every pixel, and the 4/1/2 residual token proxy.
  It computes residuals only for the selected predictors and writes compact
  columns directly, avoiding the legacy 184-byte training records.
- Weighted-row deduplication, hash order/table growth, saturation handling,
  multiplier cuts, static ranges, and exact float bits for the sampling-scaled
  split threshold. The reference denominator includes the first image's
  population before adding all gathered pixels; a missing first image
  contributes one instead.

`PreparedHistogramSamples` owns the finalized columns and copied multiplier
ranges. Callers can inspect them through const access; fitting may reorder
rows internally. Preparation publishes the result and counters only on
complete success. It does not truncate samples to meet a limit.

The default weighted-sample admission stays at **1M**; explicit private research
admission supports **16M**. The scan population is limited to **256M pixels**,
with bounded group/channel counts and at most 256 requested property values.
Very small sampling fractions that cannot support the reference's float
geometric calculation are rejected. Unsupported residual/WP arithmetic and
malformed views/options also fail without publishing partial samples.

The allocation-free, pixel-read-free storage planner replays the sampling RNG
over channel geometry to count selected samples exactly. This admission work
is part of preparation time. It bounds result columns, cuts, deduplication
storage, quantization scratch, copied multiplier ranges and one live WP state.
The bound excludes borrowed pixels, old output, allocator bookkeeping and
stack storage. Preparation uses managed allocations throughout.

## Qualification

The [verified ledger](modular-native-sample-preparation-results.json) covers
both predictor controls for every image in the frozen reference bridge study:

| Check | Result |
|---|---:|
| Prepared dimensions, weights, columns, order and cuts | 32/32 exact |
| Threshold/penalty float bits, ranges and multipliers | 32/32 exact |
| All raw tree-node fields | 32/32 exact |
| Complete native encoded bytes | 32/32 exact |
| Independent lossless decode, including preserved hidden RGB | 32/32 exact |
| Managed preparation backing within admission | 32/32 passed |

Aggregate bytes remain **8,837,251 Best / 8,591,998 all predictors**, exactly
matching the earlier reference-tree/native-entropy control. The latter remains
+0.9487% versus saved stock e9 on this corpus. These are fixed winning native
transforms, with unchanged estimated ANS; they are not an automatic native
search result. The earlier per-image regressions remain relevant.

The largest preparation peak is **531,958,752 bytes** for alpha-flower/all,
with 8,938,244 selected/distinct rows. This includes reserved sample capacity
and hash storage; it is not fitting scratch or process RSS. Every per-case
bound and peak is recorded. Large global all-predictor fits still have
substantial storage requirements despite the compact representation.

There are **101 generated differential cases** covering sparse/full sampling,
signed 16-bit values, both property orders, predictor subsets, narrow channels,
group holes and offsets, coordinate/static clamping, metadata admission,
multiplier boundaries, duplicate rows, table growth and weight saturation.
Native contract tests cover invalid input, minimum sampling, population
accounting, default/unprepared objects, every preparation allocation failure,
unchanged old output and insufficient reservations. Eight analyzer tests
reject incomplete correctness claims, changed bytes, wrong case identity,
invalid populations and failed storage bounds.

Release, scalar and ASan/UBSan suites pass; the root-build Modular foundation,
stream, fitter and preparation tests also pass. Photographic and true 16-bit
RGBA samples/trees match under scalar and sanitizer builds. Instrumentation
covers native preparation/fitting, the harness and generated reference units;
linked archives are uninstrumented and vptr checks are disabled for the pinned
no-RTTI reference. This is not whole-codec sanitizer qualification.

The immutable study retains a frozen executable, native/tool source snapshots,
compiler commands, input/fixture/policy/archive hashes, outputs and logs. Final
validation is archived separately; its native sources match the corpus study.
The analyzer verifies complete coverage against that study and the original
bridge. Preparation times from the correctness runs are diagnostic only:
there is no warmup/repetition control, storage accounting is enabled, and
some runs overlap validation builds. Do not use them as speedup evidence.

## Explicit reference exception

An additional regression reproduces a pinned-libjxl deduplication overflow.
With the Zero predictor, property 9, full sampling and two constant-one
channels of sizes 2×15,000 and 2×25,000, hash-table growth leaves two aliases
to the same row. Saturating one alias removes that slot, but the other still
points to the saturated row. Its next increment wraps the 16-bit weight.
The reference reports 80,000 selected samples but the row weights sum to
14,464, losing 65,536 samples of weight.

The native preparer detects this condition and returns `Unsupported`, leaving
the prior output intact. This is an intentional rejection of corrupt reference
population accounting, not a claim of parity on that input configuration.
It does not occur in the 32 corpus controls. No change to libjxl was made.

## Remaining work

The [paired complete-learning pilot](modular-native-learning-timing.md) now
matches reference samples/trees on all six controls and uses 0.9110× its summed
median CPU time. Large-image preparation timing is not yet qualified. Next,
qualify a [search and training-scope policy](modular-reference-search-policy.md)
that avoids repeated large candidate fits. The
[validation-scan follow-up](modular-histogram-validation-cost.md) addresses the
19 previously reported small-fit outliers. Held-out rate, correctness and
resource qualification still precede default promotion and selection of a GPU
implementation target.
