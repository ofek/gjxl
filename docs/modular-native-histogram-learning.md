# Native Modular histogram fitting

The native replacement now fits compact, quantized samples without a libjxl or
Highway dependency. This implements the **fitting** part of the
[reference-first learner plan](modular-tree-learning-cost.md#recommended-implementation-order).
Sampling/property preparation is now implemented in the separate
[native preparation milestone](modular-native-sample-preparation.md).
Automatic candidate selection remains separate. The default encoder still
uses its existing learner.

All **233** captured reference trees match exactly. On identical inputs, the
native fitter uses **0.9477×** the reference's summed median CPU time in the
initial paired study. That study had 19 small single-leaf outliers above the
proposed 1.5× target. A [validation-scan follow-up](modular-histogram-validation-cost.md)
now passes the target on all 64 affected-group fixtures, including those 19,
and rechecks all 233 trees. Neither study is an end-to-end speed result.

## Contract and reference fidelity

[`histogram_tree_learning.h`](../src/codec/modular/histogram_tree_learning.h)
defines private columnar samples: 16-bit multiplicities, token/extra-bit pairs
for the selected predictors, 32-bit static-property bins, 8-bit other bins and
signed dequantization cuts. Rows are caller-owned scratch and fitting may
partition them in place, including on a rejected fit. Only tree publication is
atomic. Invalid shapes, counts, bins, predictors, cuts and numerical parameters
are checked before fitting.

The fitter follows pinned libjxl
`e8ff09762481785938d8e4e01333ed3917571161`, preserving:

- The residual-token proxy, 1/4096 probability floor, approximate float
  logarithm, predictor-change penalty, Weighted/Zero tie preferences, and
  static/constant/no-Weighted split preferences.
- Histogram accumulation over quantized bins, prefix evaluation over active
  symbols, depth-first growth and the reference row-partition order. There is
  no per-node comparison sort or repeated threshold search to impose a leaf
  count.
- Signed thresholds, greater-than left children, forced multiplier cuts and
  all raw tree-node fields. Quantization equality belongs to the lower bin.

The AArch64 path uses explicit four-lane NEON math, including the reference FMA
polynomial and reduction order. A scalar path preserves that arithmetic order.
Matching another libjxl build's SIMD width, compiler or floating-point settings
is not implied. The native compilation uses normal production flags; only the
research adapter/reference units receive the pinned reference's special flags.

All fitting backing allocations occur before row partitioning. Scratch is
reused across nodes; allocation/resource errors leave the output tree unchanged.
An allocation-free planner bounds new scratch and result storage, excluding
the caller's sample columns and old output. Capacity/depth limits reject rather
than silently truncate a reference tree. The caller must provide the
image-dependent tree limit before using a result for global coding.

The default sample admission remains 1M. Explicit private research admission
allows up to 16M weighted samples, covering the existing large replay fixtures.
This does not raise the current public search budget or add an automatic
fallback. The representation ceiling remains 8,192 leaves and depth 2,047.

## Validation and measurement

The initial corpus replay and the repeated paired study match every raw field
of all **233** captured trees from the 16-input reference/bridge corpus.
An additional **129** independently generated differential cases exercise
signed/high-depth values, predictor ties, duplicate rows,
16-bit weight saturation, property subsets and forced multiplier boundaries.
Native contract tests cover row association, empty and zero-entropy input,
capacity/depth rejection, invalid inputs, every fitting allocation failure,
atomic output publication and the admitted scratch bound.

Release, forced-scalar and ASan/UBSan contract/differential suites pass, as do
the ordinary root-build Modular foundation, stream and new fitter tests.
Representative photographic and true 16-bit RGBA fixtures also match on the
scalar and sanitizer builds. Sanitizer coverage includes the native fitter,
harness and generated reference units; linked archives are uninstrumented and
vptr checks are excluded for the pinned no-RTTI reference. This is not a
whole-codec sanitizer qualification.

`native-verify FIXTURE` performs one exact fit and measures managed peak
backing under the computed reservation. `paired-replay FIXTURE REPEATS` warms
both fitters, then alternates reference/native execution order on identical
captured data. Both sample copies/conversions are outside the fit timer;
native validation and all fitting allocations are inside. Native managed peak
is measured during the warmup, separately from measured timing iterations.
Measured fitting calls have no installed resource reservation; the explicit
admission/accounting path is checked in the warmup. Sample-column backing is
reported separately. Neither value is process RSS, and both exclude allocator
metadata and stack storage.

The completed paired study uses one warmup and three measured repetitions per
fitter per fixture: **932 exact fits per implementation**. Its
ledger freezes the executable, native/tool source snapshots and compiler
commands, and binds the reference fixture and linked archive hashes. The
analyzer rejects incomplete ledgers, changed artifacts, nonidentical trees,
nondeterministic shapes and storage above the admission bound. It reports every
fit, every input/variant, and every case above the proposed 1.5× CPU gate.
The final native object, after adding compile-time overflow assertions, is
byte-identical to the timed native object. Final validation binaries, source
snapshots, logs and their hashes are retained separately.

## Paired fitting results

The [verified result ledger](modular-native-histogram-learning-results.json)
contains every fit and CPU/wall repetition. On an uncontrolled M4 Pro desktop,
the sum of per-fit median CPU seconds is **201.542 native / 212.663 reference**,
or **5.23% less native CPU**. The median per-fit ratio is 0.944×. These are sums
across reference and bridge fixtures, not the learning time of one encoding
configuration.

Representative bridge fixtures are below. Each cell is **native / reference
fitting CPU seconds**, with identical samples, predictor order and parameters:

| Input | Best: Weighted/Gradient | Variable: all 14 predictors |
|---|---:|---:|
| photo-macan | 0.359 / 0.392 | 1.989 / 2.233 |
| alpha-ria | 0.562 / 0.576 | 3.052 / 3.319 |
| gray-patches | 0.136 / 0.146 | 0.717 / 0.771 |
| photo-flower | 7.161 / 7.434 | 36.759 / 39.981 |
| alpha-flower | 10.660 / 11.038 | 49.095 / 52.092 |
| splines | 8.948 / 9.125 | 47.166 / 48.088 |

**214/233 fits pass the proposed 1.5× CPU target**, with a maximum ratio of
1.086× among those cases. All 19 exceptions are single-leaf stock-reference
spline fixtures: native medians are at most **0.691 ms**, and the worst relative
ratio is **3.306×**. Their summed medians are **10.812 ms native / 4.358 ms
reference**, an excess of **6.454 ms**. Do not hide this behind the aggregate or
call the strict per-fit gate passed. The ledger does not isolate how much of
that small-fit overhead is validation versus scratch setup; that attribution
was left open by that initial study. The subsequent
[validation-scan attribution and optimization](modular-histogram-validation-cost.md)
resolves those observed outliers; the original measurements above are retained
as their baseline.

Every warmup remains within its computed managed-storage reservation. Fitting
scratch plus result backing ranges from **766,124 to 779,488 bytes** in this
corpus. This excludes the caller-owned sample columns, which scale with sample
count: the compact row payload is 28 bytes for Best and 52 for Variable when all
16 properties are present, plus cuts and vector capacity. It also excludes any
old output, reference fixture copy and allocator bookkeeping.

## Remaining boundary

This API starts with finalized quantized samples. The separate native
preparation API now reproduces the reference quantizer, sampling RNG,
deduplication and property/predictor collection on all 32 bridge controls,
including complete native byte parity. Neither component reduces the number
of transform or predictor candidates. Search-policy qualification and default
promotion remain separate.

Preparation parity includes reference details that should not be approximated:
property ordering and clipped quantization maps; the small clipped histograms
used to derive property cuts (rather than sorting all pixel samples); geometric
presampling;
per-channel sampling seeds and the minimum sampling fraction; predictor state
updates on unsampled pixels; duplicate-weight saturation and hash-table growth;
and the exact sample/population denominator used to scale the split threshold.
The existing capture contract is the arbiter for these decisions.

Exact raw-tree parity establishes fitting fidelity. The earlier
[reference-tree/native-entropy rate results](modular-reference-learning.md)
remain the rate control, now reproduced with fully native preparation and
fitting. The paired fitting measurements above do not establish a complete
learning-stage speedup.
