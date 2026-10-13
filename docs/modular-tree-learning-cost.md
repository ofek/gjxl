# Why CPU Modular learning costs more than libjxl e9

The main causes are multiplicative: GJXL repeats an expensive fit for several
transform candidates, and each fit uses a sort-and-sweep implementation with
all 14 predictors. Libjxl e9 avoids a full fit per transform candidate, normally
uses two predictors, and searches compact property histograms without sorting
each node. This is primarily an algorithm and search-policy issue, rather than
evidence that the current learning loop should be moved directly to the GPU.

The extra work is not equivalent to optimizing the final compressed size more
accurately. Restricting predictors or enabling our existing quantized learner
does save work, but neither is a safe drop-in replacement at the current rate.

This study uses the CPU implementation at `f27a797`, estimated ANS selection,
262,144 samples, 256 leaves and checkpoint selection. The pinned reference is
libjxl `e8ff09762481785938d8e4e01333ed3917571161`. Research tools live in
[`tools/modular_learning_cost`](../tools/modular_learning_cost/README.md);
production sources and defaults are unchanged. The initial instrumentation
milestone is `810c52e`. The [results and artifact hashes](modular-tree-learning-cost-results.json)
contain the complete measurements.

## Separate the repeated search from one fit

Detailed instrumentation reproduced all three saved estimated-ANS search files
byte-for-byte. These are two 500×500 photographic inputs and a 1011×277 grayscale
input, not the earlier 12 MP fixed-policy ANS benchmarks.

| Input | Small 4K/16 learning passes | Large 256K/256 learning passes | Learning CPU seconds, all passes | Fit share of complete encode wall time |
|---|---:|---:|---:|---:|
| photo-macan | 9 | 9 | 39.427 | 94.30% |
| alpha-ria | 9 | 9 | 41.969 | 93.71% |
| gray-patches | 5 | 5 | 14.121 | 95.45% |

The small retained search costs about **1%** of learning time. Removing that
fallback is not the useful target. The expensive work is the additional large
fit for each base RCT, ranked RCT and palette/Squeeze context. The photographic
cases each fit six RCT contexts and three Squeeze contexts at the large budget.

The broad `kTraining` bucket was previously ambiguous. These measurements show
that actual fitting dominates it on this pilot: preparation outside the fit and
the retained tiny learner totals only 0.31, 0.41 and 0.08 seconds respectively.
RCT ranking and palette probing outside the per-learning-call timers contribute
very little here. This conclusion is scoped to these inputs and budgets.

[`workflow.cpp`](../src/codestream/modular/workflow.cpp)'s `try_learning` loop
explains the multiplicity. Checkpoints are captured during one fit; they are
not separate retraining runs. Complete candidate encodes are a different count
again: 70, 67 and 38 respectively.

## What makes one native fit expensive

[`tree_learning.cpp`](../src/codec/modular/tree_learning.cpp)'s `FindSplit`:

1. Rebuilds parent histograms for all 14 predictors.
2. Sorts the node's sample indices separately for each of 16 properties.
3. Sweeps the sorted samples, updating both child histograms for every predictor.
4. Scores up to 32 quantile boundaries. Limiting scores to 32 does **not** avoid
   sorting or sweeping all the samples.

| Input | Sorting / fit wall time | Histogram sweep / fit wall time | Other fit work |
|---|---:|---:|---:|
| photo-macan | 49.88% | 47.50% | 2.63% |
| alpha-ria | 50.46% | 47.18% | 2.36% |
| gray-patches | 43.79% | 53.82% | 2.39% |

Together, sorting and histogram sweeps consume about **98%** of fitting time.
For photo-macan's nine large fits, this means 73,168 sorts, 380,088,064 sample
visits to sorting ranges, and 380,014,896 sweep iterations. Each sweep iteration
updates 14 low and 14 high histograms: about **10.64 billion `Change` calls**.
Each call maintains count-log-count cost, occupancy and extra bits, even though
only a small subset of boundaries is scored.

The sort comparator fetches properties indirectly from 184-byte
`TrainingSample` records. At 262,144 samples that array alone is 46 MiB, before
residual codes and other scratch. This is a poor layout for repeated sorting of
one property. The contiguous-key ablation below confirms that this access
pattern matters, although changing the comparator's data layout alone leaves
the larger algorithmic costs intact.

Constant properties account for only 4.93%, 3.71% and 10.38% of the large-fit
sample visits respectively. Skipping them is sensible, but cannot explain or
remove the bulk of the photographic cost.

## What libjxl does differently

The pinned [`enc_ma.cc`](https://github.com/libjxl/libjxl/blob/e8ff09762481785938d8e4e01333ed3917571161/lib/jxl/modular/encoding/enc_ma.cc)
and [`enc_ma.h`](https://github.com/libjxl/libjxl/blob/e8ff09762481785938d8e4e01333ed3917571161/lib/jxl/modular/encoding/enc_ma.h)
show the relevant design:

- `Predictor::Best` selects Weighted and Gradient. Variable mode enables all 14.
- Non-static properties are prequantized to compact byte columns; residual
  tokens are stored separately by predictor.
- At a node, it accumulates property-bin × residual-symbol counts and scans
  histogram prefixes. There is no per-node property sort.
- Scratch and entropy work use the actual symbol range and used property range.
  `EstimateBits` uses Highway SIMD and a 1/4096 minimum probability.
- Split thresholds scale with sampling fraction, and predictor/static-property
  preferences and threshold-driven growth differ from our legacy learner.

Libjxl also supports duplicate-sample aggregation, but **all sampled rows were
distinct on this three-input pilot**. Deduplication is not an observed cause of
its speed advantage here. Neither learner's fitting loop is parallel across
nodes, so tree-level thread parallelism does not explain this comparison.

The reference controls were rerun with internal learning/fit timers. All stock
e9 controls reproduce the saved bytes. Values below are sums of CPU time per
learning event, then medians of three measured encodes after a warmup:

| Stock e9 input | Learning events | Samples per event | Leaves per event | Fit CPU seconds | Preparation + fit CPU seconds |
|---|---:|---|---|---:|---:|
| photo-macan | 1 | 488,592 | 440 | 0.549 | 0.778 |
| alpha-ria | 1 | 651,456 | 411 | 0.757 | 1.035 |
| gray-patches | 2 | 2,904 / 182,634 | 47 / 17 | 0.090 | 0.169 |

The photos use two predictors. Gray-patches uses one in its first event and two
in the second. The reference photos actually use **more samples and leaves per
fit** than the native 256K/256 runs.

The libjxl predictor ablation changes only its selected predictor set, leaving
its surrounding encode pipeline in place:

| Input | Stock bytes | All-14 bytes | Stock fit CPU s | All-14 fit CPU s |
|---|---:|---:|---:|---:|
| photo-macan | 252,153 | 248,969 | 0.549 | 3.105 |
| alpha-ria | 228,013 | 226,079 | 0.757 | 4.246 |
| gray-patches | 4,912 | 4,412 | 0.090 | 0.494 |

On the photos, all 14 predictors cost about **5.6×** as much fitting CPU for
1.26% and 0.85% smaller files. Predictor width is therefore a real speed/rate
choice in libjxl too. The ablation changes learned trees and their sizes, so it
is a workflow effect, not a fixed-node microbenchmark.

A separate reference 262K-sample control takes 0.423 and 0.342 fit CPU seconds
on the photos, but grows 1,047 and 508 leaves. Merely matching a sample cap does
not match the learning problem: sample sets, quantizer, split penalties,
transforms and tree growth still differ. Do not turn these tables into an exact
cross-encoder kernel speedup or multiply their ratios as independent factors.

## Native ablations: which shortcuts retain rate?

For all 16 inputs, freeze the current winner's transform/RCT, retrain once with
the stated learner and select its best complete-file checkpoint. There is no
fallback to the old winning tree. The baseline reproduces **all 16 current
search winners byte-for-byte**, so the fixed controls start at the current rate.

| Frozen-transform learner | Total bytes | Change from baseline | Learning CPU s, single diagnostic sweep |
|---|---:|---:|---:|
| Legacy, 14 predictors | 8,688,169 | — | 54.781 |
| Legacy, contiguous sort keys | 8,688,169 | byte-identical | 49.191 |
| Legacy, two predictors | 8,984,750 | +296,581 / +3.4136% | 31.217 |
| Existing quantized learner | 8,961,614 | +273,445 / +3.1473% | 10.854 |

The CPU column is diagnostic, not a qualified end-to-end throughput result.
This is not a new full-search sweep of the alternative learners: transform
selection is frozen, while trees/checkpoints change.

The repeated three-input pilot makes the mechanisms clearer:

- On the two photos, retaining all predictors but sorting contiguous keys
  reduces learning CPU by about 9–10%. All 16 rate files and their resolved
  trees are identical to baseline. The prototype uses an unbudgeted scratch
  vector and needs production storage accounting before adoption.
- Skipping constant properties reduces photographic learning CPU by only
  about 1–3%, with identical outputs on the three tested inputs.
- Two predictors reduce photographic learning CPU by about 37–41%; substantial
  sorting cost remains. The photo/alpha files grow by 1.45% and 0.46%.
- The structured cases are more sensitive: chessboard grows from 62,805 to
  279,977 bytes, gray-patches from 18,299 to 24,676, and alpha16-ramp from 19,866
  to 55,445. A blanket two-predictor switch would discard useful capability.

The quantized mode is much cheaper, but is a bundled alternative, not a clean
test of binning: it changes sampling allocation, predictor width, bin boundaries,
scoring and preferences. It still clears a full 2×256×240-symbol histogram array
for each node/property and scans fixed symbol ranges with scalar cost code.
Its compression regressions do not establish that compact histograms are a bad
design, and its speed does not qualify it as a replacement.

## Why more work does not buy a rate advantage

More predictors and transform trials enlarge the explored space, but both
encoders optimize approximate local objectives. They do not exhaustively
minimize final file size. Our legacy learner uses 32 adaptive quantile
boundaries, a 4/2/0 residual proxy, a per-populated-symbol/header penalty and
best-first bounded growth. Libjxl uses different property quantization, a
4/1/2 proxy, probability clipping, sampling-scaled split thresholds and growth
preferences. Final context clustering and model/header cost also interact with
the chosen tree.

Consequently, additional work in our learner can choose different, less useful
splits. The expensive exact sorts do not imply exhaustive threshold evaluation;
the selected 32 boundaries can still miss useful ones. This is a plausible
mechanism, **not a newly isolated explanation for the remaining rate gap**.

The new experiments establish the sources of execution cost and the losses
from specific shortcuts. They do not isolate which quantizer, objective or
growth difference causes the residual compression gap. The prior bounded
reference studies support further work there, but still combine several
differences. On the current 16-input corpus, the baseline gap is +2.0786% versus
saved e9, not the older approximately 9% PNG gap.

## Recommended implementation order

The revised recommendation is a **reference-first replacement of the learner**.
Keep the legacy implementation as a measured regression control, but do not
make its selected trees a permanent compatibility requirement. The earlier
recommendation to preserve its decisions while rebuilding the sort engine
would leave the objective, predictor policy and growth differences unresolved.
Exact decoded pixels and format/resource contracts remain requirements;
identical legacy compressed bytes do not.

1. **Build a research adapter to the pinned libjxl learner before another
   native prototype.** Establish two complementary controls:

   - Capture the reference learner's quantized samples, weights, predictor
     order, property maps, static ranges, multiplier information and all
     scoring/growth parameters. Replay them through the pinned learner and
     reproduce its canonical tree. This provides identical inputs for later
     comparisons of fitting time and decisions; equal sample caps alone do
     not provide that control.
   - Run the reference sampling, property preparation and learner on
     GJXL-prepared transformed planes. Import the resulting supported tree
     into GJXL and encode it through the unchanged native entropy path. Freeze
     transform choice, weighted-predictor parameters and entropy settings while
     comparing learners. Measure complete bytes and exact reconstruction,
     preparation time, fit time and peak scratch separately.

   Validate property numbering, branch direction, signed thresholds, residual
   tokens, predictor state, offsets and multipliers at the boundary. Start with
   the common representable profile and report exclusions explicitly. Native
   limits include 16 properties, a 1M training-sample admission ceiling and
   image-dependent tree limits; do not silently truncate a reference sample
   set or tree to claim parity. The old 256-leaf study budget is not a format
   limit: the native representation supports up to 8,192 leaves, subject to
   the image-dependent node limit and depth/storage validation.

   The deliverable is a paired rate/time ledger and replay fixtures, first on
   the three pilots and then the 16-input corpus. If reference learning does
   not transfer its rate benefit into the native pipeline, isolate the sample,
   property or downstream coding difference before proceeding with the port.
   Keep the reference dependency confined to research tools.

2. **Implement the proven reference behavior natively, with differential
   tests.** Port the actual quantizer and sampling policy, residual-token
   proxy, probability clipping, split penalties, predictor preferences,
   sampling-scaled thresholds and tree growth. Use compact property columns,
   tokens per predictor, weighted samples, property-bin histograms and prefix
   scoring over active bins/symbols. Reuse bounded scratch instead of sorting
   each property at every node or clearing/scanning full fixed alphabets.

   First match the captured reference inputs and canonical trees on the pinned
   CPU/arithmetic path, then match preparation from the same transformed
   planes. Isolate floating-point/SIMD tie differences explicitly; do not
   silently replace reference scoring with our current integer objective.
   Require identical complete native bytes when feeding the same imported
   tree and coding settings. Include ties, repeated values, signed/high-depth
   properties, allocation failure and resource-boundary cases.

   Use the reference two-predictor policy as part of this coherent learner,
   retaining an all-predictor research control. The failed legacy two-predictor
   ablation does not test that combination. Likewise, the existing quantized
   mode is a comparison point, not a sufficiently faithful starting oracle.
   Do not reproduce the old bounded-reference study's repeated threshold
   search to force a leaf count: those repeated fits were a research control,
   not a production growth policy.

3. **Qualify one efficient CPU fit before changing search policy.** Optimize
   the matched implementation's layout, histogram accumulation, scratch reuse
   and vectorized entropy evaluation. Compare pure fitting on identical replay
   fixtures, including the same weights, predictor set and growth parameters.
   Report sample preparation independently, and track samples, leaves, work
   counters and peak memory alongside time. Preserve the qualified reference
   decisions during these execution-only changes.

   Use warmed, serial, interleaved Release runs with both CPU and wall time;
   collect detailed attribution in separate builds. Do not infer a kernel
   speedup from unmatched native/reference sample caps or multiply the earlier
   ablation ratios. The proposed target below is a gate to measure, not a
   speedup established by the present study.

4. **Reduce repeated large fits with a separately qualified search policy.**
   Study the reference transform selection and global/local training scope
   first. Aim for one fit per selected coding scope, with a bounded additional
   candidate fit only where ambiguity justifies it; count metadata/local
   learning events separately. Large-image e9 can learn local group trees, so
   minimizing the raw fit count is not the objective. See the
   [reference policy audit and next controls](modular-reference-search-policy.md).
   A cheap native shortlist is an alternative only after its ranking has been
   tested against exhaustive search using the *new* learner. The current small
   learner's ranking cannot be assumed to transfer.

   Record shortlist recall, actual fit counts, per-image byte losses and full
   learning/encode time. Report the candidate without a hidden slow legacy
   fallback. Qualify total learning cost separately from single-fit cost so a
   fast learner cannot conceal another nine-fit portfolio.

5. **Promote after rate, correctness and resource qualification; then choose
   the GPU slice from the new profile.** Expand beyond the 16 inputs with
   held-out photos at several resolutions, structured images, grayscale,
   RGB/RGBA and true 16-bit cases. Verify all decoded samples, including hidden
   RGB where preserved, deterministic output, memory bounds and failure paths.
   Commit the adapter, faithful native learner, execution optimizations and
   search policy as separate milestones with their qualification ledgers.
   Only then select GPU histogram/prefix/scoring work from the remaining cost.

### Proposed acceptance criteria

These are initial engineering targets, not measured outcomes or a claim that
learning alone explains the remaining whole-encoder rate gap.

| Scope | Proposed gate |
|---|---|
| Correctness | Exact lossless reconstruction and existing format/resource contracts on every supported test. |
| Reference fidelity | Matching prepared samples and canonical trees on the pinned CPU/arithmetic path; identical native files for identical trees/settings. Any portability-related tie difference is separately characterized. |
| One fit | At most 1.5× pinned libjxl fitting CPU time on identical replay inputs, reporting every case and outliers rather than only an aggregate. |
| Search approximation | At most +0.2% aggregate and +1% per-image bytes versus exhaustive search with the same new learner; review absolute byte deltas separately for tiny files. |
| Complete learning stage | Aim for at most 2× stock e9 preparation-plus-all-fits time on paired input runs; include every learning event and report both CPU and wall time. This is a workflow target, not a matched-kernel comparison. |
| Final native rate | No aggregate regression from the current qualified native encoder, with explicit review of every per-image regression. Aim eventually for within +1% aggregate bytes of stock e9 on the qualification corpus. |

The last target includes transform and entropy decisions beyond learning. Use
the reference-tree/native-entropy control to locate any residual gap; do not
respond to it by increasing predictor/sample/search budgets without a measured
rate-versus-time benefit. Exact compressed-byte parity with libjxl is not
required for a native encoder using different downstream coding decisions.

**Milestone update:** the adapter and 16-input controls are implemented at
`fe7cd44`; see [the reference-learning results](modular-reference-learning.md).
The all-predictor bridge reaches +0.9487% aggregate bytes versus e9, with
per-image regressions and expensive fits still unresolved. The next milestone
is the faithful native fitter against the captured sample contract. Its private
implementation and qualification are documented in
[native histogram fitting](modular-native-histogram-learning.md). Production
policy promotion remains gated on the criteria above.

**Native fitter update:** all 233 captured trees now match exactly. Paired
fitting uses 0.9477× the reference's summed median CPU time; 214 fits pass the
proposed per-fit target. Nineteen sub-millisecond single-leaf cases exceed it
(6.454 ms combined excess), so that gate remains partially unmet.

**Native preparation update:** native quantization, sampling and deduplication
now match all 32 Best/all-predictor bridge controls, including every prepared
sample field, raw tree, complete encoded file and independent decoded image.
See [native sample preparation](modular-native-sample-preparation.md) for the
managed-storage qualification and explicit rejection of a reproduced pinned
reference weight-overflow case.

**Small-fit follow-up:** vectorizable validation reductions retain all checks
and resolve the observed small-fit overhead. The repeated study passes the
1.5× CPU target on all 64 stock-reference spline fits, including all 19 prior
outliers; all 233 trees still match. See
[validation cost](modular-histogram-validation-cost.md). Paired complete-learning
timing on the three pilots is now documented below; search-policy qualification
and default promotion remain ahead.

**Paired learning update:** native preparation plus fitting matches the captured
samples/trees on all six pilot predictor controls and uses 0.9110× the reference's
summed median CPU time. Non-fit time is close in aggregate (1.0233×), with all
individual results retained in [the timing report](modular-native-learning-timing.md).
This is matched global learning, not stock e9 search or complete encode timing.
The next policy milestone must qualify training scope as well as transform and
predictor selection: the reference uses local trees on the large corpus inputs,
while this bridge uses one global tree.

## Measurement limits and validation

All measurements use Apple M4 Pro, Release AppleClang builds and the same
hash-bound input files. Native encoding allows four CPU participants; fitting
is serial. The reference runner installs no thread pool. Runs within this
study are serial; the desktop is not isolated, and CPU frequency/cache effects
are uncontrolled. Detailed counters can affect code generation and execution.

The first wall-only native pilot contained very long pauses and is excluded
from timing conclusions. A separate pilot records CPU time as well as wall
time: one warmup and two measured iterations per native variant, with variant
order rotated between inputs. Reference variants use one warmup and three
measured fresh-process iterations, with interleaved configuration order.
The gray-patches repeated native baseline varied from 3.00 to 3.97 CPU seconds;
do not interpret its micro-optimization ratios as stable speedup estimates.

The fine-instrumentation build is supplemented by an external-call timing
control linked to the original qualified learner/search objects:

| Frozen-transform learning call | Original learner CPU s | Instrumented baseline CPU s |
|---|---:|---:|
| photo-macan | 4.301 | 4.272 |
| alpha-ria | 5.089 | 4.955 |
| gray-patches | 2.356 | 3.482 |

All three reproduce the saved winners. The photographic control confirms that
the large per-fit cost is present in the original implementation. The unstable
gray-patches pilot does not support a precise overhead estimate. Existing
coarse profiling remains enabled in the control. These are sequential runs,
not an isolated paired overhead experiment; neither these timings nor unmatched
historical runs should be presented as production cross-encoder throughput.

Validation covers 3 full-search controls, 15 repeated native pilot cases,
64 frozen-transform rate cases, 36 reference encodes, and 3 repeated original
learner controls. Each invocation decodes its first output exactly and requires
all repeats to retain those bytes. The analyzer verifies saved controls,
variant byte/tree identities where expected, input and artifact hashes, and
links to the original reference controls. No production implementation or
public default is changed by this research branch.
