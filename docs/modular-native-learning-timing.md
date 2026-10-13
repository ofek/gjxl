# Matched native Modular preparation and fitting

The native replacement now has a paired **complete learning-call** control,
including sample preparation, fitting and sample release. On the three pilot
inputs with both Best and all-predictor settings, it uses **0.9110×** the pinned
reference's summed median CPU time. Each native warmup reproduces all captured
sample fields, and every measured native/reference call reproduces its expected
tree. This supports moving the main engineering effort to selection policy and
training scope. It is not an end-to-end encode result or default promotion.

The [verified ledger](modular-native-learning-timing-results.json) records all
six cases and repetitions. Earlier [preparation qualification](modular-native-sample-preparation.md)
already checked 32 full native files and independent exact decodes. This timing
study does not repeat those encodes or infer new compression results.

## Paired control

`paired-learning INPUT POLICY FIXTURE best|all REPEATS` reconstructs the frozen
native winning transforms and canonical stream plan. Both implementations read
the same immutable, reference-aligned planes with identical strides. Options,
stream-ID holes, channel numbering, predictor resets and the expected captured
tree are shared. The pinned reference remains libjxl
`e8ff09762481785938d8e4e01333ed3917571161`.

Each case has one discarded warmup per implementation followed by three serial
measured pairs, alternating reference-first/native-first order. The native
warmup checks every prepared field before fitting can reorder rows. All calls,
including measured calls, check the complete raw tree after the timer stops.
There are **24 learning calls per implementation**, including warmups.

The total timer includes allocation, native admission/validation, preparation,
fitting and destruction of temporary samples. Transformation, stream planning,
plane copying, fixture loading, external storage-bound queries and comparison
are excluded. No managed-accounting reservation is installed during timings.
The recorded storage values are bounds, not new peak measurements; the earlier
preparation/fitter reports contain the managed-peak qualification.

Native preparation and fitting have separate nested timers. Reference fitting
uses the existing hook; its **non-fit** time is total minus fit, including sample
collection, finalization, release and wrapper overhead. It is not an isolated
preparation timer. Non-fit differences are computed per repetition before
taking medians; independently computed stage medians need not sum to the total.
The reference hook adds small event/timing overhead to its total.

## Results

Apple M4 Pro, ordinary Release settings for native code and the qualified
reference flags, no concurrent build or study during measurements. Each CPU
cell below is the median of three measured calls, in seconds:

| Input / predictors | Native preparation | Native fit | Reference fit | Native total | Reference total | Total ratio |
|---|---:|---:|---:|---:|---:|---:|
| photo-macan / Best | 0.1570 | 0.5344 | 0.5748 | 0.6915 | 0.7232 | 0.9561× |
| photo-macan / all 14 | 0.2964 | 2.9427 | 3.3063 | 3.2391 | 3.6162 | 0.8957× |
| alpha-ria / Best | 0.1770 | 0.7490 | 0.7899 | 0.9274 | 0.9745 | 0.9516× |
| alpha-ria / all 14 | 0.4599 | 4.2118 | 4.7482 | 4.6658 | 5.1420 | 0.9074× |
| gray-patches / Best | 0.0491 | 0.2001 | 0.2138 | 0.2494 | 0.2632 | 0.9475× |
| gray-patches / all 14 | 0.1007 | 1.0383 | 1.1501 | 1.1373 | 1.2568 | 0.9049× |

Summed case medians are **10.9104 native / 11.9760 reference CPU seconds**,
or 8.90% less native CPU. These sums combine two alternative predictor policies;
they are not the total for one encoding configuration. The analogous wall-time
ratio is 0.9053×. All six fitting comparisons pass the proposed 1.5× CPU gate.

Non-fit CPU sums are **1.2402 native / 1.2120 reference seconds**, a ratio of
1.0233×. Preparation is therefore close in aggregate on this pilot, with an
outlier: alpha-ria/all has 0.4599 native versus 0.3990 reference non-fit seconds
(about 15% more). It does not hide a large fitting-stage regression. The raw
ledger exposes individual variation, including the slower first native
photo-macan/Best call. The desktop was not isolated; do not use these absolute
times to infer a speedup against earlier studies captured at other times.

## Validation and remaining scope

Release and ASan/UBSan pass all seven research CTest entries: native fitter and
preparer contracts, their differential suites, and three evidence-analyzer
suites. The new analyzer tests include mismatched scope/order, missing exact
checks, changed storage bounds, nonfinite timing, invalid nesting and taking
the median of per-run differences. Additional sanitizer paired controls on
photo-macan/Best and alpha16-ramp/all reproduce samples and trees. The validated
Release executable is byte-identical to the timed frozen executable.

Sanitizers cover native preparation/fitting, the harness and generated reference
units; linked archives remain uninstrumented and vptr checks are excluded for
the pinned no-RTTI reference. Validation artifacts and hashes are bound in the
result report. Native production sources and encoder defaults did not change
in this milestone.

This pilot covers global learning with frozen native transforms and default WP
parameters. It does not qualify large-image preparation speed, local trees,
per-group transform/WP selection, a candidate shortlist, held-out rate, or the
proposed whole-learning-stage target against stock e9. The next work is the
[reference search and scope control](modular-reference-search-policy.md), not
increasing sample or predictor budgets to compensate for remaining differences.
