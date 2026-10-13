# Native histogram validation cost

The small-fit outliers in the initial native fitter study were largely caused
by scalar sample-validation scans. Replacing early-exit element checks with
integer reductions enables compiler vectorization while retaining every
admission check. The fitting objective, samples, tree decisions, allocation
policy and scratch bounds are unchanged.

The weight scan uses a 64-bit sum and minimum weight; residual validation uses
an OR reduction of extra-bit fields; property validation uses maximum-bin
reductions. Shape validation caps row counts before these scans, so malformed
weights cannot overflow the sum. Invalid columns are fully scanned before
reporting failure. Allocation and output-publication behavior is unchanged.

AppleClang's ordinary Release optimization remarks confirm the four loop
families changed from unvectorized loops to vectorized reductions. No fast-math
or special compiler flag was added to the native target.

## Evidence

The [verified report](modular-histogram-validation-scan-results.json) combines
a representative before/after attribution, repeated paired fitting on the
entire affected group, and a full-corpus correctness replay.

For spline reference fixture 7 (128,142 distinct rows), the isolated validation
test uses five serial rounds alternating before/after processes, with 1,000
warmed validation calls per round. Full fitting is measured separately with
one warmup and nine native/reference paired repetitions per binary:

| Median CPU per call | Before | After |
|---|---:|---:|
| Isolated sample validation | 0.579 ms | 0.056 ms |
| Complete native fitting, including validation and allocation | 0.753 ms | 0.309 ms |
| Paired reference fitting | 0.328 ms | 0.334 ms |

Validation alone is about 10.3× faster in this diagnostic. Complete native
fitting uses about 59% less CPU on this particular single-leaf fixture. These
independent stage measurements must not be subtracted as nested counters or
generalized to the complete learning stage.

The follow-up paired study covers **all 64 stock-reference spline fixtures**,
with one warmup and seven measured fits per implementation. Every tree matches
exactly. **64/64 pass the proposed 1.5× per-fit CPU target**; the maximum ratio
is **1.442×**, and the ratio of summed median CPU times is **0.9605×**. This
includes all 19 earlier outliers. Their new summed medians are **5.481 ms
native / 6.895 ms reference**. The earlier 233-fit timing ledger remains intact;
this is a new timing qualification of the affected group, not a rerun of all
233 paired timings.

A separate replay matches **all 233 captured trees** and checks every computed
scratch reservation. Release, forced-scalar and ASan/UBSan contract/differential
suites pass, along with the root-build fitter/preparation tests. New rejection
cases cover invalid values at the ends of weight, residual, static-property
and nonstatic-property columns. The sanitizer coverage limits remain those
documented in the native fitter and preparation reports.

All timings are from an uncontrolled M4 Pro desktop, with no concurrent build
or study process during measured runs. Frozen binaries, before/after source,
compiler remarks, command lines, raw repetitions and validation logs are
retained in the hash-bound artifacts. Reproduction uses `validation-replay
FIXTURE REPEATS`, the existing `paired-replay` mode, and
`analyze_validation_scan.py`; the report identifies all input manifests.

This resolves the observed small-fit validation overhead without bypassing
validation. Native sample preparation and fitting remain private, with encoder
defaults unchanged. The [paired complete-learning pilot](modular-native-learning-timing.md)
now also qualifies preparation plus fitting on the three small pilot inputs.
Qualification of candidate count, predictor policy and training scope is still
needed before promotion or GPU work.
