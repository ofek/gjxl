# Modular entropy efficiency implementation and qualification

## Objective and frozen control

Reduce complete adaptive-search time while retaining competitive lossless rate.
The starting hybrid is `29cb42b`, qualified at `34586ea`: full 262144-sample /
256-leaf learning with checkpoints, GPU tokens, exact GPU ANS costs, and GPU
learner statistics. It encodes 68–70 candidates on the original 12 MP photos.
The three-image mean is 99.632 s, including 34.231 s of modeling. See
[the frozen qualification](gpu-adaptive-modular-learning.md).

The implementation must address all five steps below. Completing one step is
not completion of this plan. Preserve the exact exhaustive path as a control;
new selection policies are private and explicitly reported. Do not change public
defaults or claim libjxl parity from an isolated stage improvement.

## 1. Estimated ANS selection in the hybrid

Integrate `perf/modular-ans-estimates` (`a64510e`, qualified by `f27a797`) while
retaining GPU tokens and learner statistics. Estimated selection constructs the
chosen width from existing histogram estimates and skips ordered simulation when
no exact cost is requested. A requested cost remains exact for that selected
model. Keep the original exact selector, including its GPU evaluator.

Acceptance: CPU/hybrid byte and policy equality for each selection mode; exact
mode retains frozen outputs; estimated mode records zero ANS cost submissions
when no cost is requested; invalid modes and all failures preserve outputs.
Repeat paired full-resolution complete searches and report rate differences,
timing boundaries, resource peaks and actual execution routes. Existing fixed
Gradient estimator timings do not qualify the adaptive hybrid.

Status: implemented and qualified. The complete full-resolution comparison
shows an 8.44% time reduction (1.0922x) with identical files and policies on the
three-image pilot. The raw sleep-interrupted matrix and the independently
repeated whole pair are both retained; see
[the qualification report](modular-entropy-estimated.md) and
[machine-readable evidence](modular-entropy-estimated-results.json).
The profile executable accepts `--estimated-ans` independently of
`--metal`, `--metal-ans`, or `--metal-learn` and records `ans_selection`. The
adaptive planner omits unused GPU ANS measurement scratch in estimated mode.
The prescribed unpack/token GPU continuations propagate the same selector.

Local implementation validation: three focused Release suites passed, including
55 complete CPU/hybrid comparisons, 26,119,113 independently decoded samples and
5,835 hybrid failure-atomic checks. Eight entropy/Modular/Metal regression suites
and four CPU-only suites also passed. The estimator's original CPU experiment
and its commits remain in the merged ancestry. New provider tests require zero
exact calls without a cost request and exactly one selected model when exact
cost is requested. Native hybrid tests cover all six formats, weighted trees,
transforms, large-section costing eligibility, GPU learner execution, profiling
off, managed publication and allocation failures.

Logs: `build/entropy-step1-{build,tests,regression-build,regressions,cpu-build,cpu-tests}.log`.
Run the serial frozen-binary capture and audit with:

```sh
python3 tools/modular_training_study/entropy_efficiency.py capture \
  --capture build/entropy-estimated-qualified-20261011 \
  --binary build/release/gjxl_modular_search_profile \
  --baseline build/adaptive-baseline-20261011
python3 tools/modular_training_study/entropy_efficiency.py analyze \
  --capture build/entropy-estimated-qualified-20261011 \
  --json build/entropy-estimated-raw-analysis.json
```

The capture runs three CPU estimated controls and two alternating GPU
exact/estimated pairs per large image (15 full encodes). Its auditor verifies
the complete matrix, all artifact hashes, actual routes, unchanged search work,
CPU/hybrid bytes and policies, and frozen exact outputs. Rate changes between
selectors are measured, not assumed absent.

## 2. Reuse exact entropy populations

Retain the prefix partition's aggregated raw-value populations for ANS instead
of collecting and aggregating every token again. Provide population preparation
without requesting exact prefix section costs: the old prepared-prefix helper's
mandatory cost output must not accidentally add another full token pass.
Support both selectors and the exact GPU evaluator. Account for retained
populations in the allocation-free storage plan and resource lifetimes.

Acceptance: identical populations, models, section costs when requested, winning
policies and full files against each corresponding pre-change selector; zero
duplicate ANS collection/aggregation; invalid input and allocation failure
atomicity; actual complete-call timing and memory measurements.

Status: implemented and qualified. The paired complete-call population-reuse
milestone is 1.111x faster than its estimated pre-reuse control; see
[the pipeline report](modular-entropy-pipeline.md). `OptimizeEntropyCodeAndPrepareClusters` accepts a
null cost, and prepared ANS construction accepts the same optional GPU evaluator
as the unprepared path. Both exact and estimated Modular routes retain the
selected prefix populations only through model construction. The workflow plan
includes those retained populations in prefix output and excludes duplicate
collection/aggregation scratch from the borrowing ANS optimizer. The new
`prefix_section_measurement_count` distinguishes an ordered prefix token pass
from the small model-header serialization included in the older timing field.

All 12 selected Release suites and five CPU-only suites pass. The native tests
include 55 CPU/hybrid adaptive comparisons with 26,119,113 independently decoded
samples and 5,821 failure-atomic checks, plus the prescribed unpack and token
workflow matrices. **All 1,649 complete files match the pre-reuse files byte for
byte**, for each corresponding selector. The entropy storage matrix covers
population retention without a cost, including 206 allocation failures in that
new variant. Profile tests require zero duplicate ANS collection/aggregation and
zero ordered prefix measurements in the Modular model-only route. Shared
VarDCT encoder regressions pass as well.

Logs: `build/entropy-step2-{build,tests,cpu-build,cpu-tests}.log`. Pre-reuse byte
controls and their hashes are in `build/entropy-step1-native-byte-controls/`.
The frozen milestone binary is in `build/entropy-milestones/step2/`. The final
serial comparison includes it directly, with no concurrent compilation or model
replay. The report records complete-call times, managed peaks and admission bounds.

## 3. Separate candidate evaluation and final emission

Add a per-candidate ledger covering policy, model/payload/header costs, complete
size, work, and incumbent improvement. First implement exhaustive evaluation
using exact section sizes and the real header/TOC/padding rules, then write only
the winner. Preserve candidate order and strict tie rules. Reuse existing exact
measurements; do not replace checked allocation sizes with estimates.

Then implement a private fast policy that screens cheaply and exactly evaluates
a bounded shortlist. Preserve the exhaustive route as the oracle. Measure winner
recall, aggregate and worst-case byte regret, and full-call time on the retained
mixed corpus and large photos. A shortlist may change decisions; do not describe
its pixel-exact output as byte-equivalent to exhaustive search.

Acceptance: exhaustive complete sizes/bytes/policies match actual emission for
all candidates; tiny/empty streams, transforms, alpha/16-bit data, TOC thresholds,
ties, admission and failure paths are covered. Qualify the shortlist's rate/time
trade-off from actual outputs rather than a sum of estimated savings.

Status: implemented and qualified; correctness qualification is recorded in
[the candidate evaluation contract](modular-candidate-evaluation.md).
Exact sizing is 1.137x faster than current exhaustive emission on the large pilot.
The shortlist recalls 14/19 exhaustive winners, increases aggregate bytes by
0.3733%, and regresses `splines` by 7.6910%. It remains experimental. The original
emission oracle remains the default. Neither measured sizing nor shortlist
selection changes public policy. See [full results](modular-entropy-pipeline.md).

## 4. GPU population production and reuse

Define a bounded, exact per-context population producer adjacent to resident
tokens. Derive initial token histograms and clustered raw-value populations from
the same statistics. Use bounded tile reductions and an explicit sparse or CPU
route for wide values; support the existing integer/transform contracts without
silent truncation or unbounded scratch. Small-model decisions can stay on CPU.

Acceptance: native Metal/CPU population equality, including adversarial counts
and wide values; model/file equality; checked storage and ownership, explicit
crossover/fallback counters, failure-atomic publication; complete-workflow
performance including reductions, transfers, waits and remaining CPU work.

Status: implemented and qualified on the retained corpus; see
[the population contract](modular-gpu-populations.md). The operation and complete
workflow remain private/opt-in. GPU populations plus exact sizing preserve all 19
exhaustive winning files. Complete large-image time is 1.296x faster than exact
sizing with CPU populations and 1.819x faster than the frozen original exact-ANS
control. Small/wide fallback counters, bounded storage and all waits are included;
[the report](modular-entropy-pipeline.md) describes power conditions and limits.

## 5. Controlled entropy rate study

Replay identical ordered tokens and context identities through native and pinned
libjxl model builders. Hold tree, predictor, transform and token partition fixed.
Separate payload, histogram/configuration headers and context-map costs. Study
prefix-derived versus ANS-aware partitions and cluster capacity independently;
capacity expansion requires representation/storage validation, not just a changed
constant. Existing direct-ANS policies are useful controls but change multiple
choices and must not be mislabeled as isolated ablations.

Acceptance: a reproducible frozen-token study with hashes and exact emitted
costs, measured capacity/partition cells and separate limits for remaining
tree/transform/LZ77 differences. Promote an entropy-policy change only with
material measured benefit and complete-file decoding/rate/time validation.
A negative rate result is a valid study outcome; omitted measurements are not.

Status: complete. All 266 files from 19 images and 14 model cells decode exactly;
each native baseline reproduces the exhaustive-search winner. The partition-only
control saves 1.9352% at 32 clusters and 2.6858% at 64 clusters. The native balanced
builder produces the same partitions as the reference on all 19 cases at both
capacities. Direct balanced/density policies have material 16-bit regressions,
so no automatic policy switch is promoted. A native partition plus existing
prepared-population consumer is the next policy slice to qualify in complete
searches. See [rate conclusions and per-image results](modular-entropy-rate.md),
[machine-readable cells](modular-entropy-rate-results.json), and
[method and reproduction](../tools/modular_entropy_study/README.md).

## Engineering and measurement rules

- Document and commit each substantial implementation/qualification milestone.
- Preserve other worktrees and prior captures. Record revisions and binary/input
  hashes; freeze measured binaries before later source changes.
- Validate sample-exact output with the pinned independent decoder; distinguish
  exact byte-preserving optimizations from heuristic selection changes.
- Keep reference e7/e9 defaults and complete encoder timing boundaries explicit.
  GPU timing includes synchronization and all necessary host work. Rotate paired
  order, report repetitions/ranges, and retain per-case rate and resource data.
- Keep stage counters disjoint or label their parent relationship. Instrumented
  diagnostics and sampled CPU attribution are not interchangeable with latency.
- Measure the combined implementation at the end; do not multiply isolated
  speedups or claim savings from arithmetic ceilings.

## Serial final qualification

`tools/modular_training_study/qualify_entropy_pipeline.py` freezes and checks the
original and step2/step4 binaries. It runs 128 complete calls: 16 retained mixed
images once through each of five current modes, followed by two rotated repetitions
of eight modes on each original 12 MP image. The old exact and estimated selectors
and the population-reuse milestone are measured directly; current emission,
exhaustive sizing, shortlist, GPU population sizing and combined shortlist/population
modes carry exact candidate ledgers. Corpus timing includes all learning, tokens,
modeling, waits, final emission and ledger serialization. Setup, independent decode
and file I/O are reported outside the encoder-call timer. No compile or independent
model replay is run concurrently with timed calls. A temporary idle-sleep guard is
released on exit; process heartbeat and sleep evidence remain in the capture.

```sh
python3 tools/modular_training_study/qualify_entropy_pipeline.py capture \
  --root build/entropy-pipeline-qualified-20261011
python3 tools/modular_training_study/qualify_entropy_pipeline.py analyze \
  --root build/entropy-pipeline-qualified-20261011 \
  --json docs/modular-entropy-pipeline-results.json
```

The independent audit checks the full matrix, commands, hashes, training work,
actual GPU submissions, admission, disjoint costs, each exact candidate against
real emission, exhaustive winner bytes, shortlist finalists/regret and population
byte neutrality. A heartbeat interruption requires an explicit repeated whole
comparison; no sleep subtraction or fastest-run selection. The rate study runs
serially afterward, using the newly qualified exhaustive winning policies.

The 128-call raw capture completed and passes its independent artifact audit.
The power log records battery/AC transitions during `1039-r1-measure`; therefore
the entire first 1039 comparison is repeated, preserving all original artifacts:

```sh
python3 tools/modular_training_study/recheck_entropy_pipeline.py \
  --source build/entropy-pipeline-qualified-20261011 \
  --root build/entropy-pipeline-power-recheck-20261011 \
  --image 1039 --repetition 1 \
  --reason 'Battery to AC transition during original comparison'
```

Pass that directory to `analyze --recheck` for qualified timing tables. The
analyzer identifies affected comparison groups from the retained power log and
rejects unrelated or duplicate replacements. Its raw sums remain available.
The recheck verifies AC power before, during and after every call, as well as
unchanged files, policies, candidate ledgers and training work. The reporting
tool refuses unresolved power-transition comparisons. Positive/relocated capture
audits and ten deliberately corrupted-capture cases pass. The AC recheck also
passes original/relocated audits and rejects four damaged rechecks. All 136 calls
are audited; the qualified table uses the full eight-call replacement for the
affected comparison. No other comparison is replaced.
