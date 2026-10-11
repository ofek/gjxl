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

Status: pending.

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

Status: pending.

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

Status: pending.

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

Status: pending.

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

Status: pending.

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
