# Estimated ANS model selection

This experiment branches from `5faf68d` (qualified CPU tree checkpoint search).
It tests whether histogram-based model selection can remove the ordered ANS
candidate simulation without a material rate penalty. Public defaults remain
exact. No GPU kernels or dispatch defaults change.

## Implementation milestone

`AnsSelectionMode::kEstimated` reuses the existing per-cluster estimates:
serialized histogram bits plus normalized-frequency cross entropy, HybridUint
extra bits and configuration signaling. It selects the lowest estimated-cost
alphabet width and constructs only that width's encoding tables. Context-map
and fixed per-stream state overhead are common across these width candidates. Equal estimates
retain the first (smaller) width. Clustering, configuration candidates and
histogram precision search are unchanged.

For each candidate histogram, the symbol estimate is
`sum(count[s] * (12 - log2(normalized_frequency[s])))`, with normalized
frequencies summing to 4,096. GJXL already computes this quantity in
`EstimateAnsHistogramCost`; the change reuses the cached costs to select a
width. Ordered-state and renormalization effects are omitted by this estimate,
which explains why equal or nearly equal estimates can yield different actual
payload sizes. The final writer still emits the selected ANS payload normally.

The existing exact mode remains the default. Estimated mode skips the ordered
candidate simulation when no cost output is requested. If a caller requests
`EntropyCodeCost`, the chosen model is simulated once: model bits, token bits
and per-section counts remain exact for that model. Estimates never replace
allocation bounds or exact section-size contracts. This does not promise the
same model or compressed bytes as exact selection.

The private CPU Modular option `ans_selection` reaches the shared frame encoder,
including the CPU continuation after GPU input preparation. Search and tree
checkpoints still compare actual complete files. No GPU timing qualification is
claimed. Conservative existing storage bounds cover the smaller candidate set.

Validation at this milestone:

- 27 focused Release CTests passed, including ANS emission, entropy/storage,
  Modular oracles, transforms, API/CLI, checkpoint search and CPU-side GPU contracts.
  The full legacy `modular_workflow` fault suite was excluded.
- Three ASan/UBSan tests passed: entropy, estimated ANS selection and GPU workflow
  contracts. The pinned reference archives were not sanitizer-instrumented.
- New tests independently score all original width candidates, compare interleaved,
  split and prepared-population paths, and confirm no estimated-mode simulation.
  Requested exact costs match actual serialized model and payload sizes.
- Prescribed and checkpoint-search workflows decode sample-exactly, agree across
  one/four CPU participants, and honor exact/one-byte-under admission.
- All 6,248 managed allocation-failure positions in the new small prescribed/search
  workflow fixtures preserve caller output and release resources.

The first test run passed a null cost pointer to the existing prepared-prefix
helper, contrary to that helper's contract. The test was corrected; the failing
development logs are retained and excluded from qualification.

The [implementation qualification record](modular-ans-selection-qualification.json)
binds source and retained test evidence under `build/ans-selection/` and
`build/ans-selection-sanitize/`.

## Fixed-policy rate and timing

The completed fixed study uses 35 prescribed cases: Gradient/RCT6 (identity RCT
for gray) on the 16-input checkpoint corpus plus three saved 12 MP photographs,
and the saved 256K/256 learned policies on the 16-input corpus. Each pair keeps
transforms, predictors, trees and the prefix-derived context partition fixed.
The isolated ANS calls share the same ordered token buffers and prefix object.

| Fixed cases | Exact bytes | Estimated bytes | Increase | Complete encode speedup | Isolated ANS speedup |
|---|---:|---:|---:|---:|---:|
| Gradient, 16-input corpus | 14,877,458 | 14,877,494 | 36 / 0.000242% | 2.04x | 4.98x |
| Gradient, three 12 MP photos | 36,995,793 | 36,996,364 | 571 / 0.001543% | 2.36x | 6.10x |
| Saved learned policies, 16 inputs | 8,688,162 | 8,688,169 | 7 / 0.000081% | 1.63x | 5.21x |

Speedups divide sums of per-case median times. These are paired observations
from one Apple M4 Pro desktop session. Complete encodes allow four CPU
participants; isolated ANS calls run on the calling thread without an admitted
worker group. These are not confidence-bounded throughput claims. Each fixed case has one warmup pair,
four measured pairs in alternating order, and one separate diagnostic pair.
Profiling, decoding, file output and byte comparisons are outside the ordinary
timed calls. The isolated ANS timing excludes tokenization and prefix modeling;
the complete encode includes them. The learned-policy encode does not retrain.

For the three large photos individually, complete encode medians improve from
0.825–0.845 s to 0.343–0.361 s (2.29–2.46x). Isolated ANS construction improves
5.47–6.76x. Each diagnostic estimated call builds one alphabet width and records
zero exact simulation time; all 35 pairs retain the exact mode's configuration
and histogram candidate counts. This confirms removal of the targeted work.
It is not a GPU speedup or an end-to-end speed comparison against libjxl.

All 70 retained fixed files decode sample-exactly. Every repeated/profiled
output matches its independently decoded first output. The 16 learned exact
files match the prior checkpoint study byte-for-byte; the three large Gradient
exact files match the earlier token performance controls. The worst relative
fixed-case increase is one byte on `palette-358-learned` (1,045 to 1,046 bytes,
+0.095694%); the largest absolute increase is 239 bytes on the 12 MP `1049`
photograph (+0.002191%).

The measurements run serially within this study. The desktop was not isolated:
unrelated compilation, another Modular study, browsers and OS activity were
observed during the surrounding adaptive rate sweep. No thermal-stability or
controlled-machine claim is made. Ratios on small cases are particularly
sensitive to scheduling and timing overhead.

## Adaptive search rate

The full 16-input 262,144-sample / 256-leaf checkpoint search retains actual
complete-file selection. Every exact output reproduces the saved checkpoint
control. Every estimated output selects the same RCT, transforms and tree as
its exact counterpart; candidate counts are also unchanged. This is an observed
result on this corpus, not a guarantee for arbitrary inputs.

| Adaptive corpus | Exact bytes | Estimated bytes | Increase | Exact gap vs libjxl e9 | Estimated gap vs e9 |
|---|---:|---:|---:|---:|---:|
| PNG-derived, 11 inputs | 7,863,872 | 7,863,877 | 5 / 0.000064% | +2.475426% | +2.475492% |
| All 16 inputs | 8,688,162 | 8,688,169 | 7 / 0.000081% | +2.078567% | +2.078649% |

The only increases are `photo-flower` (+2 bytes), `gray-macan` (+2),
`palette-358` (+1) and `noise16` (+2). All 32 files decode sample-exactly.
The e9 denominator uses the verified saved lossless files, including hidden
RGB under alpha; no reference encoder was rerun for this experiment.
The estimator removes work with negligible measured rate loss. It does not
recover the remaining compression gap to e9.

## Repeated adaptive timing

The serial adaptive pilot uses one warmup pair and three measured pairs per
input, with four allowed CPU participants and alternating exact/estimated order.
These calls enable the Modular profile to retain stage times and resolved
policies. The median full-search times and model-stage times are:

| Input | Exact search (s) | Estimated search (s) | Observed ratio | Exact modeling (s) | Estimated modeling (s) |
|---|---:|---:|---:|---:|---:|
| alpha-ria | 34.164 | 33.216 | 1.029x | 1.884 | 0.906 |
| gray-patches | 10.974 | 10.767 | 1.019x | 0.445 | 0.278 |
| photo-macan | 31.660 | 30.778 | 1.029x | 1.677 | 0.811 |

The ratio of summed per-input medians is **1.027x**: 76.798 s to 74.760 s.
Training still takes roughly 29.3 s on photo-macan, 31.5 s on alpha-ria and
10.4 s on gray-patches in both modes. Removing the ANS dry run therefore has
much less impact on adaptive search than on prescribed encoding. This is a
small instrumented pilot on an uncontrolled desktop, not a statistical speed
guarantee. Concurrent rate-sweep timings are excluded from these comparisons.
All six pilot files match the corresponding adaptive rate-sweep files.

## Completed qualification and decision

Across all phases, 108 retained case/mode files decoded sample-exactly. There
were 476 complete encode calls and 420 isolated ANS calls; repeated outputs and
models were stable. All 38 saved exact-byte controls passed. All 35 fixed replay
pairs kept identical token buffers/context partitions and configuration/histogram
candidate counts, while estimated mode constructed one width and recorded no
exact simulation. All 16 adaptive winners and their candidate counts remained
unchanged, and both outputs matched the separately frozen learned-policy files.

The estimated selector is a useful CPU performance path: it removes the targeted
hotspot with negligible measured rate loss. Use the exact selector as the control
for oracle/rate work and retain exact cost reporting when requested. Public
defaults remain exact in this change. Adaptive tree training remains the dominant
runtime cost in the pilot; no GPU speed or broad libjxl rate-parity claim follows.

The [structured results](modular-ans-selection-results.json) retain per-case byte
changes, timing samples/ranges, diagnostic counters and 346 artifact hashes.
The [study harness](../tools/modular_ans_study/README.md) describes reproduction,
paired order, decode checks and frozen controls. Inputs, binaries, manifests,
raw traces and retained files are under `build/ans-selection/`.
The measured runner/driver snapshot predates the reporting-only analyzer and
README additions; no production or measured runner code changed after the
implementation milestone. Development test failures and the runner smoke pilot
are excluded from the reported measurements.
