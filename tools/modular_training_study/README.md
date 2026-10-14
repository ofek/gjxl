# Native Modular training coverage study

The current [entropy efficiency plan](../../docs/modular-entropy-efficiency.md)
tracks estimated ANS selection, exact population reuse, candidate evaluation,
GPU statistics, and the controlled entropy-rate study. `gjxl_modular_search_profile`
accepts `--estimated-ans` with any CPU/Metal route; the JSON records its selection
mode. `entropy_efficiency.py` captures and audits the original three full-resolution
images with exact/estimated GPU pairs and independent CPU estimated controls.
See the plan for reproduction commands and qualification status.

The [parallel weighted-feature preparation report](../../docs/parallel-weighted-feature-preparation.md)
documents bounded CPU preparation and the Metal wavefront producer. On a Metal
benchmark build, `gjxl_modular_feature_profile INPUT REPETITIONS raw|rct|squeeze|palette`
compares one/two/four CPU participants and GPU preparation of the same prepared
frame. It emits one warm-up plus the requested measured rounds, reversing order
between rounds, and checks every feature against the serial reference. Input
reading/transforms, backend setup and the retained reference are outside its
operation timer; allocation, copies, scheduling/dispatch, completion and publication
are inside. Its reservation covers one operation; the source and reference are
additional live memory outside that reservation.

For complete encoder comparisons, `gjxl_modular_search_profile` accepts
`--cpu-features`, `--gpu-features` or `--auto-features` with a Metal route. Auto is
the workflow default and selects the producer before execution using the
qualified size/shape gate. `adaptive_gpu.feature_preparation` records actual CPU
participants, GPU work and pre-dispatch CPU fallbacks. `--gpu-features` bypasses
the size gate but still respects device and shape eligibility. Errors after work
begins propagate without retry. These switches do not change learner policy or
feature sharing, and they do not select GPU feature work in CPU-only encoding.

This opt-in CPU study measures complete raw lossless Modular files, including
transform/tree/model headers and padding. It does not modify public defaults.
Use a clean pinned libjxl source/build for independent sample-exact decoding.

Build GJXL with Metal disabled and the oracle enabled, then build this runner:

```sh
cmake -S . -B build/training-coverage -DCMAKE_BUILD_TYPE=Release \
  -DGJXL_ENABLE_METAL=OFF -DGJXL_MODULAR_ORACLE_BUILD=/absolute/pinned-libjxl-build
cmake --build build/training-coverage -j 4
cmake -S tools/modular_training_study -B build/training-coverage/study \
  -DCMAKE_BUILD_TYPE=Release -DGJXL_BUILD="$PWD/build/training-coverage" \
  -DREFERENCE_SOURCE=/absolute/pinned-libjxl-source \
  -DREFERENCE_BUILD=/absolute/pinned-libjxl-build
cmake --build build/training-coverage/study -j 4
python3 tools/modular_training_study/run.py \
  --runner build/training-coverage/study/training_runner \
  --manifest /absolute/baseline/manifest.json \
  --baseline /absolute/baseline/results.jsonl \
  --output build/training-coverage/rate
```

The manifest supplies named, hashed PNM inputs. The baseline ledger supplies
`sa` (search + ANS) output paths and hashes. The runner asserts exact baseline
file identity, then freezes each baseline winner's RCT, transform sequence,
and weighted-predictor settings. It learns and encodes all four combinations
of 4096/65536 samples and 16/64 leaves for each learner. It then runs normal
search at 65536/64 for each learner, preserving the complete baseline search
and its tie order. Every emitted file is decoded with pinned libjxl and compared
to every original integer sample, including hidden RGB under alpha.

Each input has eleven actual encodes. `encoded_bytes` is the measured size;
`retained_bytes` is the smaller of that actual file and the actual baseline.
The latter is an explicit fallback comparison, not another encode or an
inferred factorial cell. Coverage is measured for fixed-transform training;
search rows leave the coverage object zeroed because search trains many trees.
Node arrays retain the complete resolved tree. Weighted settings remain the
native defaults and are asserted unchanged during fixed-transform trials.

The `kLegacy` learner keeps all 14 predictors and its original split cost and
sampling. `kQuantized` is reference-inspired, not a bit-exact libjxl learner:

- Weighted/Gradient candidates and HybridUint 4/1/2 residual proxy.
- Up to 256 globally quantized bins per actual sampled property, reused by all
  nodes; per-property histograms avoid repeated per-node sorting.
- Reference-shaped sampling-aware split threshold, predictor-change cost,
  static/constant and no-Weighted preferences, and 1/4096 minimum probability.
- Integer Q12 entropy estimates and best-first leaf allocation under a hard cap.
- A common per-slice floor up to 1024 samples, reduced to fit the hard budget,
  followed by area-proportional quotas and deterministic jittered strata.

The reference instead uses proxy pixel/difference distributions, coordinate
and Weighted-specific bins, geometric sampling, floating-point estimates and
threshold-driven depth-first construction. Its `total_pixels` bookkeeping is
also different: here the denominator is exactly the transformed sample count.
These differences are intentional and must not be described as libjxl parity.

A single timed encode is diagnostic only. Concurrent workers, decoding, cold
state and profiling prevent these numbers from establishing throughput claims.

## Tree checkpoint and larger-budget study

The private CPU workflow adds `tree_checkpoints=true`. One training pass retains
16/32/64/... leaf trees through the requested cap, plus its final tree on early
stopping or a non-power-of-two cap. Search first encodes the final tree, then the
intermediate checkpoints. Pre-existing final-tree candidates have priority over
checkpoints on equal complete sizes, even when that final candidate is evaluated
later. The original default search remains available. Public defaults are unchanged.

A checkpoint contains the same tree as an independent lower-cap run on the same
sample set. The optional `TreeLearningCheckpoints` output and corresponding
storage bounds account for live snapshots during both learning and encoding.
Increasing the leaf cap with checkpoints preserves the lower checkpoints;
increasing the sample budget changes the training set and has no corresponding
per-image monotonicity guarantee relative to another experimental sample budget.

Build into a fresh directory (for example `build/training-checkpoints`) using
the commands above with the new directory substituted. The two native phases are:

```sh
python3 tools/modular_training_study/checkpoints.py \
  --runner build/training-checkpoints/study/training_runner \
  --manifest /absolute/baseline/manifest.json \
  --prior /absolute/training-coverage/rate/results.jsonl \
  --output build/training-checkpoints/fixed --phase checkpoints
python3 tools/modular_training_study/checkpoints.py \
  --runner build/training-checkpoints/study/training_runner \
  --manifest /absolute/baseline/manifest.json \
  --prior /absolute/training-coverage/rate/results.jsonl \
  --output build/training-checkpoints/search --phase checkpoint-search
```

The fixed phase measures 64K/256K samples crossed with 16/32/64/128/256 leaf
caps, using the existing learner and frozen baseline transforms. Each sample
budget trains once to 256 leaves and emits the saved checkpoints; training time
is recorded once per pass in stderr and individual file encoding time in JSON.
Early stopping can make several cap settings emit identical files.

The normal-search phase measures the saved 64K/64 control, then checkpoint search
at 64K/64, 64K/256 and 256K/256. Both phases reproduce the original default
baseline. Saved matching fixed and search controls must be byte-identical.
All output files receive independent sample-exact decoding.

For the reference comparison, reuse the frozen instrumented runner and data
from the separate reference attribution study. The driver checks its binary hash,
reruns default e9 and the earlier 64K/64 fixed-RCT control byte-identically, and
measures the complete 64K/256K by 64/256-leaf factorial. It verifies identical
reference transforms and channel layouts across learning budgets:

```sh
python3 tools/modular_training_study/checkpoint_reference.py \
  --reference-data /absolute/modular-libjxl-controls \
  --native-results /absolute/training-coverage/rate/results.jsonl \
  --output build/training-checkpoints/reference
python3 tools/modular_training_study/analyze_checkpoints.py \
  --data build/training-checkpoints \
  --prior /absolute/training-coverage/rate/results.jsonl \
  --output build/training-checkpoints/summary.json
```

Reference controls force global coding and the same RCT type chosen by the
native baseline. Other reference transforms, patch handling and channel layouts
remain the reference's own, verified unchanged across budgets. They are not an
assertion of identical complete native/reference transform pipelines. Reference
capacity uses the previously qualified threshold-targeting procedure; native
capacity uses best-first growth. See the retained reference-study manifest for
exact instrumentation and build provenance.

The analysis separates actual fixed-cap files, selection among measured
checkpoint files, and actual normal-search outputs. It never substitutes a
selection envelope for a newly measured full-search configuration.
# Complete-search profiling

With `GJXL_BUILD_BENCHMARKS=ON` and a pinned `GJXL_MODULAR_ORACLE_BUILD`, the main
build provides `gjxl_modular_search_profile`. It runs one complete CPU ANS search
with the original learner, the requested sample/leaf budgets, retained baseline
candidates, and checkpoint selection:

```sh
build/release/gjxl_modular_search_profile input.pam output.jxl 262144 256 4 > result.json
```

The wall timer brackets `EncodeModularImage`, including all learning and candidate
encodes. PNM input, pinned decoder validation, and output-file writing are outside
that timer. Successful output requires exact decoded samples and released workflow
resources. The JSON records the full winning tree/transform policy, encoded size,
candidate count, resource peaks, top-level phases, detailed training attribution,
and entropy diagnostics.

`training.stages` contains disjoint nested intervals inside `stages.training`;
do not add both levels into an end-to-end total. Sample gathering includes quota
preparation and retained training samples. It also includes the complete weighted
predictor scan when completed features are not shared with learning. The
split-scanning interval includes histogram updates and candidate
threshold scoring. Tree control excludes the separately attributed preparation,
histogram, sorting, and scanning intervals. Entropy diagnostics retain their
existing nesting: prefix value aggregation is inside prefix configuration search.

Append `--no-profile` for an ordinary control call. Its profile counters are zero
and its policy is null; encoded bytes and decode validation remain available.
Compare complete files to a profiled call before interpreting the attribution.
Retain binary/input/output hashes, build revision and flags, exact commands, and
machine activity alongside captures. One capture on a busy desktop is diagnostic,
not a qualified throughput comparison.

In a Metal-enabled build, append `--metal` to measure the complete hybrid search:

```sh
build/release/gjxl_modular_search_profile input.pam output.jxl 262144 256 4 --metal > result.json
```

The backend is initialized before the timer and its setup duration is recorded
separately. CPU transforms and weighted-feature preparation, call-local cache
creation/replacement, GPU token evaluation, CPU entropy processing and output
publication all remain inside the encoder-call timer. Search policy and candidate
order are identical to the CPU control. `--metal --no-profile` is supported too;
actual `gpu_submissions` and `gpu_allocations` remain recorded with profiling off.

The JSON route is `hybrid-complete-search`. The `adaptive_gpu` object contains
feature preparation/reuse/eviction counts and GPU/fallback candidate counts.
`feature_seconds` spans both learning and token consumers; its
`learning_feature_seconds` subset is inside `stages.training`, and the remainder
plus `token_seconds` is inside `stages.tokens`. These are not additional phases
or device-only timestamps. The older `gpu_input_tokens` stage
belongs to the separate prescribed unpack/RCT/token route and stays zero here.
The CPU-only build rejects `--metal` explicitly. Compare complete bytes and the
entire winning policy, including weighted parameters, before interpreting paired
CPU/hybrid timings.

Every Metal search route lets learning borrow the same completed weighted
prediction map as token generation by default. Append `--no-shared-features` for
the unshared comparison control; `--shared-features` remains an explicit enable
switch. Sharing does not change the learner, candidates or entropy policy.
The cache retains one transformed configuration per encoder call. See
[`gpu-modular-shared-features.md`](../../docs/gpu-modular-shared-features.md) for its
ownership contract, validation and paired measurements.

With sharing enabled, `feature_seconds` includes preparation requested by either
consumer. `learning_feature_seconds` is the subset within `stages.training`;
subtract it from `feature_seconds` before combining the remainder with
`token_seconds` under `stages.tokens`. It is separate from the nested training
stages, including `sample_gathering`. `feature_preparations + feature_reuses`
equals GPU token requests plus `learning_feature_uses`. The learner reports full
population counts in `weighted_prediction_samples` and `reused_prediction_samples`;
their sum is `population`. In the search route every prepared map is weighted,
so `adaptive_gpu.feature_samples + training.weighted_prediction_samples` counts
the weighted predictor evaluations actually performed by both producers.

The older frozen-study tools retain their original route assumptions; use them
with their recorded binaries and revisions. Use `shared_features.py` for new
off/on paired captures:

```sh
python3 tools/modular_training_study/shared_features.py capture build/shared-features-pilot \
  --binary build/release/gjxl_modular_search_profile \
  --inputs-manifest /path/to/entropy-pipeline-qualified/manifest.json
python3 tools/modular_training_study/shared_features.py audit build/shared-features-pilot \
  --output build/shared-features-results.json
```

This runs three small image pairs and two pairs per original 12 MP image, using
one frozen binary with explicit `--no-shared-features` and `--shared-features`
arms. New manifests use schema 2; auditing still accepts the original schema 1
pilot, whose frozen binary defaulted to unshared features. It retains the exact
source patch relative to HEAD, build configuration, input and output hashes,
complete candidate ledgers, and power/thermal/process observations. The audit
checks byte identity, every candidate decision/cost, training work, eliminated
prediction evaluations, timing nesting and unchanged admission bounds.

The frozen one-pair pilot and validation command are documented in
[`gpu-adaptive-modular-hybrid.md`](../../docs/gpu-adaptive-modular-hybrid.md).
`analyze_adaptive_hybrid.py` verifies the saved hashes, routes, complete-file and
policy parity, training work counters and cache accounting before regenerating
the report. It reports timings as diagnostic observations, alongside exact
ANS attribution and the increased managed backing/admission requirement.

`--metal-ans` enables the resident-token hybrid route plus exact GPU ANS cost
evaluation; use it instead of `--metal`. It preserves the CPU model search,
candidate pruning and ties. Its route is `hybrid-ans-complete-search`.
`adaptive_gpu.ans_gpu_groups` counts successful cost submissions,
`ans_gpu_states` counts section/candidate states, and
`ans_cpu_fallback_groups` counts explicit pre-dispatch scalar fallback. The
existing `entropy_work.ans_token_cost_ns` includes model upload, GPU wait and
conversion inside `stages.model`. Actual `gpu_submissions` is token-generation
submissions plus cost-group submissions. `--metal-ans --no-profile` also works;
backend counts remain enabled while optional attribution counters are zero.


The repeated CPU/exact-ANS hybrid qualification and regeneration command are in
[`gpu-adaptive-modular-ans.md`](../../docs/gpu-adaptive-modular-ans.md).
`analyze_adaptive_ans.py` verifies all 100 retained artifacts, the 21-capture
matrix and pair order, complete-file/policy/training parity, routes and resources
before reporting per-image medians/ranges and actual paired ratios. It rejects
incomplete captures and corrupted checksums. The small-section CPU crossover is
reported by `ans_cpu_small_groups`, a subset of `ans_cpu_fallback_groups`.


`--metal-learn` adds exact original-learner split statistics to `--metal-ans`.
Its route is `hybrid-learning-complete-search`. CPU property sorting, threshold
selection, greedy growth, checkpoints and quantized learning remain on the CPU.
The `tree_prepare_seconds` and `tree_evaluate_seconds` counters are nested within
training/split scanning; do not add them to total training time. Preparation
copies immutable residual codes and count costs once per eligible learning call;
it does not submit device work. Each `tree_gpu_groups` count is one submission
containing histogram and cost dispatches. Total submissions therefore equal GPU
token candidates plus ANS groups plus learner groups. Small-leaf properties are
an explicit subset of `tree_cpu_fallback_properties`, and quantized learning calls
are reported separately by `tree_cpu_unsupported_learning_calls`.
An eligible property with no valid threshold submits no GPU work and needs no
scalar histogram scan. It remains in the semantic `sorted_properties` count, so
GPU groups plus scalar-fallback properties need not equal all sorted properties.

The operation, validation scope and performance qualification requirements are in
[`gpu-modular-learner-statistics.md`](../../docs/gpu-modular-learner-statistics.md).
The opt-in `gjxl_modular_tree_statistics_benchmark root|tree REPETITIONS` runs
production-learner crossover probes with exact tree/checkpoint comparisons. Its
synthetic timings are diagnostic, not complete-image throughput measurements.


The completed [three-route learner qualification](../../docs/gpu-adaptive-modular-learning.md)
uses frozen revision `34586ea`, two CPU/token-ANS/learner triplets per original
full-resolution image, and three smoke captures. Regenerate it with:

```sh
python3 tools/modular_training_study/analyze_adaptive_learning.py \
  --capture build/adaptive-learning-qualified-20261011 \
  --baseline build/adaptive-baseline-20261011 \
  --json docs/gpu-adaptive-modular-learning-results.json \
  --markdown docs/gpu-adaptive-modular-learning.md
```

The analyzer checks all 89 artifact hashes, the 21-capture matrix and route order,
input/output identity, complete policies and semantic training work, native
submission accounting and resource bounds. It also verifies the saved smoke CPU
files and default libjxl reference sizes. Incomplete captures and same-size
corrupted outputs are rejected. Performance statistics use only this frozen
three-route capture; earlier ANS and synthetic crossover runs remain separate.
