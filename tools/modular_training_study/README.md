# Native Modular training coverage study

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
