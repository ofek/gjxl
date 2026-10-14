# Modular learning cost study

Research-only attribution and ablations, based on `f27a797`. Production encoder
sources and defaults are unchanged. Generated translation units override the
corresponding objects in the qualified archives at link time.

Native variants preserve the sampler, 32 per-node quantile thresholds, integer
cost proxy, best-first growth, and checkpoint selection, except where stated:

- `baseline`: all 14 predictors and the original indirect property sort.
- `two`: Gradient and Weighted in the entropy learner. The retained one/two-leaf
  learner and sample collection still compute all predictors.
- `keys`: sort contiguous `(property value, sample index)` records with exactly
  the original tie order. This scratch vector is **not budgeted** and is only a
  research prototype, not a candidate for direct production inclusion.
- `two_keys`: both of the preceding changes.
- `skip_constant`: skip a property's sort and histogram sweep if all values in
  that node are equal.
- `quantized`: the existing experimental quantized learner. This also changes
  sampling allocation, predictors, thresholds and scoring; it does not isolate
  the effect of binning.
- `unprofiled`: links the original learner/search objects from the qualified
  archives, with only external call timing and the existing coarse profile.
  The `control` phase measures instrumentation impact on the three pilot cases.

`fixed` freezes the saved winning RCT/transform sequence, retrains at 262,144
samples / 256 leaves, and selects among the resulting final tree/checkpoints.
It deliberately does not retain the saved winning tree as a fallback.
`search` runs the complete existing portfolio. Both use estimated ANS selection.
Every invocation decodes its first output sample-exactly, including hidden RGB,
and requires subsequent iterations to reproduce the bytes. Search controls
must also reproduce the prior estimated-ANS control SHA-256.

## Build

Paths below describe the retained local experiment. The native archive build is
the qualified `perf/modular-ans-estimates` build at `f27a797`. The reference source
is pinned at `e8ff09762481785938d8e4e01333ed3917571161`. Reference overrides and
adapter come from the byte-qualified attribution study at `f4300f4`; generation
records their exact hashes and leaves those files untouched.

```sh
python3 tools/modular_learning_cost/prepare.py --source . \
  --output build/learning-cost/overrides
python3 tools/modular_learning_cost/prepare_reference.py \
  --previous-data /Users/yunhocho/GitHub/gjxl-modular-libjxl-study/build/modular-libjxl-controls \
  --previous-tools /Users/yunhocho/GitHub/gjxl-modular-libjxl-study/tools/modular_libjxl_study \
  --output build/learning-cost/reference-source
cmake -S tools/modular_learning_cost -B build/learning-cost/bin \
  -DCMAKE_BUILD_TYPE=Release \
  -DGJXL_BUILD=/Users/yunhocho/GitHub/gjxl-modular-ans-estimates/build/ans-selection \
  -DREFERENCE_SOURCE=/Users/yunhocho/GitHub/gjxl/third_party/libjxl \
  -DREFERENCE_BUILD=/Users/yunhocho/GitHub/gjxl/build/pinned-libjxl \
  -DSTUDY_DATA="$PWD/build/learning-cost" \
  -DREFERENCE_STUDY_DATA="$PWD/build/learning-cost/reference-source"
cmake --build build/learning-cost/bin -j4
```

## Run serially

```sh
python3 tools/modular_learning_cost/run_native.py \
  --binaries build/learning-cost/bin \
  --ans-study /Users/yunhocho/GitHub/gjxl-modular-ans-estimates/build/ans-selection \
  --output build/learning-cost/profile-cpu --phase profile
python3 tools/modular_learning_cost/run_native.py \
  --binaries build/learning-cost/bin \
  --ans-study /Users/yunhocho/GitHub/gjxl-modular-ans-estimates/build/ans-selection \
  --output build/learning-cost/pilot-cpu --phase pilot
python3 tools/modular_learning_cost/run_native.py \
  --binaries build/learning-cost/bin \
  --ans-study /Users/yunhocho/GitHub/gjxl-modular-ans-estimates/build/ans-selection \
  --output build/learning-cost/rate --phase rate
python3 tools/modular_learning_cost/run_reference.py \
  --binary build/learning-cost/bin/reference_runner \
  --previous-data /Users/yunhocho/GitHub/gjxl-modular-libjxl-study/build/modular-libjxl-controls \
  --output build/learning-cost/reference
python3 tools/modular_learning_cost/run_native.py \
  --binaries build/learning-cost/bin \
  --ans-study /Users/yunhocho/GitHub/gjxl-modular-ans-estimates/build/ans-selection \
  --output build/learning-cost/control --phase control
python3 tools/modular_learning_cost/analyze.py \
  --root build/learning-cost \
  --ans-study /Users/yunhocho/GitHub/gjxl-modular-ans-estimates/build/ans-selection \
  --output docs/modular-tree-learning-cost-results.json
```

Manifests bind inputs, binaries and prior ledgers. Resume rejects changed
manifests or output/trace hashes. Native timing uses one warmup and two measured
iterations per variant, with variant order rotated between images. Reference
timing uses one warmup and three measured fresh-process iterations, interleaving
the configuration order. Native encoding allows four CPU participants; its
learner is serial. The reference runner installs no thread pool.

Fine native attribution uses steady-clock wall timers. CPU time is separately
recorded around each learning pass and entropy fit; it excludes process
suspension/descheduling but does not correct for frequency or cache variation.
Reference fit and complete-learning events record both time bases. All timings
are instrumented diagnostics from an uncontrolled desktop session, not release
throughput qualification. Cross-encoder sample sets, transforms, growth and
objectives differ even when a sample cap or predictor count is equal.

The first wall-only pilot (`build/learning-cost/pilot`) contained 920-second and
1,030-second pauses and is excluded from timing conclusions. The CPU-timed pilot
was captured separately; the old raw traces are retained for audit.
