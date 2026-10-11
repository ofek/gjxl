# ANS selection study

Research-only paired comparison of exact and estimated ANS alphabet selection.
Both use the same configuration and histogram searches, tree, transforms and
context partition in fixed trials. No public defaults are changed.

Build the CPU Release tree with `GJXL_MODULAR_ORACLE_BUILD` set to the clean
pinned reference build, then:

```sh
cmake -S tools/modular_ans_study -B build/ans-selection/study \
  -DCMAKE_BUILD_TYPE=Release -DGJXL_BUILD="$PWD/build/ans-selection" \
  -DREFERENCE_SOURCE=/absolute/libjxl-source \
  -DREFERENCE_BUILD=/absolute/libjxl-build
cmake --build build/ans-selection/study -j 4
python3 tools/modular_ans_study/run.py \
  --runner build/ans-selection/study/ans_runner \
  --checkpoint-data /absolute/training-checkpoints \
  --input-manifest /absolute/palette-context/rate/manifest.json \
  --profile-record /absolute/docs/libjxl-modular-entropy-profile-results.json \
  --output build/ans-selection/fixed --phase fixed --jobs 1
```

Repeat the driver with `--phase search` and a fresh `search` output directory
for all 16 adaptive checkpoint cases; two concurrent workers are allowed for
this rate-only sweep. Use `--phase timing --jobs 1` and a `timing` directory for
serial repeated adaptive timing on photo-macan, alpha-ria and gray-patches.
Run phases sequentially to keep the timed phases free of study concurrency.

Fixed cases: Gradient/RCT6 (identity RCT for gray) on the 16-input corpus and
three saved 12 MP photos, plus saved 256K/256 checkpoint winners on the 16-input
corpus. Weighted parameters are the native defaults, as in the saved study.
The driver imports the recorded transform/tree policy without relearning it.

Each fixed case performs one warmup pair and four measured pairs of complete
CPU encodes with four allowed participants, alternating exact/estimated order.
Profiling is disabled in these timed calls. Separate diagnostic calls collect
model-stage time and must reproduce ordinary bytes. The first file per mode is
independently decoded, including hidden RGB under alpha; all repetitions must
match it byte-for-byte. Diagnostic calls are excluded from ordinary timing.

Then tokenize the same prescribed policy once, freeze the token buffers and
prefix-derived context partition, and run the same paired schedule for isolated
ANS construction. These isolated calls run on the calling thread without an
admitted CPU worker group; complete encodes allow four participants. A separate diagnostic pair records candidate counts and exact
simulation time. Estimated mode must build one width, perform zero simulation,
and retain the exact mode's configuration/histogram candidate counts. The token
hash includes stream boundaries, values and contexts and is a diagnostic FNV
fingerprint, not a cryptographic artifact identifier.

Adaptive search uses 262,144 samples, a 256-leaf cap and complete-file checkpoint
selection. The rate sweep runs one pair per input without warmups. Repeated
timing runs one warmup pair plus three measured pairs for each pilot case.
Both adaptive phases enable the existing Modular profile to retain the resolved
policy, candidate count and model/learning timings. These timings are instrumented
and are reported separately from unprofiled fixed-decision measurements.

Manifests bind inputs, the runner binary, source files, prior ledgers, exact
controls and measurement settings. Every saved learned-policy exact file and
adaptive exact file must match the previous checkpoint search SHA-256. The
three large Gradient exact files must match the earlier GPU-token qualification.
This is an all-CPU experiment; it does not establish GPU speedup or equal-rate
end-to-end speedup versus libjxl.

Use a fresh output root for a new study and preserve the old qualification
artifacts. Completed cases can resume only with the same manifest, binary,
source revision and tool-source snapshot. Retain incremental
`runner.jsonl` and `stderr.log` when a case fails. Never substitute a partial
trace for a completed case or mix binaries in one output directory.

After all phases complete, validate saved hashes and summarize:

```sh
python3 tools/modular_ans_study/analyze.py \
  --root build/ans-selection \
  --prior-report docs/modular-tree-checkpoints-results.json \
  --reference-ledger /absolute/modular-rate-20261010/results.jsonl \
  --output docs/modular-ans-selection-results.json
```

The analyzer verifies the implementation qualification, measured manifests,
retained output/trace/input hashes, exact controls, saved e9 files, stable
repetitions and diagnostic candidate counts. It also requires the adaptive
pilot to reproduce the adaptive rate sweep. Speed ratios divide the sums of
per-case medians, with warmups and fixed diagnostic calls excluded. Individual
samples and ranges remain in the JSON. Concurrent search timings are explicitly
diagnostic, and single-session desktop timings have no confidence interval.

The original measured tool sources are retained in
`build/ans-selection/measurement-source/`; the final analyzer and README were
added/expanded for reporting. The runner and driver do not change between
phases. On a fresh reproduction the analyzer accepts the current source when
it matches the manifest and no frozen copy exists. Keep host-activity snapshots
and linked-library hashes alongside the measurements when available. Saved
artifacts are local ignored files; the committed report binds their hashes.
