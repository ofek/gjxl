# Reference Modular learner bridge

Research-only first milestone of the
[learner replacement plan](../../docs/modular-tree-learning-cost.md#recommended-implementation-order).
Encoder defaults remain unchanged. The native histogram fitter is a private
implementation under differential qualification. The reference is libjxl
`e8ff09762481785938d8e4e01333ed3917571161`; qualified native archives come from
`f27a797` (`perf/modular-ans-estimates`).

## Reference policy and scope controls

`reference-policy INPUT OUT trace|timing auto|buffered` changes only the input
buffering setting when `buffered` is requested; automatic mode preserves the
stock e9 API sequence. Generate/build in fresh `policy-generated` / `policy-bin`
directories using the build recipe below with those paths substituted. Keep
older generated sources intact: existing ledgers bind their hashes.

```sh
python3 tools/modular_reference_learning/run_policy_study.py \
  --binary build/reference-learning/policy-bin/reference_learning \
  --corpus build/reference-learning/corpus \
  --output build/reference-learning/reference-policy \
  --inputs photo-macan photo-flower alpha-flower chessboard splines --repeats 3
python3 tools/modular_reference_learning/test_policy_trace.py \
  --case build/reference-learning/reference-policy/photo-macan-auto--1
python3 tools/modular_reference_learning/analyze_policy_study.py \
  --study build/reference-learning/reference-policy \
  --validation build/reference-learning/reference-policy-validation \
  --output docs/modular-reference-policy-results.json
```

Use a new output directory; the runner does not overwrite or silently resume a
partial study. It freezes the binary, exact sources, generated hooks and build
configuration, and binds inputs and archives by hash. Omit `--validation` if the
archived sanitizer controls are unavailable. The negative tests above use the
nonempty photographic global-tree trace.

Trace mode writes `streams/learning-N.json` and `.i32le` payloads for every
successful learning call. Included channels have tight signed 32-bit rows;
excluded channels retain descriptors but no pixel bytes. Metadata preserves
stream-ID holes, every learning option, float bits, WP coefficients, transforms,
multipliers and the raw tree. Payloads are read back and checked sample-for-sample.
`coding-N.json` records global/local coding scope and header settings separately.
Trace overhead is explicitly flagged and excluded from timing analysis. Local
scope is captured at the call site, not inferred from the number of images.
See the [ablation report](../../docs/modular-reference-policy-ablation.md) for
results and the distinctions between workflow, learning and fitting comparisons.

## Native sample preparation

`prepare-check INPUT POLICY FIXTURE best|all [OUTPUT]` borrows native transformed
stream slices, prepares compact samples, checks every captured sample field,
fits the native tree and checks every raw node field. An optional output path
also encodes through native estimated ANS and verifies an independent exact
decode. The corpus runner checks complete byte identity with the frozen bridge:

```sh
python3 tools/modular_reference_learning/run_native_preparation.py \
  --binary build/reference-learning/bin/reference_learning \
  --corpus build/reference-learning/corpus \
  --output build/reference-learning/native-preparation
python3 tools/modular_reference_learning/analyze_native_preparation.py \
  --study build/reference-learning/native-preparation \
  --corpus build/reference-learning/corpus \
  --validation build/reference-learning/native-preparation-validation \
  --output docs/modular-native-sample-preparation-results.json
```

Use a new output directory. `--inputs` restricts a diagnostic run; omit
`--validation` when the archived final build validation is unavailable.
`preparation-self-test` runs 101 generated reference/native comparisons plus
the explicit pinned-reference weight-overflow rejection regression.
`native_preparation_test` checks the native admission, atomic publication and
managed allocation/storage contract. Both and the ledger tests run under CTest.
See the [preparation report](../../docs/modular-native-sample-preparation.md) for
profile limits and timing caveats. Defaults and candidate selection are unchanged.

## Paired complete learning

`paired-learning INPUT POLICY FIXTURE best|all REPEATS` times preparation plus
fitting on identical immutable transformed planes and strides. Both paths warm
once, then alternate order; samples are checked in the native warmup and every
measured tree is checked after timing. Native admission/validation, allocation
and temporary sample release are included; image conversion and comparisons
are excluded. No encodes run in this mode.

```sh
python3 tools/modular_reference_learning/run_native_preparation.py \
  --binary build/reference-learning/bin/reference_learning \
  --corpus build/reference-learning/corpus \
  --output build/reference-learning/native-learning-pilot \
  --mode paired-learning --repeats 3 \
  --inputs photo-macan alpha-ria gray-patches
python3 tools/modular_reference_learning/analyze_paired_learning.py \
  --study build/reference-learning/native-learning-pilot \
  --corpus build/reference-learning/corpus \
  --validation build/reference-learning/native-learning-validation \
  --output docs/modular-native-learning-timing-results.json
```

Use a new output directory; optional `--variants best all` restricts predictor
controls. The validation argument binds archived Release/ASan test logs and
representative paired controls; omit it without that local archive. The
analyzer checks fixture/policy/input hashes against the frozen bridge, exact
checks, run order, expected shapes, raw stdout and timing nesting. Reference
non-fit time is a per-run difference, including sample release and wrapper
overhead, not an isolated preparation timer. See the
[timing report](../../docs/modular-native-learning-timing.md) for the limits.

## Native fitting replay

The harness also builds the dependency-free native fitter under ordinary
production compiler settings, separately from reference-only compiler flags.
`native_fit_test` checks the private native contract; `native-self-test` generates
129 independent differential fixtures. Both are registered with CTest.

```sh
ctest --test-dir build/reference-learning/bin --output-on-failure
python3 tools/modular_reference_learning/run_native_replay.py \
  --binary build/reference-learning/bin/reference_learning \
  --corpus build/reference-learning/corpus \
  --output build/reference-learning/native-paired \
  --mode paired --repeats 3
python3 tools/modular_reference_learning/analyze_native_replay.py \
  --study build/reference-learning/native-paired \
  --validation build/reference-learning/native-validation \
  --output docs/modular-native-histogram-learning-results.json
```

Use a new output directory; the runner refuses to overwrite any study. The
runner freezes the executable and exact source snapshots, including uncommitted
files, and binds inputs, archives and compiler settings by SHA-256. Optional
`--inputs` and `--variants` restrict a diagnostic study. `--mode verify` runs a
single native fit per fixture without warmup, for correctness rather than timing.
`--validation` additionally checks the archived final-build validation used by
the checked-in report; omit it when that local archive is unavailable. The
archive includes contract-test logs, scalar/sanitizer replay results and the
timed/final native-object equivalence check. Analyzer rejection tests are also
registered with CTest.

Direct modes are `native-verify FIXTURE`, `native-replay FIXTURE REPEATS`, and
`paired-replay FIXTURE REPEATS`. Copies/conversion are outside both timers;
native validation/allocations are inside. Every fit compares all raw tree fields.
The paired mode alternates execution order after one warmup each and measures
native storage separately during its warmup. See the
[native fitting contract](../../docs/modular-native-histogram-learning.md).

`validation-replay FIXTURE REPEATS` separately times repeated admission checks
on already converted samples. The validation-scan follow-up retains full
validation inside ordinary native fitting timers. Its combined report can be
recreated with:

```sh
python3 tools/modular_reference_learning/analyze_validation_scan.py \
  --paired build/reference-learning/native-validation-scan-paired \
  --verify build/reference-learning/native-validation-scan-verify \
  --scan build/reference-learning/validation-scan \
  --validation build/reference-learning/validation-scan-validation \
  --baseline docs/modular-native-histogram-learning-results.json \
  --output docs/modular-histogram-validation-scan-results.json
```

For scalar arithmetic qualification, configure a separate build with
`-DCMAKE_CXX_FLAGS=-DGJXL_HISTOGRAM_SCALAR=1`. This validates the scalar fallback
against the pinned four-lane reference; it does not qualify another reference
architecture or compiler. Existing ASan/UBSan builds also instrument the native
fitter and its contract/differential tests.

The original bridge analyzer resolves changed tracked tool sources through
their recorded capture commit. Frozen binaries, fixtures, archives and other
data must still match their live artifact hashes; they cannot resolve through
Git history. This preserves verification of older studies after tool evolution.

`prepare.py` checks the reference revision and pristine hooked files, records
their hashes, and generates these changes outside the reference checkout:

- Rename the original `ComputeBestTree` entry point so a capture wrapper can
  call its unchanged implementation.
- Add a layout-neutral friend declaration for sample serialization.
- Time the enclosing `LearnTree` call.
- Set the existing reference codec adapter to effort 9 instead of effort 1.

The little-endian versioned fixtures contain all quantized sample columns,
weights, token/extra-bit pairs, property and predictor order, dequantization
cuts and quantization maps, plus the exact float parameter bits, static ranges,
multiplier ranges and expected tree. They are captured before the fitter
partitions samples in place. Each capture is read back from disk and replayed
to the exact tree. The disposable dedup table must already be finalized.
Capture/replay overhead is excluded from the fit timer and explicitly included
in enclosing capture-run timers. Use `timing` runs for preparation-plus-fit
comparisons. The process peak RSS includes the entire harness, encoding and
validation; it is **not** an isolated learner scratch measurement.

The bridge copies GJXL's canonical stream slices into reference images, retaining
stream-ID holes, local channel numbering, metadata, shifted channel extents and
predictor reset boundaries. It uses the reference lossless e9 learning settings
with either Best (Weighted/Gradient) or Variable (all 14) predictors. Custom WP
headers and previous-channel properties are currently outside this adapter's
profile. No sample/leaf cap or repeated threshold search is imposed. Fixtures
have an explicit research admission limit of 16M samples and 512 MiB; captures
above native's current 1M sample ceiling are flagged, not truncated.

Imported trees must pass native property, predictor, offset, multiplier,
shape/depth and image-dependent size validation. For each bridge encode, tree
tokens and **every** residual/context token must match reference tokenization
on the same stream images. GJXL then encodes the imported tree through its
unchanged estimated-ANS path; reference decoding must reconstruct every source
integer, including hidden RGB. Timing/capture repeats must retain identical
compressed bytes. Saved stock e9 controls are compared by SHA-256 when available.

## Build

```sh
python3 tools/modular_reference_learning/prepare.py \
  --reference /Users/yunhocho/GitHub/gjxl/third_party/libjxl \
  --output build/reference-learning/generated
cmake -S tools/modular_reference_learning -B build/reference-learning/bin \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DGJXL_BUILD=/Users/yunhocho/GitHub/gjxl-modular-ans-estimates/build/ans-selection \
  -DREFERENCE_SOURCE=/Users/yunhocho/GitHub/gjxl/third_party/libjxl \
  -DREFERENCE_BUILD=/Users/yunhocho/GitHub/gjxl/build/pinned-libjxl \
  -DGENERATED_SOURCE="$PWD/build/reference-learning/generated"
cmake --build build/reference-learning/bin -j4
```

## Run

Run one study process at a time. This example captures the three pilot inputs
and interleaves one warmup and two measured fresh processes for each variant.
Omit `--inputs` for the complete saved 16-input corpus, or omit
`--timing-repeats` for capture/rate qualification only.

```sh
python3 tools/modular_reference_learning/run_study.py \
  --binary build/reference-learning/bin/reference_learning \
  --generated build/reference-learning/generated \
  --ans-study /Users/yunhocho/GitHub/gjxl-modular-ans-estimates/build/ans-selection \
  --reference-study /Users/yunhocho/GitHub/gjxl-modular-libjxl-study/build/modular-libjxl-controls \
  --output build/reference-learning/pilot \
  --inputs photo-macan alpha-ria gray-patches \
  --variants reference best all --timing-repeats 2
python3 tools/modular_reference_learning/test_capture.py \
  --binary build/reference-learning/bin/reference_learning \
  --fixture build/reference-learning/pilot/gray-patches-reference--1/fixtures/fit-0.bin
```

The manifest binds inputs, source files, binary, native/reference archives and
prior ledgers/policies. Completed cases are verified before resuming. A changed
manifest needs a new output directory; a partial case requires inspection.
Replay alone uses `reference_learning replay FIXTURE REPEATS`; it copies the
immutable captured samples outside each fitting timer and verifies every tree.

The [qualified results](../../docs/modular-reference-learning.md) cover 48
corpus encodes, 233 fitting captures/replays and 45 repeated pilot encodes.
The all-predictor bridge reaches +0.9487% aggregate bytes versus e9 on frozen
native transforms, but neither predictor policy is ready for promotion.

For boundary validation, configure a second build with the same path arguments
and `-DCMAKE_BUILD_TYPE=RelWithDebInfo` plus
`-DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-sanitize=vptr -fno-omit-frame-pointer'`.
Use `UBSAN_OPTIONS=halt_on_error=1` to run `test_capture.py`, a photographic
fixture replay, and an `alpha16-ramp` bridge encode with all predictors. The
vptr exclusion is required by the pinned no-RTTI archive; the linked archives
are uninstrumented. Preserve the sanitizer executable, CMakeCache.txt, logs
(`capture-tests.log`, `photo-replay.log`, `alpha16.log`) and output (`alpha16.jxl`)
in a separate validation directory so later builds cannot overwrite evidence.

The analyzer verifies hashes and completeness before producing the report:

```sh
python3 tools/modular_reference_learning/analyze.py \
  --corpus build/reference-learning/corpus \
  --timings build/reference-learning/pilot-qualified \
  --sanitizer-validation build/reference-learning/sanitizer-validation \
  --output docs/modular-reference-learning-results.json
```

Captured sample payloads are ignored research artifacts. The study copies its
executable into the output directory so rebuilding the harness does not replace
the binary bound to completed captures. The preliminary `pilot` was made before
final compiler/error-path qualification; use `corpus` and `pilot-qualified` for
the committed measurements.
