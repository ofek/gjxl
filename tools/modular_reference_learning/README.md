# Reference Modular learner bridge

Research-only first milestone of the
[learner replacement plan](../../docs/modular-tree-learning-cost.md#recommended-implementation-order).
Production sources and defaults remain unchanged. The reference is libjxl
`e8ff09762481785938d8e4e01333ed3917571161`; qualified native archives come from
`f27a797` (`perf/modular-ans-estimates`).

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

The initial pilot demonstrated exact stock bytes and replay, matching bridge
tokens, and exact reconstruction on all three inputs. Best-predictor bridge
trees were larger than the current native winners; all-predictor bridge trees
were smaller on all three. This does not select a production predictor policy
or establish full-corpus compression parity. Keep both controls for the wider
comparison.
