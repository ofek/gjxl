# Native fixed-gradient transform cost

The private native `EstimateModularTransformCost` now matches pinned libjxl's
SIMD `EstimateCost` on the qualified Apple M4 Pro controls. It provides the
cost primitive used for palette acceptance and RCT selection, without a tree
fit, token vector, ANS model or heap allocation.

This completes the fixed-gradient estimator prerequisite in the
[reference search plan](modular-reference-search-policy.md). The subsequent
[weighted-cost milestone](modular-weighted-cost.md) also qualifies that separate
estimator. Autonomous transform/scope selection and its
integration with learning remain. Encoder defaults are unchanged.
The [results ledger](modular-selection-cost-results.json) binds the tests,
captured-plane comparisons and source/build snapshots. This milestone adds
no compression-rate or timing claim.

## Cost and storage contract

[`selection_cost.h`](../src/codec/modular/selection_cost.h) accepts borrowed,
possibly strided `ModularChannelView` planes. The implementation follows
reference revision `e8ff09762481785938d8e4e01333ed3917571161`:

- Clamped-gradient residuals, with every first-row sample in context zero.
  Subsequent rows use 17 contexts derived from the neighborhood's value range.
- Default 4/2/0 HybridUint symbols and extra bits, with separate histograms
  for each channel. This differs from the learner's 4/1/2 proxy.
- Reference `FastLog2` and four-lane Shannon accumulation, without the
  learner's probability clipping. Integer histogram cost and accumulated
  fractional remainders retain the reference's final truncation/conversion.
- Unsigned residual packing and extra-bit accumulation, including the SIMD
  reference's modulo-2^32 extra-bit sum. Integer symbol extraction avoids
  floating-point rounding near large powers of two.

The native NEON path processes four pixels at a time and uses scalar tails;
it never reads beyond the supplied plane backing. The portable path retains
the same four-lane entropy arithmetic, including explicit fused operations.
Exact equality is qualified against the Apple AArch64 reference, not every
possible libjxl SIMD width. The optional exact-reference CTest targets are
therefore restricted to Apple AArch64; the native contract test is portable.

The fixed histogram array occupies **8,772 bytes** on this target, reused per
channel. That is histogram scratch, not the entire stack frame or process RSS.
The estimator allocates no heap storage and copies no pixels. It validates
all nonempty views before scanning and preserves the prior output on error.
Empty planes are ignored. Each nonempty plane is limited to `INT32_MAX`
samples, with an additional checked combined-population bound.

## Qualification

| Check | Result |
|---|---:|
| Generated cases per backend/build | 396 |
| Exact generated costs per backend/build | 2,372 |
| Exact 19-RCT rankings per backend/build | 104 |
| Native CTest entries per build | 3 / 3 |
| Research CTest entries per build | 10 / 10 |
| Captured image/policy cases | 11 |
| Canonical captured stream slots / populated slots | 416 / 391 |
| Exact captured costs | 6,724 |
| Exact captured 19-RCT candidate sets and winners | 332 |
| Sanitizer captured cases identical to Release | 4 / 4 |
| Exact costs / RCT candidate sets in that subset | 1,343 / 66 |

Both NEON and forced-portable implementations pass the generated controls in
Release and ASan/UBSan. Inputs cover one to four channels, widths around vector
boundaries, one-row/one-column images, minimally backed strided planes, flat
and skewed populations, random 16-bit values, signed 24-bit values and full
32-bit residual edges. Four larger RGBA fixtures exercise 1,025×769 planes.
RCT comparisons use values whose transformed samples remain representable;
full-32-bit residual tests qualify the estimator separately.

For each eligible generated case, all 19 native RCT candidates match reference
pixels before their costs and stable ordering are compared. The replay checks
the same candidates and strict-improvement winner order on captured planes.
Allocation poisoning remains armed through every native estimate. Contract
tests also check invalid later channels, preserved outputs and empty planes.
Replay rejection tests cover missing/reordered candidates, false equality,
changed scope, invalid cost decomposition and changed tie winners.

The 11 capture cases are the five established images in automatic and buffered
mode, plus the 16-bit RGBA ramp in automatic mode. **These are already
transformed e9 planes.** Applying an additional hypothetical RCT exercises
the estimator and candidate order; it does not reproduce the original
preselection inputs or establish equality of complete encoder decisions.
No new files are encoded by this replay mode.

The validation manifest binds **2,887 artifacts**, including source snapshots,
native/reference archives, tested binaries, compiler settings, logs and input
captures. Native builds use ordinary compiler math/vectorization settings.
The reference archives remain uninstrumented; reference vptr checks and the
unavailable macOS LeakSanitizer are excluded. The native estimator and research
harness are instrumented. Correctness builds/checks overlap, so their elapsed
times are not benchmarks.

## Remaining selection work

Reference `try_palettes` accepts a proposal when `EstimateCost` does not
increase. RCT selection instead requires strict improvement in the ordered
19-candidate list. This estimator can support both decisions, but the native
policy must still reproduce the inputs and transform construction.

The palette source audit found additional requirements beyond GJXL's existing
lexicographically ordered explicit dictionary: implicit colors, image-order
discovery, frequency-dependent luma ordering, and the actual retained palette
size. An implicit-only palette can have no explicit entries, and frequently
used implicit colors can increase the explicit count beyond the requested
capacity. The subsequent [reference palette port](modular-reference-palette.md)
now supports these cases, including empty metadata and widths up to 1,213, and
qualifies them against the actual pinned builder and independent full decodes.

The subsequent [WP cost port](modular-weighted-cost.md) qualifies five predictor
presets, 34 property contexts, overflow handling, float accumulation and
winners directly. With palette construction also qualified, next compose
palette/RCT/WP and shared/local scope selection with preparation and one fit
per selected coding scope. Keep exhaustive new-learner controls and the
whole-stage time, rate, resource and held-out gates before default promotion.

## Reproduction

Build native targets `gjxl_modular_selection_cost_test`,
`gjxl_modular_selection_cost_reference_test` and
`gjxl_modular_selection_cost_scalar_reference_test` in fresh Release and
ASan/UBSan directories with `GJXL_MODULAR_ORACLE_BUILD` set to the pristine
pinned reference build. Run `ctest -R '^modular_selection_cost'` and retain
its output and `Testing/Temporary/LastTest.log`.

Build the research runner against the corresponding native archives, then
run its ten CTest entries. `selection_cost_replay.py run` takes `--binary`,
`--coding-study` and a new `--output` directory; `--cases` selects the four
sanitizer cases listed in the results ledger. It freezes the binary and
sources, binds the existing capture inputs, and checks every result.
`selection_cost_replay.py analyze` revalidates that individual study.

The combined qualifier additionally requires a validation manifest binding
the native/research logs, native test binaries, archives and study source
snapshots. The saved instance is `build/selection-cost-validation`; it can be
rechecked without building or encoding:

```sh
python3 tools/modular_reference_learning/analyze_selection_cost.py \
  --study build/reference-learning/selection-cost-qualified \
  --sanitizer-study build/reference-learning/selection-cost-qualified-asan \
  --validation build/selection-cost-validation \
  --output /tmp/gjxl-selection-cost-rechecked.json
```

The analyzer refuses to overwrite an existing report. Raw build/capture
artifacts remain local; the compact report and qualification code are tracked.
