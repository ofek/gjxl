# Native preparation on actual e9 learning inputs

Native sample preparation now carries **per-stream weighted-predictor
parameters**. Release replay reproduces the reference samples and trees for
all **199 captured learning calls**, including **195 fits and four empty
calls**. This closes the preparation gap exposed by the
[reference policy ablation](modular-reference-policy-ablation.md), beyond the
earlier default-WP/frozen-native-transform bridge.

The [qualification ledger](modular-native-stream-preparation-results.json)
also checks the scalar fallback, sanitizer controls, managed storage and
default-WP encoded-byte regressions. This is preparation/fitting correctness
on captured transformed streams. It does not implement the per-stream coding
policy or change encoder defaults, and it supplies no new timing or rate claim.
The subsequent [prepared-stream encoder](modular-prepared-streams.md) now also
matches captured coding tokens and exact decodes, while exposing a downstream
rate gap that is separate from this matched learning result. The
[direct-ANS follow-up](modular-prepared-entropy.md) closes that conditional
automatic-cohort aggregate gap through an explicit private policy.

## Native contract and replay

`HistogramTrainingImage` carries seven WP coefficients and four weights.
Preparation validates each header before allocation and resets predictor state
for each channel, including palette metadata channels. Invalid coefficients
or weights leave the previous samples and storage-bound output unchanged.
The parameters add no separately allocated backing. The rest of the private
sampling/admission contract is unchanged, including explicit 16M research
sample admission and rejection of the pinned reference's weight-overflow case.

The reference adapter derives the effective header from each image's options.
It only applies a WP preset when that image's predictor uses WP; a nonweighted
predictor keeps the default header, including for property 15. Eighteen new
generated comparisons cover all five presets and mixed-preset batches under
Best, all-predictor and Gradient modes. Together with the prior cases, the
preparation differential suite now has **119 cases**.

The replay reconstructs learning calls from the frozen `.json` / `.i32le`
captures at pinned libjxl `e8ff09762481785938d8e4e01333ed3917571161`:

1. Preserve image/channel descriptors, metadata prefixes, empty group-ID
   slots, exact option float bits, multiplier boxes and each effective WP
   header. Excluded channels retain descriptors without pixel backing.
   A local call keeps its actual learning range `[0,1)`; its coding stream ID
   is not substituted as the learning group coordinate.
2. Recompute reference preparation and require its raw tree to match the
   original capture. Native weights, residual token/extra-bit columns,
   property bins/cuts/order, predictor order, threshold/penalty bits and static
   and multiplier ranges must then match the recomputed reference fixture.
3. Fit the native tree and compare every raw node field with the original
   captured tree. Empty calls must produce the same trivial tree without a
   reference fitting event. Check preparation and fitting allocations against
   their respective managed bounds.

Pixels are already transformed. Reapplying transforms or using entropy options
is unnecessary for learning replay; their original metadata remains bound in
the full source captures. Tight captured rows are loaded into new aligned
planes. Original reference strides are not recorded, so this does not reproduce
the memory layout or timing of the original encode.

The versioned text projection avoids a new JSON dependency in the C++ harness.
Both sides validate ranges and slice lengths; the reader rejects truncated or
trailing data, inconsistent WP headers/presets and unsupported per-image
sampling/growth differences. The payload limit is 512 MiB, with a conservative
1 GiB bound on reconstructed aligned planes. This is a research input format,
not a codec API. Previous-channel properties remain outside the profile.

## Qualified coverage and regression found

| Captured input | Configurations | Learning calls | Fits |
|---|---|---:|---:|
| photo-macan | auto, buffered | 2 | 2 |
| photo-flower | auto, buffered | 56 | 55 |
| alpha-flower | auto, buffered | 56 | 55 |
| chessboard | auto, buffered | 18 | 17 |
| splines | auto, buffered | 66 | 65 |
| alpha16-ramp | auto | 1 | 1 |

Release and the corrected scalar fallback match all 199 calls. ASan/UBSan
matches 19 calls / 18 fits covering photo-macan, chessboard's local palette
trees and true 16-bit RGBA. All eight CTest entries pass in each build,
including 119 preparation and 129 fitting differential cases, native allocation
failure/publication tests and replay corruption/evidence tests.

The broader scalar replay exposed an existing rounding bug on alpha-flower's
automatic learning event 46. Samples matched, but raw tree node 639 differed.
AppleClang had contracted the scalar entropy accumulator's multiply/subtract
into fused instructions, unlike the reference's separate operations. A
function-scoped `FP_CONTRACT OFF` restores that rounding boundary while retaining
the explicit FMA polynomial in `FastLog2`. The failure reproduces with the
frozen old scalar executable; the corrected case and complete matrix match.
The failed partial study is retained and excluded from passing qualification.
The normal Release executable is byte-identical before and after this
scalar-only correction.

Eight default-WP bridge cases are also rechecked in both Release and scalar
builds: photo-macan, chessboard, splines and alpha16-ramp, each with Best and
all predictors. Every prepared sample, raw tree and complete native ANS file
matches its prior control, and every independently decoded source integer
matches. These are regressions against the original bridge, not encodes of
the newly captured e9 stream policies.

The largest actual-stream preparation peak is **317,440,896 bytes**, for the
buffered alpha photo's 8,938,244 Best-predictor samples. Its largest local call
uses **5,893,436 bytes**. Fitting backing is **767,152 bytes** per call under
the fixed private capacity. These are component-managed allocations, excluding
borrowed planes, reference fixtures and allocator bookkeeping; fitting also
excludes the prepared columns already retained by preparation. They are not
whole-process or whole-encoder peak measurements.

The frozen studies retain executables, native/tool sources, generated reference
hooks, compiler commands, raw outputs and SHA-256 bindings to all source
captures. The analyzer checks exact projection, complete call coverage, original
fit populations, binary identity and final test logs. Sanitizers cover native
objects, the harness and generated reference units; linked archives remain
uninstrumented, and vptr is excluded for the no-RTTI reference. Correctness
runs may overlap independent builds/tests; their times are not performance
evidence.

## Next implementation boundary

The learner can now consume these actual e9 inputs faithfully. The
[ordered-transform follow-up](modular-ordered-transforms.md) qualifies
prescribed RCT anywhere in the sequence and palettes through 1,024 entries,
and the [prepared-stream encoder](modular-prepared-entropy.md) qualifies
per-stream WP headers and global/local tree and entropy scopes. Token and
decoded-pixel equality now cover these prescribed decisions.

The [fixed-gradient estimator](modular-selection-cost.md) now matches pinned
cost values and RCT rankings on generated/captured planes, and the
[WP estimator](modular-weighted-cost.md) now matches all five reference modes.
Next port reference palette construction, then integrate selection so expensive
learning happens on selected coding scopes. Keep the exhaustive
new-learner and all-predictor controls, then apply whole-stage timing, rate,
resource and held-out gates before default promotion or GPU work. The
[acceptance criteria](modular-tree-learning-cost.md#proposed-acceptance-criteria)
remain unchanged.
