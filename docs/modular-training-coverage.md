# CPU Modular training coverage and capacity

Step #1 is committed as `3372d65`. This step implements and qualifies the native
coverage/capacity experiment, plus a separate reference-inspired learner. Public
API and CLI defaults remain unchanged. Both experimental modes retain the entire
default search and append new candidates; complete bytes decide the winner.

**The existing learner at 65,536 samples / 64 leaves reduces the PNG aggregate
gap versus pinned libjxl e9 from +9.0529% to +4.0772%.** That saves 381,836 bytes
and recovers 54.9630% of the previous 694,714-byte excess. The separate quantized
learner reaches +4.9259%, saving 316,702 bytes (45.5874% of the excess).

## Measured complete files

All 176 files were independently decoded sample-exactly. All 16 default-search
controls are byte-identical to saved `f4300f4` outputs. Reference totals come
from the retained, hashed e9 files; reference encodes were not rerun.

| Normal search + ANS | PNG bytes (11 inputs) | Gap vs e9 | All 16 bytes | Gap vs e9 |
|---|---:|---:|---:|---:|
| Baseline, 4K/16 | 8,368,624 | +9.0529% | 9,205,898 | +8.1615% |
| Existing learner, 64K/64 | 7,986,788 | +4.0772% | 8,814,109 | +3.5583% |
| Quantized learner, 64K/64 | 8,051,922 | +4.9259% | 8,888,960 | +4.4378% |

Pinned libjxl e9 totals: 7,673,910 bytes for PNG inputs; 8,511,250 bytes for all
16. No image regressed under either experimental normal-search setting.

## Fixed-transform factorial

For each image, freeze the baseline winner's RCT, palette/Squeeze sequence and
Weighted parameters before learning. All four cells below are actual native
encodes. Raw bytes expose learner regressions. Retained bytes select the smaller
of that measured file and the measured baseline, reproducing the fallback rule.
Baseline winners can be fixed or bit-width trees, so the raw 4K/16 learner cell
need not equal the baseline. This is not a pure leaf-growth study across learners:
quantized mode also changes sampling, predictors, bins and split costs.

| Learner | Samples/leaves | PNG raw bytes | With baseline fallback | Gap with fallback |
|---|---|---:|---:|---:|
| legacy | 4,096/16 | 8,378,846 | 8,368,624 | +9.0529% |
| legacy | 65,536/16 | 8,220,789 | 8,216,633 | +7.0723% |
| legacy | 4,096/64 | 8,366,999 | 8,358,938 | +8.9267% |
| legacy | 65,536/64 | 8,097,497 | 8,047,253 | +4.8651% |
| quantized | 4,096/16 | 8,542,663 | 8,321,247 | +8.4356% |
| quantized | 65,536/16 | 8,396,587 | 8,236,293 | +7.3285% |
| quantized | 4,096/64 | 8,416,129 | 8,249,715 | +7.5034% |
| quantized | 65,536/64 | 8,240,260 | 8,075,766 | +5.2367% |

For the existing learner, increasing both budgets saves 281,349 raw bytes,
with a positive factorial interaction of 111,445 bytes. With baseline fallback,
the combined saving is 321,371 bytes and the interaction is 159,694 bytes.
Normal search recovers another 60,465 bytes beyond fixed transforms with fallback.
This confirms that coverage and capacity should increase together.

The earlier libjxl fixed-RCT study reached +2.9551% at 64K/64. These native
cells do not reproduce its sampler, trees or entropy models, and the difference
must not be attributed to one mechanism without further controls.

## Implementation and interpretation

- Private `TreeLearningOptions` selects the retained learner or `kQuantized`.
  The latter uses Weighted/Gradient, HybridUint 4/1/2, global 256-bin property
  quantization, sampling-aware penalties, predictor-change costs and static/
  no-Weighted preferences. Both allocate capacity best-first and stop when
  further splits do not pass their cost test.
- Quantized sampling maximizes a common floor up to 1024 samples per transformed
  stream/channel slice, reduces that floor to fit the budget, then apportions
  the remainder by area with deterministic stratified positions. Reports retain
  population, actual samples, covered slices, floor, quota range and quota hash.
- Storage plans cover both the retained and experimental learners. Invalid
  options fail before allocation, output publication remains atomic, and all
  new working storage uses managed allocations.
- This is a native reference-inspired control, not libjxl parity. Actual-property
  quantiles replace reference proxy/coordinate/Weighted bins; integer costs,
  deterministic sampling, exact transformed-population denominator and a hard
  best-first leaf cap also differ. See the [study README](../tools/modular_training_study/README.md).

The measured result favors the existing learner for rate on this corpus.
For chessboard, the frozen quantized 64K/64 tree costs 286,479 bytes versus
107,359 with the existing learner; fallback is essential. This bundled comparison
does not isolate whether predictors, quantization, sampling or scoring cause
the difference. The quantized learner remains useful as a control.

Capacity is not monotonic in final rate: on splines, the existing frozen 64K/16
candidate costs 343,695 bytes, while 64K/64 costs 446,914. Normal 64K/64 search
finds 355,821 through other candidates. Retaining intermediate tree checkpoints
is therefore a concrete follow-up, alongside studying the 32-cluster entropy
model with richer trees. The current experiment does not establish the cause
of that regression. Neither public-default promotion nor libjxl-level parity
is claimed here.

## Per-image normal search

| Input | libjxl e9 | Baseline | Existing 64K/64 | Quantized 64K/64 |
|---|---:|---:|---:|---:|
| alpha-flower | 3,715,769 | 3,970,554 | 3,830,474 | 3,826,210 |
| alpha-ria | 228,013 | 251,925 | 238,976 | 240,097 |
| alpha16-ramp | 8,375 | 26,121 | 19,838 | 26,121 |
| chessboard | 136,329 | 141,670 | 103,400 | 141,670 |
| gray-macan | 90,153 | 94,001 | 90,842 | 91,779 |
| gray-patches | 4,912 | 25,917 | 21,588 | 25,917 |
| gray16-ramp | 2,502 | 1,431 | 1,293 | 1,431 |
| noise16 | 825,073 | 808,588 | 805,318 | 808,588 |
| palette-358 | 1,243 | 1,061 | 1,045 | 1,061 |
| palette-pattern | 1,047 | 670 | 463 | 579 |
| photo-bliznaca | 240,347 | 250,237 | 244,280 | 245,882 |
| photo-flower | 2,537,909 | 2,748,396 | 2,624,437 | 2,605,816 |
| photo-macan | 252,153 | 268,426 | 263,843 | 264,482 |
| photo-ria | 208,131 | 219,767 | 212,082 | 212,338 |
| screen-pattern | 343 | 464 | 409 | 319 |
| splines | 258,951 | 396,670 | 355,821 | 396,670 |

At 64K, quantized sampling covers all 162 photo-flower slices (404–405
samples each), 216 alpha-flower slices (303–304), and 192 splines slices
(341–342). Small metadata and one-pixel edge slices can be fully sampled;
a quota below 1024 for such slices is not missing coverage.

## Qualification and retained evidence

- 19 selected Modular CTests passed, including frozen native/reference oracles,
  conformance, transforms, policy, API/CLI, GPU handoff contracts and both learner modes.
- 30,374 exhaustive complete-workflow allocation-failure injections passed
  across prescribed/default-learner/quantized-learner prefix and ANS cases.
  The quantized learner also passed all seven direct backing-failure points.
- Legacy workflow shard 0/64 passed: 44 cases and 2,702 injected failures.
  This is not a claim that all legacy workflow failure positions were revisited.
- ASan/UBSan passed foundation, quantized coverage/encode and GPU workflow
  contract tests. Pinned reference archives were not sanitizer-instrumented.
- A final explicit floor-coverage assertion passed in release and sanitizer
  builds. Formatting-only rebuild produced a byte-identical study executable.

Artifacts are retained under `build/training-coverage/` and
`build/training-coverage-sanitize/`. The [structured results](modular-training-coverage-results.json)
include per-cell/per-image numbers, source hashes, output hashes and log hashes.
The [runner and analysis tools](../tools/modular_training_study/README.md) reproduce
the paired controls and aggregation. Step #1 artifacts remain untouched.

This is a small corpus with related RGB/gray/alpha images. Single-run timings
were gathered with concurrent workers and are diagnostic only: summed caller
encode times were 105.26 s baseline, 267.32 s existing 64K/64, and 166.79 s
quantized 64K/64. These are not throughput qualifications. No GPU implementation
or GPU capacity change is included.
