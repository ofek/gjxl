# Predeclared high-depth WP validation

This is the next control after the original-corpus
[WP ranking failure](modular-reference-wp-sweep.md), committed before encoding
the new inputs. The goal is to test whether that opportunity transfers to
other high-depth content, while keeping the efficient reference-style learner
and the runtime policy unchanged.

The [corpus specification](../tools/modular_reference_learning/wp_holdout_spec.json)
fixes 32 inputs: nine photographic RGB16 sources, three explicit derivatives
of the larger HDR source, and ten synthetic families at each of two extents.
Synthetic extents have eight and nine 256-pixel groups, exercising shared and
local trees. Families include affine gradients, wrapped gradients, curvature,
edges, palettes and noise. Gray, RGB and RGBA are represented; RGBA preserves
nonzero color beneath zero alpha. The generator records every sample hash.

Natural PNG data is read without resampling or color conversion and checked
sample-for-sample with two installed decoders before writing big-endian PAM.
Nonzero low bits are required: these are not merely expanded 8-bit values.
Input/source hashes must not match the original Modular corpus. Ancillary
color metadata is omitted; all encoder comparisons use the same declared
profile and must reconstruct the exact source integers.

Coverage is deliberately explicit. Eight natural sources are only 64×64;
`hdr_room` is 676×449. Its crop, channel projection and generated-alpha variant
share one underlying scene. Synthetic pairs also share a formula, and their
different shapes are not a pure tree-scope ablation. This study cannot close
large-natural-image or general 8-bit held-out qualification.

For each input, run the already-qualified automatic Best workflow and the
same selected-vector replay plus five uniform WP assignments. Retain every
candidate, complete bytes and exact decode. Compare stock pinned e9 and the
frozen qualified legacy native search separately. Do not discard difficult
files or reference features, including patches/LZ77, from rate totals.

One prospective rate-only control is fixed before results are known:

- On 16-bit inputs with a shared tree (at most eight groups), compare the
  selected result with uniform WP 1, retaining the smaller complete file.
- On local-tree inputs, retain the original selected result.
- Selected wins ties. This allows at most one additional fit per eligible
  image. No legacy fallback or all-predictor search is part of the control.

This is a testable proposed budget, not a runtime implementation or a speed
claim. Single-pass timings remain diagnostic. A useful rate result would
still need an implemented bounded path, complete paired CPU/wall timing and
resource qualification before promotion. The five-preset minimum is an
offline lower-bound diagnostic for search loss, not the prospective policy.

Report full-corpus and natural/synthetic strata, every per-image loss and
absolute delta, selected versus tested-minimum loss, the prospective control's
remaining loss and fit budget, and ratios to stock e9 and legacy native.
Retain all failures as evidence; policy changes motivated by this set require
a further independent qualification set.

The pre-encode corpus is frozen at `build/reference-wp-heldout-corpus-v2`:
manifest SHA-256
`7155f2dc6e4a6812f7b09eb569260145b7085d9c967d4e639c43ccc6ddf28366`.
It contains 18 RGB, five Gray and nine RGBA inputs (all 16-bit), totaling
72,295,086 packed PAM bytes. Nine images have one group, three have six,
ten have eight and ten have nine. A second preparation reproduced all 32
files byte-for-byte; all frozen source/input/tool hashes were checked.
The earlier preparation is retained as a reproducibility control.

Reproduce with installed NumPy, OpenCV and ImageMagick in a new directory:

```sh
python3 tools/modular_reference_learning/prepare_wp_holdout.py \
  --reference-testdata /path/to/pinned-libjxl/testdata \
  --original-corpus build/reference-learning/corpus \
  --out build/reference-wp-heldout-corpus-new
```
