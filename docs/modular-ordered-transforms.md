# Ordered Modular transform contract

The private CPU transform sequence now supports **RCT at any ordinary-channel
offset**, interleaved with palette and squeeze, and **exact palettes up to
1,024 entries**. This removes two representation limits exposed by the
[captured e9 stream policies](modular-reference-policy-ablation.md): palettes
larger than 256 entries and palette-before-RCT chains with shifted channel
coordinates. The existing automatic search remains capped at 256 palette
entries and retains its original candidate portfolio.

This is a prescribed-transform correctness milestone in the learning-cost
plan. It does not implement local coding policies, reference palette ordering,
new selection heuristics, or default promotion. There is no new rate or timing
claim. See the [qualification ledger](modular-ordered-transforms-results.json).

## Contract and storage

`TransformKind::kRct` requires three adjacent, equally shaped ordinary
channels and a type in `[0,42)`. Type zero is an explicit identity descriptor.
The channel range is interpreted after all preceding transforms; metadata
channels are excluded. The existing `ModularCodingPolicy::rct` still executes
first, preserving the earlier private contract and wire encoding. A sequence
retains its eight-entry and 64-channel limits.

Planning is allocation-free. RCT preserves descriptors, uses wide arithmetic,
and rejects results outside signed int32 before publishing any replacement.
Forward and reverse composition retain the old output on allocation, shape or
arithmetic failure, including when input and output alias. Existing workflow
storage planning includes each stage's image backing and replacement overlap.

The palette cap of 1,024 follows pinned libjxl's default `palette_colors`, rather
than its larger format limit. Explicit palettes retain the existing signed
lexicographic tuple order, zero-filled unused entries, and rejection when the
requested capacity is exceeded. Palette scratch is a fixed 16 KiB stack array;
only its initialized prefix is read, without clearing unused scratch capacity.
The metadata dictionary's managed backing remains part of the planned output
shape. These managed bounds exclude stack/inline objects and allocator overhead.

The stream writer encodes ordered RCT ranges/types and the complete palette
size field through 1,024. Its existing conservative header bound remains valid.
The legacy leading-RCT spelling is preserved, and explicit GPU admission still
rejects transform sequences. No GPU kernel or routing capability was expanded.

## Qualification

Release and ASan/UBSan use fresh native builds with the pinned reference at
`e8ff09762481785938d8e4e01333ed3917571161`. They do not link the older native
archives used by earlier learner studies: the private descriptor layout changed.

| Check | Result |
|---|---:|
| New prescribed cases, with per-stage sample/shape comparison and native inversion | 175 |
| New complete files, independently decoded exactly by libjxl | 350 |
| Native stream headers parsed by libjxl at all eight bit alignments | 4,520 |
| Release versus sanitizer file identity | 350 / 350 |
| Existing transform cases / independently decoded files | 453 / 906 |
| Frozen native stage/file regressions | 286 / 286 |
| Frozen reference stage/file regressions | 286 / 286 |
| Selected Release CTest entries | 11 / 11 |
| Selected ASan/UBSan CTest entries | 3 / 3 |

New cases cover every RCT type, nonzero ranges, palette-before-RCT, multiple
metadata channels, RCT after squeeze, eight ordered RCTs plus a legacy leading
RCT, and RCT-before-palette with learned contexts. Palette sizes include the
255/256/257 boundary, observed 285/338/417/662/862 sizes and the 1,024 cap.
The tests compare native tree/sample tokens with pinned libjxl prediction and
tree traversal, verify complete Prefix/ANS files and invisible RGBA samples,
and require serial/parallel bytes to agree on grouped cases.

The native-only contract test injects every allocation in both transform
directions, checks managed bounds and released partial stages, and rejects bad
RCT types/counts/ranges, mismatched dimensions/shifts, overflow and unsupported
palette sizes. Header failures preserve the prior bitstream and respect nested
allotments. Separate checks keep the search palette limit at 256. Existing
GPU contract/handoff tests remain passing with both GPU backends disabled.

ASan/UBSan instruments the native libraries, test harness and reference adapter;
the linked pinned reference archives remain uninstrumented. The run uses
`UBSAN_OPTIONS=halt_on_error=1` and `ASAN_OPTIONS=detect_leaks=0`; LeakSanitizer is
unsupported by this macOS runtime. An initial run requesting it aborted before
tests executed and is retained separately, excluded from passing qualification.
Managed allocation-leak assertions are exercised by the native contract test.
Correctness runs overlap independent builds/tests, so their elapsed times are
not performance evidence.

## Remaining policy boundary

Do not equate an ordered transform sequence with reference palette construction.
Pinned `enc_palette.cc` orders multi-channel palettes using frequency/luma
scores (including alpha) and supports implicit colors; the current native
constructor remains lexicographic and explicit-only. The capture audit found
no implicit indices in its 58 single-palette RGB streams, but their palette
ordering still needs reference-equivalent preparation. Prescribed palette
tests use an independent exact dictionary and the pinned inverse/decoder;
they do not claim equality to libjxl's heuristic forward palette selection.

The subsequent [prepared-stream coding contract](modular-prepared-streams.md)
now keeps each stream's WP and transform sequence, its global/local tree choice,
and the corresponding entropy-model scope together. Captured e9 tree/sample
tokens and decoded pixels match, but a conditional downstream rate gap remains.
Isolate that gap and implement reference palette construction and the cheap
transform/WP estimators before fitting selected scopes. The [existing whole-stage rate,
time, resource and held-out gates](modular-tree-learning-cost.md#proposed-acceptance-criteria)
still govern promotion.

To reproduce the focused checks, configure a fresh Release build with
`GJXL_ENABLE_METAL=OFF`, `GJXL_ENABLE_CUDA=OFF`, and
`GJXL_MODULAR_ORACLE_BUILD` pointing at the clean pinned reference build.
Build `gjxl_modular_transform_test` and `gjxl_modular_transform_contract_test`,
then run:

```sh
ctest --test-dir BUILD --output-on-failure -R '^modular_(ordered_transform|transform_contract)$'
```

The ledger binds the
frozen source patch, executables, build settings, logs and output-file hashes.
