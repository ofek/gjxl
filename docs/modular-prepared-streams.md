# Native coding of captured Modular stream scopes

The private CPU encoder now accepts **separately prepared streams**, each with
its own ordered transforms and weighted-predictor parameters, and either a
shared tree/model or a local tree/model. On the saved e9 controls, native tree
and sample tokens agree exactly with pinned libjxl across **391 populated
streams and 73,913,379 sample tokens**. All **22 complete Prefix/ANS files**
decode to the original integers, including invisible RGB and true 16-bit alpha.

This qualifies the coding boundary needed after the
[native preparation/fitter replay](modular-native-stream-preparation.md).
It also finds a remaining **downstream rate gap**: with the captured automatic
e9 decisions, native ANS files are **2.9747% larger** on the five-image policy
cohort. Matching learning alone does not meet the eventual rate gate.
The [qualification ledger](modular-prepared-streams-results.json) binds the
source, binaries, captures, outputs and checks.

## Contract

[`prepared_streams.h`](../src/codestream/modular/prepared_streams.h) is a private,
borrowed-input API. Source dimensions/format determine canonical global, DC
group and pixel-group slots, retaining empty slots and reserved stream-ID gaps.
Global transforms determine the full channel shape before grouping; local
transforms operate on each group's channel coordinates after grouping.
The global stream borrows only its coded prefix. Excluded full-image channels
need no backing in this interface.
The caller owns the semantic agreement of transformed samples with those
headers and the source image; this interface does not run an inverse transform
or compare against original pixels. The qualification harness does that final
comparison through the independent decoder.

A null per-stream tree selects the frame's global tree **and its shared entropy
model**. A local tree selects a tree and model serialized inside that stream.
Local tokens never enter global model optimization. Predictor history resets
for each channel, and prediction uses the actual coding stream ID and the
stream-local channel ordinal. Local learning's `[0,1)` range is a separate
coordinate convention and is not substituted here.

The encoder serializes each nonempty local tree once, retains one global model
where needed, and constructs one local model at a time. The empty global
prefix still carries its transform/WP header; it needs no local tree/model
payload because the decoder returns after the header when no channels are
coded. An unused global tree is also covered by the generated controls.
Tree/sample tokenization is shared with the existing whole-image path.

The allocation-free planner validates full/local shapes, view extents/strides,
WP fields, transforms, tree structure, and the distinct global/local decoder
tree limits. Fresh managed bounds cover token/tree arenas, stream views,
predictor scratch, model optimization, serial emission, section storage,
assembly and atomic publication. Failure preserves the old output and releases
all partial managed backing. Borrowed planes, trees, input/view ownership,
preexisting output, inline/stack storage and allocator overhead are excluded;
these bounds are not process RSS. Profiling keeps sample-model construction
separate from emission; tree-model coding remains inside emission.

This path does no fitting, transform selection or pixel copying. It is serial
and is not installed, selected by public defaults, or admitted by GPU routing.
Its representation retains the preceding
[ordered-transform limits](modular-ordered-transforms.md), including explicit
palettes through 1,024 entries and no palette deltas/implicit colors.

## Actual-capture replay

The research adapter consumes the previously hash-bound learning and coding
traces from pristine libjxl `e8ff09762481785938d8e4e01333ed3917571161`.
It does not re-encode the reference or redo its learning. It binds each coding
record to a captured image/tree by **stream ID**, checks options, shapes and
transforms, and loads the exact `.i32le` pixel slices. The palette dictionaries
and indices retain their captured ordering; the native lexicographic palette
constructor is not used to approximate them.

Source PNM dimensions determine geometry. In particular, a streaming capture's
empty global prefix can record a 2,048-pixel preparation width for a 2,268-pixel
source; treating that width as the frame width would misplace groups.
The bounded text projection rejects unsupported options/transforms, duplicate
or missing IDs, malformed trees, truncated/trailing data, invalid pixel
lengths/shifts, and mismatches between declared WP fields and reference presets.

For each populated stream, pinned `TokenizeTree` and `ModularCompress`
independently reproduce tree serialization and every sample's context/value.
The native encoder then emits Prefix and ANS files under the computed managed
bound, and the independent pinned decoder compares every source integer.
There are **401 captured coding records**, **188 populated local tree scopes**
and **seven global tree scopes**, across five inputs in automatic/buffered modes
plus the separate automatic 16-bit ramp control. Empty headers account for the
difference between 401 coding records and 391 populated streams.

## Conditional rate result

These are native complete-file sizes **given captured reference decisions**,
not results of an autonomous native search policy or a timing benchmark.
The five-image denominator is the original policy cohort, not the earlier
16-input bridge corpus. Differences are after the matched logical trees and
sample tokens: tree/model serialization, entropy decisions/emission and other
file syntax. This experiment does not isolate one of those mechanisms.

| Input | Stock automatic e9 bytes | Native ANS on its decisions | Difference |
|---|---:|---:|---:|
| photo-macan | 252,153 | 255,913 | +1.4912% |
| photo-flower | 2,537,909 | 2,583,975 | +1.8151% |
| alpha-flower | 3,715,769 | 3,771,270 | +1.4937% |
| chessboard | 136,329 | 143,830 | +5.5021% |
| splines | 258,951 | 351,413 | +35.7064% |
| **Five-image total** | **6,901,111** | **7,106,401** | **+2.9747%** |

For the same five inputs' forced-buffered decisions, the conditional total is
7,207,625 native bytes versus 6,767,272 reference bytes (**+6.5071%**).
The separate `alpha16-ramp` control is 79,681 versus 8,375 bytes (**+851.4149%**).
Its exact token/decode checks pass; the poor size must remain visible instead
of being hidden by the photo-weighted aggregate. Detailed Prefix sizes and
per-input buffered results are in the ledger.

The native path currently writes no LZ77 and retains its existing entropy
optimization choices. The subsequent [same-token entropy ablations](modular-entropy-ablation.md)
now isolate the large automatic-cohort loss to data-context grouping: a
reference partition with unchanged native coding reaches −0.0021% versus
stock on that cohort. LZ77 changes no bytes on those five images, but is
significant on the separate ramp. This is a research transplant, not a native
policy implementation, and the ramp still needs further downstream work.

## Qualification

| Check | Result |
|---|---:|
| Generated cases: six formats, six layouts, global/local/mixed scope | 108 |
| Generated exact Prefix/ANS files, Release and ASan/UBSan | 216 per build |
| Release/sanitizer generated-file identity | 216 / 216 |
| Full actual-capture replay cases / exact files | 11 / 22 |
| Selected actual-capture sanitizer cases / exact files | 4 / 8 |
| Selected Release/sanitizer capture-file identity | 8 / 8 |
| Native allocation failures injected across shared/local and Prefix/ANS | 1,470 per build |
| Release native/reference CTest entries | 8 / 8 |
| Sanitizer native/reference CTest entries | 2 / 2 |
| Research CTest entries, Release and sanitizer | 9 / 9 per build |
| Existing transform cases / exact files | 453 / 906 |
| Existing ordered transform cases / exact files | 175 / 350 |
| Frozen native and reference artifacts | 286 / 286 each |

Generated layouts include clipped pixel groups, shifted DC groups, palette
metadata, global palette followed by local RCT, local palette-before-RCT,
single-group section sharing, all five WP presets, and all six 8/16-bit
Gray/RGB/RGBA formats. Shared-tree baseline files match the older encoder.
Actual sanitizer controls cover the shared photographic tree, local palette
trees, true 16-bit RGBA, and the buffered RGBA tree with **8,289 nodes**.
Negative replay controls also reject false equality/timing/resource claims.

The largest measured managed peak in the full replay is **196,767,629 bytes**
(buffered RGBA). The caller owns the captured planes/trees and all validation
storage outside that number. Planner bounds are conservative; no production
memory, streaming-memory or performance qualification is implied.

Sanitizers instrument native libraries and the research harness/generated
reference units. Linked pinned archives remain uninstrumented; reference vptr
checks are disabled for the no-RTTI archives. Runs use
`ASAN_OPTIONS=detect_leaks=0` and `UBSAN_OPTIONS=halt_on_error=1` because this
macOS runtime does not support LeakSanitizer. Managed leak assertions pass.
Builds and correctness checks overlap; elapsed times are not benchmark data.

## Next integration gates

1. Integrate the [qualified existing direct-ANS machinery](modular-direct-ans.md)
   through an explicit policy after checking estimated width selection and
   managed bounds; retain the high-depth/LZ77 exceptions and exact controls.
2. Port reference palette construction and the cheap transform/WP estimators,
   matching values, tie order, selected transforms and prepared pixels.
3. Connect native preparation/fitting once per selected coding scope, preserving
   the shared/local entropy choice. Count and prohibit repeated full fits for
   transform candidates in the proposed policy.
4. Run paired whole-stage timing, full/held-out rate, resource and exhaustive
   new-learner comparisons under the
   [existing promotion gates](modular-tree-learning-cost.md#proposed-acceptance-criteria).
   Default promotion and GPU implementation remain separate milestones.

For reproduction, build the `gjxl_modular_prepared_stream_test` and
`gjxl_modular_prepared_stream_reference_test` targets in a fresh native build
with both GPU backends disabled and `GJXL_MODULAR_ORACLE_BUILD` set to the pinned
build. Run `ctest -R '^modular_prepared_stream(_reference)?$'` in that build.
The research executable must link those fresh native archives. Use
`run_coding_replay.py --help` and `analyze_coding_replay.py --help` for capture
and evidence inputs; each new study refuses to replace existing artifacts.
