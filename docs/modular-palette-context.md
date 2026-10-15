# CPU Modular context learning after transforms

Opt-in CPU search now learns an entropy-based tree after each existing exact
palette or Squeeze candidate. This follows [bounded RCT selection](modular-rct-search.md).
The work lives on `feat/modular-palette-context`, based on `fa4d9f3`, in a separate
worktree from the concurrent GPU implementation.

## Policy and bounds

The retained search runs first, unchanged in order. Each eligible existing
transform candidate then receives one learned tree: the all-channel exact
palette, horizontal Squeeze, vertical Squeeze, and horizontal-plus-vertical
Squeeze. Identical learned policies are skipped. Only a strictly smaller
complete codestream replaces the winner; earlier candidates win ties.
There are at most four additional complete encodes, giving ceilings of 27
color candidates and 13 grayscale candidates.

Training uses the completed transformed image and its canonical stream layout,
including palette dictionary channels, indices, Squeeze residuals, shifts and
clipped groups. The learner retains the transform sequence, RCT, and weighted
predictor parameters. Its existing 4,096-sample / 16-leaf bounds and entropy
objective are unchanged. Search storage admission now covers the transformed
token count, both sample sets, weighted state, the largest context model and
the retained winning file. Training scratch is released before encoding.

This milestone does not expand palette capacity, add per-channel or RGB-only
palette selection, change public options, or add a bitstream feature. The
existing three Squeeze choices remain bounded; there is no new Squeeze search.
Search performs additional full-image training and candidate encoding, so the
rate results do not establish a speed advantage.

## Channel-property correctness

New decoder checks exposed an existing CPU-oracle error. Tree property 0 is the
channel ordinal **within the encoded stream**, whereas `ChannelSlice::channel`
is the full-image storage index. Group streams compact away global channels,
palette metadata, channels assigned to other group types, and empty edge slices.
Using the storage index can produce invalid files when a tree distinguishes
channels after transforms.

Tokenization, training and capture/reference adapters now use the stream-local
ordinal for property 0 while retaining the original storage index for plane
access. Prescribed transformed trees are corrected as well as learned trees.
Regression tests cover a global palette dictionary followed by grouped indices,
global Squeeze lowpass channels, mixed DC/ordinary groups, and clipped edges.
The current GPU token operation accepts only unpack/RCT, for which storage and
stream-local indices coincide. Future transformed GPU support must distinguish
the two indices.

The frozen oracle is explicitly versioned to v2 without rewriting v1. Both
independently checked reference and native captures agree. Only
`palette-prefix/decisions.bin` and `palette-ans/decisions.bin` change: exactly
2,313 property-0 values in each trace change from 1 to 0. All other recorded
integers and all 284 other artifacts, including every complete file, remain
identical. `oracle-v2.json` records these two overrides and binds them to the
unchanged v1 manifest hash. The verifier never generates expected hashes.

## Compression results

The paired run uses the same 16 exact integer inputs and metadata exclusions as
the RCT qualification, with its frozen search outputs as baseline. Each of the
64 new outputs is independently decoded by pinned libjxl
`e8ff09762481785938d8e4e01333ed3917571161`. All 32 default outputs remain
byte-identical; no search output grows, and equal-size outputs retain their
previous bytes. These are lossless complete-codestream comparisons, not
same-effort or equal-time comparisons.

Search + ANS:

| Input/group | Before bytes | After bytes | Reduction |
|---|---:|---:|---:|
| Synthetic screen pattern | 3,241 | 464 | 85.68% |
| Synthetic palette pattern | 10,308 | 670 | 93.50% |
| Synthetic gray16 ramp | 1,676 | 1,431 | 14.62% |
| Synthetic alpha16 ramp | 28,470 | 26,121 | 8.25% |
| Four graphics PNG fixtures | 565,436 | 565,318 | 0.0209% |
| All 11 PNG fixtures | 8,368,742 | 8,368,624 | 0.0014% |
| Five synthetic inputs | 852,476 | 837,274 | 1.78% |
| All 16 inputs | 9,221,218 | 9,205,898 | 0.1661% |

The gains are concentrated in synthetic inputs. Only `gray-patches` improves
among the PNG fixtures (118 bytes); all photographs, `splines`, `chessboard`
and `palette-358` remain unchanged. The PNG gap to libjxl e9 remains 9.05%,
and the four-graphics gap remains 40.82%. This completes a missing policy
combination and fixes oracle correctness, but does not materially close the
measured real-fixture graphics gap. Richer palettes and LZ77 remain separate
features whose benefits require their own measurements.

Prefix results are also retained in the qualification record. No timing claim
is made: diagnostic single-run timings overlapped validation work.

## Validation and evidence

The Release policy suite passes 835 stage cases and 1,738 complete files. The
expanded transform suite passes 453 cases and 906 complete files, including
pinned predictions/tree traversal/token checks for learned transformed policies,
independent sample-exact decoding, exact admission, serial/parallel equality,
input preservation and determinism. All six formats, unaligned/padded inputs,
one-dimensional and odd sizes, group edges, palette metadata, shifted channels,
combined transforms and custom weighted parameters are covered.

The CPU suites, public API/CLI, native GPU contract/handoff checks and both
frozen native/reference gates pass. Debug ASan/UBSan foundation, transform and
policy suites pass (641 seconds overall). Native code is instrumented; pinned
libjxl archives are not, and existing reference targets disable vptr for their
no-RTTI archives. All 44 exhaustive allocation-failure cases pass across four
disjoint shards: 171,229 failure points, including 152,882 search points,
in 1934 seconds. Their union is checked against the independent allocation
counters and every expected index; all non-search counts match the previous
qualification. See the [structured record](modular-palette-context-results.json).
The allocation guard is 75,000 because expanded ANS search reaches 52,213
allocations in the existing 257x3 RGBA16 fault fixture. Four disjoint shards
cover every allocation index; at this milestone the default test ran the full
sweep. See [Modular allocation-failure qualification](modular-allocation-testing.md)
for the subsequent regular/exhaustive split and current invocation commands.

`build/modular-palette-context/` retains frozen libraries, input/output hashes,
paired ledgers, the exact source patch, reference captures, the audited v1/v2
trace difference, sanitizer logs and allocation-failure evidence. Initial runs
before the channel-property correction are retained separately and do not count
as final qualification. The checked-in structured qualification record identifies
the final source and evidence hashes.
