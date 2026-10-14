# Downstream Modular entropy ablations

**Context grouping accounts for the remaining aggregate rate gap on the saved
automatic e9 decisions.** Replacing GJXL's data-context partition with pinned
libjxl's partition, capped at the same 32-histogram ceiling, reduces the native
total from **7,106,401 to 6,900,966 bytes**. Stock e9 is 6,901,111 bytes on this
five-image cohort: the conditional difference falls from **+2.9747% to
−0.0021%**. GJXL still chooses its own integer configurations, normalizes its
ANS histograms, writes its tree codes, and emits the sample tokens.

This is a research transplant of reference grouping, **not a native clustering
implementation or an autonomous encoder result**. Pixels, transforms, WP
parameters, trees and coding scopes are captured e9 decisions. The separate
16-bit ramp retains a large gap. No timing or default-promotion claim follows.
The [machine-readable ledger](modular-entropy-ablation-results.json) contains
every measured control, bit attribution, input binding and qualification result.

The subsequent [native direct-ANS comparison](modular-direct-ans.md) finds that
GJXL already has the required native clustering machinery in another entry
point. Its rate/64 control reaches −0.3282% automatic and +0.5720% buffered
versus stock on these captured decisions. That narrows the next implementation
to width-selection and policy integration, rather than another clustering port.

## Controlled boundary

The [prepared-stream replay](modular-prepared-streams.md) previously established
identical native/reference logical tree and sample tokens. The new research
command `stream-entropy-check` holds those tokens and native frame/stream syntax
fixed while substituting reference or native tree/data entropy coding. It
imports the captured histogram options rather than reconstructing effort
defaults. Shared reference models retain the actual stream IDs, including empty
slots; local models use their own stream and image width. Reference LZ77 gets
private mutable copies of the original tokens.

Each output is a complete file decoded by the independent pinned decoder and
compared with every original source integer. The `native` control must also
match both the native prepared-stream encoder and its previously frozen file
byte for byte. Per-scope model/payload bits, tree bits, stream headers, section
padding, image/frame header and TOC bits account for the entire file.

The reference-entropy baseline uses this same native assembly. It differs from
the original stock e9 file by −5 through +2 bytes per input/configuration;
native header choices and omission of ignored empty local payloads remain.
Comparisons between entropy controls use the common-assembly baseline. Tables
explicitly labeled stock use the original e9 file instead.

## Direct ablations

All totals below cover the same five policy-study images. The true 16-bit ramp
is reported separately. These are measured cells; no missing factorial cell is
constructed by subtracting unrelated deltas.

| Entropy control | Automatic bytes | Buffered bytes |
|---|---:|---:|
| Stock e9, original file | 6,901,111 | 6,767,272 |
| Reference tree and data, common assembly | 6,901,091 | 6,767,264 |
| Native tree; reference data | 6,899,156 | 6,767,533 |
| Reference tree; native data | 7,108,333 | 7,207,357 |
| Native tree and data | 7,106,401 | 7,207,625 |
| Reference, LZ77 disabled | 6,901,091 | 6,767,264 |
| Reference, data capped at 32 histograms | 6,918,859 | 6,887,809 |
| Reference, data capped at 64 histograms | 6,901,920 | 6,806,642 |
| Reference, fixed 4/2/0 data integer coding | 6,901,638 | 6,780,014 |
| Reference, approximate ANS histogram search | 6,901,870 | 6,767,368 |
| Native coding with reference data groups, cap 32 | **6,900,966** | **6,888,089** |

LZ77 has **no size effect on any of these ten cases**. The separate RLE-only
and data-only no-LZ77 controls also leave their sizes unchanged. The native tree
coder has little aggregate cost, and actually saves bytes on the automatic
cohort. The large loss is in sample-data entropy coding.

The cap controls show that histogram capacity matters more to the large shared
buffered trees than to the local automatic trees. They do not test simply
raising GJXL's cap: the partitions and downstream optimizer are still libjxl's.
Likewise, the small effects of fixed integer coding and approximate histogram
search apply to these reference-side controls, not arbitrary native settings.

The `nonstreaming` control changes the captured entropy parameter only. It
saves 21,639 bytes in aggregate automatic mode and changes no buffered size.
This parameter affects multiple downstream choices, including integer coding;
it is not an isolated estimator or tree-scope ablation. Local/shared trees and
transformed pixels stay fixed in this experiment.

## Partition transplant

The research adapter builds the same per-context 4/2/0 populations, invokes
pinned `ClusterHistograms` with a 32-histogram limit and no LZ77, then passes the
resulting partition to the unchanged native ANS optimizer. It constructs a
valid Prefix carrier solely because that existing interface requires one.
The native optimizer reads the carrier's partition and cluster count; it does
not use its prefix lengths or integer configurations to seed the ANS search.

Every transplanted map matches the map independently emitted by the **full
reference no-LZ77/cap-32 encoder**, not merely a copy of the standalone
clusterer's output. All 195 data scopes match in both tree-coder variants
(390 comparisons). The reference-tree transplant also preserves reference
tree coding, and swapping tree coders preserves every data-scope result.

| Automatic input | Stock e9 | Native coding | Reference groups + native coding | Difference from stock |
|---|---:|---:|---:|---:|
| photo-macan | 252,153 | 255,913 | 252,961 | +0.3204% |
| photo-flower | 2,537,909 | 2,583,975 | 2,537,546 | −0.0143% |
| alpha-flower | 3,715,769 | 3,771,270 | 3,715,960 | +0.0051% |
| chessboard | 136,329 | 143,830 | 136,909 | +0.4254% |
| splines | 258,951 | 351,413 | 257,590 | −0.5256% |
| **Total** | **6,901,111** | **7,106,401** | **6,900,966** | **−0.0021%** |

Splines is the clearest control. Its reference entropy file is 258,947 bytes;
disabling LZ77 changes nothing, and capping reference clustering at 32 costs
only 158 bytes. Native data coding with the reference tree costs 351,789 bytes.
Supplying the reference partition to the native data coder reduces that to
257,967 bytes. Native tree coding then reduces it to 257,590. The grouping
change recovers the large loss without another fit or a different predictor.

The source explains why this is a plausible architectural mismatch:
[`entropy.cpp`](../src/codestream/entropy.cpp) chooses and refines partitions
using Huffman costs and a portfolio of Prefix candidates. The Modular ANS path
then retains that selected partition. Pinned
[`enc_cluster.cc`](https://github.com/libjxl/libjxl/blob/e8ff09762481785938d8e4e01333ed3917571161/lib/jxl/enc_cluster.cc)
uses Shannon-distance seeding/assignment, followed at e9 by beneficial merges
scored with ANS population cost. ANS configuration search cannot undo a poor
fixed grouping. The experiment isolates the grouping pipeline as a whole;
it does not attribute the gain to one seed, merge, tie or refinement rule.

## Remaining exceptions

The buffered transplant is **+1.7853%** versus stock, much better than the native
control's +6.5071% but outside the eventual aggregate gate. Reference coding
itself loses +1.7812% with a 32-histogram cap. Higher capacity remains relevant
to this shared-tree workflow; the 64-cap reference control still costs 39,378
bytes versus unrestricted reference entropy.

The ramp is a distinct problem and must remain visible:

| `alpha16-ramp`, automatic | Bytes |
|---|---:|
| Stock e9 | 8,375 |
| Reference entropy, common assembly | 8,371 |
| Reference with LZ77 disabled | 47,576 |
| Reference without LZ77, cap 32 | 59,513 |
| Same reference partition, native data and reference tree | 75,408 |
| Same reference partition, all-native coding | 75,409 |
| Original native coding | 79,681 |

Disabling LZ77 costs 39,205 bytes on the reference path; RLE alone has the same
poor result. Capping the no-LZ77 path at 32 adds 11,937 bytes. Even on that exact
partition, native data coding costs another 15,895 bytes. These successive
controls expose more than one mechanism, not an LZ77-only explanation.
The logged integer configurations point to a further targeted control: the
native Prefix-to-ANS path uses eight configurations, whereas the reference
selects high-depth configurations absent from that portfolio. GJXL already has
a wider portfolio in its separate direct-ANS path. That is a source-grounded
lead, not an isolated measurement of the portfolio's contribution.

## Qualification and next implementation

The first Release study emits 143 files (11 cases × 13 controls); the partition
study emits 33 (11 × 3), including 11 common controls that reproduce the first
study byte for byte. Together these provide **165 distinct measured cells**.
All files reconstruct exactly. ASan/UBSan emits **60 further files** across
Macan, local-palette chessboard, the 16-bit ramp and the buffered alpha tree;
every file matches its Release counterpart. All ten research CTest entries
pass in both current Release and sanitizer builds, including malformed-control,
false-equality and changed-map rejection tests.

The executable, native/tool source snapshots, generated reference hooks,
compiler commands, archives, captures, output files and final test logs are
hash-bound. The final validation manifest covers 2,643 artifacts. Native
libraries and the research/generated units are sanitizer-instrumented; linked
pinned reference archives are not. Vptr checks are excluded for the no-RTTI
reference, and this macOS runtime requires LeakSanitizer to be disabled.
Builds/checks may overlap; elapsed times are not benchmark evidence. The mixed
reference/native adapter has no production managed-storage qualification.

The next implementation sequence is:

1. The [existing direct-ANS comparison](modular-direct-ans.md) is now complete:
   reuse that native clustering machinery. Keep these reference-partition
   controls as the quality oracle.
2. Qualify estimated alphabet-width selection and connect the native direct
   path through an explicit private policy, using prepared context populations
   and appropriate managed bounds. Avoid a Prefix-code construction detour.
   Merely increasing the old path's cap does not address its grouping loss.
3. Isolate the high-depth integer portfolio and LZ77 requirements on the ramp
   and additional held-out structured/16-bit inputs. Do not hide these cases
   behind the photographic aggregate.
4. Complete native palette construction and cheap RCT/WP selection, then
   connect native preparation/fitting **once per selected coding scope**.
   Apply the existing complete-stage timing, exhaustive-search, held-out rate
   and resource gates before changing defaults or choosing GPU work.

Reproduce captures with `run_entropy_replay.py --help`, using the qualified
`coding-replay` study as input. It refuses to replace an existing study and
requires `no_lz77_clusters32` alongside either partition transplant. Analyze a
single study with `analyze_entropy_replay.py --study ... --output ...`; add
`--partition-study`, `--sanitizer-study` and `--validation` together to verify
the combined qualification. No reference learning or transform search is
rerun by these entropy controls.
