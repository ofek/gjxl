# Identical-token Modular entropy rate study

The prefix-derived context partition is a measurable rate weakness. On the same
ordered tokens and the same native downstream ANS policy, replacing only that
partition with the pinned libjxl e7 partition reduces aggregate complete-file size
by **1.9352% at 32 clusters** and **2.6858% at 64 clusters**. This isolates a rate
problem from predictor, tree, transform and token differences.

All **266 complete files** (19 images × 14 model cells) decode to the original
samples. Every native baseline reproduces its full-search winning file byte for
byte. The study is complete; no production policy or public default is changed.

## Fixed conditions and interpretation

Each image uses the exhaustive winner from the pipeline qualification. The
scalar producer freezes its ordered `(context, value)` tokens and stream counts
once. Every cell replays that immutable archive with the same predictor, tree,
transforms, framing and context identities. ANS is forced and LZ77 is disabled.
The reference cells call pinned libjxl's model builder with e7/e9 and streaming
policy parameters; they are not complete default libjxl encodes.

Native-on-reference-partition cells use the existing native maximum-compression
configuration, normalization and estimated-width policy with a supplied context
assignment. The 32-cluster cell changes only the partition relative to the native
baseline. The 64-cluster cell also admits a larger partition. Comparing the four
reference e7 cells isolates the capacity limit under the same reference policy.
The direct balanced/density cells are multifactor controls relative to the
baseline; their gains cannot all be attributed to clustering.

## Complete-file rate

Percentages compare summed actual file bytes across the 19 unique images. The
large photographs dominate that sum; the per-image table below retains the small
and 16-bit counterexamples.

| Model policy / maximum clusters | Aggregate bytes | Change from native baseline | Actual cluster range |
|---|---:|---:|---:|
| Native prefix-derived / 32 | 34,076,924 | +0.0000% | 5–32 |
| Native direct balanced / 32 | 33,492,801 | -1.7141% | 5–32 |
| Native direct density / 32 | 33,457,692 | -1.8172% | 5–32 |
| Native direct balanced / 64 | 33,260,601 | -2.3955% | 5–64 |
| Native direct density / 64 | 33,212,191 | -2.5376% | 5–64 |
| libjxl e7 streaming / 16 | 34,041,806 | -0.1031% | 5–16 |
| Native on e7 partition / 16 | 33,976,267 | -0.2954% | 5–16 |
| libjxl e7 streaming / 32 | 33,493,105 | -1.7132% | 5–32 |
| Native on e7 partition / 32 | 33,417,469 | -1.9352% | 5–32 |
| libjxl e7 streaming / 64 | 33,260,873 | -2.3947% | 5–64 |
| Native on e7 partition / 64 | 33,161,680 | -2.6858% | 5–64 |
| libjxl e7 streaming / 128 | 33,167,472 | -2.6688% | 5–128 |
| libjxl e9 streaming / 128 | 33,166,482 | -2.6717% | 5–128 |
| libjxl e9 nonstreaming / 128 | 33,043,322 | -3.0331% | 5–128 |

The native direct-balanced partitions match the corresponding reference e7
partitions on **all 19 images**, at both capacities 32 and 64. Thus the promising
partition is already reproducible by native code on this corpus; the production
follow-up does not need a libjxl runtime dependency. This is an observed corpus
identity, not a guarantee for arbitrary inputs.

## Payload, model headers and context maps

The table gives actual emitted bits. `Histogram/config` includes the remaining
entropy header fields; each file also has one no-LZ77 flag bit. Complete bytes
add the shared tree, image/frame and stream headers, TOC widths and padding.

| Policy | Payload bits | Context-map bits | Histogram/config bits | Total model bits |
|---|---:|---:|---:|---:|
| Native prefix-derived / 32 | 272,473,103 | 12,569 | 60,216 | 72,804 |
| Native on e7 partition / 32 | 267,192,866 | 13,434 | 63,994 | 77,447 |
| Native on e7 partition / 64 | 265,100,672 | 16,180 | 106,972 | 123,171 |
| libjxl e7 streaming / 128 | 265,086,580 | 19,089 | 164,515 | 183,623 |
| libjxl e9 nonstreaming / 128 | 264,101,946 | 18,115 | 156,890 | 175,024 |

At capacity 32, changing only the partition saves **5,280,237 payload bits**
while adding **4,643 model-header bits**. At capacity 64, the corresponding
figures are **7,372,431 payload bits saved** and **50,367 header bits added**.
The net gains come overwhelmingly from payload modeling. Context-map/header
compression is a secondary target on this corpus.

The reference e7 capacity sweep saves 548,701 bytes from 16 to 32 clusters
(1.6118% relative to capacity 16), another 232,232 bytes from 32 to 64 (0.6934%),
and another 93,401 bytes from 64 to 128 (0.2808%). Actual large-image context
counts are 256/128/128; the capacity-128 reference uses 128/127/128 clusters.
Native capacity remains limited to 64. Its representation and storage must not
be expanded to 128 by changing a single constant.

## Per-image bytes and regressions

| Image | Native baseline | Direct balanced 64 | Direct density 64 | Native on e7 partition 64 | libjxl e7 128 | libjxl e9 nonstreaming 128 |
|---|---:|---:|---:|---:|---:|---:|
| photo-macan | 260,654 | 255,596 | 255,458 | 255,383 | 255,677 | 255,320 |
| photo-ria | 211,044 | 209,466 | 209,275 | 209,267 | 209,480 | 209,217 |
| photo-bliznaca | 242,885 | 241,625 | 241,498 | 241,487 | 241,736 | 241,441 |
| photo-flower | 2,609,895 | 2,592,976 | 2,590,971 | 2,589,320 | 2,592,994 | 2,589,007 |
| gray-macan | 90,321 | 89,801 | 89,624 | 89,674 | 89,812 | 89,617 |
| alpha-ria | 237,730 | 234,861 | 234,693 | 234,696 | 234,869 | 234,677 |
| alpha-flower | 3,833,995 | 3,752,276 | 3,747,796 | 3,745,173 | 3,738,784 | 3,731,392 |
| chessboard | 62,805 | 55,106 | 54,920 | 54,835 | 55,059 | 54,469 |
| splines | 295,203 | 268,237 | 267,077 | 267,085 | 267,018 | 265,852 |
| gray-patches | 18,299 | 17,132 | 16,984 | 16,681 | 16,672 | 15,028 |
| palette-358 | 1,046 | 1,028 | 991 | 984 | 1,029 | 978 |
| gray16-ramp | 1,292 | 1,421 | 1,659 | 1,283 | 1,421 | 1,016 |
| screen-pattern | 385 | 331 | 323 | 315 | 332 | 315 |
| palette-pattern | 438 | 393 | 386 | 373 | 394 | 371 |
| noise16 | 802,311 | 802,493 | 807,338 | 802,402 | 802,504 | 802,226 |
| alpha16-ramp | 19,866 | 20,444 | 24,287 | 19,460 | 20,446 | 16,421 |
| 1029 | 10,821,744 | 10,495,827 | 10,480,535 | 10,470,961 | 10,453,354 | 10,418,747 |
| 1039 | 7,230,909 | 7,047,117 | 7,039,761 | 7,031,893 | 7,031,434 | 7,009,582 |
| 1049 | 7,336,102 | 7,174,471 | 7,148,615 | 7,130,408 | 7,154,457 | 7,107,646 |

Direct balanced 64 improves 16/19 files, but increases the small `gray16-ramp`
file by 129 bytes (9.98%) and `alpha16-ramp` by 578 bytes (2.91%). Direct density
64 also improves 16/19 files, but those regressions grow to 367 bytes (28.41%)
and 4,421 bytes (22.25%); `noise16` grows by 5,027 bytes (0.63%). A blanket direct
policy switch is therefore not qualified by its aggregate gain.

The partition-only native 64 cell improves 18/19 files. Its sole regression is
91 bytes on `noise16` (0.0113%). Reference e9 nonstreaming improves all 19 frozen
files, but its additional model-construction work is substantial.

## Diagnostic construction time

These are means of **one API-boundary call per original 12 MP image**, not
repeated latency measurements. Model time includes final model-header writing.
Reference token-representation conversion is excluded, as is separate context-map
attribution serialization. Native-on-reference-partition time excludes discovery
of the supplied partition. The initial frozen-token producer and independent
validation are outside these timers. Do not combine these figures with full
encoder latency or infer a GPU speed ratio from them.

| Policy | Mean model ms | Mean payload emission ms |
|---|---:|---:|
| Native prefix-derived / 32 | 274.946 | 228.415 |
| Native direct balanced / 32 | 88.118 | 229.947 |
| Native direct balanced / 64 | 90.101 | 264.680 |
| Native direct density / 64 | 282.259 | 272.509 |
| Native on e7 partition / 64 | 143.940 | 274.016 |
| libjxl e7 streaming / 32 | 58.836 | 236.478 |
| libjxl e7 streaming / 64 | 60.910 | 254.497 |
| libjxl e9 streaming / 128 | 93.849 | 281.804 |
| libjxl e9 nonstreaming / 128 | 1971.154 | 293.822 |

The CPU prefix-derived native builder still takes about 275 ms in this replay,
versus 59 ms for reference e7 at the same 32-cluster limit. The native direct
balanced builder is about 88–90 ms. This supports removing the prefix-derived
partition path for an ANS-specific policy. It does not show that every part of
native ANS construction is inferior: with the same reference partition, the
native downstream policy produces smaller aggregate files than reference e7.

## Why the former cost did not buy a rate advantage

The full-search qualification and this controlled replay establish separate
causes. Duplicate token collection/sorting reconstructed populations already
available to the prefix builder. Exact ordered ANS simulations refined selection
without changing the retained pilot's files. Full emission of every candidate
spent work on losing files. Removing those costs preserved the winning files on
the measured pilot and cut complete time substantially. Repeated collection and
sorting add no new model information; estimated selection still needs its own
rate qualification on additional inputs.

Separately, the ANS model inherited a partition selected by the prefix path and a
32-cluster cap. More expensive downstream searches could not fully recover the
payload loss from that partition. The partition-only and capacity controls above
measure that effect directly. Better normalization/configuration search can
still help once the partition is sound, but repeatedly running it on every
full-resolution candidate is not an efficient default strategy.

## Next implementation decision

Keep GPU populations plus exhaustive exact sizing as the measured private
baseline: it is 1.819x faster than the original complete search with identical
files. Keep the four-candidate shortlist experimental because it loses 5/19
winners and regresses `splines` by 7.69%.

The next entropy policy slice should expose the existing native balanced ANS
partition builder, combine that partition with the existing maximum-compression
prepared-population consumer, and reuse GPU per-context counts. Start with 32
clusters to isolate the partition change; qualify 64 as a separate capacity
choice, including merge/storage bounds and 16-bit fallbacks. Retain the current
exhaustive policy as the rate oracle. Repeat complete adaptive searches before
promoting that policy: frozen winning trees can change when the entropy policy
changes, and this study does not measure that interaction.

For complete-call speed, learning and full candidate token generation now account
for 87% of the exhaustive GPU-population route. Reducing those repeated passes is
the larger remaining opportunity. The entropy rate follow-up should avoid
reintroducing expensive full-population searches for every candidate.

## Evidence and reproduction

See [machine-readable cells](modular-entropy-rate-results.json),
[replay method and commands](../tools/modular_entropy_study/README.md),
[pipeline qualification](modular-entropy-pipeline.md), and
[the five-step plan](modular-entropy-efficiency.md).

Replay source revision: `994f1ace103db0dca7e056303f402c71e318a90a`. Binary SHA-256: `939b6c135bb4d52972bde254087c72052aefaac1860be1b4941362a3bf9bda12`.

Pinned libjxl: `e8ff09762481785938d8e4e01333ed3917571161`. Reference libraries and source pins are recorded in the capture manifest.

Manifest: `/Users/yunhocho/GitHub/gjxl-modular-adaptive-gpu/build/entropy-rate-qualified-20261011/manifest.json`. SHA-256: `50ac932166c7e5963d85d1edf3d4cfe39fe74c0e199f049a01c2049d5127221e`.

The artifact audit validates all 19 immutable token archives, policies, commands, output files, decoded-sample results, cost decompositions and partition assignments. The two small replay fixtures also cover empty contexts and reject malformed archives and overwrite attempts. Tree/transform learning, LZ77, default coder selection and libjxl full-encoder timing remain outside this comparison.
