# Native direct ANS on captured Modular decisions

**The required native clustering machinery already exists.** Calling GJXL's
direct-ANS optimizer on the qualified e9 token streams recovers the benefit
identified by the [reference-partition transplant](modular-entropy-ablation.md).
With its rate mode and a 64-histogram cap, the automatic five-image total is
**6,878,464 bytes**, or **−0.3282% versus stock e9**, compared with +2.9747%
through the current Prefix-derived ANS partition. Forced-buffered mode improves
from +6.5071% to +0.5720% on the same five images.

This remains a captured-decision experiment. All entropy coding is native in
these controls, but transforms, WP parameters, trees and stream scopes come
from the reference capture. The private/public encoder routing is unchanged.
No timing, production storage bound or default promotion is qualified here.
The [complete ledger](modular-direct-ans-results.json) includes every size,
bit attribution and evidence binding.

The [estimated-width follow-up](modular-direct-ans-estimated.md) now qualifies
an explicit native option that removes ordered-token cost simulation when no
cost is requested. It keeps the ramp's rate/64 size and reduces paired
sample-model CPU by 64.1% for a 27-byte automatic-cohort increase. Stream-level
integration and whole-encoder timing remain separate work.

## Existing modes under test

The adapter calls `OptimizeDirectAnsEntropyCode` directly, with the same
logical tokens and native tree/header coding as the original control. It does
not first build a Prefix model. All four new variants disable LZ77 because
this native optimizer does not implement it.

| Variant | Cluster cap | Integer configurations | Histogram search | Alphabet width |
|---|---:|---|---|---|
| `native_direct_balanced` | 32 | Fixed 4/2/0 | Approximate | First viable |
| `native_direct_high_density` | 32 | Existing 28-choice portfolio | Precise | First viable |
| `native_direct_rate` | 32 | Existing 28-choice portfolio | Precise | Exact candidate comparison |
| `native_direct_rate64` | 64 | Existing 28-choice portfolio | Precise | Exact candidate comparison |

[`ans.cpp`](../src/codestream/ans.cpp) already provides Shannon-distance
seeding/assignment with a 48-bit separation threshold. High-density/rate modes
also merge clusters using ANS population costs. This is the appropriate native
counterpart to the reference grouping pipeline; it does not require another
independent clustering port. Native floating-point arithmetic and refinements
are not claimed to reproduce every reference map.

Balanced mode skips the merge refinement. High-density and rate mode have
identical partitions at the same cap; rate mode also enables histogram/config
search optimizations and compares the viable alphabet-width candidates.
Calling either first-viable mode without a requested cost avoids the optimizer's
exact ordered-token cost passes. This follows from source, not a new timing
measurement.

## Measured sizes

| Native data mode | Automatic bytes | Difference from stock | Buffered bytes | Difference from stock |
|---|---:|---:|---:|---:|
| Original Prefix-derived ANS | 7,106,401 | +2.9747% | 7,207,625 | +6.5071% |
| Direct balanced, 32 | 6,915,818 | +0.2131% | 6,900,775 | +1.9728% |
| Direct high-density, 32 | 6,902,834 | +0.0250% | 6,893,428 | +1.8642% |
| Direct rate, 32 | 6,898,213 | −0.0420% | 6,887,215 | +1.7724% |
| Direct rate, 64 | **6,878,464** | **−0.3282%** | **6,805,981** | **+0.5720%** |

These totals use the original five-image policy cohort, excluding the separate
16-bit ramp. The reference stock totals are 6,901,111 automatic and 6,767,272
buffered bytes. The common-assembly reference entropy baseline from the prior
ablation differs by only a few header/empty-payload bytes, as documented there.

Splines falls from 351,413 native bytes to 258,611 even in balanced mode and
257,365 in rate/64 mode, versus 258,951 stock. The native direct path therefore
confirms the grouping diagnosis without a reference clusterer in model
construction. Increasing the cap from 32 to 64 in rate mode saves another
19,749 automatic bytes and 81,234 buffered bytes; these are directly measured
comparisons between otherwise identical policies.

Per-image results still matter. High-density/32's automatic chessboard is
138,378 bytes versus 136,329 stock, despite its near-zero aggregate gap.
Buffered rate/64 chessboard is 145,008 versus 140,629 stock. Neither aggregate
result substitutes for the existing per-image review and held-out gates.

## The 16-bit ramp identifies the next cost/rate control

| `alpha16-ramp`, automatic | Bytes | Data alphabet log | Data histograms |
|---|---:|---:|---:|
| Stock e9 | 8,375 | — | — |
| Reference entropy with LZ77 disabled | 47,576 | 8 | 46 |
| Original native ANS | 79,681 | 7 | 16 |
| Direct balanced, 32 | 86,230 | 7 | 32 |
| Direct high-density, 32 | 109,297 | 5 | 32 |
| Direct rate, 32 | 59,493 | 8 | 32 |
| Direct rate, 64 | **47,547** | **8** | **46** |

The fast high-density mode is not safe to select universally. Its first viable
alphabet is log 5, which excludes many useful high-depth configurations. At
the same 32-cluster partition, rate mode selects log 8 and nearly halves this
file. The reported context maps are equal, so this is a downstream selection
difference, not a different learning or grouping result.

With rate/64, native coding is within 29 bytes of the reference **without
LZ77**. Reference LZ77 saves 39,205 bytes on this exact control. Thus the wider
native portfolio/cap recovers the previously unexplained no-LZ77 discrepancy,
and LZ77 is the clear remaining feature for this ramp. This does not establish
LZ77's benefit or parsing cost on a broader corpus.

Rate mode obtains its width decision by simulating exact ordered token cost
for viable candidates, the hotspot that motivated the earlier ANS work.
The next experiment should choose the width using histogram/model estimates,
retaining the current exact mode as a control. The shared optimizer already
has an estimated-width branch used by another entry point. Extending that
explicit choice to direct rate mode is a small, testable change; assuming that
the first viable width is sufficient is contradicted by this ramp.

## Qualification and next integration

All **55 Release files** (11 cases × five modes) decode exactly. All 11 original
native controls match both prior frozen studies and the prepared-stream
encoder. The adapter verifies that each direct mode preserves every native
tree scope. ASan/UBSan covers Macan, local-palette chessboard, the true 16-bit
ramp and the buffered alpha tree: all **20 files** match Release exactly.
All ten research CTest entries pass in both builds. The extended analyzer also
reproduces the previous 15-control report unchanged apart from its own source
hash, preserving the earlier qualification.

The validation manifest binds 1,978 files, including executables, native/tool
snapshots, reference hooks, compiler commands, linked libraries, source
captures, outputs and test logs. Sanitizers instrument native libraries and
the harness/generated reference units; linked reference archives remain
uninstrumented, vptr checks are excluded, and LeakSanitizer is unavailable on
this macOS runtime. Correctness runs overlap; their durations are not timing
evidence. The original input/capture profile and unsupported features remain
unchanged.

Estimated width selection is now [qualified separately](modular-direct-ans-estimated.md),
including independent candidate-cost checks, managed storage, the captured
ramp and paired model-construction time. Next add an explicit private
prepared-stream entropy policy with stream-level managed-storage and
failure-publication tests. Keep the exact and Prefix-derived controls; default
routing requires the full rate/time/resource gates. Native palette/RCT/WP
selection and one fit per selected scope still need integration afterward.

For reproduction, use `run_entropy_replay.py` with `--variants native
native_direct_balanced native_direct_high_density native_direct_rate
native_direct_rate64`. A direct-mode study requires the `native` control.
`analyze_direct_entropy.py` checks the Release, reference-entropy, sanitizer
and frozen validation studies before writing the report; see `--help` for
paths. No learning or transform selection is rerun.
