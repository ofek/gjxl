# CPU Modular tree checkpoints and larger training budgets

Step #3 builds on the committed budget representation (`3372d65`) and
coverage experiment (`9218790`). It adds optional intermediate-tree selection
to CPU search and measures larger sample and leaf budgets. Public API/CLI
defaults and GPU capacity remain unchanged.

**Normal search with 262,144 samples, up to 256 leaves and checkpoint selection
reduces the PNG aggregate gap versus pinned libjxl e9 from +4.0772%
to +2.4754%.** This saves 122,916 bytes, recovering
39.2856% of the remaining Step #2 excess.
Relative to the original +9.0529% gap, Steps #2–#3 together recover
72.6561% of the original excess.

## Actual normal-search outputs

All sizes below are newly measured complete raw lossless Modular files with
ANS, including tree/model/transform headers and padding. The existing learner
and all 14 predictors are used for this rate sweep.

| Search setting | PNG bytes (11 inputs) | Gap vs e9 | All 16 bytes | Gap vs e9 |
|---|---:|---:|---:|---:|
| Original default, 4K/16 | 8,368,624 | +9.0529% | 9,205,898 | +8.1615% |
| Step #2, 64K/64 final tree | 7,986,788 | +4.0772% | 8,814,109 | +3.5583% |
| 64K/64 + checkpoints | 7,956,945 | +3.6883% | 8,784,266 | +3.2077% |
| 64K/256 + checkpoints | 7,946,633 | +3.5539% | 8,773,954 | +3.0866% |
| 256K/256 + checkpoints | 7,863,872 | +2.4754% | 8,688,162 | +2.0786% |

Pinned libjxl revision: `e8ff09762481785938d8e4e01333ed3917571161`.
Default e9 totals are 7,673,910 bytes for the 11 PNG-derived inputs and
8,511,250 bytes for all 16 inputs. PNG e9 controls were rerun byte-identically;
the five synthetic reference files use the verified saved ledger.

Each checkpoint search preserves the original default search. At the same
64K sample budget, the measured searches also never regress as checkpoints
are enabled and the cap increases from 64 to 256 leaves.
The 256K/256 run regressed versus the previous 64K/64 search on: alpha-flower, alpha16-ramp.
That last comparison is an observed result, not a cross-sample-budget
guarantee: changing the sample budget changes the training set.

## Frozen-transform checkpoint curves

For each input, freeze the baseline winner's RCT, palette/Squeeze sequence
and Weighted settings. Train once per sample budget to a 256-leaf cap,
capturing 16/32/64/128/256-leaf checkpoints and the final early-stop tree.
The checkpoint selection column sums the smallest measured complete file
per input among the original baseline and checkpoints through the stated cap.
It is a selection envelope over measured files, not another normal-search run.

| Sample budget | Leaf cap | Raw PNG bytes | Checkpoint selection bytes | Selected gap |
|---:|---:|---:|---:|---:|
| 65,536 | 16 | 8,220,789 | 8,216,633 | +7.0723% |
| 65,536 | 32 | 8,022,747 | 8,022,747 | +4.5458% |
| 65,536 | 64 | 8,097,497 | 7,976,561 | +3.9439% |
| 65,536 | 128 | 8,081,527 | 7,959,961 | +3.7276% |
| 65,536 | 256 | 8,079,780 | 7,958,234 | +3.7051% |
| 262,144 | 16 | 8,287,325 | 8,272,683 | +7.8027% |
| 262,144 | 32 | 8,008,795 | 8,008,795 | +4.3639% |
| 262,144 | 64 | 7,949,012 | 7,934,080 | +3.3903% |
| 262,144 | 128 | 7,926,685 | 7,908,396 | +3.0556% |
| 262,144 | 256 | 7,868,177 | 7,864,047 | +2.4777% |

Raw file size is not monotonic with leaf capacity. For splines at 64K
samples, the frozen 32-leaf tree costs 325,978 bytes; the final 55-leaf tree
costs 446,914 bytes. At 256K, its final 188-leaf tree costs 295,203 bytes.
On photo-flower at 256K, 64 leaves cost 2,609,893 bytes, while the 256-leaf
candidate costs 2,611,880 bytes. The encoder therefore compares complete
files rather than assuming the final tree is best. This study does not
attribute these reversals to a specific learner or entropy-model mechanism.

## Matching bounded-reference factorial

The frozen instrumented libjxl runner forces global coding and the RCT type
selected by the native baseline. Reference transform/channel-layout traces
are identical across the four budget cells. Native and reference pipelines
are not otherwise identical: the reference retains its own transforms,
patch handling, sampler, tree learner and entropy model.

| Sample cap | Leaf cap | PNG bytes | Gap vs default e9 |
|---:|---:|---:|---:|
| 65,536 | 64 | 7,900,684 | +2.9551% |
| 65,536 | 256 | 7,769,152 | +1.2411% |
| 262,144 | 64 | 7,888,933 | +2.8020% |
| 262,144 | 256 | 7,690,026 | +0.2100% |

All four cells are measured. The positive factorial interaction is
67,375 bytes: greater sample coverage makes added capacity more valuable.
Reference capacity uses its own split-threshold decisions, with doubling
and eight bisection steps to obtain a tree within the requested cap; it is
not an optimal bounded-tree algorithm. Native capacity uses best-first growth.
The reference result supports the value of coverage and capacity, but it
does not establish that those two changes alone make the native encoder
equivalent to libjxl.

## Implementation

- Private `ModularEncodingOptions::tree_checkpoints` enables selection.
  Both existing and quantized learners support optional checkpoint output.
  The rate sweep uses the existing learner; the quantized path is qualified
  for correctness without repeating its full rate matrix.
- One learning pass captures powers of two from 16 leaves through the cap,
  plus the final tree on early stopping or a non-power-of-two cap. Captures
  match independent lower-cap runs on the same samples. Search encodes the
  final tree first, then the other captures. Existing final-tree candidates
  win size ties, including candidates evaluated later in transform search.
- Original default candidates remain available. Increasing a power-of-two
  cap retains its lower checkpoints; larger sample budgets do not retain
  the previous experimental sample set. No mixed-sample portfolio is added.
- Managed storage plans include snapshot backings during learning and
  encoding. Failed learning leaves both tree and checkpoint outputs intact;
  failed encoding preserves the caller output and releases reservations.

## Per-image complete normal search

| Input | libjxl e9 | Step #2 64K/64 | Checkpoints 64K/64 | Checkpoints 64K/256 | Checkpoints 256K/256 |
|---|---:|---:|---:|---:|---:|
| alpha-flower | 3,715,769 | 3,830,474 | 3,830,474 | 3,830,474 | 3,833,995 |
| alpha-ria | 228,013 | 238,976 | 238,976 | 238,976 | 237,730 |
| alpha16-ramp | 8,375 | 19,838 | 19,838 | 19,838 | 19,866 |
| chessboard | 136,329 | 103,400 | 103,400 | 96,500 | 62,805 |
| gray-macan | 90,153 | 90,842 | 90,842 | 90,842 | 90,319 |
| gray-patches | 4,912 | 21,588 | 21,588 | 18,731 | 18,299 |
| gray16-ramp | 2,502 | 1,293 | 1,293 | 1,293 | 1,292 |
| noise16 | 825,073 | 805,318 | 805,318 | 805,318 | 802,309 |
| palette-358 | 1,243 | 1,045 | 1,045 | 1,045 | 1,045 |
| palette-pattern | 1,047 | 463 | 463 | 463 | 438 |
| photo-bliznaca | 240,347 | 244,280 | 244,280 | 244,280 | 242,885 |
| photo-flower | 2,537,909 | 2,624,437 | 2,624,437 | 2,624,437 | 2,609,893 |
| photo-macan | 252,153 | 263,843 | 263,843 | 263,288 | 260,654 |
| photo-ria | 208,131 | 212,082 | 212,082 | 212,082 | 211,044 |
| screen-pattern | 343 | 409 | 409 | 409 | 385 |
| splines | 258,951 | 355,821 | 325,978 | 325,978 | 295,203 |

## Qualification and evidence

- 21 selected Modular CTests passed: the checkpoint test separately, plus
  20 foundation, conformance, oracle, transform, policy, API/CLI, allocation
  and CPU-side GPU-contract tests. The exhaustive legacy `modular_workflow`
  suite was not rerun. This is not hardware GPU execution qualification.
- 58,913 complete-workflow allocation-failure injections passed across
  prescribed, existing, quantized and checkpoint prefix/ANS cases on the
  small fault-sweep fixture. The new large checkpoint learners separately
  passed every managed allocation failure: 10 existing and 11 quantized.
- Tests cover snapshot identity, non-power-of-two caps, early stopping,
  transformed-policy exact decoding, serial/four-thread byte identity,
  tie behavior, atomic failure, and exact/one-byte-under storage admission.
- Three ASan/UBSan tests passed: foundation, checkpoint learning/encoding
  and GPU workflow contracts. Build: O1 RelWithDebInfo. Pinned reference
  archives were not sanitizer-instrumented.
- All 311 measured complete files decoded sample-exactly: 176 fixed native,
  80 normal-search native and 55 reference files, including hidden RGB under
  alpha. 102 rerun controls matched saved complete-file SHA-256 hashes
  (48 fixed native, 32 search native and 22 reference).
- An initial test incorrectly required the quantized learner to reach
  exactly 256 leaves. It legitimately stopped earlier. The assertion was
  corrected to allow early stopping while still requiring every checkpoint
  to match an independent capped run. The retained initial failure log is
  excluded from passing qualification.

The [structured results](modular-tree-checkpoints-results.json) contain the
aggregate/per-image measurements and source/artifact hashes. Reproduction
commands and control semantics are in the [study README](../tools/modular_training_study/README.md).
New evidence is retained under `build/training-checkpoints/` and
`build/training-checkpoints-sanitize/`; prior Step #1/#2 artifacts are preserved.
The source snapshot includes the exact generated reference overrides and
reference harness used by the frozen binary.

This remains a small corpus with related RGB/gray/alpha images. Timings
were collected with concurrent workers and are diagnostic only. No public
default change, throughput qualification or broad libjxl parity is claimed.

## Next decision

Proceed to Step #4: replay the richer trees' actual token streams through
both entropy-model builders, holding tokens and context identities fixed.
Separate token cost, model headers and context-map cost; sweep native model
capacity beyond 32 clusters only as a controlled replay. If the measured
benefit is material, verify it in complete-file encodes before promoting
an entropy change. Keep tree/sampling differences separate from that test.
