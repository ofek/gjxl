# CPU Modular RCT selection

Opt-in CPU Modular search now ranks all 42 implemented reversible color
transforms and evaluates up to three additional RCTs beyond 0, 6, and 9.
The default and prescribed policies are unchanged. This is encoder policy
within the existing CPU/GPU arithmetic contract; it adds no bitstream feature
or public option.

The subsequent [transform context-learning milestone](modular-palette-context.md)
adds up to four learned transform candidates and corrects stream-local channel
properties. The measurements below retain their original scope.

## Bounded selection

The ranker reads at most 2,048 deterministically jittered pixel neighborhoods
directly from the packed input. It respects 256-pixel group boundaries,
unaligned backing, row padding, and 8/16-bit byte order. Alpha is unchanged and
excluded from the color-transform ranking. Grayscale returns an empty shortlist
without allocating scratch.

For each of the 42 RCTs, it transforms the sampled current/left/top/top-left
values and estimates gradient, left, and top residual entropy separately for
each color channel. The score uses the same integer Q12 logarithm and fixed
HybridUint 4/2/0 proxy as context learning, including extra bits and estimated
model overhead. It sums the best predictor estimate for each channel; actual
tree learning and entropy encoding still optimize their own policies.

Candidates are ordered by estimated cost, with ascending RCT ID breaking ties.
The shortlist excludes 0/6/9 and candidates whose sorted per-channel score
triples duplicate a retained or previously shortlisted candidate. This avoids
spending the shortlist on permutations with identical estimates. It is
heuristic deduplication, not proof of equivalent final codestreams. The method
does not claim to find the best of all 42 fully encoded transforms.

The complete existing search runs first, in its original order. Each additional
RCT then gets the fixed weighted tree and an entropy-learned tree; duplicate
trees are skipped. Only a strictly smaller complete codestream replaces the
winner. Earlier candidates win ties, so successful search cannot regress file
size against the retained search. The ceiling is 23 color candidates and 9
grayscale candidates.

Ranking scratch uses one managed allocation and an allocation-free storage
planner. Its lifetime ends before additional training/encoding. Search
admission includes coexistence with the retained winning codestream. Failure
leaves the caller's output unchanged.

## Compression qualification

The paired 2026-10-10 capture compares against context-learning commit
`2cdb74c`, using its frozen libraries and saved outputs. The pinned libjxl
reference remains `e8ff09762481785938d8e4e01333ed3917571161` at effort 9.
Inputs, sample hashes, integer interpretation, alpha preservation, sRGB/gray
declarations, raw-codestream framing, and metadata exclusions match the
[context-learning qualification](modular-context-learning.md).

Aggregate complete-codestream bytes for search+ANS:

| Group | Before | After | Reduction | Remaining gap to libjxl e9 |
|---|---:|---:|---:|---:|
| 11 PNG fixtures | 8,598,720 | 8,368,742 | 2.67% | 9.05% |
| 4 RGB photos | 3,600,658 | 3,486,826 | 3.16% | 7.67% |
| 2 RGBA photos | 4,338,611 | 4,222,479 | 2.68% | 7.07% |
| 1 grayscale photo | 94,001 | 94,001 | 0.00% | 4.27% |
| 4 graphics fixtures | 565,450 | 565,436 | 0.0025% | 40.85% |
| 5 synthetic patterns | 855,025 | 852,476 | 0.30% | 1.81% |
| All 16 inputs | 9,453,745 | 9,221,218 | 2.46% | 8.34% |

The 11-fixture gap to libjxl e9 falls from 12.05% to 9.05%. Both entropy modes
are checked: all 64 outputs decode sample-exactly through the independent
libjxl decoder, all 32 default outputs remain byte-identical, and none of the
32 search outputs grows. Every equal-size result also retains the old bytes.
Grayscale search is byte-identical. These are a small corpus's measured rate
effects, not general compression guarantees or a throughput result.

The gains are concentrated in color photographs. For example, `photo-flower`
selects RCT 27 and shrinks 3.87%; `photo-macan` selects RCT 13 and shrinks 1.13%.
Five of the 11 PNG fixtures improve; six are unchanged. The RGB/RGBA flower
variants account for 98.54% of the aggregate PNG byte savings, so the aggregate
reduction is not a typical per-image gain and includes related inputs.
The graphics gap remains substantial. Local RCT selection, richer palettes,
and learning after palette/Squeeze remain separate work.

A separate four-fixture diagnostic fully encodes both context policies for all
42 RCTs, independently checking all 336 candidate encodes. It also retains the
old search winner as a fallback. The bounded search leaves the following
headroom relative to that diagnostic:

| Fixture | Bounded search bytes | All-42 diagnostic bytes | Additional bytes |
|---|---:|---:|---:|
| photo-macan | 268,426 | 266,928 | 1,498 (0.56%) |
| photo-flower | 2,748,396 | 2,748,358 | 38 (0.0014%) |
| splines | 396,670 | 391,845 | 4,825 (1.23%) |
| alpha16-ramp | 28,470 | 27,965 | 505 (1.81%) |

The shortlist is deliberately bounded and can miss the best RCT, including
permutations with equal sampled score profiles. This diagnostic is not an
exhaustive search over trees, palettes, Squeeze, or local transforms. The
production policy was frozen before this check; these four fixtures were not
used to tune its shortlist afterward.

## Validation and retained evidence

Native ARM64 Release validation passes all eight focused Modular suites.
The policy suite covers 835 prescribed
stage cases and 1,738 complete files, including independent libjxl prediction,
token and decode checks for shortlisted RCTs, exact storage admission,
serial/parallel equality, and the complete-candidate ceiling. Ranker tests cover
determinism, constant distributions, 256-pixel boundaries, single rows/columns,
byte order, padding, alpha independence, grayscale bypass, invalid input,
managed allocation failure, and scratch release.

The foundation and policy suites also pass Debug ASan/UBSan (399 seconds).
The pinned libjxl binaries are not instrumented; existing reference targets
disable vptr checks because their pinned archives omit RTTI.

The workflow test supports `--allocation-shard INDEX COUNT` to partition
failure indices across processes. Default invocation still visits every index.
The union of shards 0 through COUNT-1 covers the same exhaustive sweep; each
shard verifies failure atomicity, resource release, admission, and recovery.
The guard is 50,000 allocations per case, since the expanded ANS search exceeds
the previous 30,000 limit. The existing 7,200-second CTest timeout is retained.

All 44 failure-injection cases pass across four disjoint shards in 1,072 seconds
(17 minutes 52 seconds). Their verified union covers all 133,331 allocation
failure points, including 114,984 search points. The initial unsharded run was
intentionally stopped to switch to this partition and is not counted as
passing evidence. All non-search allocation counts match the preceding
context-learning qualification; search counts match the independent preflight
allocation counters.

The local `build/modular-rct-search/` directory retains frozen libraries,
runner/source snapshots, input and output hashes, every sampled RCT score,
shortlists and selected policies, both entropy modes' rate comparisons, and
test logs. The checked-in
[qualification record](modular-rct-search-results.json) identifies the exact
source and evidence. Diagnostic single-run timings were collected alongside
tests and do not establish a speed or equal-time advantage. Additional
candidates increase CPU search work; GPU execution and automatic routing are
unchanged.
