# CPU Modular context learning

The first compression milestone adds entropy-based context learning to the
opt-in CPU Modular search. The default and explicitly prescribed policies keep
their existing behavior. This extends encoder policy within the existing
31-node / 16-leaf CPU and GPU oracle contract; it adds no bitstream feature or
public API option.

## Search changes

The original baseline, one-leaf and one-split policies for RCT 0/6/9, palette,
and Squeeze candidates remain in their original order. Search then evaluates
two additional policies per RCT (RCT 0 only for grayscale):

- A fixed 16-context tree using the weighted predictor and property 15. Its
  split thresholds are -127, -63, -31, -15, -7, -3, -1, 0, 1, 3, 7, 15, 31,
  63, and 127, scaled by 16 for 16-bit input.
- A recursively learned tree, up to 16 leaves, that scores all 14 predictors
  and 16 implemented properties. It can split into different distributions
  even when both child contexts use the same predictor.

Each candidate is encoded completely. Only a strictly smaller codestream
replaces the winner, including tree/model/transform overhead, TOC and padding.
Earlier candidates win ties. Consequently a successful search cannot produce
more bytes than the retained search on the same input and entropy mode. The
candidate ceiling rises from 11 to 17 for color and from 7 to 9 for grayscale.
Learned policies identical to a legacy candidate, and unavailable transforms,
are skipped.

The learner uses at most 4,096 samples. Each nonempty stream/channel slice gets
one sample when the budget permits; remaining samples are allocated by area.
Deterministic jitter within strata reduces aliasing on periodic graphics.
The original uniform samples are also retained for the legacy candidates.

For learning only, residuals are mapped to a fixed HybridUint 4/2/0 alphabet.
The score combines histogram Shannon cost, verbatim extra bits, and estimates
for model and split overhead. Integer Q12 logarithms make scoring reproducible
without floating-point ties. The learner examines up to 32 quantile-derived
boundaries per property, requires at least eight samples per child, and chooses
the greatest estimated gain at each step. Final entropy encoding still performs
its existing HybridUint and prefix/ANS model optimization. The estimated score
is a heuristic; complete-codestream comparison decides whether it helps.

Scratch memory uses managed allocation and has an allocation-free planner.
Search admission includes both sample sets, the learner workspace, 16-context
model storage, and coexistence with the retained winning codestream. Failures
do not publish partial outputs.

## Compression qualification

The 2026-10-10 paired run uses the same 16 inputs as
`build/modular-rate-20261010`: 11 PNG fixtures and five synthetic patterns.
Inputs retain exact integer samples, including RGB under transparent alpha;
ancillary PNG metadata is omitted. Results below are sums of complete raw JXL
codestream bytes for GJXL **search + ANS**, compared with the previous CPU
search and pinned libjxl `e8ff09762481785938d8e4e01333ed3917571161` at effort 9.
These are corpus-specific lossless size comparisons, not equivalent-effort or
throughput comparisons.

| Corpus | Previous bytes | New bytes | Reduction | New excess over libjxl e9 |
| --- | ---: | ---: | ---: | ---: |
| 11 PNG fixtures | 9,874,064 | 8,598,720 | 12.92% | 12.05% |
| 4 RGB photos | 4,055,266 | 3,600,658 | 11.21% | 11.18% |
| 2 RGBA photos | 4,858,400 | 4,338,611 | 10.70% | 10.01% |
| 1 grayscale photo | 117,256 | 94,001 | 19.83% | 4.27% |
| 4 graphics fixtures | 843,142 | 565,450 | 32.94% | 40.86% |
| 5 synthetic patterns | 946,004 | 855,025 | 9.62% | 2.11% |
| All 16 inputs | 10,820,068 | 9,453,745 | 12.63% | 11.07% |

The 11-fixture gap to libjxl e9 falls from 28.67% to 12.05%. All 64 native
outputs (default/search crossed with prefix/ANS) decode sample-exactly through
the independently linked pinned libjxl decoder. All 32 default outputs are
byte-identical to the baseline; none of the 32 search outputs grows.

The local evidence directory `build/modular-context-learning/rate/` retains
the input manifest, frozen libraries, runner source, per-output hashes,
`results.jsonl`, and `summary.json`. A separate `rate-final/` capture records
the final source hashes and patch and checks all 64 outputs against that run.
The checked-in [qualification record](modular-context-learning-results.json)
records test results and evidence hashes.

Concurrent GPU integration subsequently extracted the shared prepared-frame
encoder. The combined CPU path was rebuilt and rechecked with all eight focused
suites and a separate `rate-integrated/` capture. All 64 outputs remain
sample-exact and byte-identical to the frozen context-learning measurements.
The combined foundation and policy suites also pass Debug ASan/UBSan.
The qualification record distinguishes this combined snapshot from the
original feature snapshot used for the exhaustive allocation sweep.

## Validation and limits

Focused tests cover useful same-predictor splits, recursive contexts,
determinism, constant distributions, integer extrema, allocation failure,
and exact learner storage admission. Policy tests exercise fixed and learned
trees across all six integer profiles, independently compare predictions,
properties and tokens, decode both entropy modes, and compare serial/parallel
execution. Tiny-image search is also covered. The frozen prescribed CPU oracle
and original legacy learner artifacts remain unchanged.

Native ARM64 Release validation passes all eight focused Modular suites.
The final-source policy run covers 811 prescribed stage cases and 1,682 complete
files. Debug ASan/UBSan runs pass the foundation and policy suites. A saved-byte
comparison with the pre-change native qualification finds all 2,802 default
files, 626 transform files, and 286 frozen artifacts plus their manifest
unchanged.

The workflow allocation-failure sweep passes all 44 cases, including all
70,489 failure points across the four search cases. It checks unchanged outputs,
released backing and admission, retained-result ownership, and successful
recovery. Repeating complete search at each allocation boundary takes
2,530 seconds (42 minutes) on this run. Its explicit CTest timeout is 7,200
seconds, overriding the native CI command's 600-second default without dropping
failure cases.

Search now performs additional training and complete encodes. This run does
not establish a throughput improvement or a quality/speed frontier; timing was
collected concurrently with tests and is not used for performance claims.
The corpus is small and the graphics gap remains material. Expanded RCT
selection, richer palettes, LZ77, previous-channel properties, learned trees
after palette/Squeeze, and libjxl learner parity remain separate work.

GPU execution can consume the resolved bounded tree through the existing
contract. Policy training and candidate selection remain on CPU. Independent
GPU arithmetic, resource admission and transfer-inclusive performance still
need their own qualification.
