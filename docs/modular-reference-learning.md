# Pinned-reference Modular learner bridge

The first milestone of the [CPU learner replacement plan](modular-tree-learning-cost.md#recommended-implementation-order)
is implemented in research commit `fe7cd44`. The adapter can capture and replay
the actual pinned libjxl fitter, learn from GJXL-prepared stream images, and
encode the imported tree through GJXL's unchanged entropy path. Production
sources and defaults are unchanged.

The complete 16-input comparison establishes a useful rate result: the
all-predictor reference learner reduces the current aggregate gap from
**+2.0786% to +0.9487% versus stock e9** on frozen native winning transforms.
It also establishes why this is not a production configuration: the
two-predictor and all-predictor policies have different regressions, and the
all-predictor fits remain expensive. A faithful native fitting implementation
is now justified; predictor selection and search policy still need separate
qualification.

## Controls and correctness

The reference is libjxl `e8ff09762481785938d8e4e01333ed3917571161`; native
archives are the qualified estimated-ANS implementation at `f27a797`.
[Build/run instructions](../tools/modular_reference_learning/README.md) and the
[verified ledger summary](modular-reference-learning-results.json) record the
inputs, source hashes, binaries, archives and raw artifact locations.

Three controls run on each input:

- **Stock e9:** the pinned full reference encoder. Its 11 available prior
  corpus controls reproduce their saved SHA-256 values. The other five are
  newly captured synthetic controls; all 16 reconstruct the source exactly.
- **Reference Best through GJXL:** the pinned learner's Weighted/Gradient
  configuration, with reference e9 sampling, quantization and growth, on the
  current native winner's transformed stream slices.
- **Reference Variable through GJXL:** the same adapter with all 14 predictors.
  It does not retain the saved native tree as a fallback.

The bridge retains stream-ID holes, local channel numbering, metadata channel
placement, shifted shapes and predictor reset boundaries. Before native
encoding, every serialized tree token and every residual/context token must
match reference tokenization of those same stream images. The resulting files
are decoded independently by libjxl and compared against all source integers,
including hidden RGB. All **48 corpus encodes** pass these checks where
applicable; all **233 captured fitting events** reproduce their complete
reference tree after reading the fixture back from disk.

Fixtures preserve quantized properties, predictor tokens/extra-bit counts,
sample multiplicities, predictor/property order, quantization maps and cuts,
static/multiplier ranges, exact float parameter bits and the expected tree.
They capture inputs before fitting rearranges sample rows. A separate persistence
test requires byte-identical read/write roundtrips, rejects ten malformed
fixtures, and detects a valid-shaped but false expected tree. Import tests
cover field preservation and rejection of unsupported properties/predictors,
bad children, cycles/shared children, offsets and multipliers.

The persistence/rejection tests, a photographic replay, and an all-predictor
16-bit RGBA bridge encode also pass AddressSanitizer/UndefinedBehaviorSanitizer
with `UBSAN_OPTIONS=halt_on_error=1`. The sanitized RGBA file matches Release
bytes. Instrumentation covers the harness and generated reference fitting and
preparation units; linked native/reference archives are uninstrumented. Vptr
checks are excluded for interoperability with the pinned no-RTTI archive. This
is boundary/fitter validation, not a whole-codec sanitizer qualification.

## Rate results

These are complete codestream sizes, with identical source pixels and frozen
native transform choices. The bridge changes the learning algorithm and
training coverage together; this is not an isolated sampling or quantization
ablation. It also uses the reference growth thresholds rather than imposing
the previous 256-leaf learning budget.

| Encoder/control | Total bytes | Gap versus stock e9 | Change versus current native |
|---|---:|---:|---:|
| Stock libjxl e9 | 8,511,250 | — | −2.0363% |
| Current qualified native | 8,688,169 | +2.0786% | — |
| Reference Best through GJXL | 8,837,251 | +3.8302% | +1.7159% |
| Reference Variable through GJXL | 8,591,998 | +0.9487% | −1.1069% |
| Smaller of the two bridge files, selected in hindsight | 8,577,404 | +0.7773% | −1.2749% |

The hindsight row is a diagnostic bound, not an implemented selection policy
or a speed result. It runs both learners and selects by final bytes. No row
includes a fallback to the saved native tree.

| Input | Current native | Reference Best bridge | Reference Variable bridge |
|---|---:|---:|---:|
| photo-macan | 260,654 | 264,156 | 260,537 |
| photo-ria | 211,044 | 211,586 | 210,993 |
| photo-bliznaca | 242,885 | 244,409 | 242,755 |
| photo-flower | 2,609,895 | 2,552,130 | 2,563,089 |
| gray-macan | 90,321 | 91,346 | 90,463 |
| alpha-ria | 237,730 | 238,933 | 236,364 |
| alpha-flower | 3,833,995 | 3,740,479 | 3,744,114 |
| chessboard | 62,805 | 214,321 | 57,560 |
| splines | 295,203 | 366,049 | 353,406 |
| gray-patches | 18,299 | 23,477 | 17,314 |
| palette-358 | 1,046 | 1,384 | 1,087 |
| gray16-ramp | 1,292 | 1,715 | 1,318 |
| screen-pattern | 385 | 671 | 389 |
| palette-pattern | 438 | 637 | 402 |
| noise16 | 802,311 | 825,378 | 791,177 |
| alpha16-ramp | 19,866 | 60,580 | 21,030 |

More predictors do not monotonically improve final size: Best wins on both
large photographs. Conversely, Variable removes most of Best's chessboard
loss and improves the current native chessboard file. Neither bridge mode
recovers the current native result on splines or alpha16-ramp. These are
explicit blockers to promoting a blanket replacement policy, not failures of
token mapping or lossless reconstruction.

## Learning time and scope

The repeated pilot has one warmup and three measured fresh processes for each
input/control, interleaved by variant. Capture/replay is disabled in timing
processes; all 45 pilot encodes, including captures and warmups, reproduce the
corresponding corpus bytes. Values are medians of process CPU seconds, summed
over every learning event:

| Input | Stock e9 fit / preparation+fit | Best bridge fit / preparation+fit | Variable bridge fit / preparation+fit |
|---|---:|---:|---:|
| photo-macan | 0.395 / 0.499 | 0.407 / 0.516 | 2.337 / 2.567 |
| alpha-ria | 0.533 / 0.688 | 0.558 / 0.711 | 3.289 / 3.592 |
| gray-patches | 0.064 / 0.100 | 0.151 / 0.188 | 0.790 / 0.871 |

These are diagnostic measurements on an uncontrolled M4 Pro desktop. The
reference and native pipelines supply different transformed samples and trees;
this table is not an identical-input kernel benchmark. The bridge preparation
column above is reference sample preparation plus fitting; copying native
stream planes into the research adapter is recorded separately. Full encode
and validation work is outside these learning timers.

The larger images expose a separate scope issue. Stock e9 learns 54 smaller
trees on each flower input, while the bridge learns one global tree. Best's
global fit sees 6,703,683 samples / 3,665 leaves for photo-flower and 8,938,244
samples / 5,008 leaves for alpha-flower. Those trees pass the native format
validator, but the sample counts exceed the current native learner's 1M
admission limit. This reference-backed research run must not be presented as
qualified native training coverage.

The large-image corpus fit counters are single diagnostic captures, not
repeated throughput results. Capturing copies samples and replays the fitter;
enclosing capture-run learning timers include that work and must not be used
as production timings. Process peak RSS is recorded, but includes image copies,
encoding and validation, and additionally sample copies during capture. It
does not isolate learner scratch.

## Next implementation milestone

Implement a native compact histogram fitter that consumes the captured sample
contract and matches the pinned reference's tree decisions on identical inputs.
Support both predictor sets so representation and fitting work can be qualified
independently of their selection policy. Then reproduce sample preparation from
the same transformed stream images, with explicit managed-storage admission
and failure behavior. Larger reference captures are useful stress/differential
fixtures; they do not authorize silently raising native resource limits.

Before promoting an automatic policy, investigate the splines/alpha16 regressions
with reference-side controls for sampling/growth and downstream coding, and
measure a cheap predictor/transform selection policy on held-out inputs. Keep
that work separate from execution-only optimization of the faithful fitter.
The aggregate rate result supports proceeding to the native implementation;
it does not satisfy the remaining speed, per-image rate, resource or GPU gates.
