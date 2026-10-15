# Parallel Modular integration qualification

2026-10-15. Quick qualification of the reconciliation of `perf/parallel-tree-fitting` into `feat/modular`. The incoming bounded feature/token pipeline replaces the competing whole-image implementation; public explicit Metal selection is preserved. Automatic remains CPU. No codec selection or fitting policy was intentionally changed.

The same Release binary was measured with CPU participation limits of 1 and 14 on an Apple M4 Pro. These controls measure the combined parallel selection, preparation, fitting, tokenization and entropy pipeline; they do not isolate tree fitting or provide a matched pre-merge speedup.

## Validation

- 25 CTest runs passed: 19 in the Metal build and six in the CPU-only build.
- 102 complete encodes matched their CPU control byte for byte and independently decoded to exact samples; 56 were timed measurement rounds, with cold calls and warmups retained separately.
- 36 feature-map trials compared all 15,578,616 entries against the serial predictor.
- All managed domains released; recorded participants stayed within the requested CPU limits. Feature reuse, sample population, batch counts and GPU submissions passed the analyzer checks.
- Saved outputs match between the 1- and 14-participant captures, the forced-producer smoke capture, and the available prior qualification controls.
- All three captures reproduce with the current analyzer, use the same frozen executable and source patch, and match all 541 recorded current source hashes.

## End-to-end timing

Median seconds, excluding cold calls and warmups. Each public variant has two measured rounds per input. Encoder timing includes admission, preparation, selection, learning, tokenization, entropy, cleanup and output publication; it excludes input/output file I/O, domain construction and independent verification.

| Image | CPU, 1 | CPU, 14 | Metal, 1 | Metal, 14 |
|---|---:|---:|---:|---:|
| photo-macan | 0.6010 | 0.4759 | 0.5669 | 0.4691 |
| alpha-ria | 0.8316 | 0.7087 | 0.7886 | 0.6649 |
| palette-358 | 0.0120 | 0.0105 | 0.0122 | 0.0106 |
| alpha16-ramp | 0.4014 | 0.3446 | 0.3896 | 0.3435 |
| photo-large-1029 | 38.3011 | 5.3364 | 36.0805 | 5.8875 |
| Sum of image medians | 40.1471 | 6.8761 | 37.8379 | 7.3757 |

The 12 MP photograph improves 7.18x for CPU and 6.13x for Metal when increasing the CPU limit. The sums improve 5.84x and 5.13x respectively; these sums are not an average per-image speedup.

Metal uses 5.75% less aggregate time at limit 1, but 7.27% more at limit 14. On the large image at limit 14, Metal is 10.33% slower.

| Large-image interval, limit 14 | CPU (s) | Metal (s) |
|---|---:|---:|
| Transforms/selection | 0.4408 | 0.4335 |
| Training including feature preparation | 3.5134 | 4.1764 |
| Token generation | 0.2008 | 0.0903 |
| Parallel stream encoding | 1.1494 | 1.1519 |

Token generation remains faster through Metal, but the longer training interval outweighs that saving. These timings locate the difference without establishing its internal cause. Feature preparation is nested in training; GPU token work is nested in token generation. Do not add nested counters to stage totals.

The maximum measured managed backing on the large input is 808.27 MiB for CPU and 737.99 MiB for Metal. This excludes borrowed input, published output, immutable backend, driver and oracle storage; it is not total process memory.

## Weighted-feature preparation and forced-producer smoke check

The feature operation includes allocation, copying, predictor work, dispatch, waiting and publication. Preparation uses up to four CPU participants. Selected-channel generation, planning and independent comparisons are outside this timer. The microbenchmark retains completed feature batches through comparison, unlike the one-live-local-batch workflow.

| Feature case | CPU (ms) | GPU (ms) | Automatic producer (ms) |
|---|---:|---:|---:|
| photo-macan | 3.785 | 2.590 | 3.653 |
| palette-358 | 0.158 | 1.727 | 0.163 |
| alpha16-ramp | 2.528 | 2.004 | 2.567 |

The 14-participant end-to-end smoke check on `photo-macan` passed 22 encodes with four counterbalanced measured rounds. Median seconds: public CPU 0.4829, public Metal with automatic feature routing 0.4690, forced CPU features 0.4643, and forced GPU features 0.4703. Both forced feature routes use Metal token generation and mirror the public wrapper. This one-image smoke check establishes wrapper correctness, not a new producer cutoff.

## Scope and reproduction

The five cases cover RGB photographs, alpha, palette and 16-bit content. The 14-participant block ran before the 1-participant block; within each block, variant positions rotate. All recorded power snapshots show AC power, but the desktop and background services remained active. Two measured public repetitions and this small input set support an integration check, not a general performance crossover or corpus-wide claim. The earlier roughly 6% serial Metal saving should not be projected onto the fully parallel CPU baseline.

The [results ledger](modular-parallel-integration-qualification-results.json) retains per-run distributions, input/output hashes, stage and resource diagnostics, manifest hashes, test-log hashes and all three analyzer results. Raw inputs, frozen sources, executable, patches, per-call JSONL and environment snapshots remain under `build/parallel-modular-integration-20261015/{cpu-1,cpu-14,producer-smoke}`.

Destination parent: `4a92bf36af6fbceca796ee02729362f3559af071`. Incoming parent: `d6af521d28244f22ae17a407dcc5473a3d180680`.

Executable SHA-256: `7f93f959ec425cfaf22c1b929873f30a38ea72a942864770c38807d9c7dd33e1`. Qualified source patch SHA-256: `7e387e1db9ff37f00ad1c46af9c816e8540285384dc5f2b05a7b6fa74bec913d`. Only this report, its ledger and the public documentation link were added after capture.

Use the runner and analyzer documented in [the qualification tools](../tools/modular_reference_learning/README.md#public-cpumetal-qualification). Public captures use `--cpu-participants 1` or `14`, `--public-only --encode-repeats 2`; the 14-participant capture also uses `--feature-repeats 3`. The smoke capture uses the single-photo input manifest, `--cpu-participants 14 --encode-repeats 4`, and omits `--public-only`.
