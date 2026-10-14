# Adaptive Modular exact-ANS GPU qualification

The complete hybrid search preserves every CPU file and winning policy across three repeated pairs on each of the original full-resolution inputs. All full-resolution candidates use Metal tokens and exact ANS cost evaluation, with no fallback. Pinned libjxl decodes every output sample exactly.

Frozen implementation: `2009da79029a45a03676d784780a405132af7b2a`. Artifacts: `/Users/yunhocho/GitHub/gjxl-modular-adaptive-gpu/build/adaptive-ans-qualified-20261011`; 100 verified files.

Pinned libjxl: `e8ff09762481785938d8e4e01333ed3917571161`. Machine-readable results: [gpu-adaptive-modular-ans-results.json](gpu-adaptive-modular-ans-results.json).

## Timing boundary

Both routes use one CPU participant, the original 262144-sample/256-leaf learner, tree checkpoints and all retained baseline candidates. Each repetition rotates image order and alternates CPU/hybrid pair order. Encoder-call timing includes transforms, learning, feature-cache preparation, every candidate, model upload, GPU waits, CPU emission and publication. Backend initialization is outside that timer and reported separately; file reading, writing and independent decoding are also outside it. There is no sampling profiler. Background desktop activity and between-pair power/thermal snapshots are retained.

| Image | CPU seconds, median [range] | Hybrid seconds, median [range] | Paired ratio, median [range] |
|---|---:|---:|---:|
| 1029 | 207.468 [205.550–207.501] | 108.960 [108.169–109.039] | 1.903 [1.900–1.904] |
| 1039 | 193.676 [192.838–193.830] | 103.520 [103.483–103.531] | 1.871 [1.863–1.872] |
| 1049 | 191.382 [191.286–192.200] | 102.910 [102.776–103.740] | 1.859 [1.853–1.862] |

The ratio of summed per-image median CPU and hybrid times is **1.88×** (592.527 / 315.390 seconds). Paired ratios above are computed from each actual adjacent pair. This is a local three-image qualification on the captured Apple M4 Pro desktop, not an uncontended benchmark or a comparison of equal-budget GJXL/libjxl throughput.

## Attribution

The following values sum the three per-image stage medians. Each stage median is computed independently, so the table need not sum exactly to the complete-call median.

| Caller stage | CPU seconds | Hybrid seconds |
|---|---:|---:|
| input | 3.609 | 1.730 |
| transforms | 15.975 | 8.767 |
| training | 112.121 | 112.413 |
| tokens | 232.135 | 46.349 |
| model | 184.958 | 102.174 |
| emission | 43.251 | 43.646 |
| assembly | 0.248 | 0.205 |

Within `model`, exact ANS cost changes from 111.717 to 30.673 seconds. The hybrid interval includes adapter preparation, model upload, dispatch, wait and cost conversion; it is not kernel-only time. Do not add this nested interval to the table. Full per-run training, entropy and GPU counters are retained in the companion JSON.

Tree learning, model construction/normalization, weighted feature preparation and bitstream emission remain CPU work. The GPU preserves the selected algorithm and each section's ordered ANS recurrence. It prepares tokens within a SIMD group and processes independent section/model recurrences concurrently.

## Compression and resources

| Image | GJXL bytes | Saved libjxl e7 bytes | Saved libjxl e9 bytes | CPU / hybrid peak MiB | CPU / hybrid admission MiB | Hybrid setup seconds, median |
|---|---:|---:|---:|---:|---:|---:|
| 1029 | 10,821,744 | 10,602,863 | 10,558,213 | 689.3 / 1307.3 | 5021.1 / 6400.4 | 0.041 |
| 1039 | 7,230,909 | 7,303,959 | 7,214,879 | 675.7 / 1293.7 | 5021.1 / 6400.4 | 0.037 |
| 1049 | 7,336,102 | 7,181,942 | 7,072,878 | 674.9 / 1292.9 | 5021.1 / 6400.4 | 0.031 |

Summed output is 25,388,755 bytes, 1.20% above saved default libjxl e7 and 2.18% above e9. GPU execution preserves the richer CPU learner's rate exactly; it does not disable learning or reduce search. These reference sizes are verified saved artifacts, not fresh libjxl timing runs. An end-to-end speedup over libjxl is not established.

Peak backing is the maximum observed across repetitions; admission is the conservative plan. Both measure managed backing, not process RSS or driver memory. Every call fits admission and releases its resource domain. Feature caching retains transformed planes and weighted maps; ANS costing adds descriptor/model/result storage without another token arena.

## Small work and limitations

| Smoke image | Bytes | Complete seconds | Exact ANS seconds | CPU cost groups |
|---|---:|---:|---:|---:|
| photo-macan | 260,654 | 29.068 | 0.785 | 140 |
| palette-pattern | 438 | 23.108 | 0.402 | 136 |
| alpha16-ramp | 19,866 | 14.867 | 0.440 | 168 |

The smoke cases use GPU token generation and explicitly counted CPU ANS costing below 64 nonempty sections. Each matches the saved CPU file and policy and passes independent decoding. These four-CPU-participant smoke captures are single runs, not repeated small-image performance qualification. The crossover is conservative; intermediate sizes, different devices and multi-CPU full-resolution throughput remain unqualified. Public encoder defaults are unchanged.

The earlier one-lane capture was stopped after small-image regressions and is retained separately. Diagnostic SIMD probes guided the final mapping; none are mixed into these repeated statistics. See [the operation contract](gpu-ans-token-cost.md) for exact arithmetic, storage, failure behavior and test coverage.

Reproduce the report from the saved artifacts:

```sh
python3 tools/modular_training_study/analyze_adaptive_ans.py \
  --capture build/adaptive-ans-qualified-20261011 \
  --baseline build/adaptive-baseline-20261011 \
  --json docs/gpu-adaptive-modular-ans-results.json \
  --markdown docs/gpu-adaptive-modular-ans.md
```
