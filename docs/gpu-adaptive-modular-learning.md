# Adaptive Modular GPU learner qualification

The complete-search learner route preserves all CPU files, winning policies and semantic training work across two three-route repetitions on each original full-resolution input. Pinned libjxl independently decodes every output sample exactly.

Frozen implementation: `34586ea3b3f6fd3d8c45922b3be5bd5c45c3244a`. Artifacts: `/Users/yunhocho/GitHub/gjxl-modular-adaptive-gpu/build/adaptive-learning-qualified-20261011`; 89 verified files. Pinned libjxl: `e8ff09762481785938d8e4e01333ed3917571161`.

Machine-readable results: [gpu-adaptive-modular-learning-results.json](gpu-adaptive-modular-learning-results.json).

## Complete-call timing

All full-resolution calls use one CPU participant, the original learner with 262,144 samples and 256 leaves, checkpoints, and the full retained candidate set. The controls are CPU-only, GPU tokens plus exact ANS costs, and those GPU operations plus exact learner statistics. CPU property sorting, weighted sample gathering, tree decisions, model search, emission and every candidate remain inside the timer. Uploads, waits, cache creation and publication are included. Backend setup, file I/O and independent decode are outside the encoder-call timer. No sampling profiler runs.

Image and route order rotate; the second repetition reverses each image's route order. Desktop process activity and between-triplet thermal/power snapshots are retained. Values below are medians and observed ranges from two repetitions, not confidence intervals.

| Image | CPU seconds [range] | GPU tokens + ANS seconds [range] | Plus GPU learning seconds [range] |
|---|---:|---:|---:|
| 1029 | 207.330 [206.461–208.200] | 108.586 [107.598–109.574] | 102.417 [102.051–102.783] |
| 1039 | 194.457 [194.214–194.701] | 104.464 [104.081–104.847] | 98.658 [98.351–98.965] |
| 1049 | 191.751 [191.124–192.379] | 103.307 [103.243–103.370] | 97.821 [97.698–97.945] |

The ratio of summed per-image medians is **1.99×** for CPU versus the learner hybrid, and **1.058×** for adding GPU learning to the token/ANS hybrid (**5.52% less elapsed time**). Actual per-triplet ratios and all per-run values are retained in JSON. These are local observations on three images; they do not establish a speedup over libjxl or broad device/corpus performance.

## Attribution

Each cell below sums three per-image stage medians. Nested training rows must not be added to the top-level training total.

| Caller phase | CPU seconds | GPU tokens + ANS seconds | Plus GPU learning seconds |
|---|---:|---:|---:|
| input | 3.555 | 1.606 | 1.733 |
| transforms | 16.095 | 8.739 | 8.735 |
| training | 112.134 | 112.710 | 95.545 |
| tokens | 232.475 | 47.021 | 46.367 |
| model | 185.661 | 102.410 | 102.693 |
| emission | 43.287 | 43.636 | 43.594 |
| assembly | 0.261 | 0.219 | 0.213 |

| Nested training phase | CPU seconds | GPU tokens + ANS seconds | Plus GPU learning seconds |
|---|---:|---:|---:|
| rct_ranking | 0.005 | 0.003 | 0.004 |
| palette_probe | 0.000 | 0.000 | 0.000 |
| sample_gathering | 26.453 | 26.430 | 26.551 |
| simple_learning | 0.031 | 0.031 | 0.031 |
| entropy_preparation | 0.131 | 0.129 | 0.127 |
| split_histograms | 2.198 | 2.229 | 2.380 |
| property_sorting | 43.068 | 43.355 | 42.965 |
| split_scanning | 39.935 | 40.215 | 23.169 |
| tree_control | 0.313 | 0.317 | 0.317 |
| quantized_learning | 0.000 | 0.000 | 0.000 |

Within the learner hybrid's model stage, exact ANS costing accounts for 30.727 summed seconds. That interval includes the GPU adapter, upload and wait; the model stage is mixed CPU/GPU work, not an isolated remaining CPU cost.

GPU statistics evaluate the same CPU-selected thresholds and exact Q12 integer costs. Leaves below 16,384 samples use the original scalar scan. Greedy growth, strict ties, parent histograms, property sorting and checkpoints stay on the CPU. Preparation/evaluation counters include host validation, upload and completion; they are nested within split scanning. The fallback counter explicitly includes small leaves and does not mean the richer learning policy was disabled. The measured change also includes CPU threshold preflight, which skips histogram work for eligible properties with no valid threshold; it is not a kernel-only speedup.

## Rate and storage

| Image | GJXL bytes | Default libjxl e7 bytes | Default libjxl e9 bytes |
|---|---:|---:|---:|
| 1029 | 10,821,744 | 10,602,863 | 10,558,213 |
| 1039 | 7,230,909 | 7,303,959 | 7,214,879 |
| 1049 | 7,336,102 | 7,181,942 | 7,072,878 |

Summed GJXL size is 25,388,755 bytes: 1.20% above default libjxl e7 and 2.18% above e9. These are verified saved reference files. No fresh libjxl timings enter this comparison. GPU execution preserves the richer CPU search's compression exactly.

| Image | CPU / ANS hybrid / learner hybrid peak MiB | Respective admission MiB | Learner backend setup seconds |
|---|---:|---:|---:|
| 1029 | 689.3 / 1307.3 / 1307.3 | 5021.1 / 6400.4 / 6418.4 | 0.028 |
| 1039 | 675.7 / 1293.7 / 1293.7 | 5021.1 / 6400.4 / 6418.4 | 0.034 |
| 1049 | 674.9 / 1292.9 / 1292.9 | 5021.1 / 6400.4 / 6418.4 | 0.035 |

Backing and admission are managed-memory measurements, not process RSS or driver memory. Every call fits its conservative bound and releases its resource domain. Learner preparation is scoped to one learning call and reuses immutable codes/count costs across eligible properties and leaves.

## Small inputs and limits

| Smoke image | Bytes | Complete seconds | GPU learner submissions | CPU small-leaf properties |
|---|---:|---:|---:|---:|
| photo-macan | 260,654 | 23.282 | 3900 | 73312 |
| palette-pattern | 438 | 16.247 | 2884 | 36832 |
| alpha16-ramp | 19,866 | 10.674 | 2419 | 20144 |

These are single four-CPU-participant smoke runs, not repeated small-image performance measurements. All use GPU tokens and the explicit CPU ANS crossover below 64 nonempty sections. Tree checkpoints and all candidate decisions remain enabled. Other devices, intermediate shapes and multi-CPU full-image performance remain unqualified. Public encoding defaults are unchanged; this is a private Metal qualification route, not public API rollout or hosted CI evidence.

See the [learner operation contract](gpu-modular-learner-statistics.md) for exact arithmetic, storage, fault checks and crossover evidence.

Reproduce from saved artifacts:

```sh
python3 tools/modular_training_study/analyze_adaptive_learning.py \
  --capture build/adaptive-learning-qualified-20261011 \
  --baseline build/adaptive-baseline-20261011 \
  --json docs/gpu-adaptive-modular-learning-results.json \
  --markdown docs/gpu-adaptive-modular-learning.md
```
