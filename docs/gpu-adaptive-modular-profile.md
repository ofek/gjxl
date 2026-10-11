# Full-search adaptive Modular baseline

The richer GJXL policy closes most of the compressed-size gap on the three
full-resolution photographs: total size is **1.20% above default libjxl e7** and
**2.18% above default libjxl e9**. Reaching that rate currently takes 68–70 complete
candidate encodes and about 199–225 seconds per image on the CPU. These captures
establish a rate control and identify GPU work; they do not establish a speedup.

## Capture and validation

- GJXL revision: `c80bb2bc50d8776dbc1169e33cb622bfbd500615`, integrating
  `cbc2d8e` and tree-learning revision `5faf68d` through `bf1fa12`.
- Reference: pinned libjxl `e8ff09762481785938d8e4e01333ed3917571161`.
- Machine: Apple M4 Pro, 48 GiB, macOS 27.0.1, AC power. Other desktop activity
  was recorded. Release GJXL uses `-O3 -g -DNDEBUG`; Metal is built but unused.
- Inputs: the original **4000×3000 RGB8** PAM samples for 1029, 1039, and 1049.
  No resizing. Input hashes are in the results and capture manifest.
- GJXL: complete CPU ANS search, original learner, 262144 samples/256 leaves,
  intermediate checkpoints enabled, retained baseline candidates, one CPU
  participant. All input preparation, training, candidates, and publication are
  inside the encoder-call timer. Input-file reading, decoder validation, and
  output-file writing are outside it.
- libjxl: raw lossless Modular, default e7/e9 policy, no background workers,
  three repetitions. No predictor, learning, palette, RCT, or responsive-policy
  restrictions. Commands use `-d 0 -m 1 --container=0 --keep_invisible=1
  --num_reps=3 --num_threads=0` plus the effort.
- All nine final outputs decoded to exactly the original pixels. All six new
  libjxl files also match the earlier pinned profiles byte for byte.
- The 1029 GJXL call additionally has a 1 kHz Samply capture with retained dSYM
  and symbol sidecar. Other GJXL calls have explicit phase instrumentation only.

Artifact root: `build/adaptive-baseline-20261011/`. It retains a frozen GJXL
binary, symbols, CMake flags, exact commands, per-command activity logs, profiles,
compressed files, decoder checks, provenance, and hashes. The machine was not
isolated and these are diagnostic captures, not paired throughput qualification.

## Compressed size

Percentages are `(GJXL / reference - 1) * 100`; negative means smaller.

| Image | GJXL bytes | libjxl e7 bytes | vs e7 | libjxl e9 bytes | vs e9 |
|---|---:|---:|---:|---:|---:|
| 1029 | 10,821,744 | 10,602,863 | +2.06% | 10,558,213 | +2.50% |
| 1039 | 7,230,909 | 7,303,959 | −1.00% | 7,214,879 | +0.22% |
| 1049 | 7,336,102 | 7,181,942 | +2.15% | 7,072,878 | +3.72% |
| **Total** | **25,388,755** | **25,088,764** | **+1.20%** | **24,845,970** | **+2.18%** |

The winners are RCT 12 with 256 leaves for 1029, RCT 12 with 128 leaves for 1039,
and RCT 13 with 128 leaves for 1049. All use weighted leaves and property 15;
none uses additional transforms. They exceed the existing GPU token producer's
31-node limit. The smaller palette/Squeeze study remains necessary for transformed
layout coverage; three photographs are not a general-corpus rate claim.

## Complete-call diagnostic times

| Image | GJXL full search | Candidates | libjxl e7 | libjxl e9 |
|---|---:|---:|---:|---:|
| 1029 | 224.570 s | 70 | 8.52 s | 44.61 s |
| 1039 | 202.673 s | 69 | 6.09 s | 30.15 s |
| 1049 | 199.354 s | 68 | 5.92 s | 27.15 s |

The libjxl times are derived from its reported, rounded geometric-mean MP/s across
three repetitions. GJXL records one complete call, with the extra Samply observer
on 1029. These boundaries and repetitions differ. Even with those limitations,
there is no evidence of a GJXL full-search speed advantage here.

| GJXL stage | 1029 | 1039 | 1049 | Share of summed complete-call time |
|---|---:|---:|---:|---:|
| Input + transforms | 6.734 s | 6.551 s | 6.456 s | 3.2% |
| Training | 45.725 s | 40.156 s | 40.362 s | 20.1% |
| Prediction + tokens | 86.953 s | 78.776 s | 77.625 s | 38.8% |
| Modeling | 67.877 s | 63.120 s | 61.104 s | 30.7% |
| Emission | 17.159 s | 13.988 s | 13.730 s | 7.2% |
| Assembly | 0.100 s | 0.072 s | 0.061 s | <0.1% |

Top-level stages are disjoint; small uninstrumented control time remains. Training
details are nested inside training. Across the three images, property sorting
takes 47.45 seconds, split scanning/scoring 47.76 seconds, sample gathering 27.76
seconds, and parent histograms 2.76 seconds. Greedy tree control is only 0.33 seconds.

The full-resolution result differs from the 500×500 smoke capture, where learning
dominated. Training uses a bounded sample set; every complete candidate still
predicts, models, and writes the full image. On 1029, 18 learning passes scan
648 million transformed samples, while 70 encodes tokenize about 2.52 billion.

The previous CPU entropy bottleneck also remains: 1029 spends 42.49 seconds in
exact ANS token-cost simulation, nested within its 67.88 seconds of modeling.
Neither additional training nor GPU token generation alone removes that cost.

## Samply cross-check

The 1029 profile attributes only the `EncodeModularImage` subtree, excluding file
I/O and decoder verification. Approximately 99.94% of sampled encoder leaf CPU
time resolves to symbols. Sampled CPU shares are 38.68% token generation, 22.90%
ANS modeling, 7.29% prefix modeling, 7.71% emission/assembly, and about 20.45%
training. This agrees with the explicit caller-wall attribution.

The parser checks both current- and previous-sample CPU-delta attribution. The
largest category movement is about 0.22 percentage points between sorting and
split evaluation. Inclusive symbol rows overlap and must not be summed. Some
optimized sorting leaf names mention `Quantize`; the enclosing learner stack,
recorded options, and zero quantized-learner timer identify the original learner.

## Implementation priority

1. Add token evaluation for large learned trees, arbitrary transformed layouts,
   weighted predictions, and property 15. Begin with exact CPU feature generation
   and retain its cost in the complete-call measurement.
2. Reuse prepared planes and weighted features across candidate trees/checkpoints
   for the same layout and settings, with a bounded per-call cache.
3. Accelerate learner sorting/histograms and split evaluation after that path is
   qualified. Preserve integer costs, tie order, chosen trees, and complete bytes.
4. Reassess remaining modeling and repeated candidate work from paired full-call
   profiles. No claim of matching libjxl throughput follows from kernel timing.

The operation/ownership/admission boundary is specified in
[`gpu-adaptive-modular-token-contract.md`](gpu-adaptive-modular-token-contract.md).

Machine-readable results: [`gpu-adaptive-modular-profile-results.json`](gpu-adaptive-modular-profile-results.json).
Regenerate them with:

```sh
python3 tools/modular_training_study/analyze_adaptive_profile.py \
  build/adaptive-baseline-20261011 docs/gpu-adaptive-modular-profile-results.json
```
