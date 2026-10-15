# Histogram learner and GPU feature/token qualification

2026-10-14. The integrated private hybrid route preserves the improved native
learner's complete files and reduces encoder-call time by **5.6–6.7%** on the
three qualified full-resolution photographs. Both routes use the same one-fit
histogram policy. Pinned libjxl independently decodes each output after timing.

The [implementation and ownership contract](gpu-histogram-learner-integration.md)
and [raw results](gpu-histogram-learner-integration-results.json) accompany this
report. Public automatic GPU routing remains unchanged; the qualified route is
an explicit private `ReferenceWorkflowOptions::gpu` capability.

## Complete-call results

Original 4000×3000 RGB8 PAMs, one warmup followed by three measured calls per
route/image. Values below are medians in seconds. Every call produced identical
bytes across routes/repetitions and passed independent exact decoding.

| Image | CPU | Automatic hybrid | Runtime reduction | Encoded bytes, both routes |
| --- | ---: | ---: | ---: | ---: |
| 1029 | 55.500 | 52.395 | 5.60% | 10,507,251 |
| 1039 | 40.682 | 38.278 | 5.91% | 7,138,537 |
| 1049 | 37.115 | 34.645 | 6.66% | 7,015,756 |

CPU measured ranges were 55.242–56.475, 40.591–40.706 and 36.929–37.267
seconds respectively. Hybrid ranges were 52.286–52.496, 38.179–38.367 and
34.581–34.884 seconds. The 24 complete calls include six warmups. This is
matched local evidence for these inputs, revision, settings and device.

## Where the time went

Nested wall intervals, in seconds; paired entries show CPU / automatic hybrid.
The token column is the enclosing token stage, including CPU tree-header work.

| Image | Separate feature preparation, hybrid | Native sample preparation | Tree fitting | Token stage |
| --- | ---: | ---: | ---: | ---: |
| 1029 | 0.110 | 5.409 / 5.056 | 38.069 / 38.034 | 3.038 / 0.187 |
| 1039 | 0.152 | 4.627 / 4.260 | 26.780 / 26.728 | 2.364 / 0.178 |
| 1049 | 0.111 | 4.392 / 4.029 | 24.045 / 24.033 | 2.325 / 0.144 |

Most of the complete-call gain comes from the token stage, which is about
13–16× faster. Shared maps eliminate scalar weighted-feature recomputation
inside learning and serve GPU tokens as well. The CPU control computes weighted
features inside its sample/token passes, so it has no separate preparation
interval. A zero separate-feature counter does not mean it performs no WP work.

Tree fitting still accounts for **69–73% of hybrid runtime**. Selection,
quantization, sampling, deduplication, fitting, tree-header generation and
entropy coding stay on the CPU. Further improvements to the feature producer
alone have little complete-call headroom here: its entire interval is at most
0.4% of runtime.

Feature preparation, sample preparation and fitting are nested within the
training profile stage. GPU token detail is nested within the token stage.
Do not add these details to their enclosing stages. Independently computed
medians need not sum exactly; the raw per-call profiles preserve attribution.

## Actual routing and storage

The automatic gate uses transformed channel samples and selected stream shapes.
Whole streams stay together, so selected palette metadata can split batches or
make a stream ineligible for wavefront preparation. Eligibility is decided
before execution; execution errors are terminal.

| Image | GPU / CPU feature batches | GPU feature samples / all feature samples | Metal token batches |
| --- | ---: | ---: | ---: |
| 1029 | 4 / 1 | 33,079,296 / 36,000,000 | 5 |
| 1039 | 3 / 17 | 20,517,120 / 31,344,224 | 20 |
| 1049 | 3 / 5 | 24,973,280 / 29,871,324 | 8 |

Every prepared sample is reused by learning and tokenization; all hybrid
learning runs report zero scalar WP computations. Local feature lifetime is
bounded to one batch, capped at 8,388,608 samples. The largest observed live
batch had 8,341,472 samples and a 159.1 MiB feature owner, including maps, packed
planes and descriptors. Token owners survive after feature owners are released.
Explicit buffered/global mode retains its complete learning scope through the
one fit and can hold multiple batches; that mode has separate admission.

| Image | Peak managed backing, CPU / hybrid (MiB) | Peak admission commitment, CPU / hybrid (MiB) |
| --- | ---: | ---: |
| 1029 | 785.4 / 717.9 | 1608.4 / 1814.4 |
| 1039 | 697.2 / 638.8 | 1415.6 / 1594.9 |
| 1049 | 670.9 / 615.7 | 1357.8 / 1528.7 |

Split GPU tokens use less backing than the CPU token representation. They are
retained while later batches fit, so stagewise admission commitments increase.
Admission includes conservative reservations; it is not resident-set size.
Managed figures exclude caller input, reference decode buffers, stack, allocator
bookkeeping and backend initialization. Every completed encode releases its
managed resources after output release.

## Feature-producer crossover

Two exact center crops of 1029 provide controls on either side of the existing
4,194,304-sample gate. There is no resampling or color conversion. Both controls
keep Metal token generation and the native learner unchanged; only the feature
producer changes. One warmup and three measured calls per route/crop produced
32 additional exact files.

| Crop | Transformed samples | CPU feature producer, 4 participants (ms) | GPU feature producer (ms) | Automatic choice |
| --- | ---: | ---: | ---: | --- |
| 1024×1024 | 3,145,728 | 21.861 | 13.538 | CPU |
| 1280×1280 | 4,915,200 | 34.253 | 16.930 | GPU |

These intervals include planning/admission, allocation, copying, preparation,
dispatch/wait where applicable, and publication. GPU preparation is 1.61× and
2.02× faster in these two controls. The complete-call medians are:

| Crop | Ordinary CPU (s) | CPU features + Metal tokens (s) | GPU features + Metal tokens (s) | Automatic hybrid (s) |
| --- | ---: | ---: | ---: | ---: |
| 1024×1024 | 5.480 | 5.203 | 5.188 | 5.196 |
| 1280×1280 | 7.938 | 7.570 | 7.517 | 7.543 |

Keep the existing conservative threshold and minimum average slice area of
16,384. The accepted larger control favors GPU preparation. The smaller
rejected control also favors it, but two crops from one image do not establish
a replacement crossover for tiny/narrow/palette-heavy workloads. Total-runtime
differences between feature producers are small relative to fitting and local
timing variation. These measurements do not justify promoting the public
automatic route.

## Correctness and failure coverage

- The integration suite produced 56 CPU/hybrid identical prefix/direct-ANS
  files and verified 23,266,682 decoded samples. Coverage includes gray/RGB/RGBA,
  8/16 bits, both byte orders, invisible alpha, eight/nine-group tree scope,
  all predictors, mixed headers, multiple batches and 600-color palette
  metadata wider than 256.
- Compared exact feature entries, prepared samples, learned trees, tokens,
  selected transforms/WP headers and canonical stream identities. The stream
  batch suite includes 999,744 feature/token entries and legacy ABI regressions.
- Checked completed-token borrowing without copying or double charging,
  malformed borrows before allocation, frame moves, detached packed input,
  memory admission and resource release. The complete hybrid test exercised
  159 allocation/submission/completion failures without changing prior outputs.
- Selected CPU and Metal checks passed. AddressSanitizer/UndefinedBehaviorSanitizer
  CPU runs of shared consumers, stream batches and the reference workflow passed;
  leak detection was disabled and UBSan halted on errors. These instrument host
  code, not GPU shaders. The exhaustive legacy candidate allocation sweep was
  excluded.
- All 56 performance-qualification calls, including warmups, independently
  decoded 992,974,848 source channel samples and preserved exact file hashes
  and learning-work counts. Development pilots are excluded from the evidence.

## Reproduction and limits

Encoder implementation: `736a0c921bae1e273092c3a3f1878d22db9473c9` on
`feat/parallel-weighted-feature-prep`, incorporating the improved learner at
`81467ca`. Both studies use the same frozen executable SHA-256:
`d50796469fcbbfb2146fd1f8f15d403312fb4ec8a23561a98794a0f82b2d93c9`.
The later `6eb61ba` commit only improves runner dependency inventory.
The pinned oracle source is libjxl `e8ff09762481785938d8e4e01333ed3917571161`.

Apple M4 Pro, 48 GiB, macOS 27.0.1, Apple Clang 21, Release
`-O3 -g -DNDEBUG`, Metal enabled and CUDA disabled. Ordinary desktop activity
continued. Both studies ran on battery (70%→66% for the full-resolution run,
66% at both endpoints for the crops). These results should be kept separate
from earlier AC-powered cohorts.
A post-run power-settings snapshot reports battery `powermode 1` and no
recorded thermal/performance warnings. Those settings were not sampled during
individual calls; the timestamped supplemental snapshot is retained in the JSON.

The complete native API interval includes selection, feature/sample preparation,
fitting, tokenization, entropy, admission, output copy and prepared-frame teardown.
Input/output I/O, backend initialization, reference source unpacking and
independent decoding are outside the timer. Policy is automatic/local tree
buffering, Best predictors and estimated direct ANS with 64 clusters. Both
routes have the same four-participant CPU ceiling; the ordinary CPU workflow
remains serial. Calls run in fresh serial processes with rotating route order.

[The driver and runner instructions](../tools/modular_reference_learning/README.md#cpu-hybrid-histogram-qualification)
describe reproduction. Raw local archives are under
`build/histogram-gpu-integration/qualification-20261014-{cpu-auto,crossover}`;
each includes the frozen executable, input/output hashes, build configuration,
power snapshots, raw calls and a completed manifest. The JSON report preserves
those manifests and every measurement. Crop origins and test-log hashes are
included as well.

The first runner recorded `@rpath` decoder install names without resolving
their hashes. A supplemental inventory captured all five decoder/Brotli files
during the full-resolution run, verified modification/change times predating
the run, and rechecked unchanged hashes and times at completion. The crossover
runner resolves and checks these dependencies automatically. Encoder arithmetic
and Metal shaders are contained in the identical frozen executable.

The next substantial performance target is CPU tree fitting. Public API/backend
routing and a broader corpus/device qualification are separate follow-up work.
