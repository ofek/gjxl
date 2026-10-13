# Integrated Modular learning and encoder timing

The native reference-style **Best** workflow now meets the whole-learning CPU
target on the qualified corpus. On the **15 matched learning problems**, it
uses **0.8983× stock e9's learning CPU time**, or **10.17% less**. Scope counts,
sample counts, distinct rows and tree-node counts match, and every timed file
reproduces its qualified complete bytes.

Across **all 16 inputs**, including the patch exception, native Best uses
**0.8902× the reference complete CPU interval** and **0.8879× its wall interval**:
10.98% less CPU time and 11.21% less elapsed time. Its previously qualified
aggregate rate is **+0.3905% versus e9**. These results concern the private
workflow, not a change to the public default.

The [timing ledger](modular-reference-workflow-timing-results.json) retains all
per-image medians, minima, maxima and three measured observations. The
[integration qualification](modular-reference-workflow.md) documents the
rate, exact decisions, allocation behavior and remaining feature limitations.

## Boundaries and controls

The study runs 192 serial, fresh-process encodes: 16 inputs × stock e9/native
Best/native all × one discarded warmup plus three measured repetitions. Order
rotates per input and repetition, so each variant occupies each measured
position once. There were no concurrent agent builds, tests or encodes.
The desktop remains uncontrolled; this is a three-repeat, one-device study,
not a statistical confidence interval or a cross-platform claim.

The native complete interval includes selection, sample preparation, fitting,
coding, admission and destruction of the prepared frame. Stock e9 uses the
reference encoder API wrapper interval. Both exclude file IO, independent
source unpacking for verification, reference decoding, trace IO, and output
IO. No thread pool is supplied. The native caller creates its shared resource
domain before the timed interval. Native preparation and coding subintervals
are also retained for diagnosis.

Learning means **all scope preparation plus fitting**. Native learning also
includes stage planning/admission, input validation, releasing prepared
columns, and compacting completed trees. Reference learning sums every
`LearnTree` interval, including empty global calls. Transform selection is a
separate native stage. This gives a complete learning-stage comparison;
the earlier identical-input fit-only study remains a separate control.

The existing opt-in profiler now records process CPU alongside caller wall
time and subtracts nested scope time. Worker CPU is included in the process
counter when the caller joins workers. No clocks are read without an active
profile session. All native stage sums are checked against the complete
interval, allowing only clock-resolution tolerance.

Every one of the 192 outputs exactly matches the frozen bytes for its input
and variant and independently decodes to the source integers. Sixty matched
Best/reference workload checks cover all 15 inputs × four iterations. The
`gray-patches` exception stays in the full-corpus totals and is excluded only
from the matched-work subset: stock e9 has an auxiliary patch frame and
different main-frame pixels there.

## Results

Times below are **sums of per-image medians**, in seconds. Ratios divide those
sums; they are not averages of per-image ratios.

| Scope / metric | Stock e9 | Native Best | Native all |
|---|---:|---:|---:|
| Matched 15: learning CPU | 34.9581 | 31.4024 | 160.8402 |
| Matched 15: learning CPU / e9 | 1.0000× | **0.8983×** | 4.6009× |
| All 16: learning CPU | 35.0991 | 31.6839 | 162.0870 |
| All 16: complete CPU | 47.1294 | 41.9539 | 172.2664 |
| All 16: complete wall | 47.6501 | 42.3108 | 173.8309 |
| All 16: complete CPU / e9 | 1.0000× | **0.8902×** | 3.6552× |
| All 16: learning share of complete CPU | 74.47% | **75.52%** | 94.09% |
| All 16: complete bytes / e9 | 1.0000× | 1.0039× | 0.9853× |

Best's learning stage remains the main CPU consumer, at 75.52% of its complete
interval. That share is close to e9's 74.47%. Its per-image matched learning
ratios range from 0.8571× to 0.9739×. Best is below e9 on every matched input
in this study.

| Input | e9 learning CPU | Best learning CPU | Best / e9 | All learning CPU |
|---|---:|---:|---:|---:|
| photo-macan | 0.7043 | 0.6457 | 0.917× | 3.1075 |
| photo-ria | 0.6843 | 0.6531 | 0.954× | 2.9967 |
| photo-bliznaca | 0.7082 | 0.6546 | 0.924× | 2.9571 |
| photo-flower | 10.2549 | 9.0078 | 0.878× | 49.6368 |
| gray-macan | 0.2308 | 0.2105 | 0.912× | 1.1161 |
| alpha-ria | 0.9631 | 0.9015 | 0.936× | 4.2861 |
| alpha-flower | 13.6554 | 12.2580 | 0.898× | 68.7253 |
| chessboard | 1.1460 | 0.9823 | 0.857× | 4.2762 |
| splines | 5.2724 | 4.8141 | 0.913× | 20.8521 |
| gray-patches (different work) | 0.1410 | 0.2814 | 1.996× | 1.2468 |
| palette-358 | 0.0103 | 0.0095 | 0.917× | 0.0378 |
| gray16-ramp | 0.0544 | 0.0514 | 0.946× | 0.1027 |
| screen-pattern | 0.0574 | 0.0559 | 0.974× | 0.2377 |
| palette-pattern | 0.0596 | 0.0579 | 0.972× | 0.2336 |
| noise16 | 0.6848 | 0.6400 | 0.935× | 0.4488 |
| alpha16-ramp | 0.4723 | 0.4600 | 0.974× | 1.8257 |

Native all saves another 158,214 bytes, or **1.8517% relative to Best**, but
uses **5.1158× Best's learning CPU** and **4.1061× its complete CPU** on this
corpus. A blanket all-predictor switch would miss the performance objective.
There are useful exceptions: all is both smaller and faster on `noise16`,
and gives large rate gains on graphics. Keep this as an explicit rate/cost
control when evaluating a future selection policy.

The complete-encoder ratios compare different entropy implementations and
the rates above. In particular, the missing native LZ77 path changes both
rate and coding cost on alpha16-ramp. Its complete-encode timing advantage
does not establish equivalent entropy work. Its learning problem does match,
so the learning-stage comparison remains applicable.

## Validation, provenance, and next work

The instrumented build passes the workflow test, including 98 exact files and
5,138 allocation failures. Its files match the integration archive. Existing
profile consumers `modular_policy` and `modular_gpu_token_handoff` also pass.
The 113-artifact supplement `build/reference-workflow-profile-validation`
binds those tests and the unchanged files; its manifest SHA-256 is
`55848fce26597c9a4aae953a23600c314b61b91f462bd5c1320587b4f426a11c`.
This profiling change follows the
19-test Release/ASan integration qualification; the timing executables are
ordinary Release builds, not sanitizer measurements.

`build/reference-workflow-timing-study/manifest.json` binds 1,183 artifacts:
source, executables, build settings/libraries, complete outputs and logs.
Manifest SHA-256:
`1ddd21ae706b3d667db913d63c18aa02ca35f1b5dc243928f278d818249381bc`.
The source snapshot is authoritative for the instrumented pre-commit build;
the base integration is commit `1d8a698`. The device/toolchain are the same
Apple M4 Pro / macOS / Apple Clang setup used for the preceding qualification.

Recompute without encoding:

```sh
python3 tools/modular_reference_learning/analyze_reference_workflow_timing.py \
  build/reference-workflow-timing-study \
  --out /tmp/gjxl-reference-workflow-timing-rechecked.json
```

The current-corpus whole-learning aim of ≤2× stock e9 is satisfied by Best.
Next apply the exhaustive-new-learner and held-out/per-image gates before
default promotion. The patch-frame and high-depth entropy outliers remain
open. Further CPU/GPU optimization should target the now-measured workload
after those policy decisions; it need not preserve the old repeated-fitting
implementation.
