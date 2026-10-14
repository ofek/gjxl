# Estimated width selection for native direct ANS

The private native direct-ANS optimizer can now select its alphabet width
from histogram/configuration estimates. On the 11 captured e9 cases,
sample-model construction uses **0.9754 s summed median CPU**, versus
**2.7167 s** for exact direct rate/64: **64.1% less CPU, or 2.79× faster**.
The five automatic-mode images cost **27 additional bytes in total**; the
buffered total is one byte smaller. The 16-bit ramp remains **47,547 bytes**.

This qualifies the optimizer change and its managed-storage contract.
It does **not** qualify whole-encoder timing or connect a new policy to the
prepared-stream/public encoder. Trees, transforms, WP parameters and stream
scopes are still captured reference decisions. The
[full ledger](modular-direct-ans-estimated-results.json) binds rate, counters,
independent reconstruction, allocation tests and timing evidence.

The subsequent [prepared-stream integration](modular-prepared-entropy.md) now
qualifies an explicit private routing option and its stream-level resource
contract. It reproduces all 55 files here; public-default and whole-encoder
performance qualification remain separate.

## Implementation and contract

`OptimizeDirectAnsEntropyCode` now accepts the existing `AnsSelectionMode`
as an optional final argument. The default remains exact. Estimated selection
is explicitly supported for `kRateOptimized`; balanced/high-density modes
retain their existing policies and reject that override before allocation.

The change reuses the shared optimizer's existing estimated-width branch:

1. Build the same direct partition, 28-configuration candidates and precise
   normalized histograms as the exact mode.
2. For each viable width, sum the best per-cluster histogram/payload estimate,
   extra bits and integer-configuration signaling cost. Choose the minimum,
   retaining the existing width/tie order.
3. Build only that width's final model and encoder tables. When the caller
   does not request a cost, return without simulating the ordered ANS tokens.
4. If the caller requests a cost, measure that one selected model exactly.
   The cost request does not change the selected model.

This is deliberately different from the first-viable-width shortcut that
failed the 16-bit ramp in the [direct-mode comparison](modular-direct-ans.md).
The new immediate-optimizer counter records the number of model cost
evaluations; deferred external measurements are outside that counter.

The existing rate-mode storage planner conservatively covers estimated
selection. Tests exercise both requested/no-cost forms, initial context maps,
empty/full-width data, cluster caps through 64, admitted parallel work,
retained ownership, underplanned jobs and allocation failure. No smaller
memory bound is claimed merely because fewer candidates are constructed.

## Rate and decision evidence

All **33 original native/exact controls** reproduce the preceding frozen
files byte for byte. On each cap, estimated selection preserves all **195 data
context maps**, integer-candidate counts and histogram-candidate counts.

| Work across 195 data scopes, at either cap | Exact | Estimated |
|---|---:|---:|
| Final width models constructed | 780 | 195 |
| Ordered-token model cost evaluations | 714 | **0** |

These counters come from separate profiled correctness runs. The timing runs
disable detailed profiling entirely. Unit tests independently recompute each
candidate's histogram estimate, verify the chosen model belongs to the exact
candidate set, and compare requested exact costs with actually emitted bits.

| Five-image policy cohort | Exact/32 | Estimated/32 | Exact/64 | Estimated/64 |
|---|---:|---:|---:|---:|
| Automatic bytes | 6,898,213 | 6,898,242 | 6,878,464 | 6,878,491 |
| Buffered bytes | 6,887,215 | 6,887,220 | 6,805,981 | 6,805,980 |

Estimated/64 remains **−0.3278% versus stock e9** automatically and
**+0.5720%** buffered on this cohort. The largest absolute per-file change
across both caps is **22 bytes**, on automatic alpha-flower at cap 64.
The largest percentage increase is **0.001109%**, three bytes on buffered
splines at cap 32. The separate ramp is unchanged at either cap: 59,493 bytes
for 32 and 47,547 for 64. LZ77 remains the missing mechanism for its large
gap to stock e9's 8,375 bytes.

Exact selection minimizes its model-plus-ordered-token objective, not complete
file size after section padding. Buffered splines at cap 64 illustrates the
distinction: estimated selection adds 18 model/payload bits but removes 26
padding bits, saving one file byte. This is not a better exact-cost estimate.

## Paired model timing

The timing boundary encloses native **sample-model optimization**: partition
construction, configuration/histogram work, width selection and encoder-table
construction. It excludes loading/capture validation, sample preparation,
learning, tree coding, tokenization, final model/token emission, frame assembly
and independent decoding. Tree-model intervals are recorded separately and
excluded from the following numbers.

Each of the 11 cases runs in fresh serial processes with one warmup and five
measured rounds. Both case order and the three variant orders rotate. The
optimizer has no admitted worker pool in this harness and runs serially.
All **198 timed files** match the corresponding unprofiled rate-control files
byte for byte. Every file also decodes exactly. Durations are summed over data
scopes within one file, then medians are taken per case/variant.

| Data-model construction, sum of case medians | CPU seconds | Wall seconds |
|---|---:|---:|
| Original native Prefix-derived ANS | 2.9904 | 3.0110 |
| Direct rate/64, exact width | 2.7167 | 2.7382 |
| Direct rate/64, estimated width | **0.9754** | **0.9827** |

The estimated/exact CPU ratio is **0.3590** overall and **0.1870–0.6649** per
case. CPU reductions occur on every case. These sums describe this 11-case
component workload, including both policy controls, not an image-weighted
whole-encoder speedup or a comparison with libjxl's full runtime. Raw per-round
CPU/wall values and each case's medians are retained in the ledger.

The machine is an Apple M4 Pro with 48 GiB, macOS 27.0.1, using Release native
archives and the ordinary native compiler settings. The desktop is not
isolated; Spotlight activity was observed at timing start. CPU time is the
primary comparison, with rotated paired ordering and wall time reported as
additional evidence. No builds or other qualification processes overlap this
timing study.

## Validation and next boundary

All six selected native CTest entries pass in Release and ASan/UBSan:
entropy, entropy storage planning, estimated selection, its workflow fault
sweep, prepared-stream coding and independent prepared-stream reconstruction.
The new direct-mode storage tests inject **179 allocation failures per build**
across requested/no-cost forms and verify atomic outputs and released backing.
All ten research CTest entries pass in both builds, including timed/unprofiled
evidence separation and rejection of false zero-simulation claims.

The rate replay emits **55 exact Release files**, and the four sanitizer
capture controls emit **20 files identical to Release**. The latter cover a
shared photographic model, local palettes, true 16-bit RGBA, and the large
buffered alpha tree. Linked pinned reference archives are uninstrumented;
reference vptr checks and unavailable macOS LeakSanitizer are excluded as in
the earlier studies. Native code and the harness are instrumented.

The validation manifest binds **2,815 artifacts**, including the new native
test binaries/logs, source snapshots, archives and all rate/timing outputs.
`analyze_estimated_entropy.py` verifies those bindings, unchanged exact
controls, per-scope work counters, reconstruction, resource-test logs and the
serial timing protocol before producing the report. `entropy_model_timing.py`
provides explicit `run` and `analyze` commands; timing evidence never replaces
the separate profiled rate controls.

An explicit private prepared-stream entropy policy is now
[qualified separately](modular-prepared-entropy.md), including stream-level
storage, failure publication and complete-file comparison with these controls.
Native palette/RCT/WP selection,
one fit per selected scope, held-out rate and whole-stage performance remain
required before public defaults or GPU routing change.
