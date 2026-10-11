# Private exact Modular GPU populations

The hybrid can now produce a completed per-context histogram of raw residual
values beside its resident canonical tokens. This is an optional capability and
private workflow flag, `adaptive_populations` (`--metal-populations` in the profile
tool). Public defaults are unchanged.

The Metal operation supports up to 256 contexts, values in `[0,4096)`, and at most
`UINT32_MAX` tokens. Context/value counts are exact integers. Each group processes
2048 tokens with 128 threads; groups share up to 32 independent counter shards.
Per-thread runs reduce contention for repeated bins. A second kernel sums shards
into context-major uint64 counts, with an explicit dependency inside one joined
submission. Counts cannot overflow a uint32 shard because no bin can exceed the
bounded total token count.

The shape-only planner bounds counters at 64 MiB plus two status words and the
completed output at 8 MiB. Input token arenas are borrowed and excluded from this
operation bound. Each device range must have the exact logical extent, native
ownership and proper alignment; shared-buffer input ranges must be disjoint.
The completed output owns its backing and remains valid after token owners die.
Allocation, submission, completion and data-dependent context errors preserve the
old output and drain work before releasing its backing. No device failure is
silently retried on CPU.

Values outside the supported range are counted in a separate exact counter.
They never wrap, truncate or alias a dense bin. A successful operation with any
wide values causes the workflow to discard the partial dense table and run its
complete existing CPU population producer. Unsupported capabilities/shapes and
small inputs also select CPU explicitly. Initially, the crossover requires at
least 262144 tokens and at least `4 * contexts * 4096` tokens. These are conservative
private thresholds to qualify, not a universal dispatch policy. Shortlist sample
modeling bypasses the full-image population operation; its exact finalists and
final winner may use it.

The CPU model builder derives its original HybridUint histograms and extra-bit
counts from exact populations, preserving the original clustering policy. It
merges counts through the final composed context map, including any initial
context remapping, to produce sorted unique weighted values. These drive the
same prefix configuration search and the existing prepared-population ANS builder.
Optional per-section costs still use authoritative ordered tokens. No ordered
token collection or sorting is needed for population-backed model construction.

The complete adaptive storage plan adds independent GPU scratch/output maxima
and CPU merge storage to its existing reservation. This accounts for resident
tokens, cached features, model work and exact ANS scratch that can coexist. The
plan is deliberately conservative; actual peak backing is measured separately.

## Attribution

`population_seconds` includes allocation, zeroing, dispatch, reduction, completion
wait and completed-output borrowing; it is nested inside the model wall stage.
`population_gpu_calls` and `population_tokens` record actual producer work.
`population_cpu_small`, `population_cpu_unsupported`, `population_cpu_wide` and
`population_screen_skips` identify each explicit fallback/bypass reason.
`prefix_population_models` records consumed complete dense tables;
`prefix_histogram_tokens` records ordered CPU histogram reads. The population
merge timer is separate from configuration search. The initial-histogram timer
also includes CPU derivation from the dense table and is therefore not zero.

## Local qualification

Native tests cover 22 population cases and 16,992,753 token counts, including a
single bin above 2^24, empty bins, context 255, partial threadgroups, disjoint
shared-buffer ranges and guards, wide values, foreign/impostor owners, invalid
contexts, exact admission and one-byte-short admission, injected allocation /
submission / completion failures and completed-output lifetime.

CPU tests compare prefix models, retained populations, section costs and both ANS
selectors for 1/3/17/65/256 contexts, with and without initial context remapping.
They check 64-bit counts above 2^40, overflow rejection, all bounded-plan thresholds,
and 341 allocation failures with distinct prior outputs.

Complete native workflow tests cover all six formats, prefix and ANS, both ANS
selectors, a full measured adaptive search, small-image fallback, wide-value
fallback, byte equality and pinned-decoder sample equality. Eleven Release suites
and six CPU-only suites pass, including shared entropy/VarDCT regressions, both
candidate evaluation suites and the older adaptive hybrid failure tests.

Logs: `build/entropy-step4-{build,tests,cpu-build,cpu-tests,final-build,final-tests}.log`.
The frozen milestone binary is saved under `build/entropy-milestones/step4/`.
Complete-workflow performance, coverage and combined rate/time results are still
pending; operation correctness alone does not establish an end-to-end gain.
