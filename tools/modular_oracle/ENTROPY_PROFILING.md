# Modular entropy attribution

This diagnostic extends the prescribed three-route benchmark without changing
encoder policy or public dispatch. `profile` appends the existing entropy work
counters and ANS candidate counts to its CSV. `measure` and `qualify` leave those
counters zero. The old wall-stage columns and route meanings are unchanged.

`prefix_value_aggregation_work_ns` is a new **nested** counter within
`prefix_config_search_work_ns`. The latter retains its original inclusive
meaning for existing consumers. Subtract aggregation before making a disjoint
breakdown. Entropy counters are nested within `model_ns`; never add them to the
complete-call wall stages. Current Modular modeling explicitly uses one CPU
participant, so the analyzer checks that disjoint work fits within its parent.
That check must be revisited if modeling becomes parallel.

## Frozen build and inputs

Build `gjxl_modular_gpu_benchmark` in Release with Metal enabled, Metal stage
profiling disabled, and no sanitizers. Retain the exact source snapshot, base
revision plus overlay hashes, build logs/cache, binary hash and linked libraries.
The pinned Modular reference decoder can remain linked. New inputs require the
existing `qualify` gate before collecting measurements.

Prepare a new run directory containing:

- `cases.tsv` in the existing nine-column prescribed-token format, with one job
  per case. Use the same inputs and policies across every route/backing.
- `expected/CASE.jxl` from an identified qualification. The benchmark compares
  every complete encode directly with these bytes. Reusing qualified outputs
  establishes byte identity; it does not constitute a fresh independent decode.

For the October 10 study, the 14 cases are selected from the committed token
qualification: tiny RGB8, 1 MP RGB8/four participants, 4K RGB8/four participants,
4K RGBA16/four participants, 1 MP RGB16 noise/ANS, and each of photos 1029, 1039
and 1049 with gradient prefix, split7 prefix and gradient ANS/four participants.

## Collection

```sh
python3 tools/modular_oracle/run_entropy_profile.py FROZEN_BINARY RUN_DIRECTORY
python3 tools/modular_oracle/analyze_entropy_profile.py RUN_DIRECTORY
```

The default run uses one ordinary process (seed 17), two profiled complete-call
processes (137/239), and two isolated model-replay processes (181/283). Each case
has two warmups and six measured triplets per process, balancing all six route
orders. Case order is shuffled in every process. Setup is recorded separately.
The driver refuses existing captures, records input/expected/binary hashes,
disables Metal validation for timing, and records host activity and thermal
reports without stopping other applications. These are fresh processes in one
session, not independent quiet-session performance confirmation.

## Isolated token-backing replay

The benchmark's `replay ... tokens` mode prepares canonical CPU and composed GPU
tokens once per case, then verifies every token and stream boundary. It creates
a host split-array copy of those same CPU values/contexts and replays the same
entropy optimizer over:

1. CPU interleaved `{uint32 context, uint32 value}` tokens.
2. Host split `uint32` values and `uint16` contexts.
3. Borrowed split arrays in the completed Metal token arena.

All three backings remain alive. Every resulting `EntropyCode`, including
context maps, configurations and code tables, must equal an untimed CPU model.
The timer includes prefix optimization and, when selected, ANS optimization.
Preparation, copying, token comparison, model comparison and final model
destruction are outside it. No GPU work is submitted in the timed interval.
Temporary prefix-model replacement inside ANS optimization is included, as in
the complete workflow.

These warmed replays separate representation and allocation-backing effects
under shared resident conditions. They do not reproduce the original handoff's
first-touch/cache state, isolate a hardware cause, or measure the complete-call
cost of changing representations. The replay's resident copies are diagnostic
storage outside the workflow's full-call resource-admission comparison.

## Analysis and interpretation

The analyzer validates matrix completeness, balanced ordering, unchanged inputs,
byte fingerprints, device submission/allocation counts, resource envelopes,
matching policies, zero ordinary counters, candidate-count equality and nested
timing boundaries. The benchmark additionally performs full byte/model equality
checks. It retains per-cohort results and computes signed stage deltas from
matched rounds. Positive `tokens_minus_cpu`, `host_split_minus_interleaved` or
`metal_split_minus_host_split` means the second representation took longer.

Reported stage medians need not sum to the median model duration. Fractions are
computed within each call before taking their median. Unattributed model time
remains visible and includes uninstrumented bookkeeping, cleanup and timer gaps.
No profiled stage is subtracted from an ordinary complete-call measurement.
Small differences need repeatable evidence before motivating an implementation
change; the current study selects the next target, not an automatic GPU policy.
