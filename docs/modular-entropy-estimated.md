# Estimated ANS selection in complete adaptive GPU search

The private estimated selector reduces complete-call time by **8.44% (1.0922x)**
on the three retained full-resolution photos, with **identical complete files
and winning policies** in this pilot. This qualifies step 1 of
[the entropy efficiency plan](modular-entropy-efficiency.md). It does not establish
libjxl speed parity, a general zero-rate-regret guarantee, or an automatic public
dispatch policy.

## Frozen conditions

- Binary revision: `cd9d1aa931d8fe9fd0c651018c8409e0f3abdcc5`, frozen from a clean
  checkout before subsequent source edits. Binary SHA-256:
  `b66d5b4f7257071cebdef5104e9067b9185559712342667813be5f66387d8648`.
- Images 1029, 1039, 1049: original 4000 x 3000 RGB8 samples, no resizing.
- Full adaptive search: 262144 learning samples, 256 leaves, tree checkpoints,
  one CPU participant, GPU tokens and GPU learner statistics for both routes.
  Exact selection uses the GPU ordered ANS cost evaluator; estimated selection
  constructs one selected width and makes no ordered cost request.
- Complete encoder-call wall time includes host work, allocation, device waits,
  all candidates, emission, and assembly. Backend creation, input/output file I/O,
  and pinned independent decoding are outside the timer. No warmup was removed.
- Apple M4 Pro, active desktop, on battery. These are paired latency observations,
  not an isolated-throughput or thermal-envelope claim. Do not compare their
  absolute times directly with the earlier 99.632-second session.

The original capture has three CPU estimated controls and two alternating GPU
exact/estimated pairs per image: **15 complete calls, 62 hashed artifacts**.
Exact outputs match the frozen pre-integration CPU controls. Estimated GPU
outputs match the new CPU estimated controls. Both selectors happened to choose
the same files here. Independent libjxl decoding checked every original sample.

## Complete-call results

Each timing is the median of two uninterrupted repetitions. Both members of one
sleep-interrupted pair were repeated, as detailed below.

| Image | Exact ANS selection | Estimated ANS selection | Speed ratio | Bytes, either selector |
|---|---:|---:|---:|---:|
| 1029 | 140.579 s | 128.666 s | 1.0926x | 10,821,744 |
| 1039 | 136.280 s | 124.710 s | 1.0928x | 7,230,909 |
| 1049 | 133.970 s | 122.771 s | 1.0912x | 7,336,102 |
| Sum of per-image medians | 410.829 s | 376.148 s | **1.0922x** | **25,388,755** |

Exact ranges were 139.752–141.406, 136.093–136.468, and 133.877–134.063 seconds.
Estimated ranges were 127.934–129.398, 124.684–124.737, and 121.155–124.388 seconds.
The mean model stage fell from **45.141 to 33.941 seconds**. Estimated runs report
zero ANS cost time, submissions, states, and CPU cost fallbacks. Candidate counts
(70, 69, 68) and semantic learning work are unchanged. The rest of the pipeline
still performs the full exhaustive search and emits each candidate.

Actual managed peak backing is unchanged between selectors: 1,370,775,407,
1,356,513,009, and 1,355,708,723 bytes for the three images. The conservative
admission bound drops from 6,730,159,807 to 6,724,216,103 bytes because the
estimated route does not reserve unused GPU cost scratch. These figures cover
managed resources, not total process RSS or Metal driver memory.

## Sleep interruption and retained raw evidence

The original `1039-r2-estimated` call took **193.193 seconds**, with 87.505 seconds
attributed to emission versus 19.608 seconds in its first repetition. Modeling
remained normal (34.752 versus 35.023 seconds). Its process-activity trace has a
72.438887-second gap between 14:04:04.546234 and 14:05:16.985121 UTC. macOS power
logs record Idle Sleep at 10:03:14 EDT and Maintenance Sleep at 10:04:06 EDT during
that call.

The original matrix is retained without adjustment. Its raw sum of per-image
medians is **409.305 seconds exact versus 410.402 seconds estimated (0.9973x)**;
it must not be silently presented as the uninterrupted result. A separate
capture repeats **both** members of the second 1039 pair, in the original
exact/estimated order, using the same frozen binary and inputs under a temporary
`caffeinate -i` idle-sleep guard. Those complete calls took **136.468 and 124.684
seconds**. No sleep duration was subtracted and no fastest repetition was chosen.
The original interrupted call remains part of the correctness evidence.

The raw capture, independent replacement pair, selection rule, power-log hash,
per-repetition values, stages, and resource peaks are all retained in
[`modular-entropy-estimated-results.json`](modular-entropy-estimated-results.json).

## Audit and reproduction

Artifacts:

- `build/entropy-estimated-qualified-20261011/`
- `build/entropy-estimated-sleep-recheck-20261011/`
- `build/entropy-estimated-sleep-events-20261011.log`
- `build/entropy-step1-audit-tests.log`

The auditor checks inventories and hashes, exact matrix/order/commands, input and
reference provenance, actual CPU/GPU routes, semantic work, complete file and
policy equality, stage nesting, and resource bounds. The replacement audit also
checks unchanged work and absence of another long process-observation gap.
Positive original/relocated captures pass; seven independent damaged captures
are rejected: incomplete capture, altered file, missing inventory entry, missing
matrix row, unexpected exact costing, wrong GPU route, and exceeded admission.

```sh
python3 tools/modular_training_study/test_entropy_efficiency_audit.py \
  build/entropy-estimated-qualified-20261011
python3 tools/modular_training_study/analyze_entropy_recheck.py \
  --source build/entropy-estimated-qualified-20261011 \
  --recheck build/entropy-estimated-sleep-recheck-20261011 \
  --sleep-events build/entropy-estimated-sleep-events-20261011.log \
  --json docs/modular-entropy-estimated-results.json
```

Implementation validation is recorded in the main plan: focused native
CPU/hybrid/independent-decoder tests, shared entropy regressions, failure-atomic
allocation sweeps, and CPU-only builds. No new libjxl encode or model comparison
is implied by this selector qualification; the identical-token rate study is a
separate later step.
