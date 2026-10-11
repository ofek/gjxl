# Adaptive Modular GPU implementation

## Objective and comparison boundary

Accelerate the richer lossless Modular search while retaining its compression
decisions. Qualification must time the **complete search**, including learning,
all candidate encodes, CPU fallbacks, and output publication. A replay of the
winning prescribed policy is useful diagnostically but is not an adaptive
end-to-end result.

The initial comparison uses the original full-resolution 1029, 1039, and 1049
inputs and pinned libjxl's **default** lossless effort 7 and 9 policies. Report
encoded bytes and complete-call time separately. Same-numbered efforts are not
equivalent algorithms; neither a reduced libjxl policy nor a faster prescribed
GJXL encode establishes a speedup at comparable compression.

## Integration baseline

- Worktree: `gjxl-modular-adaptive-gpu`, branch `feat/modular-adaptive-gpu`.
- GPU parent: `cbc2d8eda26be311356c66504049fda4e9a576ac`.
- Learning parent: `5faf68d53bf959ea1605e73e56b1cb5a3cce3434` from
  `feat/modular-tree-learning`.
- Pinned libjxl: `e8ff09762481785938d8e4e01333ed3917571161`.
- Initial device: Apple M4 Pro; Release builds retain debug symbols for profiling.

The integration keeps both the completed GPU token arena accounting and the
larger learned-tree/header/publication bounds. GPU-input/CPU-token continuation
accepts large CPU trees. The original GPU token producer still supports only
31 nodes, nonweighted prediction, and prescribed RCT-only input. Large trees are
borrowed and rejected before allocation or GPU submission on that route.

Existing public encoding defaults remain unchanged. Experimental budgets and
checkpoint selection are private qualification controls.

## Implementation milestones

1. **Integrate and establish the CPU control.** Validate the merged learner,
   storage admission, completed-token handoff, Metal token equality, and pinned
   decoder round trips. Retain exact binaries, input/output hashes, options,
   resource settings, and test logs.
2. **Measure the complete search.** Add opt-in attribution for feature gathering,
   learner preparation, split evaluation, and tree growth, alongside existing
   input/transform/token/model/emission stages. Measure the original learner at
   expanded budgets with checkpoint selection; do not replace it with the
   different quantized learner. Compare full-resolution compressed sizes with
   default libjxl, and collect sampling profiles to verify hot functions.
3. **Extend the GPU operation and storage contract.** Separate transformed plane
   identity from stream-local channel properties; accommodate dynamic trees,
   canonical slice bounds, and exact weighted predictions/property 15. Specify
   ownership, admission, overflow, failure publication, and fallback before
   dispatch implementation. A CPU weighted-feature bridge is an initial valid
   implementation, with its cost included in the complete workflow.
4. **Implement and qualify GPU work.** Use the measurements to prioritize reusable
   transformed preparation, feature production, and bulk learner statistics.
   Keep greedy tree control and deterministic reductions exact. Verify tokens,
   selected trees, complete bytes, and decode against the CPU control. Include
   unsupported cases through an explicit, measured CPU fallback.
5. **Optimize the complete path.** Reuse work across candidates/checkpoints where
   it is policy-independent; bound retained memory and include cache creation in
   timing. Measure paired CPU/GPU calls with the same policies and thread budget.
   Publish limitations and commit each independently validated milestone.

## Dependencies and constraints

The expanded search retains the old candidate set, then compares larger trees
and intermediate checkpoints using complete encoded size. Candidate order and
tie handling are observable policy and must remain stable.

Weighted prediction scans every pixel within each stream/channel slice, even
when training keeps only a sample. Its nonlinear error state prevents arbitrary
pixel independence. Separate slices are independent. Weighted features depend
on the transformed input and predictor settings, not on the selected tree, so
reuse across checkpoint encodes is possible with explicit storage bounds.

Residual coding, property generation other than weighted history, histogram
construction, sorted property scans, and candidate split costs expose bulk
parallel work. Greedy leaf selection remains ordered; exact integer costs and
strict tie rules must survive GPU reduction. Small-leaf CPU work may remain
faster than repeated GPU submissions and waits.

Palette and Squeeze winners are part of the richer search. Forward Squeeze
pairs and palette lookup expose parallelism, but transformed layouts differ
from the current four equal-sized RCT planes. Optimizing only untransformed,
nonweighted, small trees does not cover the rate-improving winners.

## Evidence status

The integrated Release/Metal build passed all 27 focused Modular tests, including
learning budget/coverage/checkpoint allocation failures, the frozen and direct
libjxl oracles, API/CLI round trips, and both Metal token submission routes.
The separate `modular_workflow --allocation-shard 0 64` also passed all its
workflow/admission assertions and every 64th injected allocation failure. The
other 63 allocation shards were not run. Logs are retained in
`build/adaptive-integration/`.

Complete-search instrumentation and a pinned-decoder capture tool are now
implemented; see `tools/modular_training_study/README.md`. The profile test checks
disabled/nested sinks, accumulation, disjoint training attribution, selected
policy equality, and complete-byte equality. The budget, coverage, checkpoint,
and frozen-oracle tests also pass after instrumentation.

Six complete searches (profiling enabled/disabled for `photo-macan`,
`palette-pattern`, and `alpha16-ramp`, all at 262144 samples/256 leaves with
checkpoints and four CPU participants) decode exactly and match the saved
tree-learning branch's complete files byte for byte. These exercise photographic,
palette, and 16-bit alpha input. Captures are retained in
`build/adaptive-profile-smoke/`; their timing is diagnostic.

The full-resolution CPU control and default libjxl e7/e9 captures are complete:
see [`gpu-adaptive-modular-profile.md`](gpu-adaptive-modular-profile.md). All
outputs decode exactly. Summed GJXL bytes are 1.20% above e7 and 2.18% above e9;
complete CPU searches take 199–225 seconds per image in these diagnostic runs.
Token generation is the largest phase (38.8%), followed by modeling (30.7%) and
training (20.1%). No GPU speedup has been established by this effort yet.

The [adaptive token contract](gpu-adaptive-modular-token-contract.md), private
operation interfaces, fixed-width records, tree eligibility, and storage planner
are implemented and tested with Metal enabled and disabled. CPU transformed-plane
and weighted-feature preparation, Metal learned-tree evaluation, and completed
token continuation are now implemented. Device qualification on Apple M4 Pro
matches 3,778,462 scalar tokens and 34 complete files, with exact pinned-decoder
reconstruction of every source sample. Large trees, weighted property 15, palette
and Squeeze layouts, signed arithmetic, bounded storage and failure publication
are covered; the contract document records the precise scope.

The new operation is connected to full adaptive search through a private option.
One call retains at most one transformed layout and its weighted features, with
composed admission and explicit fallback accounting. All 25 complete CPU/hybrid
workflow cases match selected policies, candidate counts and complete bytes, with
pinned-decoder reconstruction; cache-key/lifetime cases and 1,770 failure checks
also pass. The original candidate order and tie behavior remain intact.

The profile capture tool now accepts `--metal` and reports actual GPU submissions,
feature preparations/reuses, GPU candidates and CPU fallbacks. CPU transformation,
weighted-history construction and every candidate remain inside the timed call.
Paired full-resolution measurements are next; no adaptive end-to-end GPU speedup
has yet been established. The original prescribed GPU path remains available with
its original eligibility; public encoding defaults are unchanged.
