# Estimated ANS model selection

This experiment branches from `5faf68d` (qualified CPU tree checkpoint search).
It tests whether histogram-based model selection can remove the ordered ANS
candidate simulation without a material rate penalty. Public defaults remain
exact. No GPU kernels or dispatch defaults change.

## Implementation milestone

`AnsSelectionMode::kEstimated` reuses the existing per-cluster estimates:
serialized histogram bits plus normalized-frequency cross entropy, HybridUint
extra bits and configuration signaling. It selects the lowest estimated-cost
alphabet width and constructs only that width's encoding tables. Context-map
and stream-state costs are common across these width candidates. Equal estimates
retain the first (smaller) width. Clustering, configuration candidates and
histogram precision search are unchanged.

The existing exact mode remains the default. Estimated mode skips the ordered
candidate simulation when no cost output is requested. If a caller requests
`EntropyCodeCost`, the chosen model is simulated once: model bits, token bits
and per-section counts remain exact for that model. Estimates never replace
allocation bounds or exact section-size contracts. This does not promise the
same model or compressed bytes as exact selection.

The private CPU Modular option `ans_selection` reaches the shared frame encoder,
including the CPU continuation after GPU input preparation. Search and tree
checkpoints still compare actual complete files. No GPU timing qualification is
claimed. Conservative existing storage bounds cover the smaller candidate set.

Validation at this milestone:

- 27 focused Release CTests passed, including ANS emission, entropy/storage,
  Modular oracles, transforms, API/CLI, checkpoint search and CPU-side GPU contracts.
  The full legacy `modular_workflow` fault suite was excluded.
- Three ASan/UBSan tests passed: entropy, estimated ANS selection and GPU workflow
  contracts. The pinned reference archives were not sanitizer-instrumented.
- New tests independently score all original width candidates, compare interleaved,
  split and prepared-population paths, and confirm no estimated-mode simulation.
  Requested exact costs match actual serialized model and payload sizes.
- Prescribed and checkpoint-search workflows decode sample-exactly, agree across
  one/four CPU participants, and honor exact/one-byte-under admission.
- All 6,248 managed allocation-failure positions in the new small prescribed/search
  workflow fixtures preserve caller output and release resources.

The first test run passed a null cost pointer to the existing prepared-prefix
helper, contrary to that helper's contract. The test was corrected; the failing
development logs are retained and excluded from qualification.

The [implementation qualification record](modular-ans-selection-qualification.json)
binds source and retained test evidence under `build/ans-selection/` and
`build/ans-selection-sanitize/`.

## Rate and timing study (pending)

1. Use the three saved 12 MP photographic inputs from the entropy profile and
   the 16-input checkpoint corpus. Fix prediction, transforms and trees first;
   include both prescribed Gradient and the saved learned checkpoint winners.
2. Compare exact/estimated models using identical ordered token buffers and a
   shared context partition. Record configuration/histogram/width counts and
   the exact-simulation counter. Compare complete files separately.
3. Repeat ordinary complete encodes and isolated model calls in alternating
   exact/estimated order, excluding explicit warmups. Gather separate diagnostic
   profiles and require them to preserve ordinary bytes. Run timed cases serially.
4. Repeat the adaptive 256K/256 checkpoint search with both policies on all 16
   inputs, requiring old exact controls to remain byte-identical. Add repeated
   timing on representative adaptive cases; report these separately from fixed
   decisions and from any concurrent rate-only runs.
5. Verify retained files with the pinned decoder, record per-image and aggregate
   rate changes, preserve hashes and provenance, and decide whether estimated
   selection merits promotion. No rate or speed benefit is established yet.
