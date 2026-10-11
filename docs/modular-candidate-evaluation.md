# Private Modular candidate evaluation

The private search has three explicitly selected evaluation modes. The candidate
traversal, learning budgets, checkpoints and token generation remain shared.

- `kEncodeAll` is the unchanged selection control: write each complete candidate
  and retain the smallest file. With an observer, obtain its exact payload costs
  directly from emission rather than running an additional ANS simulation.
- `kMeasureAll` builds each complete model and requests exact section costs for
  that model, serializes the real compact tree/model/stream headers, then measures
  image/frame headers, TOC widths and section padding. It retains only the winning
  policy, re-creates that candidate and emits it once. That additional token/model
  pass is included in complete-call timing and actual GPU operation counters.
- `kShortlist` samples deterministic interval midpoints from each candidate's
  complete canonical tokens. The sample bound is `32768 + stream_count` (and
  never exceeds the full token count). It builds a sample model and estimates
  payload cost using actual normalized frequencies and extra-bit counts. Keep the
  baseline plus three lowest estimates, exactly evaluate these in original order,
  and emit the winner. This still performs all tree learning and full tokenization;
  it is an experiment in reducing full-population modeling and repeated emission.

Earlier candidates win ordinary size ties. A final learned tree outranks a
checkpoint on equal bytes, including finals encountered later. Exhaustive sizing
preserves these rules and winning policy even when profiling is disabled.
Shortlist score ties retain earlier candidates. Its score is a selection heuristic;
it is never an allocation bound or an exact size claim. Exact finalist comparison
and final emission still use the caller's selected exact/estimated ANS model policy.

The candidate ledger records original index, checkpoint status, policy, incumbent
improvement, elapsed evaluation time and whether the record is exact. Exact costs
partition the complete file into payload, tree, entropy model (including the
no-LZ77 flag), stream headers (including default DC matrices), image/frame headers,
TOC and section padding. Their sum equals eight times the actual file size.
A screened record has estimated bits and sample count, and no exact file cost.
Shortlist finalists receive a second, exact record with their original indices.
The final winner emission is excluded from the ledger's candidate count but is
included in the encoder wall time and actual emission/token work counters.

The observer borrows policy owners only during its callback. Diagnostic storage
is caller-owned and must not retain managed encoder owners. The profile tool
stores plain serialized rows and writes them after timing; row serialization is
included in complete-call time when `--candidate-ledger` is enabled. Report that
setting in comparisons. Observer failure aborts public output publication.

## Resource and failure contract

The allocation-free workflow/search plans explicitly include cost vectors,
compact serialization scratch, sample storage and up to four retained finalist
policies. Estimated ANS selection with exact sizing also reserves GPU exact-cost
scratch: the earlier estimated model-only path did not need it. Completed tokens
and models are borrowed only while their owners remain alive. All errors retain
prior public output, and the exact-cost APIs retain their previous result on
failure. Oversized TOC entries and integer overflow are rejected.

Empty Modular streams contribute neither a stream header nor an ANS terminal
state. Nonempty streams are assigned to their actual TOC sections before padding;
this matters for single-group files, where multiple logical contributions share
the global section.

## Local validation and qualification boundary

The size test compares actual files across all six integer formats, tiny images,
weighted prediction, transforms and multi-group boundaries; the pinned decoder
checks the original samples. It also tests all TOC encoding thresholds and
measurement allocation-failure atomicity.

The search test covers all six formats, prefix and both ANS selectors, checkpoints,
constant/palette data, CPU and native Metal, profiling disabled, admission one byte
below the plan, invalid modes and observer failures. Every exhaustive candidate
cost and policy/order matches actual emission. Shortlist exact finalist costs
match the same control, the minimum is emitted exactly once, the baseline remains
eligible and decoded samples agree. CPU-only builds exercise the same contract.
The existing training/profile and native adaptive workflow suites remain gates.
CI explicitly builds both new tests before running its Modular qualification gate.

Logs: `build/entropy-step3-{build,tests,cpu-build,cpu-tests}.log`. The tested profile
binary will be frozen under `build/entropy-milestones/step3/`. Corpus qualification
must still measure complete calls, winner recall and byte regret; unit-test
success is not evidence of a speedup or rate neutrality.

The profile executable accepts `--measure-all` or `--shortlist`, and independently
`--candidate-ledger`, alongside the existing Metal and ANS-selector switches.
The default remains exhaustive emission.
