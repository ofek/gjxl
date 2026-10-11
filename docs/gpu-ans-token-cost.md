# Exact ANS token-cost evaluation on GPU

## Purpose and policy boundary

The [adaptive hybrid pilot](gpu-adaptive-modular-hybrid.md) attributes 109.474
seconds, or 27.6% of aggregate complete-search time, to exact ANS token-cost
simulation. This operation targets that measured work without changing the
entropy policy or the learned-tree search.

`OptimizeAnsEntropyCodeImpl` still constructs and normalizes models, selects
HybridUint configurations, prunes equivalent alphabet widths, applies lower
bounds, groups candidates and breaks ties. The GPU operation measures the same
surviving models against the same ordered tokens. It does not introduce an
estimated cost or choose a different model. Final bitstream emission remains CPU
work in this slice.

## Operation and independence

`GpuAnsTokenCost` in `src/gpu/ops/ans_token_cost.h` is an optional private backend
capability. It borrows completed immutable split token buffers (uint32 values and
uint16 contexts), an ordered partition into sections, and one to four model
views. Each model has at most 65,536 contexts and 64 clusters; a cluster contains
one valid HybridUint configuration and a normalized 4,096-entry rANS table with
at most 256 symbols. Models in a batch need not share configurations or tables;
the caller can submit the optimizer's existing equivalent-symbol groups.

There is one independent recurrence per **section and model**. Within that task,
tokens run in reverse order, starting from `0x13 << 16`:

1. Map the token's context to a cluster and encode its value as HybridUint.
2. Add its extra-bit count.
3. If `state >> 20 >= frequency`, add 16 bits and shift the state right by 16.
4. Compute the exact quotient using the existing `ceil(2^44 / frequency)`
   reciprocal, then look up the symbol/remainder in the model's reverse map.
5. Update the state to `(quotient << 12) + reverse_map[offset + remainder]`.

Every section includes 32 final-state bits, including empty sections. There is no
padding in this cost. Model bits are supplied by the existing CPU writer and
added by the caller. Splitting one section into independent token chunks would
change its state history and is outside this operation.

Results are candidate-major, then section-major records: uint64 bit count,
uint32 final state, uint32 reserved zero. Keeping the final state allows device
qualification to distinguish equal costs from equal recurrence. The caller
performs checked summation and retains the optimizer's original candidate order
and strict tie rules.

## Storage and ownership

The input ranges must belong to the selected backend, have the advertised exact
logical sizes and natural alignment, fit their buffers, and not overlap. Empty
sections are allowed; together all sections must partition a nonempty token
arena without gaps or duplicate tokens. Borrowed producer owners remain alive
through completion. Native Metal buffer validation is additionally required by
the implementation; a matching numeric backend identifier alone is insufficient.

`ComputeAnsTokenCostStoragePlan` is allocation-free and preserves its old output
on failure. It accepts shape maxima before model construction, allowing the
workflow to reserve capacity before allocation. For T tokens, S sections, C
candidates, Q contexts per candidate and K clusters per candidate:

- Section descriptors use 16S bytes; model descriptors use 32C bytes.
- Context maps reserve CQ bytes and clusters reserve 11,520CK bytes. Each cluster
  slot contains four uint32 configuration/symbol-count words, 256 uint16
  frequencies, 256 uint16 offsets, 256 uint64 reciprocals and 4,096 uint16 reverse
  map entries, followed by zero padding.
- Arena regions start at 256-byte boundaries. A zero-initialized uint32 status
  word follows the cluster slots. Each model descriptor records its uint64 map
  and cluster offsets, uint32 context and cluster counts, and two reserved words.
- Completed result backing is 16SC bytes. The transient peak additionally counts
  the device descriptor/model arena, one fresh host staging copy of that arena,
  and fresh caller section and cluster-view vectors. Direct initialization of
  shared memory may use less than this conservative bound.
- T does not introduce another token allocation. Input tokens, CPU model owners,
  previously published output, cached features and optimizer scratch are composed
  separately by the workflow. Managed backing excludes inline objects, allocator
  metadata, backend initialization and driver memory.

The capability is synchronous and failure-atomic. Invalid descriptors, models,
ownership, ranges or admission fail before device work. The device checks actual
token contexts, absent/out-of-range symbols and state/reverse-map bounds. Unknown
status bits are protocol errors. After submission, failures are terminal for
that attempt; outstanding work is drained before backing is released and the
old result owner is preserved. A missing capability or an unsupported preflight
may take an explicitly counted CPU route before dispatch.

## Implementation sequence and evidence

1. Define the operation, validation and storage contract. Implemented with
   boundary, malformed-model, exact-capacity and allocation-free validation tests,
   including borrowed views of a real scalar-optimized ANS model.
2. Implement a native Metal section/candidate kernel and completed-result owner.
   Implemented and qualified against scalar section bits and final states across
   alphabet widths, HybridUint configurations, cluster maps and edge values.
3. Route only the existing exact-cost call through the capability for resident
   adaptive tokens. Compose storage and preserve all optimizer decisions and
   error/owner lifetimes. Implemented behind `adaptive_ans_cost`, with submissions,
   fallback counts and the existing nested exact-cost elapsed interval.
4. Qualify complete-file/policy equality, pinned decoding, exact admission,
   injected allocation/submission/completion failures and disabled profiling.
5. Repeat paired full-resolution complete searches. Include model upload, waits,
   cache creation, all candidates, CPU work and publication. Preserve compressed
   sizes and document resource changes before making throughput claims.

Metal now advertises this capability. The standalone operation borrows resident
values/contexts, directly initializes shared descriptor/model backing, dispatches
one lane per section/candidate and publishes a completed shared result arena
after waiting and validating status. No token copy or token-sized scratch is
introduced. The new kernel is included in external and embedded libraries.
Encoder routing is now implemented as described below; complete-path performance
measurements are still required. The earlier hybrid pilot predates this operation.

`ans_token_cost_plan` passes in the Release/Metal and CPU-only builds. Tests cover
normalized native model tables, malformed configuration/frequency/offset/map and
reciprocal data, disjoint resident ranges and foreign owners, empty sections in a
nonempty partition, count overflow, exact device capacity and one byte less,
failure-atomic plans and device status decoding. Logs are retained under
`build/adaptive-integration/ans-cost-contract-*`.

Native qualification on Apple M4 Pro passes **829 cases, 224,848 exact bit-count
and final-state pairs, and 1,713,380 token recurrences**. Coverage includes all 816
valid HybridUint configurations, real scalar-generated models across alphabet
widths 5–8, arbitrary valid-range reverse-map permutations, frequency extremes 1
and 4,096 (including native single-symbol tables), edge uint32 values, context ID
65,535, 64 clusters, mixed-model batches, empty and irregular sections, and long
ordered streams. The reference uses ordinary division and is checked against the
production scalar counter; selected cases also check the final writer's bit count
and emitted 32-bit state. General synthetic table cases establish arithmetic
equivalence, not independent decoder-valid model serialization.

Disjoint ranges sharing one buffer, naturally aligned nonzero offsets, input and
padding immutability, completed-result lifetime and exact live backing are
checked. Exact native allocation capacity and one byte less behave as planned.
Eight explicit failure-atomic checks cover allocation, submission, completion,
malformed sections, foreign and nonnative buffer owners, and data-dependent
context/absent-symbol failures. Failed attempts preserve the prior result and
release all new backing and pending work. There is no managed host vector
allocation inside the native operation; caller-view allocation fault coverage
belongs to the upcoming workflow integration.

The full Release build succeeds. All 19 selected ANS/entropy, Metal Modular,
submission-storage, external/embedded-library and Make/Ninja build-graph tests
pass. Logs are retained under `build/adaptive-integration/ans-cost-*`; the final
test details and frozen native test artifacts are recorded in
`build/ans-cost-device-20261011/manifest.json`. This establishes operation
correctness, not a complete-encode speedup or changed compression rate.

## Complete-search integration

The private `ModularEncodingOptions::adaptive_ans_cost` switch requires an
`adaptive_gpu` backend. The profile tool's `--metal-ans` option enables both;
`--metal` retains the earlier token-only hybrid route as a control. Public
encoding defaults remain unchanged. Prefix mode never invokes ANS costing.

`ModularGpuAnsCost` validates that the optimizer's split streams partition the
completed token owner's host views, then borrows the same native device ranges.
It builds small section/model views for the existing candidate group, measures
the group synchronously, validates the completed result, performs checked
summation and moves exact costs back to the unchanged selector. It does not
recollect tokens, normalize models, change pruning, reorder candidates or cache
mutable model state. The internal `AnsTokenCostEvaluator` hook is an explicit
call argument, not a global or thread-local override.

If the capability is absent or shape/device-limit preflight is unsupported, the
adapter explicitly requests the original scalar cost measurement. A provider
error, including Unsupported after work has begun, propagates immediately. The
existing CPU optimizer and its other clients continue through the old entry
point with no provider. The provider contract test checks identical models and
exact costs for handled and declined requests, and unchanged public model/cost
outputs after a terminal provider error.

The complete workflow adds the worst-case four-candidate cost-operation peak for
each transformed layout to its existing conservative CPU-plus-feature bound.
This covers GPU dispatch/result backing and caller views while the CPU models,
resident tokens and feature cache are alive. Admission includes the bound even
when a capability later falls back. The CPU optimizer's existing cost-vector
bound covers result conversion and selection; no second token arena is added.

Profiling reports `ans_gpu_groups`, `ans_gpu_states` and
`ans_cpu_fallback_groups` inside `adaptive_gpu`. Successful GPU groups each add
one actual submission. Cost upload, dispatch, wait and conversion are included in
`entropy_work.ans_token_cost_ns`, itself nested in `stages.model`; those intervals
must not be summed again. With profiling disabled, actual backend submission and
allocation totals remain available. The capture route is
`hybrid-ans-complete-search`.

Integrated tests pass 39 complete CPU/hybrid file and selected-policy matches,
including 14 additional ANS-enabled/prefix-bypass cases covering all six formats,
palette patterns, expanded checkpoint search, one/four CPU participants and
grouped Squeeze/weighted prediction. Pinned libjxl decodes 92,908 source samples
exactly. Profile-disabled and owned publication paths match too. Capability
fallback is explicit and byte exact; ANS-inclusive one-byte-short admission and
an ANS switch without a GPU are rejected before publication. The original and
ANS-enabled tiny-input complete-search allocation sweeps plus existing device
and late-candidate faults pass 3,905 injected failure-atomic checks. This count
does not mean exhaustive allocation sweeps on the photographic corpus.

The full Release build and all 19 selected integrated ANS/entropy, Modular GPU,
profile, embedded-library and submission-storage regressions pass. Test logs use
the `build/adaptive-integration/ans-cost-integrated-*` prefix. The CPU-only ANS
contract/provider, Modular profile and GPU-token handoff tests also pass. Paired performance
captures must now establish whether the exact operation improves complete calls
on the original full-resolution inputs.
