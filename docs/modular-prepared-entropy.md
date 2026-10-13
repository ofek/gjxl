# Direct ANS in the private prepared-stream encoder

The private prepared-stream encoder now supports direct rate-optimized ANS
with exact or estimated width selection and an explicit histogram cap.
All **55 captured-case files** match the preceding standalone entropy
controls byte for byte. The new path has stream-level managed-storage and
atomic-failure qualification; it is ready for composition with native
preparation and learning.

This is a coding integration milestone. Transform, WP, tree and scope decisions
are still prescribed. It does not qualify an autonomous native search policy,
whole-encoder timing, public defaults or GPU routing. The
[qualification ledger](modular-prepared-entropy-results.json) records the
file comparisons, resource measurements and evidence bindings.

## API and storage contract

[`PreparedModularEntropyOptions`](../src/codestream/modular/prepared_streams.h)
is shared by the planner and encoder. It selects Prefix, ANS using the existing
Prefix-derived partition, or direct rate ANS. The legacy overloads retain their
previous plans and output bytes. A caller selects the qualified estimated/64
path explicitly:

```cpp
PreparedModularEntropyOptions options{
    .policy = PreparedModularEntropyPolicy::kDirectRateAns,
    .selection = AnsSelectionMode::kEstimated,
    .maximum_ans_clusters = 64,
};
if (auto status = ComputePreparedModularStreamsStoragePlan(input, options, &plan);
    !status.ok()) return status;
// Admit plan.working.peak_bytes using the caller's resource reservation.
return EncodePreparedModularStreams(input, options, &output);
```

Options are validated before allocation or output publication. Direct models use the existing rate
optimizer; estimated selection requests no exact token cost. Tree coding
retains its previous policy. A shared tree retains one shared data model, and
each populated local stream constructs one local model. Local tokens remain
excluded from shared optimization.

The planner uses the direct-rate optimization bound and the actual requested
cluster cap for model ownership, serialized model size and writing scratch.
It no longer applies the Prefix path's 32-cluster bound to direct/64 models.
Exact and estimated selection deliberately share a conservative bound.
The captured controls exercise **244 data scopes with more than 32 histograms**
across the two 64-cap variants.

The existing borrowed-input exclusions still apply: source/transformed planes,
trees, view ownership, prior outputs, stack storage and allocator overhead
are caller-owned or outside the managed bound. Fresh tokenization, model work,
emission, sections, assembly and result publication are covered. These numbers
are not process RSS or a production streaming-memory claim.

## Qualification

| Check | Result |
|---|---:|
| Captured Release files identical to standalone controls | 55 / 55 |
| Captured sanitizer files identical to Release | 20 / 20 |
| Generated cases, across formats/transforms/model scopes | 108 |
| Generated exact files per build | 648 |
| Generated Release/sanitizer file identity | 648 / 648 |
| Historical generated Prefix/ANS controls unchanged per build | 216 / 216 |
| Native allocation failures injected per build | 4,350 |
| Of those, new direct exact/estimated paths | 2,880 |
| Native CTest entries per build | 5 / 5 |
| Research CTest entries per build | 10 / 10 |

Generated controls cover six integer formats, shared/local/mixed model scopes,
empty slots, clipped groups, shifted DC groups, ordered palettes/RCT/squeeze
and the existing WP presets. Exact and estimated direct policies run at caps
32 and 64 alongside the original Prefix and ANS controls. Every generated
file is decoded independently. The capture replay also compares native sample
and tree tokens with pinned libjxl before the integrated file comparison.

Failure sweeps check every managed allocation in a small shared/local fixture,
including model construction, tree/data emission and final publication.
Failure preserves a sentinel output and releases partial backing. Invalid
options allocate nothing; underplanned jobs fail atomically. Successful runs
check both peak usage and retained output ownership.

All correctness checks use fresh Release and ASan/UBSan builds against pinned
libjxl `e8ff09762481785938d8e4e01333ed3917571161`. Native libraries and the
research harness are instrumented in the sanitizer build. Linked reference
archives are uninstrumented, reference vptr checks are excluded, and unavailable
macOS LeakSanitizer is disabled. Managed leak checks pass. Builds and checks
overlap, so their durations are not performance evidence.

The validation manifest binds **3,797 artifacts**, including source snapshots,
linked archives, tested binaries, generated files, capture files and logs.

## Rate, resources and the next step

Byte identity preserves the [estimated-width results](modular-direct-ans-estimated.md):
direct estimated/64 totals **6,878,491 bytes** on the five automatic-mode images
(−0.3278% versus stock e9) and **6,805,980 bytes** buffered (+0.5720%).
Those are conditional results on captured e9 decisions. The separate 16-bit
ramp remains **47,547 bytes**, versus stock e9's 8,375; LZ77 is still absent.

The largest measured managed peak is **201,172,590 bytes** for both exact and
estimated direct/64, on buffered RGBA. The original Prefix-derived ANS control
peaks at 196,767,629 bytes. Width estimation does not reduce the maximum observed
managed peak on this cohort. All variants remain within their computed bounds.
The earlier **2.79× sample-model CPU speedup** is a separate component benchmark;
this integration adds no whole-encoder timing claim.

The [fixed-gradient cost primitive](modular-selection-cost.md) is now qualified
against pinned libjxl values and RCT rankings. Next port the reference palette
constructor and WP cost, integrate selection, then compose native
preparation/fitting once per selected shared or local coding
scope. Full/held-out rate, exhaustive-policy comparisons and paired whole-stage
time/resource checks remain the gates for promoting a default.

For reproduction, build the prepared-stream tests in fresh native Release and
sanitizer directories, retaining their generated-file outputs. Run the five
`native`/direct exact/estimated capture controls with `run_entropy_replay.py`.
`analyze_prepared_entropy.py` verifies the frozen native/research logs, generated
controls, prior standalone files, integrated file equality and memory bounds
before producing this report. Its `--help` lists the required evidence roots.
