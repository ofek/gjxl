# Native weighted-predictor selection cost

The private native `EstimateModularWeightedCost` now matches pinned libjxl's
`EstimateWPCost` for all five lossless e9 presets on the qualified M4 Pro
controls. Together with the [fixed-gradient estimator](modular-selection-cost.md),
this provides both cost primitives needed to select transforms and WP settings
before fitting a tree.

The [qualification ledger](modular-weighted-cost-results.json) records exact
costs, rankings, overflow cases, storage bounds and coding regressions.
Palette construction and complete native transform/scope selection still need
implementation and integration. This milestone changes no encoder default and
adds no rate or timing claim.

## Behavior and resource contract

[`selection_cost.h`](../src/codec/modular/selection_cost.h) exposes the five
reference parameter presets in candidate order, the estimator and its storage
planner. Modes outside `[0, 5)` are rejected before allocation. The estimator
uses borrowed, potentially strided planes, with the same view/population limits
as the gradient estimator. Empty planes contribute nothing.

Each channel starts with fresh WP history. The cost uses its weighted residuals,
34 contexts derived from the predictor's error property, and 4/2/0 HybridUint
symbols. Histogram entropy is accumulated as float in reference context/channel
order; extra bits are accumulated as `size_t` and converted for the final float
addition. This differs from the gradient proxy's integer/fractional truncation.
The two estimators share the qualified four-lane Shannon arithmetic. Native
code does not depend on Highway or libjxl.

History overflow and residual overflow have separate meanings in the reference.
History errors are narrowed modulo 2^32; the prediction is also narrowed before
checking the residual. A residual outside signed 32-bit range marks that
candidate unhealthy. The reference finishes the current channel and returns
max-float, preventing that candidate from winning on cost. The native estimator
does the same and exposes `residual_overflow` for attribution.

The existing predictor's checked `Update` behavior remains in use by coding and
training. A separate `UpdateWrapping` is used by this selection estimator. This
distinction matters: generated controls contain **22 finite-cost candidates**
that need wrapped history. Rejecting them merely because the checked history
update failed would not reproduce the reference.

The allocation-free, pixel-read-free planner bounds one live channel's five
two-row history arrays: **40 × (maximum nonempty width + 2) bytes**, or zero
for empty input. Arrays use preparation ownership and release before return;
the result retains no managed allocation. The fixed histogram scratch is
**17,544 bytes** on the stack. Borrowed inputs, stack objects and allocator
overhead are outside the managed bound; it is not an RSS bound.

Invalid inputs, admission failures and allocation failures preserve the prior
result. A successfully estimated unhealthy candidate returns an OK status and
max-float cost, rather than a partial failure. Its histogram cost covers prior
healthy channels and its extra-bit counter includes the rejected channel.

## Qualification

| Check | Result |
|---|---:|
| Generated cases per backend/build | 396 |
| Exact five-preset costs per backend/build | 1,980 |
| Exact rankings/tie orderings per backend/build | 396 |
| Residual-overflow rejections per backend/build | 523 |
| Directly compared wrapped history updates per backend/build | 72,463 |
| Finite-cost candidates requiring wrapped history per backend/build | 22 |
| Injected allocation failures per build | 50 |
| Native / research CTest entries per build | 6 / 10, all passed |
| Captured image/policy cases | 11 |
| Exact captured costs | 2,080 |
| Populated-stream rankings / empty-slot ties | 391 / 25 |
| Sanitizer captured cases identical to Release | 4 / 4 |
| Exact sanitizer costs | 445 |
| Maximum observed managed history peak | 34,560 bytes |
| Generated coding files unchanged per build | 648 / 648 |

The native NEON-entropy and forced-portable-entropy implementations both pass
Release and ASan/UBSan controls. The generated set retains the gradient study's
narrow, strided, flat, skewed, signed 24-bit, full-32-bit and larger RGBA cases.
Small fixtures compare every native prediction and error property directly
with reference WP state, including after a checked update rejects and wrapping
is required. Every cost and ranking is compared with the reference function.

Because `EstimateWPCost` is private to `enc_modular.cc`, the test build extracts
its body **verbatim** from pristine reference
`e8ff09762481785938d8e4e01333ed3917571161`. A whole-source hash guards extraction;
only its enclosing test namespace changes. The generated oracle retains the
reference compiler settings and calls the pinned histogram implementation.
The qualifier verifies the extracted text and compiler settings. Production
libraries contain only the native implementation.

Allocation sweeps cover all ten history allocations across two unequal-width
channels, for each of the five modes. They check ownership, atomic failure,
recovery, underplanned admission and released partial storage. Empty scopes
allocate nothing. Replay validation rejects changed costs, decompositions,
candidate order, winners, scope and resource claims.

The captured controls use the established five images in automatic and
buffered mode plus the automatic 16-bit RGBA ramp. These are already transformed
planes; they establish estimator/ranking parity on those inputs, not the
original encoder's full sequence of decisions. All captured candidates are
healthy; the generated tests cover rejection. The four-case sanitizer subset
covers RGB, palette, RGBA and true 16-bit input.

Gradient regressions still pass **2,372 costs and 104 RCT rankings** per generated
backend/build, plus **6,724 captured costs and 332 candidate sets**. The 648
generated files per build cover 108 cases across six entropy controls. All
match the previous prepared-stream milestone byte for byte and match across
Release/sanitizer builds, with independent reference decoding in the tests.

The validation freeze binds **5,257 artifacts**. Native code, the research
harness and the extracted WP body/inline state are instrumented in the
sanitizer build. Linked reference archives remain uninstrumented, reference
vptr checks are excluded, and unavailable macOS LeakSanitizer is disabled.
Managed history leak checks pass. Correctness builds/checks overlap, so their
durations are not performance evidence.

## Next integration boundary and reproduction

Next port reference palette construction, including implicit colors, ordering
and actual palette-size behavior. Then connect palette/RCT/WP and shared/local
scope selection to native preparation and one fit per selected scope. The
exhaustive new-learner, held-out rate, whole-stage timing and resource gates in
the [search plan](modular-reference-search-policy.md) remain unchanged.

The existing selection-cost native tests now include WP controls. Run those,
`modular_stream`, `modular_prepared_stream` and
`modular_prepared_stream_reference` in fresh Release/sanitizer builds. The
research runner's ten CTest entries cover the new `stream-wp-cost-check` mode.
`selection_cost_replay.py run --weighted` records its five-preset costs and
bounds; omit `--weighted` for the gradient regression control. Both use the
existing frozen coding captures and perform no new encodes.

The combined qualifier rechecks source/build bindings, both replay types,
generated tests, resource failures and the prior generated coding files:

```sh
python3 tools/modular_reference_learning/analyze_weighted_cost.py \
  --study build/reference-learning/weighted-cost-replay \
  --sanitizer-study build/reference-learning/weighted-cost-replay-asan \
  --gradient-study build/reference-learning/weighted-cost-gradient \
  --gradient-sanitizer-study build/reference-learning/weighted-cost-gradient-asan \
  --validation build/weighted-cost-validation-v2 \
  --baseline-validation build/prepared-entropy-validation \
  --output /tmp/gjxl-weighted-cost-rechecked.json
```

The output path must be new. Raw artifacts stay local; the compact report,
source extraction helper and qualification code are tracked.
