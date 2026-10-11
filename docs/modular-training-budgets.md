# Configurable CPU Modular training budgets

This is the storage and correctness foundation for expanding the learner, based
on the qualified palette/context implementation `f4300f4`. The default remains
4,096 entropy-training samples and at most 16 learned leaves. Sampling,
split scoring, predictor choices, transform enumeration, and complete-file
candidate selection retain their existing algorithms. The retained bit-width
candidates always use their original 4,096-sample budget.

The experimental controls are private CPU workflow options; the public API and
CLI retain their existing behavior. For example:

```cpp
#include "codestream/modular/workflow.h"

gjxl::modular_internal::ModularEncodingOptions options;
options.search = true;
options.entropy = gjxl::EntropyCodingMode::kAns;
options.learning = {.maximum_samples = 65536, .maximum_leaves = 64};
// EncodeModularImage(input, options, &output), using the private integer view.
```

`LearnEntropyTree`, both `LearnModularPolicies` overloads,
`ComputeTreeLearningStorageBound`, and `ComputeModularSearchStoragePlan` accept
matching `TreeLearningOptions`. Callers must pass the same options to planning
and execution. Budgets are upper bounds: small inputs use fewer samples, and
nonpositive split gains stop growth before the leaf budget is exhausted.

## Representation, arithmetic, and decoder limits

The supported research ceiling is 1,048,576 training samples and 8,192 leaves
(16,383 nodes). The sample and leaf budgets are independent. Zero and
out-of-range settings fail validation before managed allocation, preserving
outputs. These ceilings are safety bounds, not selected compression policies or
claims that the current learner efficiently exploits those budgets.

Histogram counters and sample indices are 32-bit. The Q12 count-log-count table
and its multiplication use signed 64-bit arithmetic. This removes the old
65,536-sample counter wrap and fixed-point table overflow; 65,537 samples also
exercise indices that cannot fit the old 16-bit representation. The configured
sample ceiling keeps every learning cost comfortably within signed 64-bit range.

Child indices, breadth-first node order, and leaf contexts are 16-bit. Trees up
to 31 nodes remain inline; larger node arrays and tree-token headers use managed,
exact-size backing. Policy copies retain only live nodes, so a small result of a
large-budget attempt becomes inline again. Equality compares active nodes, not
unused training capacity. Learning scratch allocates for the sample count and
leaf budget rather than for the maximum supported configuration.

Two pinned-libjxl decoder limits also apply:

- The global-tree node bound is based on `1024 + source_samples / 16`, subject
  to the decoder's larger absolute cap. Native planning uses a conservative
  complete-binary-tree leaf bound and the stricter native representation cap.
  Source samples count original channels and pixels, not transformed channel
  storage or training samples. Search clamps its effective leaf budget to this
  bound; oversized prescribed trees are rejected before admission.
- Tree depth is at most 2,047, counting the root. The learner stops splitting at
  that depth. Validation rejects deeper or contradictory trees without recursion.

Validation remains allocation-free, using bounded parent/visit/layout arrays
and checking ancestral property ranges. Its worst-case work is O(nodes × depth).
These inline control arrays are excluded from managed-backing accounting, as
other inline records are; they do not grow with input image dimensions.

## Admission, publication, and GPU boundary

Storage plans cover training samples, all learner workspaces, the returned tree,
large policy copies, tree-token headers, entropy work, and overlap with the
retained winning file. Search plans conservatively include sixteen large-policy
backings for retained candidates and replacement temporaries. Direct learner
bounds exclude caller-owned input samples and any old output tree; callers
must account for those separately when they share a reservation.

The workflow planner distinguishes leaf contexts from entropy clusters. The
existing prefix partition still has at most 32 clusters, including when it feeds
ANS. Larger trees therefore work with the existing entropy writer without
pretending that each leaf requires its own encoded histogram. This change does
not improve clustering or increase its capacity.

CPU workflow options are borrowed until admission; large policies are not copied
before the resource checks. Resolved-policy profiling copies complete before
publishing output. The same ownership correction applies to the GPU-input / CPU-
token continuation. Allocation failure preserves the caller's encoded output.

Resident GPU token generation retains its explicit 31-node limit and unchanged
32-byte node descriptor ABI. A larger valid CPU tree returns `Unsupported` from
GPU token eligibility without allocating or altering its output. Unused learning
capacity does not make an otherwise small tree GPU-ineligible. The separate GPU
input-preparation path can still hand samples to CPU tokenization with a larger
prescribed tree; this does not imply resident GPU support for those trees.

## Qualification

The structured evidence record is
[modular-training-budgets-results.json](modular-training-budgets-results.json).
Raw binaries, logs, source hashes, input/output hashes, and complete files are
retained in `build/training-budgets/`; sanitizer artifacts are in
`build/training-budgets-sanitize/`.

All 17 selected Release CTest suites passed, including both frozen oracle gates
(286 artifacts each), policy/transform/conformance checks, public API/CLI checks,
and the CPU-side GPU contracts. All 64 final legacy files are byte-identical to
the saved baseline and independently decoded sample-exactly. Four ASan/UBSan
suites passed with native code built at `-O1 -g`; pinned libjxl libraries are not
instrumented. An earlier unoptimized Debug sanitizer attempt reached the
300-second timeout without a sanitizer diagnostic; that incomplete run is
retained under `build/training-budgets-asan/` and excluded from passing evidence.

The dedicated budget tests cover invalid limits, allocation-free planning,
65,536/65,537-sample histograms, deterministic 16/64/196-leaf learning,
8,192-leaf representation validation, the depth limit, widened node/context
indices, GPU rejection, and large-buffer-to-inline policy copies. Complete-file
checks exercise 64/256/1,024-leaf prescribed trees, larger-budget search,
serial/parallel equality, exact admission, undersized admission, and independent
sample-exact decoding. Larger-budget training also runs after palette and Squeeze
transforms, preserving the retained learner's policies and original RGBA bytes.

All six fresh learner backing allocations are fault-injected. A separate test
exhaustively injects every managed allocation failure in four small complete
workflows (prescribed/search × prefix/ANS), including resolved-policy profiling:
847, 988, 4,281, and 5,926 failure points, respectively (12,042 total).
The existing broader workflow fault suite is sampled with shard 0 of 64; that is
not a repeat of its earlier exhaustive 44-case qualification.

Legacy compatibility compares all four existing default/search × prefix/ANS
configurations on the same 16 inputs against the saved `f4300f4` files. This is
a byte-identity and correctness gate, not a fresh compression-gain experiment.
Higher-budget rate attribution and a reference-compatible split scorer belong
to the next milestone. No speed, rate-parity, or GPU-performance claim is made
by this foundation change.

To reproduce the main checks with the clean pinned libjxl build:

```sh
cmake -S . -B build/training-budgets -DCMAKE_BUILD_TYPE=Release \
  -DGJXL_ENABLE_METAL=OFF -DGJXL_MODULAR_ORACLE_BUILD=/path/to/pinned-libjxl-build
cmake --build build/training-budgets -j 8
ctest --test-dir build/training-budgets -R '^modular_' \
  -E '^modular_workflow$' --output-on-failure
build/training-budgets/gjxl_modular_workflow_test --allocation-shard 0 64
```
