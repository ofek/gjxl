# Modular GPU handoff after P2.6

The native scalar implementation and its prescribed decisions are the oracle.
Start with [the frozen fixture protocol](../tools/modular_oracle/README.md),
[architecture](modular-architecture.md), and [qualification](modular-p2.6.md).
The public API currently resolves automatic Modular execution to CPU; explicit
Metal/CUDA requests remain unsupported/unavailable. Private Metal unpack/RCT,
owning preparation, explicit host readback and a private GPU-assisted complete
encoder path are implemented for qualification. Public backend routing remains
unchanged pending end-to-end performance evidence.

## Implemented capability boundary

| Area | Phase 2 contract |
| --- | --- |
| Input | Gray/RGB/unassociated RGBA; unsigned 8/16-bit sRGB; explicit 16-bit byte order; checked offsets, strides and backing; exact invisible RGB |
| Frame | One regular final frame, one pass; 256-pixel groups and 2048-pixel DC groups; no XYB, filtering, resampling, responsive passes or animation |
| Channels | Signed int32 planar samples; separate per-channel dimensions/shifts/roles; metadata prefix; up to 64 transformed channels |
| Predictors | IDs 0–13, validated weighted parameters, integer arithmetic; reset at every stream/channel boundary |
| Tree | One prescribed global tree; properties 0–15; at most 31 nodes/16 leaves; exact signed offsets and positive multipliers; no local or previous-channel trees |
| Transforms | Optional first RCT, types 0–41; up to eight explicit transforms; exact palettes up to 256 colors over 1–4 channels; explicit scalar horizontal/vertical squeeze over 1–19 channels with nonempty residuals |
| Entropy | Shared native prefix or ANS, resolved HybridUint/context maps; canonical stream and section order; no LZ77 |
| Policy | Default gradient leaf and identity transforms; opt-in bounded CPU search, at most 23 color/9 gray complete candidates, including [context learning](modular-context-learning.md) and [bounded RCT selection](modular-rct-search.md); earlier candidates win ties |
| Execution | Serial oracle and up to four participants over independent streams; shared domain admission/CPU limits, atomic publication, joined workers on failure |
| Entry points | Installed C++ `gjxl/modular.hpp`, sized C options, Rust safe wrapper, explicit integer PGM/PPM/PAM CLI route |

Unsupported features require a new CPU contract and independent qualification
before GPU work. Preserve the separation between whole-image Modular policy
and VarDCT's use of shared integer-stream mechanisms.

## First slice: operation and storage contract

The private [input operation](../src/gpu/ops/modular_input.h) and
[storage planner](../src/gpu/ops/modular_input_storage_plan.h) define unpacking
plus one prescribed RCT. They build with both GPU backends disabled and are not
installed headers. The Metal implementation below consumes this contract;
workflow dispatch remains unchanged.

`GpuModularInputPreparation` is an optional capability of an existing backend.
It accepts all six packed integer formats and explicit byte order, with RCT 0
for grayscale or 0–41 for color. Successful preparation is synchronous and
returns a `PreparedGpuModularInput` owning completed, immutable int32 device
planes and their resource charges. Plane views preserve source extents, channel
roles and zero shifts; there are no metadata channels. Alpha and invisible RGB
follow the scalar oracle. The backend must outlive the owner, and all consumers
must complete before the owner is destroyed. Failure preserves prior output and
drains submitted work before releasing backing. Host readback is a separate step.

`ComputeModularInputStoragePlan` accepts shapes without pixel backing. The CPU
view validator and GPU planner share `ComputePackedModularInputSize`, including
format, area, byte-order, stride, pointer-range and final-row rules. Input
offsets are represented by the caller's subspan. The planner reports:

| Backing / phase | Contract |
| --- | --- |
| Upload | `(height - 1) * row_stride_bytes + meaningful_row_bytes`; includes inter-row padding, excludes trailing padding |
| Device result | One arena of dense int32 planes with 256-byte aligned, disjoint offsets |
| Preparation peak | Upload plus device result; upload is released after completion |
| Optional readback peak | Device result plus a fresh managed `ModularImage`, including channel records |
| Component peak | Maximum of preparation and readback, excluding prior outputs and downstream work |

Successful planning allocates no backing and initializes no device. Supplying
backend limits additionally checks each extent axis, input stride, and the
upload/output allocations separately. Limits are inclusive and nonzero. Missing
limits mean arithmetic-only planning; runtime implementations must supply their
actual limits before allocation. Invalid requests leave the old plan unchanged.
Callers still compose old-output overlap, diagnostics and the remaining encoder
phases into complete-call admission.

The CPU-only `modular_gpu_input` test covers the format/RCT metadata matrix,
odd offsets/strides, exact backing ends, plane ranges and alignment, device-limit
boundaries, invalid inputs, large shape-only plans, allocation-free planning and
actual managed host-destination capacity. It is included in native Modular CI.
The qualification below covers arithmetic, borrowed-buffer submission and owning
preparation/readback, complete-call admission and GPU-assisted files.

## First slice: fused Metal kernel

[`modular_input.metal`](../src/gpu/metal/kernels/modular_input.metal) runs one
thread per pixel in an 8-by-8 threadgroup. It loads packed bytes directly,
performs the prescribed permutation/RCT, and writes disjoint dense int32 planes.
Byte-wise 16-bit loads preserve odd offsets/strides and both endian orders.
Alpha passes through and invisible RGB is retained. The source-bound proof beside
the kernel justifies signed 32-bit arithmetic; negative odd differences use
explicit floor-halving. A shared host/shader parameter header uses 64-bit byte
addresses and planar element offsets, independently of the 32-bit grid axes.

[`SubmitMetalModularInput`](../src/gpu/metal/metal_modular_input.h) is the private
borrowed-buffer launch boundary. It validates the shared storage plan against
the device's maximum buffer length, checks native backend ownership, addressed
ranges and input/output overlap, then submits one kernel. It allocates no sample
backing. Buffer offsets are carried in parameters, so binding alignment does not
restrict a packed source's byte offset. File-loaded and embedded shader libraries
include the kernel; its pipeline is created with the backend.

The caller owns upload/output buffers and must wait before accessing output or
releasing either buffer, including on completion failure. A rejected launch
preserves buffers and an existing submission; a device error leaves output
contents unspecified. The owning preparation capability wraps this primitive
with the completion and publication rules below. No public routing changes.

`metal_modular_input` requires a working Metal device (no skip/fallback). It
checks all formats/endian orders/RCTs against scalar preparation, plus pinned
libjxl when `GJXL_MODULAR_ORACLE_BUILD` is configured. Fixtures include source
extrema and negative odd differences, transparent colors, unaligned input,
exact final-row ends, one-dimensional/odd extents and 255/256/257 boundaries.
Sentinels cover inter-plane padding and surrounding output. Tests also check
invalid/native/foreign buffers, overlap, submission/completion faults, repeated
waits, resource charges, concurrent independent calls, and both library paths.
Native CI's Metal build and test stages include this target; CPU-only builds
continue to exercise the contract/planner without a Metal dependency.

## First slice: ownership, resource charges and failure handling

[`PrepareGpuModularInput`](../src/gpu/ops/modular_input.cpp) queries the optional
backend capability and prepares a private candidate owner. Metal supports the
operation in both library paths. It validates the packed backing and device
limits before allocation, allocates a fresh minimal upload and a separate output
arena, copies only the addressed source span, then submits the fused kernel.
Only successful completion can publish the candidate. The submission and upload
are released before publication, leaving exactly one charged device backing.
There is no cache or intermediate host sample allocation.

[`ReadbackGpuModularInput`](../src/gpu/ops/modular_input.h) separately validates
the backend and all dense, disjoint plane views, allocates a fresh managed
`ModularImage`, and copies through `GpuBackend::CopyDeviceToHost`. It requires
completed samples and never assumes a device buffer is CPU-accessible. The host
candidate is published only after every plane copy succeeds. Its storage remains
valid and charged after the device owner and producing reservation are destroyed.

Both operations inherit the caller's reservation and use `kPreparation` resource
charges. They do not reserve again or transfer charges from a prior output.
Without an admitted context, standalone private calls use the existing unbounded
default ledger; this is accounting, not full-workflow admission. A caller planning
preparation plus readback reserves the maximum of upload-plus-device and
device-plus-host capacities, with previous device/host outputs charged separately.
An insufficient reservation fails without escaping its capacity. The complete
workflow below composes these phases with the remaining encoder work.

A scope guard drains any committed work on status failures or exceptions before
destroying either backing or its tickets. The internal kernel seam requires a
submitted handle even when submission reports a later error. Metal's cleanup
fence waits without constructing a `Status` or allocating error diagnostics, so
an exception from normal completion handling cannot bypass the fence. Normal
waits and cleanup fences suspend and then restore ambient CPU participation.
Allocation, launch, completion and readback failures preserve prior outputs.

`modular_gpu_preparation` is a CPU-only lifecycle test using a backend double.
It checks exact upload/readback peaks, retained-owner charges, old/new output
overlap, undersized reservations, standalone accounting, allocation/upload/
submission/completion/copy failures and exceptions after submission or during
waits. It detects destruction while work still borrows a backing and is included
in native Modular CI. It does not qualify kernel arithmetic.

The physical `metal_modular_input` matrix exercises both raw dispatch and owning
preparation/readback for every fixture, comparing every sample with the scalar
oracle and configured pinned libjxl. It checks actual phase peaks, independent
host lifetime, failure-atomic replacement, CPU suspension, concurrent calls and
both library paths. Qualification artifacts for step 3 are retained under
`build/modular-preparation-qualification/`, using committed CPU sources to keep
concurrent CPU policy edits outside this slice's evidence.

## First slice: private encoder continuation

[`EncodeGpuModularImage` and `EncodeGpuModularImageOwned`](../src/codestream/modular/gpu_workflow.h)
accept an already initialized backend with the preparation capability. They
support all six input formats, explicit byte order, prescribed RCT and prescribed
CPU tree/predictor/entropy decisions. Search and additional transforms are
rejected before allocation or submission. Explicit failure is returned without
CPU fallback. These are private experiment/qualification entry points; the
installed API, automatic backend choice and CLI retain their existing behavior.

After the synchronous GPU preparation, the workflow reads back a fresh managed
image and releases the device owner. `ModularEncoderFrame::AdoptPreparedRct`
validates the source shape and channel roles, reconstructs canonical metadata,
and moves that image into a completed frame without re-unpacking, repeating RCT,
or copying samples. It is a private handoff requiring already qualified sample
provenance, not a general unchecked transformed-input API.

The CPU workflow and GPU workflow use the same
[`EncodePreparedModularFrame`](../src/codestream/modular/prepared_encoder.cpp)
continuation: canonical stream planning, CPU prediction/tokenization, model
selection, entropy emission, section assembly and final file construction.
Samples are released after tokenization and before modeling, matching the
existing CPU plan. Prescribed decisions and serialization therefore have one
implementation. The scalar unpack/RCT implementation remains the oracle.

`ComputeGpuModularWorkflowStoragePlan` is allocation-free and composes the fresh
call as `max(input.peak_bytes, cpu.working.peak_bytes)`. The input peak includes
both upload-plus-device and device-plus-host phases. The CPU peak already includes
the host image, which becomes the continuation's input. Releasing device storage
before tokenization is required for this bound. Complete memory admission occurs
before CPU participation or GPU sample allocation. Nested same-domain calls share
the outer reservation; foreign domains are rejected. Existing managed output
remains charged separately, so replacement requires capacity for the old result
plus the fresh-call reservation. Public vector output is published and uncharged
only after every fallible stage succeeds.

`modular_gpu_workflow` checks shape-only phase composition (including an
upload-dominated padded input), adoption without copying, early rejection and
admission ordering in CPU-only builds. `metal_modular_workflow` compares complete
files byte-for-byte with scalar encoding and, when the pinned oracle is configured,
independently decodes every file to the original integers. The matrix covers all
formats, endian orders and RCTs, both coders, every predictor, a weighted split
tree, alpha/invisible RGB, clipped groups and DC-group boundaries, and serial/
parallel CPU continuations. It also checks memory/CPU caps, nested and concurrent
calls, owned-output replacement, GPU faults and every managed host allocation
boundary through final publication. File-loaded and embedded libraries are both
exercised. Files, case ledger and build/test provenance are retained under
`build/modular-workflow-qualification/` using committed CPU policy sources.

The local M4 Pro checkpoint passed 797 complete CPU/GPU byte comparisons,
6,735,280 original-sample comparisons through the pinned decoder and 786 injected
workflow failures. CPU regression gates passed, including all 286 frozen
stage/file artifacts and the public API backend checks. ASan/UBSan passed the
GPU workflow with the repository's existing `metal-ubsan.supp` exclusion for
null calls inside `third_party/metal-cpp`. The unsuppressed run stops in the
vendor `NSObject.hpp` wrapper during backend initialization; its diagnostic is
retained separately. Native code remains instrumented; the pinned libjxl
binaries are not rebuilt with sanitizers.

[Step 5 performance qualification](modular-gpu-performance.md) includes upload,
synchronization, readback, the CPU continuation and publication. The M4 Pro
32-case pilot found no demonstrated complete-call GPU win: 30 cases favored CPU
and two noise cases were inconclusive. Keep automatic routing on CPU and use
the private path to qualify a larger resident preparation/prediction boundary.

## Proposed first operations

Introduce private operation contracts under `src/gpu/ops/`, implementations in
the existing Metal/CUDA directories, and orchestration in
`src/codestream/modular/`. Keep algorithms and the oracle under
`src/codec/modular/`. Do not make GPU types part of the installed Modular API.

1. Start with exact integer unpacking plus a prescribed RCT. This is a compact
   qualification unit with independent per-pixel arithmetic and clear bounds.
   Measure upload/readback as part of the experiment; its small CPU contribution
   alone does not justify automatic GPU dispatch.
2. Add prescribed nonweighted prediction/property/tree/residual generation over
   immutable transformed planes. This has more potential work to amortize
   transfer costs. Compute row-boundary and previous-gradient properties exactly;
   avoid introducing a dependency on a previous thread's execution order.
3. Evaluate fixed-model symbol mapping and histogram accumulation, followed by
   explicit squeeze operations where supported by profiling. Keep model search,
   normalization, tree learning, palette dictionary construction and final
   assembly on CPU initially. GPU palette lookup may consume an already sorted
   CPU dictionary; it must preserve exact indices and zero-fill capacity.
4. Treat weighted prediction as a separate design task. Its evolving error rows
   and updates are order-dependent. Parallelize independent stream/channel
   instances first; any finer schedule needs an exact state-equivalence proof
   and tests before performance comparisons.

The baseline shows different bottlenecks by policy and corpus: weighted
tokenization matters for the prescribed screen/alpha/photo cases, global model
optimization matters for noise, and search adds substantial training and repeated
candidate work. Transforms are a convenient first correctness unit, not a claim
that transform acceleration alone materially speeds complete encoding. Repeat
measurements on target hardware; the recorded Windows x64-emulation timings
cannot select Metal/CUDA dispatch thresholds.

## Required operation contract

| Boundary | Required specification and gate |
| --- | --- |
| Inputs | Immutable borrowed planes, explicit extents/element strides/shifts/roles and metadata count; canonical stream ID and slice origin; validated prescribed transforms/tree/WP/model; no implicit RGB, padding or float conversion |
| Arithmetic | Signed int32 storage, sufficiently wide signed intermediates and exact rounding/division; reject unrepresentable values before publication; no saturation, reassociation or reduced precision |
| Outputs | Disjoint owned plane/token ranges in canonical order; exact context/value pairs; stable mapping back to source channel and stream; no pointers into temporary readback storage |
| Memory | Checked device and host capacities, staging/readback/scratch and old/new-output overlap included in admission; retain every owner until its last fence completes; distinguish reserved capacity from observed backing |
| Synchronization | State resets per stream/channel; explicit dependencies between transforms and between WP updates; section assembly only after all producers complete; shared-domain scheduling and GPU lifetime rules |
| Failure | Allocation/launch/device/readback errors preserve caller output; drain all submitted work and release tickets; explicit backend requests report failure; any permitted automatic fallback starts from intact input |
| Equality | Compare every transformed sample, property/prediction, leaf, token, population and stream bit count for fixed decisions, then complete bytes and pinned-decoder samples; timing or size improvements cannot waive equality |

Extend the frozen scalar capture to expose additional state when a new GPU
boundary needs it. Keep compact deterministic fixture hashes separate from
policy choices and benchmark measurements. Do not use optimized CPU output as
the only reference: retain pinned stage checks and independent decoding.

Before enabling dispatch, qualify all six input formats, odd strides and endian
layouts, one-dimensional/odd/clipped groups, signed extrema, transparent colors,
empty streams, palette metadata, shifted DC groups, both coders, and resource and
concurrency failures. Reuse the broader P2.0–P2.5 matrices in addition to the
compact frozen corpus. Run sanitizer and native target-architecture gates where
available, then measure end-to-end latency and concurrent throughput with
transfers, host decisions and publication included. Keep the CPU-only installed
build and serial oracle usable without a GPU or libjxl runtime.
