# Modular GPU handoff after P2.6

The native scalar implementation and its prescribed decisions are the oracle.
Start with [the frozen fixture protocol](../tools/modular_oracle/README.md),
[architecture](modular-architecture.md), and [qualification](modular-p2.6.md).
The public API currently resolves automatic Modular execution to CPU; explicit
Metal/CUDA requests remain unsupported/unavailable. A private Metal unpack/RCT
kernel is implemented for qualification; owning preparation and encoder
integration remain subsequent steps.

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
| Policy | Default gradient leaf and identity transforms; opt-in bounded CPU search, at most 11 color/7 gray complete candidates, with earlier candidate winning ties |
| Execution | Serial oracle and up to four participants over independent streams; shared domain admission/CPU limits, atomic publication, joined workers on failure |
| Entry points | Installed C++ `gjxl/modular.hpp`, sized C options, Rust safe wrapper, explicit integer PGM/PPM/PAM CLI route |

Unsupported features require a new CPU contract and independent qualification
before GPU work. Preserve the separation between whole-image Modular policy
and VarDCT's use of shared integer-stream mechanisms.

## First slice: operation and storage contract

The private [input operation](../src/gpu/ops/modular_input.h) and
[storage planner](../src/gpu/ops/modular_input_storage_plan.h) define unpacking
plus one prescribed RCT. They build with both GPU backends disabled and are not
installed headers. No Metal/CUDA implementation or workflow dispatch is enabled
by this contract.

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
The kernel qualification below covers arithmetic and borrowed-buffer submission.
Owning preparation, admission across the full workflow and complete GPU-assisted
files remain subsequent qualification gates.

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
contents unspecified. The `PreparedGpuModularInput` capability is still disabled:
step 3 will add the candidate owner, synchronous completion, resource admission
and failure-atomic publication using this primitive. No public routing changes.

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
