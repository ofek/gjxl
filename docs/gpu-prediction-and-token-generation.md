# GPU-resident prescribed prediction and token generation

Status: proposed next Modular GPU implementation slice, following the
[unpack/RCT performance qualification](modular-gpu-performance.md).

The next slice should consume unpacked/RCT planes directly on the GPU and hand
completed token buffers to the existing CPU entropy encoder. Completed shared
token buffers and explicit ownership are part of the initial design: copying a
larger token representation back to CPU could consume the savings.

The saved profiles give this a more meaningful opportunity than unpack/RCT
alone. Preparation plus tokenization accounts for about 24% of serial 4K RGB8
encoding, or 20% with four CPU participants. Eliminating that work entirely
would imply ceilings of roughly 1.31× and 1.24×, respectively. These are upper
bounds derived from the saved stage profiles, not predicted speedups. GPU
execution, synchronization and CPU consumption still cost time, and CPU
modeling and entropy emission remain major costs.

```mermaid
flowchart LR
    A[Packed input] --> B[GPU: unpack and RCT]
    B --> C[Resident sample planes]
    C --> D[GPU: prediction, tree lookup, residual tokens]
    D --> E[Completed shared token buffers]
    E --> F[CPU: models, entropy coding, assembly]
```

## Scope

The completed slice should support all six existing packed input formats and
their prescribed RCTs, all 13 nonweighted predictors, and prescribed trees using
properties 0–14. Keep the current 31-node/16-leaf limits and exact leaf
offset/multiplier semantics. Both prefix and ANS use the same resulting tokens.

Weighted prediction, property 15, additional transforms and policy search remain
outside this slice. A supplied tree must satisfy the same eligibility checks
regardless of how it was selected. Tree learning, entropy-model selection,
entropy coding and codestream assembly remain on CPU. The new path remains a
private qualification entry point until complete-call performance evidence
supports a dispatch policy.

## 1. Define the operation, eligibility and storage contract

Specify immutable source planes, canonical stream/channel slices, the prescribed
tree, token ordering, ownership and failure behavior. Each sample produces
exactly one token, so the CPU can calculate every output offset before launch.
No GPU sizing pass, compaction or atomic append is needed. Reject unsupported
policies before allocation or submission.

Use compact `uint32` values plus `uint16` contexts, matching the existing
[entropy stream interface](../src/codestream/entropy.h). Preserve canonical
stream order, channel order within each stream, and row-major sample order
within each slice, including empty streams and their canonical IDs.

The storage plan must account for actual overlapping lifetimes: upload, sample
planes, token buffers, descriptors, status storage and the CPU continuation.
The composed path can keep upload, planes and tokens alive together until its
completion fence; reusing the previous slice's phase maximum without accounting
for this overlap is insufficient. Retained token backing remains charged through
CPU consumption. Existing output stays charged separately during replacement.

The first deliverable is the private operation/storage contract and CPU-only
eligibility, layout and planning tests. Keep GPU types out of installed APIs.

## 2. Establish the CPU consumer and ownership boundary

Adapt
[`PreparedModularTokens`](../src/codestream/modular/tokenization.h)
to carry a move-only backing owner alongside its stream views. Preserve the
scalar producer and use `EntropyTokenStreamView::Split` for GPU output. The CPU
continuation should consume the same values and contexts regardless of producer.

Metal's completed-buffer borrowing can avoid an explicit token memcpy. Publish
host-readable views only after successful completion and validation. The owner
must remain alive through model selection and every entropy worker, then release
before assembly. Borrowing a view must not duplicate its backing charge or imply
that arbitrary device buffers are CPU-accessible.

Qualify this ownership change independently before adding the shader. Verify
move/lifetime behavior, CPU output equivalence, replacement failure and resource
release through the existing
[`EncodeModularFrame`](../src/codestream/modular/frame_encoder.cpp) consumer.

## 3. Implement and qualify the Metal token kernel

Start with the prescribed gradient leaf, then extend to the full nonweighted
scope. Assign independent sample work to threads and write directly into
predetermined token ranges. The current
[scalar tokenization](../src/codestream/modular/tokenization.cpp),
[predictor arithmetic](../src/codec/modular/scalar_prediction.h) and
[tree properties](../src/codec/modular/tree.h) remain the oracle.

The main correctness details are:

- Neighbors and coordinates are slice-local, including group-edge substitutions.
  A neighboring sample outside the slice must not leak into prediction simply
  because it exists in the full resident plane.
- Property 8 uses the previous pixel's unclamped gradient, with zero at each row
  start. Recompute that gradient from immutable neighbors using the same edge
  rules; it does not require serial execution or a separate scan.
- Preserve predictor division rounding, select ties, tree branch direction and
  breadth-first leaf context numbering.
- Use sufficiently wide arithmetic for offsets and residuals. Check exact
  divisibility and signed range before applying the existing signed packing.

Data-dependent failures should set a checked device status and prevent
publication. Complete or drain submitted work before releasing any backing or
resource ticket on failure. A specialized gradient-leaf path may avoid generic
tree work while preserving exactly the same output contract.

## 4. Compose preparation and tokenization into the complete encoder

Feed tokens directly into `EncodeModularFrame`, bypassing host plane
reconstruction and CPU tokenization. Keep CPU tree serialization, metadata,
model decisions, stream headers and final publication identical for the same
prescribed policy.

For the performance path, encode unpack/RCT and token generation into one command
buffer with explicit GPU dependencies and one completion wait. Keep them as
separate kernels initially. Release upload and sample planes after completion;
retain only token backing and required host metadata through CPU consumption.
Preserve the existing standalone preparation contract.

Compose full-call admission with the new lifetimes, preserve atomic output
replacement and return explicit failures without an unplanned CPU retry.

## 5. Qualify correctness, resources and failures

Compare intermediate predictions, properties and leaf selections, then every
token against CPU. Require complete byte equality and independent decoding to
the original integer samples.

Cover all supported predictors/properties/RCTs, signed extrema, leaf offsets and
multipliers, transparent colors, narrow images, 255/256/257 boundaries,
DC-group ID changes and empty streams. Include both byte orders, odd input
offsets/strides, clipped groups, both entropy coders and serial/parallel CPU
continuations.

Extend allocation/submission/completion fault injection, owner replacement,
concurrent calls, memory caps and CPU-participant checks. Include CPU-only
builds, Metal validation and sanitizer gates. Preserve prior output on failure,
join all consumers before releasing token backing, and verify charges return to
their expected state.

## 6. Run paired end-to-end performance qualification

Compare CPU, the existing GPU-preparation path and the new GPU-token path using
identical prescribed decisions. Test one/four CPU participants and four-call
workloads under matching resource caps. Retain ordinary complete-call timings
separately from stage diagnostics, following the
[existing paired measurement protocol](../tools/modular_oracle/GPU_PERFORMANCE.md).

Add larger natural photographs and prescribed split trees to the existing
matrix. Include allocation, upload, synchronization, token handoff, CPU
continuation and publication. Record setup separately, preserve byte/resource
checks, and retain source, input, binary and raw measurement provenance.

As a proposed promotion bar, require a repeatable complete-call gain of at least
5%, intervals excluding parity, and no material concurrency regression within
the claimed eligibility range. Define the acceptance cases and regression
tolerance before collecting the qualification results. Compare against the
relevant four-participant CPU path as well as the serial oracle. A kernel
speedup alone does not qualify dispatch.

The saved performance baseline is revision `155e60e`; the planning checkout was
at `2cdb74c`, which adds CPU context learning. Freeze a new common revision for
all comparison paths when qualifying this slice, and preserve the older results
as evidence for their original revision.

Histogram generation can follow as a separate slice once these measurements
show whether modeling deserves GPU work. It is not required to establish this
token producer/consumer boundary.
