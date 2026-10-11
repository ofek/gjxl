// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/modular/gpu_token_handoff.h"
#include "codestream/modular/storage_plan.h"
#include "codestream/modular/workflow.h"

namespace gjxl::modular_gpu_internal {
struct GpuModularTokenWorkflowStoragePlan {
  GpuModularTokenStoragePlan generation;
  resource_budget_internal::HostStorageBound handoff;
  modular_internal::ModularWorkflowStoragePlan cpu;
  size_t handoff_peak_bytes = 0, peak_bytes = 0;
  bool operator==(const GpuModularTokenWorkflowStoragePlan &) const = default;
};
// Allocation-free complete-call admission. Generation includes overlapping
// upload/planes/tokens/dispatch/layout. Handoff additionally creates views.
// CPU continuation uses compact tokens once, and only canonical layout; GPU
// descriptors/ranges must be released before modeling. Old output is separate.
[[nodiscard]] Status ComputeGpuModularTokenWorkflowStoragePlan(
    Extent2D extent, size_t row_stride_bytes, modular_internal::PackedModularFormat format,
    modular_internal::SampleByteOrder byte_order,
    const modular_internal::ModularEncodingOptions &options,
    const ModularInputDeviceLimits &input_limits, const ModularTokenDeviceLimits &token_limits,
    GpuModularTokenWorkflowStoragePlan *out);

// Private qualification routes. Require both GPU input/token capabilities.
// One unpack/RCT+token submission/wait, no planar readback or CPU tokenization.
// Full admission precedes CPU participation and allocations; uses an existing
// same-domain reservation, preserves output on any failure, and never retries
// on CPU. Backend is initialized by the caller and outlives CPU consumption.
[[nodiscard]] Status
EncodeGpuModularTokensImageOwned(GpuBackend &backend,
                                 modular_internal::PackedModularImageView input,
                                 const modular_internal::ModularEncodingOptions &options,
                                 codestream_internal::CodestreamBuffer *out);
[[nodiscard]] Status
EncodeGpuModularTokensImage(GpuBackend &backend, modular_internal::PackedModularImageView input,
                            const modular_internal::ModularEncodingOptions &options,
                            std::vector<uint8_t> *out);
} // namespace gjxl::modular_gpu_internal
