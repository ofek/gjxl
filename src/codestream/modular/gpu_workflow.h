// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/modular/storage_plan.h"
#include "codestream/modular/workflow.h"
#include "gpu/ops/modular_input_storage_plan.h"

namespace gjxl::modular_gpu_internal {
struct GpuModularWorkflowStoragePlan {
  ModularInputStoragePlan input;
  modular_internal::ModularWorkflowStoragePlan cpu;
  size_t peak_bytes = 0;
  bool operator==(const GpuModularWorkflowStoragePlan &) const = default;
};
// Shape-only, allocation-free complete-call bound for a prescribed policy with
// no additional transforms/search. GPU preparation/readback finishes and frees
// all device backing before CPU tokenization. The host image then occupies the
// existing CPU preparation bound. Peak = max(input.peak_bytes, cpu.working.peak_bytes).
// Existing managed output remains charged separately, as in the CPU workflow;
// callers must provide room for old output + this fresh-call reservation.
[[nodiscard]] Status ComputeGpuModularWorkflowStoragePlan(
    Extent2D extent, size_t row_stride_bytes, modular_internal::PackedModularFormat format,
    modular_internal::SampleByteOrder byte_order,
    const modular_internal::ModularEncodingOptions &options, const ModularInputDeviceLimits &limits,
    GpuModularWorkflowStoragePlan *out);

// Private qualification/experiment routes. Backend is already initialized and
// must expose GpuModularInputPreparation; failures never silently select CPU.
// Public backend routing is unchanged. Admission covers GPU and CPU phases,
// consumes an existing same-domain reservation if present, and precedes CPU
// participation and all sample allocations. Output replacement is atomic.
[[nodiscard]] Status
EncodeGpuModularImageOwned(GpuBackend &backend, modular_internal::PackedModularImageView input,
                           const modular_internal::ModularEncodingOptions &options,
                           codestream_internal::CodestreamBuffer *out);
[[nodiscard]] Status EncodeGpuModularImage(GpuBackend &backend,
                                           modular_internal::PackedModularImageView input,
                                           const modular_internal::ModularEncodingOptions &options,
                                           std::vector<uint8_t> *out);
} // namespace gjxl::modular_gpu_internal
