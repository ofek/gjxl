// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/modular/stream_plan.h"
#include "gpu/ops/modular_input_storage_plan.h"
#include "gpu/ops/modular_token_storage_plan.h"

namespace gjxl::modular_gpu_internal {
struct GpuModularTokenPlanOptions {
  Extent2D extent;
  size_t row_stride_bytes = 0;
  modular_internal::SampleByteOrder byte_order = modular_internal::SampleByteOrder::kLittleEndian;
  ModularTokenizationOptions tokenization;
  bool operator==(const GpuModularTokenPlanOptions &) const = default;
};
struct GpuModularTokenStoragePlan {
  ModularInputStoragePlan input;
  ModularTokenEligibility eligibility;
  modular_internal::ModularFrameGeometry geometry;
  modular_internal::ModularStreamStoragePlan streams;
  ModularTokenStoragePlan tokens;
  // Canonical streams/slices plus GPU slice descriptors and host token ranges.
  resource_budget_internal::HostStorageBound host_layout;
  // Full candidate overlap through the composed unpack/token completion fence:
  // upload + planes + token output + dispatch + staging + canonical host layout.
  size_t generation_peak_bytes = 0;
  // Token arena + host layout after upload/planes/dispatch/staging are released.
  size_t retained_bytes = 0;
  bool operator==(const GpuModularTokenStoragePlan &) const = default;
};
// Allocation-free on success; validates policy, packed geometry, canonical IDs,
// arithmetic and supplied limits without pixels or backend creation. Null limits
// mean arithmetic-only planning; a backend must supply both actual limit sets.
// This bounds preparation, NOT the complete encoder. Continuation must add its
// modeling/emission work to retained token/layout backing exactly once and take
// the maximum with generation/assembly/publication phases. Old outputs, caller
// input, diagnostics, driver/setup memory and inline objects are excluded.
// Failed planning preserves *out.
[[nodiscard]] Status
ComputeGpuModularTokenStoragePlan(const GpuModularTokenPlanOptions &options,
                                  GpuModularTokenStoragePlan *out,
                                  const ModularInputDeviceLimits *input_limits = nullptr,
                                  const ModularTokenDeviceLimits *token_limits = nullptr);

struct GpuModularTokenPlan {
  GpuModularTokenPlanOptions options;
  GpuModularTokenStoragePlan storage;
  // Final source/RCT/policy metadata, matching EncodePreparedModularFrame.
  codec_internal::ImageMetadata metadata;
  modular_internal::ModularStreamPlan layout;
  resource_budget_internal::ManagedVector<ModularTokenSlice> slices;
  resource_budget_internal::ManagedVector<ModularTokenRange> ranges;
  GpuModularTokenPlan() = default;
  GpuModularTokenPlan(const GpuModularTokenPlan &) = delete;
  GpuModularTokenPlan &operator=(const GpuModularTokenPlan &) = delete;
  GpuModularTokenPlan(GpuModularTokenPlan &&) noexcept = default;
  GpuModularTokenPlan &operator=(GpuModularTokenPlan &&) noexcept = default;
};
// Materializes only metadata, never samples/tokens/device backing. Uses managed
// kPreparation allocations, including default tracking without an ambient scope.
// Caller admits the full operation before building, using host_layout's bound.
// Canonical BuildModularStreamPlan ordering and IDs are retained, including empty
// streams. Each range indexes the compact value/context arrays by ELEMENT, not
// byte; slices partition [0, token_count) without gaps or overlap. Preserve this
// candidate unmodified while its descriptors/views are consumed. On allocation
// or validation failure, keep the previous plan and its charges unchanged.
[[nodiscard]] Status
BuildGpuModularTokenPlan(const GpuModularTokenPlanOptions &options, GpuModularTokenPlan *out,
                         const ModularInputDeviceLimits *input_limits = nullptr,
                         const ModularTokenDeviceLimits *token_limits = nullptr);
} // namespace gjxl::modular_gpu_internal
