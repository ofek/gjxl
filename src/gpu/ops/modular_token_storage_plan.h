// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "core/host_storage_bound.h"
#include "gpu/ops/modular_tokenization.h"

namespace gjxl::modular_gpu_internal {
inline constexpr size_t kModularTokenAlignment = 256;
struct ModularTokenStoragePlan {
  size_t token_count = 0, slice_count = 0;
  // One retained arena: values at zero, contexts at the next aligned offset.
  size_t values_bytes = 0, contexts_offset = 0, contexts_bytes = 0, output_bytes = 0;
  // Separate transient arena: slices at zero, then all 31 node slots, then a
  // uint32 status word. Regions start at 256-byte aligned offsets.
  size_t slices_bytes = 0, nodes_offset = 0, status_offset = 0, dispatch_bytes = 0;
  // Serialized dispatch upload; a backend may initialize shared storage directly
  // and use less, but must not allocate unplanned staging. Fresh exact vector.
  resource_budget_internal::HostStorageBound staging;
  size_t production_peak_bytes = 0; // output + dispatch + staging
  size_t retained_bytes = 0;        // output only; views do not duplicate charges
  bool operator==(const ModularTokenStoragePlan &) const = default;
};
// Shape-only, allocation-free on success, atomic output. Counts are supplied by
// the canonical codestream planner; nonzero slices <= tokens. Checks arithmetic,
// pointer ranges and individual buffer limits (when supplied), not pixel values.
// Excludes input backing, canonical host layout, old output, CPU entropy work,
// diagnostics, backend setup/driver memory and inline/control objects. The
// composed preparation planner adds input/upload/layout overlap. CPU continuation
// must retain output_bytes exactly once through joined entropy consumption.
[[nodiscard]] Status
ComputeModularTokenStoragePlan(size_t tokens, size_t slices, ModularTokenStoragePlan *out,
                               const ModularTokenDeviceLimits *limits = nullptr);
} // namespace gjxl::modular_gpu_internal
