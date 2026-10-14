// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "core/managed_allocator.h"
#include "gpu/metal/kernels/modular_token_params.h"
#include "gpu/ops/modular_tokenization.h"

namespace gjxl::metal_internal {
using ModularTokenDiagnostic = gjxl_modular_tokens::Diagnostic;
// Private qualification hook, excluded from production storage/performance.
// Uses the generic token kernel's exact neighborhood/property/prediction/lookup
// code, writes a guarded diagnostic arena, checks token padding/trailing guards,
// and copies traces only after completion/status/guard validation. Allocation
// uses ambient managed tracking; old diagnostic output is preserved on failure.
[[nodiscard]] Status
DiagnoseMetalModularTokens(GpuBackend &backend,
                           const modular_gpu_internal::PreparedGpuModularInput &input,
                           const modular_gpu_internal::ModularTokenizationOptions &options,
                           std::span<const modular_gpu_internal::ModularTokenSlice> slices,
                           resource_budget_internal::ManagedVector<ModularTokenDiagnostic> *out);
} // namespace gjxl::metal_internal
