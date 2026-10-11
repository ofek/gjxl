// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "gpu/metal/kernels/modular_adaptive_params.h"
#include "gpu/ops/modular_adaptive_tokens.h"
namespace gjxl::metal_internal {
using AdaptiveModularDiagnostic = gjxl_modular_adaptive::Diagnostic;
// Qualification-only guarded trace, excluded from production storage/timing.
[[nodiscard]] Status DiagnoseMetalAdaptiveModularTokens(
    GpuBackend &backend, const modular_gpu_internal::PreparedGpuAdaptiveModularFeatures &features,
    const modular_internal::DecisionTree &tree,
    resource_budget_internal::ManagedVector<AdaptiveModularDiagnostic> *out);
} // namespace gjxl::metal_internal
