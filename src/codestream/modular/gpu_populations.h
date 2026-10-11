// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/modular/tokenization.h"
#include "gpu/ops/modular_populations.h"
namespace gjxl::modular_internal {
// Exact dense counts are worthwhile only above both fixed and shape-dependent
// crossovers. A null successful output explicitly selects the complete CPU path.
// Unsupported/small/wide cases are separately counted. Other errors propagate.
[[nodiscard]] Status PrepareModularPopulations(GpuBackend &backend,
    const PreparedModularTokens &tokens,
    std::unique_ptr<modular_gpu_internal::PreparedGpuModularPopulations> *out);
} // namespace gjxl::modular_internal
