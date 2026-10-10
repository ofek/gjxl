// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/modular/gpu_token_plan.h"
#include "codestream/modular/tokenization.h"

namespace gjxl::modular_gpu_internal {
// Fresh stream-view metadata only. No token arena charge/copy and no full-call
// admission. Add this bound to the already-live plan and completed token owner;
// any old output remains charged separately. Failure preserves *out.
[[nodiscard]] Status
ComputeGpuModularTokenHandoffStorageBound(size_t streams,
                                          resource_budget_internal::HostStorageBound *out);

// Private completed-token boundary. Plan must be the unmodified canonical
// BuildGpuModularTokenPlan result used by the producer, which proves token
// contents and successful completion/status validation. Checks policy, counts,
// range coverage and borrowed view sizes before allocating stream metadata.
// Does not scan tokens, map device views, submit work or fall back to a copy.
// Unsupported host borrowing is propagated. Success consumes backing and
// atomically replaces *out. Any failure preserves both, including allocation
// failure; passing out->gpu_backing as backing is invalid. The backend outlives
// the new owner. Plan itself is not borrowed by the resulting token views.
// Views use managed kSerializer storage (also tracked without an ambient scope).
[[nodiscard]] Status AdoptGpuModularTokens(const GpuModularTokenPlan &plan,
                                           std::unique_ptr<PreparedGpuModularTokens> &&backing,
                                           modular_internal::PreparedModularTokens *out);
} // namespace gjxl::modular_gpu_internal
