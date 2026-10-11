// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/modular/storage_plan.h"
namespace gjxl::modular_internal {
// Internal handoff: plan/tokens have canonical provenance. Consumes either CPU
// tokens or completed GPU-owned split views; joins all entropy workers before
// releasing the owner, before assembly, also on failure after consumption.
// Argument validation failures leave tokens alone. Caller owns admission and
// atomic publication; GPU continuation needs its own composed storage bound.
[[nodiscard]] Status EncodeModularFrame(const codec_internal::ImageMetadata &metadata,
                                        const ModularStreamPlan &layout,
                                        PreparedModularTokens &&tokens, EntropyCodingMode mode,
                                        const ModularWorkflowStoragePlan &storage,
                                        codestream_internal::CodestreamBuffer *out,
                                        AnsSelectionMode selection = AnsSelectionMode::kExact,
                                        GpuBackend *ans_gpu = nullptr);
} // namespace gjxl::modular_internal
