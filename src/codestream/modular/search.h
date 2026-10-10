// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/modular/storage_plan.h"
#include "codec/modular/tree_learning.h"
namespace gjxl::modular_internal {
// Allocation-free bounded transform policy enumeration. A zero palette size
// means ineligible; exact probing stops after the 257th distinct tuple.
[[nodiscard]] Status ProbePalette(PackedModularImageView input, uint16_t *colors);
[[nodiscard]] Status BuildTransformCandidates(Extent2D extent, PackedModularFormat format,
                                              uint16_t palette_colors,
                                              std::array<ModularCodingPolicy, 4> *out,
                                              size_t *count);
// Deterministic uniform sampling in canonical stream/channel/raster order.
// The retained candidates use the original uniform samples and bit-width cost.
// Optional entropy learning uses a separate, bounded stratified sample set.
[[nodiscard]] Status LearnModularPolicies(const ModularEncoderFrame &frame,
                                          const ModularStreamPlan &plan, uint8_t rct,
                                          ModularCodingPolicy *single, ModularCodingPolicy *split,
                                          ModularCodingPolicy *entropy = nullptr);
[[nodiscard]] Status ComputeModularSearchStoragePlan(Extent2D extent, PackedModularFormat format,
                                                     EntropyCodingMode mode,
                                                     ModularWorkflowStoragePlan *out,
                                                     size_t cpu_threads = 1);
} // namespace gjxl::modular_internal
