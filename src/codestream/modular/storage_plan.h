// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/entropy_storage_plan.h"
#include "codestream/modular/tokenization.h"

namespace gjxl::modular_internal {
struct ModularWorkflowStoragePlan {
  ModularFrameGeometry geometry;
  size_t participants = 1;
  size_t tokens = 0, streams = 0;
  size_t maximum_global_bits = 0, maximum_group_bits = 0;
  size_t maximum_codestream_bytes = 0;
  resource_budget_internal::HostStorageBound preparation, modeling, emission;
  resource_budget_internal::HostStorageBound assembly, publication, output, working;
  bool operator==(const ModularWorkflowStoragePlan &) const = default;
};
// maximum_tree_leaves optionally bounds future learned trees (including their
// storage capacity); zero uses the prescribed tree. Input policy backing and
// old outputs are caller-owned. The prefix-derived ANS model still has at most
// kMaximumPrefixClusters clusters, independently of the number of leaves.
// Shape-only, allocation-free; bounds fresh work and output. An existing managed
// result stays charged separately in its domain during replacement.
[[nodiscard]] Status ComputeModularWorkflowStoragePlan(Extent2D extent, EntropyCodingMode mode,
                                                       ModularWorkflowStoragePlan *out);
[[nodiscard]] Status ComputeModularWorkflowStoragePlan(Extent2D extent, PackedModularFormat format,
                                                       EntropyCodingMode mode,
                                                       ModularWorkflowStoragePlan *out);
[[nodiscard]] Status ComputeModularWorkflowStoragePlan(Extent2D extent, PackedModularFormat format,
                                                       EntropyCodingMode mode,
                                                       const ModularCodingPolicy &policy,
                                                       ModularWorkflowStoragePlan *out,
                                                       size_t cpu_threads = 1,
                                                       size_t maximum_tree_leaves = 0,
                                                       bool candidate_evaluation = false);
// Completed-token continuation only: replaces scalar token backing with the
// supplied retained arena capacity, and omits image preparation/tokenization.
// Includes canonical layout and fresh stream views, models, emission workers,
// assembly and publication. Caller separately accounts for producing the arena
// and any extra metadata still live during handoff. No charge is acquired here.
[[nodiscard]] Status ComputeModularTokenConsumerStoragePlan(
    Extent2D extent, PackedModularFormat format, EntropyCodingMode mode,
    const ModularCodingPolicy &policy, size_t completed_token_bytes,
    ModularWorkflowStoragePlan *out, size_t cpu_threads = 1);
} // namespace gjxl::modular_internal
