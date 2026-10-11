// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/tree_learning.h"

namespace gjxl::modular_internal {
// Internal implementation; callers validate budgets and catch allocation errors.
[[nodiscard]] Status LearnQuantizedEntropyTree(std::span<const TrainingSample> samples,
                                               DecisionTree *out, TreeLearningOptions options,
                                               size_t population,
                                               TreeLearningCheckpoints *checkpoints);
[[nodiscard]] Status
ComputeQuantizedTreeLearningStorageBound(resource_budget_internal::HostStorageBound *out,
                                         TreeLearningOptions options);
} // namespace gjxl::modular_internal
