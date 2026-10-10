// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/tree.h"

namespace gjxl::modular_internal {
inline constexpr size_t kMaximumTrainingSamples = 4096;
struct TrainingSample {
  PredictionProperties properties;
  std::array<int32_t, 14> residuals;
};
// Retained bit-width learner: these candidates remain available to search.
[[nodiscard]] Status LearnDecisionTrees(std::span<const TrainingSample> samples,
                                        DecisionTree *single, DecisionTree *split);
// Deterministic histogram-based learner, bounded by kMaximumTreeLeaves. Its
// cost is an integer estimate, not a serialized size; the caller must compare
// complete codestreams. No entropy-writer or stream-plan dependencies.
[[nodiscard]] Status LearnEntropyTree(std::span<const TrainingSample> samples, DecisionTree *out);
[[nodiscard]] Status
ComputeTreeLearningStorageBound(resource_budget_internal::HostStorageBound *out);
[[nodiscard]] Status MakeFixedWeightedTree(uint32_t bits, DecisionTree *out);
} // namespace gjxl::modular_internal
