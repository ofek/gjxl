// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/tree.h"
#include "codec/modular/tree_learning_cost.h"

namespace gjxl::modular_internal {
inline constexpr size_t kDefaultTrainingSamples = 4096;
inline constexpr size_t kDefaultTreeLeaves = 16;
inline constexpr size_t kMaximumTrainingSamples = 1 << 20;
enum class TreeLearningMode { kLegacy, kQuantized };
struct TreeLearningOptions {
  size_t maximum_samples = kDefaultTrainingSamples;
  size_t maximum_leaves = kDefaultTreeLeaves;
  TreeLearningMode mode = TreeLearningMode::kLegacy;
  bool operator==(const TreeLearningOptions &) const = default;
};
[[nodiscard]] inline Status ValidateTreeLearningOptions(TreeLearningOptions options) {
  if (!options.maximum_samples || options.maximum_samples > kMaximumTrainingSamples ||
      !options.maximum_leaves || options.maximum_leaves > kMaximumTreeLeaves ||
      (options.mode != TreeLearningMode::kLegacy && options.mode != TreeLearningMode::kQuantized))
    return Status::InvalidArgument("Invalid Modular tree learning budget");
  return Status::Ok();
}
// Powers of two from 16 through the representation ceiling, plus an early
// final tree in place of the first unreached checkpoint. Default storage is inline.
inline constexpr size_t kMaximumTreeCheckpoints = 10;
static_assert((size_t{16} << (kMaximumTreeCheckpoints - 1)) == kMaximumTreeLeaves);
struct TreeLearningCheckpoints {
  std::array<DecisionTree, kMaximumTreeCheckpoints> trees;
  size_t size = 0;
  bool operator==(const TreeLearningCheckpoints &) const = default;
  // Learner-internal: copies active nodes only; may throw a managed allocation
  // failure. Call after predictors are finalized at this stage of tree growth.
  [[nodiscard]] Status Capture(const DecisionTree &tree, bool final = false);
};
struct TrainingSample {
  PredictionProperties properties;
  std::array<int32_t, 14> residuals;
};
// Retained bit-width learner: these candidates remain available to search.
[[nodiscard]] Status LearnDecisionTrees(std::span<const TrainingSample> samples,
                                        DecisionTree *single, DecisionTree *split);
// Deterministic histogram-based learners, bounded by options.maximum_leaves.
// Quantized mode uses Weighted/Gradient, 256 global property bins and a
// sampling-aware split penalty. population=0 means all samples are observed.
// This is a native reference-inspired control, not a bit-exact libjxl learner. Its
// cost is an integer estimate, not a serialized size; the caller must compare
// complete codestreams. Optional checkpoints preserve the same growth path as
// independent lower-cap runs on these samples. Both outputs publish on success.
// No entropy-writer or stream-plan dependencies.
[[nodiscard]] Status LearnEntropyTree(std::span<const TrainingSample> samples, DecisionTree *out,
                                      TreeLearningOptions options = {}, size_t population = 0,
                                      TreeLearningCheckpoints *checkpoints = nullptr);
// Private original-learner entry point. The provider changes only histogram
// cost evaluation for the caller-selected thresholds. Provider errors propagate;
// both tree/checkpoint outputs still publish together only on complete success.
[[nodiscard]] Status LearnEntropyTreeWithCostEvaluator(
    std::span<const TrainingSample> samples, DecisionTree *out, TreeSplitCostEvaluator &evaluator,
    TreeLearningOptions options = {}, size_t population = 0,
    TreeLearningCheckpoints *checkpoints = nullptr);
// Allocation-free; bounds fresh learning scratch plus the returned tree. Input
// samples and any old output are caller-owned and must be admitted separately.
[[nodiscard]] Status
ComputeTreeLearningStorageBound(resource_budget_internal::HostStorageBound *out,
                                TreeLearningOptions options = {}, bool retain_checkpoints = false);
// Snapshot backings only, also needed while search encodes retained checkpoints.
[[nodiscard]] Status
ComputeTreeLearningCheckpointStorageBound(size_t maximum_leaves,
                                          resource_budget_internal::HostStorageBound *out);
[[nodiscard]] Status MakeFixedWeightedTree(uint32_t bits, DecisionTree *out);
} // namespace gjxl::modular_internal
