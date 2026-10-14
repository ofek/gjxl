// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/entropy_estimate.h"
#include "core/status.h"
#include <array>
#include <cstddef>
#include <span>

namespace gjxl::modular_internal {
inline constexpr size_t kTreeCostPredictors = 14;
inline constexpr size_t kTreeCostSymbols = 128;
inline constexpr size_t kTreeCostThresholds = 32;
inline constexpr size_t kTreeCostMinimumSide = 8;
using TreeLearningCodes =
    std::array<entropy_estimate::ResidualCode, kTreeCostPredictors>;
struct TreeSplitCosts {
  std::array<int64_t, kTreeCostPredictors> low{}, high{};
  bool operator==(const TreeSplitCosts &) const = default;
};
// Private synchronous hook for one sorted property of one original-learner
// leaf. The caller owns sampling, sorting, threshold rules, predictor/gain tie
// order, tree growth and checkpoints. Providers are scoped to one learning
// call; immutable codes/count costs may be prepared once and reused within that
// call.
class TreeSplitCostEvaluator {
public:
  virtual ~TreeSplitCostEvaluator() = default;
  // Allows small leaves to keep the original scan without preparing thresholds.
  [[nodiscard]] virtual bool
  WantsEvaluation(size_t leaf_samples) const noexcept = 0;
  // OK with handled=false declines before device work and preserves costs.
  // OK with handled=true fills exactly one cost pair per low count. Any error
  // is terminal and preserves costs; the caller must not retry after that
  // error.
  [[nodiscard]] virtual Status
  TryMeasure(std::span<const TreeLearningCodes> codes,
             std::span<const int64_t> count_log_count,
             std::span<const uint32_t> order,
             std::span<const uint32_t> low_counts,
             std::span<TreeSplitCosts> costs, bool *handled) = 0;
};
} // namespace gjxl::modular_internal
