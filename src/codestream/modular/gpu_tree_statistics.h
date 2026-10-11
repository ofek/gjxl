// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/tree_learning_cost.h"
#include "gpu/ops/modular_tree_statistics.h"

namespace gjxl::modular_internal {
// M4 Pro production-learner probes include first preparation, upload and wait.
// Smaller leaves stay on the scalar scan; this changes scheduling only.
// See docs/gpu-modular-learner-statistics.md for the measured scope.
inline constexpr size_t kMinimumGpuTreeStatisticsLeafSamples = 16384;
// One original-learner invocation owns this adapter. Prepared codes/count costs
// are immutable and reused across its properties/leaves only. No global cache.
class ModularGpuTreeStatistics final : public TreeSplitCostEvaluator {
public:
  explicit ModularGpuTreeStatistics(
      GpuBackend &backend,
      size_t minimum_leaf = kMinimumGpuTreeStatisticsLeafSamples)
      : backend_(backend),
        operation_(tree_gpu_internal::QueryGpuModularTreeStatistics(backend)),
        minimum_leaf_(minimum_leaf) {}
  bool WantsEvaluation(size_t leaf_samples) const noexcept override;
  Status TryMeasure(std::span<const TreeLearningCodes> codes,
                    std::span<const int64_t> count_log_count,
                    std::span<const uint32_t> order,
                    std::span<const uint32_t> low_counts,
                    std::span<TreeSplitCosts> costs, bool *handled) override;

private:
  GpuBackend &backend_;
  tree_gpu_internal::GpuModularTreeStatistics *operation_;
  size_t minimum_leaf_;
  const TreeLearningCodes *codes_ = nullptr;
  const int64_t *count_log_count_ = nullptr;
  size_t samples_ = 0;
  std::unique_ptr<tree_gpu_internal::PreparedGpuTreeStatistics> prepared_;
};
} // namespace gjxl::modular_internal
