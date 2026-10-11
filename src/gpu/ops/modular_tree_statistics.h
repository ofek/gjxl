// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/tree_learning_cost.h"
#include "core/host_storage_bound.h"
#include "gpu/backend.h"
#include <span>

namespace gjxl::tree_gpu_internal {
using modular_internal::TreeLearningCodes;
using modular_internal::TreeSplitCosts;
inline constexpr size_t kMaximumTreeStatisticsSamples = 1 << 20;
struct TreeStatisticsDeviceLimits {
  size_t maximum_samples = 0, maximum_leaf_samples = 0,
         maximum_buffer_bytes = 0;
};
struct TreeStatisticsPreparationInput {
  std::span<const TreeLearningCodes> codes;
  std::span<const int64_t> count_log_count;
};
struct TreeStatisticsPreparationPlan {
  size_t sample_count = 0, codes_bytes = 0, count_costs_offset = 0;
  size_t storage_bytes = 0, retained_bytes = 0, peak_bytes = 0;
  bool operator==(const TreeStatisticsPreparationPlan &) const = default;
};
struct TreeStatisticsStoragePlan {
  size_t sample_count = 0, leaf_count = 0, threshold_count = 0;
  size_t prepared_bytes = 0, indices_bytes = 0, boundaries_offset = 0;
  size_t histograms_offset = 0, histograms_bytes = 0, status_offset = 0;
  size_t dispatch_bytes = 0, output_bytes = 0, retained_bytes = 0,
         peak_bytes = 0;
  bool operator==(const TreeStatisticsStoragePlan &) const = default;
};
// Allocation-free and failure-atomic. Preparation includes one conservative
// fresh host staging arena. Evaluation peak includes live prepared backing,
// output, dispatch/scratch and one fresh dispatch staging arena. It excludes
// borrowed CPU codes/table/order, existing CPU learner scratch, inline
// controls, caller-owned old output, backend initialization and driver memory.
[[nodiscard]] Status ComputeTreeStatisticsPreparationPlan(
    size_t samples, TreeStatisticsPreparationPlan *out,
    const TreeStatisticsDeviceLimits *limits = nullptr);
[[nodiscard]] Status ComputeTreeStatisticsStoragePlan(
    size_t samples, size_t leaf_samples, size_t thresholds,
    TreeStatisticsStoragePlan *out,
    const TreeStatisticsDeviceLimits *limits = nullptr);
// Validates every supplied code and the canonical integer Q12 count table;
// performs no allocation, backend access or submission.
[[nodiscard]] Status ValidateTreeStatisticsPreparation(
    const TreeStatisticsPreparationInput &input,
    TreeStatisticsPreparationPlan *out,
    const TreeStatisticsDeviceLimits *limits = nullptr);
// Strictly increasing interior low counts, at least eight samples per side.
// Empty candidate lists are handled by the learner without invoking the device.
[[nodiscard]] Status
ValidateTreeStatisticsThresholds(size_t leaf_samples,
                                 std::span<const uint32_t> low_counts);

class PreparedGpuTreeStatistics {
public:
  virtual ~PreparedGpuTreeStatistics() = default;
  PreparedGpuTreeStatistics(const PreparedGpuTreeStatistics &) = delete;
  PreparedGpuTreeStatistics &
  operator=(const PreparedGpuTreeStatistics &) = delete;
  [[nodiscard]] virtual BackendId backend_id() const noexcept = 0;
  [[nodiscard]] virtual size_t sample_count() const noexcept = 0;

protected:
  PreparedGpuTreeStatistics() = default;
};
struct TreeStatisticsInput {
  const PreparedGpuTreeStatistics *prepared = nullptr;
  // Ordered leaf membership; repeated valid indices are a supported multiset.
  // The learner separately guarantees its unique leaf membership and sorting.
  std::span<const uint32_t> order, low_counts;
};
[[nodiscard]] Status
ValidateTreeStatisticsInput(const TreeStatisticsInput &input, BackendId backend,
                            TreeStatisticsStoragePlan *out,
                            const TreeStatisticsDeviceLimits *limits = nullptr);
inline constexpr uint32_t kTreeStatisticsIndexError = 1,
                          kTreeStatisticsCodeError = 2,
                          kTreeStatisticsHistogramError = 4;
[[nodiscard]] Status CheckTreeStatisticsDeviceStatus(uint32_t status);

class PreparedGpuTreeSplitCosts {
public:
  virtual ~PreparedGpuTreeSplitCosts() = default;
  PreparedGpuTreeSplitCosts(const PreparedGpuTreeSplitCosts &) = delete;
  PreparedGpuTreeSplitCosts &
  operator=(const PreparedGpuTreeSplitCosts &) = delete;
  [[nodiscard]] virtual BackendId backend_id() const noexcept = 0;
  // Completed, immutable, threshold-major low/high arrays, predictor wire
  // order. No allocation/copy/second resource charge; the owner outlives
  // borrowed spans.
  [[nodiscard]] virtual std::span<const TreeSplitCosts>
  costs() const noexcept = 0;

protected:
  PreparedGpuTreeSplitCosts() = default;
};
// Optional private capability. Both calls consume ambient admission, complete
// synchronously and publish only on success. Implementations validate native
// owner provenance, check device index/code/histogram errors, and drain pending
// work before releasing borrowed inputs or failed backing. No scalar retry.
class GpuModularTreeStatistics {
public:
  virtual ~GpuModularTreeStatistics() = default;
  [[nodiscard]] virtual TreeStatisticsDeviceLimits
  tree_statistics_limits() const noexcept = 0;
  [[nodiscard]] virtual Status
  PrepareTreeStatistics(const TreeStatisticsPreparationInput &input,
                        std::unique_ptr<PreparedGpuTreeStatistics> *out) = 0;
  [[nodiscard]] virtual Status
  MeasureTreeSplitCosts(const TreeStatisticsInput &input,
                        std::unique_ptr<PreparedGpuTreeSplitCosts> *out) = 0;
};
[[nodiscard]] inline GpuModularTreeStatistics *
QueryGpuModularTreeStatistics(GpuBackend &backend) noexcept {
  return dynamic_cast<GpuModularTreeStatistics *>(&backend);
}
} // namespace gjxl::tree_gpu_internal
