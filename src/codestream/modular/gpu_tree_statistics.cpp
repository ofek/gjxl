// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/gpu_tree_statistics.h"
#include "codec/modular/profile.h"
#include <algorithm>

namespace gjxl::modular_internal {
using namespace tree_gpu_internal;
namespace {
void Fallback(bool small) {
  if (active_modular_profile) {
    ++active_modular_profile->adaptive_gpu.tree_cpu_fallback_properties;
    active_modular_profile->adaptive_gpu.tree_cpu_small_properties += small;
  }
}
class WorkTimer {
public:
  explicit WorkTimer(double AdaptiveGpuWorkProfile::*field)
      : sink_(active_modular_profile), field_(field) {
    if (sink_)
      start_ = std::chrono::steady_clock::now();
  }
  ~WorkTimer() {
    if (sink_)
      sink_->adaptive_gpu.*field_ +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        start_)
              .count();
  }

private:
  ModularProfile *sink_;
  double AdaptiveGpuWorkProfile::*field_;
  std::chrono::steady_clock::time_point start_{};
};
} // namespace
bool ModularGpuTreeStatistics::WantsEvaluation(
    size_t leaf_samples) const noexcept {
  if (!operation_ || leaf_samples < minimum_leaf_) {
    Fallback(operation_ && leaf_samples < minimum_leaf_);
    return false;
  }
  return true;
}
Status ModularGpuTreeStatistics::TryMeasure(
    std::span<const TreeLearningCodes> codes,
    std::span<const int64_t> count_log_count, std::span<const uint32_t> order,
    std::span<const uint32_t> low_counts, std::span<TreeSplitCosts> costs,
    bool *handled) {
  if (!handled || codes.empty() || count_log_count.size() != codes.size() + 1 ||
      costs.size() != low_counts.size() || low_counts.empty())
    return Status::InvalidArgument(
        "Invalid Modular learner statistics request");
  *handled = false;
  if (prepared_ &&
      (codes_ != codes.data() || count_log_count_ != count_log_count.data() ||
       samples_ != codes.size()))
    return Status::InvalidArgument(
        "Modular learner statistics crossed its sample lifetime");
  if (!WantsEvaluation(order.size()))
    return Status::Ok();
  if (auto s = ValidateTreeStatisticsThresholds(order.size(), low_counts);
      !s.ok())
    return s;
  const auto limits = operation_->tree_statistics_limits();
  TreeStatisticsStoragePlan plan;
  const auto preflight = ComputeTreeStatisticsStoragePlan(
      codes.size(), order.size(), low_counts.size(), &plan, &limits);
  if (preflight.code() == StatusCode::kUnsupported) {
    Fallback(false);
    return Status::Ok();
  }
  if (!preflight.ok())
    return preflight;
  // Keep a first preparation temporary until the whole attempt succeeds. A
  // failed measurement must not strand a new retained owner in this adapter.
  std::unique_ptr<PreparedGpuTreeStatistics> candidate;
  if (!prepared_) {
    WorkTimer timer(&AdaptiveGpuWorkProfile::tree_prepare_seconds);
    if (auto s = operation_->PrepareTreeStatistics({codes, count_log_count},
                                                   &candidate);
        !s.ok())
      return s;
    if (!candidate || candidate->backend_id() != backend_.id() ||
        candidate->sample_count() != codes.size())
      return Status::Internal(
          "Modular learner statistics preparation identity differs");
  }
  WorkTimer timer(&AdaptiveGpuWorkProfile::tree_evaluate_seconds);
  std::unique_ptr<PreparedGpuTreeSplitCosts> output;
  if (auto s = operation_->MeasureTreeSplitCosts(
          {prepared_ ? prepared_.get() : candidate.get(), order, low_counts},
          &output);
      !s.ok())
    return s;
  if (!output || output->backend_id() != backend_.id() ||
      output->costs().size() != costs.size())
    return Status::Internal(
        "Modular learner statistics result identity differs");
  const int64_t maximum = count_log_count[order.size()] +
                          (29 * static_cast<int64_t>(order.size()) +
                           2 * static_cast<int64_t>(kTreeCostSymbols) + 8) *
                              4096;
  for (const auto &result : output->costs())
    for (size_t p = 0; p < kTreeCostPredictors; ++p)
      if (result.low[p] < 0 || result.low[p] > maximum || result.high[p] < 0 ||
          result.high[p] > maximum)
        return Status::Internal(
            "Modular learner statistics returned an invalid cost");
  if (candidate) {
    prepared_ = std::move(candidate);
    codes_ = codes.data();
    count_log_count_ = count_log_count.data();
    samples_ = codes.size();
    if (active_modular_profile)
      ++active_modular_profile->adaptive_gpu.tree_gpu_preparations;
  }
  std::copy(output->costs().begin(), output->costs().end(), costs.begin());
  if (active_modular_profile) {
    auto &g = active_modular_profile->adaptive_gpu;
    ++g.tree_gpu_groups;
    g.tree_gpu_samples += order.size();
  }
  *handled = true;
  return Status::Ok();
}
} // namespace gjxl::modular_internal
