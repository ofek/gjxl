// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/ops/modular_tree_statistics.h"
#include "codec/modular/tree_learning.h"
#include <type_traits>

namespace gjxl::tree_gpu_internal {
using namespace modular_internal;
using namespace resource_budget_internal;
static_assert(kMaximumTreeStatisticsSamples == kMaximumTrainingSamples);
static_assert(sizeof(entropy_estimate::ResidualCode) == 2 &&
              offsetof(entropy_estimate::ResidualCode, extra_bits) == 1);
static_assert(sizeof(TreeLearningCodes) == 28 &&
              sizeof(TreeSplitCosts) == 224 && alignof(TreeSplitCosts) == 8 &&
              offsetof(TreeSplitCosts, high) == 112);
static_assert(std::is_trivially_copyable_v<TreeLearningCodes> &&
              std::is_trivially_copyable_v<TreeSplitCosts>);
namespace {
// Samples are capped at 2^20 and thresholds at 32 before arithmetic, so these
// aligned regions and their summed peaks fit even 32-bit size_t/PTRDIFF_MAX.
size_t Align(size_t bytes) { return (bytes + 255) & ~size_t{255}; }
bool ValidLimits(const TreeStatisticsDeviceLimits *limits) {
  return !limits || (limits->maximum_samples && limits->maximum_leaf_samples &&
                     limits->maximum_buffer_bytes);
}
Status Overflow() {
  return Status::InvalidArgument("Modular tree statistics storage overflow");
}
} // namespace
Status
ComputeTreeStatisticsPreparationPlan(size_t samples,
                                     TreeStatisticsPreparationPlan *out,
                                     const TreeStatisticsDeviceLimits *limits) {
  if (!out || !samples || samples > kMaximumTreeStatisticsSamples ||
      !ValidLimits(limits))
    return Status::InvalidArgument(
        "Invalid Modular tree statistics preparation shape");
  TreeStatisticsPreparationPlan p;
  p.sample_count = samples;
  p.codes_bytes = samples * sizeof(TreeLearningCodes);
  p.count_costs_offset = Align(p.codes_bytes);
  p.storage_bytes = p.count_costs_offset + (samples + 1) * sizeof(int64_t);
  if (limits && (samples > limits->maximum_samples ||
                 p.storage_bytes > limits->maximum_buffer_bytes))
    return Status::Unsupported(
        "Modular tree statistics preparation exceeds device limits");
  HostStorageBound bound{p.storage_bytes, p.storage_bytes};
  if (!bound.AddVector<std::byte>(p.storage_bytes,
                                  VectorCapacityPolicy::kFreshExact))
    return Overflow();
  p.retained_bytes = p.storage_bytes;
  p.peak_bytes = bound.peak_bytes;
  *out = p;
  return Status::Ok();
}
Status ComputeTreeStatisticsStoragePlan(
    size_t samples, size_t leaf_samples, size_t thresholds,
    TreeStatisticsStoragePlan *out, const TreeStatisticsDeviceLimits *limits) {
  if (!out || leaf_samples < 2 * kTreeCostMinimumSide ||
      leaf_samples > samples || !thresholds ||
      thresholds > kTreeCostThresholds ||
      thresholds > leaf_samples - 2 * kTreeCostMinimumSide + 1)
    return Status::InvalidArgument(
        "Invalid Modular tree statistics evaluation shape");
  TreeStatisticsPreparationPlan prepared;
  if (auto s = ComputeTreeStatisticsPreparationPlan(samples, &prepared, limits);
      !s.ok())
    return s;
  TreeStatisticsStoragePlan p;
  p.sample_count = samples;
  p.leaf_count = leaf_samples;
  p.threshold_count = thresholds;
  p.prepared_bytes = prepared.retained_bytes;
  p.indices_bytes = leaf_samples * sizeof(uint32_t);
  p.boundaries_offset = Align(p.indices_bytes);
  p.histograms_offset =
      Align(p.boundaries_offset + (thresholds + 2) * sizeof(uint32_t));
  p.histograms_bytes = (thresholds + 1) * kTreeCostPredictors *
                       kTreeCostSymbols * sizeof(uint32_t);
  p.status_offset = Align(p.histograms_offset + p.histograms_bytes);
  p.dispatch_bytes = p.status_offset + sizeof(uint32_t);
  p.output_bytes = thresholds * sizeof(TreeSplitCosts);
  if (limits && (leaf_samples > limits->maximum_leaf_samples ||
                 p.dispatch_bytes > limits->maximum_buffer_bytes ||
                 p.output_bytes > limits->maximum_buffer_bytes))
    return Status::Unsupported(
        "Modular tree statistics evaluation exceeds device limits");
  HostStorageBound bound{p.prepared_bytes, p.prepared_bytes};
  if (!bound.Add({p.output_bytes, p.output_bytes}) ||
      !bound.Add({p.dispatch_bytes, p.dispatch_bytes}) ||
      !bound.AddVector<std::byte>(p.dispatch_bytes,
                                  VectorCapacityPolicy::kFreshExact))
    return Overflow();
  p.retained_bytes = p.output_bytes;
  p.peak_bytes = bound.peak_bytes;
  *out = p;
  return Status::Ok();
}
Status
ValidateTreeStatisticsPreparation(const TreeStatisticsPreparationInput &input,
                                  TreeStatisticsPreparationPlan *out,
                                  const TreeStatisticsDeviceLimits *limits) {
  TreeStatisticsPreparationPlan plan;
  if (!out || input.count_log_count.size() != input.codes.size() + 1)
    return Status::InvalidArgument(
        "Invalid Modular tree statistics preparation extent");
  if (auto s = ComputeTreeStatisticsPreparationPlan(input.codes.size(), &plan,
                                                    limits);
      !s.ok())
    return s;
  if (input.count_log_count[0] != 0)
    return Status::InvalidArgument(
        "Nonzero Modular tree statistics zero-count cost");
  for (size_t i = 1; i < input.count_log_count.size(); ++i)
    if (input.count_log_count[i] !=
        static_cast<int64_t>(i) *
            entropy_estimate::Log2(static_cast<uint32_t>(i)))
      return Status::InvalidArgument(
          "Noncanonical Modular tree statistics count cost");
  for (const auto &sample : input.codes)
    for (const auto code : sample) {
      const unsigned extra = code.symbol < 16 ? 0 : 2 + (code.symbol - 16) / 4;
      if (code.symbol >= kTreeCostSymbols || code.extra_bits != extra)
        return Status::InvalidArgument(
            "Invalid Modular tree statistics residual code");
    }
  *out = plan;
  return Status::Ok();
}
Status ValidateTreeStatisticsThresholds(size_t leaf_samples,
                                        std::span<const uint32_t> low_counts) {
  if (leaf_samples < 2 * kTreeCostMinimumSide ||
      leaf_samples > kMaximumTreeStatisticsSamples || low_counts.empty() ||
      low_counts.size() > kTreeCostThresholds)
    return Status::InvalidArgument(
        "Invalid Modular tree statistics thresholds");
  uint32_t previous = 0;
  for (const uint32_t low : low_counts) {
    if (low < kTreeCostMinimumSide ||
        low > leaf_samples - kTreeCostMinimumSide || low <= previous)
      return Status::InvalidArgument(
          "Modular tree statistics thresholds are not an interior partition");
    previous = low;
  }
  return Status::Ok();
}
Status ValidateTreeStatisticsInput(const TreeStatisticsInput &input,
                                   BackendId backend,
                                   TreeStatisticsStoragePlan *out,
                                   const TreeStatisticsDeviceLimits *limits) {
  if (!out || !backend || !input.prepared ||
      input.prepared->backend_id() != backend)
    return Status::InvalidArgument(
        "Invalid Modular tree statistics owner/output");
  if (auto s = ValidateTreeStatisticsThresholds(input.order.size(),
                                                input.low_counts);
      !s.ok())
    return s;
  return ComputeTreeStatisticsStoragePlan(input.prepared->sample_count(),
                                          input.order.size(),
                                          input.low_counts.size(), out, limits);
}
Status CheckTreeStatisticsDeviceStatus(uint32_t status) {
  if (status & ~(kTreeStatisticsIndexError | kTreeStatisticsCodeError |
                 kTreeStatisticsHistogramError))
    return Status::Internal("Unknown Modular tree statistics device status");
  if (status & kTreeStatisticsIndexError)
    return Status::InvalidArgument(
        "Modular tree statistics sample index is out of range");
  if (status & kTreeStatisticsCodeError)
    return Status::InvalidArgument(
        "Modular tree statistics residual code is invalid");
  if (status & kTreeStatisticsHistogramError)
    return Status::Internal(
        "Modular tree statistics histogram count is invalid");
  return Status::Ok();
}
} // namespace gjxl::tree_gpu_internal
