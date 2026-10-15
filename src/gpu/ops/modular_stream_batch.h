// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "gpu/ops/modular_adaptive_tokens.h"

namespace gjxl::modular_gpu_internal {
// One nonempty, already transformed channel. The caller preserves canonical
// stream/channel order and decoder scope limits. Neighborhoods reset per job.
struct ModularStreamChannel {
  modular_internal::ModularChannelView channel;
  modular_internal::WeightedPredictorParameters weighted;
  uint32_t stream_id = 0, channel_id = 0, tree_index = 0;
};
struct ModularStreamBatchPlan {
  AdaptiveModularTokenStoragePlan tokens;
  size_t maximum_slice_tokens = 0;
  size_t parameters_offset = 0, feature_status_offset = 0;
  resource_budget_internal::HostStorageBound preparation, generation;
};
// Validates borrowed views and computes fresh backing only. No pixel reads or
// allocations. node_count is the sum across the supplied tree table (one during
// feature preparation); tree offsets use 32 bits. Old/borrowed owners are extra.
[[nodiscard]] Status
ComputeModularStreamBatchPlan(std::span<const ModularStreamChannel> channels, size_t node_count,
                              ModularStreamBatchPlan *out,
                              const ModularTokenDeviceLimits *limits = nullptr);

class PreparedGpuModularStreamFeatures {
public:
  virtual ~PreparedGpuModularStreamFeatures() = default;
  [[nodiscard]] virtual size_t token_count() const noexcept = 0;
  [[nodiscard]] virtual Status
  BorrowWeightedHostView(std::span<const AdaptiveModularWeightedFeature> *out) const = 0;
};
// Batched counterpart to the legacy single-policy adaptive operation. All
// features use checked coding/training arithmetic, never WP-selection wrapping.
// Completed owners retain charges; backend outlives owners. Failure preserves
// outputs and drains submissions. Oversized WP channels use the CPU producer
// before dispatch; device failures are never retried on CPU.
class GpuModularStreamGeneration {
public:
  virtual ~GpuModularStreamGeneration() = default;
  [[nodiscard]] virtual ModularTokenDeviceLimits modular_stream_limits() const noexcept = 0;
  [[nodiscard]] virtual Status
  PrepareModularStreamFeatures(std::span<const ModularStreamChannel> channels,
                               std::unique_ptr<PreparedGpuModularStreamFeatures> *out,
                               modular_internal::WeightedFeatureBackend producer =
                                   modular_internal::WeightedFeatureBackend::kAuto) = 0;
  [[nodiscard]] virtual Status
  GenerateModularStreamTokens(const PreparedGpuModularStreamFeatures &features,
                              std::span<const modular_internal::DecisionTree *const> trees,
                              std::unique_ptr<PreparedGpuModularTokens> *out) = 0;
};
[[nodiscard]] inline GpuModularStreamGeneration *
QueryGpuModularStreamGeneration(GpuBackend &backend) {
  return dynamic_cast<GpuModularStreamGeneration *>(&backend);
}
} // namespace gjxl::modular_gpu_internal
