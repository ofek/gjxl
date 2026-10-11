// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "core/host_storage_bound.h"
#include "gpu/ops/modular_tokenization.h"

namespace gjxl::modular_gpu_internal {
// Separate ABI from the original RCT-only token producer. Every neighborhood
// coordinate is slice-local; the plane offset already includes the rectangle
// origin. Property channel is the stream-local ordinal, not a plane index.
struct AdaptiveModularTokenSlice {
  uint64_t token_offset = 0, plane_offset_elements = 0, plane_stride_elements = 0;
  uint32_t stream_id = 0, property_channel = 0, width = 0, height = 0;
  uint32_t reserved[2]{};
  bool operator==(const AdaptiveModularTokenSlice &) const = default;
};
struct AdaptiveModularWeightedFeature {
  int64_t prediction = 0, property15 = 0;
  bool operator==(const AdaptiveModularWeightedFeature &) const = default;
};
struct AdaptiveModularTreeEligibility {
  size_t node_count = 0, context_count = 0;
  bool weighted = false;
  bool operator==(const AdaptiveModularTreeEligibility &) const = default;
};
// Borrowed validation, no policy/node copies. The full workflow also validates
// original-image decoder limits and the transform/layout match. All 16 current
// properties and all 14 predictors use the CPU tree representation limits.
[[nodiscard]] Status
ResolveAdaptiveModularTreeEligibility(const modular_internal::DecisionTree &tree,
                                      AdaptiveModularTreeEligibility *out);

struct AdaptiveModularTokenStoragePlan {
  size_t token_count = 0, slice_count = 0, node_count = 0;
  size_t plane_bytes = 0, weighted_bytes = 0;
  size_t values_bytes = 0, contexts_offset = 0, output_bytes = 0;
  size_t slices_bytes = 0, nodes_offset = 0, status_offset = 0, dispatch_bytes = 0;
  // Completed features own plane/weighted arenas and one host slice copy.
  resource_budget_internal::HostStorageBound features;
  // Includes the borrowed managed CPU frame, weighted staging and five two-row
  // arrays per participant. Original packed input and old output are separate.
  size_t bridge_peak_bytes = 0;
  // Features + output + device dispatch + fresh exact host dispatch staging.
  size_t generation_peak_bytes = 0;
  // Features remain cached during CPU consumption; count each arena only once.
  size_t retained_bytes = 0;
  bool operator==(const AdaptiveModularTokenStoragePlan &) const = default;
};
// Allocation-free, checked shape bound. The canonical planner supplies S and
// proves coverage; this function validates channel descriptors/count arithmetic.
// It conservatively bounds weighted state by the widest transformed channel.
// Active nodes must be an odd count within the CPU representation limit.
[[nodiscard]] Status ComputeAdaptiveModularTokenStoragePlan(
    std::span<const modular_internal::ChannelDescriptor> channels, size_t metadata_channels,
    size_t slices, size_t nodes, bool weighted, size_t cpu_participants,
    AdaptiveModularTokenStoragePlan *out, const ModularTokenDeviceLimits *limits = nullptr);

// Completed, immutable, backend-native features. The frame's original metadata
// is retained for the image-dependent decoder limit, independently of transformed
// sample count. Backend outlives this owner; one encoder call scopes its cache.
class PreparedGpuAdaptiveModularFeatures {
public:
  virtual ~PreparedGpuAdaptiveModularFeatures() = default;
  PreparedGpuAdaptiveModularFeatures(const PreparedGpuAdaptiveModularFeatures &) = delete;
  PreparedGpuAdaptiveModularFeatures &
  operator=(const PreparedGpuAdaptiveModularFeatures &) = delete;
  [[nodiscard]] virtual BackendId backend_id() const noexcept = 0;
  [[nodiscard]] virtual const codec_internal::ImageMetadata &metadata() const noexcept = 0;
  [[nodiscard]] virtual size_t token_count() const noexcept = 0;
  [[nodiscard]] virtual std::span<const AdaptiveModularTokenSlice> slices() const noexcept = 0;
  [[nodiscard]] virtual bool has_weighted_features() const noexcept = 0;

protected:
  PreparedGpuAdaptiveModularFeatures() = default;
};
// Optional private capability. The CPU bridge generates weighted history before
// publication; token evaluation consumes it without dependence between pixels.
// Both calls consume an admitted ambient reservation, preserve old output on
// failure, drain pending work, and never silently retry on CPU.
class GpuAdaptiveModularTokenGeneration {
public:
  virtual ~GpuAdaptiveModularTokenGeneration() = default;
  [[nodiscard]] virtual ModularTokenDeviceLimits adaptive_modular_limits() const noexcept = 0;
  [[nodiscard]] virtual Status
  PrepareAdaptiveModularFeatures(const modular_internal::ModularEncoderFrame &frame,
                                 std::span<const AdaptiveModularTokenSlice> slices,
                                 const modular_internal::WeightedPredictorParameters &parameters,
                                 bool weighted,
                                 std::unique_ptr<PreparedGpuAdaptiveModularFeatures> *out) = 0;
  [[nodiscard]] virtual Status
  GenerateAdaptiveModularTokens(const PreparedGpuAdaptiveModularFeatures &features,
                                const modular_internal::DecisionTree &tree,
                                std::unique_ptr<PreparedGpuModularTokens> *out) = 0;
};
[[nodiscard]] inline GpuAdaptiveModularTokenGeneration *
QueryGpuAdaptiveModularTokenGeneration(GpuBackend &backend) noexcept {
  return dynamic_cast<GpuAdaptiveModularTokenGeneration *>(&backend);
}
} // namespace gjxl::modular_gpu_internal
