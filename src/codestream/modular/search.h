// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/modular/storage_plan.h"
#include "codec/modular/tree_learning.h"
namespace gjxl { class GpuBackend; }
namespace gjxl::modular_internal {
inline constexpr size_t kMaximumRctSamples = 2048;
inline constexpr size_t kMaximumAdditionalRcts = 3;
struct RctRanking {
  // Q12 estimates for all 42 RCTs; no serialized-size or optimality guarantee.
  std::array<int64_t, 42> costs{};
  std::array<uint8_t, kMaximumAdditionalRcts> shortlist{};
  size_t count = 0, samples = 0;
  bool operator==(const RctRanking &) const = default;
};
// Rank sampled gradient/left/top residual distributions. Skip 0/6/9 and
// duplicate sorted per-channel score profiles. Grayscale returns no candidates.
// Output is replaced only on success; scratch uses managed preparation storage.
[[nodiscard]] Status RankModularRcts(PackedModularImageView input, RctRanking *out);
[[nodiscard]] Status
ComputeRctRankingStorageBound(resource_budget_internal::HostStorageBound *out);
// Allocation-free bounded transform policy enumeration. A zero palette size
// means ineligible; exact probing stops after the 257th distinct tuple.
[[nodiscard]] Status ProbePalette(PackedModularImageView input, uint16_t *colors);
[[nodiscard]] Status BuildTransformCandidates(Extent2D extent, PackedModularFormat format,
                                              uint16_t palette_colors,
                                              std::array<ModularCodingPolicy, 4> *out,
                                              size_t *count);
struct TrainingCoverage {
  size_t population = 0, samples = 0, slices = 0, sampled_slices = 0;
  size_t minimum_slice_samples = 0, maximum_slice_samples = 0;
  size_t slice_floor = 0;
  uint64_t quota_hash = 14695981039346656037ULL;
  // Diagnostic fingerprints include sample positions, all properties and all
  // predictor residuals. Computed only when the caller requests coverage.
  uint64_t original_sample_hash = 14695981039346656037ULL;
  uint64_t entropy_sample_hash = 14695981039346656037ULL;
  bool operator==(const TrainingCoverage &) const = default;
};
// Deterministic uniform sampling in canonical stream/channel/raster order.
// The retained candidates use the original uniform samples and bit-width cost.
// Optional entropy learning uses a separate, bounded stratified sample set.
// An optional completed feature span must describe this exact frame, canonical
// plan and weighted parameters, and remain immutable/alive through the call.
// Empty selects the scalar oracle; a nonempty span must cover every sample.
[[nodiscard]] Status LearnModularPolicies(const ModularEncoderFrame &frame,
                                          const ModularStreamPlan &plan, uint8_t rct,
                                          ModularCodingPolicy *single, ModularCodingPolicy *split,
                                          ModularCodingPolicy *entropy = nullptr,
                                          TreeLearningOptions learning = {},
                                          TrainingCoverage *coverage = nullptr,
                                          TreeLearningCheckpoints *checkpoints = nullptr,
                                          GpuBackend *tree_gpu = nullptr,
                                          std::span<const WeightedPredictorFeature> features = {});
// Frame and layout must describe the completed transforms in base. Learn on
// those channels (including metadata), preserving transforms and WP parameters.
[[nodiscard]] Status
LearnModularPolicies(const ModularEncoderFrame &frame, const ModularStreamPlan &plan,
                     const ModularCodingPolicy &base, ModularCodingPolicy *single,
                     ModularCodingPolicy *split, ModularCodingPolicy *entropy = nullptr,
                     TreeLearningOptions learning = {}, TrainingCoverage *coverage = nullptr,
                     TreeLearningCheckpoints *checkpoints = nullptr,
                     GpuBackend *tree_gpu = nullptr,
                     std::span<const WeightedPredictorFeature> features = {});
[[nodiscard]] Status
ComputeModularSearchStoragePlan(Extent2D extent, PackedModularFormat format, EntropyCodingMode mode,
                                ModularWorkflowStoragePlan *out, size_t cpu_threads = 1,
                                TreeLearningOptions learning = {}, bool tree_checkpoints = false,
                                bool candidate_evaluation = false);
} // namespace gjxl::modular_internal
