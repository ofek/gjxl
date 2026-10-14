// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/adaptive_gpu_workflow.h"
#include "codec/modular/profile.h"
#include "codestream/modular/gpu_token_handoff.h"
#include "codestream/modular/prepared_encoder.h"
#include "codestream/modular/search.h"
#include "gpu/ops/modular_adaptive_layout.h"
#include "gpu/ops/ans_token_cost.h"
#include "gpu/ops/modular_tree_statistics.h"
#include "gpu/ops/modular_populations.h"
#include "codestream/entropy_population.h"
#include <algorithm>

namespace gjxl::modular_internal {
using namespace modular_gpu_internal;
using namespace resource_budget_internal;
namespace {
Status AdditionalBound(const ModularInputProfile &source, const ModularFrameGeometry &geometry,
                       const TransformSequence &transforms, size_t nodes, bool ans_cost,
                       size_t participants, size_t *out, size_t *samples) {
  std::array<ChannelShape, kMaximumTransforms + 1> shapes;
  if (auto s = PlanTransforms(source.channels(), 0, transforms, &shapes); !s.ok())
    return s;
  const auto &shape = shapes[transforms.size];
  ModularStreamStoragePlan layout;
  if (auto s = ComputeModularStreamStoragePlan(geometry, shape.channels(), shape.metadata, &layout);
      !s.ok())
    return s;
  size_t total = 0;
  for (const auto &c : shape.channels()) {
    size_t area;
    if (!c.extent.try_area(&area) || area > SIZE_MAX - total)
      return Status::InvalidArgument("Adaptive Modular sample count overflow");
    total += area;
  }
  // The shape planner includes potentially empty edge slices. Actual canonical
  // descriptors are nonempty, so both this bound and the sample count bound S.
  const size_t slices = std::min(total, layout.maximum_slices);
  AdaptiveModularTokenStoragePlan tokens;
  if (auto s = ComputeAdaptiveModularTokenStoragePlan(shape.channels(), shape.metadata, slices,
                                                      nodes, true, participants, &tokens);
      !s.ok())
    return s;
  HostStorageBound additional = layout.owned;
  const size_t operation = std::max(tokens.bridge_peak_bytes, tokens.generation_peak_bytes);
  if (!additional.Add({operation, operation}) ||
      !additional.AddVector<AdaptiveModularTokenSlice>(slices, VectorCapacityPolicy::kFreshExact))
    return Status::InvalidArgument("Adaptive Modular additional storage overflow");
  if (ans_cost) {
    ans_gpu_internal::AnsTokenCostStoragePlan cost;
    const size_t contexts = (nodes + 1) / 2;
    if (auto s = ans_gpu_internal::ComputeAnsTokenCostStoragePlan(
          total, layout.streams, ans_gpu_internal::kMaximumCostCandidates, contexts,
          std::min(contexts, ans_gpu_internal::kMaximumCostClusters), &cost); !s.ok())
      return s;
    if (!additional.Add({cost.peak_bytes, cost.peak_bytes}))
      return Status::InvalidArgument("Adaptive Modular ANS storage overflow");
  }
  *out = additional.peak_bytes;
  *samples = total;
  return Status::Ok();
}
class WorkTimer {
public:
  explicit WorkTimer(double AdaptiveGpuWorkProfile::*field,
                     double AdaptiveGpuWorkProfile::*subset = nullptr)
      : sink_(active_modular_profile), field_(field), subset_(subset) {
    if (sink_)
      start_ = std::chrono::steady_clock::now();
  }
  ~WorkTimer() {
    if (sink_) {
      const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
      sink_->adaptive_gpu.*field_ += seconds;
      if (subset_) sink_->adaptive_gpu.*subset_ += seconds;
    }
  }

private:
  ModularProfile *sink_;
  double AdaptiveGpuWorkProfile::*field_;
  double AdaptiveGpuWorkProfile::*subset_;
  std::chrono::steady_clock::time_point start_{};
};
} // namespace
Status ComputeAdaptiveModularWorkflowStoragePlan(Extent2D extent, PackedModularFormat format,
                                                 const ModularEncodingOptions &options,
                                                 AdaptiveModularWorkflowStoragePlan *out) {
  if (!out || (options.search && options.coding != ModularCodingPolicy{}) ||
      (options.adaptive_tree_statistics && !options.search))
    return Status::InvalidArgument("Invalid adaptive Modular workflow plan request");
  if (auto s = ValidateTreeLearningOptions(options.learning); !s.ok())
    return s;
  if (auto s = ValidateAnsSelectionMode(options.ans_selection); !s.ok())
    return s;
  AdaptiveModularWorkflowStoragePlan result;
  const bool candidate_evaluation = options.search_evaluation != ModularSearchEvaluation::kEncodeAll ||
                                    options.candidate_observer;
  const auto cpu_status =
      options.search
          ? ComputeModularSearchStoragePlan(extent, format, options.entropy, &result.cpu,
                                            options.cpu_thread_count, options.learning,
                                            options.tree_checkpoints, candidate_evaluation)
          : ComputeModularWorkflowStoragePlan(extent, format, options.entropy, options.coding,
                                              &result.cpu, options.cpu_thread_count, 0, candidate_evaluation);
  if (!cpu_status.ok())
    return cpu_status;
  ModularInputProfile source;
  if (auto s = ResolveModularInput(extent, format, &source); !s.ok())
    return s;
  const size_t leaves = std::min(
      std::max(kDefaultTreeLeaves, options.learning.maximum_leaves),
      MaximumGlobalTreeLeaves(uint64_t{extent.width} * extent.height * source.channel_count));
  const size_t nodes = options.search ? 2 * leaves - 1 : options.coding.tree.size;
  size_t maximum_samples = 0;
  const bool ans_cost = options.adaptive_ans_cost && options.entropy == EntropyCodingMode::kAns &&
                        (options.ans_selection == AnsSelectionMode::kExact ||
                         options.search_evaluation != ModularSearchEvaluation::kEncodeAll);
  if (auto s = AdditionalBound(source, result.cpu.geometry, options.coding.transforms, nodes,
                               ans_cost, result.cpu.participants, &result.additional_peak_bytes, &maximum_samples);
      !s.ok())
    return s;
  if (options.search) {
    std::array<ModularCodingPolicy, 4> candidates;
    size_t count;
    if (auto s =
            BuildTransformCandidates(extent, format, kMaximumPaletteColors, &candidates, &count);
        !s.ok())
      return s;
    for (size_t i = 0; i < count; ++i) {
      size_t bound, samples;
      if (auto s =
              AdditionalBound(source, result.cpu.geometry, candidates[i].transforms, nodes, ans_cost,
                              result.cpu.participants, &bound, &samples);
          !s.ok())
        return s;
      result.additional_peak_bytes = std::max(result.additional_peak_bytes, bound);
      maximum_samples = std::max(maximum_samples, samples);
    }
  }
  if (options.adaptive_tree_statistics && options.search) {
    // A retained feature layout can differ from the layout currently learning.
    // Add the independent maxima, not only the largest same-layout sum.
    const size_t samples = std::min(maximum_samples,
        std::max(kDefaultTrainingSamples, options.learning.maximum_samples));
    if (samples >= 2 * kTreeCostMinimumSide) {
      tree_gpu_internal::TreeStatisticsPreparationPlan prepared;
      tree_gpu_internal::TreeStatisticsStoragePlan evaluated;
      if (auto s = tree_gpu_internal::ComputeTreeStatisticsPreparationPlan(samples, &prepared); !s.ok())
        return s;
      if (auto s = tree_gpu_internal::ComputeTreeStatisticsStoragePlan(samples, samples,
              std::min(kTreeCostThresholds, samples - 2 * kTreeCostMinimumSide + 1), &evaluated); !s.ok())
        return s;
      const size_t peak = std::max(prepared.peak_bytes, evaluated.peak_bytes);
      if (peak > SIZE_MAX - result.additional_peak_bytes)
        return Status::InvalidArgument("Adaptive Modular learner statistics storage overflow");
      result.additional_peak_bytes += peak;
    }
  }
  if (options.adaptive_populations && maximum_samples >= 262144 &&
      (options.search || maximum_samples <= UINT32_MAX)) {
    // Reserve the independent scratch/output maxima, because shard count changes
    // with context shape. Retained counts can coexist with CPU model/merge work
    // and exact ANS scratch. Never rely on slack in the older CPU reservation.
    const size_t contexts = std::min(kMaximumPopulationContexts, (nodes + 1) / 2);
    HostStorageBound populations, merge;
    const size_t device = kMaximumPopulationScratchBytes + 8 + contexts * kPopulationValueLimit * 8;
    // A palette can move the largest search layout above the operation limit
    // while another layout remains eligible. Bound all potentially eligible
    // layouts; a maximum above the limit does not prove that all exceed it.
    if (auto s = codestream_internal::ComputeEntropyPopulationMergeStorageBound(
          std::min(maximum_samples, size_t{UINT32_MAX}),
          contexts, kPopulationValueLimit, std::min(contexts, kMaximumPrefixClusters), &merge); !s.ok()) return s;
    if (!populations.Add({device, device}) || !populations.Add(merge) ||
        populations.peak_bytes > SIZE_MAX - result.additional_peak_bytes)
      return Status::InvalidArgument("Adaptive Modular population storage overflow");
    result.additional_peak_bytes += populations.peak_bytes;
  }
  if (result.additional_peak_bytes > SIZE_MAX - result.cpu.working.peak_bytes)
    return Status::InvalidArgument("Adaptive Modular complete workflow storage overflow");
  result.peak_bytes = result.cpu.working.peak_bytes + result.additional_peak_bytes;
  *out = result;
  return Status::Ok();
}
void AdaptiveModularTokenCache::Evict() {
  if (features_ && active_modular_profile)
    ++active_modular_profile->adaptive_gpu.evictions;
  features_.reset();
  layout_ = {};
}
Status AdaptiveModularTokenCache::Prepare(const ModularCodingPolicy &policy, bool learning,
                                          ModularEncoderFrame *frame, bool *ready) {
  ProfileScope profile(learning ? ProfileStage::kTraining : ProfileStage::kTokens);
  *ready = false;
  if (!operation_) {
    return ModularEncoderFrame::Prepare(input_, policy, frame);
  }
  const auto limits = operation_->adaptive_modular_limits();
  AdaptiveModularTreeEligibility eligibility;
  if (auto s = ResolveAdaptiveModularTreeEligibility(policy.tree, &eligibility); !s.ok())
    return s;
  const bool weighted = learning || anticipate_weighted_ || eligibility.weighted;
  const size_t nodes = learning ? 1 : policy.tree.size;
  AdaptiveModularTokenStoragePlan plan;
  if (features_ && rct_ == policy.rct && transforms_ == policy.transforms &&
      weighted_ == policy.weighted && (features_->has_weighted_features() || !weighted)) {
    const auto status = ComputeAdaptiveModularTokenStoragePlan(
        shape_.channels(), shape_.metadata, features_->slices().size(), nodes,
        features_->has_weighted_features(), feature_participants_, &plan, &limits);
    if (status.code() == StatusCode::kUnsupported)
      return ModularEncoderFrame::Prepare(input_, policy, frame);
    if (!status.ok())
      return status;
    if (learning)
      if (auto s = ModularEncoderFrame::Prepare(input_, policy, frame); !s.ok())
        return s;
    if (active_modular_profile)
      ++active_modular_profile->adaptive_gpu.feature_reuses;
    *ready = true;
    return Status::Ok();
  }
  Evict();
  if (auto s = ModularEncoderFrame::Prepare(input_, policy, frame); !s.ok())
    return s;
  AdaptiveModularLayout adaptive;
  if (auto s = ComputeAdaptiveModularLayout(*frame, &adaptive); !s.ok())
    return s;
  const auto status = ComputeAdaptiveModularTokenStoragePlan(
      adaptive.shape.channels(), adaptive.shape.metadata, adaptive.slices, nodes,
      weighted, feature_participants_, &plan, &limits);
  if (status.code() == StatusCode::kUnsupported) {
    return Status::Ok();
  }
  if (!status.ok())
    return status;
  ModularStreamPlan layout;
  if (auto s = BuildModularStreamPlan(adaptive.geometry, adaptive.shape.channels(),
                                      adaptive.shape.metadata, &layout);
      !s.ok())
    return s;
  ManagedVector<AdaptiveModularTokenSlice> slices;
  if (auto s = BuildAdaptiveModularSlices(*frame, &slices); !s.ok())
    return s;
  std::unique_ptr<PreparedGpuAdaptiveModularFeatures> features;
  {
    WorkTimer timer(&AdaptiveGpuWorkProfile::feature_seconds,
                    learning ? &AdaptiveGpuWorkProfile::learning_feature_seconds : nullptr);
    // Every search layout may later use weighted leaves/property 15. Retain one
    // exact history for all its trees, including initially unweighted candidates.
    // Any failure after this preflight (including Unsupported from weighted
    // range validation) is terminal, never a silent CPU retry.
    if (auto s = operation_->PrepareAdaptiveModularFeatures(*frame, slices, policy.weighted,
                                                            weighted, &features, feature_participants_, feature_backend_);
        !s.ok())
      return s;
  }
  if (!features || features->token_count() != adaptive.tokens ||
      features->metadata() != frame->metadata() ||
      features->has_weighted_features() != weighted ||
      features->slices().size() != slices.size() ||
      !std::equal(slices.begin(), slices.end(), features->slices().begin()))
    return Status::Internal("Adaptive feature operation returned an inconsistent owner");
  if (active_modular_profile) {
    ++active_modular_profile->adaptive_gpu.feature_preparations;
    active_modular_profile->adaptive_gpu.feature_samples += adaptive.tokens;
  }
  shape_ = adaptive.shape;
  rct_ = policy.rct;
  transforms_ = policy.transforms;
  weighted_ = policy.weighted;
  layout_ = std::move(layout);
  features_ = std::move(features);
  if (!learning) *frame = {};
  *ready = true;
  return Status::Ok();
}
Status AdaptiveModularTokenCache::Learn(const ModularCodingPolicy &base, ModularCodingPolicy *single,
                                       ModularCodingPolicy *split, ModularCodingPolicy *entropy,
                                       TreeLearningOptions learning, TrainingCoverage *coverage,
                                       TreeLearningCheckpoints *checkpoints, GpuBackend *tree_gpu) {
  ModularEncoderFrame frame;
  bool ready;
  if (auto s = Prepare(base, true, &frame, &ready); !s.ok()) return s;
  if (!ready) {
    if (active_modular_profile) ++active_modular_profile->adaptive_gpu.learning_feature_fallbacks;
    ChannelShape shape;
    if (auto s = DescribeImage(frame.image(), &shape); !s.ok()) return s;
    ModularFrameGeometry geometry;
    if (auto s = ModularFrameGeometry::Create(frame.metadata().extent, &geometry); !s.ok()) return s;
    ModularStreamPlan layout;
    if (auto s = BuildModularStreamPlan(geometry, shape.channels(), shape.metadata, &layout); !s.ok()) return s;
    return LearnModularPolicies(frame, layout, base, single, split, entropy, learning, coverage,
                                checkpoints, tree_gpu);
  }
  std::span<const WeightedPredictorFeature> features;
  // Host borrowing is part of this private shared-feature contract. A failure
  // here is terminal, including Unsupported; the producer has already run.
  if (auto s = features_->BorrowWeightedHostView(&features); !s.ok()) return s;
  if (!features.data() || features.size() != features_->token_count())
    return Status::Internal("Adaptive feature owner returned an incomplete weighted map");
  if (active_modular_profile) ++active_modular_profile->adaptive_gpu.learning_feature_uses;
  return LearnModularPolicies(frame, layout_, base, single, split, entropy, learning, coverage,
                              checkpoints, tree_gpu, features);
}
Status AdaptiveModularTokenCache::Encode(const ModularCodingPolicy &policy, EntropyCodingMode mode,
                                         const ModularWorkflowStoragePlan &storage,
                                         codestream_internal::CodestreamBuffer *out,
                                         AnsSelectionMode selection, ModularCandidateEvaluation *evaluation) {
  if (auto s = ValidateAnsSelectionMode(selection); !s.ok())
    return s;
  ModularEncoderFrame fallback;
  bool ready;
  if (auto s = Prepare(policy, false, &fallback, &ready); !s.ok())
    return s;
  if (!ready) {
    const auto status = EncodePreparedModularFrame(std::move(fallback), policy, mode, storage, out,
                                                   selection, evaluation, population_backend_);
    if (status.ok() && active_modular_profile)
      ++active_modular_profile->adaptive_gpu.cpu_fallback_candidates;
    return status;
  }
  PreparedModularTokens tokens;
  {
    ProfileScope profile(ProfileStage::kTokens);
    WorkTimer timer(&AdaptiveGpuWorkProfile::token_seconds);
    std::unique_ptr<PreparedGpuModularTokens> owner;
    if (auto s = operation_->GenerateAdaptiveModularTokens(*features_, policy.tree, &owner);
        !s.ok())
      return s;
    if (auto s = AdoptAdaptiveGpuModularTokens(layout_, policy, std::move(owner), &tokens); !s.ok())
      return s;
  }
  auto metadata = features_->metadata();
  if (policy != ModularCodingPolicy{})
    metadata.modular_16_bit_buffer_sufficient = false;
  const auto status = EncodeModularFrame(metadata, layout_, std::move(tokens), mode, storage, out,
                                         selection, ans_backend_, evaluation, population_backend_);
  if (status.ok() && active_modular_profile)
    ++active_modular_profile->adaptive_gpu.encoded_candidates;
  return status;
}
} // namespace gjxl::modular_internal
