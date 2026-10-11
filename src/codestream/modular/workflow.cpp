// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/workflow.h"
#include "codestream/modular/adaptive_gpu_workflow.h"
#include "codec/modular/profile.h"
#include "codestream/modular/prepared_encoder.h"
#include "codestream/modular/search.h"
#include "codestream/workflow_admission_scope.h"
#include "core/publication_output.h"
#include "core/thread_budget.h"
#include <optional>

namespace gjxl::modular_internal {
namespace {
Status EncodeSearch(PackedModularImageView input, const ModularEncodingOptions &options,
                    resource_budget_internal::PublicationOutput<uint8_t> output);
Status Encode(PackedModularImageView input, const ModularEncodingOptions &options,
              resource_budget_internal::PublicationOutput<uint8_t> output,
              AdaptiveModularTokenCache *cache = nullptr) try {
  if (output == nullptr)
    return Status::InvalidArgument("Null Modular workflow output");
  if (auto s = input.Validate(); !s.ok())
    return s;
  if (auto s = ValidateTreeLearningOptions(options.learning); !s.ok())
    return s;
  if (options.search)
    return EncodeSearch(input, options, output);
  ModularWorkflowStoragePlan storage;
  if (auto s =
          ComputeModularWorkflowStoragePlan(input.extent, input.format, options.entropy,
                                            options.coding, &storage, options.cpu_thread_count);
      !s.ok())
    return s;
  size_t peak = storage.working.peak_bytes;
  if (options.adaptive_gpu && !cache) {
    AdaptiveModularWorkflowStoragePlan adaptive;
    if (auto s = ComputeAdaptiveModularWorkflowStoragePlan(input.extent, input.format, options,
                                                           &adaptive);
        !s.ok())
      return s;
    peak = adaptive.peak_bytes;
  }
  codestream_internal::WorkflowAdmission admission;
  if (auto s = admission.Start(peak, options.execution_domain); !s.ok())
    return s;
  thread_budget_internal::CpuExecutionScope execution;
  if (auto s = execution.Start(options.execution_domain, options.cpu_thread_count); !s.ok())
    return s;
  thread_budget_internal::EncodeScope threads(options.cpu_thread_count);
  resource_budget_internal::ManagedHostScope managed(
      resource_budget_internal::ResourceClass::kPreparation);
  std::optional<AdaptiveModularTokenCache> local_cache;
  if (options.adaptive_gpu && !cache) {
    local_cache.emplace(*options.adaptive_gpu, input, false);
    cache = &*local_cache;
  }
  codestream_internal::CodestreamBuffer candidate;
  if (cache) {
    if (auto s = cache->Encode(options.coding, options.entropy, storage, &candidate); !s.ok())
      return s;
  } else {
    ModularEncoderFrame frame;
    if (auto s = ModularEncoderFrame::Prepare(input, options.coding, &frame); !s.ok())
      return s;
    if (auto s = EncodePreparedModularFrame(std::move(frame), options.coding, options.entropy,
                                             storage, &candidate);
        !s.ok())
      return s;
  }
  if (active_modular_profile) {
    active_modular_profile->resolved = options.coding;
    ++active_modular_profile->encoded_candidates;
  }
  output.Publish(std::move(candidate));
  return Status::Ok();
} catch (const resource_budget_internal::ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Modular workflow allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Modular workflow storage overflow");
}
Status EncodeSearch(PackedModularImageView input, const ModularEncodingOptions &requested,
                    resource_budget_internal::PublicationOutput<uint8_t> output) {
  if (requested.coding != ModularCodingPolicy{})
    return Status::InvalidArgument("Search cannot override a prescribed Modular policy");
  ModularWorkflowStoragePlan storage;
  if (auto s = ComputeModularSearchStoragePlan(input.extent, input.format, requested.entropy,
                                               &storage, requested.cpu_thread_count,
                                               requested.learning, requested.tree_checkpoints);
      !s.ok())
    return s;
  size_t peak = storage.working.peak_bytes;
  if (requested.adaptive_gpu) {
    AdaptiveModularWorkflowStoragePlan adaptive;
    if (auto s = ComputeAdaptiveModularWorkflowStoragePlan(input.extent, input.format, requested,
                                                           &adaptive);
        !s.ok())
      return s;
    peak = adaptive.peak_bytes;
  }
  codestream_internal::WorkflowAdmission admission;
  if (auto s = admission.Start(peak, requested.execution_domain); !s.ok())
    return s;
  thread_budget_internal::CpuExecutionScope execution;
  if (auto s = execution.Start(requested.execution_domain, requested.cpu_thread_count); !s.ok())
    return s;
  resource_budget_internal::ManagedHostScope managed(
      resource_budget_internal::ResourceClass::kPreparation);
  thread_budget_internal::EncodeScope threads(requested.cpu_thread_count);
  std::optional<AdaptiveModularTokenCache> adaptive_cache;
  if (requested.adaptive_gpu)
    adaptive_cache.emplace(*requested.adaptive_gpu, input);
  auto *cache = adaptive_cache ? &*adaptive_cache : nullptr;
  auto options = requested;
  options.search = false;
  // Always retain the exact default learner and candidate order. Experimental
  // budgets/modes append candidates after this complete baseline search.
  options.learning = {};
  ModularCodingPolicy best_policy;
  codestream_internal::CodestreamBuffer best;
  if (auto s = Encode(input, options, &best, cache); !s.ok())
    return s;
  ModularInputProfile profile;
  if (auto s = ResolveModularInput(input.extent, input.format, &profile); !s.ok())
    return s;
  // Append the new policies only after all retained candidates. This preserves
  // the old search winner on equal sizes, including its transform decisions.
  std::array<ModularCodingPolicy, 6> context_candidates;
  size_t context_count = 0;
  DecisionTree fixed;
  if (auto s = MakeFixedWeightedTree(profile.metadata.bits, &fixed); !s.ok())
    return s;
  for (uint8_t rct : {uint8_t{0}, uint8_t{6}, uint8_t{9}}) {
    if (rct && profile.channel_count < 3)
      continue;
    ModularCodingPolicy single, split, entropy;
    {
      ModularEncoderFrame frame;
      ModularStreamPlan layout;
      if (auto s = ModularEncoderFrame::Prepare(input, rct, &frame); !s.ok())
        return s;
      if (auto s = BuildModularStreamPlan(storage.geometry, profile.channels(), 0, &layout);
          !s.ok())
        return s;
      if (auto s =
              LearnModularPolicies(frame, layout, rct, &single, &split, &entropy, options.learning);
          !s.ok())
        return s;
    }
    auto &fixed_candidate = context_candidates[context_count++];
    fixed_candidate.rct = rct;
    fixed_candidate.tree = fixed;
    if (entropy != single && entropy != split)
      context_candidates[context_count++] = entropy;
    for (size_t i = 0; i < 2; ++i) {
      const auto &policy = i ? split : single;
      if (policy == ModularCodingPolicy{} || (i && split == single))
        continue;
      options.coding = policy;
      codestream_internal::CodestreamBuffer candidate;
      if (auto s = Encode(input, options, &candidate, cache); !s.ok())
        return s;
      // Compare complete bytes, including transforms/tree/models/TOC/padding.
      // Baseline and earlier candidates win ties. Replacement releases the old winner.
      if (candidate.view().size() < best.view().size()) {
        best = std::move(candidate);
        if (active_modular_profile) best_policy = options.coding;
      }
    }
  }
  uint16_t palette_colors;
  if (auto s = ProbePalette(input, &palette_colors); !s.ok())
    return s;
  std::array<ModularCodingPolicy, 4> candidates;
  size_t count;
  if (auto s =
          BuildTransformCandidates(input.extent, input.format, palette_colors, &candidates, &count);
      !s.ok())
    return s;
  for (size_t i = 0; i < count; ++i) {
    options.coding = candidates[i];
    codestream_internal::CodestreamBuffer candidate;
    if (auto s = Encode(input, options, &candidate, cache); !s.ok())
      return s;
    if (candidate.view().size() < best.view().size()) {
      best = std::move(candidate);
      if (active_modular_profile) best_policy = options.coding;
    }
  }
  for (size_t i = 0; i < context_count; ++i) {
    options.coding = context_candidates[i];
    codestream_internal::CodestreamBuffer candidate;
    if (auto s = Encode(input, options, &candidate, cache); !s.ok())
      return s;
    if (candidate.view().size() < best.view().size()) {
      best = std::move(candidate);
      if (active_modular_profile)
        best_policy = options.coding;
    }
  }
  // Keep the complete retained search as a fallback. Rank all 42 transforms
  // cheaply, then train/encode only a bounded shortlist of additional RCTs.
  RctRanking ranking;
  if (auto s = RankModularRcts(input, &ranking); !s.ok())
    return s;
  for (size_t i = 0; i < ranking.count; ++i) {
    const uint8_t rct = ranking.shortlist[i];
    ModularCodingPolicy single, split, learned, weighted;
    {
      ModularEncoderFrame frame;
      ModularStreamPlan layout;
      if (auto s = ModularEncoderFrame::Prepare(input, rct, &frame); !s.ok())
        return s;
      if (auto s = BuildModularStreamPlan(storage.geometry, profile.channels(), 0, &layout);
          !s.ok())
        return s;
      if (auto s =
              LearnModularPolicies(frame, layout, rct, &single, &split, &learned, options.learning);
          !s.ok())
        return s;
    }
    weighted.rct = rct;
    weighted.tree = fixed;
    for (size_t kind = 0; kind < 2; ++kind) {
      const auto &policy = kind ? learned : weighted;
      if (kind && learned == weighted)
        continue;
      options.coding = policy;
      codestream_internal::CodestreamBuffer candidate;
      if (auto s = Encode(input, options, &candidate, cache); !s.ok())
        return s;
      if (candidate.view().size() < best.view().size()) {
        best = std::move(candidate);
        if (active_modular_profile)
          best_policy = policy;
      }
    }
  }
  // Append learned transform candidates after the complete retained search.
  // Train on the same metadata/index/residual planes and slice boundaries as
  // tokenization, then preserve the transform sequence in the learned policy.
  for (size_t i = 0; i < count; ++i) {
    ModularCodingPolicy single, split, learned;
    {
      ModularEncoderFrame frame;
      if (auto s = ModularEncoderFrame::Prepare(input, candidates[i], &frame); !s.ok())
        return s;
      ChannelShape shape;
      if (auto s = DescribeImage(frame.image(), &shape); !s.ok())
        return s;
      ModularStreamPlan layout;
      if (auto s = BuildModularStreamPlan(storage.geometry, shape.channels(), shape.metadata,
                                          &layout); !s.ok())
        return s;
      if (auto s = LearnModularPolicies(frame, layout, candidates[i], &single, &split, &learned,
                                        options.learning);
          !s.ok())
        return s;
    }
    if (learned == candidates[i])
      continue;
    options.coding = learned;
    codestream_internal::CodestreamBuffer candidate;
    if (auto s = Encode(input, options, &candidate, cache); !s.ok())
      return s;
    if (candidate.view().size() < best.view().size()) {
      best = std::move(candidate);
      if (active_modular_profile)
        best_policy = learned;
    }
  }
  if (requested.learning != TreeLearningOptions{} || requested.tree_checkpoints) {
    bool best_is_checkpoint = false;
    auto try_learning = [&](const ModularCodingPolicy &base) -> Status {
      ModularCodingPolicy single, split, learned;
      TreeLearningCheckpoints checkpoints;
      {
        ModularEncoderFrame frame;
        if (auto s = ModularEncoderFrame::Prepare(input, base, &frame); !s.ok())
          return s;
        ChannelShape shape;
        if (auto s = DescribeImage(frame.image(), &shape); !s.ok())
          return s;
        ModularStreamPlan layout;
        if (auto s =
                BuildModularStreamPlan(storage.geometry, shape.channels(), shape.metadata, &layout);
            !s.ok())
          return s;
        if (auto s = LearnModularPolicies(frame, layout, base, &single, &split, &learned,
                                          requested.learning, nullptr,
                                          requested.tree_checkpoints ? &checkpoints : nullptr);
            !s.ok())
          return s;
      }
      auto consider = [&](const DecisionTree &tree, bool checkpoint) -> Status {
        options.coding.rct = learned.rct;
        options.coding.transforms = learned.transforms;
        options.coding.weighted = learned.weighted;
        options.coding.tree = tree;
        codestream_internal::CodestreamBuffer candidate;
        if (auto s = Encode(input, options, &candidate, cache); !s.ok())
          return s;
        // All pre-existing final-tree candidates outrank checkpoints on ties,
        // including final candidates from later transforms in the search.
        if (candidate.view().size() < best.view().size() ||
            (candidate.view().size() == best.view().size() && best_is_checkpoint && !checkpoint)) {
          best = std::move(candidate);
          best_is_checkpoint = checkpoint;
          if (active_modular_profile)
            best_policy = options.coding;
        }
        return Status::Ok();
      };
      if (auto s = consider(learned.tree, false); !s.ok())
        return s;
      for (size_t i = 0; i < checkpoints.size; ++i)
        if (checkpoints.trees[i].size != learned.tree.size)
          if (auto s = consider(checkpoints.trees[i], true); !s.ok())
            return s;
      return Status::Ok();
    };
    for (uint8_t rct : {uint8_t{0}, uint8_t{6}, uint8_t{9}}) {
      if (rct && profile.channel_count < 3)
        continue;
      ModularCodingPolicy base;
      base.rct = rct;
      if (auto s = try_learning(base); !s.ok())
        return s;
    }
    for (size_t i = 0; i < ranking.count; ++i) {
      ModularCodingPolicy base;
      base.rct = ranking.shortlist[i];
      if (auto s = try_learning(base); !s.ok())
        return s;
    }
    for (size_t i = 0; i < count; ++i)
      if (auto s = try_learning(candidates[i]); !s.ok())
        return s;
  }
  if (active_modular_profile)
    active_modular_profile->resolved = std::move(best_policy);
  output.Publish(std::move(best));
  return Status::Ok();
}
} // namespace
Status EncodeRgb8ModularOwned(Rgb8View input, const ModularEncodingOptions &options,
                              codestream_internal::CodestreamBuffer *out) {
  return Encode(input.packed(), options, out);
}
Status EncodeRgb8Modular(Rgb8View input, const ModularEncodingOptions &options,
                         std::vector<uint8_t> *out) {
  return Encode(input.packed(), options, out);
}
Status EncodeModularImageOwned(PackedModularImageView input, const ModularEncodingOptions &options,
                               codestream_internal::CodestreamBuffer *out) {
  return Encode(input, options, out);
}
Status EncodeModularImage(PackedModularImageView input, const ModularEncodingOptions &options,
                          std::vector<uint8_t> *out) {
  return Encode(input, options, out);
}
} // namespace gjxl::modular_internal
