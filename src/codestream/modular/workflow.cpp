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
#include <chrono>
#include <algorithm>

namespace gjxl::modular_internal {
namespace {
Status EncodeSearch(PackedModularImageView input, const ModularEncodingOptions &options,
                    resource_budget_internal::PublicationOutput<uint8_t> output);
Status Encode(PackedModularImageView input, const ModularEncodingOptions &options,
              resource_budget_internal::PublicationOutput<uint8_t> output,
              AdaptiveModularTokenCache *cache = nullptr,
              ModularCandidateEvaluation *evaluation = nullptr, bool count_candidate = true) try {
  if (output == nullptr && (!evaluation || evaluation->operation == ModularCandidateOperation::kEmit))
    return Status::InvalidArgument("Null Modular workflow output");
  if (auto s = input.Validate(); !s.ok())
    return s;
  if (options.adaptive_ans_cost && !options.adaptive_gpu)
    return Status::InvalidArgument("GPU ANS cost requires the adaptive GPU route");
  if (options.adaptive_tree_statistics && (!options.adaptive_gpu || !options.search))
    return Status::InvalidArgument("GPU learner statistics requires the adaptive GPU search route");
  if (auto s = ValidateTreeLearningOptions(options.learning); !s.ok())
    return s;
  if (auto s = ValidateAnsSelectionMode(options.ans_selection); !s.ok())
    return s;
  if (options.search_evaluation != ModularSearchEvaluation::kEncodeAll &&
      options.search_evaluation != ModularSearchEvaluation::kMeasureAll &&
      options.search_evaluation != ModularSearchEvaluation::kShortlist)
    return Status::InvalidArgument("Invalid Modular search evaluation mode");
  if (!options.search && !cache && !evaluation &&
      (options.search_evaluation != ModularSearchEvaluation::kEncodeAll || options.candidate_observer))
    return Status::InvalidArgument("Candidate evaluation requires Modular search");
  if (options.search)
    return EncodeSearch(input, options, output);
  ModularWorkflowStoragePlan storage;
  if (auto s =
          ComputeModularWorkflowStoragePlan(input.extent, input.format, options.entropy,
                                            options.coding, &storage, options.cpu_thread_count, 0, evaluation != nullptr);
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
    local_cache.emplace(*options.adaptive_gpu, input, false, options.adaptive_ans_cost);
    cache = &*local_cache;
  }
  codestream_internal::CodestreamBuffer candidate;
  if (cache) {
    if (auto s = cache->Encode(options.coding, options.entropy, storage, &candidate,
                                options.ans_selection, evaluation); !s.ok())
      return s;
  } else {
    ModularEncoderFrame frame;
    if (auto s = ModularEncoderFrame::Prepare(input, options.coding, &frame); !s.ok())
      return s;
    if (auto s = EncodePreparedModularFrame(std::move(frame), options.coding, options.entropy,
                                             storage, &candidate, options.ans_selection, evaluation);
        !s.ok())
      return s;
  }
  if (active_modular_profile) {
    active_modular_profile->resolved = options.coding;
    if (count_candidate) ++active_modular_profile->encoded_candidates;
  }
  if (!evaluation || evaluation->operation == ModularCandidateOperation::kEmit)
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
                                               requested.learning, requested.tree_checkpoints,
                                               requested.search_evaluation != ModularSearchEvaluation::kEncodeAll ||
                                               requested.candidate_observer);
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
    adaptive_cache.emplace(*requested.adaptive_gpu, input, true, requested.adaptive_ans_cost);
  auto *cache = adaptive_cache ? &*adaptive_cache : nullptr;
  auto *tree_gpu = requested.adaptive_tree_statistics ? requested.adaptive_gpu : nullptr;
  auto options = requested;
  options.search = false;
  options.adaptive_tree_statistics = false;
  // Always retain the exact default learner and candidate order. Experimental
  // budgets/modes append candidates after this complete baseline search.
  options.learning = {};
  // The traversal below is shared by the emission oracle and exact sizing.
  // Keep the winning policy even when profiling is disabled: measurement must
  // re-create and emit that policy once, after all candidate owners are released.
  options.search_evaluation = ModularSearchEvaluation::kEncodeAll;
  options.candidate_observer = nullptr;
  const bool shortlist = requested.search_evaluation == ModularSearchEvaluation::kShortlist;
  const bool measure = requested.search_evaluation != ModularSearchEvaluation::kEncodeAll;
  struct Finalist {
    ModularCodingPolicy policy;
    size_t index = 0;
    bool checkpoint = false;
    double bits = 0;
  };
  std::array<Finalist, kMaximumSearchShortlist> finalists;
  size_t finalist_count = 0;
  ModularCodingPolicy best_policy;
  codestream_internal::CodestreamBuffer best;
  size_t best_bytes = SIZE_MAX, candidate_index = 0;
  bool best_is_checkpoint = false;
  auto consider = [&](const ModularCodingPolicy &policy, bool checkpoint = false) -> Status {
    options.coding = policy;
    ModularCandidateEvaluation evaluation;
    evaluation.operation = shortlist ? ModularCandidateOperation::kScreen :
                           measure ? ModularCandidateOperation::kMeasure : ModularCandidateOperation::kEmit;
    codestream_internal::CodestreamBuffer candidate;
    const auto start = std::chrono::steady_clock::now();
    if (auto s = Encode(input, options, &candidate, cache,
                         measure || requested.candidate_observer ? &evaluation : nullptr); !s.ok())
      return s;
    const size_t bytes = measure ? evaluation.cost.bytes : candidate.view().size();
    const bool improved = !shortlist && (bytes < best_bytes ||
                          (bytes == best_bytes && best_is_checkpoint && !checkpoint));
    if (shortlist) {
      // Always retain the baseline and the three lowest sampled estimates.
      // Stable estimate ties retain earlier candidates; exact finalists below
      // are visited in original order with the original checkpoint tie rule.
      size_t position = candidate_index == 0 ? 0 : 1;
      while (position < finalist_count && evaluation.estimate.bits >= finalists[position].bits)
        ++position;
      if (position < finalists.size()) {
        finalist_count = std::min(finalist_count + 1, finalists.size());
        for (size_t i = finalist_count - 1; i > position; --i)
          finalists[i] = std::move(finalists[i - 1]);
        finalists[position] = {policy, candidate_index, checkpoint, evaluation.estimate.bits};
      }
    }
    if (improved) {
      best_bytes = bytes;
      best_is_checkpoint = checkpoint;
      best_policy = policy;
      if (!measure) best = std::move(candidate);
    }
    if (requested.candidate_observer) {
      const ModularCandidateRecord record{
          candidate_index, checkpoint, improved,
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
          evaluation.cost, !shortlist, evaluation.estimate.bits, evaluation.estimate.sampled_tokens};
      if (auto s = requested.candidate_observer->Candidate(policy, record); !s.ok()) return s;
    }
    ++candidate_index;
    return Status::Ok();
  };
  if (auto s = consider({}); !s.ok()) return s;
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
              LearnModularPolicies(frame, layout, rct, &single, &split, &entropy, options.learning, nullptr, nullptr, tree_gpu);
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
      if (auto s = consider(policy); !s.ok()) return s;
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
    if (auto s = consider(candidates[i]); !s.ok()) return s;
  }
  for (size_t i = 0; i < context_count; ++i) {
    if (auto s = consider(context_candidates[i]); !s.ok()) return s;
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
              LearnModularPolicies(frame, layout, rct, &single, &split, &learned, options.learning, nullptr, nullptr, tree_gpu);
          !s.ok())
        return s;
    }
    weighted.rct = rct;
    weighted.tree = fixed;
    for (size_t kind = 0; kind < 2; ++kind) {
      const auto &policy = kind ? learned : weighted;
      if (kind && learned == weighted)
        continue;
      if (auto s = consider(policy); !s.ok()) return s;
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
                                        options.learning, nullptr, nullptr, tree_gpu);
          !s.ok())
        return s;
    }
    if (learned == candidates[i])
      continue;
    if (auto s = consider(learned); !s.ok()) return s;
  }
  if (requested.learning != TreeLearningOptions{} || requested.tree_checkpoints) {
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
                                          requested.tree_checkpoints ? &checkpoints : nullptr, tree_gpu);
            !s.ok())
          return s;
      }
      if (auto s = consider(learned); !s.ok()) return s;
      for (size_t i = 0; i < checkpoints.size; ++i) {
        if (checkpoints.trees[i].size == learned.tree.size) continue;
        auto policy = learned;
        policy.tree = checkpoints.trees[i];
        if (auto s = consider(policy, true); !s.ok()) return s;
      }
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
  if (shortlist) {
    std::sort(finalists.begin(), finalists.begin() + finalist_count,
              [](const auto &a, const auto &b) { return a.index < b.index; });
    for (size_t i = 0; i < finalist_count; ++i) {
      const auto &finalist = finalists[i];
      options.coding = finalist.policy;
      ModularCandidateEvaluation evaluation;
      evaluation.operation = ModularCandidateOperation::kMeasure;
      const auto start = std::chrono::steady_clock::now();
      if (auto s = Encode(input, options, nullptr, cache, &evaluation, false); !s.ok()) return s;
      const size_t bytes = evaluation.cost.bytes;
      const bool improved = bytes < best_bytes ||
                            (bytes == best_bytes && best_is_checkpoint && !finalist.checkpoint);
      if (improved) {
        best_bytes = bytes;
        best_policy = finalist.policy;
        best_is_checkpoint = finalist.checkpoint;
      }
      if (requested.candidate_observer) {
        const ModularCandidateRecord record{
            finalist.index, finalist.checkpoint, improved,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
            evaluation.cost, true, finalist.bits, 0};
        if (auto s = requested.candidate_observer->Candidate(finalist.policy, record); !s.ok()) return s;
      }
    }
  }
  if (measure) {
    options.coding = best_policy;
    if (auto s = Encode(input, options, &best, cache, nullptr, false); !s.ok()) return s;
    if (best.view().size() != best_bytes)
      return Status::Internal("Measured Modular winner differs from final emission");
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
