// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/frame_encoder.h"
#include "codestream/modular/gpu_ans_cost.h"
#include "codestream/modular/gpu_populations.h"
#include "codestream/entropy_population.h"
#include "codec/modular/profile.h"
#include "codestream/headers_internal.h"
#include "codestream/modular/stream_encoder.h"
#include "codestream/modular/tree_codec.h"
#include "codestream/sections.h"
#include "codestream/modular/parallel.h"

namespace gjxl::modular_internal {
using namespace codestream_internal;
Status EncodeModularFrame(const codec_internal::ImageMetadata &metadata,
                          const ModularStreamPlan &layout, PreparedModularTokens &&input,
                          EntropyCodingMode mode, const ModularWorkflowStoragePlan &storage,
                          CodestreamBuffer *out, AnsSelectionMode selection,
                          GpuBackend *ans_gpu, ModularCandidateEvaluation *evaluation, GpuBackend *population_gpu) try {
  if (auto s = ValidateAnsSelectionMode(selection); !s.ok())
    return s;
  const auto operation = evaluation ? evaluation->operation : ModularCandidateOperation::kEmit;
  const bool emit = operation == ModularCandidateOperation::kEmit;
  if ((!out && emit) || input.streams.size() != layout.streams.size() ||
      (operation != ModularCandidateOperation::kEmit && operation != ModularCandidateOperation::kMeasure &&
       operation != ModularCandidateOperation::kScreen))
    return Status::InvalidArgument("Invalid Modular frame handoff");
  ModularCandidateEvaluation result;
  result.operation = operation;
  Storage<BitWriter> sections;
  {
    PreparedModularTokens tokens = std::move(input);
    if (operation == ModularCandidateOperation::kScreen) {
      if (population_gpu && active_modular_profile)
        ++active_modular_profile->adaptive_gpu.population_screen_skips;
      ProfileScope profile_scope(ProfileStage::kModel);
      if (auto s = ScreenModularCandidate(metadata, layout, tokens, mode, &result.estimate); !s.ok())
        return s;
      *evaluation = result;
      if (active_modular_profile) ++active_modular_profile->screened_candidates;
      return Status::Ok();
    }
    EntropyCode model;
    EntropyCodeCost model_cost;
    auto *requested_cost = emit ? nullptr : &model_cost;
    {
      ProfileScope profile_scope(ProfileStage::kModel);
      thread_budget_internal::EncodeScope serial(1);
      auto *entropy_profile = active_modular_profile ? &active_modular_profile->entropy_work
                                                    : nullptr;
      const EntropyCodeOptions options{.context_count = static_cast<uint32_t>(tokens.context_count)};
      std::unique_ptr<modular_gpu_internal::PreparedGpuModularPopulations> population_owner;
      if (population_gpu)
        if (auto s = PrepareModularPopulations(*population_gpu, tokens, &population_owner); !s.ok()) return s;
      EntropyContextPopulations dense;
      if (population_owner) dense = {tokens.context_count, modular_gpu_internal::kPopulationValueLimit,
                                      population_owner->counts()};
      const auto prefix_model = [&](PreparedEntropyClusters *prepared, EntropyCodeCost *cost) {
        if (population_owner)
          return OptimizeEntropyCodeFromContextPopulations(tokens.streams, options, dense, &model,
                                                             cost, prepared, entropy_profile);
        if (prepared)
          return OptimizeEntropyCodeAndPrepareClusters(tokens.streams, options, &model, cost,
                                                         prepared, entropy_profile);
        return OptimizeEntropyCode(tokens.streams, options, &model, cost, entropy_profile);
      };
      if (mode == EntropyCodingMode::kAns) {
        // The selected prefix partition already computed these exact raw-value
        // populations. Keep them through ANS model construction, then release
        // them before payload emission. No prefix section costs are requested.
        PreparedEntropyClusters populations;
        if (auto s = prefix_model(&populations, nullptr);
            !s.ok())
          return s;
        EntropyCode ans;
        if (ans_gpu) {
          ModularGpuAnsCost evaluator(*ans_gpu, tokens);
          if (auto s = OptimizeAnsEntropyCodeWithPreparedClusters(
                tokens.streams, model, populations, &ans, requested_cost, entropy_profile, selection,
                &evaluator); !s.ok())
            return s;
        } else {
          if (auto s = OptimizeAnsEntropyCodeWithPreparedClusters(
                tokens.streams, model, populations, &ans, requested_cost, entropy_profile, selection); !s.ok())
            return s;
        }
        model = std::move(ans);
      } else if (auto s = prefix_model(nullptr, requested_cost); !s.ok()) {
        return s;
      }
    }
    if (!emit) {
      ProfileScope profile_scope(ProfileStage::kModel);
      if (auto s = MeasureModularCandidate(metadata, layout, tokens, model, model_cost, &result.cost); !s.ok())
        return s;
      *evaluation = result;
      if (active_modular_profile) ++active_modular_profile->exactly_measured_candidates;
      return Status::Ok();
    }
    ProfileScope emission_scope(ProfileStage::kEmission);
    if (evaluation) model_cost.section_token_bits.resize(tokens.streams.size(), 0);
    sections.resize(layout.geometry.section_count());
    {
      thread_budget_internal::EncodeScope serial(1);
      auto &global = sections[0];
      if (auto s = global.WithMaxBits(
              storage.maximum_global_bits,
              [&]() -> Status {
                if (auto s = global.WriteBits(1, 1); !s.ok())
                  return s; // default DC matrices
                if (auto s = WriteGlobalTreeInTransaction(
                        std::span(tokens.tree_tokens).first(tokens.tree_token_count), &global);
                    !s.ok())
                  return s;
                const size_t model_begin = global.bits_written();
                if (auto s = WriteGlobalModelInTransaction(model, &global); !s.ok())
                  return s;
                if (evaluation) model_cost.model_bits = global.bits_written() - model_begin - 1;
                if (auto s = WriteCodingStreamHeader(tokens.policy.weighted, tokens.policy.rct,
                                                     tokens.policy.transforms, &global);
                    !s.ok())
                  return s;
                if (tokens.streams[0].size()) {
                  const size_t begin = global.bits_written();
                  if (auto s = WriteStreamTokensWithValidatedModel(tokens.streams[0], model, &global); !s.ok())
                    return s;
                  if (evaluation) model_cost.section_token_bits[0] = global.bits_written() - begin;
                }
                return Status::Ok();
              });
          !s.ok())
        return s;
    }
    size_t active_streams = 0;
    for (size_t i = 1; i < tokens.streams.size(); ++i)
      active_streams += tokens.streams[i].size() != 0;
    const size_t participants = std::min(storage.participants, std::max(size_t{1}, active_streams));
    if (auto status = RunModularStreams(
            layout.streams.size() - 1, participants,
            thread_budget_internal::WorkerLaunchSite::kModularEmission,
            [&](size_t index) -> Status {
              const size_t i = index + 1;
              if (tokens.streams[i].size() == 0)
                return Status::Ok();
              auto &section = sections[layout.streams[i].section];
              if (auto s = section.WithMaxBits(
                      storage.maximum_group_bits,
                      [&]() -> Status {
                        if (auto s = WriteCodingStreamHeader(tokens.policy.weighted, 0, &section);
                            !s.ok())
                          return s;
                        const size_t begin = section.bits_written();
                        if (auto s = WriteStreamTokensWithValidatedModel(tokens.streams[i], model,
                                                                         &section); !s.ok()) return s;
                        if (evaluation) model_cost.section_token_bits[i] = section.bits_written() - begin;
                        return Status::Ok();
                      });
                  !s.ok())
                return s;
              return Status::Ok();
            });
        !status.ok())
      return status;
    if (evaluation)
      if (auto s = MeasureModularCandidate(metadata, layout, tokens, model, model_cost, &result.cost); !s.ok())
        return s;
  } // Tokens and model are no longer live during final assembly.
  ProfileScope assembly_scope(ProfileStage::kAssembly);
  BitWriter file;
  if (auto s = WriteImageHeader(metadata, &file); !s.ok())
    return s;
  if (auto s = WriteFrameHeader(metadata, {}, &file); !s.ok())
    return s;
  if (auto s = WriteTocAndSections(sections, &file); !s.ok())
    return s;
  // Clear backing as well as elements before allocating the published copy.
  Storage<BitWriter>().swap(sections);
  if (evaluation && result.cost.bytes != file.padded_size())
    return Status::Internal("Modular complete candidate cost differs from emitted bytes");
  if (auto s = CodestreamBuffer::CopyFrom(file.padded_bytes(), out,
              resource_budget_internal::ResourceClass::kRetainedResult); !s.ok()) return s;
  if (evaluation) *evaluation = result;
  if (active_modular_profile) ++active_modular_profile->candidate_emissions;
  return Status::Ok();
} catch (const resource_budget_internal::ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Modular frame allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Modular frame storage overflow");
}
} // namespace gjxl::modular_internal
