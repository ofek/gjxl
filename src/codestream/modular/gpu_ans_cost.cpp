// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/gpu_ans_cost.h"
#include "codec/modular/profile.h"
#include <algorithm>

namespace gjxl::modular_internal {
using namespace ans_gpu_internal;
using namespace codestream_internal;
using namespace resource_budget_internal;
Status ModularGpuAnsCost::TryMeasure(std::span<const EntropyTokenStreamView> sections,
                                     std::span<const EntropyCode *const> codes,
                                     std::span<const uint64_t> model_bits,
                                     std::span<EntropyCodeCost> costs, bool *handled) try {
  if (!handled || codes.empty() || codes.size() > kMaximumCostCandidates ||
      model_bits.size() != codes.size() || costs.size() != codes.size())
    return Status::InvalidArgument("Invalid Modular GPU ANS measurement");
  *handled = false;
  const auto fallback = [] {
    if (active_modular_profile)
      ++active_modular_profile->adaptive_gpu.ans_cpu_fallback_groups;
    return Status::Ok();
  };
  if (!operation_ || !tokens_.gpu_backing)
    return fallback();
  const auto &owner = *tokens_.gpu_backing;
  if (owner.backend_id() != backend_.id() || sections.data() != tokens_.streams.data() ||
      sections.size() != tokens_.streams.size())
    return Status::InvalidArgument("Modular GPU ANS owner/stream provenance differs");
  const size_t active_sections = std::count_if(
      sections.begin(), sections.end(), [](const auto section) { return section.size() != 0; });
  if (active_sections < kMinimumGpuAnsCostSections) {
    if (active_modular_profile)
      ++active_modular_profile->adaptive_gpu.ans_cpu_small_groups;
    return fallback();
  }
  size_t max_contexts = 0, max_clusters = 0;
  for (const auto *code : codes) {
    if (!code || code->mode != EntropyCodingMode::kAns ||
        code->context_count != code->context_map.size() ||
        code->uint_configs.size() != code->ans_histograms.size())
      return Status::InvalidArgument("Invalid Modular GPU ANS model");
    max_contexts = std::max(max_contexts, code->context_map.size());
    max_clusters = std::max(max_clusters, code->ans_histograms.size());
  }
  AnsTokenCostStoragePlan plan;
  const auto limits = operation_->ans_token_cost_limits();
  const auto preflight =
      ComputeAnsTokenCostStoragePlan(owner.token_count(), sections.size(), codes.size(),
                                     max_contexts, max_clusters, &plan, &limits);
  if (preflight.code() == StatusCode::kUnsupported)
    return fallback();
  if (!preflight.ok())
    return preflight;
  const ManagedHostScope resources(ResourceClass::kSerializer);
  modular_gpu_internal::ModularTokenHostView host;
  if (auto s = owner.BorrowHostView(&host); !s.ok())
    return s;
  if (host.values.size() != owner.token_count() || host.contexts.size() != owner.token_count() ||
      !host.values.data() || !host.contexts.data())
    return Status::InvalidArgument("Modular GPU ANS token extent differs");
  Storage<AnsTokenCostSection> ranges(sections.size());
  size_t offset = 0;
  for (size_t i = 0; i < sections.size(); ++i) {
    const auto section = sections[i];
    if (!section.valid() || !section.split || section.size() > owner.token_count() - offset ||
        section.values.data() != host.values.data() + offset ||
        section.contexts.data() != host.contexts.data() + offset)
      return Status::InvalidArgument("Modular GPU ANS split stream differs from resident tokens");
    ranges[i] = {offset, section.size()};
    offset += section.size();
  }
  if (offset != owner.token_count())
    return Status::InvalidArgument("Modular GPU ANS stream coverage differs");
  Storage<AnsTokenCostClusterView> clusters(codes.size() * max_clusters);
  std::array<AnsTokenCostModelView, kMaximumCostCandidates> models;
  for (size_t i = 0; i < codes.size(); ++i) {
    const auto &code = *codes[i];
    auto views = std::span(clusters).subspan(i * max_clusters, code.ans_histograms.size());
    for (size_t j = 0; j < views.size(); ++j) {
      const auto c = code.uint_configs[j];
      const auto &h = code.ans_histograms[j];
      views[j] = {c.split_exponent,  c.msb_in_token, c.lsb_in_token,          h.frequencies,
                  h.reverse_offsets, h.reverse_map,  h.reciprocal_frequencies};
    }
    models[i] = {code.context_map, views};
  }
  AnsTokenCostInput input{owner.values(), owner.contexts(), owner.token_count(), ranges,
                          std::span(models).first(codes.size())};
  std::unique_ptr<PreparedGpuAnsTokenCosts> output;
  // No fallback beyond preflight, even if a backend reports Unsupported after
  // beginning work. The operation drains failures before its borrowed inputs die.
  if (auto s = operation_->MeasureAnsTokenCosts(input, &output); !s.ok())
    return s;
  if (!output || output->backend_id() != backend_.id() ||
      output->section_count() != sections.size() || output->candidate_count() != codes.size() ||
      output->results().size() != codes.size() * sections.size())
    return Status::Internal("Modular GPU ANS result dimensions differ");
  std::array<EntropyCodeCost, kMaximumCostCandidates> candidate;
  for (size_t c = 0; c < codes.size(); ++c) {
    auto &cost = candidate[c];
    cost.model_bits = model_bits[c];
    cost.cluster_count = codes[c]->ans_histograms.size();
    cost.section_token_bits.resize(sections.size());
    for (size_t s = 0; s < sections.size(); ++s) {
      const auto result = output->results()[c * sections.size() + s];
      if (result.reserved || result.bits < 32 || result.bits > 32 + 47 * ranges[s].count)
        return Status::Internal("Modular GPU ANS returned an invalid cost");
      if (result.bits > UINT64_MAX - cost.token_bits)
        return Status::InvalidArgument("ANS token cost overflow");
      cost.section_token_bits[s] = result.bits;
      cost.token_bits += result.bits;
    }
  }
  for (size_t c = 0; c < codes.size(); ++c)
    costs[c] = std::move(candidate[c]);
  *handled = true;
  if (active_modular_profile) {
    ++active_modular_profile->adaptive_gpu.ans_gpu_groups;
    active_modular_profile->adaptive_gpu.ans_gpu_states += codes.size() * sections.size();
  }
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Modular GPU ANS allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Modular GPU ANS storage overflow");
}
} // namespace gjxl::modular_internal
