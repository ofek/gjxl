// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/ops/ans_token_cost.h"
#include <algorithm>
#include <type_traits>

namespace gjxl::ans_gpu_internal {
using namespace resource_budget_internal;
static_assert(sizeof(AnsTokenCostSection) == 16 && alignof(AnsTokenCostSection) == 8);
static_assert(sizeof(AnsTokenCostResult) == 16 && alignof(AnsTokenCostResult) == 8 &&
              offsetof(AnsTokenCostResult, final_state) == 8);
static_assert(std::is_trivially_copyable_v<AnsTokenCostSection> &&
              std::is_trivially_copyable_v<AnsTokenCostResult>);
namespace {
bool Add(size_t bytes, size_t *sum) {
  if (bytes > size_t(PTRDIFF_MAX) - *sum)
    return false;
  *sum += bytes;
  return true;
}
bool Align(size_t *bytes) {
  if (*bytes > size_t(PTRDIFF_MAX) - 255)
    return false;
  *bytes = (*bytes + 255) & ~size_t{255};
  return true;
}
Status Overflow() { return Status::InvalidArgument("ANS token-cost storage overflow"); }
} // namespace
Status ComputeAnsTokenCostStoragePlan(size_t tokens, size_t sections, size_t candidates,
                                      size_t maximum_contexts, size_t maximum_clusters,
                                      AnsTokenCostStoragePlan *out,
                                      const AnsTokenCostDeviceLimits *limits) {
  if (!out || !tokens || !sections || sections > UINT32_MAX || !candidates ||
      candidates > kMaximumCostCandidates || !maximum_contexts ||
      maximum_contexts > kMaximumCostContexts || !maximum_clusters ||
      maximum_clusters > kMaximumCostClusters || maximum_clusters > maximum_contexts)
    return Status::InvalidArgument("Invalid ANS token-cost shape");
  if (limits &&
      (!limits->maximum_tokens || !limits->maximum_sections || !limits->maximum_buffer_bytes))
    return Status::InvalidArgument("Invalid ANS token-cost limits");
  if (tokens > size_t(PTRDIFF_MAX) / sizeof(uint32_t) || tokens > (UINT64_MAX - 32) / 47 ||
      sections > size_t(PTRDIFF_MAX) / sizeof(AnsTokenCostResult) / candidates)
    return Overflow();
  AnsTokenCostStoragePlan p;
  p.token_count = tokens;
  p.section_count = sections;
  p.candidate_count = candidates;
  p.maximum_contexts = maximum_contexts;
  p.maximum_clusters = maximum_clusters;
  p.sections_bytes = sections * sizeof(AnsTokenCostSection);
  p.models_offset = p.sections_bytes;
  if (!Align(&p.models_offset))
    return Overflow();
  p.context_maps_offset = p.models_offset;
  if (!Add(candidates * kCostModelBytes, &p.context_maps_offset) || !Align(&p.context_maps_offset))
    return Overflow();
  p.clusters_offset = p.context_maps_offset;
  if (!Add(candidates * maximum_contexts, &p.clusters_offset) || !Align(&p.clusters_offset))
    return Overflow();
  p.status_offset = p.clusters_offset;
  if (!Add(candidates * maximum_clusters * kCostClusterBytes, &p.status_offset))
    return Overflow();
  p.dispatch_bytes = p.status_offset;
  if (!Add(sizeof(uint32_t), &p.dispatch_bytes))
    return Overflow();
  p.output_bytes = sections * candidates * sizeof(AnsTokenCostResult);
  if (limits && (tokens > limits->maximum_tokens || sections > limits->maximum_sections ||
                 p.dispatch_bytes > limits->maximum_buffer_bytes ||
                 p.output_bytes > limits->maximum_buffer_bytes))
    return Status::Unsupported("ANS token-cost storage exceeds backend limits");
  HostStorageBound work{p.output_bytes, p.output_bytes};
  if (!work.Add({p.dispatch_bytes, p.dispatch_bytes}) ||
      !work.AddVector<std::byte>(p.dispatch_bytes, VectorCapacityPolicy::kFreshExact) ||
      !work.AddVector<AnsTokenCostSection>(sections, VectorCapacityPolicy::kFreshExact) ||
      !work.AddVector<AnsTokenCostClusterView>(candidates * maximum_clusters,
                                               VectorCapacityPolicy::kFreshExact))
    return Overflow();
  p.retained_bytes = p.output_bytes;
  p.peak_bytes = work.peak_bytes;
  *out = p;
  return Status::Ok();
}
Status ValidateAnsTokenCostSections(size_t tokens, std::span<const AnsTokenCostSection> sections) {
  if (!tokens || sections.empty() || sections.size() > UINT32_MAX ||
      tokens > size_t(PTRDIFF_MAX) / sizeof(uint32_t) || tokens > (UINT64_MAX - 32) / 47)
    return Status::InvalidArgument("Invalid ANS token-cost sections");
  uint64_t offset = 0;
  for (const auto &section : sections) {
    if (section.offset != offset || section.count > tokens - offset)
      return Status::InvalidArgument("ANS token-cost sections do not partition tokens");
    offset += section.count;
  }
  return offset == tokens ? Status::Ok()
                          : Status::InvalidArgument("ANS token-cost sections omit tokens");
}
Status ValidateAnsTokenCostModels(std::span<const AnsTokenCostModelView> models) {
  if (models.empty() || models.size() > kMaximumCostCandidates)
    return Status::InvalidArgument("Invalid ANS token-cost candidates");
  for (const auto &model : models) {
    if (model.context_map.empty() || model.context_map.size() > kMaximumCostContexts ||
        model.clusters.empty() || model.clusters.size() > kMaximumCostClusters ||
        model.clusters.size() > model.context_map.size())
      return Status::InvalidArgument("Invalid ANS token-cost model extent");
    for (const auto cluster : model.context_map)
      if (cluster >= model.clusters.size())
        return Status::InvalidArgument("ANS token-cost context map is out of range");
    for (const auto &cluster : model.clusters) {
      const size_t symbols = cluster.frequencies.size();
      if (cluster.split_exponent > 15 || cluster.msb_in_token > cluster.split_exponent ||
          cluster.lsb_in_token > cluster.split_exponent - cluster.msb_in_token || !symbols ||
          symbols > kCostAlphabetSize || cluster.reverse_offsets.size() != symbols ||
          cluster.reciprocal_frequencies.size() != symbols ||
          cluster.reverse_map.size() != kCostTableSize)
        return Status::InvalidArgument("Invalid ANS token-cost cluster extent/configuration");
      uint32_t sum = 0;
      for (size_t symbol = 0; symbol < symbols; ++symbol) {
        const uint32_t frequency = cluster.frequencies[symbol];
        const uint64_t reciprocal =
            frequency ? ((uint64_t{1} << 44) + frequency - 1) / frequency : 0;
        if (frequency > kCostTableSize - sum || cluster.reverse_offsets[symbol] != sum ||
            cluster.reciprocal_frequencies[symbol] != reciprocal)
          return Status::InvalidArgument("Invalid ANS token-cost frequency/reciprocal/offset");
        sum += frequency;
      }
      if (sum != kCostTableSize)
        return Status::InvalidArgument("ANS token-cost frequencies are not normalized");
      for (const auto entry : cluster.reverse_map)
        if (entry >= kCostTableSize)
          return Status::InvalidArgument("ANS token-cost reverse map is out of range");
    }
  }
  return Status::Ok();
}
Status ValidateAnsTokenCostInput(const AnsTokenCostInput &input, BackendId backend,
                                 AnsTokenCostStoragePlan *out,
                                 const AnsTokenCostDeviceLimits *limits) {
  if (!out || !backend)
    return Status::InvalidArgument("Invalid ANS token-cost output/backend");
  if (auto s = ValidateAnsTokenCostSections(input.token_count, input.sections); !s.ok())
    return s;
  if (auto s = ValidateAnsTokenCostModels(input.models); !s.ok())
    return s;
  size_t contexts = 0, clusters = 0;
  for (const auto &model : input.models) {
    contexts = std::max(contexts, model.context_map.size());
    clusters = std::max(clusters, model.clusters.size());
  }
  const auto range_ok = [backend](DeviceMemoryRange range, size_t count, size_t element) {
    return range.buffer && range.buffer->backend_id() == backend &&
           range.offset_bytes % element == 0 && range.size_bytes == count * element &&
           range.offset_bytes <= range.buffer->size_bytes() &&
           range.size_bytes <= range.buffer->size_bytes() - range.offset_bytes;
  };
  if (!range_ok(input.values, input.token_count, sizeof(uint32_t)) ||
      !range_ok(input.contexts, input.token_count, sizeof(uint16_t)))
    return Status::InvalidArgument("Invalid ANS token-cost resident ranges");
  if (input.values.buffer == input.contexts.buffer &&
      input.values.offset_bytes < input.contexts.offset_bytes + input.contexts.size_bytes &&
      input.contexts.offset_bytes < input.values.offset_bytes + input.values.size_bytes)
    return Status::InvalidArgument("ANS token-cost resident ranges overlap");
  return ComputeAnsTokenCostStoragePlan(input.token_count, input.sections.size(),
                                        input.models.size(), contexts, clusters, out, limits);
}
Status CheckAnsTokenCostDeviceStatus(uint32_t status) {
  if (status & ~(kCostContextError | kCostSymbolError | kCostStateError))
    return Status::Internal("Unknown ANS token-cost device status");
  if (status & kCostContextError)
    return Status::InvalidArgument("ANS token-cost context is out of range");
  if (status & kCostSymbolError)
    return Status::InvalidArgument("ANS token-cost symbol is absent");
  if (status & kCostStateError)
    return Status::InvalidArgument("ANS token-cost state or reverse map is invalid");
  return Status::Ok();
}
} // namespace gjxl::ans_gpu_internal
