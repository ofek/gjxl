// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/ops/modular_stream_batch.h"
#include "codec/modular/weighted_feature_math.h"

namespace gjxl::modular_gpu_internal {
using namespace resource_budget_internal;
Status ComputeModularStreamBatchPlan(std::span<const ModularStreamChannel> channels,
                                     size_t node_count, ModularStreamBatchPlan *out,
                                     const ModularTokenDeviceLimits *limits) {
  auto invalid = [] { return Status::InvalidArgument("Invalid Modular stream batch/storage"); };
  if (!out || channels.empty() || channels.size() > UINT32_MAX || !node_count ||
      node_count > UINT32_MAX ||
      (limits &&
       (!limits->maximum_tokens || !limits->maximum_slices || !limits->maximum_buffer_bytes)))
    return invalid();
  ModularStreamBatchPlan result;
  auto &p = result.tokens;
  p.slice_count = channels.size();
  p.node_count = node_count;
  size_t max_width = 0;
  const auto add = [](size_t bytes, size_t *value) {
    if (bytes > size_t(PTRDIFF_MAX) - *value)
      return false;
    *value += bytes;
    return true;
  };
  const auto align = [](size_t *value) {
    if (*value > size_t(PTRDIFF_MAX) - 255)
      return false;
    *value = (*value + 255) & ~size_t{255};
    return true;
  };
  for (const auto &job : channels) {
    if (!job.weighted.valid() || job.stream_id >= 65535 || job.channel_id >= 65535)
      return invalid();
    if (auto s = job.channel.Validate(); !s.ok())
      return s;
    size_t area;
    if (!job.channel.descriptor.extent.try_area(&area) || !area || area > UINT32_MAX ||
        job.channel.descriptor.extent.width > UINT32_MAX ||
        job.channel.descriptor.extent.height > UINT32_MAX ||
        area > size_t(PTRDIFF_MAX) / sizeof(AdaptiveModularWeightedFeature) ||
        !add(area, &p.token_count) || !align(&p.plane_bytes) ||
        !add(area * sizeof(int32_t), &p.plane_bytes))
      return invalid();
    max_width = std::max(max_width, job.channel.descriptor.extent.width);
    result.maximum_slice_tokens = std::max(result.maximum_slice_tokens, area);
  }
  if (p.token_count > size_t(PTRDIFF_MAX) / sizeof(AdaptiveModularWeightedFeature) ||
      channels.size() > size_t(PTRDIFF_MAX) / sizeof(AdaptiveModularTokenSlice) ||
      node_count > size_t(PTRDIFF_MAX) / sizeof(ModularTokenNode))
    return invalid();
  p.weighted_bytes = p.token_count * sizeof(AdaptiveModularWeightedFeature);
  p.values_bytes = p.token_count * sizeof(uint32_t);
  p.contexts_offset = p.values_bytes;
  if (!align(&p.contexts_offset))
    return invalid();
  p.output_bytes = p.contexts_offset;
  if (!add(p.token_count * sizeof(uint16_t), &p.output_bytes))
    return invalid();
  p.slices_bytes = channels.size() * sizeof(AdaptiveModularTokenSlice);
  p.nodes_offset = p.slices_bytes;
  if (!align(&p.nodes_offset))
    return invalid();
  p.status_offset = p.nodes_offset;
  if (!add(node_count * sizeof(ModularTokenNode), &p.status_offset) || !align(&p.status_offset))
    return invalid();
  p.dispatch_bytes = p.status_offset;
  if (!add(sizeof(uint32_t), &p.dispatch_bytes))
    return invalid();
  result.parameters_offset = p.nodes_offset;
  result.feature_status_offset = result.parameters_offset;
  if (!add(channels.size() * sizeof(gjxl_weighted_features::Parameters),
           &result.feature_status_offset) ||
      !align(&result.feature_status_offset))
    return invalid();
  p.feature_dispatch_bytes = result.feature_status_offset;
  if (!add(sizeof(uint32_t), &p.feature_dispatch_bytes))
    return invalid();
  if (limits &&
      (p.token_count > limits->maximum_tokens || channels.size() > limits->maximum_slices ||
       std::max({p.plane_bytes, p.weighted_bytes, p.output_bytes, p.dispatch_bytes,
                 p.feature_dispatch_bytes}) > limits->maximum_buffer_bytes))
    return Status::Unsupported("Modular stream batch exceeds backend limits");
  if (!p.features.Add({p.plane_bytes, p.plane_bytes}) ||
      !p.features.Add({p.weighted_bytes, p.weighted_bytes}) ||
      !p.features.AddVector<AdaptiveModularTokenSlice>(channels.size(),
                                                       VectorCapacityPolicy::kFreshExact))
    return invalid();
  result.preparation = p.features;
  HostStorageBound scratch;
  if (auto s = modular_internal::ComputeWeightedPredictorScratchBound(max_width, 1, &scratch);
      !s.ok())
    return s;
  if (!result.preparation.Add(scratch) || !result.preparation.Add({0, p.feature_dispatch_bytes}) ||
      !result.generation.Add({p.output_bytes, p.output_bytes}) ||
      !result.generation.Add({0, p.dispatch_bytes}) ||
      !result.generation.AddVector<uint32_t>(node_count, VectorCapacityPolicy::kFreshExact))
    return invalid(); // At least one node per tree bounds the offset table.
  *out = result;
  return Status::Ok();
}
} // namespace gjxl::modular_gpu_internal
