// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/ops/modular_adaptive_tokens.h"
#include <algorithm>
#include <type_traits>

namespace gjxl::modular_gpu_internal {
using namespace modular_internal;
using namespace resource_budget_internal;
static_assert(std::is_trivially_copyable_v<AdaptiveModularTokenSlice> &&
              std::is_standard_layout_v<AdaptiveModularTokenSlice>);
static_assert(sizeof(AdaptiveModularTokenSlice) == 48 &&
              offsetof(AdaptiveModularTokenSlice, stream_id) == 24 &&
              offsetof(AdaptiveModularTokenSlice, width) == 32);
static_assert(std::is_trivially_copyable_v<AdaptiveModularWeightedFeature> &&
              sizeof(AdaptiveModularWeightedFeature) == 16);
namespace {
bool Add(size_t bytes, size_t *sum) {
  if (bytes > SIZE_MAX - *sum)
    return false;
  *sum += bytes;
  return true;
}
bool Align(size_t bytes, size_t *out) {
  if (bytes > SIZE_MAX - 255)
    return false;
  *out = (bytes + 255) & ~size_t{255};
  return true;
}
Status Overflow() { return Status::InvalidArgument("Adaptive Modular storage overflow"); }
} // namespace
Status ResolveAdaptiveModularTreeEligibility(const DecisionTree &tree,
                                             AdaptiveModularTreeEligibility *out) {
  if (!out)
    return Status::InvalidArgument("Null adaptive Modular tree eligibility");
  TreeLayout layout;
  if (auto s = ValidateTree(tree, &layout); !s.ok())
    return s;
  *out = {tree.size, layout.leaves, layout.weighted};
  return Status::Ok();
}
Status ComputeAdaptiveModularTokenStoragePlan(std::span<const ChannelDescriptor> channels,
                                              size_t metadata_channels, size_t slices, size_t nodes,
                                              bool weighted, size_t cpu_participants,
                                              AdaptiveModularTokenStoragePlan *out,
                                              const ModularTokenDeviceLimits *limits) {
  if (!out || !slices || !nodes || !(nodes & 1) || nodes > kMaximumTreeNodes || !cpu_participants ||
      cpu_participants > 256)
    return Status::InvalidArgument("Invalid adaptive Modular storage request");
  if (limits &&
      (!limits->maximum_tokens || !limits->maximum_slices || !limits->maximum_buffer_bytes))
    return Status::InvalidArgument("Invalid adaptive Modular device limits");
  if (auto s = ValidateChannelDescriptors(channels, metadata_channels); !s.ok())
    return s;
  AdaptiveModularTokenStoragePlan p;
  p.slice_count = slices;
  p.node_count = nodes;
  size_t maximum_width = 0;
  for (const auto &channel : channels) {
    size_t area;
    if (!channel.extent.try_area(&area) || area > size_t(PTRDIFF_MAX) / sizeof(int32_t) ||
        !Add(area, &p.token_count) || !Align(p.plane_bytes, &p.plane_bytes) ||
        !Add(area * sizeof(int32_t), &p.plane_bytes))
      return Overflow();
    maximum_width = std::max(maximum_width, channel.extent.width);
  }
  if (!p.token_count || slices > p.token_count ||
      p.token_count > size_t(PTRDIFF_MAX) / sizeof(uint32_t) ||
      (weighted && p.token_count > size_t(PTRDIFF_MAX) / sizeof(AdaptiveModularWeightedFeature)) ||
      slices > size_t(PTRDIFF_MAX) / sizeof(AdaptiveModularTokenSlice))
    return Overflow();
  p.weighted_bytes = weighted ? p.token_count * sizeof(AdaptiveModularWeightedFeature) : 0;
  p.values_bytes = p.token_count * sizeof(uint32_t);
  if (!Align(p.values_bytes, &p.contexts_offset))
    return Overflow();
  p.output_bytes = p.contexts_offset;
  p.slices_bytes = slices * sizeof(AdaptiveModularTokenSlice);
  if (!Add(p.token_count * sizeof(uint16_t), &p.output_bytes) ||
      !Align(p.slices_bytes, &p.nodes_offset))
    return Overflow();
  size_t nodes_end = p.nodes_offset;
  if (!Add(nodes * sizeof(ModularTokenNode), &nodes_end) || !Align(nodes_end, &p.status_offset))
    return Overflow();
  p.dispatch_bytes = p.status_offset;
  if (!Add(sizeof(uint32_t), &p.dispatch_bytes) || p.plane_bytes > PTRDIFF_MAX ||
      p.output_bytes > PTRDIFF_MAX || p.dispatch_bytes > PTRDIFF_MAX)
    return Overflow();
  if (limits && (p.token_count > limits->maximum_tokens || slices > limits->maximum_slices ||
                 p.plane_bytes > limits->maximum_buffer_bytes ||
                 p.weighted_bytes > limits->maximum_buffer_bytes ||
                 p.output_bytes > limits->maximum_buffer_bytes ||
                 p.dispatch_bytes > limits->maximum_buffer_bytes))
    return Status::Unsupported("Adaptive Modular storage exceeds backend limits");
  if (!p.features.Add({p.plane_bytes, p.plane_bytes}) ||
      !p.features.Add({p.weighted_bytes, p.weighted_bytes}) ||
      !p.features.AddVector<AdaptiveModularTokenSlice>(slices, VectorCapacityPolicy::kFreshExact))
    return Overflow();
  HostStorageBound bridge = p.features, image;
  if (auto s = ComputeModularImageStorageBound(channels, metadata_channels, &image); !s.ok())
    return s;
  if (!bridge.Add(image))
    return Overflow();
  if (weighted) {
    if (maximum_width > (SIZE_MAX / 2) - 2 ||
        !bridge.AddVector<AdaptiveModularWeightedFeature>(p.token_count,
                                                          VectorCapacityPolicy::kFreshExact) ||
        !bridge.AddVector<uint32_t>((maximum_width + 2) * 2, VectorCapacityPolicy::kFreshExact,
                                    5 * cpu_participants))
      return Overflow();
  }
  p.bridge_peak_bytes = bridge.peak_bytes;
  HostStorageBound generation = p.features;
  if (!generation.Add({p.output_bytes, p.output_bytes}))
    return Overflow();
  p.retained_bytes = generation.retained_bytes;
  if (!generation.Add({p.dispatch_bytes, p.dispatch_bytes}) ||
      !generation.AddVector<std::byte>(p.dispatch_bytes, VectorCapacityPolicy::kFreshExact))
    return Overflow();
  p.generation_peak_bytes = generation.peak_bytes;
  *out = p;
  return Status::Ok();
}
} // namespace gjxl::modular_gpu_internal
