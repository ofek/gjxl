// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/ops/modular_token_storage_plan.h"
#include <cstddef>
#include <type_traits>

namespace gjxl::modular_gpu_internal {
using namespace modular_internal;
using namespace resource_budget_internal;
static_assert(std::is_standard_layout_v<ModularTokenSlice> &&
              std::is_trivially_copyable_v<ModularTokenSlice>);
static_assert(sizeof(ModularTokenSlice) == 32 && offsetof(ModularTokenSlice, stream_id) == 8 &&
              offsetof(ModularTokenSlice, height) == 28);
static_assert(std::is_standard_layout_v<ModularTokenNode> &&
              std::is_trivially_copyable_v<ModularTokenNode>);
static_assert(sizeof(ModularTokenNode) == 32 && offsetof(ModularTokenNode, offset) == 20 &&
              offsetof(ModularTokenNode, context) == 28);

Status ResolveModularTokenEligibility(const ModularTokenizationOptions &options,
                                      ModularTokenEligibility *out) {
  return ResolveModularTokenEligibility(options.format, options.coding, options.search, out);
}

Status ResolveModularTokenEligibility(PackedModularFormat format, const ModularCodingPolicy &coding,
                                      bool search, ModularTokenEligibility *out) {
  if (!out)
    return Status::InvalidArgument("Null GPU Modular token eligibility output");
  ModularInputProfile source;
  if (auto s = ResolveModularInput({1, 1}, format, &source); !s.ok())
    return s;
  if (auto s = ValidateCodingPolicy(coding); !s.ok())
    return s;
  if (coding.rct && source.channel_count < 3)
    return Status::InvalidArgument("GPU Modular token RCT requires color input");
  TreeLayout tree;
  if (auto s = ValidateTree(coding.tree, &tree); !s.ok())
    return s;
  if (coding.tree.size > kMaximumGpuTreeNodes)
    return Status::Unsupported("GPU Modular tree exceeds backend node capacity");
  if (search || coding.transforms.size || tree.weighted)
    return Status::Unsupported("GPU Modular tokens require prescribed nonweighted RCT-only policy");
  ModularTokenEligibility result;
  result.channel_count = source.channel_count;
  result.context_count = tree.leaves;
  result.node_count = coding.tree.size;
  for (size_t i = 0; i < result.node_count; ++i) {
    const auto &n = coding.tree.nodes[i];
    result.nodes[i] = {
        n.property, n.split,      n.left,         n.right, static_cast<uint32_t>(n.predictor),
        n.offset,   n.multiplier, tree.context[i]};
  }
  const auto &root = result.nodes[0];
  result.gradient_leaf = result.node_count == 1 &&
                         root.predictor == static_cast<uint32_t>(Predictor::kGradient) &&
                         root.offset == 0 && root.multiplier == 1;
  *out = result;
  return Status::Ok();
}

Status CheckModularTokenDeviceStatus(uint32_t status) {
  if (status & ~kModularTokenResidualError)
    return Status::Internal("Unknown GPU Modular token status bits");
  if (status)
    return Status::InvalidArgument("Modular leaf cannot represent residual exactly");
  return Status::Ok();
}

namespace {
bool Align(size_t bytes, size_t *out) {
  if (bytes > SIZE_MAX - (kModularTokenAlignment - 1))
    return false;
  *out = (bytes + kModularTokenAlignment - 1) & ~(kModularTokenAlignment - 1);
  return true;
}
Status Overflow() { return Status::InvalidArgument("GPU Modular token storage overflow"); }
} // namespace

Status ComputeModularTokenStoragePlan(size_t tokens, size_t slices, ModularTokenStoragePlan *out,
                                      const ModularTokenDeviceLimits *limits) {
  if (!out || !tokens || !slices || slices > tokens)
    return Status::InvalidArgument("Invalid GPU Modular token storage request");
  if (limits &&
      (!limits->maximum_tokens || !limits->maximum_slices || !limits->maximum_buffer_bytes))
    return Status::InvalidArgument("Invalid GPU Modular token limits");
  if (tokens > size_t(PTRDIFF_MAX) / sizeof(uint32_t) ||
      slices > size_t(PTRDIFF_MAX) / sizeof(ModularTokenSlice))
    return Overflow();
  ModularTokenStoragePlan p;
  p.token_count = tokens;
  p.slice_count = slices;
  p.values_bytes = tokens * sizeof(uint32_t);
  p.contexts_bytes = tokens * sizeof(uint16_t);
  p.slices_bytes = slices * sizeof(ModularTokenSlice);
  constexpr size_t nodes = sizeof(ModularTokenNode) * kMaximumGpuTreeNodes;
  if (!Align(p.values_bytes, &p.contexts_offset) ||
      p.contexts_bytes > SIZE_MAX - p.contexts_offset || !Align(p.slices_bytes, &p.nodes_offset) ||
      nodes > SIZE_MAX - p.nodes_offset || !Align(p.nodes_offset + nodes, &p.status_offset) ||
      sizeof(uint32_t) > SIZE_MAX - p.status_offset)
    return Overflow();
  p.output_bytes = p.contexts_offset + p.contexts_bytes;
  p.dispatch_bytes = p.status_offset + sizeof(uint32_t);
  if (p.output_bytes > PTRDIFF_MAX || p.dispatch_bytes > PTRDIFF_MAX ||
      !p.staging.AddVector<std::byte>(p.dispatch_bytes, VectorCapacityPolicy::kFreshExact))
    return Overflow();
  if (limits && (tokens > limits->maximum_tokens || slices > limits->maximum_slices ||
                 p.output_bytes > limits->maximum_buffer_bytes ||
                 p.dispatch_bytes > limits->maximum_buffer_bytes))
    return Status::Unsupported("GPU Modular token storage exceeds backend limits");
  HostStorageBound peak = p.staging;
  if (!peak.Add({p.output_bytes, p.output_bytes}) ||
      !peak.Add({p.dispatch_bytes, p.dispatch_bytes}))
    return Overflow();
  p.production_peak_bytes = peak.peak_bytes;
  p.retained_bytes = p.output_bytes;
  *out = p;
  return Status::Ok();
}
} // namespace gjxl::modular_gpu_internal
