// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/gpu_token_handoff.h"
#include <stdexcept>

namespace gjxl::modular_gpu_internal {
using namespace modular_internal;
using namespace resource_budget_internal;

Status ComputeGpuModularTokenHandoffStorageBound(size_t streams, HostStorageBound *out) {
  if (!out || !streams)
    return Status::InvalidArgument("Invalid GPU Modular token handoff storage request");
  HostStorageBound result;
  if (!result.AddVector<EntropyTokenStreamView>(streams, VectorCapacityPolicy::kFreshExact))
    return Status::InvalidArgument("GPU Modular token handoff storage overflow");
  *out = result;
  return Status::Ok();
}

Status AdoptGpuModularTokens(const GpuModularTokenPlan &plan,
                             std::unique_ptr<PreparedGpuModularTokens> &&backing,
                             PreparedModularTokens *out) try {
  if (!out || !backing || &backing == &out->gpu_backing)
    return Status::InvalidArgument("Invalid GPU Modular token handoff");
  GpuModularTokenStoragePlan expected;
  if (auto s = ComputeGpuModularTokenStoragePlan(plan.options, &expected); !s.ok())
    return s;
  if (plan.storage != expected || plan.layout.geometry != expected.geometry ||
      plan.layout.streams.size() != expected.streams.streams ||
      plan.ranges.size() != expected.streams.streams ||
      backing->token_count() != expected.tokens.token_count ||
      backing->context_count() != expected.eligibility.context_count)
    return Status::InvalidArgument("GPU Modular token plan/owner mismatch");
  size_t next = 0;
  for (size_t i = 0; i < plan.ranges.size(); ++i) {
    const auto range = plan.ranges[i];
    const auto &stream = plan.layout.streams[i];
    if (range.offset != next || range.count > expected.tokens.token_count - next ||
        stream.slice_begin > plan.layout.slices.size() ||
        stream.slice_count > plan.layout.slices.size() - stream.slice_begin)
      return Status::InvalidArgument("Invalid GPU Modular token stream range");
    size_t count = 0;
    for (const auto &slice :
         std::span(plan.layout.slices).subspan(stream.slice_begin, stream.slice_count)) {
      size_t area;
      if (!slice.rect.extent.try_area(&area) || area > range.count - count)
        return Status::InvalidArgument("GPU Modular token slice coverage mismatch");
      count += area;
    }
    if (count != range.count)
      return Status::InvalidArgument("GPU Modular token stream coverage mismatch");
    next += range.count;
  }
  if (next != expected.tokens.token_count)
    return Status::InvalidArgument("Incomplete GPU Modular token coverage");
  ModularTokenHostView view;
  if (auto s = backing->BorrowHostView(&view); !s.ok())
    return s;
  if (view.values.size() != next || view.contexts.size() != next || !view.values.data() ||
      !view.contexts.data())
    return Status::InvalidArgument("GPU Modular token host view mismatch");
  TreeLayout tree;
  if (auto s = ValidateTree(plan.options.tokenization.coding.tree, &tree); !s.ok())
    return s;
  auto result = PrepareModularTokenHeader(plan.options.tokenization.coding, tree);
  const ManagedHostScope resources(ResourceClass::kSerializer);
  result.streams.resize(plan.ranges.size());
  for (size_t i = 0; i < plan.ranges.size(); ++i) {
    const auto range = plan.ranges[i];
    result.streams[i] =
        EntropyTokenStreamView::Split(view.values.subspan(range.offset, range.count),
                                      view.contexts.subspan(range.offset, range.count));
  }
  result.gpu_backing = std::move(backing);
  *out = std::move(result);
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("GPU Modular token handoff allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("GPU Modular token handoff storage overflow");
}
Status AdoptAdaptiveGpuModularTokens(
    const ModularStreamPlan &layout, const ModularCodingPolicy &policy,
    std::unique_ptr<PreparedGpuModularTokens> &&backing,
    PreparedModularTokens *out) try {
  if (!out || !backing || &backing == &out->gpu_backing ||
      layout.streams.empty())
    return Status::InvalidArgument(
        "Invalid adaptive GPU Modular token handoff");
  if (!policy.weighted.valid() || policy.rct >= 42)
    return Status::InvalidArgument("Invalid adaptive GPU Modular coding policy");
  if (auto s = ValidateTransforms(policy.transforms); !s.ok())
    return s;
  TreeLayout tree;
  if (auto s = ValidateTree(policy.tree, &tree); !s.ok())
    return s;
  if (backing->context_count() != tree.leaves)
    return Status::InvalidArgument(
        "Adaptive GPU Modular context count mismatch");
  size_t total = 0, next_slice = 0;
  for (const auto &stream : layout.streams) {
    if (stream.slice_begin != next_slice || next_slice > layout.slices.size() ||
        stream.slice_count > layout.slices.size() - next_slice)
      return Status::InvalidArgument(
          "Invalid adaptive GPU Modular stream slices");
    for (const auto &slice :
         std::span(layout.slices).subspan(next_slice, stream.slice_count)) {
      size_t area;
      if (!slice.rect.extent.try_area(&area) ||
          total > backing->token_count() ||
          area > backing->token_count() - total)
        return Status::InvalidArgument(
            "Invalid adaptive GPU Modular slice coverage");
      total += area;
    }
    next_slice += stream.slice_count;
  }
  if (next_slice != layout.slices.size() || total != backing->token_count() ||
      !total)
    return Status::InvalidArgument(
        "Incomplete adaptive GPU Modular token coverage");
  ModularTokenHostView view;
  if (auto s = backing->BorrowHostView(&view); !s.ok())
    return s;
  if (view.values.size() != total || view.contexts.size() != total ||
      !view.values.data() || !view.contexts.data())
    return Status::InvalidArgument(
        "Adaptive GPU Modular token host view mismatch");
  const ManagedHostScope resources(ResourceClass::kSerializer);
  auto result = PrepareModularTokenHeader(policy, tree);
  result.streams.resize(layout.streams.size());
  size_t next = 0;
  for (size_t i = 0; i < layout.streams.size(); ++i) {
    const auto &stream = layout.streams[i];
    size_t count = 0;
    for (const auto &slice :
         std::span(layout.slices)
             .subspan(stream.slice_begin, stream.slice_count))
      count += slice.rect.extent.width * slice.rect.extent.height;
    result.streams[i] = EntropyTokenStreamView::Split(
        view.values.subspan(next, count), view.contexts.subspan(next, count));
    next += count;
  }
  result.gpu_backing = std::move(backing);
  *out = std::move(result);
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory(
      "Adaptive GPU Modular token handoff allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument(
      "Adaptive GPU Modular token handoff storage overflow");
}
} // namespace gjxl::modular_gpu_internal
