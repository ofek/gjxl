// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/gpu_token_plan.h"
#include <stdexcept>

namespace gjxl::modular_gpu_internal {
using namespace modular_internal;
using namespace resource_budget_internal;
namespace {
Status Overflow() {
  return Status::InvalidArgument("GPU Modular token preparation storage overflow");
}
} // namespace

Status ComputeGpuModularTokenStoragePlan(const GpuModularTokenPlanOptions &options,
                                         GpuModularTokenStoragePlan *out,
                                         const ModularInputDeviceLimits *input_limits,
                                         const ModularTokenDeviceLimits *token_limits) {
  if (!out)
    return Status::InvalidArgument("Null GPU Modular token preparation plan");
  GpuModularTokenStoragePlan p;
  if (auto s = ResolveModularTokenEligibility(options.tokenization, &p.eligibility); !s.ok())
    return s;
  const ModularInputStorageOptions input{
      options.extent,     options.row_stride_bytes,        options.tokenization.format,
      options.byte_order, options.tokenization.coding.rct, false};
  if (auto s = ComputeModularInputStoragePlan(input, &p.input, input_limits); !s.ok())
    return s;
  if (auto s = ModularFrameGeometry::Create(options.extent, &p.geometry); !s.ok())
    return s;
  if (auto s = ComputeModularStreamStoragePlan(
          p.geometry, std::span(p.input.channels).first(p.input.channel_count), 0, &p.streams);
      !s.ok())
    return s;
  size_t area;
  if (!options.extent.try_area(&area) || area > SIZE_MAX / p.input.channel_count)
    return Overflow();
  // For unshifted equal-size source channels the canonical slice bound is exact:
  // all channels are global, or every channel contributes to every 256px group.
  if (auto s = ComputeModularTokenStoragePlan(area * p.input.channel_count,
                                              p.streams.maximum_slices, &p.tokens, token_limits);
      !s.ok())
    return s;
  p.host_layout = p.streams.owned;
  if (!p.host_layout.AddVector<ModularTokenSlice>(p.tokens.slice_count,
                                                  VectorCapacityPolicy::kFreshExact) ||
      !p.host_layout.AddVector<ModularTokenRange>(p.streams.streams,
                                                  VectorCapacityPolicy::kFreshExact))
    return Overflow();
  HostStorageBound live = p.host_layout;
  if (!live.Add({p.input.preparation_peak_bytes, p.input.preparation_peak_bytes}) ||
      !live.Add({p.tokens.production_peak_bytes, p.tokens.production_peak_bytes}))
    return Overflow();
  p.generation_peak_bytes = live.peak_bytes;
  if (p.tokens.retained_bytes > SIZE_MAX - p.host_layout.retained_bytes)
    return Overflow();
  p.retained_bytes = p.tokens.retained_bytes + p.host_layout.retained_bytes;
  *out = p;
  return Status::Ok();
}

Status BuildGpuModularTokenPlan(const GpuModularTokenPlanOptions &options, GpuModularTokenPlan *out,
                                const ModularInputDeviceLimits *input_limits,
                                const ModularTokenDeviceLimits *token_limits) try {
  if (!out)
    return Status::InvalidArgument("Null GPU Modular token layout output");
  GpuModularTokenPlan result;
  if (auto s =
          ComputeGpuModularTokenStoragePlan(options, &result.storage, input_limits, token_limits);
      !s.ok())
    return s;
  result.options = options;
  result.metadata = result.storage.input.metadata;
  if (options.tokenization.coding != ModularCodingPolicy{})
    result.metadata.modular_16_bit_buffer_sufficient = false;
  const ManagedHostScope resources(ResourceClass::kPreparation);
  const auto &p = result.storage;
  if (auto s = BuildModularStreamPlan(
          p.geometry, std::span(p.input.channels).first(p.input.channel_count), 0, &result.layout);
      !s.ok())
    return s;
  result.slices.reserve(p.tokens.slice_count);
  result.ranges.resize(p.streams.streams);
  size_t next = 0;
  for (size_t i = 0; i < result.layout.streams.size(); ++i) {
    const auto &stream = result.layout.streams[i];
    const size_t begin = next;
    for (const auto &slice :
         std::span(result.layout.slices).subspan(stream.slice_begin, stream.slice_count)) {
      const auto &r = slice.rect;
      size_t count;
      if (!r.extent.try_area(&count) || next > p.tokens.token_count ||
          count > p.tokens.token_count - next)
        return Status::Internal("Canonical GPU Modular token coverage overflow");
      // Canonical geometry bounds all coordinates to <= kMaximumDimension and
      // IDs to INT32_MAX; only token offsets require 64 bits in the device ABI.
      result.slices.push_back({next, static_cast<uint32_t>(stream.id),
                               static_cast<uint32_t>(slice.channel), static_cast<uint32_t>(r.x),
                               static_cast<uint32_t>(r.y), static_cast<uint32_t>(r.extent.width),
                               static_cast<uint32_t>(r.extent.height)});
      next += count;
    }
    result.ranges[i] = {begin, next - begin};
  }
  if (next != p.tokens.token_count || result.slices.size() != p.tokens.slice_count)
    return Status::Internal("Canonical GPU Modular token coverage mismatch");
  *out = std::move(result);
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("GPU Modular token layout allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("GPU Modular token layout storage overflow");
}
} // namespace gjxl::modular_gpu_internal
