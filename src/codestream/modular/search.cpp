// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/search.h"
#include "codec/modular/profile.h"
#include <algorithm>
namespace gjxl::modular_internal {
using namespace resource_budget_internal;
Status ProbePalette(PackedModularImageView input, uint16_t *colors) {
  ProfileScope profile_scope(ProfileStage::kTraining);
  if (!colors)
    return Status::InvalidArgument("Null palette probe output");
  if (auto s = input.Validate(); !s.ok())
    return s;
  ModularInputProfile p;
  if (auto s = ResolveModularInput(input.extent, input.format, &p); !s.ok())
    return s;
  std::array<std::array<int32_t, 4>, kMaximumPaletteColors> entries{};
  size_t used = 0;
  for (size_t y = 0; y < input.extent.height; ++y)
    for (size_t x = 0; x < input.extent.width; ++x) {
      std::array<int32_t, 4> color{};
      for (size_t c = 0; c < p.channel_count; ++c) {
        const size_t i = y * input.row_stride + (x * p.channel_count + c) * p.bytes_per_sample;
        uint32_t value = input.bytes[i];
        if (p.bytes_per_sample == 2)
          value = input.byte_order == SampleByteOrder::kLittleEndian
                      ? value | (uint32_t{input.bytes[i + 1]} << 8)
                      : (value << 8) | input.bytes[i + 1];
        color[c] = static_cast<int32_t>(value);
      }
      auto pos = std::lower_bound(entries.begin(), entries.begin() + used, color);
      if (pos != entries.begin() + used && *pos == color)
        continue;
      if (used == kMaximumPaletteColors) {
        *colors = 0;
        return Status::Ok();
      }
      std::move_backward(pos, entries.begin() + used, entries.begin() + used + 1);
      *pos = color;
      ++used;
    }
  *colors = static_cast<uint16_t>(used);
  return Status::Ok();
}
Status BuildTransformCandidates(Extent2D extent, PackedModularFormat format, uint16_t colors,
                                std::array<ModularCodingPolicy, 4> *out, size_t *count) {
  if (!out || !count || colors > kMaximumPaletteColors)
    return Status::InvalidArgument("Invalid transform search output");
  ModularInputProfile p;
  if (auto s = ResolveModularInput(extent, format, &p); !s.ok())
    return s;
  std::array<ModularCodingPolicy, 4> candidates{};
  size_t n = 0;
  const auto channels = static_cast<uint8_t>(p.channel_count);
  if (colors) {
    auto &policy = candidates[n++];
    policy.transforms.size = 1;
    policy.transforms.entries[0] = {TransformKind::kPalette, 0, channels, colors};
  }
  for (unsigned kind = 0; kind < 3; ++kind) {
    if ((kind != 1 && extent.width < 2) || (kind != 0 && extent.height < 2))
      continue;
    auto &policy = candidates[n++];
    policy.rct = channels >= 3 ? 6 : 0;
    if (kind != 1)
      policy.transforms.entries[policy.transforms.size++] = {
          TransformKind::kSqueeze, 0, channels, 1, true, false};
    if (kind != 0)
      policy.transforms.entries[policy.transforms.size++] = {
          TransformKind::kSqueeze, 0, channels, 1, false, false};
  }
  *out = candidates;
  *count = n;
  return Status::Ok();
}
Status ComputeModularSearchStoragePlan(Extent2D extent, PackedModularFormat format,
                                       EntropyCodingMode mode, ModularWorkflowStoragePlan *out,
                                       size_t cpu_threads) {
  if (!out)
    return Status::InvalidArgument("Null Modular search plan");
  ModularCodingPolicy largest;
  if (auto s = MakeFixedWeightedTree(16, &largest.tree); !s.ok())
    return s;
  ModularWorkflowStoragePlan plan, baseline;
  if (auto s = ComputeModularWorkflowStoragePlan(extent, format, mode, largest, &plan, cpu_threads);
      !s.ok())
    return s;
  if (auto s = ComputeModularWorkflowStoragePlan(extent, format, mode, {}, &baseline, cpu_threads);
      !s.ok())
    return s;
  // Training keeps prepared planes, layout, and sample records, with one WP
  // state at a time. Preparation's token allocation is conservatively retained
  // in this bound, although actual training does not allocate tokens.
  HostStorageBound learning;
  if (auto s = ComputeTreeLearningStorageBound(&learning); !s.ok())
    return s;
  if (!plan.preparation.AddVector<TrainingSample>(std::min(plan.tokens, kMaximumTrainingSamples),
                                                  VectorCapacityPolicy::kFreshExact, 2) ||
      !plan.preparation.Add(learning))
    return Status::InvalidArgument("Modular search storage overflow");
  plan.working.peak_bytes =
      std::max({plan.working.peak_bytes, plan.preparation.peak_bytes, baseline.working.peak_bytes});
  std::array<ModularCodingPolicy, 4> candidates;
  size_t count;
  if (auto s = BuildTransformCandidates(extent, format, kMaximumPaletteColors, &candidates, &count);
      !s.ok())
    return s;
  for (size_t i = 0; i < count; ++i) {
    ModularWorkflowStoragePlan candidate;
    if (auto s = ComputeModularWorkflowStoragePlan(extent, format, mode, candidates[i], &candidate,
                                                   cpu_threads);
        !s.ok())
      return s;
    plan.working.peak_bytes = std::max(plan.working.peak_bytes, candidate.working.peak_bytes);
    plan.maximum_codestream_bytes =
        std::max(plan.maximum_codestream_bytes, candidate.maximum_codestream_bytes);
    if (candidate.output.peak_bytes > plan.output.peak_bytes)
      plan.output = candidate.output;
  }
  // A winning complete codestream coexists with each following training/encode.
  if (plan.output.peak_bytes > SIZE_MAX - plan.working.peak_bytes)
    return Status::InvalidArgument("Modular search candidate overlap overflow");
  plan.working.peak_bytes += plan.output.peak_bytes;
  *out = plan;
  return Status::Ok();
}
Status LearnModularPolicies(const ModularEncoderFrame &frame, const ModularStreamPlan &plan,
                            uint8_t rct, ModularCodingPolicy *single, ModularCodingPolicy *split,
                            ModularCodingPolicy *entropy) {
  ProfileScope profile_scope(ProfileStage::kTraining);
  if (!single || !split || rct >= 42 || plan.geometry.source() != frame.metadata().extent)
    return Status::InvalidArgument("Null Modular learned policy");
  size_t total = 0;
  for (size_t c = 0; c < frame.image().channel_count(); ++c) {
    size_t area;
    if (!frame.image().view(c).descriptor.extent.try_area(&area) || area > SIZE_MAX - total)
      return Status::InvalidArgument("Invalid Modular training frame");
    total += area;
  }
  if (!total)
    return Status::InvalidArgument("Empty Modular training frame");
  const size_t stride = total / kMaximumTrainingSamples + (total % kMaximumTrainingSamples != 0);
  ManagedVector<TrainingSample> samples;
  samples.resize(std::min(total, kMaximumTrainingSamples));
  ManagedVector<TrainingSample> stratified;
  if (entropy)
    stratified.resize(samples.size());
  // Give every nonempty stream/channel slice one sample when the budget
  // permits, then distribute the remaining samples by area. Within each
  // stratum, a deterministic hash avoids regular-stride aliasing on patterns.
  size_t slices = 0;
  for (const auto &slice : plan.slices)
    slices += slice.rect.extent.width != 0 && slice.rect.extent.height != 0;
  const size_t guaranteed = slices <= stratified.size() ? slices : 0;
  const size_t proportional = stratified.size() - guaranteed;
  if (entropy && total > SIZE_MAX / kMaximumTrainingSamples)
    return Status::InvalidArgument("Modular sampling extent overflow");
  auto mix = [](uint64_t value) {
    value = (value ^ (value >> 30)) * uint64_t{0xbf58476d1ce4e5b9};
    value = (value ^ (value >> 27)) * uint64_t{0x94d049bb133111eb};
    return value ^ (value >> 31);
  };
  size_t sampled = 0, sampled_slices = 0;
  size_t ordinal = 0, used = 0;
  for (const auto &stream : plan.streams) {
    if (stream.slice_begin > plan.slices.size() ||
        stream.slice_count > plan.slices.size() - stream.slice_begin)
      return Status::InvalidArgument("Invalid Modular training slices");
    for (const auto &slice :
         std::span(plan.slices).subspan(stream.slice_begin, stream.slice_count)) {
      if (slice.channel >= frame.image().channel_count())
        return Status::InvalidArgument("Invalid Modular training channel");
      ModularChannelView view;
      if (auto s = BorrowChannelSlice(frame.image().view(slice.channel), slice.rect, &view);
          !s.ok())
        return s;
      const auto [w, h] = view.descriptor.extent;
      const size_t area = w * h;
      if (ordinal > total || area > total - ordinal)
        return Status::InvalidArgument("Overlapping Modular training slices");
      // Distribute only the excess area after the guaranteed sample. This
      // keeps quotas within a slice even when many slices contain one pixel.
      const size_t weighted_total = total - guaranteed;
      const size_t weighted_begin = ordinal - (guaranteed ? sampled_slices : 0);
      const size_t weighted_area = area - (guaranteed && area ? 1 : 0);
      size_t quota = 0;
      if (entropy && area) {
        quota = guaranteed ? 1 : 0;
        if (weighted_total)
          quota += (weighted_begin + weighted_area) * proportional / weighted_total -
                   weighted_begin * proportional / weighted_total;
      }
      auto position = [&](size_t index) -> size_t {
        if (index == quota)
          return area;
        const size_t begin = index * area / quota, end = (index + 1) * area / quota;
        return begin + mix(uint64_t{stream.id} * 0x9e3779b97f4a7c15ULL +
                           uint64_t{slice.channel} * 0xd1b54a32d192ed03ULL + index) %
                           (end - begin);
      };
      size_t local_sample = 0, next_sample = position(0), local = 0;
      WeightedPredictor<ResourceClass::kPreparation> wp(w);
      for (size_t y = 0; y < h; ++y) {
        int64_t previous_gradient = 0;
        for (size_t x = 0; x < w; ++x, ++ordinal, ++local) {
          const auto n = Neighbors(view, x, y);
          const auto weighted = wp.Predict(x, y, n.top, n.left, n.top_right, n.top_left, n.top_top);
          const auto properties =
              Properties(n, slice.channel, stream.id, x, y, previous_gradient, weighted.second);
          previous_gradient = properties[9];
          const bool original_sample = ordinal % stride == 0;
          const bool entropy_sample = entropy && local == next_sample;
          if (original_sample || entropy_sample) {
            if ((original_sample && used == samples.size()) ||
                (entropy_sample && sampled == stratified.size()))
              return Status::InvalidArgument("Overlapping Modular training samples");
            TrainingSample sample;
            sample.properties = properties;
            for (unsigned p = 0; p < 14; ++p) {
              const int64_t residual =
                  int64_t{view.Row(y)[x]} - Predict(static_cast<Predictor>(p), n, weighted.first);
              if (residual < INT32_MIN || residual > INT32_MAX)
                return Status::Unsupported("Modular training residual exceeds supported range");
              sample.residuals[p] = static_cast<int32_t>(residual);
            }
            if (original_sample)
              samples[used++] = sample;
            if (entropy_sample) {
              stratified[sampled++] = sample;
              next_sample = position(++local_sample);
            }
          }
          if (!wp.Update(view.Row(y)[x], x, y))
            return Status::Unsupported("Modular training weighted state exceeds supported range");
        }
      }
      sampled_slices += area != 0;
    }
  }
  if (ordinal != total || !used)
    return Status::InvalidArgument("Incomplete Modular training coverage");
  ModularCodingPolicy leaf, branch;
  leaf.rct = branch.rct = rct;
  if (auto status = LearnDecisionTrees(std::span(samples).first(used), &leaf.tree, &branch.tree);
      !status.ok())
    return status;
  ModularCodingPolicy learned;
  learned.rct = rct;
  if (entropy) {
    if (sampled != stratified.size())
      return Status::InvalidArgument("Incomplete Modular entropy sampling coverage");
    if (auto s = LearnEntropyTree(stratified, &learned.tree); !s.ok())
      return s;
    *entropy = learned;
  }
  *single = leaf;
  *split = branch;
  return Status::Ok();
}
} // namespace gjxl::modular_internal
