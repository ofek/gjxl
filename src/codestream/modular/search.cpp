// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/search.h"
#include "codec/modular/entropy_estimate.h"
#include "codec/modular/profile.h"
#include "codec/modular/transform/rct.h"
#include <algorithm>
#include <new>
#include <numeric>
namespace gjxl::modular_internal {
using namespace resource_budget_internal;
namespace {
uint64_t Mix(uint64_t value) {
  value = (value ^ (value >> 30)) * uint64_t{0xbf58476d1ce4e5b9};
  value = (value ^ (value >> 27)) * uint64_t{0x94d049bb133111eb};
  return value ^ (value >> 31);
}
struct RctSample {
  // Current, left, top, top-left, with stream-boundary fallbacks applied.
  std::array<std::array<int32_t, 3>, 4> rgb{};
};
struct RctWorkspace {
  std::array<RctSample, kMaximumRctSamples> samples{};
  std::array<uint32_t, kMaximumRctSamples + 1> count_log_count{};
  std::array<std::array<std::array<uint16_t, 128>, 3>, 3> counts{};
  std::array<std::array<int64_t, 3>, 3> extra_bits{};
};
} // namespace
Status ComputeRctRankingStorageBound(HostStorageBound *out) {
  if (!out)
    return Status::InvalidArgument("Null Modular RCT ranking storage output");
  HostStorageBound bound;
  if (!bound.AddVector<RctWorkspace>(1, VectorCapacityPolicy::kFreshExact))
    return Status::InvalidArgument("Modular RCT ranking storage overflow");
  *out = bound;
  return Status::Ok();
}
Status RankModularRcts(PackedModularImageView input, RctRanking *out) try {
  ProfileScope profile_scope(ProfileStage::kTraining);
  if (!out)
    return Status::InvalidArgument("Null Modular RCT ranking output");
  if (auto s = input.Validate(); !s.ok())
    return s;
  ModularInputProfile profile;
  if (auto s = ResolveModularInput(input.extent, input.format, &profile); !s.ok())
    return s;
  RctRanking result;
  if (profile.channel_count < 3) {
    *out = result;
    return Status::Ok();
  }
  ManagedVector<RctWorkspace> scratch(1);
  auto &w = scratch[0];
  const size_t width = input.extent.width, area = width * input.extent.height;
  result.samples = std::min(area, kMaximumRctSamples);
  auto read = [&](size_t x, size_t y) {
    std::array<int32_t, 3> rgb;
    for (size_t c = 0; c < 3; ++c) {
      const size_t offset = y * input.row_stride +
                            (x * profile.channel_count + c) * profile.bytes_per_sample;
      uint32_t value = input.bytes[offset];
      if (profile.bytes_per_sample == 2)
        value = input.byte_order == SampleByteOrder::kLittleEndian
                    ? value | (uint32_t{input.bytes[offset + 1]} << 8)
                    : (value << 8) | input.bytes[offset + 1];
      rgb[c] = static_cast<int32_t>(value);
    }
    return rgb;
  };
  // Divide before multiplying to keep strata arithmetic valid on 32-bit hosts.
  auto boundary = [&](size_t i) {
    return (area / result.samples) * i + (area % result.samples) * i / result.samples;
  };
  for (size_t i = 0; i < result.samples; ++i) {
    const size_t begin = boundary(i), end = boundary(i + 1);
    const size_t pixel = begin + Mix(i) % (end - begin);
    const size_t x = pixel % width, y = pixel / width;
    const bool left = x % kGroupDimension != 0, top = y % kGroupDimension != 0;
    auto &s = w.samples[i].rgb;
    s[0] = read(x, y);
    if (left)
      s[1] = read(x - 1, y);
    else if (top)
      s[1] = read(x, y - 1);
    s[2] = top ? read(x, y - 1) : s[1];
    s[3] = left && top ? read(x - 1, y - 1) : s[1];
  }
  for (uint32_t n = 1; n <= result.samples; ++n)
    w.count_log_count[n] = n * entropy_estimate::Log2(n);
  std::array<std::array<int64_t, 3>, 42> profiles;
  for (uint8_t rct = 0; rct < 42; ++rct) {
    w.counts = {};
    w.extra_bits = {};
    for (size_t i = 0; i < result.samples; ++i) {
      std::array<std::array<int32_t, 3>, 4> values;
      for (size_t n = 0; n < values.size(); ++n)
        values[n] = ForwardRct(w.samples[i].rgb[n], rct);
      for (size_t c = 0; c < 3; ++c) {
        const int32_t left = values[1][c], top = values[2][c];
        const int32_t gradient = std::clamp(left + top - values[3][c], std::min(left, top),
                                          std::max(left, top));
        const std::array<int32_t, 3> predictors{gradient, left, top};
        for (size_t p = 0; p < predictors.size(); ++p) {
          const auto code = entropy_estimate::EncodeResidual(values[0][c] - predictors[p]);
          ++w.counts[c][p][code.symbol];
          w.extra_bits[c][p] += code.extra_bits;
        }
      }
    }
    for (size_t c = 0; c < 3; ++c) {
      int64_t best = INT64_MAX;
      for (size_t p = 0; p < 3; ++p) {
        int64_t cost = w.count_log_count[result.samples] + (w.extra_bits[c][p] + 8) * 4096;
        for (auto count : w.counts[c][p])
          if (count)
            cost += int64_t{2 * 4096} - w.count_log_count[count];
        best = std::min(best, cost);
      }
      profiles[rct][c] = best;
      result.costs[rct] += best;
    }
    // Channel permutations often have identical estimates. This is heuristic
    // deduplication, not a claim that their final learned codestreams are equal.
    std::sort(profiles[rct].begin(), profiles[rct].end());
  }
  std::array<uint8_t, 42> order;
  std::iota(order.begin(), order.end(), uint8_t{0});
  std::sort(order.begin(), order.end(), [&](uint8_t a, uint8_t b) {
    return result.costs[a] != result.costs[b] ? result.costs[a] < result.costs[b] : a < b;
  });
  for (uint8_t rct : order) {
    bool duplicate = profiles[rct] == profiles[0] || profiles[rct] == profiles[6] ||
                     profiles[rct] == profiles[9];
    for (size_t i = 0; i < result.count; ++i)
      duplicate |= profiles[rct] == profiles[result.shortlist[i]];
    if (!duplicate) {
      result.shortlist[result.count++] = rct;
      if (result.count == result.shortlist.size())
        break;
    }
  }
  *out = result;
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Modular RCT ranking allocation failed");
}
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
                                       size_t cpu_threads, TreeLearningOptions learning_options) {
  if (!out)
    return Status::InvalidArgument("Null Modular search plan");
  if (auto s = ValidateTreeLearningOptions(learning_options); !s.ok())
    return s;
  ModularInputProfile source;
  if (auto s = ResolveModularInput(extent, format, &source); !s.ok())
    return s;
  learning_options.maximum_leaves = std::min(
      learning_options.maximum_leaves,
      MaximumGlobalTreeLeaves(uint64_t{extent.width} * extent.height * source.channel_count));
  const size_t maximum_leaves = std::max(kDefaultTreeLeaves, learning_options.maximum_leaves);
  ModularCodingPolicy largest;
  if (auto s = MakeFixedWeightedTree(16, &largest.tree); !s.ok())
    return s;
  ModularWorkflowStoragePlan plan, baseline;
  if (auto s = ComputeModularWorkflowStoragePlan(extent, format, mode, largest, &plan, cpu_threads,
                                                 maximum_leaves);
      !s.ok())
    return s;
  if (auto s = ComputeModularWorkflowStoragePlan(extent, format, mode, {}, &baseline, cpu_threads);
      !s.ok())
    return s;
  // Training keeps prepared planes, layout, and sample records, with one WP
  // state at a time. Preparation's token allocation is conservatively retained
  // in this bound, although actual training does not allocate tokens.
  HostStorageBound learning, ranking;
  if (auto s = ComputeTreeLearningStorageBound(&learning, learning_options); !s.ok())
    return s;
  HostStorageBound retained_learning;
  if (auto s = ComputeTreeLearningStorageBound(&retained_learning); !s.ok())
    return s;
  learning.peak_bytes = std::max(learning.peak_bytes, retained_learning.peak_bytes);
  if (auto s = ComputeRctRankingStorageBound(&ranking); !s.ok())
    return s;
  auto add_learning = [&](ModularWorkflowStoragePlan &p) -> Status {
    if (!p.preparation.AddVector<TrainingSample>(std::min(p.tokens, kDefaultTrainingSamples),
                                                 VectorCapacityPolicy::kFreshExact) ||
        !p.preparation.AddVector<TrainingSample>(
            std::min(p.tokens, std::max(kDefaultTrainingSamples, learning_options.maximum_samples)),
            VectorCapacityPolicy::kFreshExact) ||
        !p.preparation.Add(learning))
      return Status::InvalidArgument("Modular search storage overflow");
    p.working.peak_bytes = std::max(p.working.peak_bytes, p.preparation.peak_bytes);
    return Status::Ok();
  };
  if (auto s = add_learning(plan); !s.ok())
    return s;
  plan.working.peak_bytes =
      std::max({plan.working.peak_bytes, plan.preparation.peak_bytes, baseline.working.peak_bytes,
                ranking.peak_bytes});
  std::array<ModularCodingPolicy, 4> candidates;
  size_t count;
  if (auto s = BuildTransformCandidates(extent, format, kMaximumPaletteColors, &candidates, &count);
      !s.ok())
    return s;
  for (size_t i = 0; i < count; ++i) {
    // Transformed candidates may now learn any supported tree. Include the
    // largest context model and WP rows, plus samples of the transformed
    // token count (palette metadata can exceed the original pixel count).
    candidates[i].tree = largest.tree;
    ModularWorkflowStoragePlan candidate;
    if (auto s = ComputeModularWorkflowStoragePlan(extent, format, mode, candidates[i], &candidate,
                                                   cpu_threads, maximum_leaves);
        !s.ok())
      return s;
    if (auto s = add_learning(candidate); !s.ok())
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
  // Large learned policies can coexist in six context candidates, the options,
  // winning/profile policies, learner outputs and temporary replacement copies.
  // Sixteen fresh exact node backings conservatively cover that overlap; the
  // per-candidate plan already covers token-header storage. Inline trees add 0.
  const size_t nodes = 2 * maximum_leaves - 1;
  HostStorageBound policies;
  if (nodes > kInlineTreeNodes &&
      !policies.AddVector<DecisionNode>(nodes, VectorCapacityPolicy::kFreshExact, 16))
    return Status::InvalidArgument("Modular search policy storage overflow");
  if (!plan.working.Add(policies))
    return Status::InvalidArgument("Modular search policy overlap overflow");
  *out = plan;
  return Status::Ok();
}
Status LearnModularPolicies(const ModularEncoderFrame &frame, const ModularStreamPlan &plan,
                            uint8_t rct, ModularCodingPolicy *single, ModularCodingPolicy *split,
                            ModularCodingPolicy *entropy, TreeLearningOptions learning,
                            TrainingCoverage *coverage) {
  ModularCodingPolicy base;
  base.rct = rct;
  return LearnModularPolicies(frame, plan, base, single, split, entropy, learning, coverage);
}
Status LearnModularPolicies(const ModularEncoderFrame &frame, const ModularStreamPlan &plan,
                            const ModularCodingPolicy &base, ModularCodingPolicy *single,
                            ModularCodingPolicy *split, ModularCodingPolicy *entropy,
                            TreeLearningOptions learning, TrainingCoverage *coverage) try {
  if (auto s = ValidateTreeLearningOptions(learning); !s.ok())
    return s;
  const auto &metadata = frame.metadata();
  const size_t source_channels =
      (metadata.color == codec_internal::SourceColor::kGraySrgb ? 1 : 3) +
      static_cast<size_t>(metadata.alpha.has_value());
  learning.maximum_leaves = std::min(
      learning.maximum_leaves, MaximumGlobalTreeLeaves(uint64_t{metadata.extent.width} *
                                                       metadata.extent.height * source_channels));
  ProfileScope profile_scope(ProfileStage::kTraining);
  if (!single || !split || plan.geometry.source() != frame.metadata().extent)
    return Status::InvalidArgument("Null Modular learned policy");
  if (auto s = ValidateCodingPolicy(base); !s.ok())
    return s;
  size_t total = 0;
  for (size_t c = 0; c < frame.image().channel_count(); ++c) {
    size_t area;
    if (!frame.image().view(c).descriptor.extent.try_area(&area) || area > SIZE_MAX - total)
      return Status::InvalidArgument("Invalid Modular training frame");
    total += area;
  }
  if (!total)
    return Status::InvalidArgument("Empty Modular training frame");
  const size_t stride = total / kDefaultTrainingSamples + (total % kDefaultTrainingSamples != 0);
  ManagedVector<TrainingSample> samples;
  samples.resize(std::min(total, kDefaultTrainingSamples));
  ManagedVector<TrainingSample> stratified;
  if (entropy)
    stratified.resize(std::min(total, learning.maximum_samples));
  // Give every nonempty stream/channel slice one sample when the budget
  // permits, then distribute the remaining samples by area. Within each
  // stratum, a deterministic hash avoids regular-stride aliasing on patterns.
  size_t slices = 0;
  for (const auto &slice : plan.slices)
    slices += slice.rect.extent.width != 0 && slice.rect.extent.height != 0;
  size_t slice_floor = slices <= stratified.size() ? 1 : 0;
  auto floor_samples = [&](size_t floor) {
    size_t sum = 0;
    for (const auto &slice : plan.slices)
      sum += std::min(floor, slice.rect.extent.width * slice.rect.extent.height);
    return sum;
  };
  if (entropy && learning.mode == TreeLearningMode::kQuantized) {
    // Maximize a common floor up to the reference's 1024 samples per slice,
    // then allocate the remainder by area. Small slices can be fully covered.
    size_t low = 0, high = 1025;
    while (low + 1 < high) {
      const size_t middle = (low + high) / 2;
      if (floor_samples(middle) <= stratified.size())
        low = middle;
      else
        high = middle;
    }
    slice_floor = low;
  }
  const size_t guaranteed = floor_samples(slice_floor);
  TrainingCoverage stats;
  stats.population = total;
  stats.samples = stratified.size();
  stats.slices = slices;
  stats.slice_floor = slice_floor;
  stats.minimum_slice_samples = SIZE_MAX;
  const size_t proportional = stratified.size() - guaranteed;
  if (entropy && total > SIZE_MAX / learning.maximum_samples)
    return Status::InvalidArgument("Modular sampling extent overflow");
  size_t sampled = 0, guaranteed_used = 0;
  size_t ordinal = 0, used = 0;
  for (const auto &stream : plan.streams) {
    if (stream.slice_begin > plan.slices.size() ||
        stream.slice_count > plan.slices.size() - stream.slice_begin)
      return Status::InvalidArgument("Invalid Modular training slices");
    size_t property_channel = 0;
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
      const size_t weighted_begin = ordinal - guaranteed_used;
      const size_t weighted_area = area - std::min(area, slice_floor);
      size_t quota = 0;
      if (entropy && area) {
        quota = std::min(area, slice_floor);
        if (weighted_total)
          quota += (weighted_begin + weighted_area) * proportional / weighted_total -
                   weighted_begin * proportional / weighted_total;
      }
      if (entropy && area) {
        stats.sampled_slices += quota != 0;
        stats.minimum_slice_samples = std::min(stats.minimum_slice_samples, quota);
        stats.maximum_slice_samples = std::max(stats.maximum_slice_samples, quota);
        stats.quota_hash = (stats.quota_hash ^ stream.id) * 1099511628211ULL;
        stats.quota_hash = (stats.quota_hash ^ slice.channel) * 1099511628211ULL;
        stats.quota_hash = (stats.quota_hash ^ area) * 1099511628211ULL;
        stats.quota_hash = (stats.quota_hash ^ quota) * 1099511628211ULL;
      }
      auto position = [&](size_t index) -> size_t {
        if (index == quota)
          return area;
        const size_t begin = index * area / quota, end = (index + 1) * area / quota;
        return begin + Mix(uint64_t{stream.id} * 0x9e3779b97f4a7c15ULL +
                           uint64_t{slice.channel} * 0xd1b54a32d192ed03ULL + index) %
                           (end - begin);
      };
      size_t local_sample = 0, next_sample = position(0), local = 0;
      WeightedPredictor<ResourceClass::kPreparation> wp(w, base.weighted);
      for (size_t y = 0; y < h; ++y) {
        int64_t previous_gradient = 0;
        for (size_t x = 0; x < w; ++x, ++ordinal, ++local) {
          const auto n = Neighbors(view, x, y);
          const auto weighted = wp.Predict(x, y, n.top, n.left, n.top_right, n.top_left, n.top_top);
          const auto properties =
              Properties(n, property_channel, stream.id, x, y, previous_gradient, weighted.second);
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
      guaranteed_used += std::min(area, slice_floor);
      ++property_channel;
    }
  }
  if (ordinal != total || !used)
    return Status::InvalidArgument("Incomplete Modular training coverage");
  // The prescribed tree is irrelevant to learning; retain only transform and
  // prediction settings, avoiding copies of an unrelated large tree.
  ModularCodingPolicy settings;
  settings.rct = base.rct;
  settings.weighted = base.weighted;
  settings.transforms = base.transforms;
  ModularCodingPolicy leaf = settings, branch = settings;
  if (auto status = LearnDecisionTrees(std::span(samples).first(used), &leaf.tree, &branch.tree);
      !status.ok())
    return status;
  ModularCodingPolicy learned = settings;
  if (entropy) {
    if (sampled != stratified.size())
      return Status::InvalidArgument("Incomplete Modular entropy sampling coverage");
    if (auto s = LearnEntropyTree(stratified, &learned.tree, learning, total); !s.ok())
      return s;
    *entropy = std::move(learned);
  }
  *single = std::move(leaf);
  *split = std::move(branch);
  if (coverage) {
    if (stats.minimum_slice_samples == SIZE_MAX)
      stats.minimum_slice_samples = 0;
    *coverage = stats;
  }
  return Status::Ok();
} catch (const resource_budget_internal::ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Modular training allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Modular training storage overflow");
}
} // namespace gjxl::modular_internal
