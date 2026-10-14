// Copyright (c) the JPEG XL Project Authors.
// Copyright (c) 2026 Yunho Cho
// SPDX-License-Identifier: BSD-3-Clause
// Lossless selection policy adapted from pinned libjxl enc_modular.cc.
#include "codec/modular/reference_selection.h"
#include "codec/modular/transform/rct.h"
#include <algorithm>
#include <cmath>
#include <functional>

namespace gjxl::modular_internal {
namespace {
using namespace resource_budget_internal;
using Views = std::array<ModularChannelView, kMaximumTransformedChannels>;
Status Invalid() {
  return Status::InvalidArgument("Invalid reference Modular selection");
}
bool EqualGeometry(ModularChannelView a, ModularChannelView b) {
  return a.descriptor.extent == b.descriptor.extent &&
         a.descriptor.hshift == b.descriptor.hshift &&
         a.descriptor.vshift == b.descriptor.vshift;
}
float ChannelPercent(const ReferenceSelectionOptions &o) {
  return o.palettes == ReferencePaletteStage::kGlobal ? 95.f : 80.f;
}
uint32_t ChannelLimit(float pixels, int64_t range,
                      const ReferenceSelectionOptions &o) {
  // Preserve the reference's float pixel count and double percent expression.
  return std::min(static_cast<int>(pixels / 16),
                  static_cast<int>(ChannelPercent(o) / 100. * range));
}
Status Validate(std::span<const ModularChannelView> input, size_t metadata,
                const ReferenceSelectionOptions &o) {
  if (input.empty() || input.size() > kMaximumTransformedChannels ||
      metadata > input.size() || input.size() - metadata > 4 ||
      o.bit_depth == 0 || o.bit_depth > 16 || o.palette_colors < -1024 ||
      o.palette_colors > 1024 ||
      (o.palettes != ReferencePaletteStage::kNone &&
       o.palettes != ReferencePaletteStage::kGlobal &&
       o.palettes != ReferencePaletteStage::kLocal))
    return Invalid();
  const size_t normal = input.size() - metadata;
  if (o.palettes != ReferencePaletteStage::kNone &&
      input.size() + normal > kMaximumTransformedChannels)
    return Invalid();
  size_t pixels = 0;
  for (size_t c = 0; c < input.size(); ++c) {
    if (auto s = input[c].Validate(); !s.ok())
      return s;
    size_t area;
    if ((input[c].descriptor.role == ChannelRole::kMetadata) !=
            (c < metadata) ||
        !input[c].descriptor.extent.try_area(&area) ||
        area > INT32_MAX - pixels)
      return Invalid();
    pixels += area;
  }
  if (o.select_rct && normal >= 3 &&
      (!EqualGeometry(input[metadata], input[metadata + 1]) ||
       !EqualGeometry(input[metadata], input[metadata + 2])))
    return Invalid();
  return Status::Ok();
}
std::pair<int32_t, int32_t> MinMax(ModularChannelView view) {
  int32_t lo = INT32_MAX, hi = INT32_MIN;
  const auto [w, h] = view.descriptor.extent;
  for (size_t y = 0; y < h; ++y)
    for (size_t x = 0; x < w; ++x) {
      lo = std::min(lo, view.Row(y)[x]);
      hi = std::max(hi, view.Row(y)[x]);
    }
  return {lo, hi};
}
std::span<const ModularChannelView> ImageViews(const ModularImage &image,
                                               Views *views) {
  for (size_t c = 0; c < image.channel_count(); ++c)
    (*views)[c] = image.view(c);
  return std::span(*views).first(image.channel_count());
}
Status Cost(std::span<const ModularChannelView> views,
            ReferenceSelectionTrace *trace, float *cost) {
  ModularTransformCost estimate;
  if (auto s = EstimateModularTransformCost(views, &estimate); !s.ok())
    return s;
  assert(trace->gradient_count < trace->gradient_costs.size());
  trace->gradient_costs[trace->gradient_count++] = *cost = estimate.bits;
  return Status::Ok();
}
Status Copy(std::span<const ModularChannelView> input, size_t metadata,
            ModularImage *out) {
  std::array<ChannelDescriptor, kMaximumTransformedChannels> descriptors;
  for (size_t c = 0; c < input.size(); ++c)
    descriptors[c] = input[c].descriptor;
  if (auto s = ModularImage::Create(std::span(descriptors).first(input.size()),
                                    metadata, out);
      !s.ok())
    return s;
  for (size_t c = 0; c < input.size(); ++c) {
    const auto [w, h] = input[c].descriptor.extent;
    if (w != 0)
      for (size_t y = 0; y < h; ++y)
        std::copy_n(input[c].Row(y), w, out->samples(c).data() + y * w);
  }
  return Status::Ok();
}
void Rct(std::span<const ModularChannelView> input, uint8_t type,
         ModularImage *image, size_t begin) {
  const auto [w, h] = input[0].descriptor.extent;
  for (size_t y = 0; y < h; ++y)
    for (size_t x = 0; x < w; ++x) {
      const auto value = ForwardRct(
          {input[0].Row(y)[x], input[1].Row(y)[x], input[2].Row(y)[x]}, type);
      // Read all three values before writing, including for in-place winners.
      for (size_t c = 0; c < 3; ++c)
        image->samples(begin + c)[y * w + x] = value[c];
    }
}
} // namespace

Status ComputeReferenceSelectionStorageBound(
    std::span<const ModularChannelView> input, size_t metadata,
    const ReferenceSelectionOptions &o, HostStorageBound *out) {
  if (!out)
    return Invalid();
  if (auto s = Validate(input, metadata, o); !s.ok())
    return s;
  const size_t normal = input.size() - metadata;
  const bool palettes =
      o.palettes != ReferencePaletteStage::kNone && normal != 0;
  const bool rct = o.select_rct && normal >= 3;
  HostStorageBound image, rct_scratch;
  size_t max_width = 0;
  for (auto v : input)
    max_width = std::max(max_width, v.descriptor.extent.width);
  if (palettes || rct) {
    if (!image.AddVector<ModularImage::Channel>(
            input.size() + (palettes ? normal : 0),
            VectorCapacityPolicy::kFreshExact))
      return Invalid();
    for (auto v : input) {
      size_t area = v.descriptor.extent.width * v.descriptor.extent.height;
      if (!image.AddVector<int32_t>(area, VectorCapacityPolicy::kFreshExact))
        return Invalid();
    }
  }
  if (palettes) {
    const float pixels =
        input[0].descriptor.extent.width * input[0].descriptor.extent.height;
    const uint32_t limit = ChannelLimit(pixels, 65536, o);
    for (size_t c = metadata; c < input.size(); ++c) {
      const auto extent = input[c].descriptor.extent;
      const size_t colors =
          std::min(size_t{limit}, extent.width * extent.height);
      if (!image.AddVector<int32_t>(colors, VectorCapacityPolicy::kFreshExact))
        return Invalid();
      max_width = std::max(max_width, colors);
    }
    if (normal > 1 && o.palette_colors != 0) {
      const size_t colors = std::abs(o.palette_colors) + kImplicitPaletteColors;
      if (!image.AddVector<int32_t>(normal * colors,
                                    VectorCapacityPolicy::kFreshExact))
        return Invalid();
      max_width = std::max(max_width, colors);
    }
  }
  if (rct) {
    const auto extent = input[metadata].descriptor.extent;
    if (!rct_scratch.AddVector<ModularImage::Channel>(
            3, VectorCapacityPolicy::kFreshExact) ||
        !rct_scratch.AddVector<int32_t>(extent.width * extent.height,
                                        VectorCapacityPolicy::kFreshExact, 3))
      return Invalid();
  }
  HostStorageBound bound = image;
  if (palettes) {
    // Admitted normal samples and resulting palette indices are within a
    // 16-bit unsigned span, so the wide constructor needs only its bitmap.
    if (!bound.Add({0, image.peak_bytes}))
      return Invalid();
  }
  HostStorageBound after = image;
  rct_scratch.retained_bytes = 0;
  if (!after.Add(rct_scratch))
    return Invalid();
  bound.peak_bytes = std::max(bound.peak_bytes, after.peak_bytes);
  if (o.select_wp) {
    // Histories are released per channel and per mode; RCT scratch is gone.
    HostStorageBound history;
    if (!history.AddVector<int64_t>(max_width + 2,
                                    VectorCapacityPolicy::kFreshExact, 5))
      return Invalid();
    history.retained_bytes = 0;
    after = image;
    if (!after.Add(history))
      return Invalid();
    bound.peak_bytes = std::max(bound.peak_bytes, after.peak_bytes);
  }
  *out = bound;
  return Status::Ok();
}

Status SelectReferenceModularEncoding(std::span<const ModularChannelView> input,
                                      size_t metadata,
                                      const ReferenceSelectionOptions &o,
                                      ReferenceSelectionResult *out) {
  if (!out)
    return Invalid();
  HostStorageBound bound;
  if (auto s =
          ComputeReferenceSelectionStorageBound(input, metadata, o, &bound);
      !s.ok())
    return s;
  bool exact_alias = out->image.channel_count() == input.size() &&
                     out->image.metadata_channels() == metadata;
  if (exact_alias)
    for (size_t c = 0; c < input.size(); ++c) {
      const auto previous = out->image.view(c);
      exact_alias &= input[c].descriptor == previous.descriptor &&
                     input[c].stride == previous.stride &&
                     input[c].backing.data() == previous.backing.data() &&
                     input[c].backing.size() == previous.backing.size();
    }
  if (!exact_alias) {
    const std::less<const int32_t *> before;
    for (auto view : input)
      if (!view.backing.empty())
        for (size_t c = 0; c < out->image.channel_count(); ++c) {
          const auto previous = out->image.view(c).backing;
          if (!previous.empty() &&
              before(view.backing.data(), previous.data() + previous.size()) &&
              before(previous.data(),
                     view.backing.data() + view.backing.size()))
            return Invalid();
        }
  }
  // The supported raw/palette domain avoids reference RCT's int32 wrapping and
  // bounds single-channel proposals without silently capping their heuristic.
  for (size_t c = metadata; c < input.size(); ++c) {
    const auto [lo, hi] = MinMax(input[c]);
    if (lo < 0 || hi > 65535)
      return Invalid();
  }
  auto context = CurrentResourceContext();
  context.resource_class = ResourceClass::kPreparation;
  ResourceContextScope scope(context);
  ReferenceSelectionResult result;
  Views storage;
  auto current = input;
  auto propose = [&](uint8_t begin, uint8_t count, uint32_t limit, bool ordered,
                     float before, bool *accepted) -> Status {
    assert(result.trace.palette_count < result.trace.palettes.size());
    auto &decision = result.trace.palettes[result.trace.palette_count++];
    decision = {begin, count, limit, false, false, before, 0};
    *accepted = false;
    for (size_t c = 1; c < count; ++c)
      if (!EqualGeometry(current[begin], current[begin + c]))
        return Status::Ok(); // Reference CheckEqualChannels rejects this
                             // proposal.
    ReferencePaletteCandidate candidate;
    if (auto s = TryBuildReferencePalette(
            current, metadata, {begin, count, o.bit_depth, limit, ordered},
            &candidate);
        !s.ok())
      return s;
    if (!candidate.applied)
      return Status::Ok();
    decision.built = true;
    Views trial;
    if (auto s = Cost(ImageViews(candidate.image, &trial), &result.trace,
                      &decision.after);
        !s.ok())
      return s;
    if (decision.after > before)
      return Status::Ok();
    decision.accepted = *accepted = true;
    result.transforms.entries[result.transforms.size++] = candidate.transform;
    result.image = std::move(candidate.image);
    metadata = result.image.metadata_channels();
    current = ImageViews(result.image, &storage);
    return Status::Ok();
  };
  if (o.palettes != ReferencePaletteStage::kNone) {
    float before = 0;
    const float pixels = current[0].descriptor.extent.width *
                         current[0].descriptor.extent.height;
    const size_t normal = current.size() - metadata;
    bool did_palette = false;
    if (o.palette_colors != 0) {
      if (auto s = Cost(current, &result.trace, &before); !s.ok())
        return s;
      const uint32_t limit =
          std::min(static_cast<int>(before * .0005f + pixels / 128 + 128),
                   std::abs(o.palette_colors));
      if (normal > 1)
        if (auto s = propose(metadata, normal, limit, o.palette_colors >= 0,
                             before, &did_palette);
            !s.ok())
          return s;
      if (!did_palette && normal > 3)
        if (auto s = propose(metadata, normal - 1, limit, o.palette_colors >= 0,
                             before, &did_palette);
            !s.ok())
          return s;
    }
    const size_t remaining = current.size() - metadata - did_palette;
    if (remaining > 0 && (did_palette || before == 0))
      if (auto s = Cost(current, &result.trace, &before); !s.ok())
        return s;
    for (size_t i = did_palette; i < remaining + did_palette; ++i) {
      const auto [lo, hi] = MinMax(current[metadata + i]);
      const uint32_t limit = ChannelLimit(pixels, int64_t{hi} - lo + 1, o);
      bool accepted;
      // Reference deliberately retains this baseline for all single channels.
      if (auto s = propose(metadata + i, 1, limit, true, before, &accepted);
          !s.ok())
        return s;
    }
  }
  if (o.select_rct && current.size() - metadata >= 3) {
    ModularImage scratch;
    const std::array descriptors{current[metadata].descriptor,
                                 current[metadata + 1].descriptor,
                                 current[metadata + 2].descriptor};
    if (auto s = ModularImage::Create(descriptors, 0, &scratch); !s.ok())
      return s;
    Views trial;
    std::copy(current.begin(), current.end(), trial.begin());
    const auto candidates = std::span(trial).first(current.size());
    float best;
    if (auto s = Cost(current, &result.trace, &best); !s.ok())
      return s;
    ++result.trace.rct_count;
    for (size_t c = 0; c < 3; ++c)
      trial[metadata + c] = scratch.view(c);
    for (size_t i = 1; i < kSelectionRcts.size(); ++i) {
      Rct(current.subspan(metadata, 3), kSelectionRcts[i], &scratch, 0);
      float cost;
      if (auto s = Cost(candidates, &result.trace, &cost); !s.ok())
        return s;
      ++result.trace.rct_count;
      if (cost < best) {
        best = cost;
        result.trace.rct = kSelectionRcts[i];
      }
    }
    if (result.trace.rct != 0) {
      if (result.image.channel_count() == 0) {
        if (auto s = Copy(current, metadata, &result.image); !s.ok())
          return s;
        current = ImageViews(result.image, &storage);
      }
      Rct(current.subspan(metadata, 3), result.trace.rct, &result.image,
          metadata);
      TransformDescriptor transform;
      transform.kind = TransformKind::kRct;
      transform.begin = metadata;
      transform.count = 3;
      transform.rct_type = result.trace.rct;
      result.transforms.entries[result.transforms.size++] = transform;
    }
  } // Release the three RCT trial planes before WP histories.
  if (o.select_wp) {
    float best = std::numeric_limits<float>::max();
    for (size_t mode = 0; mode < kSelectionWeightedPresets.size(); ++mode) {
      ModularWeightedCost cost;
      if (auto s = EstimateModularWeightedCost(current, mode, &cost); !s.ok())
        return s;
      result.trace.wp_costs[result.trace.wp_count++] = cost.bits;
      if (cost.bits < best) {
        best = cost.bits;
        result.wp_mode = mode;
      }
    }
  }
  if (exact_alias && result.image.channel_count() == 0)
    result.image = std::move(out->image);
  *out = std::move(result);
  return Status::Ok();
}
} // namespace gjxl::modular_internal
