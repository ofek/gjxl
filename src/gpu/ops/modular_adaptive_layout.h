// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/geometry.h"
#include "gpu/ops/modular_adaptive_tokens.h"

namespace gjxl::modular_gpu_internal {
struct AdaptiveModularLayout {
  modular_internal::ChannelShape shape;
  modular_internal::ModularFrameGeometry geometry;
  std::array<size_t, modular_internal::kMaximumTransformedChannels> plane_offsets{};
  size_t plane_bytes = 0, tokens = 0, slices = 0, maximum_slice_tokens = 0;
};
// Shape/metadata/backing validation and canonical descriptor enumeration are
// allocation-free. Offsets are into the packed resident plane arena, in bytes.
[[nodiscard]] Status
ComputeAdaptiveModularLayout(const modular_internal::ModularEncoderFrame &frame,
                             AdaptiveModularLayout *out);
// Call only with a freshly validated layout for frame. The callback receives
// physical channel/rectangle separately from the device's stream-local ordinal.
template <class Visitor>
Status VisitAdaptiveModularSlices(const AdaptiveModularLayout &layout, Visitor &&visit) {
  using namespace modular_internal;
  const auto channels = layout.shape.channels();
  size_t global = layout.shape.metadata;
  while (global < channels.size() && channels[global].extent.width <= kGroupDimension &&
         channels[global].extent.height <= kGroupDimension)
    ++global;
  size_t next = 0;
  auto emit = [&](size_t c, size_t stream, size_t ordinal, ModularRect rect) -> Status {
    const auto &e = rect.extent;
    const auto stride = channels[c].extent.width;
    const AdaptiveModularTokenSlice slice{next,
                                          layout.plane_offsets[c] / sizeof(int32_t) +
                                              rect.y * stride + rect.x,
                                          stride,
                                          static_cast<uint32_t>(stream),
                                          static_cast<uint32_t>(ordinal),
                                          static_cast<uint32_t>(e.width),
                                          static_cast<uint32_t>(e.height),
                                          {0, 0}};
    next += e.width * e.height;
    return visit(slice, c, rect);
  };
  for (size_t c = 0; c < global; ++c)
    if (auto s = emit(c, 0, c, {0, 0, channels[c].extent}); !s.ok())
      return s;
  for (bool dc : {true, false}) {
    const size_t groups = dc ? layout.geometry.dc_group_count() : layout.geometry.group_count();
    for (size_t group = 0; group < groups; ++group) {
      ModularRect rect;
      if (auto s = layout.geometry.GroupRect(group, dc, &rect); !s.ok())
        return s;
      const size_t stream = dc ? 1 + layout.geometry.dc_group_count() + group
                               : 18 + 3 * layout.geometry.dc_group_count() + group;
      size_t ordinal = 0;
      for (size_t c = global; c < channels.size(); ++c) {
        const auto &ch = channels[c];
        if ((std::min(ch.hshift, ch.vshift) >= 3) != dc)
          continue;
        const size_t x = rect.x >> ch.hshift, y = rect.y >> ch.vshift;
        if (x >= ch.extent.width || y >= ch.extent.height)
          continue;
        const Extent2D e{std::min(rect.extent.width >> ch.hshift, ch.extent.width - x),
                         std::min(rect.extent.height >> ch.vshift, ch.extent.height - y)};
        if (!e.empty())
          if (auto s = emit(c, stream, ordinal++, {x, y, e}); !s.ok())
            return s;
      }
    }
  }
  return Status::Ok();
}
// Fresh owned descriptors; callers admit this backing separately from the
// completed feature owner's slice copy. Failure preserves the old vector.
[[nodiscard]] Status
BuildAdaptiveModularSlices(const modular_internal::ModularEncoderFrame &frame,
                           resource_budget_internal::ManagedVector<AdaptiveModularTokenSlice> *out);
[[nodiscard]] Status
ValidateAdaptiveModularSlices(const AdaptiveModularLayout &layout,
                              std::span<const AdaptiveModularTokenSlice> slices);
} // namespace gjxl::modular_gpu_internal
