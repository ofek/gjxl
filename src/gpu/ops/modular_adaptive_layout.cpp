// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/ops/modular_adaptive_layout.h"

namespace gjxl::modular_gpu_internal {
using namespace modular_internal;
using namespace resource_budget_internal;
Status ComputeAdaptiveModularLayout(const ModularEncoderFrame &frame, AdaptiveModularLayout *out) {
  if (!out)
    return Status::InvalidArgument("Null adaptive Modular layout");
  AdaptiveModularLayout p;
  if (auto s = codec_internal::ValidateImageMetadata(frame.metadata()); !s.ok())
    return s;
  if (auto s = DescribeImage(frame.image(), &p.shape); !s.ok())
    return s;
  if (auto s = ModularFrameGeometry::Create(frame.metadata().extent, &p.geometry); !s.ok())
    return s;
  if (18 + 3 * p.geometry.dc_group_count() + p.geometry.group_count() - 1 > INT32_MAX)
    return Status::Unsupported("Adaptive Modular stream IDs exceed the property range");
  bool global = true;
  for (size_t c = 0; c < p.shape.count; ++c) {
    const auto view = frame.image().view(c);
    if (auto s = view.Validate(); !s.ok())
      return s;
    const auto &ch = view.descriptor;
    size_t area;
    if (ch.extent.width > UINT32_MAX || ch.extent.height > UINT32_MAX ||
        !ch.extent.try_area(&area) || area > size_t(PTRDIFF_MAX) / sizeof(int32_t) ||
        area > SIZE_MAX - p.tokens || p.plane_bytes > SIZE_MAX - 255)
      return Status::InvalidArgument("Adaptive Modular plane extent overflow");
    p.plane_bytes = (p.plane_bytes + 255) & ~size_t{255};
    p.plane_offsets[c] = p.plane_bytes;
    if (p.plane_bytes > PTRDIFF_MAX || area * sizeof(int32_t) > size_t(PTRDIFF_MAX) - p.plane_bytes)
      return Status::InvalidArgument("Adaptive Modular plane arena overflow");
    p.plane_bytes += area * sizeof(int32_t);
    p.tokens += area;
    if (c < p.shape.metadata)
      continue;
    const auto width = p.geometry.source().ceil_div(size_t{1} << ch.hshift).width;
    const auto height = p.geometry.source().ceil_div(size_t{1} << ch.vshift).height;
    if (ch.extent.width > width || ch.extent.height > height)
      return Status::InvalidArgument("Adaptive Modular channel exceeds its source extent");
    global &= ch.extent.width <= kGroupDimension && ch.extent.height <= kGroupDimension;
    const unsigned bits = std::min(ch.hshift, ch.vshift) >= 3 ? 11 : 8;
    if (!global && (ch.hshift > bits || ch.vshift > bits))
      return Status::Unsupported("Adaptive Modular channel shift exceeds group dimensions");
  }
  size_t covered = 0;
  if (auto s = VisitAdaptiveModularSlices(
          p,
          [&](const auto &slice, size_t, ModularRect) {
            const uint64_t area = uint64_t{slice.width} * slice.height;
            if (area > UINT32_MAX || area > p.tokens - covered || p.slices == UINT32_MAX)
              return Status::Unsupported("Adaptive Modular dispatch exceeds index limits");
            ++p.slices;
            covered += static_cast<size_t>(area);
            p.maximum_slice_tokens = std::max(p.maximum_slice_tokens, static_cast<size_t>(area));
            return Status::Ok();
          });
      !s.ok())
    return s;
  if (covered != p.tokens || !p.slices)
    return Status::InvalidArgument("Incomplete adaptive Modular slice coverage");
  *out = p;
  return Status::Ok();
}
Status ValidateAdaptiveModularSlices(const AdaptiveModularLayout &layout,
                                     std::span<const AdaptiveModularTokenSlice> slices) {
  if (slices.size() != layout.slices)
    return Status::InvalidArgument("Adaptive Modular slice count mismatch");
  size_t index = 0;
  return VisitAdaptiveModularSlices(layout, [&](const auto &expected, size_t, ModularRect) {
    if (index >= slices.size() || slices[index++] != expected)
      return Status::InvalidArgument("Noncanonical adaptive Modular slice");
    return Status::Ok();
  });
}
Status BuildAdaptiveModularSlices(const ModularEncoderFrame &frame,
                                  ManagedVector<AdaptiveModularTokenSlice> *out) try {
  if (!out)
    return Status::InvalidArgument("Null adaptive Modular slices");
  AdaptiveModularLayout layout;
  if (auto s = ComputeAdaptiveModularLayout(frame, &layout); !s.ok())
    return s;
  ManagedVector<AdaptiveModularTokenSlice> slices;
  slices.reserve(layout.slices);
  if (auto s = VisitAdaptiveModularSlices(layout,
                                          [&](const auto &slice, size_t, ModularRect) {
                                            slices.push_back(slice);
                                            return Status::Ok();
                                          });
      !s.ok())
    return s;
  *out = std::move(slices);
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Adaptive Modular slice allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Adaptive Modular slice storage overflow");
}
} // namespace gjxl::modular_gpu_internal
