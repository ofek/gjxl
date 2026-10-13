// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/image.h"
#include <array>

namespace gjxl::modular_internal {
// Deliberately bounded CPU subset. Entries execute in order, in the channel
// coordinates produced by the preceding entries. The legacy policy.rct still
// precedes this sequence. Palette deltas are not supported.
inline constexpr size_t kMaximumTransforms = 8;
inline constexpr size_t kMaximumTransformedChannels = 64;
inline constexpr size_t kMaximumPaletteColors = 1024;
inline constexpr size_t kImplicitPaletteColors = 189;
inline constexpr size_t kMaximumReferencePaletteColors = kMaximumPaletteColors + kImplicitPaletteColors;
inline constexpr uint32_t kMaximumReferenceChannelPaletteColors = 5376 + 65535;
enum class TransformKind : uint8_t { kPalette, kSqueeze, kRct };
struct TransformDescriptor {
  TransformKind kind = TransformKind::kPalette;
  uint8_t begin = 0, count = 1;
  uint32_t colors = 1;
  bool horizontal = true, in_place = true;
  uint8_t rct_type = 0; // RCT requires count=3; type 0 is an explicit identity.
  // Zero keeps the legacy explicit dictionary/capacity construction. Nonzero
  // selects reference lossless palette construction and implicit index values;
  // colors is then the actual explicit width, possibly zero. The frame depth
  // is not serialized here and must agree with the enclosing frame's depth.
  uint8_t palette_bit_depth = 0;
  bool palette_ordered = true;
  bool operator==(const TransformDescriptor &) const = default;
};
struct TransformSequence {
  std::array<TransformDescriptor, kMaximumTransforms> entries{};
  size_t size = 0;
  bool operator==(const TransformSequence &) const = default;
};
struct ChannelShape {
  std::array<ChannelDescriptor, kMaximumTransformedChannels> descriptors{};
  size_t count = 0, metadata = 0;
  std::span<const ChannelDescriptor> channels() const {
    return std::span(descriptors).first(count);
  }
};
[[nodiscard]] Status ValidateTransforms(const TransformSequence &sequence);
[[nodiscard]] Status ValidateTransformBitDepth(const TransformSequence &sequence, uint32_t bit_depth);
[[nodiscard]] Status DescribeImage(const ModularImage &image, ChannelShape *out);
// Allocation-free. Each shape corresponds to the image before/after one step.
[[nodiscard]] Status PlanTransforms(std::span<const ChannelDescriptor> source, size_t metadata,
                                    const TransformSequence &sequence,
                                    std::array<ChannelShape, kMaximumTransforms + 1> *out);
// Borrow input, publish only on success. Inverse requires original descriptors
// to restore roles as well as dimensions. Neither operation mutates input.
[[nodiscard]] Status ForwardTransforms(const ModularImage &input, const TransformSequence &sequence,
                                       ModularImage *out);
[[nodiscard]] Status InverseTransforms(const ModularImage &input,
                                       std::span<const ChannelDescriptor> original, size_t metadata,
                                       const TransformSequence &sequence, ModularImage *out);
} // namespace gjxl::modular_internal
