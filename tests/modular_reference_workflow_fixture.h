// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/modular/reference_workflow.h"
#include "modular_reference.h"

namespace gjxl::test::modular_workflow_fixture {
using namespace gjxl::modular_internal;
namespace reference = gjxl::test::modular_reference;
struct Fixture {
  std::vector<uint8_t> bytes;
  reference::IntegerImage original;
  size_t stride;
  PackedModularFormat format;
  SampleByteOrder order;
  PackedModularImageView view() const {
    return {bytes, {original.width, original.height}, stride, format, order};
  }
};
inline Fixture Make(Extent2D extent, size_t channels, unsigned bits, unsigned pattern,
                    SampleByteOrder order = SampleByteOrder::kBigEndian) {
  Fixture f;
  f.original = {extent.width, extent.height, uint32_t(channels), bits, {}};
  f.stride = extent.width * channels * (bits / 8) + 7;
  f.bytes.resize(f.stride * extent.height, 0xAD);
  f.order = order;
  f.format =
      channels == 1   ? (bits == 8 ? PackedModularFormat::kGray8 : PackedModularFormat::kGray16)
      : channels == 3 ? (bits == 8 ? PackedModularFormat::kRgb8 : PackedModularFormat::kRgb16)
                      : (bits == 8 ? PackedModularFormat::kRgba8 : PackedModularFormat::kRgba16);
  uint32_t rng = 3571;
  for (size_t y = 0; y < extent.height; ++y)
    for (size_t x = 0; x < extent.width; ++x)
      for (size_t c = 0; c < channels; ++c) {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        uint32_t value = pattern == 0   ? 17 + c * 31
                         : pattern == 1 ? ((x / 3 + y / 5 + c) % 5) * (bits == 8 ? 47 : 12017)
                         : pattern == 2 ? x * 521 + y * 997 + c * 2029
                                        : rng;
        value &= (1u << bits) - 1;
        // Include invisible color values and both transparent/opaque alpha.
        if (c == 3 && (x + y) % 3 == 0)
          value = 0;
        f.original.samples.push_back(uint16_t(value));
        const size_t pos = y * f.stride + (x * channels + c) * (bits / 8);
        if (bits == 8)
          f.bytes[pos] = value;
        else {
          f.bytes[pos] = order == SampleByteOrder::kBigEndian ? value >> 8 : value;
          f.bytes[pos + 1] = order == SampleByteOrder::kBigEndian ? value : value >> 8;
        }
      }
  return f;
}
} // namespace gjxl::test::modular_workflow_fixture
