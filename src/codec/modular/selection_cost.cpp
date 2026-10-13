// Copyright (c) the JPEG XL Project Authors.
// Copyright (c) 2026 Yunho Cho
// SPDX-License-Identifier: BSD-3-Clause
// Adapted from pinned libjxl e8ff0976 enc_modular_simd.cc, enc_cluster.cc and
// fast_math-inl.h. Native borrowed views and bounded storage; no Highway or
// libjxl dependency. Integer vector loads never rely on input padding.
#include "codec/modular/selection_cost.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>

#if (defined(__aarch64__) || defined(_M_ARM64)) && !defined(GJXL_SELECTION_COST_SCALAR)
#include <arm_neon.h>
#define GJXL_SELECTION_NEON 1
#endif

namespace gjxl::modular_internal {
namespace {
constexpr std::array<uint32_t, 16> kCuts = {1,  3,  5,  7,   11,  15,  23,  31,
                                            47, 63, 95, 127, 191, 255, 392, 500};
constexpr auto ContextMap() {
  std::array<uint8_t, 501> map{};
  for (size_t value = 0; value < map.size(); ++value)
    for (auto cut : kCuts)
      map[value] += value >= cut;
  return map;
}
constexpr auto kContextMap = ContextMap();
struct Histogram {
  std::array<int32_t, 128> counts{};
  uint32_t total = 0;
};
static_assert(sizeof(std::array<Histogram, 17>) < 9 * 1024);

#if GJXL_SELECTION_NEON
float32x4_t FastLog2(float32x4_t x) {
  const auto bits = vreinterpretq_s32_f32(x);
  const auto exponent = vshrq_n_s32(vsubq_s32(bits, vdupq_n_s32(0x3f2aaaab)), 23);
  const auto mantissa = vreinterpretq_f32_s32(vsubq_s32(bits, vshlq_n_s32(exponent, 23)));
  const auto z = vsubq_f32(mantissa, vdupq_n_f32(1.0f));
  auto p = vfmaq_f32(vdupq_n_f32(1.4287160470083755E+00f), vdupq_n_f32(7.4245873327820566E-01f), z);
  auto q = vfmaq_f32(vdupq_n_f32(1.0096718572241148E+00f), vdupq_n_f32(1.7409343003366853E-01f), z);
  p = vfmaq_f32(vdupq_n_f32(-1.8503833400518310E-06f), p, z);
  q = vfmaq_f32(vdupq_n_f32(9.9032814277590719E-01f), q, z);
  return vaddq_f32(vdivq_f32(p, q), vcvtq_f32_s32(exponent));
}
#else
float FastLog2(float x) {
  const uint32_t bits = std::bit_cast<uint32_t>(x);
  const int32_t exponent = std::bit_cast<int32_t>(bits - 0x3f2aaaabu) >> 23;
  const float mantissa = std::bit_cast<float>(bits - (static_cast<uint32_t>(exponent) << 23));
  const float z = mantissa - 1.0f;
  float p = std::fma(7.4245873327820566E-01f, z, 1.4287160470083755E+00f);
  float q = std::fma(1.7409343003366853E-01f, z, 1.0096718572241148E+00f);
  p = std::fma(p, z, -1.8503833400518310E-06f);
  q = std::fma(q, z, 9.9032814277590719E-01f);
  return p / q + static_cast<float>(exponent);
}
#endif
float Shannon(const Histogram &h) {
#pragma STDC FP_CONTRACT OFF
  if (!h.total)
    return 0;
  const float total = static_cast<float>(h.total), inv = 1.0f / total;
#if GJXL_SELECTION_NEON
  auto bits = vdupq_n_f32(0);
  for (size_t i = 0; i < h.counts.size(); i += 4) {
    const auto c = vcvtq_f32_s32(vld1q_s32(h.counts.data() + i));
    const auto cost = vsubq_f32(vdupq_n_f32(0), vmulq_f32(c, FastLog2(vmulq_n_f32(c, inv))));
    bits = vaddq_f32(bits, vbslq_f32(vceqq_f32(c, vdupq_n_f32(total)), vdupq_n_f32(0), cost));
  }
  return vaddvq_f32(bits);
#else
  std::array<float, 4> bits{};
  for (size_t i = 0; i < h.counts.size(); ++i) {
    const float c = static_cast<float>(h.counts[i]);
    bits[i % 4] += c == total ? 0.0f : 0.0f - c * FastLog2(inv * c);
  }
  return (bits[0] + bits[1]) + (bits[2] + bits[3]);
#endif
}
uint32_t Pack(uint32_t residual) { return (residual << 1) ^ (0u - (residual >> 31)); }
void Add(uint32_t packed, Histogram *h, uint32_t *extra) {
  uint32_t token = packed;
  if (packed >= 16) {
    const uint32_t nbits = std::bit_width(packed) - 3;
    token = 8 + 4 * nbits + ((packed >> nbits) & 3);
    *extra += nbits;
  }
  ++h->counts[token];
  ++h->total;
}
void Collect(const ModularChannelView &ch, std::array<Histogram, 17> *histograms, uint32_t *extra) {
  const auto [w, h] = ch.descriptor.extent;
  for (size_t y = 0; y < h; ++y) {
    const int32_t *row = ch.Row(y), *top = y ? ch.Row(y - 1) : nullptr;
    size_t x = 0;
#if GJXL_SELECTION_NEON
    for (; x + 4 <= w; x += 4) {
      const auto center = vld1q_s32(row + x);
      const int32_t edge = y ? top[0] : 0;
      const auto left = x ? vld1q_s32(row + x - 1) : vextq_s32(vdupq_n_s32(edge), center, 3);
      auto prediction = left;
      auto diff = vdupq_n_u32(0);
      if (y) {
        const auto north = vld1q_s32(top + x);
        const auto northwest = x ? vld1q_s32(top + x - 1) : vextq_s32(vdupq_n_s32(edge), north, 3);
        const auto low = vminq_s32(left, north), high = vmaxq_s32(left, north);
        const auto gradient = vsubq_s32(vaddq_s32(north, left), northwest);
        prediction = vbslq_s32(vcgtq_s32(northwest, high), low,
                               vbslq_s32(vcltq_s32(northwest, low), high, gradient));
        diff =
            vreinterpretq_u32_s32(vsubq_s32(vmaxq_s32(high, northwest), vminq_s32(low, northwest)));
        diff = vminq_u32(diff, vdupq_n_u32(500));
      }
      const auto residual = vreinterpretq_u32_s32(vsubq_s32(center, prediction));
      const auto packed =
          veorq_u32(vshlq_n_u32(residual, 1), vsubq_u32(vdupq_n_u32(0), vshrq_n_u32(residual, 31)));
      // Integer extraction is exact even near powers of two at full bit depth.
      const auto nbits = vsubq_u32(vdupq_n_u32(29), vclzq_u32(vmaxq_u32(packed, vdupq_n_u32(16))));
      const auto symbol =
          vaddq_u32(vdupq_n_u32(8),
                    vaddq_u32(vshlq_n_u32(nbits, 2),
                              vandq_u32(vshlq_u32(packed, vnegq_s32(vreinterpretq_s32_u32(nbits))),
                                        vdupq_n_u32(3))));
      const auto literal = vcltq_u32(packed, vdupq_n_u32(16));
      *extra += vaddvq_u32(vbslq_u32(literal, vdupq_n_u32(0), nbits));
      std::array<uint32_t, 4> symbols, contexts;
      vst1q_u32(symbols.data(), vbslq_u32(literal, packed, symbol));
      vst1q_u32(contexts.data(), diff);
      for (size_t lane = 0; lane < 4; ++lane) {
        auto &histogram = (*histograms)[kContextMap[contexts[lane]]];
        ++histogram.counts[symbols[lane]];
        ++histogram.total;
      }
    }
#endif
    for (; x < w; ++x) {
      const int32_t left = x ? row[x - 1] : y ? top[0] : 0;
      int32_t prediction = left;
      size_t context = 0;
      if (y) {
        const int32_t north = top[x], northwest = x ? top[x - 1] : top[0];
        const int32_t low = std::min(left, north), high = std::max(left, north);
        prediction = static_cast<int32_t>(
            std::clamp(int64_t{north} + left - northwest, int64_t{low}, int64_t{high}));
        const uint32_t diff = static_cast<uint32_t>(std::max(high, northwest)) -
                              static_cast<uint32_t>(std::min(low, northwest));
        context = kContextMap[std::min(diff, 500u)];
      }
      Add(Pack(static_cast<uint32_t>(row[x]) - static_cast<uint32_t>(prediction)),
          &(*histograms)[context], extra);
    }
  }
}
} // namespace
Status EstimateModularTransformCost(std::span<const ModularChannelView> channels,
                                    ModularTransformCost *out) {
  if (!out)
    return Status::InvalidArgument("Null Modular transform cost output");
  ModularTransformCost cost;
  for (const auto &ch : channels) {
    if (ch.descriptor.extent.empty())
      continue;
    if (auto s = ch.Validate(); !s.ok())
      return s;
    size_t area;
    if (!ch.descriptor.extent.try_area(&area) || area > INT32_MAX ||
        area > (SIZE_MAX / 128) - cost.pixels)
      return Status::InvalidArgument("Modular transform cost population overflow");
    cost.pixels += area;
  }
  for (const auto &ch : channels) {
    if (ch.descriptor.extent.empty())
      continue;
    std::array<Histogram, 17> histograms{};
    Collect(ch, &histograms, &cost.extra_bits);
    for (const auto &histogram : histograms) {
      const float bits = Shannon(histogram);
      const size_t integer = static_cast<size_t>(bits);
      cost.histogram_bits += integer;
      cost.histogram_fraction += bits - static_cast<float>(integer);
    }
  }
  cost.bits = static_cast<float>(cost.histogram_bits + cost.extra_bits +
                                 static_cast<size_t>(cost.histogram_fraction));
  *out = cost;
  return Status::Ok();
}
} // namespace gjxl::modular_internal
