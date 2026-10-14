// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
// Scalar arithmetic adapted from pinned libjxl.
#pragma once
#include <array>
#include <cstdint>

namespace gjxl::modular_internal {
// Wide results allow composed transforms to reject overflow before plane storage.
// All kernels require a validated type in [0, 42).
[[nodiscard]] inline std::array<int64_t, 3> ForwardRctWide(std::array<int32_t, 3> rgb,
                                                         uint8_t type) {
  const unsigned permutation = type / 7, transform = type % 7;
  int64_t a = rgb[permutation % 3];
  int64_t b = rgb[(permutation + 1 + permutation / 3) % 3];
  int64_t c = rgb[(permutation + 2 - permutation / 3) % 3];
  if (transform == 6) {
    const int64_t co = a - c, tmp = c + (co >> 1), cg = b - tmp;
    a = tmp + (cg >> 1);
    b = co;
    c = cg;
  } else {
    if (transform / 2 == 1)
      b -= a;
    if (transform / 2 == 2)
      b -= (a + c) >> 1;
    if (transform & 1)
      c -= a;
  }
  return {a, b, c};
}
[[nodiscard]] inline std::array<int64_t, 3> InverseRctWide(std::array<int32_t, 3> transformed,
                                                         uint8_t type) {
  const unsigned permutation = type / 7, transform = type % 7;
  int64_t a = transformed[0], b = transformed[1], c = transformed[2];
  if (transform == 6) {
    const int64_t tmp = a - (c >> 1), g = c + tmp, blue = tmp - (b >> 1);
    a = blue + b;
    b = g;
    c = blue;
  } else {
    if (transform & 1)
      c += a;
    if (transform / 2 == 1)
      b += a;
    if (transform / 2 == 2)
      b += (a + c) >> 1;
  }
  std::array<int64_t, 3> rgb;
  rgb[permutation % 3] = a;
  rgb[(permutation + 1 + permutation / 3) % 3] = b;
  rgb[(permutation + 2 - permutation / 3) % 3] = c;
  return rgb;
}
// Existing unpack/ranking kernels operate on bounded 8/16-bit source samples.
[[nodiscard]] inline std::array<int32_t, 3> ForwardRct(std::array<int32_t, 3> rgb, uint8_t type) {
  const auto v = ForwardRctWide(rgb, type);
  return {static_cast<int32_t>(v[0]), static_cast<int32_t>(v[1]), static_cast<int32_t>(v[2])};
}
[[nodiscard]] inline std::array<int32_t, 3> InverseRct(std::array<int32_t, 3> transformed,
                                                     uint8_t type) {
  const auto v = InverseRctWide(transformed, type);
  return {static_cast<int32_t>(v[0]), static_cast<int32_t>(v[1]), static_cast<int32_t>(v[2])};
}
} // namespace gjxl::modular_internal
