// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include <bit>
#include <cstdint>

namespace gjxl::modular_internal::entropy_estimate {
// Integer Q12 log2 for positive counts. Used only for encoder policy scoring.
inline uint32_t Log2(uint32_t n) {
  const unsigned exponent = std::bit_width(n) - 1;
  uint64_t value = uint64_t{n} << (31 - exponent);
  uint32_t result = exponent << 12;
  for (unsigned bit = 12; bit-- > 0;) {
    value = (value * value) >> 31;
    if (value >= (uint64_t{1} << 32)) {
      value >>= 1;
      result |= uint32_t{1} << bit;
    }
  }
  return result;
}
struct ResidualCode {
  uint8_t symbol = 0, extra_bits = 0;
};
inline ResidualCode EncodeResidual(int32_t residual) {
  const uint32_t value =
      (static_cast<uint32_t>(residual) << 1) ^ (residual < 0 ? UINT32_MAX : 0);
  if (value < 16)
    return {static_cast<uint8_t>(value), 0};
  const unsigned exponent = std::bit_width(value) - 1;
  // Fixed HybridUint 4/2/0 proxy; actual encoding optimizes its own alphabet.
  return {static_cast<uint8_t>(16 + 4 * (exponent - 4) +
                               ((value >> (exponent - 2)) & 3)),
          static_cast<uint8_t>(exponent - 2)};
}
} // namespace gjxl::modular_internal::entropy_estimate
