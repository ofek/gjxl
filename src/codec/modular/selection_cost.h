// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/image.h"

namespace gjxl::modular_internal {

// Cheap fixed-gradient proxy used before palette/RCT selection. This matches
// the pinned reference's SIMD estimator, including first-row contexts,
// per-channel histograms, 4/2/0 integers, and integer/fractional accumulation.
// Four-lane entropy arithmetic targets the qualified AArch64 reference. The
// portable implementation preserves that arithmetic, not arbitrary reference
// SIMD widths. It is distinct from the clipped 4/1/2 tree-training objective.
struct ModularTransformCost {
  float bits = 0;
  // Retained for differential attribution; integer histogram cost and summed
  // fractional remainders precede the final reference truncation/conversion.
  size_t histogram_bits = 0;
  float histogram_fraction = 0;
  uint32_t extra_bits = 0; // Reference SIMD accumulation is modulo 2^32.
  size_t pixels = 0;
  bool operator==(const ModularTransformCost &) const = default;
};

// Borrowed, possibly strided planes. Empty planes are ignored, as in libjxl;
// nonempty planes require valid views and at most INT32_MAX samples per plane.
// No managed allocation or pixel copying. Fixed histogram scratch is < 9 KiB
// on the stack; previous output is preserved on validation failure. Does not
// choose/apply transforms, fit trees, emit files, or change encoder defaults.
[[nodiscard]] Status EstimateModularTransformCost(std::span<const ModularChannelView> channels,
                                                  ModularTransformCost *out);

} // namespace gjxl::modular_internal
