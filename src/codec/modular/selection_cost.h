// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/image.h"
#include "codec/modular/prediction.h"

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

// Reference lossless e9 WP candidate order. Preset zero is the default header.
inline constexpr std::array<WeightedPredictorParameters, 5> kSelectionWeightedPresets{{
    {},
    {{8, 8, 4, 0, 3, 23, 2}, {13, 12, 12, 11}},
    {{10, 9, 7, 0, 0, 16, 9}, {13, 12, 13, 12}},
    {{16, 8, 0, 16, 0, 23, 0}, {13, 13, 12, 12}},
    {{10, 10, 5, 5, 5, 12, 4}, {13, 12, 12, 12}},
}};
struct ModularWeightedCost {
  float bits = 0;
  float histogram_bits = 0;
  size_t extra_bits = 0;
  size_t pixels = 0; // Validated input population, including rejected candidates.
  // Residual overflow is a successful estimate of max-float, as in libjxl.
  // Histogram cost stops before the rejected channel; extra bits include it.
  bool residual_overflow = false;
  bool operator==(const ModularWeightedCost &) const = default;
};

// Allocation- and pixel-read-free planning for one live channel's five WP
// history arrays. Excludes borrowed planes, the result, allocator overhead and
// fixed stack histograms (<18 KiB). Candidate mode must be in [0, 5).
[[nodiscard]] Status
ComputeModularWeightedCostStorageBound(std::span<const ModularChannelView> channels, size_t mode,
                                       resource_budget_internal::HostStorageBound *out);
// Same view/population and atomic-output contract as the gradient estimator.
// WP histories use kPreparation ownership and release before return. History
// narrowing wraps as in the reference; residual overflow rejects the candidate.
[[nodiscard]] Status EstimateModularWeightedCost(std::span<const ModularChannelView> channels,
                                                 size_t mode, ModularWeightedCost *out);

} // namespace gjxl::modular_internal
