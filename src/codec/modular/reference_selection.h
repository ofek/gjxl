// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/selection_cost.h"
#include "codec/modular/transform/reference_palette.h"

namespace gjxl::modular_internal {
// The two palette passes have different channel-compaction thresholds. A
// single-group frame uses kGlobal followed by RCT/WP, without a kLocal pass.
enum class ReferencePaletteStage : uint8_t { kNone, kGlobal, kLocal };
struct ReferenceSelectionOptions {
  ReferencePaletteStage palettes = ReferencePaletteStage::kGlobal;
  uint8_t bit_depth = 8;
  // Zero disables multichannel palettes; negative selects first-occurrence
  // ordering. Channel compaction is controlled by palettes, independently.
  int16_t palette_colors = 1024;
  bool select_rct = true, select_wp = true;
};
inline constexpr std::array<uint8_t, 19> kSelectionRcts{
    0, 6, 5, 10, 26, 40, 12, 19, 8, 4, 9, 15, 16, 17, 32, 33, 2, 1, 3};
struct ReferencePaletteDecision {
  uint8_t begin = 0, count = 0;
  uint32_t maximum_colors = 0;
  bool built = false, accepted = false;
  float before = 0, after = 0; // after is meaningful only when built.
  bool operator==(const ReferencePaletteDecision &) const = default;
};
struct ReferenceSelectionTrace {
  std::array<ReferencePaletteDecision, 6> palettes{};
  size_t palette_count = 0;
  // Calls in execution order, including palette baselines/accepted and rejected
  // constructions, then the 19 RCT candidates when eligible. No tree fitting.
  std::array<float, 32> gradient_costs{};
  size_t gradient_count = 0, rct_count = 0, wp_count = 0;
  std::array<float, 5> wp_costs{};
  uint8_t rct = 0;
  bool operator==(const ReferenceSelectionTrace &) const = default;
};
struct ReferenceSelectionResult {
  // Empty when no transform was accepted: the caller retains its borrowed
  // input. If that input exactly aliases the previous output image, identity
  // selection retains its ownership here. Otherwise contains the complete
  // transformed image, including meta.
  ModularImage image;
  TransformSequence transforms;
  uint8_t wp_mode = 0;
  ReferenceSelectionTrace trace;
};

// Private lossless e9 selection, before tree preparation/fitting. Accepts
// 1..16-bit frame depths and at most four normal planes, before any RCT. Normal
// samples must be in [0,65535], including indices from a preceding palette
// pass. Metadata may be signed. Views may be strided; empty palette metadata
// is supported. Total input population is at most INT32_MAX. The first three
// normal planes must have equal geometry/shifts when RCT selection is enabled.
// Output and borrowed input are unchanged on error. Exact full-image aliasing
// into the previous output is supported; partial overlap is rejected. All new
// backing has kPreparation ownership.
// This does not choose frame/group scope or invoke the learner/entropy coder.
[[nodiscard]] Status SelectReferenceModularEncoding(
    std::span<const ModularChannelView> input, size_t metadata,
    const ReferenceSelectionOptions &options, ReferenceSelectionResult *out);

// Shape-only, allocation- and pixel-read-free bound for a fresh result plus
// transient candidates/RCT planes/WP history. Add any pre-existing output's
// backing separately. Excludes borrowed input, allocator overhead, and fixed
// stack state (the palette constructor uses <35 KiB).
[[nodiscard]] Status ComputeReferenceSelectionStorageBound(
    std::span<const ModularChannelView> input, size_t metadata,
    const ReferenceSelectionOptions &options,
    resource_budget_internal::HostStorageBound *out);
} // namespace gjxl::modular_internal
