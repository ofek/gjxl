// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/transform/transform.h"

namespace gjxl::modular_internal {
struct ReferencePaletteOptions {
  uint8_t begin = 0, count = 1, bit_depth = 8;
  uint32_t maximum_colors = kMaximumPaletteColors;
  bool ordered = true;
};
struct ReferencePaletteCandidate {
  bool applied = false;
  TransformDescriptor transform;
  ModularImage image;
};
// Lossless, zero-delta reference construction for 1..4 equal normal channels,
// bit depths 1..24. Multichannel requested limits are 0..1024; implicit entries
// do not consume that limit and can add another 189 explicit entries. A single
// channel accepts limits through the wire maximum, 70911.
// A rejected proposal returns OK with applied=false and an empty image. Errors
// leave output untouched; input is always borrowed, including on rejection.
[[nodiscard]] Status TryBuildReferencePalette(std::span<const ModularChannelView> input,
                                              size_t metadata,
                                              const ReferencePaletteOptions &options,
                                              ReferencePaletteCandidate *out);
// Allocation/pixel-read-free bound for one fresh candidate's managed backing.
// Excludes borrowed input, a pre-existing output, allocator overhead and the
// fixed stack dictionaries. Large single-channel proposals can also use bounded
// scratch, released before return; rejected proposals can allocate that scratch.
[[nodiscard]] Status
ComputeReferencePaletteStorageBound(std::span<const ModularChannelView> input, size_t metadata,
                                    const ReferencePaletteOptions &options,
                                    resource_budget_internal::HostStorageBound *out);
// Scratch only for one single-channel forward construction. Used by the
// composition planner in addition to output images; never needed by inverse
// transforms or already prepared streams. Extent is the selected source plane.
[[nodiscard]] Status
ComputeReferenceChannelPaletteScratchBound(Extent2D extent, uint32_t maximum_colors,
                                           resource_budget_internal::HostStorageBound *out);
} // namespace gjxl::modular_internal
