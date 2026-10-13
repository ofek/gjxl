// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/modular/storage_plan.h"

namespace gjxl::modular_internal {
struct PreparedModularStreamView {
  // Already transformed, in coding order. Global stream 0 includes only its
  // coded prefix; excluded full-image channels need no pixel backing here.
  std::span<const ModularChannelView> channels;
  WeightedPredictorParameters weighted;
  uint8_t rct = 0; // Optional leading RCT, preceding the ordered sequence.
  TransformSequence transforms;
  // Null selects the frame's shared tree AND entropy model. A non-null tree
  // selects a local tree AND a model built from this stream's tokens alone.
  const DecisionTree *local_tree = nullptr;
};
struct PreparedModularStreamsView {
  Extent2D source;
  PackedModularFormat format = PackedModularFormat::kRgb8;
  const DecisionTree *global_tree = nullptr;
  // Canonical order: global, all DC groups, all pixel groups (including empty
  // slots). IDs/sections are derived from geometry, never caller-renumbered.
  std::span<const PreparedModularStreamView> streams;
};
struct PreparedModularStreamTokens {
  codestream_internal::Storage<EntropyToken> tokens;
  codestream_internal::Storage<EntropyToken> tree_tokens;
  std::span<const EntropyToken> global_tree_tokens;
  codestream_internal::Storage<EntropyTokenStreamView> streams;
  codestream_internal::Storage<std::span<const EntropyToken>> local_tree_tokens;
  PreparedModularStreamTokens() = default;
  PreparedModularStreamTokens(const PreparedModularStreamTokens &) = delete;
  PreparedModularStreamTokens &operator=(const PreparedModularStreamTokens &) = delete;
  PreparedModularStreamTokens(PreparedModularStreamTokens &&) noexcept = default;
  PreparedModularStreamTokens &operator=(PreparedModularStreamTokens &&) noexcept = default;
};
// Allocation-free validation/planning. Borrowed samples, views, trees, prior
// outputs and their lifetimes remain the caller's responsibility. Managed bounds
// cover fresh tokenization, serial modeling/emission, assembly and publication.
// Global transforms define full-image shape before grouping; group transforms
// define each tile's shape after grouping. All provided descriptors must match.
[[nodiscard]] Status
ComputePreparedModularStreamsStoragePlan(const PreparedModularStreamsView &input,
                                         EntropyCodingMode mode, ModularWorkflowStoragePlan *out);
[[nodiscard]] Status TokenizePreparedModularStreams(const PreparedModularStreamsView &input,
                                                    PreparedModularStreamTokens *out);
// Complete private CPU encode with prescribed decisions; no transform search,
// sample fitting, pixel copy or GPU work. The caller owns admission. All borrows
// must stay immutable/live until return. Failure preserves the previous output.
[[nodiscard]] Status
EncodePreparedModularStreams(const PreparedModularStreamsView &input, EntropyCodingMode mode,
                             codestream_internal::CodestreamBuffer *out,
                             AnsSelectionMode selection = AnsSelectionMode::kExact);
} // namespace gjxl::modular_internal
