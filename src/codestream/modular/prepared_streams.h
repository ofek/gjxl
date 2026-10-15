// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/modular/storage_plan.h"
#include "codec/modular/weighted_features.h"
#include <optional>

namespace gjxl::modular_internal {
// Borrowed result from a completed, checked token producer for exactly these
// immutable channels, parameters, coding ID and tree. Owner/backend and all
// identities remain live until the last consumer finishes. Contents are trusted
// producer results; validation checks identities/shape without rescanning tokens.
struct PreparedModularStreamTokenView {
  EntropyTokenStreamView tokens;
  std::span<const ModularChannelView> channels;
  const DecisionTree *tree = nullptr;
  WeightedPredictorParameters weighted;
  size_t stream_id = 0;
};
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
  // Optional parallel array of completed channel-local maps; see
  // WeightedFeatureView. Owners remain live through tokenization, separately
  // charged by the caller. Empty entries retain scalar WP for those channels.
  std::span<const WeightedFeatureView> weighted_features;
  // Skips sample tokenization and borrows the completed backing without copying
  // or charging it again. CPU tree-header tokenization and entropy stay intact.
  std::optional<PreparedModularStreamTokenView> completed_tokens;
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
enum class PreparedModularEntropyPolicy {
  kPrefix,
  kAnsFromPrefix,
  kDirectRateAns,
};
struct PreparedModularEntropyOptions {
  PreparedModularEntropyPolicy policy = PreparedModularEntropyPolicy::kAnsFromPrefix;
  // Ignored for Prefix. Requested ANS models use this selection policy without
  // an exact-cost request. Tree coding always retains its existing policy.
  AnsSelectionMode selection = AnsSelectionMode::kExact;
  // Direct rate ANS only, in [1, kMaximumAnsClusters]. Other policies require
  // this default value. The optimizer and storage planner share this limit.
  size_t maximum_ans_clusters = kDefaultDirectAnsClusters;
  // Zero uses the admitted caller's CPU allowance; one is the serial control.
  // Unadmitted or explicitly nested calls stay serial. Local streams use outer
  // workers; a shared/single model may use the existing ANS cluster workers.
  size_t cpu_participants = 0;
};
[[nodiscard]] Status ValidatePreparedModularEntropyOptions(const PreparedModularEntropyOptions &options);
// Allocation-free validation/planning. Borrowed samples, views, trees, prior
// outputs and their lifetimes remain the caller's responsibility. Managed
// bounds exclude borrowed completed tokens and cover fresh tokenization,
// modeling/emission at the resolved CPU allowance, assembly and
// publication. Global transforms define full-image shape before grouping; group
// transforms define each tile's shape after grouping. All provided descriptors
// must match. Plan and encode must use the same options and CPU scope. The
// returned participants is the resolved allowance, not a guaranteed grant.
[[nodiscard]] Status
ComputePreparedModularStreamsStoragePlan(const PreparedModularStreamsView &input,
                                         EntropyCodingMode mode, ModularWorkflowStoragePlan *out);
[[nodiscard]] Status
ComputePreparedModularStreamsStoragePlan(const PreparedModularStreamsView &input,
                                         const PreparedModularEntropyOptions &options,
                                         ModularWorkflowStoragePlan *out);
[[nodiscard]] Status TokenizePreparedModularStreams(const PreparedModularStreamsView &input,
                                                    PreparedModularStreamTokens *out);
// Complete private CPU encode with prescribed decisions; no transform search,
// sample fitting, pixel copy or GPU work. The caller owns admission. All
// borrows must stay immutable/live until return. Failure preserves the previous
// output.
[[nodiscard]] Status
EncodePreparedModularStreams(const PreparedModularStreamsView &input, EntropyCodingMode mode,
                             codestream_internal::CodestreamBuffer *out,
                             AnsSelectionMode selection = AnsSelectionMode::kExact);
[[nodiscard]] Status EncodePreparedModularStreams(const PreparedModularStreamsView &input,
                                                  const PreparedModularEntropyOptions &options,
                                                  codestream_internal::CodestreamBuffer *out);
} // namespace gjxl::modular_internal
