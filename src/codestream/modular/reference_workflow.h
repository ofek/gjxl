// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/histogram_sample_preparation.h"
#include "codec/modular/reference_selection.h"
#include "codestream/modular/prepared_streams.h"

namespace gjxl::modular_internal {
enum class ReferenceBuffering : uint8_t { kAutomatic, kBuffered };
enum class ReferencePredictors : uint8_t { kBest, kAll };
struct ReferenceWorkflowOptions {
  ReferenceBuffering buffering = ReferenceBuffering::kAutomatic;
  ReferencePredictors predictors = ReferencePredictors::kBest;
  size_t maximum_samples = kHistogramMaximumSamples;
  PreparedModularEntropyOptions entropy{
      PreparedModularEntropyPolicy::kDirectRateAns,
      AnsSelectionMode::kEstimated, 64};
};
struct ReferenceLearningStats {
  size_t first_group = 0, stop_group = 0;
  HistogramPreparationStats preparation;
  size_t distinct_samples = 0, tree_nodes = 0;
};
struct ReferenceWorkflowStats {
  bool local_trees = false;
  size_t selection_calls = 0, learning_calls = 0, fits = 0,
         empty_learning_calls = 0;
  size_t scanned_pixels = 0, selected_samples = 0, distinct_samples = 0,
         tree_nodes = 0;
  // Largest individual stage reservation, excluding backing retained by other
  // stages. The caller's ResourceBudget observes the complete managed peak.
  size_t maximum_stage_bytes = 0;
};
// A lossless source has at most four normal planes. One reference palette pass
// adds at most four metadata planes; local passes start without global meta.
inline constexpr size_t kReferenceStreamChannels = 8;
struct ReferenceModularStream {
  size_t id = 0, channel_count = 0, metadata_channels = 0;
  Extent2D learning_extent;
  std::array<ModularChannelView, kReferenceStreamChannels> channels{};
  ReferenceSelectionResult selection;
  DecisionTree tree;
  ReferenceLearningStats learning;
  [[nodiscard]] std::span<const ModularChannelView> training_channels() const {
    return std::span(channels).first(channel_count);
  }
};
// Owns all samples/trees behind view(). Channel spans and local-tree pointers
// point into fixed managed allocations, so moves preserve their addresses.
// The global learning image retains excluded descriptors, while view() exposes
// only the coded global prefix. Empty canonical stream-ID slots are preserved
// during global preparation; local learning uses [0,1).
class ReferenceModularFrame {
public:
  ReferenceModularFrame() = default;
  ReferenceModularFrame(const ReferenceModularFrame &) = delete;
  ReferenceModularFrame &operator=(const ReferenceModularFrame &) = delete;
  ReferenceModularFrame(ReferenceModularFrame &&) noexcept = default;
  ReferenceModularFrame &operator=(ReferenceModularFrame &&) noexcept = default;
  [[nodiscard]] PreparedModularStreamsView view() const;
  [[nodiscard]] std::span<const ReferenceModularStream> streams() const {
    return records_;
  }
  [[nodiscard]] const ReferenceWorkflowStats &stats() const { return stats_; }

private:
  friend Status
  PrepareReferenceModularFrame(PackedModularImageView,
                               const ReferenceWorkflowOptions &,
                               const resource_budget_internal::ResourceBudget &,
                               ReferenceModularFrame *);
  ModularEncoderFrame source_;
  Extent2D extent_;
  PackedModularFormat format_ = PackedModularFormat::kRgb8;
  resource_budget_internal::ManagedVector<ReferenceModularStream> records_;
  resource_budget_internal::ManagedVector<PreparedModularStreamView> streams_;
  ReferenceWorkflowStats stats_;
};

// Private ordinary lossless e9 profile: fixed 256-pixel groups, preserved
// invisible RGB, no responsive transforms, patches, custom trees or extra
// channels beyond unassociated alpha. Automatic mode uses local trees above
// eight groups and skips global palettes. Buffered mode retains global
// palettes and one shared tree. Each selected scope is prepared once and fit
// once; empty learning calls produce a trivial tree without invoking the
// fitter. Caps reject rather than truncate. The result owns its samples and
// does not borrow the packed input.
//
// Staged, nonblocking admission in the caller's managed-memory domain. Before
// every allocating stage, its existing planner bounds the fresh backing and
// TryReserve checks capacity alongside all retained backing. No whole-workflow
// bound is claimed. Busy capacity returns Unavailable; failures preserve *out.
// The caller must account for borrowed packed bytes and previous outputs too.
[[nodiscard]] Status PrepareReferenceModularFrame(
    PackedModularImageView input, const ReferenceWorkflowOptions &options,
    const resource_budget_internal::ResourceBudget &budget,
    ReferenceModularFrame *out);
[[nodiscard]] Status EncodeReferenceModularImage(
    PackedModularImageView input, const ReferenceWorkflowOptions &options,
    const resource_budget_internal::ResourceBudget &budget,
    codestream_internal::CodestreamBuffer *out,
    ReferenceWorkflowStats *stats = nullptr);
} // namespace gjxl::modular_internal
