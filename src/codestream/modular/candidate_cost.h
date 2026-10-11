// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/modular/storage_plan.h"

namespace gjxl::modular_internal {
// Disjoint emitted bit categories, including section padding and TOC alignment.
// Empty Modular streams emit neither a stream header nor an ANS terminal state.
// The tree and entropy models use their real serializers, not entropy estimates.
struct ModularCandidateCost {
  uint64_t payload_bits = 0, tree_bits = 0, model_bits = 0;
  uint64_t stream_header_bits = 0, frame_header_bits = 0, toc_bits = 0;
  uint64_t padding_bits = 0;
  size_t bytes = 0, tokens = 0, contexts = 0, clusters = 0;
  bool operator==(const ModularCandidateCost &) const = default;
};

enum class ModularSearchEvaluation { kEncodeAll, kMeasureAll, kShortlist };
struct ModularCandidateRecord {
  size_t index = 0;
  bool checkpoint = false, improved = false;
  double seconds = 0;
  ModularCandidateCost cost;
  bool exact = true;
  double estimated_bits = 0;
  size_t sampled_tokens = 0;
};
// Optional research ledger. Policy is borrowed only during the callback. The
// observer owns its diagnostic storage and must not retain managed policy/model
// owners in the encoder's domain. Errors abort atomic output publication.
class ModularCandidateObserver {
public:
  virtual ~ModularCandidateObserver() = default;
  [[nodiscard]] virtual Status Candidate(const ModularCodingPolicy &policy,
                                         const ModularCandidateRecord &record) = 0;
};

// Canonical internal handoff: layout, tokens, and model have already been
// validated together. token_cost contains exact section costs for this model,
// in stream order. Only compact headers are serialized. Failure preserves out.
[[nodiscard]] Status MeasureModularCandidate(
    const codec_internal::ImageMetadata &metadata, const ModularStreamPlan &layout,
    const PreparedModularTokens &tokens, const EntropyCode &model,
    const EntropyCodeCost &token_cost, ModularCandidateCost *out);

// Shared checked complete-file accounting. section_bits are unpadded and indexed
// by TOC section, not Modular stream. Used by both evaluation and boundary tests.
// Accumulates header/TOC/padding into a local copy, publishing only on success.
[[nodiscard]] Status MeasureModularSections(
    const codec_internal::ImageMetadata &metadata, std::span<const uint64_t> section_bits,
    ModularCandidateCost *cost);
} // namespace gjxl::modular_internal
