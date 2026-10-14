// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/frame.h"
#include "codestream/entropy.h"
#include "codestream/modular/stream_plan.h"
#include "gpu/ops/modular_tokenization.h"

namespace gjxl::modular_internal {
// Composition helpers shared by full-image and separately prepared streams.
// The caller validates tree/layout, channel view, static IDs and WP parameters,
// provides exact output extents, owns unpublished output, and catches allocation
// errors. A channel failure may leave its unpublished token span partly written.
[[nodiscard]] size_t ModularTreeTokenCount(const DecisionTree &tree, const TreeLayout &layout);
void TokenizeValidatedModularTree(const DecisionTree &tree, const TreeLayout &layout,
                                  std::span<EntropyToken> tokens);
[[nodiscard]] Status TokenizeValidatedModularChannel(
    const ModularChannelView &view, size_t channel, size_t stream_id, const DecisionTree &tree,
    const TreeLayout &layout, const WeightedPredictorParameters &weighted,
    std::span<EntropyToken> tokens);
inline constexpr std::array<EntropyToken, 5> kGradientTreeTokens{
    {{1, 0}, {2, 5}, {3, 0}, {4, 0}, {5, 0}}};
struct PreparedModularTokens {
  InlineStorage<EntropyToken, 5 * kInlineTreeNodes> tree_tokens;
  size_t tree_token_count = 0, context_count = 1;
  ModularCodingPolicy policy;
  // Exactly one backing representation is populated by a producer. A completed
  // GPU owner keeps borrowed split views alive without copying or recharging
  // its arena. The backend must outlive this object and all joined consumers.
  std::unique_ptr<modular_gpu_internal::PreparedGpuModularTokens> gpu_backing;
  codestream_internal::Storage<EntropyToken> tokens;
  // Declared last so views are destroyed before either backing representation.
  codestream_internal::Storage<EntropyTokenStreamView> streams;
  PreparedModularTokens() = default;
  PreparedModularTokens(const PreparedModularTokens &) = delete;
  PreparedModularTokens &operator=(const PreparedModularTokens &) = delete;
  PreparedModularTokens(PreparedModularTokens &&) noexcept = default;
  PreparedModularTokens &operator=(PreparedModularTokens &&) noexcept = default;
};
// Shared scalar/GPU wire metadata, with no sample tokens. Larger trees allocate managed
// header/policy storage. Caller supplies a validated policy and its ValidateTree result.
[[nodiscard]] PreparedModularTokens PrepareModularTokenHeader(
    const ModularCodingPolicy &policy, const TreeLayout &layout);
// Plan must be the canonical BuildModularStreamPlan result for this frame.
// Views refer to the moved token backing. Output replacement is atomic.
[[nodiscard]] Status TokenizeIdentity(const ModularEncoderFrame &frame,
                                      const ModularStreamPlan &plan, PreparedModularTokens *out);
[[nodiscard]] Status TokenizeModular(const ModularEncoderFrame &frame,
                                     const ModularStreamPlan &plan,
                                     const ModularCodingPolicy &policy, PreparedModularTokens *out,
                                     size_t participants = 1);
// Compatibility spelling for the original private RGB8 workflow.
[[nodiscard]] inline Status TokenizeRgb8(const ModularEncoderFrame &frame,
                                         const ModularStreamPlan &plan,
                                         PreparedModularTokens *out) {
  return TokenizeIdentity(frame, plan, out);
}
} // namespace gjxl::modular_internal
