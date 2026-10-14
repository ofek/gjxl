// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/tokenization.h"
#include "codec/modular/profile.h"
#include "codec/modular/gradient.h"
#include "codestream/modular/parallel.h"

namespace gjxl::modular_internal {
size_t ModularTreeTokenCount(const DecisionTree &tree, const TreeLayout &layout) {
  return 5 * layout.leaves + 2 * (tree.size - layout.leaves);
}
void TokenizeValidatedModularTree(const DecisionTree &tree, const TreeLayout &layout,
                                  std::span<EntropyToken> tokens) {
  size_t next = 0;
  auto token = [&](uint32_t context, uint32_t value) { tokens[next++] = {context, value}; };
  for (size_t i = 0; i < tree.size; ++i) {
    const auto &node = tree.nodes[layout.breadth_first[i]];
    token(1, node.property + 1);
    if (node.property == -1) {
      token(2, static_cast<uint32_t>(node.predictor));
      token(3, PackSigned(node.offset));
      const unsigned shift = std::countr_zero(node.multiplier);
      token(4, shift);
      token(5, (node.multiplier >> shift) - 1);
    } else
      token(0, PackSigned(node.split));
  }
}
PreparedModularTokens PrepareModularTokenHeader(const ModularCodingPolicy &policy,
                                                const TreeLayout &layout) {
  PreparedModularTokens result;
  result.policy = policy;
  result.context_count = layout.leaves;
  result.tree_token_count = ModularTreeTokenCount(policy.tree, layout);
  result.tree_tokens.resize(result.tree_token_count);
  TokenizeValidatedModularTree(policy.tree, layout,
                               std::span(result.tree_tokens).first(result.tree_token_count));
  return result;
}
Status TokenizeValidatedModularChannel(const ModularChannelView &view, size_t channel,
                                       size_t stream_id, const DecisionTree &tree,
                                       const TreeLayout &layout,
                                       const WeightedPredictorParameters &parameters,
                                       std::span<EntropyToken> tokens,
                                       std::span<const WeightedPredictorFeature> features) {
  const auto [w, h] = view.descriptor.extent;
  if (tokens.size() != w * h || (!features.empty() && features.size() != tokens.size()))
    return Status::InvalidArgument("Invalid Modular channel token extent");
  if (!w || !h)
    return Status::Ok();
  std::optional<WeightedPredictor<resource_budget_internal::ResourceClass::kPreparation>> wp;
  if (layout.weighted && features.empty())
    wp.emplace(w, parameters);
  size_t next = 0;
  for (size_t y = 0; y < h; ++y) {
    int64_t previous_gradient = 0;
    for (size_t x = 0; x < w; ++x) {
      const auto neighbors = Neighbors(view, x, y);
      std::pair<int64_t, int64_t> weighted{};
      if (wp)
        weighted = wp->Predict(x, y, neighbors.top, neighbors.left, neighbors.top_right,
                               neighbors.top_left, neighbors.top_top);
      else if (layout.weighted && !features.empty())
        weighted = {features[next].prediction, features[next].property15};
      const auto properties = Properties(neighbors, channel, stream_id, x, y,
                                         previous_gradient, weighted.second);
      previous_gradient = properties[9];
      const size_t leaf = Lookup(tree, properties);
      const auto &node = tree.nodes[leaf];
      const int64_t residual = int64_t{view.Row(y)[x]} -
                               Predict(node.predictor, neighbors, weighted.first) - node.offset;
      if (residual % node.multiplier != 0 || residual / node.multiplier < INT32_MIN ||
          residual / node.multiplier > INT32_MAX)
        return Status::InvalidArgument("Modular leaf cannot represent residual exactly");
      tokens[next++] = {layout.context[leaf],
                        PackSigned(static_cast<int32_t>(residual / node.multiplier))};
      if (wp && !wp->Update(view.Row(y)[x], x, y))
        return Status::Unsupported("Modular weighted state exceeds supported range");
    }
  }
  return Status::Ok();
}
Status TokenizeIdentity(const ModularEncoderFrame &frame, const ModularStreamPlan &plan,
                        PreparedModularTokens *out) {
  return TokenizeModular(frame, plan, {}, out);
}
Status TokenizeModular(const ModularEncoderFrame &frame, const ModularStreamPlan &plan,
                       const ModularCodingPolicy &policy, PreparedModularTokens *out,
                       size_t participants) try {
  ProfileScope profile_scope(ProfileStage::kTokens);
  const auto extent = frame.metadata().extent;
  const size_t channels = frame.image().channel_count();
  size_t total = 0;
  if (!out || !channels || plan.geometry.source() != extent)
    return Status::InvalidArgument("Invalid Modular tokenization input");
  for (size_t c = 0; c < channels; ++c) {
    size_t area;
    if (!frame.image().view(c).descriptor.extent.try_area(&area) || area > SIZE_MAX - total)
      return Status::InvalidArgument("Modular token count overflow");
    total += area;
  }
  if (auto s = ValidateCodingPolicy(policy); !s.ok())
    return s;
  if (auto s = ValidateTransformBitDepth(policy.transforms, frame.metadata().bits); !s.ok())
    return s;
  TreeLayout tree_layout;
  if (auto s = ValidateTree(policy.tree, &tree_layout); !s.ok())
    return s;
  auto result = PrepareModularTokenHeader(policy, tree_layout);
  result.tokens.resize(total);
  result.streams.resize(plan.streams.size());
  codestream_internal::Storage<size_t> offsets(plan.streams.size() + 1, 0);
  for (size_t i = 0; i < plan.streams.size(); ++i) {
    const auto &stream = plan.streams[i];
    if (stream.slice_begin > plan.slices.size() ||
        stream.slice_count > plan.slices.size() - stream.slice_begin)
      return Status::InvalidArgument("Invalid Modular stream slices");
    size_t count = offsets[i];
    for (const auto &slice :
         std::span(plan.slices).subspan(stream.slice_begin, stream.slice_count)) {
      size_t area;
      if (!slice.rect.extent.try_area(&area) || count > total || area > total - count)
        return Status::InvalidArgument("Invalid Modular slice coverage");
      count += area;
    }
    offsets[i + 1] = count;
  }
  size_t active_streams = 0;
  for (size_t i = 0; i < plan.streams.size(); ++i)
    active_streams += offsets[i + 1] != offsets[i];
  participants = std::min(participants, std::max(size_t{1}, active_streams));
  if (offsets.back() != total)
    return Status::InvalidArgument("Incomplete Modular token coverage");
  if (auto status = RunModularStreams(
          plan.streams.size(), participants,
          thread_budget_internal::WorkerLaunchSite::kModularTokenization,
          [&](size_t i) -> Status {
            const auto &stream = plan.streams[i];
            size_t next = offsets[i];
            const size_t begin = next;
            size_t property_channel = 0;
            for (const auto &slice :
                 std::span(plan.slices).subspan(stream.slice_begin, stream.slice_count)) {
              if (slice.channel >= channels)
                return Status::InvalidArgument("Invalid Modular identity channel index");
              ModularChannelView view;
              if (auto s = BorrowChannelSlice(frame.image().view(slice.channel), slice.rect, &view);
                  !s.ok())
                return s;
              const auto [w, h] = view.descriptor.extent;
              if (w * h > offsets[i + 1] - next)
                return Status::InvalidArgument("Overlapping Modular token count");
              if (auto s = TokenizeValidatedModularChannel(
                      view, property_channel, stream.id, policy.tree, tree_layout, policy.weighted,
                      std::span(result.tokens).subspan(next, w * h)); !s.ok())
                return s;
              next += w * h;
              ++property_channel;
            }
            result.streams[i] = EntropyTokenStreamView::Interleaved(
                std::span(result.tokens).subspan(begin, next - begin));
            if (next != offsets[i + 1])
              return Status::InvalidArgument("Incomplete Modular stream coverage");
            return Status::Ok();
          });
      !status.ok())
    return status;
  *out = std::move(result);
  return Status::Ok();
} catch (const resource_budget_internal::ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Modular token allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Modular token storage overflow");
}
} // namespace gjxl::modular_internal
