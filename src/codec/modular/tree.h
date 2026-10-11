// Copyright (c) the JPEG XL Project Authors.
// SPDX-License-Identifier: BSD-3-Clause
// Scalar arithmetic adapted from pinned libjxl.
#pragma once
#include "codec/modular/inline_storage.h"
#include "codec/modular/scalar_prediction.h"

namespace gjxl::modular_internal {
// Representation limits, independent of encoder learning policy and GPU limits.
inline constexpr size_t kInlineTreeNodes = 31;
inline constexpr size_t kMaximumTreeLeaves = 8192;
inline constexpr size_t kMaximumTreeNodes = 2 * kMaximumTreeLeaves - 1;
// The pinned decoder rejects a stack height of 2048 (root counts as one).
inline constexpr size_t kMaximumTreeDepth = 2047;
// Global trees also have an image-dependent decoder bound. Count original
// source samples, not transformed samples or the training subset. Saturating
// before arithmetic also makes this helper safe for hostile size requests.
inline size_t MaximumGlobalTreeLeaves(uint64_t source_samples) {
  const uint64_t nodes = 1024 + std::min(source_samples, uint64_t{1} << 26) / 16;
  return std::min<uint64_t>(kMaximumTreeLeaves, (nodes + 1) / 2);
}
using TreeIndex = uint16_t;
static_assert(kMaximumTreeNodes <= UINT16_MAX);
inline constexpr size_t kPropertyCount = 16; // No previous-channel properties yet.
struct DecisionNode {
  int32_t property = -1, split = 0;
  TreeIndex left = 0, right = 0; // > split and <= split, respectively.
  Predictor predictor = Predictor::kGradient;
  int32_t offset = 0;
  uint32_t multiplier = 1;
  bool operator==(const DecisionNode &) const = default;
};
struct DecisionTree {
  InlineStorage<DecisionNode, kInlineTreeNodes> nodes;
  size_t size = 1;
  DecisionTree() = default;
  // Policies copy only live nodes, not the learner's reserved capacity. This
  // also keeps GPU-eligible trees inline after a large-budget learning attempt.
  DecisionTree(const DecisionTree &other) : size(other.size) {
    const size_t count = std::min(size, other.nodes.size());
    nodes.resize(count);
    std::copy_n(other.nodes.begin(), count, nodes.begin());
  }
  DecisionTree(DecisionTree &&) noexcept = default;
  DecisionTree &operator=(DecisionTree &&) noexcept = default;
  DecisionTree &operator=(const DecisionTree &other) {
    if (this != &other) {
      DecisionTree copy(other);
      *this = std::move(copy);
    }
    return *this;
  }
  bool operator==(const DecisionTree &other) const {
    if (size != other.size)
      return false;
    if (size > nodes.size() || size > other.nodes.size())
      return nodes == other.nodes;
    return std::equal(nodes.begin(), nodes.begin() + size, other.nodes.begin());
  }
};
struct TreeLayout {
  std::array<TreeIndex, kMaximumTreeNodes> breadth_first{}, context{};
  size_t leaves = 0;
  bool weighted = false;
};
[[nodiscard]] inline Status ValidateTree(const DecisionTree &tree, TreeLayout *out) {
  if (!out || tree.size == 0 || tree.size > kMaximumTreeNodes || tree.size > tree.nodes.size())
    return Status::InvalidArgument("Invalid Modular tree size");
  TreeLayout result;
  std::array<bool, kMaximumTreeNodes> seen{};
  seen[0] = true;
  // Parent links keep validation allocation-free without a property-range
  // array per node. Check only ancestors using this node's property. This is
  // O(nodes * depth), bounded by the representation limit, with no recursion.
  std::array<TreeIndex, kMaximumTreeNodes> parents{};
  size_t queued = 1;
  for (size_t i = 0; i < queued; ++i) {
    const size_t id = result.breadth_first[i];
    const auto &n = tree.nodes[id];
    int32_t lower = INT32_MIN, upper = INT32_MAX;
    size_t depth = 1;
    for (size_t child = id; child != 0;) {
      if (++depth > kMaximumTreeDepth)
        return Status::InvalidArgument("Modular tree exceeds supported depth");
      const auto parent = parents[child];
      const auto &ancestor = tree.nodes[parent];
      if (n.property != -1 && ancestor.property == n.property) {
        if (ancestor.left == child)
          lower = std::max(lower, ancestor.split + 1);
        else
          upper = std::min(upper, ancestor.split);
      }
      child = parent;
    }
    if (n.property == -1) {
      if (static_cast<unsigned>(n.predictor) >= 14 || !n.multiplier || n.multiplier > INT32_MAX)
        return Status::InvalidArgument("Invalid Modular tree leaf");
      result.context[id] = static_cast<TreeIndex>(result.leaves++);
      result.weighted |= n.predictor == Predictor::kWeighted;
    } else {
      if (n.property < 0 || static_cast<size_t>(n.property) >= kPropertyCount ||
          queued + 2 > tree.size)
        return Status::InvalidArgument("Unsupported Modular tree property or shape");
      if (n.split < lower || n.split >= upper)
        return Status::InvalidArgument("Contradictory Modular tree split");
      result.weighted |= n.property == 15;
      for (auto child : {n.left, n.right}) {
        if (child >= tree.size || seen[child])
          return Status::InvalidArgument("Cyclic or shared Modular tree child");
        seen[child] = true;
        parents[child] = static_cast<TreeIndex>(id);
        result.breadth_first[queued++] = child;
      }
    }
  }
  if (queued != tree.size || result.leaves > kMaximumTreeLeaves)
    return Status::InvalidArgument("Unreachable Modular tree node");
  *out = result;
  return Status::Ok();
}
using PredictionProperties = std::array<int64_t, kPropertyCount>;
[[nodiscard]] inline PredictionProperties Properties(PredictionNeighborhood n, size_t channel,
                                                     size_t stream, size_t x, size_t y,
                                                     int64_t previous_gradient,
                                                     int64_t weighted_property) {
  return {static_cast<int64_t>(channel),
          static_cast<int64_t>(stream),
          static_cast<int64_t>(y),
          static_cast<int64_t>(x),
          std::abs(n.top),
          std::abs(n.left),
          n.top,
          n.left,
          n.left - previous_gradient,
          n.left + n.top - n.top_left,
          n.left - n.top_left,
          n.top_left - n.top,
          n.top - n.top_right,
          n.top - n.top_top,
          n.left - n.left_left,
          weighted_property};
}
[[nodiscard]] inline size_t Lookup(const DecisionTree &tree, const PredictionProperties &p) {
  size_t id = 0;
  while (tree.nodes[id].property != -1) {
    const auto &n = tree.nodes[id];
    id = p[n.property] > n.split ? n.left : n.right;
  }
  return id;
}
} // namespace gjxl::modular_internal
