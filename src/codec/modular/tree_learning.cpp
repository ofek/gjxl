// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/tree_learning.h"
#include "codec/modular/entropy_estimate.h"
#include <algorithm>
#include <bit>
#include <new>
#include <numeric>
namespace gjxl::modular_internal {
namespace {
uint32_t FoldSigned(int32_t value) {
  return (static_cast<uint32_t>(value) << 1) ^ (value < 0 ? UINT32_MAX : 0);
}
unsigned ResidualWidth(int32_t residual) { return 1 + std::bit_width(FoldSigned(residual)); }
} // namespace
Status LearnDecisionTrees(std::span<const TrainingSample> samples, DecisionTree *single,
                          DecisionTree *split) {
  if (!single || !split || samples.empty() || samples.size() > kDefaultTrainingSamples)
    return Status::InvalidArgument("Invalid Modular training samples");
  const size_t used = samples.size();
  std::array<uint64_t, 14> costs{};
  for (size_t i = 0; i < used; ++i)
    for (size_t p = 0; p < 14; ++p)
      costs[p] += ResidualWidth(samples[i].residuals[p]);
  auto best_predictor = [](const auto &values) {
    size_t best = 5; // Gradient wins equal scores, then ascending wire ID.
    for (size_t p = 0; p < 14; ++p)
      if (values[p] < values[best])
        best = p;
    return best;
  };
  DecisionTree leaf, branch;
  leaf.nodes[0].predictor = static_cast<Predictor>(best_predictor(costs));
  branch = leaf;
  uint64_t best_cost = costs[static_cast<size_t>(leaf.nodes[0].predictor)];
  for (size_t property = 0; property < kPropertyCount; ++property) {
    int64_t sum = 0;
    for (size_t i = 0; i < used; ++i)
      sum += samples[i].properties[property];
    for (int64_t threshold : {int64_t{0}, sum / static_cast<int64_t>(used)}) {
      if (threshold < INT32_MIN || threshold > INT32_MAX)
        continue;
      std::array<std::array<uint64_t, 14>, 2> sides{};
      std::array<size_t, 2> counts{};
      for (size_t i = 0; i < used; ++i) {
        const size_t side = samples[i].properties[property] > threshold ? 0 : 1;
        ++counts[side];
        for (size_t p = 0; p < 14; ++p)
          sides[side][p] += ResidualWidth(samples[i].residuals[p]);
      }
      if (!counts[0] || !counts[1])
        continue;
      const size_t left = best_predictor(sides[0]), right = best_predictor(sides[1]);
      const uint64_t cost = sides[0][left] + sides[1][right] + 32;
      if (cost >= best_cost)
        continue;
      best_cost = cost;
      branch.size = 3;
      branch.nodes[0] = {.property = static_cast<int32_t>(property),
                         .split = static_cast<int32_t>(threshold),
                         .left = 1,
                         .right = 2};
      branch.nodes[1].predictor = static_cast<Predictor>(left);
      branch.nodes[2].predictor = static_cast<Predictor>(right);
    }
  }
  *single = leaf;
  *split = branch;
  return Status::Ok();
}

namespace {
constexpr size_t kPredictors = 14, kSymbols = 128, kThresholds = 32;
constexpr int64_t kCostUnit = 4096;
constexpr size_t kMinimumLeafSamples = 8;

using entropy_estimate::Log2;
using entropy_estimate::ResidualCode;
using entropy_estimate::EncodeResidual;
struct Histogram {
  std::array<uint32_t, kSymbols> counts{};
  int64_t sum_count_log_count = 0, extra_bits = 0;
  size_t occupied = 0;
};
struct Split {
  int64_t gain = 0;
  int32_t property = -1, threshold = 0;
};
struct Leaf {
  size_t begin = 0, end = 0, node = 0, depth = 1;
  Split split;
};
struct LearningWorkspace {
  resource_budget_internal::ManagedVector<int64_t> count_log_count;
  resource_budget_internal::ManagedVector<uint32_t> order;
  resource_budget_internal::ManagedVector<std::array<ResidualCode, kPredictors>> codes;
  std::array<Histogram, kPredictors> parent{}, low{}, high{};
  resource_budget_internal::ManagedVector<Leaf> leaves;
};
void Change(Histogram &h, ResidualCode code, int delta, const LearningWorkspace &w) {
  auto &count = h.counts[code.symbol];
  h.sum_count_log_count -= w.count_log_count[count];
  if (delta > 0) {
    h.occupied += count == 0;
    ++count;
  } else {
    --count;
    h.occupied -= count == 0;
  }
  h.sum_count_log_count += w.count_log_count[count];
  h.extra_bits += delta * int64_t{code.extra_bits};
}
int64_t Cost(const Histogram &h, size_t samples, const LearningWorkspace &w) {
  // Shannon symbol cost plus verbatim extra bits. Penalize the extra model
  // population/header as well as the split itself to avoid tiny contexts.
  return w.count_log_count[samples] - h.sum_count_log_count +
         (h.extra_bits + 2 * static_cast<int64_t>(h.occupied) + 8) * kCostUnit;
}
std::pair<size_t, int64_t> Best(const std::array<Histogram, kPredictors> &histograms, size_t count,
                                const LearningWorkspace &w) {
  size_t predictor = 5; // Same deterministic tie order as the retained learner.
  int64_t cost = Cost(histograms[predictor], count, w);
  for (size_t p = 0; p < kPredictors; ++p) {
    const int64_t candidate = Cost(histograms[p], count, w);
    if (candidate < cost) {
      predictor = p;
      cost = candidate;
    }
  }
  return {predictor, cost};
}
void FindSplit(std::span<const TrainingSample> samples, Leaf &leaf, DecisionTree &tree,
               LearningWorkspace &w) {
  const size_t count = leaf.end - leaf.begin;
  w.parent = {};
  for (size_t i = leaf.begin; i < leaf.end; ++i)
    for (size_t p = 0; p < kPredictors; ++p)
      Change(w.parent[p], w.codes[w.order[i]][p], 1, w);
  const auto [predictor, cost] = Best(w.parent, count, w);
  tree.nodes[leaf.node].predictor = static_cast<Predictor>(predictor);
  leaf.split = {};
  if (count < 2 * kMinimumLeafSamples || leaf.depth >= kMaximumTreeDepth)
    return;
  for (size_t property = 0; property < kPropertyCount; ++property) {
    auto begin = w.order.begin() + leaf.begin, end = w.order.begin() + leaf.end;
    std::sort(begin, end, [&](uint32_t a, uint32_t b) {
      const auto x = samples[a].properties[property], y = samples[b].properties[property];
      return x != y ? x < y : a < b;
    });
    w.low = {};
    w.high = w.parent;
    size_t quantile = 1;
    for (size_t i = 0; i + 1 < count; ++i) {
      for (size_t p = 0; p < kPredictors; ++p) {
        const auto code = w.codes[begin[i]][p];
        Change(w.low[p], code, 1, w);
        Change(w.high[p], code, -1, w);
      }
      const size_t low = i + 1, high = count - low;
      if (low < kMinimumLeafSamples || high < kMinimumLeafSamples ||
          low * (kThresholds + 1) < quantile * count)
        continue;
      const int64_t threshold = samples[begin[i]].properties[property];
      if (threshold == samples[begin[i + 1]].properties[property] || threshold < INT32_MIN ||
          threshold >= INT32_MAX)
        continue;
      // Use actual property boundaries near quantiles, not just zero/mean.
      while (quantile * count <= low * (kThresholds + 1))
        ++quantile;
      const int64_t split_bits = 12 + ResidualWidth(static_cast<int32_t>(threshold));
      const int64_t gain =
          cost - Best(w.low, low, w).second - Best(w.high, high, w).second - split_bits * kCostUnit;
      if (gain > leaf.split.gain)
        leaf.split = {gain, static_cast<int32_t>(property), static_cast<int32_t>(threshold)};
    }
  }
}
} // namespace

Status ComputeTreeLearningStorageBound(resource_budget_internal::HostStorageBound *out,
                                       TreeLearningOptions options) {
  using namespace resource_budget_internal;
  if (!out)
    return Status::InvalidArgument("Null Modular learning storage output");
  if (auto s = ValidateTreeLearningOptions(options); !s.ok())
    return s;
  HostStorageBound bound;
  const size_t nodes = 2 * options.maximum_leaves - 1;
  if (!bound.AddVector<LearningWorkspace>(1, VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<int64_t>(options.maximum_samples + 1, VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<uint32_t>(options.maximum_samples, VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<std::array<ResidualCode, kPredictors>>(options.maximum_samples,
                                                              VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<Leaf>(options.maximum_leaves, VectorCapacityPolicy::kFreshExact) ||
      (nodes > kInlineTreeNodes &&
       !bound.AddVector<DecisionNode>(nodes, VectorCapacityPolicy::kFreshExact)))
    return Status::InvalidArgument("Modular learning storage overflow");
  *out = bound;
  return Status::Ok();
}
Status LearnEntropyTree(std::span<const TrainingSample> samples, DecisionTree *out,
                        TreeLearningOptions options) try {
  if (auto s = ValidateTreeLearningOptions(options); !s.ok())
    return s;
  if (!out || samples.empty() || samples.size() > options.maximum_samples)
    return Status::InvalidArgument("Invalid Modular entropy training samples");
  resource_budget_internal::ManagedVector<LearningWorkspace> storage(1);
  auto &w = storage[0];
  w.count_log_count.resize(samples.size() + 1);
  w.order.resize(samples.size());
  w.codes.resize(samples.size());
  w.leaves.resize(options.maximum_leaves);
  for (uint32_t n = 1; n <= samples.size(); ++n)
    w.count_log_count[n] = int64_t{n} * Log2(n);
  std::iota(w.order.begin(), w.order.end(), uint32_t{0});
  for (size_t i = 0; i < samples.size(); ++i)
    for (size_t p = 0; p < kPredictors; ++p)
      w.codes[i][p] = EncodeResidual(samples[i].residuals[p]);
  DecisionTree tree;
  tree.nodes.resize(2 * options.maximum_leaves - 1);
  w.leaves[0] = {0, samples.size(), 0, 1, {}};
  size_t leaves = 1;
  FindSplit(samples, w.leaves[0], tree, w);
  while (leaves < options.maximum_leaves) {
    size_t best = 0;
    for (size_t i = 1; i < leaves; ++i)
      if (w.leaves[i].split.gain > w.leaves[best].split.gain)
        best = i;
    const auto leaf = w.leaves[best];
    if (leaf.split.gain <= 0)
      break;
    const auto middle =
        std::partition(w.order.begin() + leaf.begin, w.order.begin() + leaf.end, [&](uint32_t i) {
          return samples[i].properties[leaf.split.property] > leaf.split.threshold;
        });
    const size_t boundary = static_cast<size_t>(middle - w.order.begin());
    const TreeIndex left = static_cast<TreeIndex>(tree.size++);
    const TreeIndex right = static_cast<TreeIndex>(tree.size++);
    tree.nodes[leaf.node] = {.property = leaf.split.property,
                             .split = leaf.split.threshold,
                             .left = left,
                             .right = right};
    w.leaves[best] = {leaf.begin, boundary, left, leaf.depth + 1, {}};
    w.leaves[leaves] = {boundary, leaf.end, right, leaf.depth + 1, {}};
    FindSplit(samples, w.leaves[best], tree, w);
    FindSplit(samples, w.leaves[leaves++], tree, w);
  }
  TreeLayout layout;
  if (auto s = ValidateTree(tree, &layout); !s.ok())
    return s;
  *out = std::move(tree);
  return Status::Ok();
} catch (const resource_budget_internal::ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Modular learning allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Modular learning storage overflow");
}

Status MakeFixedWeightedTree(uint32_t bits, DecisionTree *out) {
  if (!out || (bits != 8 && bits != 16))
    return Status::InvalidArgument("Invalid fixed Modular tree depth");
  constexpr std::array<int32_t, 15> cutoffs{-127, -63, -31, -15, -7, -3, -1, 0,
                                            1,    3,   7,   15,  31, 63, 127};
  const int32_t scale = bits == 16 ? 16 : 1;
  DecisionTree tree;
  auto fill = [&](auto &&self, size_t begin, size_t end, size_t id) -> void {
    if (begin == end) {
      tree.nodes[id].predictor = Predictor::kWeighted;
      return;
    }
    const size_t mid = (begin + end) / 2;
    const TreeIndex left = static_cast<TreeIndex>(tree.size++);
    const TreeIndex right = static_cast<TreeIndex>(tree.size++);
    tree.nodes[id] = {.property = 15, .split = cutoffs[mid] * scale, .left = left, .right = right};
    self(self, mid + 1, end, left);
    self(self, begin, mid, right);
  };
  fill(fill, 0, cutoffs.size(), 0);
  *out = std::move(tree);
  return Status::Ok();
}
} // namespace gjxl::modular_internal
