// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/quantized_tree_learning.h"
#include "codec/modular/entropy_estimate.h"
#include <algorithm>
#include <numeric>

namespace gjxl::modular_internal {
namespace {
// Native implementation of the histogram-learning mechanisms studied in
// libjxl enc_ma.cc / enc_encoding.cc (e8ff0976). Deliberate differences:
// quantiles use actual training properties; sampling is bounded/stratified;
// integer Q12 costs and best-first capacity allocation ensure deterministic
// bounded work. No libjxl code or runtime dependency is used here.
constexpr size_t kBins = 256, kSymbols = 240, kPredictors = 2;
constexpr int64_t kUnit = 4096;
constexpr std::array<Predictor, 2> kPredictor{Predictor::kWeighted, Predictor::kGradient};
using entropy_estimate::ResidualCode;
using resource_budget_internal::ManagedVector;
struct Histogram {
  std::array<uint32_t, kSymbols> counts{};
  int64_t extra = 0;
};
struct Split {
  int64_t gain = 0, cost = INT64_MAX;
  int32_t property = -1, threshold = 0;
  size_t low_predictor = 0, high_predictor = 0;
};
struct Leaf {
  size_t begin = 0, end = 0, node = 0, depth = 1, predictor = 0;
  Split split;
};
struct Workspace {
  ManagedVector<int64_t> logs;
  ManagedVector<uint32_t> order;
  ManagedVector<std::array<ResidualCode, kPredictors>> codes;
  ManagedVector<std::array<uint8_t, kPropertyCount>> properties;
  ManagedVector<Leaf> leaves;
  std::array<std::array<int32_t, kBins - 1>, kPropertyCount> thresholds{};
  std::array<size_t, kPropertyCount> threshold_count{};
  std::array<std::array<Histogram, kBins>, kPredictors> bins{};
  std::array<size_t, kBins> populations{};
  std::array<Histogram, kPredictors> parent{}, low{}, high{};
  int64_t penalty = 0, change_penalty = 0;
};
ResidualCode Code(int32_t residual) {
  const uint32_t value = (static_cast<uint32_t>(residual) << 1) ^ (residual < 0 ? UINT32_MAX : 0);
  if (value < 16)
    return {static_cast<uint8_t>(value), 0};
  const unsigned exponent = std::bit_width(value) - 1;
  // HybridUint 4/1/2, matching the reference learning proxy.
  return {static_cast<uint8_t>(16 + 8 * (exponent - 4) + 4 * ((value >> (exponent - 1)) & 1) +
                               (value & 3)),
          static_cast<uint8_t>(exponent - 3)};
}
void Add(Histogram &h, ResidualCode code) {
  ++h.counts[code.symbol];
  h.extra += code.extra_bits;
}
int64_t Cost(const Histogram &h, size_t count, const Workspace &w) {
  int64_t cost = h.extra * kUnit;
  for (uint32_t n : h.counts)
    if (n)
      cost += int64_t{n} * std::min(w.logs[count] - w.logs[n], 12 * kUnit);
  return cost;
}
struct Choice {
  size_t predictor;
  int64_t cost;
};
Choice Best(const std::array<Histogram, kPredictors> &h, size_t count, size_t parent,
            const Workspace &w) {
  Choice best{1, Cost(h[1], count, w)}; // Prefer Gradient on a true tie.
  int64_t best_score = best.cost;
  const int64_t weighted_cost = Cost(h[0], count, w);
  // Like the reference, the penalty chooses the predictor; split acceptance
  // uses its unpenalized cost. Moving away from Weighted is never penalized.
  const int64_t weighted_score = weighted_cost + (parent == 1 ? w.change_penalty : 0);
  if (weighted_score < best_score)
    best = {0, weighted_cost};
  return best;
}
void Quantize(std::span<const TrainingSample> samples, Workspace &w) {
  for (size_t p = 0; p < kPropertyCount; ++p) {
    std::sort(w.order.begin(), w.order.end(), [&](uint32_t a, uint32_t b) {
      const auto x = samples[a].properties[p], y = samples[b].properties[p];
      return x != y ? x < y : a < b;
    });
    size_t quantile = 1, bins = 0;
    for (size_t i = 0; i < samples.size(); ++i) {
      w.properties[w.order[i]][p] = static_cast<uint8_t>(bins);
      if (i + 1 == samples.size())
        break;
      const int64_t value = samples[w.order[i]].properties[p];
      if (value == samples[w.order[i + 1]].properties[p] || value < INT32_MIN ||
          value >= INT32_MAX || (i + 1) * kBins < quantile * samples.size())
        continue;
      w.thresholds[p][bins++] = static_cast<int32_t>(value);
      while (quantile * samples.size() <= (i + 1) * kBins)
        ++quantile;
    }
    w.threshold_count[p] = bins;
  }
  std::iota(w.order.begin(), w.order.end(), uint32_t{0});
}
void FindSplit(Leaf &leaf, Workspace &w) {
  leaf.split = {};
  if (leaf.end - leaf.begin < 2 || leaf.depth >= kMaximumTreeDepth)
    return;
  w.parent = {};
  for (size_t i = leaf.begin; i < leaf.end; ++i)
    for (size_t p = 0; p < kPredictors; ++p)
      Add(w.parent[p], w.codes[w.order[i]][p]);
  const int64_t base = Cost(w.parent[leaf.predictor], leaf.end - leaf.begin, w);
  if (base <= w.penalty)
    return;
  // Reference preferences: static constant, static, no-WP, then general.
  std::array<Split, 4> best;
  for (size_t property = 0; property < kPropertyCount; ++property) {
    if (!w.threshold_count[property])
      continue;
    w.bins = {};
    w.populations = {};
    for (size_t i = leaf.begin; i < leaf.end; ++i) {
      const size_t sample = w.order[i], bin = w.properties[sample][property];
      ++w.populations[bin];
      for (size_t p = 0; p < kPredictors; ++p)
        Add(w.bins[p][bin], w.codes[sample][p]);
    }
    w.low = {};
    w.high = w.parent;
    size_t low = 0;
    for (size_t b = 0; b < w.threshold_count[property]; ++b) {
      if (!w.populations[b])
        continue;
      low += w.populations[b];
      const size_t high = leaf.end - leaf.begin - low;
      if (!high)
        break;
      for (size_t p = 0; p < kPredictors; ++p) {
        for (size_t s = 0; s < kSymbols; ++s) {
          w.low[p].counts[s] += w.bins[p][b].counts[s];
          w.high[p].counts[s] -= w.bins[p][b].counts[s];
        }
        w.low[p].extra += w.bins[p][b].extra;
        w.high[p].extra -= w.bins[p][b].extra;
      }
      const auto l = Best(w.low, low, leaf.predictor, w);
      const auto h = Best(w.high, high, leaf.predictor, w);
      const int64_t cost = l.cost + h.cost;
      const size_t category =
          property < 2 ? (l.cost == 0 || h.cost == 0 ? 0 : 1)
                       : (property != 15 && l.predictor == 1 && h.predictor == 1 ? 2 : 3);
      if (cost < best[category].cost)
        best[category] = {base - cost - w.penalty,   cost,        static_cast<int32_t>(property),
                          w.thresholds[property][b], l.predictor, h.predictor};
    }
  }
  Split selected = best[3];
  // Lossless e9's fast-decode multiplier is 1.001.
  for (size_t category : {size_t{2}, size_t{1}})
    if (best[category].gain > 0 &&
        (selected.cost == INT64_MAX || best[category].cost <= selected.cost + selected.cost / 1000))
      selected = best[category];
  if (best[0].gain > 0)
    selected = best[0];
  if (selected.gain > 0)
    leaf.split = selected;
}
} // namespace
Status ComputeQuantizedTreeLearningStorageBound(resource_budget_internal::HostStorageBound *out,
                                                TreeLearningOptions options) {
  using namespace resource_budget_internal;
  HostStorageBound bound;
  const size_t nodes = 2 * options.maximum_leaves - 1;
  if (!bound.AddVector<Workspace>(1, VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<int64_t>(options.maximum_samples + 1, VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<uint32_t>(options.maximum_samples, VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<std::array<ResidualCode, kPredictors>>(options.maximum_samples,
                                                              VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<std::array<uint8_t, kPropertyCount>>(options.maximum_samples,
                                                            VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<Leaf>(options.maximum_leaves, VectorCapacityPolicy::kFreshExact) ||
      (nodes > kInlineTreeNodes &&
       !bound.AddVector<DecisionNode>(nodes, VectorCapacityPolicy::kFreshExact)))
    return Status::InvalidArgument("Quantized Modular learning storage overflow");
  *out = bound;
  return Status::Ok();
}
Status LearnQuantizedEntropyTree(std::span<const TrainingSample> samples, DecisionTree *out,
                                 TreeLearningOptions options, size_t population,
                                 TreeLearningCheckpoints *checkpoints) {
  ManagedVector<Workspace> storage(1);
  auto &w = storage[0];
  w.logs.resize(samples.size() + 1);
  w.order.resize(samples.size());
  w.codes.resize(samples.size());
  w.properties.resize(samples.size());
  w.leaves.resize(options.maximum_leaves);
  for (uint32_t i = 1; i <= samples.size(); ++i)
    w.logs[i] = entropy_estimate::Log2(i);
  std::iota(w.order.begin(), w.order.end(), uint32_t{0});
  for (size_t i = 0; i < samples.size(); ++i)
    for (size_t p = 0; p < kPredictors; ++p)
      w.codes[i][p] = Code(samples[i].residuals[static_cast<size_t>(kPredictor[p])]);
  Quantize(samples, w);
  // e9 (speed tier 1): 89 * (0.1 + 0.9 * sampled/population). Unlike
  // libjxl's gather bookkeeping, population is exactly the transformed pixels.
  w.penalty = 89 * kUnit / 10 +
              static_cast<int64_t>((uint64_t{89 * 9 * kUnit} * samples.size() / population) / 10);
  w.change_penalty = 800 * kUnit * kUnit / (100 * kUnit + w.penalty);
  DecisionTree tree;
  TreeLearningCheckpoints retained;
  tree.nodes.resize(2 * options.maximum_leaves - 1);
  tree.nodes[0].predictor = kPredictor[0];
  w.leaves[0] = {0, samples.size(), 0, 1, 0, {}};
  FindSplit(w.leaves[0], w);
  size_t leaves = 1;
  while (leaves < options.maximum_leaves) {
    size_t best = 0;
    for (size_t i = 1; i < leaves; ++i)
      if (w.leaves[i].split.gain > w.leaves[best].split.gain)
        best = i;
    const Leaf leaf = w.leaves[best];
    if (leaf.split.gain <= 0)
      break;
    const auto mid =
        std::partition(w.order.begin() + leaf.begin, w.order.begin() + leaf.end, [&](uint32_t i) {
          return samples[i].properties[leaf.split.property] > leaf.split.threshold;
        });
    const size_t boundary = static_cast<size_t>(mid - w.order.begin());
    const auto left = static_cast<TreeIndex>(tree.size++),
               right = static_cast<TreeIndex>(tree.size++);
    tree.nodes[leaf.node] = {.property = leaf.split.property,
                             .split = leaf.split.threshold,
                             .left = left,
                             .right = right};
    tree.nodes[left].predictor = kPredictor[leaf.split.high_predictor];
    tree.nodes[right].predictor = kPredictor[leaf.split.low_predictor];
    w.leaves[best] = {leaf.begin, boundary, left, leaf.depth + 1, leaf.split.high_predictor, {}};
    w.leaves[leaves] = {boundary, leaf.end, right, leaf.depth + 1, leaf.split.low_predictor, {}};
    FindSplit(w.leaves[best], w);
    FindSplit(w.leaves[leaves++], w);
    if (checkpoints)
      if (auto s = retained.Capture(tree); !s.ok())
        return s;
  }
  TreeLayout layout;
  if (auto s = ValidateTree(tree, &layout); !s.ok())
    return s;
  if (checkpoints) {
    if (auto s = retained.Capture(tree, true); !s.ok())
      return s;
    *checkpoints = std::move(retained);
  }
  *out = std::move(tree);
  return Status::Ok();
}
} // namespace gjxl::modular_internal
