// Copyright (c) the JPEG XL Project Authors.
// Copyright (c) 2026 Yunho Cho
// SPDX-License-Identifier: BSD-3-Clause
// Adapted from pinned libjxl e8ff0976 enc_ma.cc and fast_math-inl.h.
// Native storage, validation and SIMD operations; no libjxl/Highway dependency.
#include "codec/modular/histogram_tree_learning.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

namespace gjxl::modular_internal {
namespace {
using resource_budget_internal::ManagedVector;
constexpr size_t kSymbols = 256;
// The last (unevaluated) property bin is cleared after all predictors, so its
// histogram may temporarily hold every predictor's weighted population.
static_assert(kHistogramMaximumSamples <= INT32_MAX / kHistogramPredictors);
static_assert(kHistogramMaximumSamples <= UINT32_MAX / 63);
size_t Padded(size_t value) { return (value + 3) & ~size_t{3}; }

// Four-lane order matches the pinned AArch64 reference. Other architectures
// use the same explicit FMA polynomial and reduction order, without promising
// identical decisions to a reference compiled for a different SIMD width.
#if (defined(__aarch64__) || defined(_M_ARM64)) &&                             \
    !defined(GJXL_HISTOGRAM_SCALAR)
float32x4_t FastLog2(float32x4_t x) {
  const auto bits = vreinterpretq_s32_f32(x);
  const auto exponent =
      vshrq_n_s32(vsubq_s32(bits, vdupq_n_s32(0x3f2aaaab)), 23);
  const auto mantissa =
      vreinterpretq_f32_s32(vsubq_s32(bits, vshlq_n_s32(exponent, 23)));
  const auto z = vsubq_f32(mantissa, vdupq_n_f32(1.0f));
  auto p = vfmaq_f32(vdupq_n_f32(1.4287160470083755E+00f),
                     vdupq_n_f32(7.4245873327820566E-01f), z);
  auto q = vfmaq_f32(vdupq_n_f32(1.0096718572241148E+00f),
                     vdupq_n_f32(1.7409343003366853E-01f), z);
  p = vfmaq_f32(vdupq_n_f32(-1.8503833400518310E-06f), p, z);
  q = vfmaq_f32(vdupq_n_f32(9.9032814277590719E-01f), q, z);
  return vaddq_f32(vdivq_f32(p, q), vcvtq_f32_s32(exponent));
}
float EstimateBits(const int32_t *counts, size_t symbols) {
  auto total = vdupq_n_s32(0);
  for (size_t i = 0; i < symbols; i += 4)
    total = vaddq_s32(total, vld1q_s32(counts + i));
  total = vdupq_n_s32(vaddvq_s32(total));
  const auto inv_total = vdupq_n_f32(1.0f / vgetq_lane_s32(total, 0));
  const auto minimum = vdupq_n_f32(1.0f / 4096);
  auto bits = vdupq_n_f32(0);
  for (size_t i = 0; i < symbols; i += 4) {
    const auto c = vld1q_s32(counts + i);
    const auto f = vcvtq_f32_s32(c);
    const auto probability = vmaxq_f32(vmulq_f32(f, inv_total), minimum);
    const auto log =
        vbslq_f32(vceqq_s32(c, total), vdupq_n_f32(0), FastLog2(probability));
    bits = vsubq_f32(bits, vmulq_f32(f, log));
  }
  return vaddvq_f32(bits);
}
#else
float FastLog2(float x) {
  const uint32_t bits = std::bit_cast<uint32_t>(x);
  const int32_t exponent = std::bit_cast<int32_t>(bits - 0x3f2aaaabu) >> 23;
  const float mantissa =
      std::bit_cast<float>(bits - (static_cast<uint32_t>(exponent) << 23));
  const float z = mantissa - 1.0f;
  float p = std::fma(7.4245873327820566E-01f, z, 1.4287160470083755E+00f);
  float q = std::fma(1.7409343003366853E-01f, z, 1.0096718572241148E+00f);
  p = std::fma(p, z, -1.8503833400518310E-06f);
  q = std::fma(q, z, 9.9032814277590719E-01f);
  return p / q + static_cast<float>(exponent);
}
float EstimateBits(const int32_t *counts, size_t symbols) {
  // Match the reference's separate multiply and subtract. Implicit contraction
  // changes rounding in close split/predictor decisions; explicit std::fma in
  // FastLog2 remains intentional. Scope this to the scalar fallback.
#pragma STDC FP_CONTRACT OFF
  int32_t total = 0;
  for (size_t i = 0; i < symbols; ++i)
    total += counts[i];
  const float inv_total = 1.0f / total;
  std::array<float, 4> bits{};
  for (size_t i = 0; i < symbols; i += 4) {
    for (size_t lane = 0; lane < 4; ++lane) {
      const int32_t c = counts[i + lane];
      const float log =
          c == total ? 0.0f : FastLog2(std::max(c * inv_total, 1.0f / 4096));
      bits[lane] -= static_cast<float>(c) * log;
    }
  }
  return (bits[0] + bits[1]) + (bits[2] + bits[3]);
}
#endif

struct SamplesView {
  HistogramTrainingSamples &data;
  size_t NumDistinctSamples() const { return data.weights.size(); }
  size_t NumPredictors() const { return data.predictor_count; }
  size_t NumProperties() const { return data.property_count; }
  size_t NumStaticProps() const { return data.static_property_count; }
  size_t Count(size_t row) const { return data.weights[row]; }
  const auto &RTokens(size_t pred) const { return data.residuals[pred]; }
  size_t Token(size_t pred, size_t row) const {
    return data.residuals[pred][row].symbol;
  }
  size_t NumPropertyValues(size_t prop) const {
    return data.thresholds[prop].size() + 1;
  }
  Predictor PredictorFromIndex(size_t pred) const {
    return data.predictors[pred];
  }
  size_t PropertyFromIndex(size_t prop) const { return data.properties[prop]; }
  size_t PredictorIndex(Predictor pred) const {
    return std::find(data.predictors.begin(),
                     data.predictors.begin() + data.predictor_count, pred) -
           data.predictors.begin();
  }
  template <bool S> size_t Property(size_t prop, size_t row) const {
    if constexpr (S)
      return data.static_values[prop][row];
    else
      return data.values[prop][row];
  }
  int32_t UnquantizeProperty(size_t prop, size_t bin) const {
    return data.thresholds[prop][bin];
  }
  uint32_t QuantizeStaticProperty(size_t prop, uint32_t value) const {
    // This is the reference's clipped lookup mapping, derived from its cuts.
    const int64_t clipped = std::clamp<int64_t>(value, -511, 511);
    const auto &cuts = data.thresholds[prop];
    return std::lower_bound(cuts.begin(), cuts.end(), clipped) - cuts.begin();
  }
  void Swap(size_t a, size_t b) { data.Swap(a, b); }
};

void MakeSplitNode(size_t pos, int property, int splitval, Predictor low,
                   Predictor high, DecisionTree *tree) {
  auto &parent = tree->nodes[pos];
  parent.left = static_cast<TreeIndex>(tree->size);
  parent.right = static_cast<TreeIndex>(tree->size + 1);
  parent.split = splitval;
  parent.property = property;
  // Wire-left is strictly greater. Predictor names refer to property values.
  tree->nodes[tree->size] = DecisionNode{};
  tree->nodes[tree->size++].predictor = high;
  tree->nodes[tree->size] = DecisionNode{};
  tree->nodes[tree->size++].predictor = low;
}

enum class IntersectionType { kNone, kPartial, kInside };
IntersectionType BoxIntersects(HistogramStaticRange needle,
                               HistogramStaticRange haystack, uint32_t &axis,
                               uint32_t &value) {
  bool partial = false;
  for (size_t i = 0; i < kHistogramStaticProperties; ++i) {
    if (haystack[i][0] >= needle[i][1] || haystack[i][1] <= needle[i][0])
      return IntersectionType::kNone;
    if (haystack[i][0] <= needle[i][0] && haystack[i][1] >= needle[i][1])
      continue;
    partial = true;
    axis = static_cast<uint32_t>(i);
    value = haystack[i][0] > needle[i][0] && haystack[i][0] < needle[i][1]
                ? haystack[i][0] - 1
                : haystack[i][1] - 1;
  }
  return partial ? IntersectionType::kPartial : IntersectionType::kInside;
}
template <bool S>
void SplitTreeSamples(SamplesView &samples, size_t begin, size_t pos,
                      size_t end, size_t prop, uint32_t value) {
  size_t low = begin, high = pos;
  do {
    while (low < pos && samples.Property<S>(prop, low) <= value)
      ++low;
    while (high < end && samples.Property<S>(prop, high) > value)
      ++high;
    if (low < pos && high < end)
      samples.Swap(low, high);
    ++low;
    ++high;
  } while (low < pos && high < end);
}
template <bool S>
void CollectExtraBitsIncrease(
    const SamplesView &samples,
    const ManagedVector<HistogramResidualToken> &tokens,
    ManagedVector<int32_t> &increase, ManagedVector<size_t> &extra_increase,
    size_t begin, size_t end, size_t prop, size_t symbols) {
  for (size_t i = begin; i < end; ++i) {
    const auto token = tokens[i];
    const size_t count = samples.Count(i);
    const size_t bin = samples.Property<S>(prop, i);
    increase[bin * symbols + token.symbol] += count;
    extra_increase[bin] += token.extra_bits * count;
  }
}

struct NodeInfo {
  size_t pos, begin, end;
  HistogramStaticRange static_prop_range;
  size_t depth;
};
struct CostInfo {
  float cost = std::numeric_limits<float>::max();
  float extra_cost = 0;
  Predictor pred = Predictor::kZero;
  float Cost() const { return cost + extra_cost; }
};
struct Workspace {
  ManagedVector<NodeInfo> nodes;
  ManagedVector<int32_t> counts, above, below, used, increase;
  ManagedVector<uint32_t> total_extra;
  ManagedVector<size_t> extra_increase;
  ManagedVector<CostInfo> costs_l, costs_r;
};

Status ValidateShape(const HistogramTrainingSamples &s,
                     HistogramTreeLearningOptions o) {
  if (!std::isfinite(o.split_threshold) || o.split_threshold < 0 ||
      !std::isfinite(o.fast_decode_multiplier) ||
      o.fast_decode_multiplier < 1 || !o.maximum_samples ||
      o.maximum_samples > kHistogramMaximumSamples || !o.maximum_leaves ||
      o.maximum_leaves > kMaximumTreeLeaves || !o.maximum_depth ||
      o.maximum_depth > kMaximumTreeDepth ||
      o.multipliers.size() > kHistogramMaximumStaticBins ||
      !s.predictor_count || s.predictor_count > kHistogramPredictors ||
      !s.property_count || s.property_count > kPropertyCount ||
      s.static_property_count > kHistogramStaticProperties ||
      s.static_property_count > s.property_count ||
      s.total_samples > o.maximum_samples || s.weights.size() > s.total_samples)
    return Status::InvalidArgument(
        "Invalid histogram Modular learning shape or options");
  std::array<bool, kHistogramPredictors> seen_predictors{};
  for (size_t p = 0; p < s.predictor_count; ++p) {
    const size_t pred = static_cast<size_t>(s.predictors[p]);
    if (pred >= kHistogramPredictors || seen_predictors[pred] ||
        s.residuals[p].size() != s.weights.size())
      return Status::InvalidArgument(
          "Invalid histogram Modular predictor column");
    seen_predictors[pred] = true;
  }
  std::array<bool, kPropertyCount> seen_properties{};
  for (size_t p = 0; p < s.property_count; ++p) {
    const size_t prop = s.properties[p];
    const bool stat = p < s.static_property_count;
    if (prop >= kPropertyCount || seen_properties[prop] ||
        (stat ? prop != p : prop < kHistogramStaticProperties) ||
        s.thresholds[p].size() >= (stat ? kHistogramMaximumStaticBins : 256) ||
        (stat ? s.static_values[p].size()
              : s.values[p - s.static_property_count].size()) !=
            s.weights.size())
      return Status::InvalidArgument(
          "Invalid histogram Modular property column");
    seen_properties[prop] = true;
  }
  for (auto range : o.static_range)
    if (range[0] >= range[1])
      return Status::InvalidArgument("Invalid histogram static range");
  for (const auto &m : o.multipliers) {
    if (!m.multiplier || m.multiplier > INT32_MAX)
      return Status::InvalidArgument("Invalid histogram multiplier");
    for (auto range : m.range)
      if (range[0] >= range[1])
        return Status::InvalidArgument("Invalid histogram multiplier range");
  }
  return Status::Ok();
}
size_t MaximumBins(const HistogramTrainingSamples &s) {
  size_t bins = 1;
  for (size_t p = 0; p < s.property_count; ++p)
    bins = std::max(bins, s.thresholds[p].size() + 1);
  return bins;
}

Status FindBestSplit(SamplesView &tree_samples, Workspace &w,
                     HistogramTreeLearningOptions options, float threshold,
                     std::span<const HistogramMultiplierRange> mul_info,
                     HistogramStaticRange initial_static_prop_range,
                     float fast_decode_multiplier, DecisionTree *tree) {
  auto &nodes = w.nodes;
  nodes.push_back(NodeInfo{0, 0, tree_samples.NumDistinctSamples(),
                           initial_static_prop_range, 1});

  size_t num_predictors = tree_samples.NumPredictors();
  size_t num_properties = tree_samples.NumProperties();

  // TODO(veluca): consider parallelizing the search (processing multiple nodes
  // at a time).
  while (!nodes.empty()) {
    size_t pos = nodes.back().pos;
    size_t begin = nodes.back().begin;
    size_t end = nodes.back().end;

    HistogramStaticRange static_prop_range = nodes.back().static_prop_range;
    const size_t depth = nodes.back().depth;
    nodes.pop_back();
    if (begin == end)
      continue;

    struct SplitInfo {
      size_t prop = 0;
      uint32_t val = 0;
      size_t pos = 0;
      float lcost = std::numeric_limits<float>::max();
      float rcost = std::numeric_limits<float>::max();
      Predictor lpred = Predictor::kZero;
      Predictor rpred = Predictor::kZero;
      float Cost() const { return lcost + rcost; }
    };

    SplitInfo best_split_static_constant;
    SplitInfo best_split_static;
    SplitInfo best_split_nonstatic;
    SplitInfo best_split_nowp;

    // Compute the maximum token in the range.
    size_t max_symbols = 0;
    for (size_t pred = 0; pred < num_predictors; pred++) {
      for (size_t i = begin; i < end; i++) {
        uint32_t tok = tree_samples.Token(pred, i);
        max_symbols = max_symbols > tok + 1 ? max_symbols : tok + 1;
      }
    }
    max_symbols = Padded(max_symbols);
    auto &counts = w.counts;
    auto &tot_extra_bits = w.total_extra;
    std::fill_n(counts.begin(), max_symbols * num_predictors, int32_t{0});
    for (size_t pred = 0; pred < num_predictors; pred++) {
      size_t extra_bits = 0;
      const auto &rtokens = tree_samples.RTokens(pred);
      for (size_t i = begin; i < end; i++) {
        const HistogramResidualToken &rt = rtokens[i];
        size_t count = tree_samples.Count(i);
        size_t eb = rt.extra_bits * count;
        counts[pred * max_symbols + rt.symbol] += count;
        extra_bits += eb;
      }
      tot_extra_bits[pred] = extra_bits;
    }

    float base_bits;
    {
      size_t pred = tree_samples.PredictorIndex(tree->nodes[pos].predictor);
      base_bits =
          EstimateBits(counts.data() + pred * max_symbols, max_symbols) +
          tot_extra_bits[pred];
    }

    SplitInfo *best = &best_split_nonstatic;

    SplitInfo forced_split;
    // The multiplier ranges cut halfway through the current ranges of static
    // properties. We do this even if the current node is not a leaf, to
    // minimize the number of nodes in the resulting tree.
    for (const auto &mmi : mul_info) {
      uint32_t axis;
      uint32_t val;
      IntersectionType t =
          BoxIntersects(static_prop_range, mmi.range, axis, val);
      if (t == IntersectionType::kNone)
        continue;
      if (t == IntersectionType::kInside) {
        tree->nodes[pos].multiplier = mmi.multiplier;
        break;
      }
      if (t == IntersectionType::kPartial) {
        if (axis >= tree_samples.NumStaticProps())
          return Status::InvalidArgument(
              "Multiplier split requires a static property");
        forced_split.val = tree_samples.QuantizeStaticProperty(axis, val);
        if (forced_split.val + 1 >= tree_samples.NumPropertyValues(axis) ||
            tree_samples.UnquantizeProperty(axis, forced_split.val) !=
                static_cast<int64_t>(val))
          return Status::InvalidArgument(
              "Multiplier boundary is absent from quantization cuts");
        forced_split.prop = axis;
        forced_split.lcost = forced_split.rcost = base_bits / 2 - threshold;
        forced_split.lpred = forced_split.rpred = tree->nodes[pos].predictor;
        best = &forced_split;
        best->pos = begin;

        if (best->prop < tree_samples.NumStaticProps()) {
          for (size_t x = begin; x < end; x++) {
            if (tree_samples.Property<true>(best->prop, x) <= best->val) {
              best->pos++;
            }
          }
        } else {
          size_t prop = best->prop - tree_samples.NumStaticProps();
          for (size_t x = begin; x < end; x++) {
            if (tree_samples.Property<false>(prop, x) <= best->val) {
              best->pos++;
            }
          }
        }
        break;
      }
    }

    if (best != &forced_split) {
      auto &prop_value_used_count = w.used;
      auto &count_increase = w.increase;
      auto &extra_bits_increase = w.extra_increase;
      // For each property, compute which of its values are used, and what
      // tokens correspond to those usages. Then, iterate through the values,
      // and compute the entropy of each side of the split (of the form `prop >
      // threshold`). Finally, find the split that minimizes the cost.
      auto &costs_l = w.costs_l;
      auto &costs_r = w.costs_r;
      auto &counts_above = w.above;
      auto &counts_below = w.below;

      // The lower the threshold, the higher the expected noisiness of the
      // estimate. Thus, discourage changing predictors.
      float change_pred_penalty = 800.0f / (100.0f + threshold);
      for (size_t prop = 0; prop < num_properties && base_bits > threshold;
           prop++) {
        const size_t prop_size = tree_samples.NumPropertyValues(prop);
        std::fill_n(prop_value_used_count.begin(), prop_size, int32_t{0});

        size_t first_used = prop_size;
        size_t last_used = 0;

        // TODO(veluca): consider finding multiple splits along a single
        // property at the same time, possibly with a bottom-up approach.
        if (prop < tree_samples.NumStaticProps()) {
          for (size_t i = begin; i < end; i++) {
            size_t p = tree_samples.Property<true>(prop, i);
            prop_value_used_count[p]++;
            last_used = std::max(last_used, p);
            first_used = std::min(first_used, p);
          }
        } else {
          size_t prop_idx = prop - tree_samples.NumStaticProps();
          for (size_t i = begin; i < end; i++) {
            size_t p = tree_samples.Property<false>(prop_idx, i);
            prop_value_used_count[p]++;
            last_used = std::max(last_used, p);
            first_used = std::min(first_used, p);
          }
        }
        std::fill_n(costs_l.begin(), last_used - first_used, CostInfo{});
        std::fill_n(costs_r.begin(), last_used - first_used, CostInfo{});
        // For all predictors, compute the right and left costs of each split.
        for (size_t pred = 0; pred < num_predictors; pred++) {
          // Compute cost and histogram increments for each property value.
          const auto &rtokens = tree_samples.RTokens(pred);
          if (prop < tree_samples.NumStaticProps()) {
            CollectExtraBitsIncrease<true>(tree_samples, rtokens,
                                           count_increase, extra_bits_increase,
                                           begin, end, prop, max_symbols);
          } else {
            CollectExtraBitsIncrease<false>(
                tree_samples, rtokens, count_increase, extra_bits_increase,
                begin, end, prop - tree_samples.NumStaticProps(), max_symbols);
          }
          memcpy(counts_above.data(), counts.data() + pred * max_symbols,
                 max_symbols * sizeof counts_above[0]);
          memset(counts_below.data(), 0, max_symbols * sizeof counts_below[0]);
          size_t extra_bits_below = 0;
          // Exclude last used: this ensures neither counts_above nor
          // counts_below is empty.
          for (size_t i = first_used; i < last_used; i++) {
            if (!prop_value_used_count[i])
              continue;
            extra_bits_below += extra_bits_increase[i];
            // The increase for this property value has been used, and will not
            // be used again: clear it. Also below.
            extra_bits_increase[i] = 0;
            for (size_t sym = 0; sym < max_symbols; sym++) {
              counts_above[sym] -= count_increase[i * max_symbols + sym];
              counts_below[sym] += count_increase[i * max_symbols + sym];
              count_increase[i * max_symbols + sym] = 0;
            }
            float rcost = EstimateBits(counts_above.data(), max_symbols) +
                          tot_extra_bits[pred] - extra_bits_below;
            float lcost = EstimateBits(counts_below.data(), max_symbols) +
                          extra_bits_below;

            float penalty = 0;
            // Never discourage moving away from the Weighted predictor.
            if (tree_samples.PredictorFromIndex(pred) !=
                    tree->nodes[pos].predictor &&
                tree->nodes[pos].predictor != Predictor::kWeighted) {
              penalty = change_pred_penalty;
            }
            // If everything else is equal, disfavour Weighted (slower) and
            // favour Zero (faster if it's the only predictor used in a
            // group+channel combination)
            if (tree_samples.PredictorFromIndex(pred) == Predictor::kWeighted) {
              penalty += 1e-8;
            }
            if (tree_samples.PredictorFromIndex(pred) == Predictor::kZero) {
              penalty -= 1e-8;
            }
            if (rcost + penalty < costs_r[i - first_used].Cost()) {
              costs_r[i - first_used].cost = rcost;
              costs_r[i - first_used].extra_cost = penalty;
              costs_r[i - first_used].pred =
                  tree_samples.PredictorFromIndex(pred);
            }
            if (lcost + penalty < costs_l[i - first_used].Cost()) {
              costs_l[i - first_used].cost = lcost;
              costs_l[i - first_used].extra_cost = penalty;
              costs_l[i - first_used].pred =
                  tree_samples.PredictorFromIndex(pred);
            }
          }
        }
        // Iterate through the possible splits and find the one with minimum sum
        // of costs of the two sides.
        size_t split = begin;
        for (size_t i = first_used; i < last_used; i++) {
          if (!prop_value_used_count[i])
            continue;
          split += prop_value_used_count[i];
          float rcost = costs_r[i - first_used].cost;
          float lcost = costs_l[i - first_used].cost;

          bool uses_wp = tree_samples.PropertyFromIndex(prop) == 15 ||
                         costs_l[i - first_used].pred == Predictor::kWeighted ||
                         costs_r[i - first_used].pred == Predictor::kWeighted;
          bool zero_entropy_side = rcost == 0 || lcost == 0;

          SplitInfo &best_ref =
              tree_samples.PropertyFromIndex(prop) < kHistogramStaticProperties
                  ? (zero_entropy_side ? best_split_static_constant
                                       : best_split_static)
                  : (uses_wp ? best_split_nonstatic : best_split_nowp);
          if (lcost + rcost < best_ref.Cost()) {
            best_ref.prop = prop;
            best_ref.val = i;
            best_ref.pos = split;
            best_ref.lcost = lcost;
            best_ref.lpred = costs_l[i - first_used].pred;
            best_ref.rcost = rcost;
            best_ref.rpred = costs_r[i - first_used].pred;
          }
        }
        // Clear extra_bits_increase and cost_increase for last_used.
        extra_bits_increase[last_used] = 0;
        for (size_t sym = 0; sym < max_symbols; sym++) {
          count_increase[last_used * max_symbols + sym] = 0;
        }
      }

      // Try to avoid introducing WP.
      if (best_split_nowp.Cost() + threshold < base_bits &&
          best_split_nowp.Cost() <= fast_decode_multiplier * best->Cost()) {
        best = &best_split_nowp;
      }
      // Split along static props if possible and not significantly more
      // expensive.
      if (best_split_static.Cost() + threshold < base_bits &&
          best_split_static.Cost() <= fast_decode_multiplier * best->Cost()) {
        best = &best_split_static;
      }
      // Split along static props to create constant nodes if possible.
      if (best_split_static_constant.Cost() + threshold < base_bits) {
        best = &best_split_static_constant;
      }
    }

    if (best->Cost() + threshold < base_bits) {
      uint32_t p = tree_samples.PropertyFromIndex(best->prop);
      int32_t dequant = tree_samples.UnquantizeProperty(best->prop, best->val);
      // Split node and try to split children.
      if (tree->size + 2 > 2 * options.maximum_leaves - 1 ||
          depth >= options.maximum_depth)
        return Status::Unsupported(
            "Histogram Modular fit exceeds tree capacity or depth");
      MakeSplitNode(pos, p, dequant, best->lpred, best->rpred, tree);
      // "Sort" according to winning property
      if (best->prop < tree_samples.NumStaticProps()) {
        SplitTreeSamples<true>(tree_samples, begin, best->pos, end, best->prop,
                               best->val);
      } else {
        SplitTreeSamples<false>(tree_samples, begin, best->pos, end,
                                best->prop - tree_samples.NumStaticProps(),
                                best->val);
      }
      auto new_sp_range = static_prop_range;
      if (p < kHistogramStaticProperties) {
        new_sp_range[p][1] = dequant + 1;
      }
      nodes.push_back(NodeInfo{tree->nodes[pos].right, begin, best->pos,
                               new_sp_range, depth + 1});
      new_sp_range = static_prop_range;
      if (p < kHistogramStaticProperties) {
        new_sp_range[p][0] = dequant + 1;
      }
      nodes.push_back(NodeInfo{tree->nodes[pos].left, best->pos, end,
                               new_sp_range, depth + 1});
    }
  }
  return Status::Ok();
}

} // namespace

void HistogramTrainingSamples::Swap(size_t a, size_t b) {
  if (a == b)
    return;
  for (size_t p = 0; p < predictor_count; ++p)
    std::swap(residuals[p][a], residuals[p][b]);
  for (size_t p = 0; p < static_property_count; ++p)
    std::swap(static_values[p][a], static_values[p][b]);
  for (size_t p = 0; p < property_count - static_property_count; ++p)
    std::swap(values[p][a], values[p][b]);
  std::swap(weights[a], weights[b]);
}

Status ValidateHistogramTrainingSamples(const HistogramTrainingSamples &s,
                                        HistogramTreeLearningOptions o) {
  if (auto status = ValidateShape(s, o); !status.ok())
    return status;
  // Reductions keep complete admission checks while allowing vectorization.
  // Shape validation bounds row count by 2^24; uint16 weights therefore fit
  // in this uint64 sum even for malformed populations.
  uint64_t total = 0;
  uint16_t minimum_weight = UINT16_MAX;
  for (auto count : s.weights) {
    total += count;
    minimum_weight = std::min(minimum_weight, count);
  }
  if (!minimum_weight || total > o.maximum_samples)
    return Status::InvalidArgument("Invalid histogram sample weights");
  if (total != s.total_samples)
    return Status::InvalidArgument("Histogram sample count mismatch");
  for (size_t pred = 0; pred < s.predictor_count; ++pred) {
    uint8_t extra_bits = 0;
    for (auto token : s.residuals[pred])
      extra_bits |= token.extra_bits;
    if (extra_bits > 63)
      return Status::InvalidArgument("Invalid histogram residual token");
  }
  for (size_t p = 0; p < s.property_count; ++p) {
    const auto &cuts = s.thresholds[p];
    if (!std::is_sorted(cuts.begin(), cuts.end()) ||
        std::adjacent_find(cuts.begin(), cuts.end()) != cuts.end() ||
        (!cuts.empty() && (cuts.back() == INT32_MAX ||
                           (p < s.static_property_count && cuts.front() < 0))))
      return Status::InvalidArgument("Invalid histogram property cuts");
    if (p < s.static_property_count) {
      uint32_t maximum_bin = 0;
      for (auto bin : s.static_values[p])
        maximum_bin = std::max(maximum_bin, bin);
      if (maximum_bin > cuts.size())
        return Status::InvalidArgument("Invalid histogram static bin");
    } else {
      uint8_t maximum_bin = 0;
      for (auto bin : s.values[p - s.static_property_count])
        maximum_bin = std::max(maximum_bin, bin);
      if (maximum_bin > cuts.size())
        return Status::InvalidArgument("Invalid histogram property bin");
    }
  }
  return Status::Ok();
}

Status ComputeHistogramTreeLearningStorageBound(
    const HistogramTrainingSamples &s, HistogramTreeLearningOptions options,
    resource_budget_internal::HostStorageBound *out) {
  if (!out)
    return Status::InvalidArgument("Null histogram learning storage output");
  if (auto status = ValidateShape(s, options); !status.ok())
    return status;
  using namespace resource_budget_internal;
  HostStorageBound bound;
  const size_t bins = MaximumBins(s), nodes = 2 * options.maximum_leaves - 1;
  if (!bound.AddVector<Workspace>(1, VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<NodeInfo>(options.maximum_depth,
                                 VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<int32_t>(kSymbols * s.predictor_count,
                                VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<int32_t>(kSymbols, VectorCapacityPolicy::kFreshExact,
                                2) ||
      !bound.AddVector<int32_t>(bins, VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<int32_t>(bins * kSymbols,
                                VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<uint32_t>(s.predictor_count,
                                 VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<size_t>(bins, VectorCapacityPolicy::kFreshExact) ||
      !bound.AddVector<CostInfo>(bins, VectorCapacityPolicy::kFreshExact, 2) ||
      (nodes > kInlineTreeNodes &&
       !bound.AddVector<DecisionNode>(nodes,
                                      VectorCapacityPolicy::kFreshExact)))
    return Status::InvalidArgument(
        "Histogram Modular learning storage overflow");
  *out = bound;
  return Status::Ok();
}

Status LearnHistogramTree(HistogramTrainingSamples &s, DecisionTree *out,
                          HistogramTreeLearningOptions options) try {
  if (!out)
    return Status::InvalidArgument("Null histogram tree output");
  if (auto status = ValidateHistogramTrainingSamples(s, options); !status.ok())
    return status;
  ManagedVector<Workspace> storage(1);
  auto &w = storage[0];
  const size_t bins = MaximumBins(s);
  w.nodes.reserve(options.maximum_depth);
  w.counts.resize(kSymbols * s.predictor_count);
  w.above.resize(kSymbols);
  w.below.resize(kSymbols);
  w.total_extra.resize(s.predictor_count);
  w.used.resize(bins);
  w.increase.resize(bins * kSymbols);
  w.extra_increase.resize(bins);
  w.costs_l.resize(bins);
  w.costs_r.resize(bins);
  DecisionTree result;
  result.nodes.resize(2 * options.maximum_leaves - 1);
  result.nodes[0].predictor = s.predictors[0];
  SamplesView view{s};
  if (auto status = FindBestSplit(view, w, options, options.split_threshold,
                                  options.multipliers, options.static_range,
                                  options.fast_decode_multiplier, &result);
      !status.ok())
    return status;
  TreeLayout layout;
  if (auto status = ValidateTree(result, &layout); !status.ok())
    return status;
  if (result.size <= kInlineTreeNodes)
    result.nodes.resize(result.size); // Releases only; cannot allocate.
  *out = std::move(result);
  return Status::Ok();
} catch (const resource_budget_internal::ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Histogram Modular learning allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Histogram Modular learning storage overflow");
}
} // namespace gjxl::modular_internal
