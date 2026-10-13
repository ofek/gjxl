// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/histogram_tree_learning.h"
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(const Status &status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
HistogramTrainingSamples Separated(bool weighted = false) {
  HistogramTrainingSamples s;
  s.predictor_count = 1;
  s.predictors[0] = Predictor::kGradient;
  s.property_count = 1;
  s.properties[0] = 9;
  s.thresholds[0] = {-7};
  for (size_t i = 0; i < 128; ++i) {
    s.weights.push_back(weighted ? 65535 : 1);
    s.total_samples += s.weights.back();
    s.values[0].push_back(i % 2);
    s.residuals[0].push_back({static_cast<uint8_t>(i % 2 ? 17 : 3), 0});
  }
  return s;
}
void DecisionsAndWeights() {
  for (bool weighted : {false, true}) {
    auto s = Separated(weighted);
    HistogramTreeLearningOptions options;
    options.maximum_samples = kHistogramMaximumSamples;
    options.maximum_leaves = 16;
    DecisionTree tree;
    Ok(LearnHistogramTree(s, &tree, options));
    Check(tree.size == 3 && tree.nodes[0].property == 9 &&
              tree.nodes[0].split == -7 && tree.nodes[0].left == 1 &&
              tree.nodes[0].right == 2 &&
              tree.nodes[1].predictor == Predictor::kGradient &&
              tree.nodes[2].predictor == Predictor::kGradient,
          "Weighted signed split or predictor changed");
    for (size_t i = 0; i < s.weights.size(); ++i)
      Check(s.residuals[0][i].symbol == (s.values[0][i] ? 17 : 3),
            "Partition broke row association");
    DecisionTree repeated;
    Ok(LearnHistogramTree(s, &repeated, options));
    Check(repeated == tree, "Fitting changed after row partitioning");
  }
  auto s = Separated();
  for (auto &token : s.residuals[0])
    token = {0, 0};
  DecisionTree tree;
  Ok(LearnHistogramTree(s, &tree));
  Check(tree.size == 1, "Zero entropy made a useless split");
  s.weights.clear();
  s.values[0].clear();
  s.residuals[0].clear();
  s.total_samples = 0;
  Ok(LearnHistogramTree(s, &tree));
  Check(tree.size == 1 && tree.nodes[0].predictor == Predictor::kGradient,
        "Empty finalized samples lost root predictor");
}
void MultipliersAndLimits() {
  auto s = Separated();
  s.properties[0] = 0;
  s.static_property_count = 1;
  s.static_values[0].assign(s.values[0].begin(), s.values[0].end());
  s.values[0].clear();
  s.thresholds[0] = {0};
  for (auto &token : s.residuals[0])
    token = {0, 0};
  const std::array<HistogramMultiplierRange, 2> multipliers{
      {{{{{0, 1}, {0, 1}}}, 2}, {{{{1, 2}, {0, 1}}}, 4}}};
  HistogramTreeLearningOptions options;
  options.static_range = {{{0, 2}, {0, 1}}};
  options.multipliers = multipliers;
  DecisionTree tree;
  Ok(LearnHistogramTree(s, &tree, options));
  Check(tree.size == 3 && tree.nodes[0].property == 0 &&
            tree.nodes[0].split == 0 && tree.nodes[1].multiplier == 4 &&
            tree.nodes[2].multiplier == 2,
        "Forced multiplier boundary or child direction changed");
  const auto saved = tree;
  options.maximum_leaves = 1;
  Check(LearnHistogramTree(s, &tree, options).code() ==
                StatusCode::kUnsupported &&
            tree == saved,
        "Leaf capacity silently truncated or published");
  options.maximum_leaves = 16;
  options.maximum_depth = 1;
  Check(LearnHistogramTree(s, &tree, options).code() ==
                StatusCode::kUnsupported &&
            tree == saved,
        "Depth capacity silently truncated or published");
  options.maximum_depth = kMaximumTreeDepth;
  s.thresholds[0].clear();
  std::fill(s.static_values[0].begin(), s.static_values[0].end(), 0);
  Check(!LearnHistogramTree(s, &tree, options).ok() && tree == saved,
        "Missing multiplier cut was accepted or published");
}
void ValidationAndFailures() {
  const auto original = Separated();
  auto rejects = [&](auto modify, HistogramTreeLearningOptions options = {}) {
    auto s = original;
    modify(s);
    DecisionTree tree;
    tree.nodes[0].predictor = Predictor::kZero;
    const auto saved = tree;
    ArmNextManagedHostAllocationFailureForTest();
    const auto status = LearnHistogramTree(s, &tree, options);
    Check(status.code() == StatusCode::kInvalidArgument && tree == saved &&
              ManagedHostAllocationFailurePendingForTest(),
          "Invalid samples allocated or published");
    DisarmManagedHostAllocationFailureForTest();
  };
  rejects([](auto &s) { s.predictor_count = 15; });
  rejects([](auto &s) { s.predictors[0] = Predictor::kInvalid; });
  rejects([](auto &s) { s.property_count = 17; });
  rejects([](auto &s) { s.properties[0] = 16; });
  rejects([](auto &s) { s.properties[0] = 0; });
  rejects([](auto &s) { s.weights[0] = 0; });
  rejects([](auto &s) { s.weights[0] = 2; });
  rejects([](auto &s) { s.total_samples = kMaximumTrainingSamples + 1; });
  rejects([](auto &s) { s.residuals[0].pop_back(); });
  rejects([](auto &s) { s.values[0][0] = 2; });
  rejects([](auto &s) { s.thresholds[0] = {0, -1}; });
  rejects([](auto &s) { s.thresholds[0] = {0, 0}; });
  rejects([](auto &s) { s.thresholds[0] = {INT32_MAX}; });
  rejects([](auto &s) { s.residuals[0][0].extra_bits = 64; });
  // Reductions must retain rejection across SIMD blocks and their tails.
  rejects([](auto &s) { s.weights.back() = 0; });
  rejects([](auto &s) { s.weights.back() = UINT16_MAX; });
  rejects([](auto &s) { s.values[0].back() = UINT8_MAX; });
  rejects([](auto &s) { s.residuals[0].back().extra_bits = 128; });
  rejects([](auto &s) {
    s.static_property_count = 1;
    s.properties[0] = 0;
    s.thresholds[0] = {0};
    s.static_values[0].assign(s.weights.size(), 0);
    s.static_values[0].back() = UINT32_MAX;
    s.values[0].clear();
  });
  HistogramTreeLearningOptions options;
  options.split_threshold = std::numeric_limits<float>::quiet_NaN();
  rejects([](auto &) {}, options);
  options = {};
  options.maximum_samples = SIZE_MAX;
  rejects([](auto &) {}, options);
  options = {};
  options.maximum_depth = 0;
  rejects([](auto &) {}, options);

  options = {};
  HostStorageBound bound;
  ArmNextManagedHostAllocationFailureForTest();
  Ok(ComputeHistogramTreeLearningStorageBound(original, options, &bound));
  Check(ManagedHostAllocationFailurePendingForTest(),
        "Storage planner allocated");
  DisarmManagedHostAllocationFailureForTest();
  const auto saved_bound = bound;
  options.maximum_leaves = 0;
  Check(!ComputeHistogramTreeLearningStorageBound(original, options, &bound)
                .ok() &&
            bound == saved_bound,
        "Invalid storage plan changed output");
  options = {};
  size_t failures = 0;
  for (size_t fail_at = 0; fail_at < 32; ++fail_at) {
    auto s = original;
    DecisionTree tree, saved = tree;
    ArmManagedHostAllocationFailureAfterForTest(fail_at);
    const auto status = LearnHistogramTree(s, &tree, options);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (status.ok()) {
      Check(pending && failures >= 10,
            "Failure sweep did not cover all fitting allocations");
      break;
    }
    Check(status.code() == StatusCode::kOutOfMemory && tree == saved &&
              !pending,
          "Allocation failure escaped or changed output");
    Check(s.weights == original.weights &&
              s.residuals[0] == original.residuals[0] &&
              s.values[0] == original.values[0],
          "Fitting allocated after beginning to partition rows");
    ++failures;
  }
  Check(failures < 32, "Allocation sweep never succeeded");
  ResourceBudget budget(bound.peak_bytes);
  ResourceReservation reservation;
  Ok(budget.Reserve(bound.peak_bytes, &reservation));
  auto samples = original;
  {
    ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
    DecisionTree result;
    Ok(LearnHistogramTree(samples, &result));
    Check(budget.snapshot().peak_backing_bytes <= bound.peak_bytes,
          "Fitting exceeded storage bound");
  }
  Check(budget.snapshot().total.backing_count == 0,
        "Fitting leaked managed storage");
  ResourceBudget short_budget(bound.peak_bytes - 1);
  ResourceReservation short_reservation;
  Ok(short_budget.Reserve(bound.peak_bytes - 1, &short_reservation));
  samples = original;
  DecisionTree unchanged;
  const auto saved_output = unchanged;
  {
    ResourceContextScope scope(
        {&short_reservation, ResourceClass::kPreparation});
    Check(LearnHistogramTree(samples, &unchanged).resource_plan_exceeded() &&
              unchanged == saved_output &&
              samples.weights == original.weights &&
              samples.values[0] == original.values[0],
          "Fitting escaped an insufficient reservation or published output");
  }
  Check(short_budget.snapshot().total.backing_count == 0,
        "Rejected fitting leaked admitted storage");
}
} // namespace
int main() try {
  DecisionsAndWeights();
  MultipliersAndLimits();
  ValidationAndFailures();
  std::cout << "Histogram learning decisions, weights, limits, storage and "
               "failure checks passed\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
