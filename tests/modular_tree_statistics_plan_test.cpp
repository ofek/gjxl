// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/profile.h"
#include "codec/modular/tree_learning.h"
#include "gpu/metal/kernels/modular_tree_statistics_params.h"
#include "gpu/ops/modular_tree_statistics.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::tree_gpu_internal;
using namespace gjxl::resource_budget_internal;
static_assert(sizeof(gjxl_tree_statistics::Code) == 2 &&
              sizeof(gjxl_tree_statistics::Params) == 32 &&
              offsetof(gjxl_tree_statistics::Params, count_costs_offset) == 16);
static_assert(sizeof(gjxl_tree_statistics::Result) == sizeof(TreeSplitCosts) &&
              alignof(gjxl_tree_statistics::Result) ==
                  alignof(TreeSplitCosts) &&
              offsetof(gjxl_tree_statistics::Result, high) ==
                  offsetof(TreeSplitCosts, high));
static_assert(gjxl_tree_statistics::kIndexError == kTreeStatisticsIndexError &&
              gjxl_tree_statistics::kCodeError == kTreeStatisticsCodeError &&
              gjxl_tree_statistics::kHistogramError ==
                  kTreeStatisticsHistogramError);
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(const Status &status) {
  Check(status.ok(), "Modular tree statistics contract failed");
}
void Shapes() {
  TreeStatisticsPreparationPlan prep;
  TreeStatisticsStoragePlan plan;
  ArmNextManagedHostAllocationFailureForTest();
  Ok(ComputeTreeStatisticsPreparationPlan(64, &prep));
  Check(prep.codes_bytes == 1792 && prep.count_costs_offset == 1792 &&
            prep.storage_bytes == 2312 && prep.retained_bytes == 2312 &&
            prep.peak_bytes == 4624,
        "Prepared codes/count table layout differs");
  const auto saved_prep = prep;
  auto limits = TreeStatisticsDeviceLimits{64, 64, prep.storage_bytes};
  Ok(ComputeTreeStatisticsPreparationPlan(64, &prep, &limits));
  --limits.maximum_buffer_bytes;
  Check(ComputeTreeStatisticsPreparationPlan(64, &prep, &limits).code() ==
                StatusCode::kUnsupported &&
            prep == saved_prep,
        "Preparation one-byte-short limit changed prior output");
  for (const size_t n :
       {size_t{0}, kMaximumTreeStatisticsSamples + 1, SIZE_MAX})
    Check(!ComputeTreeStatisticsPreparationPlan(n, &prep).ok() &&
              prep == saved_prep,
          "Invalid preparation shape published a plan");
  Ok(ComputeTreeStatisticsStoragePlan(64, 64, 1, &plan));
  Check(plan.prepared_bytes == 2312 && plan.indices_bytes == 256 &&
            plan.boundaries_offset == 256 && plan.histograms_offset == 512 &&
            plan.histograms_bytes == 14336 && plan.status_offset == 14848 &&
            plan.dispatch_bytes == 14852 && plan.output_bytes == 224 &&
            plan.retained_bytes == 224 && plan.peak_bytes == 32240,
        "Evaluation arena or live prepared overlap differs");
  Ok(ComputeTreeStatisticsStoragePlan(4096, 4096, 32, &plan));
  Check(plan.histograms_bytes == 236544 && plan.output_bytes == 7168 &&
            plan.peak_bytes == 661008,
        "Maximum threshold layout or peak differs");
  const auto saved_plan = plan;
  limits = {4096, 4096, std::max(plan.dispatch_bytes, plan.prepared_bytes)};
  Ok(ComputeTreeStatisticsStoragePlan(4096, 4096, 32, &plan, &limits));
  --limits.maximum_buffer_bytes;
  Check(
      ComputeTreeStatisticsStoragePlan(4096, 4096, 32, &plan, &limits).code() ==
              StatusCode::kUnsupported &&
          plan == saved_plan,
      "Evaluation one-byte-short device limit changed prior plan");
  limits = {4095, 4096, SIZE_MAX};
  Check(
      ComputeTreeStatisticsStoragePlan(4096, 4096, 32, &plan, &limits).code() ==
              StatusCode::kUnsupported &&
          plan == saved_plan,
      "Prepared sample limit was ignored");
  limits = {4096, 4095, SIZE_MAX};
  Check(
      ComputeTreeStatisticsStoragePlan(4096, 4096, 32, &plan, &limits).code() ==
              StatusCode::kUnsupported &&
          plan == saved_plan,
      "Leaf sample limit was ignored");
  limits = {};
  Check(
      !ComputeTreeStatisticsStoragePlan(4096, 4096, 32, &plan, &limits).ok() &&
          plan == saved_plan,
      "Zero device limits were accepted");
  for (const auto shape :
       std::array<std::array<size_t, 3>, 9>{{{0, 16, 1},
                                             {16, 15, 1},
                                             {16, 17, 1},
                                             {16, 16, 0},
                                             {16, 16, 2},
                                             {4096, 4096, 33},
                                             {SIZE_MAX, 16, 1},
                                             {SIZE_MAX, SIZE_MAX, 1},
                                             {4096, 16, SIZE_MAX}}})
    Check(!ComputeTreeStatisticsStoragePlan(shape[0], shape[1], shape[2], &plan)
                  .ok() &&
              plan == saved_plan,
          "Invalid statistics shape changed prior plan");
  Check(!ComputeTreeStatisticsPreparationPlan(64, nullptr).ok() &&
            !ComputeTreeStatisticsStoragePlan(64, 64, 1, nullptr).ok(),
        "Null plan accepted");
  Ok(ComputeTreeStatisticsStoragePlan(16, 16, 1, &plan));
  Ok(ComputeTreeStatisticsStoragePlan(
      kMaximumTreeStatisticsSamples, kMaximumTreeStatisticsSamples, 32, &plan));
  Check(plan.peak_bytes < INT32_MAX && plan.dispatch_bytes < UINT32_MAX &&
            plan.prepared_bytes < UINT32_MAX,
        "Fixed-width device offset ceiling was exceeded");
  Check(ManagedHostAllocationFailurePendingForTest(),
        "Statistics planning allocated");
  DisarmManagedHostAllocationFailureForTest();
}
void Preparation() {
  std::vector<TreeLearningCodes> codes(257);
  std::vector<int64_t> table(codes.size() + 1);
  for (uint32_t i = 1; i < table.size(); ++i)
    table[i] = int64_t{i} * entropy_estimate::Log2(i);
  for (size_t i = 0; i < codes.size(); ++i)
    for (size_t p = 0; p < 14; ++p) {
      const auto symbol = static_cast<uint8_t>((i + p) % 128);
      codes[i][p] = {symbol, static_cast<uint8_t>(
                                 symbol < 16 ? 0 : 2 + (symbol - 16) / 4)};
    }
  codes[0][0] = entropy_estimate::EncodeResidual(INT32_MIN);
  codes[0][1] = entropy_estimate::EncodeResidual(INT32_MAX);
  TreeStatisticsPreparationInput input{codes, table};
  TreeStatisticsPreparationPlan plan;
  ArmNextManagedHostAllocationFailureForTest();
  Ok(ValidateTreeStatisticsPreparation(input, &plan));
  const auto saved = plan;
  const auto reject = [&] {
    Check(!ValidateTreeStatisticsPreparation(input, &plan).ok() &&
              plan == saved,
          "Invalid prepared data published a plan");
  };
  ++table[0];
  reject();
  --table[0];
  ++table[17];
  reject();
  --table[17];
  --table.back();
  reject();
  ++table.back();
  codes[100][3].symbol = 128;
  reject();
  codes[100][3] = {16, 1};
  reject();
  codes[100][3] = {0, 1};
  reject();
  codes[100][3] = {127, 29};
  Ok(ValidateTreeStatisticsPreparation(input, &plan));
  input.count_log_count = std::span(table).first(table.size() - 1);
  reject();
  input.count_log_count = table;
  input.codes = {};
  reject();
  input.codes = codes;
  Check(!ValidateTreeStatisticsPreparation(input, nullptr).ok(),
        "Null validated plan accepted");
  Check(ManagedHostAllocationFailurePendingForTest(),
        "Statistics validation allocated");
  DisarmManagedHostAllocationFailureForTest();
}
void Inputs() {
  struct Owner final : PreparedGpuTreeStatistics {
    BackendId backend_id() const noexcept override { return 7; }
    size_t sample_count() const noexcept override { return 64; }
  } owner;
  std::array<uint32_t, 64> order;
  std::iota(order.begin(), order.end(), 0u);
  std::array<uint32_t, 3> counts{8, 31, 56};
  TreeStatisticsInput input{&owner, order, counts};
  TreeStatisticsStoragePlan plan;
  ArmNextManagedHostAllocationFailureForTest();
  Ok(ValidateTreeStatisticsInput(input, 7, &plan));
  const auto saved = plan;
  const auto reject = [&] {
    Check(!ValidateTreeStatisticsInput(input, 7, &plan).ok() && plan == saved,
          "Invalid leaf input published a plan");
  };
  counts[0] = 7;
  reject();
  counts[0] = 8;
  counts[1] = 8;
  reject();
  counts[1] = 31;
  counts[2] = 57;
  reject();
  counts[2] = 56;
  std::swap(counts[0], counts[1]);
  reject();
  std::swap(counts[0], counts[1]);
  input.low_counts = {};
  reject();
  input.low_counts = counts;
  input.order = std::span(order).first(15);
  reject();
  input.order = order;
  input.prepared = nullptr;
  reject();
  input.prepared = &owner;
  Check(!ValidateTreeStatisticsInput(input, 8, &plan).ok() && plan == saved,
        "Foreign statistics owner accepted");
  Check(!ValidateTreeStatisticsInput(input, 0, &plan).ok() && plan == saved,
        "Zero backend ID accepted");
  // The host preflight is descriptor-only. The native kernel must catch this
  // bad data index before loading prepared codes; it cannot publish results.
  order[0] = UINT32_MAX;
  Ok(ValidateTreeStatisticsInput(input, 7, &plan));
  std::fill(order.begin(), order.end(), 3); // Valid multiset, not a leaf claim.
  Ok(ValidateTreeStatisticsInput(input, 7, &plan));
  Check(!ValidateTreeStatisticsInput(input, 7, nullptr).ok(),
        "Null input-plan output accepted");
  Check(ManagedHostAllocationFailurePendingForTest(),
        "Leaf preflight allocated");
  DisarmManagedHostAllocationFailureForTest();
  Ok(CheckTreeStatisticsDeviceStatus(0));
  for (auto bit : {kTreeStatisticsIndexError, kTreeStatisticsCodeError})
    Check(CheckTreeStatisticsDeviceStatus(bit).code() ==
              StatusCode::kInvalidArgument,
          "Statistics data error mapping differs");
  Check(CheckTreeStatisticsDeviceStatus(kTreeStatisticsHistogramError).code() ==
                StatusCode::kInternal &&
            CheckTreeStatisticsDeviceStatus(8).code() == StatusCode::kInternal,
        "Statistics protocol error mapping differs");
}
void LearnerHook() {
  struct Evaluator final : TreeSplitCostEvaluator {
    unsigned mode = 1;
    size_t calls = 0;
    bool WantsEvaluation(size_t count) const noexcept override {
      return count >= 16;
    }
    Status TryMeasure(std::span<const TreeLearningCodes> codes,
                      std::span<const int64_t> table,
                      std::span<const uint32_t> order,
                      std::span<const uint32_t> low_counts,
                      std::span<TreeSplitCosts> costs, bool *handled) override {
      ++calls;
      *handled = false;
      if (mode == 0)
        return Status::Ok();
      if (mode == 2)
        return Status::Unsupported(
            "Injected terminal learner provider failure");
      if (mode == 4) {
        *handled = true;
        return Status::Ok(); // Deliberately incomplete provider output.
      }
      for (size_t t = 0; t < low_counts.size(); ++t)
        for (unsigned side = 0; side < 2; ++side) {
          const size_t begin = side ? low_counts[t] : 0;
          const size_t end = side ? order.size() : low_counts[t];
          auto &out = side ? costs[t].high : costs[t].low;
          for (size_t predictor = 0; predictor < 14; ++predictor) {
            std::array<uint32_t, 128> counts{};
            int64_t extra = 0;
            for (size_t i = begin; i < end; ++i) {
              const auto code = codes[order[i]][predictor];
              ++counts[code.symbol];
              extra += code.extra_bits;
            }
            int64_t result = table[end - begin] + (extra + 8) * 4096;
            for (auto count : counts)
              if (count)
                result += 2 * 4096 - table[count];
            out[predictor] = result;
          }
        }
      if (mode == 3)
        costs.back().high.back() = INT64_MAX;
      *handled = true;
      return Status::Ok();
    }
  } evaluator;
  std::vector<TrainingSample> samples(1024);
  for (size_t i = 0; i < samples.size(); ++i) {
    auto &s = samples[i];
    s.properties[0] = i < 64    ? INT64_MIN
                      : i < 96  ? INT32_MIN
                      : i < 128 ? -1
                      : i < 160 ? 0
                      : i < 192 ? int64_t{INT32_MAX} - 1
                                : INT64_MAX;
    s.properties[1] = static_cast<int64_t>((i / 5) % 3);
    s.properties[2] = static_cast<int64_t>(i / 32);
    s.properties[3] = static_cast<int64_t>(i);
    s.properties[15] =
        s.properties[2]; // Equal property gains retain earlier IDs.
    for (size_t p = 0; p < 14; ++p)
      s.residuals[p] = p == (i / 32) % 14 ? 0 : INT32_MAX;
  }
  const TreeLearningOptions options{1024, 64};
  DecisionTree expected;
  TreeLearningCheckpoints checkpoints;
  ModularProfile scalar_profile;
  {
    ProfileSession session(&scalar_profile);
    Ok(LearnEntropyTree(samples, &expected, options, samples.size(),
                        &checkpoints));
  }
  Check(checkpoints.size >= 2,
        "Provider fixture did not exercise intermediate checkpoints");
  for (unsigned mode : {0u, 1u}) {
    evaluator.mode = mode;
    evaluator.calls = 0;
    DecisionTree actual;
    TreeLearningCheckpoints retained;
    ModularProfile provider_profile;
    {
      ProfileSession session(&provider_profile);
      Ok(LearnEntropyTreeWithCostEvaluator(samples, &actual, evaluator, options,
                                           samples.size(), &retained));
    }
    Check(evaluator.calls > 0 && actual == expected && retained == checkpoints,
          "Handled/declined histogram provider changed tree or checkpoints");
    const auto &a = scalar_profile.training_work;
    const auto &b = provider_profile.training_work;
    Check(a.split_evaluations == b.split_evaluations &&
              a.sorted_properties == b.sorted_properties &&
              a.sorted_samples == b.sorted_samples,
          "Provider changed semantic learner work counters");
  }
  for (unsigned mode : {2u, 3u, 4u}) {
    evaluator.mode = mode;
    DecisionTree actual = expected;
    TreeLearningCheckpoints retained = checkpoints;
    const auto status = LearnEntropyTreeWithCostEvaluator(
        samples, &actual, evaluator, options, samples.size(), &retained);
    Check(status.code() == (mode == 2 ? StatusCode::kUnsupported
                                      : StatusCode::kInternal) &&
              actual == expected && retained == checkpoints,
          "Provider error was retried or changed published tree/checkpoints");
  }
  // Constant properties cause no provider call; all-predictor ties retain the
  // original Gradient preference and final-only checkpoint behavior.
  samples.assign(16, {});
  evaluator.calls = 0;
  evaluator.mode = 1;
  DecisionTree actual;
  Ok(LearnEntropyTreeWithCostEvaluator(samples, &actual, evaluator));
  Check(evaluator.calls == 0 && actual.size == 1 &&
            actual.nodes[0].predictor == Predictor::kGradient,
        "Constant/tied leaf changed policy or invoked an empty device batch");
  const auto saved = actual;
  auto quantized = options;
  quantized.mode = TreeLearningMode::kQuantized;
  Check(
      LearnEntropyTreeWithCostEvaluator(samples, &actual, evaluator, quantized)
                  .code() == StatusCode::kUnsupported &&
          actual == saved,
      "Quantized learner entered the original-learner provider contract");
}
} // namespace
int main() try {
  Shapes();
  Preparation();
  Inputs();
  LearnerHook();
  std::cout
      << "Modular learner statistics storage, input and ABI contracts passed\n";
} catch (const std::exception &error) {
  DisarmManagedHostAllocationFailureForTest();
  std::cerr << error.what() << '\n';
  return 1;
}
