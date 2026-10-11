// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "core/resource_context.h"
#include "codec/modular/profile.h"
#include "codec/modular/tree_learning.h"
#include "codestream/modular/gpu_tree_statistics.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/modular_tree_statistics.h"
#include <algorithm>
#include <bit>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::tree_gpu_internal;
using namespace gjxl::resource_budget_internal;
size_t cases = 0, predictor_costs = 0, samples_checked = 0, faults = 0;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(const Status &s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
GpuModularTreeStatistics &Operation(GpuBackend &gpu) {
  auto *op = QueryGpuModularTreeStatistics(gpu);
  Check(op != nullptr, "Metal did not advertise tree statistics");
  return *op;
}
struct Fixture {
  std::vector<TreeLearningCodes> codes;
  std::vector<int64_t> table;
  std::vector<uint32_t> order, low;
  explicit Fixture(size_t n, unsigned pattern = 0)
      : codes(n), table(n + 1), order(n) {
    std::iota(order.begin(), order.end(), 0u);
    for (uint32_t i = 1; i <= n; ++i)
      table[i] = int64_t{i} * entropy_estimate::Log2(i);
    for (size_t i = 0; i < n; ++i)
      for (size_t p = 0; p < 14; ++p) {
        if (pattern == 0) {
          codes[i][p] = {};
        } else if (pattern == 1) {
          const uint32_t word =
              static_cast<uint32_t>(i * 7919 + p * 104729 + (i << (p % 17)));
          codes[i][p] =
              entropy_estimate::EncodeResidual(std::bit_cast<int32_t>(word));
        } else {
          const uint8_t symbol =
              static_cast<uint8_t>(pattern >= 128 ? pattern - 128
                                   : pattern == 2 ? (i + p * 17) % 128
                                                  : (i % 127 ? p : 127));
          codes[i][p] = {symbol, static_cast<uint8_t>(
                                     symbol < 16 ? 0 : 2 + (symbol - 16) / 4)};
        }
      }
    low = {static_cast<uint32_t>(n / 2)};
  }
  TreeStatisticsPreparationInput input() const { return {codes, table}; }
};
struct Histogram {
  std::array<uint32_t, 128> counts{};
  uint64_t extra = 0;
};
int64_t Cost(const Histogram &h, uint32_t population) {
  // Independent ordinary histogram formula, reading each residual's stored
  // extra-bit field rather than deriving it from the symbol as the GPU does.
  int64_t result = int64_t{population} * entropy_estimate::Log2(population);
  size_t occupied = 0;
  for (auto count : h.counts)
    if (count) {
      result -= int64_t{count} * entropy_estimate::Log2(count);
      ++occupied;
    }
  return result + (static_cast<int64_t>(h.extra) +
                   2 * static_cast<int64_t>(occupied) + 8) *
                      4096;
}
std::vector<TreeSplitCosts> Reference(const Fixture &f) {
  std::array<Histogram, 14> low{}, high{};
  for (const auto sample : f.order)
    for (size_t p = 0; p < 14; ++p) {
      const auto code = f.codes[sample][p];
      ++high[p].counts[code.symbol];
      high[p].extra += code.extra_bits;
    }
  std::vector<TreeSplitCosts> result(f.low.size());
  size_t cursor = 0;
  for (size_t t = 0; t < f.low.size(); ++t) {
    for (; cursor < f.low[t]; ++cursor)
      for (size_t p = 0; p < 14; ++p) {
        const auto code = f.codes[f.order[cursor]][p];
        ++low[p].counts[code.symbol];
        --high[p].counts[code.symbol];
        low[p].extra += code.extra_bits;
        high[p].extra -= code.extra_bits;
      }
    for (size_t p = 0; p < 14; ++p) {
      result[t].low[p] = Cost(low[p], f.low[t]);
      result[t].high[p] =
          Cost(high[p], static_cast<uint32_t>(f.order.size()) - f.low[t]);
    }
  }
  return result;
}
void Compare(GpuBackend &gpu, Fixture &f) {
  const auto expected = Reference(f);
  const auto code_copy = f.codes;
  const auto table_copy = f.table;
  const auto order_copy = f.order;
  const auto low_copy = f.low;
  const auto before = gpu.stats();
  std::unique_ptr<PreparedGpuTreeStatistics> prepared;
  Ok(Operation(gpu).PrepareTreeStatistics(f.input(), &prepared));
  Check(prepared && prepared->backend_id() == gpu.id() &&
            prepared->sample_count() == f.codes.size(),
        "Prepared tree statistics identity differs");
  std::unique_ptr<PreparedGpuTreeSplitCosts> output;
  for (unsigned reuse = 0; reuse < 2; ++reuse) {
    Ok(Operation(gpu).MeasureTreeSplitCosts({prepared.get(), f.order, f.low},
                                            &output));
    Check(output && output->backend_id() == gpu.id() &&
              std::ranges::equal(output->costs(), expected),
          "GPU low/high predictor costs differ from scalar histograms");
    ++cases;
    predictor_costs += f.low.size() * 2 * 14;
    samples_checked += f.order.size();
  }
  Check(gpu.stats().committed_submissions == before.committed_submissions + 2,
        "Tree preparation submitted GPU work or evaluation omitted/duplicated "
        "a submission");
  Check(std::memcmp(f.codes.data(), code_copy.data(),
                    f.codes.size() * sizeof(TreeLearningCodes)) == 0 &&
            f.table == table_copy && f.order == order_copy && f.low == low_copy,
        "Statistics operation modified borrowed CPU data");
  prepared.reset();
  f.codes.clear();
  f.table.clear();
  Check(std::ranges::equal(output->costs(), expected),
        "Completed costs borrow prepared/input lifetime");
}
void Cases(GpuBackend &gpu, bool extensive) {
  for (size_t n :
       {size_t{16}, size_t{47}, size_t{257}, size_t{4096}, size_t{65536}})
    for (unsigned pattern = 0; pattern < 4; ++pattern) {
      Fixture f(n, pattern);
      std::reverse(f.order.begin(), f.order.end());
      if (n >= 47) {
        f.low.clear();
        for (uint32_t i = 0; i < 32; ++i)
          f.low.push_back(8 + i * static_cast<uint32_t>(n - 16) / 31);
      }
      Compare(gpu, f);
    }
  {
    Fixture f(8192, 2);
    for (size_t i = 0; i < f.order.size(); ++i)
      f.order[i] = static_cast<uint32_t>((i * 7919) % f.codes.size());
    f.order.resize(1024);
    f.low = {8, 9, 31, 509, 511, 512, 1016};
    Compare(gpu, f);
  }
  {
    Fixture f(512, 1);
    std::fill(f.order.begin(), f.order.end(), 511);
    f.low = {8, 256, 504};
    Compare(gpu, f);
  }
  if (!extensive)
    return;
  for (unsigned symbol = 0; symbol < 128; ++symbol) {
    Fixture f(257, symbol + 128);
    f.low = {8, 17, 128, 249};
    Compare(gpu, f);
  }
  {
    Fixture f(kMaximumTreeStatisticsSamples, 1);
    f.low = {8, static_cast<uint32_t>(f.codes.size() / 2),
             static_cast<uint32_t>(f.codes.size() - 8)};
    f.codes[0][0] = entropy_estimate::EncodeResidual(INT32_MIN);
    f.codes.back()[13] = entropy_estimate::EncodeResidual(INT32_MAX);
    Compare(gpu, f);
  }
}
void StorageAndFailures(GpuBackend &gpu, GpuBackend &foreign) {
  Fixture f(257, 2);
  f.low = {8, 128, 249};
  auto &op = Operation(gpu);
  TreeStatisticsPreparationPlan pp;
  TreeStatisticsStoragePlan ep;
  Ok(ComputeTreeStatisticsPreparationPlan(f.codes.size(), &pp));
  Ok(ComputeTreeStatisticsStoragePlan(f.codes.size(), f.order.size(),
                                      f.low.size(), &ep));
  const size_t exact = pp.storage_bytes + ep.dispatch_bytes + ep.output_bytes;
  for (size_t capacity :
       {exact - 1, exact, std::max(pp.peak_bytes, ep.peak_bytes)}) {
    ResourceBudget budget(capacity);
    ResourceReservation reservation;
    Ok(budget.TryReserve(capacity, &reservation));
    ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
    std::unique_ptr<PreparedGpuTreeStatistics> prepared;
    std::unique_ptr<PreparedGpuTreeSplitCosts> result;
    Ok(op.PrepareTreeStatistics(f.input(), &prepared));
    const auto status =
        op.MeasureTreeSplitCosts({prepared.get(), f.order, f.low}, &result);
    Check(capacity < exact ? status.resource_plan_exceeded() : status.ok(),
          "Native statistics capacity boundary differs");
    if (result)
      Check(budget.snapshot().total.live_capacity_bytes ==
                pp.retained_bytes + ep.retained_bytes,
            "Statistics result retains transient dispatch backing");
    result.reset();
    prepared.reset();
    Check(budget.snapshot().total.live_capacity_bytes == 0 &&
              budget.snapshot().total.pending_count == 0 &&
              budget.snapshot().peak_backing_bytes <= capacity,
          "Statistics storage leaked or exceeded admission");
  }
  std::unique_ptr<PreparedGpuTreeStatistics> prepared;
  std::unique_ptr<PreparedGpuTreeSplitCosts> old;
  Ok(op.PrepareTreeStatistics(f.input(), &prepared));
  Ok(op.MeasureTreeSplitCosts({prepared.get(), f.order, f.low}, &old));
  const auto *old_prepared = prepared.get();
  const auto *old_costs = old.get();
  const auto expected = Reference(f);
  const auto live =
      DefaultResourceBudget().snapshot().total.live_capacity_bytes;
  const auto intact = [&] {
    Check(prepared.get() == old_prepared && old.get() == old_costs &&
              std::ranges::equal(old->costs(), expected) &&
              DefaultResourceBudget().snapshot().total.live_capacity_bytes ==
                  live &&
              DefaultResourceBudget().snapshot().total.pending_count == 0,
          "Statistics failure changed published output or leaked backing");
    ++faults;
  };
  Ok(ArmNextMetalAllocationFailureForTest(gpu));
  Check(op.PrepareTreeStatistics(f.input(), &prepared).code() ==
            StatusCode::kOutOfMemory,
        "Prepared statistics allocation failure differs");
  intact();
  for (unsigned kind = 0; kind < 3; ++kind) {
    if (kind == 0)
      Ok(ArmNextMetalAllocationFailureForTest(gpu));
    else
      Ok(ArmNextMetalSubmissionFailureForTest(gpu, kind == 1, kind == 2));
    const auto s =
        op.MeasureTreeSplitCosts({prepared.get(), f.order, f.low}, &old);
    Check(s.code() == (kind == 0   ? StatusCode::kOutOfMemory
                       : kind == 1 ? StatusCode::kSubmissionFailed
                                   : StatusCode::kDeviceError),
          "Injected statistics device failure differs");
    intact();
  }
  const auto before = gpu.stats();
  f.low[0] = 7;
  Check(!op.MeasureTreeSplitCosts({prepared.get(), f.order, f.low}, &old).ok(),
        "Bad low count accepted");
  intact();
  f.low[0] = 8;
  ++f.table[17];
  Check(!op.PrepareTreeStatistics(f.input(), &prepared).ok(),
        "Noncanonical count table accepted");
  intact();
  --f.table[17];
  f.codes[0][0] = {128, 0};
  Check(!op.PrepareTreeStatistics(f.input(), &prepared).ok(),
        "Bad residual code accepted");
  intact();
  f.codes[0][0] = {};
  Check(!Operation(foreign)
             .MeasureTreeSplitCosts({prepared.get(), f.order, f.low}, &old)
             .ok(),
        "Foreign prepared owner accepted");
  intact();
  struct Impostor final : PreparedGpuTreeStatistics {
    BackendId id;
    size_t count;
    BackendId backend_id() const noexcept override { return id; }
    size_t sample_count() const noexcept override { return count; }
  } impostor;
  impostor.id = gpu.id();
  impostor.count = f.codes.size();
  Check(!op.MeasureTreeSplitCosts({&impostor, f.order, f.low}, &old).ok(),
        "Nonnative prepared owner accepted");
  intact();
  Check(gpu.stats().successful_allocations == before.successful_allocations &&
            gpu.stats().committed_submissions == before.committed_submissions,
        "Malformed statistics preflight reached device work");
  f.order[17] = static_cast<uint32_t>(f.codes.size());
  Check(
      op.MeasureTreeSplitCosts({prepared.get(), f.order, f.low}, &old).code() ==
          StatusCode::kInvalidArgument,
      "Device index error failed to prevent result publication");
  intact();
  Check(gpu.stats().committed_submissions == before.committed_submissions + 1,
        "Device index failure was not checked on the GPU");
}
void LearnerIntegration(GpuBackend &gpu) {
  std::vector<TrainingSample> samples(8192);
  for (size_t i = 0; i < samples.size(); ++i) {
    auto &s = samples[i];
    s.properties[0] = i < 512    ? INT64_MIN
                      : i < 768 ? INT32_MIN
                      : i < 1024 ? -1
                      : i < 1280 ? 0
                      : i < 1536 ? int64_t{INT32_MAX} - 1
                                 : INT64_MAX;
    s.properties[1] = static_cast<int64_t>((i / 5) % 3);
    s.properties[2] = static_cast<int64_t>(i / 128);
    s.properties[3] = static_cast<int64_t>(i);
    s.properties[15] = s.properties[2];
    for (size_t p = 0; p < 14; ++p)
      s.residuals[p] = p == (i / 128) % 14 ? 0 : INT32_MAX;
  }
  const TreeLearningOptions options{samples.size(), 64};
  DecisionTree expected;
  TreeLearningCheckpoints checkpoints;
  ModularProfile scalar;
  {
    ProfileSession session(&scalar);
    Ok(LearnEntropyTree(samples, &expected, options, samples.size(),
                        &checkpoints));
  }
  Check(checkpoints.size >= 2,
        "Native learner fixture omitted intermediate checkpoints");
  for (const bool profile : {false, true}) {
    DecisionTree actual;
    TreeLearningCheckpoints retained;
    ModularProfile measured;
    const auto before = gpu.stats();
    {
      ProfileSession session(profile ? &measured : nullptr);
      ModularGpuTreeStatistics evaluator(gpu, 512);
      Ok(LearnEntropyTreeWithCostEvaluator(samples, &actual, evaluator, options,
                                           samples.size(), &retained));
    }
    Check(actual == expected && retained == checkpoints &&
              gpu.stats().committed_submissions > before.committed_submissions,
          "Native statistics changed a complete tree or checkpoint");
    if (profile) {
      Check(measured.adaptive_gpu.tree_gpu_preparations == 1 &&
                measured.adaptive_gpu.tree_gpu_groups ==
                    gpu.stats().committed_submissions -
                        before.committed_submissions &&
                scalar.training_work.split_evaluations ==
                    measured.training_work.split_evaluations &&
                scalar.training_work.sorted_properties ==
                    measured.training_work.sorted_properties &&
                scalar.training_work.sorted_samples ==
                    measured.training_work.sorted_samples,
            "Native learner profile changed semantic work/submission counts");
    }
  }
  struct FailingEvaluator final : TreeSplitCostEvaluator {
    GpuBackend &gpu;
    ModularGpuTreeStatistics inner;
    unsigned calls = 0, fail_at, kind;
    FailingEvaluator(GpuBackend &backend, unsigned at, unsigned type)
        : gpu(backend), inner(backend, 512), fail_at(at), kind(type) {}
    bool WantsEvaluation(size_t n) const noexcept override {
      return inner.WantsEvaluation(n);
    }
    Status TryMeasure(std::span<const TreeLearningCodes> codes,
                      std::span<const int64_t> table,
                      std::span<const uint32_t> order,
                      std::span<const uint32_t> low,
                      std::span<TreeSplitCosts> costs, bool *handled) override {
      if (++calls == fail_at) {
        if (kind == 0)
          Ok(ArmNextMetalAllocationFailureForTest(gpu));
        else
          Ok(ArmNextMetalSubmissionFailureForTest(gpu, kind == 1, kind == 2));
      }
      return inner.TryMeasure(codes, table, order, low, costs, handled);
    }
  };
  // First-attempt failures must release temporary prepared backing; failures
  // after reuse may keep only the existing call-local prepared owner alive.
  for (unsigned at : {1u, 2u})
    for (unsigned kind = 0; kind < 3; ++kind) {
      DecisionTree actual = expected;
      TreeLearningCheckpoints retained = checkpoints;
      const auto live =
          DefaultResourceBudget().snapshot().total.live_capacity_bytes;
      {
        FailingEvaluator evaluator(gpu, at, kind);
        const auto status = LearnEntropyTreeWithCostEvaluator(
            samples, &actual, evaluator, options, samples.size(), &retained);
        Check(status.code() ==
                      (kind == 0   ? StatusCode::kOutOfMemory
                       : kind == 1 ? StatusCode::kSubmissionFailed
                                   : StatusCode::kDeviceError) &&
                  evaluator.calls == at && actual == expected &&
                  retained == checkpoints &&
                  DefaultResourceBudget().snapshot().total.pending_count == 0,
              "Native learner retried a device error or changed published outputs");
        if (at == 1)
          Check(DefaultResourceBudget().snapshot().total.live_capacity_bytes == live,
                "Failed first learner evaluation retained prepared backing");
      }
      Check(DefaultResourceBudget().snapshot().total.live_capacity_bytes == live,
            "Failed learner leaked its cached preparation");
      ++faults;
    }
}
} // namespace
int main() try {
  {
    std::unique_ptr<GpuBackend> gpu, embedded;
    Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &gpu));
    Ok(CreateEmbeddedMetalBackend({}, &embedded));
    std::cout << "Tree statistics device: " << gpu->name() << std::endl;
    Cases(*gpu, true);
    Cases(*embedded, false);
    StorageAndFailures(*gpu, *embedded);
    LearnerIntegration(*gpu);
  }
  Check(DefaultResourceBudget().snapshot().total.live_capacity_bytes == 0 &&
            DefaultResourceBudget().snapshot().total.pending_count == 0,
        "Tree statistics tests leaked backing");
  std::cout << cases << " exact batches, " << predictor_costs
            << " low/high predictor costs, " << samples_checked
            << " indexed samples, " << faults << " failure-atomic checks\n";
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
