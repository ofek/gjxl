// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/weighted_features.h"
#include "core/parallel_work_internal.h"
#include <iostream>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
using namespace gjxl::thread_budget_internal;
void Check(bool v, const char *message) {
  if (!v)
    throw std::runtime_error(message);
}
void Ok(const Status &s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
constexpr WeightedPredictorFeature kCanary{INT64_MIN, INT64_MAX};
struct Fixture {
  std::vector<std::vector<int32_t>> planes;
  std::vector<WeightedFeatureSlice> slices;
  std::vector<WeightedPredictorFeature> expected;
  WeightedPredictorParameters parameters{{7, 15, 11, 5, 3, 2, 1}, {15, 0, 7, 3}};
  size_t maximum_width = 0;
  Fixture() {
    // Deliberately shrink and grow row widths on a reused worker, retaining
    // strided subviews, odd edges and signed transformed samples.
    size_t next = 0;
    for (auto extent : {Extent2D{257, 31},
                        {1, 19},
                        {3, 1},
                        {256, 257},
                        {17, 19},
                        {385, 3},
                        {1, 1},
                        {255, 256},
                        {3, 5},
                        {128, 65},
                        {2, 37},
                        {257, 1}}) {
      const size_t stride = extent.width + 7, origin = stride + 3;
      auto &plane = planes.emplace_back((extent.height + 2) * stride, INT32_MIN);
      for (size_t y = 0; y < extent.height; ++y)
        for (size_t x = 0; x < extent.width; ++x)
          plane[origin + y * stride + x] =
              int32_t((x * 19397 + y * 47713 + x * y * 997) % 131071) - 65535;
      slices.push_back({{std::span(plane).subspan(origin), {extent}, stride}, next});
      next += extent.width * extent.height;
      maximum_width = std::max(maximum_width, extent.width);
    }
    expected.resize(next);
    for (const auto &slice : slices)
      Ok(PrepareWeightedPredictorFeatures(
          slice.input, parameters,
          std::span(expected).subspan(slice.feature_offset,
                                      slice.input.descriptor.extent.width *
                                          slice.input.descriptor.extent.height)));
  }
};
void Empty(const ExecutionDomain &domain) {
  const auto s = domain.snapshot();
  Check(!s.active_cpu_participants && !s.reserved_cpu_workers && !s.suspended_cpu_workers &&
            !s.waiting_cpu_callers &&
            s.peak_cpu_protected_slots <= s.effective_cpu_participant_limit,
        "Feature execution leaked or exceeded CPU participation");
}
void Compare(Fixture &f, size_t requested, size_t limit, bool nested = false) {
  WeightedFeatureBatchStoragePlan plan;
  Ok(ComputeWeightedFeatureBatchStoragePlan(f.slices.size(), f.maximum_width, requested, &plan));
  ResourceBudget budget(plan.execution.peak_bytes);
  ResourceReservation reservation;
  Ok(budget.Reserve(plan.execution.peak_bytes, &reservation));
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({0, limit}, &domain));
  std::vector<WeightedPredictorFeature> guarded(f.expected.size() + 2, kCanary);
  FeaturePreparationProfile profile;
  {
    CpuExecutionScope cpu;
    Ok(cpu.Start(domain, requested));
    EncodeScope encode(requested);
    ResourceContextScope resources({&reservation, ResourceClass::kPreparation, true, domain.get()});
    std::optional<ParallelScope> outer;
    if (nested)
      outer.emplace(requested, nullptr, CurrentResourceContext());
    Ok(PrepareWeightedPredictorFeatureBatch(f.slices, f.parameters,
                                            std::span(guarded).subspan(1, f.expected.size()),
                                            requested, &profile));
  }
  const size_t participants = nested ? 1 : std::min(requested, limit);
  Check(guarded.front() == kCanary && guarded.back() == kCanary &&
            std::equal(f.expected.begin(), f.expected.end(), guarded.begin() + 1),
        "Parallel feature results or output guards differ");
  Check(profile.peak_participants == participants && profile.parallel_calls == (participants > 1) &&
            profile.slice_jobs == f.slices.size() && profile.prediction_seconds > 0 &&
            budget.snapshot().peak_backing_bytes <= plan.execution.peak_bytes &&
            !budget.snapshot().total.backing_count,
        "Batch profiling or admitted scratch differs");
  Empty(*domain);
}
void Invalid(Fixture &f) {
  const auto original = f.slices;
  std::vector<WeightedPredictorFeature> output(f.expected.size(), kCanary);
  for (size_t fault = 0; fault < 5; ++fault) {
    f.slices = original;
    if (fault == 0)
      ++f.slices.back().feature_offset;
    if (fault == 1)
      --f.slices.back().feature_offset;
    if (fault == 2)
      f.slices.back().input.stride = 0;
    if (fault == 3)
      f.slices.pop_back();
    FeaturePreparationProfile profile;
    ArmNextManagedHostAllocationFailureForTest();
    const auto s = PrepareWeightedPredictorFeatureBatch(f.slices, f.parameters, output,
                                                        fault == 4 ? 0 : 4, &profile);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(s.code() == StatusCode::kInvalidArgument && pending &&
              profile == FeaturePreparationProfile{} &&
              std::all_of(output.begin(), output.end(), [](auto p) { return p == kCanary; }),
          "Malformed late job was not rejected before allocation/stores");
  }
  f.slices = original;
  WeightedFeatureBatchStoragePlan plan, unchanged;
  Ok(ComputeWeightedFeatureBatchStoragePlan(12, 385, 4, &plan));
  unchanged = plan;
  ArmNextManagedHostAllocationFailureForTest();
  Check(!ComputeWeightedFeatureBatchStoragePlan(SIZE_MAX, 385, 4, &plan).ok() &&
            !ComputeWeightedFeatureBatchStoragePlan(12, SIZE_MAX, 4, &plan).ok() &&
            !ComputeWeightedFeatureBatchStoragePlan(12, 385, 257, &plan).ok() &&
            plan == unchanged && ManagedHostAllocationFailurePendingForTest(),
        "Batch plan overflow allocated or mutated output");
  DisarmManagedHostAllocationFailureForTest();
  FeaturePreparationProfile serial;
  Ok(PrepareWeightedPredictorFeatureBatch(f.slices, f.parameters, output, 4, &serial));
  Check(output == f.expected && serial.peak_participants == 1, "Unadmitted caller spawned workers");
}
void Failures(Fixture &f) {
  WeightedFeatureBatchStoragePlan plan;
  Ok(ComputeWeightedFeatureBatchStoragePlan(f.slices.size(), f.maximum_width, 4, &plan));
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({0, 4}, &domain));
  // One predictor-record backing, five row backings per worker, then indexed
  // statuses and thread records. All 23 backings precede worker launch.
  for (size_t fail = 0; fail <= 23; ++fail) {
    ResourceBudget budget(plan.execution.peak_bytes);
    ResourceReservation reservation;
    Ok(budget.Reserve(plan.execution.peak_bytes, &reservation));
    std::vector<WeightedPredictorFeature> output(f.expected.size(), kCanary);
    Status status;
    bool pending;
    {
      CpuExecutionScope cpu;
      Ok(cpu.Start(domain, 4));
      EncodeScope encode(4);
      ResourceContextScope resources({&reservation, ResourceClass::kPreparation});
      ArmManagedHostAllocationFailureAfterForTest(fail);
      status = PrepareWeightedPredictorFeatureBatch(f.slices, f.parameters, output, 4);
      pending = ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
    }
    Check(!budget.snapshot().total.backing_count &&
              budget.snapshot().peak_backing_bytes <= plan.execution.peak_bytes,
          "Failed batch leaked memory or exceeded its bound");
    if (fail < 23)
      Check(status.code() == StatusCode::kOutOfMemory && !pending &&
                std::all_of(output.begin(), output.end(), [](auto p) { return p == kCanary; }),
            "Scratch/scheduler failure started work or lost its error");
    else
      Check(status.ok() && pending && output == f.expected,
            "Allocation sweep missed a backing or changed results");
    Empty(*domain);
  }
  for (auto kind : {WorkerLaunchFailureKind::kSystemError, WorkerLaunchFailureKind::kBadAlloc})
    for (size_t before : {size_t{0}, size_t{1}, size_t{2}}) {
      ResourceBudget budget(plan.execution.peak_bytes);
      ResourceReservation reservation;
      Ok(budget.Reserve(plan.execution.peak_bytes, &reservation));
      std::vector<WeightedPredictorFeature> output(f.expected.size(), kCanary);
      WorkerLaunchFaultForTesting fault{WorkerLaunchSite::kModularWeightedFeatures, before, kind};
      Status status;
      {
        CpuExecutionScope cpu;
        Ok(cpu.Start(domain, 4));
        EncodeScope encode(4);
        ResourceContextScope resources({&reservation, ResourceClass::kPreparation});
        WorkerLaunchFaultScopeForTesting inject(&fault);
        status = PrepareWeightedPredictorFeatureBatch(f.slices, f.parameters, output, 4);
      }
      Check(fault.triggered && fault.launched_in_group == before &&
                status.code() == (kind == WorkerLaunchFailureKind::kBadAlloc
                                      ? StatusCode::kOutOfMemory
                                      : StatusCode::kInternal) &&
                !budget.snapshot().total.backing_count,
            "Partial worker launch was retried, lost its failure or leaked storage");
      Empty(*domain);
    }
}
} // namespace
int main() try {
  Fixture f;
  for (size_t participants : {size_t{1}, size_t{2}, size_t{4}})
    Compare(f, participants, 4);
  Compare(f, 4, 2);
  Compare(f, 4, 1);
  Compare(f, 4, 4, true);
  Invalid(f);
  Failures(f);
  std::cout << "Weighted feature batches match the serial oracle at 1/2/4 participants; budgets "
               "and failures pass\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
