// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/selection_cost.h"
#include <array>
#include <iostream>
#include <stdexcept>
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
namespace {
void Check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
void Contract() {
  const std::array<int32_t, 4> samples{0, 1, 2, 3};
  const ModularChannelView channel{samples, {{4, 1}}, 4};
  ModularTransformCost cost;
  ArmNextManagedHostAllocationFailureForTest();
  const auto s = EstimateModularTransformCost({&channel, 1}, &cost);
  const bool pending = ManagedHostAllocationFailurePendingForTest();
  DisarmManagedHostAllocationFailureForTest();
  // Left residuals are 0,1,1,1: four symbols, entropy floor 3, no extra bits.
  Check(s.ok() && pending && cost.bits == 3 && cost.extra_bits == 0 && cost.pixels == 4,
        "Transform cost allocated or changed the fixed-gradient objective");
  const auto previous = cost;
  for (size_t error = 0; error < 5; ++error) {
    auto bad = channel;
    if (error == 0)
      bad.stride = 3;
    if (error == 1)
      bad.backing = bad.backing.first(3);
    if (error == 2)
      bad.descriptor.hshift = 31;
    if (error == 3)
      bad.descriptor.extent.height = SIZE_MAX;
    if (error == 4)
      bad.descriptor.role = static_cast<ChannelRole>(99);
    // The valid first channel must not be scanned/published before validation
    // of later channels finishes.
    const std::array channels{channel, bad};
    ArmNextManagedHostAllocationFailureForTest();
    const auto status = EstimateModularTransformCost(channels, &cost);
    const bool unallocated = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(status.code() == StatusCode::kInvalidArgument && unallocated && cost == previous,
          "Invalid view allocated or published a partial cost");
  }
  Check(EstimateModularTransformCost({&channel, 1}, nullptr).code() == StatusCode::kInvalidArgument,
        "Null cost accepted");
  const std::array<ModularChannelView, 2> empty = {ModularChannelView{{}, {{0, 17}}, 0},
                                                   ModularChannelView{{}, {{9, 0}}, 9}};
  Check(EstimateModularTransformCost(empty, &cost).ok() && cost == ModularTransformCost{},
        "Empty planes changed estimator cost");
}
void WeightedContract() {
  const std::array<int32_t, 9> pixels{3, 1, 7, 8, 2, 6, 3, 9, 4};
  const std::array<ModularChannelView, 3> channels{{{pixels, {{3, 3}}, 3},
                                                    {std::span(pixels).first(4), {{1, 4}}, 1},
                                                    {{}, {{10000, 0}}, 10000}}};
  size_t failures = 0;
  for (size_t mode = 0; mode < 5; ++mode) {
    HostStorageBound bound;
    ArmNextManagedHostAllocationFailureForTest();
    const auto planned = ComputeModularWeightedCostStorageBound(channels, mode, &bound);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(planned.ok() && pending && bound.peak_bytes == 200 && bound.retained_bytes == 0,
          "WP planner allocated or changed scratch lifetime/bound");
    ModularWeightedCost expected;
    Ok(EstimateModularWeightedCost(channels, mode, &expected));
    Check(expected.pixels == 13 && !expected.residual_overflow, "WP population/rejection changed");
    ResourceBudget budget(bound.peak_bytes);
    ResourceReservation reservation;
    Ok(budget.Reserve(bound.peak_bytes, &reservation));
    const ModularWeightedCost sentinel{.bits = 17, .pixels = 99};
    for (size_t fail = 0; fail <= 10; ++fail) {
      ModularWeightedCost cost = sentinel;
      ResourceContextScope scope({&reservation, ResourceClass::kInput});
      ArmManagedHostClassAllocationFailureAfterForTest(ResourceClass::kPreparation, fail);
      const auto status = EstimateModularWeightedCost(channels, mode, &cost);
      DisarmManagedHostAllocationFailureForTest();
      if (fail == 10)
        Check(status.ok() && cost == expected, "WP allocation recovery changed cost");
      else {
        Check(status.code() == StatusCode::kOutOfMemory && cost == sentinel,
              "WP failure changed output or allocation ownership");
        ++failures;
      }
      Check(budget.snapshot().total.backing_count == 0 &&
                budget.snapshot().total.pending_count == 0 &&
                budget.snapshot().peak_backing_bytes <= bound.peak_bytes,
            "WP history exceeded its bound or leaked backing");
    }
    ResourceBudget tiny(1);
    ResourceReservation tiny_reservation;
    Ok(tiny.Reserve(1, &tiny_reservation));
    ModularWeightedCost cost = sentinel;
    {
      ResourceContextScope scope({&tiny_reservation, ResourceClass::kPreparation});
      Check(EstimateModularWeightedCost(channels, mode, &cost).code() == StatusCode::kOutOfMemory &&
                cost == sentinel && tiny.snapshot().total.backing_count == 0,
            "Underplanned WP estimate changed output or leaked backing");
    }
    for (size_t bad = 0; bad < 3; ++bad) {
      auto views = channels;
      if (bad == 0)
        views[1].stride = 0;
      const size_t selected = bad == 1 ? 5 : bad == 2 ? SIZE_MAX : mode;
      auto output_bound = bound;
      ArmNextManagedHostAllocationFailureForTest();
      const auto plan_status =
          ComputeModularWeightedCostStorageBound(views, selected, &output_bound);
      const auto status = EstimateModularWeightedCost(views, selected, &cost);
      const bool unallocated = ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      Check(plan_status.code() == StatusCode::kInvalidArgument &&
                status.code() == plan_status.code() && unallocated && output_bound == bound &&
                cost == sentinel,
            "Invalid WP request allocated or changed outputs");
    }
    const std::array<int32_t, 1> high{INT32_MAX};
    const ModularChannelView view{high, {{1, 1}}, 1};
    Ok(EstimateModularWeightedCost({&view, 1}, mode, &cost));
    Check(cost.bits == 29 && !cost.residual_overflow,
          "WP history overflow incorrectly rejected a finite residual");
    const std::array<int32_t, 2> opposite{INT32_MAX, INT32_MIN};
    const ModularChannelView rejected{opposite, {{2, 1}}, 2};
    Ok(EstimateModularWeightedCost({&rejected, 1}, mode, &cost));
    Check(cost.bits == std::numeric_limits<float>::max() && cost.residual_overflow,
          "WP residual overflow did not reject the candidate");
    ArmNextManagedHostAllocationFailureForTest();
    const auto empty = EstimateModularWeightedCost(std::span(channels).last(1), mode, &cost);
    const bool empty_unallocated = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(empty.ok() && empty_unallocated && cost == ModularWeightedCost{},
          "Empty WP input allocated or changed cost");
  }
  Check(ComputeModularWeightedCostStorageBound(channels, 0, nullptr).code() ==
                StatusCode::kInvalidArgument &&
            EstimateModularWeightedCost(channels, 0, nullptr).code() ==
                StatusCode::kInvalidArgument,
        "Null WP output accepted");
  std::cout << "Weighted-cost contracts: " << failures << " allocation failures qualified\n";
}
} // namespace
int main() try {
  Contract();
  WeightedContract();
  std::cout << "Modular transform-cost allocation and validation contracts passed\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
