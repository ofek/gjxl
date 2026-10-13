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
} // namespace
int main() try {
  Contract();
  std::cout << "Modular transform-cost allocation and validation contracts passed\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
