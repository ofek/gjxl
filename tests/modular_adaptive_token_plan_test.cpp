// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/ops/modular_adaptive_tokens.h"
#include "codec/modular/weighted_features.h"
#include <thread>
#include <iostream>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::modular_gpu_internal;
using namespace gjxl::resource_budget_internal;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(const Status &s) { Check(s.ok(), "Adaptive token planning failed"); }
void Trees() {
  DecisionTree tree;
  tree.nodes.resize(511);
  tree.size = 511;
  for (size_t i = 0; i < 255; ++i)
    tree.nodes[2 * i] = {.property = 15,
                         .split = static_cast<int32_t>(i),
                         .left = static_cast<TreeIndex>(2 * i + 2),
                         .right = static_cast<TreeIndex>(2 * i + 1)};
  for (size_t i = 1; i < tree.size; i += 2)
    tree.nodes[i].predictor = Predictor::kWeighted;
  AdaptiveModularTreeEligibility result;
  ArmNextManagedHostAllocationFailureForTest();
  Ok(ResolveAdaptiveModularTreeEligibility(tree, &result));
  Check(ManagedHostAllocationFailurePendingForTest() && result.node_count == 511 &&
            result.context_count == 256 && result.weighted,
        "Large weighted tree was copied or narrowed to the old GPU ceiling");
  const auto previous = result;
  tree.nodes[0].left = 0;
  Check(!ResolveAdaptiveModularTreeEligibility(tree, &result).ok() && result == previous &&
            ManagedHostAllocationFailurePendingForTest(),
        "Malformed tree changed eligibility output or allocated");
  DisarmManagedHostAllocationFailureForTest();
}
void Layouts() {
  Check(PreferGpuWeightedFeatures(4194304, 256) &&
            !PreferGpuWeightedFeatures(4194303, 255) &&
            !PreferGpuWeightedFeatures(4194304, 257) &&
            !PreferGpuWeightedFeatures(4194304, 0) &&
            PreferGpuWeightedFeatures(SIZE_MAX, 1),
        "Automatic weighted feature crossover or overflow handling changed");
  AdaptiveModularTokenSlice tile;
  tile.width = tile.height = 256;
  Check(GpuWeightedFeatureSlicesSupported({&tile,1}), "Common GPU feature shape rejected");
  tile.width = 257;
  Check(!GpuWeightedFeatureSlicesSupported({&tile,1}), "Wide GPU feature shape accepted");
  tile.width = 256;
  tile.height = 257;
  Check(!GpuWeightedFeatureSlicesSupported({&tile,1}), "Tall GPU feature shape accepted");
  tile.height = 0;
  Check(!GpuWeightedFeatureSlicesSupported({&tile,1}) && !GpuWeightedFeatureSlicesSupported({}),
        "Empty GPU feature shape accepted");
  const std::array<ChannelDescriptor, 3> channels{{{{3, 5}}, {{3, 5}}, {{3, 5}}}};
  AdaptiveModularTokenStoragePlan weighted, plain;
  ArmNextManagedHostAllocationFailureForTest();
  Ok(ComputeAdaptiveModularTokenStoragePlan(channels, 0, 3, 511, true, 1, &weighted));
  Ok(ComputeAdaptiveModularTokenStoragePlan(channels, 0, 3, 511, false, 1, &plain));
  Check(ManagedHostAllocationFailurePendingForTest(), "Adaptive storage planner allocated");
  DisarmManagedHostAllocationFailureForTest();
  Check(weighted.token_count == 45 && weighted.plane_bytes == 572 &&
            weighted.weighted_bytes == 720 && weighted.output_bytes == 346 &&
            weighted.nodes_offset == 256 && weighted.status_offset == 16640 &&
            weighted.dispatch_bytes == 16644,
        "Plane alignment, dynamic nodes, or compact output ABI changed");
  Check(plain.weighted_bytes == 0 &&
            plain.features.retained_bytes + 720 == weighted.features.retained_bytes &&
            weighted.bridge_peak_bytes == plain.bridge_peak_bytes + 1440 + 5 * 2 * 5 * 4 +
                                              3 * sizeof(WeightedFeatureSlice) &&
            weighted.retained_bytes == weighted.features.retained_bytes + weighted.output_bytes &&
            weighted.generation_peak_bytes == weighted.retained_bytes + 2 * weighted.dispatch_bytes,
        "Bridge, retained features, or dispatch overlap omitted");
  AdaptiveModularTokenStoragePlan parallel;
  Ok(ComputeAdaptiveModularTokenStoragePlan(channels, 0, 3, 511, true, 4, &parallel));
  Check(parallel.bridge_peak_bytes == weighted.bridge_peak_bytes + 2 * 5 * 2 * 5 * 4 +
              3 * sizeof(WeightedPredictor<ResourceClass::kPreparation>) + 3 * sizeof(Status) +
              2 * sizeof(std::thread) &&
            parallel.features == weighted.features &&
            parallel.generation_peak_bytes == weighted.generation_peak_bytes,
        "Shared scratch planning changed storage beyond the additional CPU participants");
  auto limit = ModularTokenDeviceLimits{45, 3, weighted.dispatch_bytes};
  AdaptiveModularTokenStoragePlan admitted;
  Ok(ComputeAdaptiveModularTokenStoragePlan(channels, 0, 3, 511, true, 1, &admitted, &limit));
  Check(admitted == weighted, "Device admission changed layout");
  --limit.maximum_buffer_bytes;
  Check(ComputeAdaptiveModularTokenStoragePlan(channels, 0, 3, 511, true, 1, &admitted, &limit)
                    .code() == StatusCode::kUnsupported &&
            admitted == weighted,
        "One byte below the required device capacity was accepted");
  // Metadata plus differently sized, shifted residual planes are legal here.
  const std::array<ChannelDescriptor, 3> transformed{
      {{{7, 3}, 0, 0, ChannelRole::kMetadata}, {{129, 257}, 1, 0}, {{128, 257}, 1, 0}}};
  Ok(ComputeAdaptiveModularTokenStoragePlan(transformed, 1, 7, 511, true, 4, &admitted));
  Check(admitted.token_count == 66070 && admitted.plane_bytes > 4 * admitted.token_count &&
            admitted.node_count == 511,
        "Transformed channels were flattened to the original image dimensions");
  const auto previous = admitted;
  ArmNextManagedHostAllocationFailureForTest();
  for (size_t nodes : {size_t{0}, size_t{2}, kMaximumTreeNodes + 2})
    Check(!ComputeAdaptiveModularTokenStoragePlan(channels, 0, 3, nodes, true, 1, &admitted).ok() &&
              admitted == previous,
          "Invalid node count changed output");
  Check(
      !ComputeAdaptiveModularTokenStoragePlan(channels, 0, SIZE_MAX, 31, true, 1, &admitted).ok() &&
          !ComputeAdaptiveModularTokenStoragePlan(channels, 0, 3, 31, true, 0, &admitted).ok() &&
          !ComputeAdaptiveModularTokenStoragePlan(channels, 0, 3, 31, true, 257, &admitted).ok() &&
          !ComputeAdaptiveModularTokenStoragePlan(transformed, 0, 7, 31, true, 1, &admitted).ok() &&
          admitted == previous && ManagedHostAllocationFailurePendingForTest(),
      "Invalid adaptive shape allocated or changed output");
  DisarmManagedHostAllocationFailureForTest();
}
} // namespace
int main() try {
  Trees();
  Layouts();
  std::cout << "Adaptive Modular tree eligibility and storage contracts passed\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
