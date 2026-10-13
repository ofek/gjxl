// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/weighted_features.h"
#include "codec/modular/scalar_prediction.h"
#include <stdexcept>
#include <type_traits>

namespace gjxl::modular_internal {
using namespace resource_budget_internal;
static_assert(std::is_trivially_copyable_v<WeightedPredictorFeature> &&
              std::is_standard_layout_v<WeightedPredictorFeature> &&
              sizeof(WeightedPredictorFeature) == 16 &&
              offsetof(WeightedPredictorFeature, prediction) == 0 &&
              offsetof(WeightedPredictorFeature, property15) == 8);

Status ComputeWeightedPredictorScratchBound(size_t maximum_width, size_t participants,
                                            HostStorageBound *out) {
  if (!out || !maximum_width || !participants || participants > 256 ||
      maximum_width > SIZE_MAX / 2 - 2)
    return Status::InvalidArgument("Invalid weighted predictor scratch request");
  const size_t rows = (maximum_width + 2) * 2;
  HostStorageBound bound;
  if (!bound.AddVector<uint32_t>(rows, VectorCapacityPolicy::kFreshExact, 4 * participants) ||
      !bound.AddVector<int32_t>(rows, VectorCapacityPolicy::kFreshExact, participants))
    return Status::InvalidArgument("Weighted predictor scratch overflow");
  *out = bound;
  return Status::Ok();
}

Status PrepareWeightedPredictorFeatures(ModularChannelView slice,
                                        const WeightedPredictorParameters &parameters,
                                        std::span<WeightedPredictorFeature> output,
                                        FeaturePreparationProfile *profile) try {
  if (!parameters.valid())
    return Status::InvalidArgument("Invalid weighted predictor parameters");
  if (auto s = slice.Validate(); !s.ok())
    return s;
  const auto [width, height] = slice.descriptor.extent;
  size_t count;
  if (!slice.descriptor.extent.try_area(&count) ||
      count > size_t(PTRDIFF_MAX) / sizeof(WeightedPredictorFeature) || output.size() != count ||
      !output.data())
    return Status::InvalidArgument("Invalid weighted feature output span");
  HostStorageBound scratch;
  if (auto s = ComputeWeightedPredictorScratchBound(width, 1, &scratch); !s.ok())
    return s;
  FeaturePreparationScope allocation(profile, &FeaturePreparationProfile::scratch_seconds);
  WeightedPredictor<ResourceClass::kPreparation> predictor(width, parameters);
  allocation.Stop();
  const FeaturePreparationScope computation(profile,
                                            &FeaturePreparationProfile::prediction_seconds);
  for (size_t y = 0; y < height; ++y)
    for (size_t x = 0; x < width; ++x) {
      const auto n = Neighbors(slice, x, y);
      const auto p = predictor.Predict(x, y, n.top, n.left, n.top_right, n.top_left, n.top_top);
      output[y * width + x] = {p.first, p.second};
      if (!predictor.Update(slice.Row(y)[x], x, y))
        return Status::Unsupported("Adaptive Modular weighted state exceeds supported range");
    }
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Weighted predictor scratch allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Weighted predictor scratch storage overflow");
}
} // namespace gjxl::modular_internal
