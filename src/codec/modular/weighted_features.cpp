// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/weighted_features.h"
#include "codec/modular/scalar_prediction.h"
#include "core/parallel_work_internal.h"
#include <stdexcept>
#include <type_traits>

namespace gjxl::modular_internal {
using namespace resource_budget_internal;
static_assert(std::is_trivially_copyable_v<WeightedPredictorFeature> &&
              std::is_standard_layout_v<WeightedPredictorFeature> &&
              sizeof(WeightedPredictorFeature) == 16 &&
              offsetof(WeightedPredictorFeature, prediction) == 0 &&
              offsetof(WeightedPredictorFeature, property15) == 8);
namespace {
using PredictorState = WeightedPredictor<ResourceClass::kPreparation>;
template <class T> using PreparationStorage = ManagedVector<T, ResourceClass::kPreparation>;
Status ProduceSlice(ModularChannelView slice, PredictorState &predictor,
                    std::span<WeightedPredictorFeature> output) {
  const auto [width, height] = slice.descriptor.extent;
  for (size_t y = 0; y < height; ++y)
    for (size_t x = 0; x < width; ++x) {
      const auto n = Neighbors(slice, x, y);
      const auto p = predictor.Predict(x, y, n.top, n.left, n.top_right, n.top_left, n.top_top);
      output[y * width + x] = {p.first, p.second};
      if (!predictor.Update(slice.Row(y)[x], x, y))
        return Status::Unsupported("Adaptive Modular weighted state exceeds supported range");
    }
  return Status::Ok();
}
} // namespace

Status ValidateWeightedFeatureView(const WeightedFeatureView &features,
                                   const ModularChannelView &source,
                                   const WeightedPredictorParameters &parameters) {
  if (features.values.empty())
    return Status::Ok();
  size_t count;
  if (!parameters.valid() || features.parameters != parameters ||
      features.source.backing.data() != source.backing.data() ||
      features.source.descriptor != source.descriptor ||
      features.source.stride != source.stride ||
      !source.descriptor.extent.try_area(&count) ||
      count > size_t(PTRDIFF_MAX) / sizeof(WeightedPredictorFeature) ||
      features.values.size() != count || !features.values.data())
    return Status::InvalidArgument("Mismatched completed weighted feature view");
  if (auto s = source.Validate(); !s.ok())
    return s;
  return features.source.Validate();
}

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
  const size_t width = slice.descriptor.extent.width;
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
  return ProduceSlice(slice, predictor, output);
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Weighted predictor scratch allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Weighted predictor scratch storage overflow");
}

Status ComputeWeightedFeatureBatchStoragePlan(size_t slices, size_t maximum_width,
                                              size_t maximum_participants,
                                              WeightedFeatureBatchStoragePlan *out) {
  if (!out || !slices || !maximum_participants || maximum_participants > 256)
    return Status::InvalidArgument("Invalid weighted feature batch storage request");
  const size_t participants = std::min(slices, maximum_participants);
  WeightedFeatureBatchStoragePlan plan;
  if (auto s = ComputeWeightedPredictorScratchBound(maximum_width, participants, &plan.execution);
      !s.ok())
    return s;
  if (!plan.jobs.AddVector<WeightedFeatureSlice>(slices, VectorCapacityPolicy::kFreshExact) ||
      (participants > 1 &&
       (!plan.execution.AddVector<PredictorState>(participants,
                                                  VectorCapacityPolicy::kFreshExact) ||
        !plan.execution.AddVector<Status>(slices, VectorCapacityPolicy::kFreshExact) ||
        !plan.execution.AddVector<std::thread>(participants - 1,
                                               VectorCapacityPolicy::kFreshExact))))
    return Status::InvalidArgument("Weighted feature batch storage overflow");
  *out = plan;
  return Status::Ok();
}

Status PrepareWeightedPredictorFeatureBatchWithParameters(
    std::span<const WeightedFeatureSlice> slices,
    std::span<const WeightedPredictorParameters> parameters,
    std::span<WeightedPredictorFeature> output, size_t maximum_participants,
    FeaturePreparationProfile *profile) try {
  using namespace thread_budget_internal;
  if ((parameters.size() != 1 && parameters.size() != slices.size()) || parameters.empty() ||
      slices.empty() || !maximum_participants || maximum_participants > 256 || !output.data() ||
      output.size() > size_t(PTRDIFF_MAX) / sizeof(WeightedPredictorFeature))
    return Status::InvalidArgument("Invalid weighted feature batch request");
  for (const auto &p : parameters)
    if (!p.valid())
      return Status::InvalidArgument("Invalid weighted feature batch parameters");
  size_t next = 0, maximum_width = 0;
  for (const auto &slice : slices) {
    if (auto s = slice.input.Validate(); !s.ok())
      return s;
    size_t count;
    if (!slice.input.descriptor.extent.try_area(&count) || slice.feature_offset != next ||
        count > output.size() - next)
      return Status::InvalidArgument("Weighted feature batch has invalid output coverage");
    next += count;
    maximum_width = std::max(maximum_width, slice.input.descriptor.extent.width);
  }
  if (next != output.size())
    return Status::InvalidArgument("Weighted feature batch does not cover output");
  WeightedFeatureBatchStoragePlan plan;
  if (auto s = ComputeWeightedFeatureBatchStoragePlan(slices.size(), maximum_width,
                                                      maximum_participants, &plan);
      !s.ok())
    return s;
  size_t desired = std::min(maximum_participants, slices.size());
  if (CpuThreadCount())
    desired = std::min(desired, CpuThreadCount());
  if (!HasCpuParticipation() || InExplicitParallelScope())
    desired = 1;
  CpuWorkerGroup group(desired);
  const size_t participants = group.participants();
  if (profile) {
    profile->slice_jobs += slices.size();
    profile->parallel_calls += participants > 1;
    profile->peak_participants = std::max(profile->peak_participants, participants);
  }
  const auto run = [&](size_t i, PredictorState &predictor) {
    const auto &slice = slices[i];
    const auto [width, height] = slice.input.descriptor.extent;
    if (!predictor.ResetForWidth(width, parameters[parameters.size() == 1 ? 0 : i]))
      return Status::Internal("Weighted predictor scratch is narrower than its slice");
    return ProduceSlice(slice.input, predictor,
                        output.subspan(slice.feature_offset, width * height));
  };
  FeaturePreparationScope allocation(profile, &FeaturePreparationProfile::scratch_seconds);
  if (participants == 1) {
    PredictorState predictor(maximum_width, parameters.front());
    allocation.Stop();
    const FeaturePreparationScope computation(profile,
                                              &FeaturePreparationProfile::prediction_seconds);
    for (size_t i = 0; i < slices.size(); ++i)
      if (auto s = run(i, predictor); !s.ok())
        return s;
    return Status::Ok();
  }
  PreparationStorage<PredictorState> predictors;
  predictors.reserve(participants);
  for (size_t i = 0; i < participants; ++i)
    predictors.emplace_back(maximum_width, parameters.front());
  allocation.Stop();
  const FeaturePreparationScope computation(profile,
                                            &FeaturePreparationProfile::prediction_seconds);
  constexpr ParallelWorkErrors errors{"Weighted feature worker allocation failed",
                                      "Weighted feature worker failed",
                                      StatusCode::kInvalidArgument,
                                      "Weighted feature worker storage overflow",
                                      "Weighted feature worker launch allocation failed",
                                      LaunchFailureAction::kReturnError,
                                      "Unable to start weighted feature workers"};
  return RunParallelWork<PreparationStorage>(
      slices.size(), group, participants - 1, WorkerLaunchSite::kModularWeightedFeatures, errors,
      [&](size_t i, size_t worker) { return run(i, predictors[worker]); });
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Weighted feature batch allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Weighted feature batch storage overflow");
}
Status PrepareWeightedPredictorFeatureBatch(std::span<const WeightedFeatureSlice> slices,
                                            const WeightedPredictorParameters &parameters,
                                            std::span<WeightedPredictorFeature> output,
                                            size_t maximum_participants,
                                            FeaturePreparationProfile *profile) {
  return PrepareWeightedPredictorFeatureBatchWithParameters(slices, {&parameters, 1}, output,
                                                            maximum_participants, profile);
}
} // namespace gjxl::modular_internal
