// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/weighted_features.h"
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
constexpr WeightedPredictorFeature kCanary{INT64_MIN, INT64_MAX};
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(const Status &status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
std::vector<WeightedPredictorFeature> Reference(ModularChannelView v,
                                                WeightedPredictorParameters parameters) {
  const auto [width, height] = v.descriptor.extent;
  WeightedPredictor<ResourceClass::kPreparation> predictor(width, parameters);
  std::vector<WeightedPredictorFeature> result;
  for (size_t y = 0; y < height; ++y)
    for (size_t x = 0; x < width; ++x) {
      // Deliberately form slice-local neighbors here, independently of the
      // producer's Neighbors adapter and output span indexing.
      const int64_t w = x ? v.Row(y)[x - 1] : y ? v.Row(y - 1)[x] : 0;
      const int64_t n = y ? v.Row(y - 1)[x] : w;
      const int64_t ne = y && x + 1 < width ? v.Row(y - 1)[x + 1] : n;
      const int64_t nw = x && y ? v.Row(y - 1)[x - 1] : w;
      const int64_t nn = y > 1 ? v.Row(y - 2)[x] : n;
      const auto [prediction, property] = predictor.Predict(x, y, n, w, ne, nw, nn);
      result.push_back({prediction, property});
      Check(predictor.Update(v.Row(y)[x], x, y), "Reference exceeded predictor range");
    }
  return result;
}
void Slices() {
  std::array<WeightedPredictorParameters, 4> parameters{};
  parameters[1] = {{7, 15, 11, 5, 3, 2, 1}, {15, 0, 7, 3}};
  parameters[2] = {{0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0}};
  parameters[3] = {{31, 31, 31, 31, 31, 31, 31}, {15, 15, 15, 15}};
  for (const auto extent :
       {Extent2D{1, 1}, {1, 31}, {31, 1}, {3, 5}, {17, 19}, {256, 256}, {385, 3}}) {
    const auto [width, height] = extent;
    const size_t stride = width + 7, origin = stride + 3, count = width * height;
    std::vector<int32_t> backing((height + 2) * stride, INT32_MIN);
    for (size_t y = 0; y < height; ++y)
      for (size_t x = 0; x < width; ++x)
        backing[origin + y * stride + x] =
            static_cast<int32_t>((x * 19397 + y * 47713 + x * y * 997) % 131071) - 65535;
    const auto original = backing;
    ModularChannelView slice{std::span(backing).subspan(origin), {extent}, stride};
    for (const auto &p : parameters) {
      const auto expected = Reference(slice, p);
      std::vector<WeightedPredictorFeature> guarded(count + 4, kCanary);
      const auto output = std::span(guarded).subspan(2, count);
      FeaturePreparationProfile profile;
      Ok(PrepareWeightedPredictorFeatures(slice, p, output, &profile));
      Check(std::equal(output.begin(), output.end(), expected.begin()) && backing == original &&
                guarded[0] == kCanary && guarded[1] == kCanary && guarded[count + 2] == kCanary &&
                guarded[count + 3] == kCanary,
            "Strided slice changed features, input or adjacent output");
      Check(profile.scratch_seconds >= 0 && profile.prediction_seconds >= 0 &&
                profile.allocation_seconds == 0 && profile.copy_seconds == 0 &&
                profile.publication_seconds == 0,
            "Scalar producer attributed work to the enclosing owner");
      const auto first = profile;
      Ok(PrepareWeightedPredictorFeatures(slice, p, output, &profile));
      Check(profile.scratch_seconds >= first.scratch_seconds &&
                profile.prediction_seconds >= first.prediction_seconds,
            "Feature timings did not accumulate");
      const auto before_disabled = profile;
      Ok(PrepareWeightedPredictorFeatures(slice, p, output));
      Check(std::equal(output.begin(), output.end(), expected.begin()) &&
                profile == before_disabled,
            "Repeated/disabled-profile production retained predictor state");
    }
  }
}
void InvalidInputs() {
  std::array<int32_t, 4> values{1, 2, 3, 4};
  const ModularChannelView good{values, {{2, 2}}, 2};
  std::array<WeightedPredictorFeature, 4> output;
  output.fill(kCanary);
  const auto before = output;
  auto rejected = [&](ModularChannelView input, WeightedPredictorParameters p,
                      std::span<WeightedPredictorFeature> destination) {
    FeaturePreparationProfile profile;
    ArmNextManagedHostAllocationFailureForTest();
    const auto status = PrepareWeightedPredictorFeatures(input, p, destination, &profile);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(status.code() == StatusCode::kInvalidArgument && pending && output == before &&
              profile == FeaturePreparationProfile{},
          "Invalid feature request allocated, wrote output or entered timed work");
  };
  auto invalid = good;
  invalid.stride = 1;
  rejected(invalid, {}, output);
  invalid = good;
  invalid.backing = std::span(values).first(3);
  rejected(invalid, {}, output);
  invalid = good;
  invalid.descriptor.extent.height = 0;
  rejected(invalid, {}, output);
  invalid = good;
  invalid.stride = SIZE_MAX;
  rejected(invalid, {}, output);
  auto p = WeightedPredictorParameters{};
  p.coefficients[0] = 32;
  rejected(good, p, output);
  p = {};
  p.weights[3] = 16;
  rejected(good, p, output);
  rejected(good, {}, std::span(output).first(3));
  rejected(good, {}, {});
  invalid = good;
  invalid.descriptor.extent = {1, 1};
  rejected(invalid, {}, output); // Oversized output is also invalid.
}
void StorageAndFailures() {
  HostStorageBound serial, parallel;
  ArmNextManagedHostAllocationFailureForTest();
  Ok(ComputeWeightedPredictorScratchBound(257, 1, &serial));
  Ok(ComputeWeightedPredictorScratchBound(257, 4, &parallel));
  Check(ManagedHostAllocationFailurePendingForTest() &&
            serial.retained_bytes == 5 * 2 * (257 + 2) * sizeof(uint32_t) &&
            serial.peak_bytes == serial.retained_bytes &&
            parallel.peak_bytes == 4 * serial.peak_bytes,
        "Scratch plan allocated or omitted per-participant state");
  for (const auto [width, participants] : {std::pair<size_t, size_t>{0, 1},
                                           {1, 0},
                                           {1, 257},
                                           {SIZE_MAX, 1},
                                           {size_t(PTRDIFF_MAX) / sizeof(uint32_t), 256}}) {
    auto unchanged = serial;
    Check(!ComputeWeightedPredictorScratchBound(width, participants, &unchanged).ok() &&
              unchanged == serial && ManagedHostAllocationFailurePendingForTest(),
          "Invalid/overflowed scratch plan changed its output or allocated");
  }
  Check(!ComputeWeightedPredictorScratchBound(1, 1, nullptr).ok(), "Null plan accepted");
  DisarmManagedHostAllocationFailureForTest();

  std::array<int32_t, 6> values{10, -20, 30, -40, 50, -60};
  const ModularChannelView slice{values, {{3, 2}}, 3};
  HostStorageBound bound;
  Ok(ComputeWeightedPredictorScratchBound(3, 1, &bound));
  for (size_t fail = 0; fail <= 5; ++fail) {
    std::array<WeightedPredictorFeature, 6> output;
    output.fill(kCanary);
    const auto before = output;
    ResourceBudget budget(bound.peak_bytes);
    ResourceReservation reservation;
    Ok(budget.Reserve(bound.peak_bytes, &reservation));
    Status status;
    bool pending;
    {
      ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
      ArmManagedHostAllocationFailureAfterForTest(fail);
      status = PrepareWeightedPredictorFeatures(slice, {}, output);
      pending = ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
    }
    Check(budget.snapshot().peak_backing_bytes <= bound.peak_bytes &&
              budget.snapshot().total.backing_count == 0,
          "Weighted preparation exceeded its plan or leaked scratch");
    if (fail < 5)
      Check(status.code() == StatusCode::kOutOfMemory && !pending && output == before,
            "Scratch allocation failure wrote output or lost its status");
    else {
      Ok(status);
      const auto expected = Reference(slice, {});
      Check(pending && std::equal(output.begin(), output.end(), expected.begin()),
            "Successful production missed a scratch allocation or changed features");
    }
  }
  {
    std::array<WeightedPredictorFeature, 6> output;
    output.fill(kCanary);
    const auto before = output;
    ResourceBudget budget(bound.peak_bytes - 1);
    ResourceReservation reservation;
    Ok(budget.Reserve(bound.peak_bytes - 1, &reservation));
    Status status;
    {
      ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
      status = PrepareWeightedPredictorFeatures(slice, {}, output);
    }
    Check(status.resource_plan_exceeded() && output == before &&
              budget.snapshot().total.backing_count == 0,
          "Under-admitted scratch escaped its reservation or lost its failure reason");
  }
  // A numerical failure is different from preflight failure: the caller must
  // discard this unpublished destination. There is no range-based CPU retry.
  values[1] = INT32_MAX;
  std::array<WeightedPredictorFeature, 6> output;
  output.fill(kCanary);
  Check(PrepareWeightedPredictorFeatures(slice, {}, output).code() == StatusCode::kUnsupported &&
            output[0] != kCanary && output[1] != kCanary && output[2] == kCanary,
        "Range failure was hidden or wrote past the failing pixel");
}
} // namespace
int main() try {
  Slices();
  InvalidInputs();
  StorageAndFailures();
  std::cout << "Weighted feature slices preserve scalar results, boundaries and storage failures\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
