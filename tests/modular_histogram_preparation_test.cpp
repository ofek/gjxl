// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/histogram_sample_preparation.h"
#include <iostream>
#include <limits>
#include <numeric>
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
bool Same(const HistogramTrainingSamples &a,
          const HistogramTrainingSamples &b) {
  return a.total_samples == b.total_samples &&
         a.predictor_count == b.predictor_count &&
         a.property_count == b.property_count &&
         a.static_property_count == b.static_property_count &&
         a.predictors == b.predictors && a.properties == b.properties &&
         a.weights == b.weights && a.residuals == b.residuals &&
         a.values == b.values && a.static_values == b.static_values &&
         a.thresholds == b.thresholds;
}
struct Image {
  std::vector<int32_t> pixels;
  std::array<ModularChannelView, 2> channels;
  std::array<HistogramTrainingImage, 1> images;
  Image() : pixels(33 * 41 * 2) {
    for (size_t i = 0; i < pixels.size(); ++i)
      pixels[i] = static_cast<int32_t>((i * 197 + i / 29) % 65536) - 32768;
    for (size_t c = 0; c < channels.size(); ++c)
      channels[c] = {
          std::span(pixels).subspan(c * 33 * 41, 33 * 41), {{33, 41}}, 33};
    images[0] = {{33, 41}, channels, 0};
  }
};
void AdmissionAndPublication() {
  Image image;
  HistogramSamplingOptions options;
  options.maximum_property_values = 8; // WP still needs 15 fixed cuts.
  PreparedHistogramSamples output;
  Ok(PrepareHistogramSamples(image.images, &output, options));
  const auto saved = output.samples();
  const auto *saved_pointer = output.samples().weights.data();
  const auto threshold = output.tree_options().split_threshold;
  auto unchanged = [&] {
    return saved_pointer == output.samples().weights.data() &&
           Same(saved, output.samples()) &&
           threshold == output.tree_options().split_threshold;
  };
  auto rejects = [&](auto modify) {
    auto bad = options;
    modify(bad);
    ArmNextManagedHostAllocationFailureForTest();
    HostStorageBound bound;
    const auto planned = ComputeHistogramSamplePreparationStorageBound(
        image.images, bad, &bound);
    const auto status = PrepareHistogramSamples(image.images, &output, bad);
    Check(!planned.ok() && status.code() == planned.code() && unchanged() &&
              ManagedHostAllocationFailurePendingForTest(),
          "Invalid options allocated or published");
    DisarmManagedHostAllocationFailureForTest();
  };
  rejects([](auto &o) { o.sampling_fraction = 0; });
  rejects([](auto &o) {
    o.sampling_fraction = std::numeric_limits<float>::quiet_NaN();
  });
  rejects([](auto &o) { o.sampling_fraction = 1.1; });
  rejects([](auto &o) {
    o.sampling_fraction = std::numeric_limits<float>::denorm_min();
  });
  rejects([](auto &o) { o.maximum_samples = 1; });
  rejects([](auto &o) { o.maximum_samples = kHistogramMaximumSamples + 1; });
  rejects([](auto &o) { o.maximum_property_values = 257; });
  rejects([](auto &o) { o.maximum_property_values = 1; });
  rejects([](auto &o) { o.predictor_count = 15; });
  rejects([](auto &o) { o.predictors[1] = o.predictors[0]; });
  rejects([](auto &o) { o.predictors[0] = Predictor::kInvalid; });
  rejects([](auto &o) { o.properties[0] = 1; });
  rejects([](auto &o) { o.properties[2] = 16; });
  rejects([](auto &o) { o.first_group = UINT32_MAX; });
  rejects([](auto &o) { o.split_threshold = -1; });
  rejects([](auto &o) { o.fast_decode_multiplier = 0.99; });
  image.channels[0].stride = 1;
  Check(!PrepareHistogramSamples(image.images, &output, options).ok() &&
            unchanged(),
        "Invalid view changed output");
  image.channels[0].stride = 33;
  image.images[0].metadata_channels = 1;
  Check(!PrepareHistogramSamples(image.images, &output, options).ok() &&
            unchanged(),
        "Invalid metadata changed output");
  image.images[0].metadata_channels = 0;
  Check(!PrepareHistogramSamples({}, &output, options).ok() && unchanged(),
        "Empty image list accepted");
  Check(!PrepareHistogramSamples(image.images, nullptr, options).ok(),
        "Null preparation output accepted");
  PreparedHistogramSamples unprepared;
  DecisionTree tree, old_tree = tree;
  Check(!LearnPreparedHistogramTree(unprepared, &tree).ok() && tree == old_tree,
        "Default unprepared samples fitted");

  HostStorageBound bound;
  ArmNextManagedHostAllocationFailureForTest();
  Ok(ComputeHistogramSamplePreparationStorageBound(image.images, options,
                                                   &bound));
  Check(ManagedHostAllocationFailurePendingForTest(),
        "Storage planner allocated");
  DisarmManagedHostAllocationFailureForTest();
  const auto old_bound = bound;
  auto bad_options = options;
  bad_options.property_count = 17;
  Check(!ComputeHistogramSamplePreparationStorageBound(image.images,
                                                       bad_options, &bound)
                .ok() &&
            bound == old_bound,
        "Invalid preparation storage planning changed output");
  size_t failures = 0;
  for (; failures < 100; ++failures) {
    ArmManagedHostAllocationFailureAfterForTest(failures);
    const auto status = PrepareHistogramSamples(image.images, &output, options);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (status.ok()) {
      Check(pending, "Failure injection unexpectedly consumed on success");
      break;
    }
    Check(status.code() == StatusCode::kOutOfMemory && !pending && unchanged(),
          "Allocation failure changed prepared output or escaped");
  }
  Check(failures >= 40 && failures < 100,
        "Allocation failure sweep incomplete");
}
void StorageAndPopulation() {
  Image image;
  for (size_t bins : {2, 8, 32, 256}) {
    HistogramSamplingOptions options;
    options.maximum_property_values = bins;
    HostStorageBound bound;
    Ok(ComputeHistogramSamplePreparationStorageBound(image.images, options,
                                                     &bound));
    ResourceBudget budget(bound.peak_bytes);
    ResourceReservation reservation;
    Ok(budget.Reserve(bound.peak_bytes, &reservation));
    size_t peak;
    {
      ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
      PreparedHistogramSamples prepared;
      Ok(PrepareHistogramSamples(image.images, &prepared, options));
      peak = budget.snapshot().peak_backing_bytes;
      Check(peak <= bound.peak_bytes, "Preparation exceeded its storage bound");
      Check(prepared.stats().scanned_pixels == image.pixels.size() &&
                prepared.stats().population == 2 * image.pixels.size() &&
                prepared.stats().selected_samples ==
                    prepared.samples().total_samples &&
                std::accumulate(prepared.samples().weights.begin(),
                                prepared.samples().weights.end(),
                                size_t{0}) == prepared.samples().total_samples,
            "Preparation lost selected population");
    }
    Check(budget.snapshot().total.backing_count == 0,
          "Preparation leaked managed storage");
    ResourceBudget short_budget(peak - 1);
    ResourceReservation short_reservation;
    Ok(short_budget.Reserve(peak - 1, &short_reservation));
    {
      ResourceContextScope scope(
          {&short_reservation, ResourceClass::kPreparation});
      PreparedHistogramSamples prepared;
      Check(PrepareHistogramSamples(image.images, &prepared, options)
                    .resource_plan_exceeded() &&
                prepared.samples().weights.empty(),
            "Preparation escaped its admitted storage");
    }
    Check(short_budget.snapshot().total.backing_count == 0,
          "Failed preparation leaked managed storage");
  }
  // Tiny planes select every pixel, regardless of the requested fraction.
  std::array<int32_t, 3> pixels{0, -1, 65535};
  std::array<ModularChannelView, 1> channels{{{pixels, {{1, 3}}, 1}}};
  std::array<HistogramTrainingImage, 3> images{{{}, {{1, 3}, channels}, {}}};
  PreparedHistogramSamples prepared;
  HistogramSamplingOptions options;
  options.sampling_fraction = .01f;
  options.first_group = 7;
  Ok(PrepareHistogramSamples(images, &prepared, options));
  Check(prepared.stats().selected_samples == 3 &&
            prepared.stats().population == 4 &&
            prepared.stats().quantization_samples == 0 &&
            prepared.tree_options().static_range[1] ==
                std::array<uint32_t, 2>{7, 10},
        "Width-one quantization, group holes or minimum sampling changed");
}
} // namespace
int main() try {
  AdmissionAndPublication();
  StorageAndPopulation();
  std::cout << "Histogram preparation admission, population, storage and "
               "failure checks passed\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
