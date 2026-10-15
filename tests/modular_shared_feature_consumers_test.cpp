// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/histogram_sample_preparation.h"
#include "codestream/modular/prepared_streams.h"
#include "codestream/modular/tokenization.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
bool Same(const HistogramTrainingSamples &a, const HistogramTrainingSamples &b) {
  return a.total_samples == b.total_samples && a.predictor_count == b.predictor_count &&
         a.property_count == b.property_count &&
         a.static_property_count == b.static_property_count && a.predictors == b.predictors &&
         a.properties == b.properties && a.weights == b.weights && a.residuals == b.residuals &&
         a.static_values == b.static_values && a.values == b.values && a.thresholds == b.thresholds;
}
bool Same(const HistogramPreparationStats &a, const HistogramPreparationStats &b) {
  return a.scanned_pixels == b.scanned_pixels && a.population == b.population &&
         a.selected_samples == b.selected_samples &&
         a.quantization_samples == b.quantization_samples && a.channels == b.channels;
}
struct Fixture {
  std::vector<int32_t> pixels;
  std::array<ModularChannelView, 2> channels;
  std::array<std::vector<WeightedPredictorFeature>, 2> maps;
  std::array<WeightedFeatureView, 2> features;
  WeightedPredictorParameters parameters;
  Fixture(Extent2D extent, WeightedPredictorParameters p) : parameters(p) {
    const size_t stride = extent.width + 7, plane = (extent.height + 2) * stride;
    pixels.resize(2 * plane, INT32_MIN);
    for (size_t c = 0; c < 2; ++c) {
      const size_t offset = c * plane + stride + 3;
      channels[c] = {std::span(pixels).subspan(offset), {extent}, stride};
      for (size_t y = 0; y < extent.height; ++y)
        for (size_t x = 0; x < extent.width; ++x)
          pixels[offset + y * stride + x] =
              static_cast<int32_t>((x * 19397 + y * 47713 + x * y * 997 + c * 433) % 131071) -
              65535;
      maps[c].resize(extent.width * extent.height);
      Ok(PrepareWeightedPredictorFeatures(channels[c], parameters, maps[c]));
      features[c] = {channels[c], parameters, maps[c]};
    }
  }
};

size_t PreparationParity() {
  size_t cases = 0;
  const std::array<WeightedPredictorParameters, 4> parameters{
      WeightedPredictorParameters{},
      {{8, 8, 4, 0, 3, 23, 2}, {13, 12, 12, 12}},
      {{7, 15, 11, 5, 3, 2, 1}, {15, 0, 7, 3}},
      {{0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0}}};
  for (const auto extent : {Extent2D{1, 17}, {31, 1}, {33, 41}, {257, 5}})
    for (const auto &p : parameters)
      for (bool all : {false, true}) {
        Fixture f(extent, p);
        HistogramSamplingOptions options;
        options.first_group = 7;
        options.maximum_property_values = 8;
        if (all) {
          options.predictor_count = 14;
          for (size_t i = 0; i < 14; ++i)
            options.predictors[i] = static_cast<Predictor>(i);
        }
        std::array<HistogramTrainingImage, 3> images{{{}, {extent, f.channels, 0, p}, {}}};
        PreparedHistogramSamples scalar;
        Ok(PrepareHistogramSamples(images, &scalar, options));
        HostStorageBound scalar_bound;
        Ok(ComputeHistogramSamplePreparationStorageBound(images, options, &scalar_bound));
        const auto expected = scalar.samples();
        const auto stats = scalar.stats();
        const auto threshold = scalar.tree_options().split_threshold;
        DecisionTree expected_tree;
        Ok(LearnPreparedHistogramTree(scalar, &expected_tree, 16));
        for (size_t covered : {size_t{1}, size_t{2}}) {
          auto features = f.features;
          if (covered == 1)
            features[1] = {};
          images[1].weighted_features = features;
          HostStorageBound bound;
          Ok(ComputeHistogramSamplePreparationStorageBound(images, options, &bound));
          Check(covered == 1 ? bound == scalar_bound : bound.peak_bytes < scalar_bound.peak_bytes,
                "Cached preparation did not account for remaining WP scratch");
          ResourceBudget budget(bound.peak_bytes);
          ResourceReservation reservation;
          Ok(budget.Reserve(bound.peak_bytes, &reservation));
          PreparedHistogramSamples cached;
          {
            ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
            Ok(PrepareHistogramSamples(images, &cached, options));
            Check(budget.snapshot().peak_backing_bytes <= bound.peak_bytes,
                  "Cached preparation exceeded its bound");
          }
          Check(Same(expected, cached.samples()) && Same(stats, cached.stats()) &&
                    threshold == cached.tree_options().split_threshold,
                "Shared features changed native training columns, counters or threshold");
          DecisionTree actual_tree;
          Ok(LearnPreparedHistogramTree(cached, &actual_tree, 16));
          Check(actual_tree == expected_tree, "Shared features changed the learned tree");
          TreeLayout layout;
          Ok(ValidateTree(actual_tree, &layout));
          for (size_t c = 0; c < f.channels.size(); ++c) {
            std::vector<EntropyToken> a(f.maps[c].size()), b(a.size());
            Ok(TokenizeValidatedModularChannel(f.channels[c], c, 29, actual_tree, layout, p, a));
            ArmNextManagedHostAllocationFailureForTest();
            const auto status = TokenizeValidatedModularChannel(f.channels[c], c, 29, actual_tree,
                                                                layout, p, b, f.maps[c]);
            const bool pending = ManagedHostAllocationFailurePendingForTest();
            DisarmManagedHostAllocationFailureForTest();
            Ok(status);
            Check(a == b && pending, "Cached channel tokenization changed tokens or allocated WP");
          }
          ++cases;
        }
      }
  return cases;
}

void MetadataAndExclusion() {
  Fixture f({17, 3}, {});
  // A zero-width palette channel keeps its ordinal. The final channel is
  // excluded by the learner's maximum size, with no borrowed pixel backing.
  std::array<ModularChannelView, 4> channels{
      f.channels[0], ModularChannelView{{}, {{0, 1}, 0, 0, ChannelRole::kMetadata}, 0},
      f.channels[1], ModularChannelView{{}, {{1024, 1024}}, 1024}};
  channels[0].descriptor.role = ChannelRole::kMetadata;
  auto features = std::array<WeightedFeatureView, 4>{
      WeightedFeatureView{channels[0], {}, f.maps[0]}, {}, f.features[1], {}};
  std::array<HistogramTrainingImage, 1> images{{{{1024, 1024}, channels, 2}}};
  HistogramSamplingOptions options;
  options.maximum_channel_size = 256;
  PreparedHistogramSamples scalar, cached;
  Ok(PrepareHistogramSamples(images, &scalar, options));
  images[0].weighted_features = features;
  Ok(PrepareHistogramSamples(images, &cached, options));
  Check(Same(scalar.samples(), cached.samples()) && Same(scalar.stats(), cached.stats()),
        "Metadata, empty or excluded channel semantics changed");
}

void ValidationAndPublication() {
  Fixture f({33, 41}, {});
  std::array<HistogramTrainingImage, 1> images{{{{33, 41}, f.channels, 0, {}, f.features}}};
  PreparedHistogramSamples output;
  Ok(PrepareHistogramSamples(images, &output));
  const auto saved = output.samples();
  const auto *saved_pointer = output.samples().weights.data();
  for (size_t fault = 0; fault < 7; ++fault) {
    auto invalid = f.features;
    switch (fault) {
    case 0:
      invalid[0].values = invalid[0].values.first(1);
      break;
    case 1:
      invalid[0].parameters.coefficients[0] ^= 1;
      break;
    case 2:
      invalid[0].source = f.channels[1];
      break;
    case 3:
      ++invalid[0].source.stride;
      break;
    case 4:
      ++invalid[0].source.descriptor.extent.width;
      break;
    case 5:
      invalid[0].source.backing = invalid[0].source.backing.first(1);
      break;
    default:
      break;
    }
    images[0].weighted_features = std::span(invalid).first(fault == 6 ? 1 : 2);
    HostStorageBound bound;
    bound.peak_bytes = 17;
    const auto saved_bound = bound;
    ArmNextManagedHostAllocationFailureForTest();
    const auto planned = ComputeHistogramSamplePreparationStorageBound(images, {}, &bound);
    const auto status = PrepareHistogramSamples(images, &output);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(planned.code() == StatusCode::kInvalidArgument && status.code() == planned.code() &&
              pending && bound == saved_bound && Same(saved, output.samples()) &&
              saved_pointer == output.samples().weights.data(),
          "Invalid feature borrow allocated or changed published preparation");
  }
  images[0].weighted_features = f.features;
  ArmNextManagedHostAllocationFailureForTest();
  const auto status = PrepareHistogramSamples(images, &output);
  const bool pending = ManagedHostAllocationFailurePendingForTest();
  DisarmManagedHostAllocationFailureForTest();
  Check(status.code() == StatusCode::kOutOfMemory && !pending && Same(saved, output.samples()) &&
            saved_pointer == output.samples().weights.data(),
        "Failed cached preparation changed its old output");
}

void StreamParity() {
  std::array<int32_t, 9> pixels{5, 1, 7, 8, 2, 6, 3, 9, 4};
  std::array<ModularChannelView, 1> channels{{{pixels, {{3, 3}}, 3}}};
  const WeightedPredictorParameters parameters{{8, 8, 4, 0, 3, 23, 2}, {13, 12, 12, 12}};
  std::array<WeightedPredictorFeature, 9> map;
  Ok(PrepareWeightedPredictorFeatures(channels[0], parameters, map));
  std::array<WeightedFeatureView, 1> features{{{channels[0], parameters, map}}};
  DecisionTree tree;
  tree.size = 3;
  tree.nodes[0] = {.property = 15, .split = 0, .left = 1, .right = 2};
  tree.nodes[1].predictor = Predictor::kWeighted;
  tree.nodes[2].predictor = Predictor::kGradient;
  for (bool local : {false, true})
    for (const auto policy :
         {PreparedModularEntropyPolicy::kPrefix, PreparedModularEntropyPolicy::kAnsFromPrefix,
          PreparedModularEntropyPolicy::kDirectRateAns}) {
      std::array<PreparedModularStreamView, 3> streams{};
      streams[0].channels = channels;
      streams[0].weighted = parameters;
      streams[0].local_tree = local ? &tree : nullptr;
      PreparedModularStreamsView input{
          {3, 3}, PackedModularFormat::kGray8, local ? nullptr : &tree, streams};
      const PreparedModularEntropyOptions options{policy, AnsSelectionMode::kEstimated};
      codestream_internal::CodestreamBuffer scalar, cached;
      Ok(EncodePreparedModularStreams(input, options, &scalar));
      streams[0].weighted_features = features;
      ModularWorkflowStoragePlan plan;
      Ok(ComputePreparedModularStreamsStoragePlan(input, options, &plan));
      ResourceBudget budget(plan.working.peak_bytes);
      ResourceReservation reservation;
      Ok(budget.Reserve(plan.working.peak_bytes, &reservation));
      {
        ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
        Ok(EncodePreparedModularStreams(input, options, &cached));
      }
      Check(std::ranges::equal(scalar.view(), cached.view()) &&
                budget.snapshot().peak_backing_bytes <= plan.working.peak_bytes,
            "Shared features changed complete bytes or exceeded stream bound");
      PreparedModularStreamTokens completed, borrowed;
      Ok(TokenizePreparedModularStreams(input, &completed));
      streams[0].completed_tokens =
          PreparedModularStreamTokenView{completed.streams[0], channels, &tree, parameters, 0};
      ModularWorkflowStoragePlan borrowed_plan;
      Ok(ComputePreparedModularStreamsStoragePlan(input, options, &borrowed_plan));
      Ok(TokenizePreparedModularStreams(input, &borrowed));
      Check(borrowed.tokens.empty() &&
                borrowed.streams[0].tokens.data() == completed.tokens.data() &&
                borrowed_plan.preparation.peak_bytes + pixels.size() * sizeof(EntropyToken) ==
                    plan.preparation.peak_bytes,
            "Completed tokens were copied or charged twice");
      Ok(EncodePreparedModularStreams(input, options, &cached));
      Check(std::ranges::equal(scalar.view(), cached.view()),
            "Completed token continuation changed bytes");
      const auto ready = *streams[0].completed_tokens;
      for (size_t fault = 0; fault < 5; ++fault) {
        auto bad = ready;
        DecisionTree other_tree;
        auto other_channels = channels;
        if (fault == 0)
          bad.tokens.tokens = bad.tokens.tokens.first(1);
        if (fault == 1)
          bad.tree = &other_tree;
        if (fault == 2)
          bad.weighted.weights[0] ^= 1;
        if (fault == 3)
          ++bad.stream_id;
        if (fault == 4) {
          ++other_channels[0].stride;
          bad.channels = other_channels;
        }
        streams[0].completed_tokens = bad;
        const auto prior = borrowed_plan;
        ArmNextManagedHostAllocationFailureForTest();
        const auto planned =
            ComputePreparedModularStreamsStoragePlan(input, options, &borrowed_plan);
        const auto encoded = EncodePreparedModularStreams(input, options, &cached);
        const bool pending = ManagedHostAllocationFailurePendingForTest();
        DisarmManagedHostAllocationFailureForTest();
        Check(planned.code() == StatusCode::kInvalidArgument && encoded.code() == planned.code() &&
                  pending && borrowed_plan == prior &&
                  std::ranges::equal(scalar.view(), cached.view()),
              "Invalid completed token borrow allocated or published");
      }
      streams[0].completed_tokens.reset();
      features[0].parameters.weights[0] ^= 1;
      const auto saved_plan = plan;
      ArmNextManagedHostAllocationFailureForTest();
      const auto planned = ComputePreparedModularStreamsStoragePlan(input, options, &plan);
      const auto encoded = EncodePreparedModularStreams(input, options, &cached);
      const bool pending = ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      Check(planned.code() == StatusCode::kInvalidArgument && encoded.code() == planned.code() &&
                pending && plan == saved_plan && std::ranges::equal(scalar.view(), cached.view()),
            "Mismatched stream features allocated or changed bytes/plan");
      features[0].parameters = parameters;
    }
}
} // namespace
int main() try {
  const auto cases = PreparationParity();
  MetadataAndExclusion();
  ValidationAndPublication();
  StreamParity();
  std::cout << cases
            << " exact sample/tree cases; token, metadata, byte, admission and "
               "publication checks passed\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
