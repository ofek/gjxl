// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/profile.h"
#include "codestream/modular/workflow.h"
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(const Status &s) { Check(s.ok(), "Modular encode failed"); }
void Test() {
  constexpr size_t width = 257, height = 17;
  std::vector<uint8_t> pixels(width * height * 3);
  for (size_t i = 0; i < pixels.size(); ++i)
    pixels[i] = static_cast<uint8_t>((i * 31 + i / 7) % 251);
  const PackedModularImageView input{pixels, {width, height}, width * 3};
  for (auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
    ModularEncodingOptions options;
    options.entropy = mode;
    options.cpu_thread_count = 4;
    std::vector<uint8_t> ordinary, profiled;
    Ok(EncodeModularImage(input, options, &ordinary));
    ModularProfile outer, inner;
    {
      ProfileSession session(&outer);
      {
        ProfileSession disabled(nullptr);
        Ok(EncodeModularImage(input, options, &profiled));
      }
      Check(outer.entropy_work == codestream_internal::EntropyWorkProfile{},
            "Disabled profile wrote entropy counters");
      {
        ProfileSession nested(&inner);
        Ok(EncodeModularImage(input, options, &profiled));
      }
      Check(outer.entropy_work == codestream_internal::EntropyWorkProfile{},
            "Nested profile wrote to the outer sink");
      Ok(EncodeModularImage(input, options, &profiled));
      const auto first = outer.entropy_work.prefix_histogram_build_nanoseconds;
      Ok(EncodeModularImage(input, options, &profiled));
      Check(outer.encoded_candidates == 2 &&
                outer.entropy_work.prefix_histogram_build_nanoseconds > first,
            "Repeated calls did not accumulate entropy counters");
    }
    Check(!active_modular_profile && ordinary == profiled && inner.encoded_candidates == 1,
          "Profiling changed bytes or leaked its sink");
    const auto &p = inner.entropy_work;
    Check(p.prefix_histogram_build_nanoseconds > 0 && p.prefix_value_collection_nanoseconds > 0 &&
              p.prefix_value_aggregation_nanoseconds > 0 &&
              p.prefix_value_aggregation_nanoseconds <= p.prefix_config_search_nanoseconds &&
              p.prefix_config_search_nanoseconds <
                  inner.seconds[static_cast<size_t>(ProfileStage::kModel)] * 1e9,
          "Missing or incorrectly nested prefix counters");
    Check((p.ans_value_collection_nanoseconds > 0) == (mode == EntropyCodingMode::kAns) &&
              (p.ans_histogram_candidate_count > 0) == (mode == EntropyCodingMode::kAns),
          "ANS counters disagree with the selected mode");
    auto accumulated = p;
    codestream_internal::AccumulateEntropyWorkProfile(p, &accumulated);
    Check(accumulated.prefix_value_aggregation_nanoseconds ==
              2 * p.prefix_value_aggregation_nanoseconds,
          "Shared aggregation omitted the nested counter");
  }
}
void Training() {
  constexpr size_t width = 33, height = 17;
  std::vector<uint8_t> pixels(width * height * 3);
  for (size_t i = 0; i < pixels.size(); ++i)
    pixels[i] = static_cast<uint8_t>((i * 31 + i / 7) % 251);
  const PackedModularImageView input{pixels, {width, height}, width * 3};
  ModularEncodingOptions options;
  options.entropy = EntropyCodingMode::kAns;
  options.search = true;
  options.learning = {1024, 32};
  options.tree_checkpoints = true;
  std::vector<uint8_t> ordinary, profiled;
  ModularProfile outer, inner;
  {
    ProfileSession session(&outer);
    {
      ProfileSession disabled(nullptr);
      Ok(EncodeModularImage(input, options, &ordinary));
    }
    Check(outer.training_work == TrainingWorkProfile{}, "Disabled training profile wrote counters");
    {
      ProfileSession nested(&inner);
      Ok(EncodeModularImage(input, options, &profiled));
    }
    Check(outer.training_work == TrainingWorkProfile{},
          "Nested training profile changed outer sink");
    Ok(EncodeModularImage(input, options, &profiled));
    const auto first = outer.training_work;
    Ok(EncodeModularImage(input, options, &profiled));
    Check(outer.training_work.learning_calls == first.learning_calls * 2 &&
              outer.training_work.sorted_samples == first.sorted_samples * 2,
          "Training work did not accumulate across calls");
  }
  Check(!active_modular_profile && ordinary == profiled && inner.resolved == outer.resolved,
        "Training profiling changed the selected policy or complete bytes");
  const auto &work = inner.training_work;
  const double nested = std::accumulate(work.seconds.begin(), work.seconds.end(), 0.0);
  Check(work.learning_calls > 0 && work.population >= work.samples && work.samples > 0 &&
            work.split_evaluations > 0 && work.sorted_properties % kPropertyCount == 0 &&
            work.sorted_samples >= work.sorted_properties && nested > 0 &&
            nested <= inner.seconds[static_cast<size_t>(ProfileStage::kTraining)] + 1e-6,
        "Training counters missing or intervals overlap their training parent");
  for (auto stage : {TrainingProfileStage::kSampleGathering, TrainingProfileStage::kSimpleLearning,
                     TrainingProfileStage::kEntropyPreparation,
                     TrainingProfileStage::kSplitHistograms, TrainingProfileStage::kPropertySorting,
                     TrainingProfileStage::kSplitScanning, TrainingProfileStage::kTreeControl})
    Check(work.seconds[static_cast<size_t>(stage)] > 0, "Missing detailed training stage");
  Check(work.seconds[static_cast<size_t>(TrainingProfileStage::kQuantizedLearning)] == 0,
        "Original learner reported quantized training");
}
} // namespace
int main() try {
  Test();
  Training();
  std::cout << "Modular entropy profiles preserve bytes, nesting and accumulation\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
