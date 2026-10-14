// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/entropy_population.h"
#include "codestream/profile_internal.h"
#include "codestream/ans_internal.h"
#include "gpu/ops/modular_populations.h"
#include "codestream/modular/adaptive_gpu_workflow.h"
#include <iostream>
#include <stdexcept>
namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::resource_budget_internal;
using namespace gjxl::modular_gpu_internal;
void Check(bool b, const char *m) { if (!b) throw std::runtime_error(m); }
void Ok(const Status &s) { if (!s.ok()) throw std::runtime_error(std::string(s.message())); }
void Models() {
  for (size_t contexts : {1, 3, 17, 65, 256}) {
    std::vector<EntropyToken> tokens(8193);
    std::vector<uint64_t> counts(contexts * 4096);
    for (size_t i = 0; i < tokens.size(); ++i) {
      tokens[i] = {uint32_t(i % contexts), uint32_t((i * 7919 + i / 31) % 4096)};
      ++counts[tokens[i].context * 4096 + tokens[i].value];
    }
    const EntropyContextPopulations populations{contexts, 4096, counts};
    const auto span = std::span<const EntropyToken>(tokens);
    const std::array<EntropyTokenStreamView, 4> streams{
        EntropyTokenStreamView::Interleaved(span.first(0)),
        EntropyTokenStreamView::Interleaved(span.first(4097)),
        EntropyTokenStreamView::Interleaved(span.subspan(4097)),
        EntropyTokenStreamView::Interleaved(span.last(0))};
    for (bool mapped : {false, true}) {
      std::vector<uint8_t> map(contexts);
      for (size_t i = 0; i < contexts; ++i) map[i] = uint8_t(i % std::min(contexts, size_t{7}));
      EntropyCodeOptions options{.context_count = uint32_t(contexts)};
      if (mapped) { options.initial_context_map = map; options.initial_histogram_count = std::min(contexts, size_t{7}); }
      EntropyCode expected, actual;
      EntropyCodeCost expected_cost, actual_cost;
      PreparedEntropyClusters expected_values, actual_values;
      EntropyWorkProfile profile;
      Ok(OptimizeEntropyCodeAndPrepareClusters(streams, options, &expected, &expected_cost, &expected_values));
      Ok(OptimizeEntropyCodeFromContextPopulations(streams, options, populations, &actual,
                                                    &actual_cost, &actual_values, &profile));
      Check(expected == actual && expected_cost == actual_cost && expected_values == actual_values,
            "Dense populations changed prefix model, exact costs or retained values");
      Check(profile.prefix_histogram_token_count == 0 && profile.prefix_population_model_count == 1 &&
            !profile.prefix_value_collection_nanoseconds && !profile.prefix_value_aggregation_nanoseconds,
            "Dense populations scanned ordered values during model construction");
      for (auto selection : {AnsSelectionMode::kExact, AnsSelectionMode::kEstimated}) {
        EntropyCode a, b;
        EntropyCodeCost ac, bc;
        Ok(OptimizeAnsEntropyCodeWithPreparedClusters(streams, expected, expected_values, &a, &ac, nullptr, selection));
        Ok(OptimizeAnsEntropyCodeWithPreparedClusters(streams, actual, actual_values, &b, &bc, nullptr, selection));
        Check(a == b && ac == bc, "Dense populations changed downstream ANS model or costs");
      }
      if (contexts == 3 && !mapped) {
        // Model and preparation outputs are jointly failure-atomic.
        size_t failures = 0;
        for (size_t n = 0; n < 10000; ++n) {
          actual = expected; actual_cost = expected_cost; actual_values = expected_values;
          actual.context_count = 91;
          actual_cost.token_bits = 73;
          actual_values.context_count = 97;
          const auto saved_code = actual;
          const auto saved_cost = actual_cost;
          const auto saved_values = actual_values;
          ArmManagedHostAllocationFailureAfterForTest(n);
          const auto status = OptimizeEntropyCodeFromContextPopulations(streams, options, populations,
              &actual, &actual_cost, &actual_values);
          const bool pending = ManagedHostAllocationFailurePendingForTest();
          DisarmManagedHostAllocationFailureForTest();
          if (status.ok()) {
            Check(pending && failures > 5 && actual == expected && actual_cost == expected_cost &&
                  actual_values == expected_values, "Population fault sweep success differs");
            break;
          }
          Check(actual == saved_code && actual_cost == saved_cost && actual_values == saved_values,
                "Population optimizer failure changed outputs");
          Check(!pending && status.code() == StatusCode::kOutOfMemory, "Unexpected population failure");
          ++failures;
          Check(n != 9999, "Population failure sweep did not terminate");
        }
        std::cout << failures << " population model allocation failures preserve outputs\n";
      }
    }
  }
}
void CountsAndPlans() {
  {
    // Shape-only: baseline has UINT32_MAX tokens, but the palette metadata
    // pushes the largest layout over that limit. The eligible baseline still
    // needs the population reservation. No giant input/backing is allocated.
    gjxl::modular_internal::ModularEncodingOptions options;
    options.search = true;
    options.learning = {262144, 256};
    gjxl::modular_internal::AdaptiveModularWorkflowStoragePlan ordinary, population;
    constexpr Extent2D extent{65537, 65535};
    constexpr auto format = gjxl::modular_internal::PackedModularFormat::kGray8;
    ArmNextManagedHostAllocationFailureForTest();
    Ok(ComputeAdaptiveModularWorkflowStoragePlan(extent, format, options, &ordinary));
    options.adaptive_populations = true;
    Ok(ComputeAdaptiveModularWorkflowStoragePlan(extent, format, options, &population));
    Check(ManagedHostAllocationFailurePendingForTest(), "Large search population planner allocated");
    DisarmManagedHostAllocationFailureForTest();
    Check(population.peak_bytes > ordinary.peak_bytes + kMaximumPopulationScratchBytes,
          "Largest unsupported layout hid eligible population storage");
  }
  std::array<uint64_t, 6> counts{uint64_t{1} << 40, 0, 1, 7, 8, 9};
  const std::array<uint8_t, 2> map{0, 0};
  Storage<Storage<WeightedValue>> merged;
  Ok(MergeEntropyContextPopulations({2, 3, counts}, map, 1, &merged));
  Check(merged.size() == 1 && merged[0] == Storage<WeightedValue>{{0, (uint64_t{1} << 40) + 7}, {1, 8}, {2, 10}},
        "Wide integer population counts lost precision");
  const auto sentinel = merged;
  counts[0] = UINT64_MAX;
  Check(!MergeEntropyContextPopulations({2, 3, counts}, map, 1, &merged).ok() && merged == sentinel,
        "Merge overflow changed output");
  Check(!ValidateEntropyContextPopulations({2, 3, counts}, 1).ok(), "Population sum overflow accepted");
  using P = ModularPopulationStoragePlan;
  for (size_t contexts : {1, 17, 129, 256}) {
    for (size_t tokens : {size_t{1}, size_t{2049}, size_t{UINT32_MAX}}) {
      P plan;
      ArmNextManagedHostAllocationFailureForTest();
      Ok(ComputeModularPopulationStoragePlan(tokens, contexts, &plan));
      Check(ManagedHostAllocationFailurePendingForTest(), "GPU population planner allocated");
      DisarmManagedHostAllocationFailureForTest();
      Check(plan.scratch_bytes <= kMaximumPopulationScratchBytes + 8 &&
            plan.output_bytes == contexts * 4096 * 8 && plan.shards > 0 && plan.shards <= 32 &&
            plan.peak_bytes == plan.scratch_bytes + plan.output_bytes, "GPU population storage is not bounded");
      ModularPopulationDeviceLimits limits{tokens, std::max(plan.output_bytes, plan.scratch_bytes)};
      P bounded;
      Ok(ComputeModularPopulationStoragePlan(tokens, contexts, &bounded, &limits));
      Check(bounded == plan, "Inclusive population device limits differ");
      --limits.maximum_buffer_bytes;
      Check(ComputeModularPopulationStoragePlan(tokens, contexts, &bounded, &limits).code() == StatusCode::kUnsupported &&
            bounded == plan, "Population device limit rejection changed output");
    }
  }
  P sentinel_plan{.tokens = 73}, plan = sentinel_plan;
  Check(ComputeModularPopulationStoragePlan(100, 257, &plan).code() == StatusCode::kUnsupported && plan == sentinel_plan,
        "Large-context fallback boundary differs");
  Check(ComputeModularPopulationStoragePlan(size_t{UINT32_MAX} + 1, 1, &plan).code() == StatusCode::kUnsupported,
        "Overflowing atomic count accepted");
  Check(!ComputeModularPopulationStoragePlan(0, 1, &plan).ok(), "Empty populations accepted");
  struct Buffer : DeviceBuffer { Buffer(BackendId id, size_t n) : DeviceBuffer(BackendKind::kMetal, id, n) {} } buffer(7, 800);
  ModularPopulationInput input{{&buffer, 4, 400}, {&buffer, 408, 200}, 100, 3};
  Ok(ValidateModularPopulationInput(input, 7, &plan));
  const auto valid = plan;
  input.contexts.offset_bytes = 400;
  Check(!ValidateModularPopulationInput(input, 7, &plan).ok() && plan == valid, "Overlapping population ranges accepted");
  input.contexts.offset_bytes = 409;
  Check(!ValidateModularPopulationInput(input, 7, &plan).ok(), "Misaligned population range accepted");
  input.contexts.offset_bytes = 408;
  Check(!ValidateModularPopulationInput(input, 8, &plan).ok(), "Foreign population backing accepted");
}
}
int main() try { CountsAndPlans(); Models(); std::cout << "Population models, counts, storage and failures agree\n"; return 0; }
catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
