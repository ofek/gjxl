// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/ans_internal.h"
#include "codestream/entropy_internal.h"
#include "codestream/modular/search.h"
#include "codestream/modular/workflow.h"
#include "codestream/profile_internal.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#ifdef GJXL_ANS_SELECTION_REFERENCE
#include "modular_reference.h"
#endif

namespace {
using namespace gjxl;
using namespace gjxl::codestream_internal;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
constexpr auto kEstimated = AnsSelectionMode::kEstimated;
void Check(bool v, const char *message) {
  if (!v)
    throw std::runtime_error(message);
}
void Ok(const Status &s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
void Empty(const ExecutionDomain &d) {
  const auto s = d.snapshot();
  Check(!s.live_capacity_bytes && !s.reserved_unbacked_bytes && !s.active_reservations &&
            !s.active_cpu_participants,
        "Estimated selection leaked resources");
}
double EstimatedBits(std::span<const EntropyTokenStreamView> streams, const EntropyCode &code) {
  BitWriter header;
  Ok(WriteEntropyCode(code, &header));
  std::vector<std::array<uint64_t, 256>> counts(code.ans_histograms.size());
  uint64_t extra = 32 * streams.size();
  for (const auto stream : streams)
    for (size_t i = 0; i < stream.size(); ++i) {
      const auto token = stream[i];
      const auto cluster = code.context_map[token.context];
      HybridUintToken encoded;
      Ok(EncodeHybridUint(token.value, code.uint_configs[cluster], &encoded));
      ++counts[cluster][encoded.symbol];
      extra += encoded.extra_bit_count;
    }
  double bits = header.bits_written() + extra;
  for (size_t c = 0; c < counts.size(); ++c)
    for (size_t s = 0; s < 256; ++s)
      if (counts[c][s])
        bits += counts[c][s] * (12.0 - std::log2(code.ans_histograms[c].frequencies[s]));
  return bits;
}
void Models() {
  for (uint32_t fixture = 0; fixture < 5; ++fixture) {
    std::vector<std::vector<EntropyToken>> input(4);
    const size_t n = fixture == 0 ? 0 : (fixture == 1 ? 3 : 4097);
    for (size_t s = 1; s < input.size(); ++s)
      for (size_t i = 0; i < n / s; ++i) {
        const uint32_t value = fixture < 3    ? uint32_t(i % 2)
                               : fixture == 3 ? uint32_t((i * 19 + s) % 1024)
                                              : uint32_t(i * 987654321u + s * 71u);
        input[s].push_back({uint32_t(i % 3), value});
      }
    std::vector<std::vector<uint32_t>> values(input.size());
    std::vector<std::vector<uint16_t>> contexts(input.size());
    std::vector<EntropyTokenStreamView> streams;
    for (size_t s = 0; s < input.size(); ++s) {
      for (auto t : input[s]) {
        values[s].push_back(t.value);
        contexts[s].push_back(t.context);
      }
      streams.push_back(EntropyTokenStreamView::Split(values[s], contexts[s]));
    }
    EntropyCode prefix, estimated, with_cost, interleaved, prepared_model, exact;
    PreparedEntropyClusters populations;
    EntropyCodeCost prefix_cost;
    Ok(OptimizeEntropyCodeAndPrepareClusters(streams, {.context_count = 3}, &prefix, &prefix_cost,
                                             &populations));
    EntropyWorkProfile estimate_profile, exact_profile;
    Ok(OptimizeAnsEntropyCode(streams, prefix, &estimated, nullptr, &estimate_profile, kEstimated));
    Check(estimate_profile.ans_token_cost_nanoseconds == 0 &&
              estimate_profile.ans_alphabet_width_candidate_count == 1,
          "Estimated selection simulated candidates or built extra widths");
    Ok(OptimizeAnsEntropyCode(streams, prefix, &exact, nullptr, &exact_profile));
    Check(estimate_profile.ans_uint_config_candidate_count ==
                  exact_profile.ans_uint_config_candidate_count &&
              estimate_profile.ans_histogram_candidate_count ==
                  exact_profile.ans_histogram_candidate_count,
          "Estimation changed upstream candidate search");
    EntropyCodeCost cost;
    Ok(OptimizeAnsEntropyCode(streams, prefix, &with_cost, &cost, nullptr, kEstimated));
    Ok(OptimizeAnsEntropyCode(input, prefix, &interleaved, nullptr, nullptr, kEstimated));
    Ok(OptimizeAnsEntropyCodeWithPreparedClusters(streams, prefix, populations, &prepared_model,
                                                  nullptr, nullptr, kEstimated));
    Check(estimated == with_cost && estimated == interleaved && estimated == prepared_model,
          "Estimated model depends on cost request or token representation");
    BitWriter header;
    Ok(WriteEntropyCode(estimated, &header));
    uint64_t total = 0;
    for (size_t s = 0; s < streams.size(); ++s) {
      BitWriter payload;
      Ok(WriteTokenStream(streams[s], estimated, &payload));
      Check(payload.bits_written() == cost.section_token_bits[s], "Requested cost is not exact");
      total += payload.bits_written();
    }
    Check(total == cost.token_bits && header.bits_written() == cost.model_bits,
          "Exact cost totals changed");
    PreparedAnsEntropyCode candidates;
    Ok(PrepareAnsEntropyCodeWithPreparedClusters(streams, prefix, populations, &candidates));
    const double chosen = EstimatedBits(streams, estimated);
    bool found = false;
    for (const auto &c : candidates.candidates) {
      found |= c.code == estimated;
      Check(chosen <= EstimatedBits(streams, c.code) + 1e-6,
            "Selected width did not minimize independently computed estimate");
    }
    Check(found, "Estimated model was not in the original candidate set");
    const auto old = estimated;
    const auto old_cost = cost;
    ArmNextManagedHostAllocationFailureForTest();
    Check(OptimizeAnsEntropyCode(streams, prefix, &estimated, &cost, nullptr,
                                 static_cast<AnsSelectionMode>(99))
                      .code() == StatusCode::kInvalidArgument &&
              ManagedHostAllocationFailurePendingForTest() && estimated == old && cost == old_cost,
          "Invalid selection allocated or published output");
    DisarmManagedHostAllocationFailureForTest();
  }
  std::cout
      << "Estimated models, candidate coverage, representations and exact-cost contracts passed\n";
}
void Workflow(bool faults) {
  const Extent2D extent = faults ? Extent2D{3, 4} : Extent2D{257, 17};
  std::vector<uint8_t> pixels(size_t(extent.width) * extent.height);
  for (size_t i = 0; i < pixels.size(); ++i)
    pixels[i] = uint8_t((i / 31 + i % 7) % 256);
  const PackedModularImageView view{pixels, extent, extent.width, PackedModularFormat::kGray8};
  for (bool search : {false, true}) {
    ModularEncodingOptions options;
    options.entropy = EntropyCodingMode::kAns;
    options.ans_selection = kEstimated;
    options.search = search;
    options.learning = {4096, 32};
    options.tree_checkpoints = search;
    ModularWorkflowStoragePlan plan;
    Ok(search ? ComputeModularSearchStoragePlan(extent, view.format, options.entropy, &plan, 1,
                                                options.learning, options.tree_checkpoints)
              : ComputeModularWorkflowStoragePlan(extent, view.format, options.entropy,
                                                  options.coding, &plan));
    Ok(ExecutionDomain::Create({.managed_memory_bytes = plan.working.peak_bytes},
                               &options.execution_domain));
    std::vector<uint8_t> encoded;
    Ok(EncodeModularImage(view, options, &encoded));
    Empty(*options.execution_domain);
#ifdef GJXL_ANS_SELECTION_REFERENCE
    const auto decoded = test::modular_reference::DecodeLossless(encoded);
    Check(decoded.samples.size() == pixels.size(), "Estimated decode shape differs");
    for (size_t i = 0; i < pixels.size(); ++i)
      Check(decoded.samples[i] == pixels[i], "Estimated encode is not lossless");
#endif
    if (faults) {
      size_t failure = 0;
      for (; failure < 20000; ++failure) {
        std::vector<uint8_t> out{7, 8, 9};
        ArmManagedHostAllocationFailureAfterForTest(failure);
        const auto status = EncodeModularImage(view, options, &out);
        const bool pending = ManagedHostAllocationFailurePendingForTest();
        DisarmManagedHostAllocationFailureForTest();
        Empty(*options.execution_domain);
        if (status.ok()) {
          Check(pending && out == encoded, "Fault sweep changed successful bytes");
          break;
        }
        Check(!pending && status.code() == StatusCode::kOutOfMemory &&
                  out == std::vector<uint8_t>({7, 8, 9}),
              "Allocation failure changed output or status");
      }
      Check(failure < 20000, "Fault sweep did not finish");
      std::cout << "Estimated workflow search " << search << ": " << failure
                << " allocation failures\n";
    } else {
      options.execution_domain = {};
      options.cpu_thread_count = 4;
      std::vector<uint8_t> parallel;
      Ok(EncodeModularImage(view, options, &parallel));
      Check(parallel == encoded, "Parallel estimated selection changed bytes");
      options.cpu_thread_count = 1;
      Ok(ExecutionDomain::Create({.managed_memory_bytes = plan.working.peak_bytes - 1},
                                 &options.execution_domain));
      parallel = {7, 8, 9};
      Check(!EncodeModularImage(view, options, &parallel).ok() &&
                parallel == std::vector<uint8_t>({7, 8, 9}),
            "Under-budget encode published output");
      Empty(*options.execution_domain);
    }
  }
}
} // namespace
int main(int argc, char **argv) try {
  const bool faults = argc == 2 && std::string_view(argv[1]) == "--allocations";
  if (!faults)
    Models();
  Workflow(faults);
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
