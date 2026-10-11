// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/ans_internal.h"
#include "gpu/ops/ans_token_cost.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::ans_gpu_internal;
using namespace gjxl::resource_budget_internal;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(const Status &s) { Check(s.ok(), "ANS token-cost contract failed"); }
struct Buffer : DeviceBuffer {
  explicit Buffer(BackendId id, size_t bytes) : DeviceBuffer(BackendKind::kMetal, id, bytes) {}
};
void Shapes() {
  AnsTokenCostStoragePlan p;
  ArmNextManagedHostAllocationFailureForTest();
  Ok(ComputeAnsTokenCostStoragePlan(36000000, 193, 4, 256, 32, &p));
  Check(p.sections_bytes == 3088 && p.models_offset == 3328 && p.context_maps_offset == 3584 &&
            p.clusters_offset == 4608 && p.status_offset == 1479168 &&
            p.dispatch_bytes == 1479172 && p.output_bytes == 12352 &&
            p.retained_bytes == p.output_bytes &&
            p.peak_bytes > 2 * p.dispatch_bytes + p.output_bytes,
        "ANS cost arena layout or transient overlap changed");
  auto limits = AnsTokenCostDeviceLimits{36000000, 193, p.dispatch_bytes};
  const auto saved = p;
  Ok(ComputeAnsTokenCostStoragePlan(36000000, 193, 4, 256, 32, &p, &limits));
  Check(p == saved, "Device preflight changed ANS cost layout");
  --limits.maximum_buffer_bytes;
  Check(ComputeAnsTokenCostStoragePlan(36000000, 193, 4, 256, 32, &p, &limits).code() ==
                StatusCode::kUnsupported &&
            p == saved,
        "ANS cost one-byte-short limit changed output or succeeded");
  for (auto dims : std::array<std::array<size_t, 5>, 10>{{{0, 1, 1, 1, 1},
                                                          {1, 0, 1, 1, 1},
                                                          {1, 1, 0, 1, 1},
                                                          {1, 1, 5, 1, 1},
                                                          {1, 1, 1, 0, 1},
                                                          {1, 1, 1, 65537, 1},
                                                          {1, 1, 1, 1, 2},
                                                          {1, 1, 1, 256, 65},
                                                          {SIZE_MAX, 1, 1, 1, 1},
                                                          {1, SIZE_MAX, 1, 1, 1}}})
    Check(!ComputeAnsTokenCostStoragePlan(dims[0], dims[1], dims[2], dims[3], dims[4], &p).ok() &&
              p == saved,
          "Invalid ANS cost shape published a plan");
  Ok(ComputeAnsTokenCostStoragePlan(1, 3, 1, 65536, 64, &p));
  Check(ManagedHostAllocationFailurePendingForTest(), "ANS cost planner allocated");
  DisarmManagedHostAllocationFailureForTest();
}
void Inputs() {
  std::array<uint16_t, 3> frequencies{2048, 0, 2048}, offsets{0, 2048, 2048};
  std::array<uint64_t, 3> reciprocals{uint64_t{1} << 33, 0, uint64_t{1} << 33};
  std::array<uint16_t, 4096> map;
  std::iota(map.begin(), map.end(), uint16_t{0});
  std::array<uint8_t, 2> contexts{0, 0};
  AnsTokenCostClusterView cluster{4, 2, 0, frequencies, offsets, map, reciprocals};
  AnsTokenCostModelView model{contexts, std::span(&cluster, 1)};
  std::array<AnsTokenCostSection, 4> sections{{{0, 0}, {0, 3}, {3, 2}, {5, 0}}};
  Buffer buffer(7, 30);
  AnsTokenCostInput input{{&buffer, 0, 20}, {&buffer, 20, 10}, 5, sections, std::span(&model, 1)};
  AnsTokenCostStoragePlan plan;
  ArmNextManagedHostAllocationFailureForTest();
  Ok(ValidateAnsTokenCostInput(input, 7, &plan));
  const auto saved = plan;
  auto reject = [&] {
    Check(!ValidateAnsTokenCostInput(input, 7, &plan).ok() && plan == saved,
          "Invalid ANS cost input accepted or published a plan");
  };
  sections[2].offset = 2;
  reject();
  sections[2].offset = 3;
  sections[2].count = UINT64_MAX;
  reject();
  sections[2].count = 2;
  sections[3].offset = 4;
  reject();
  sections[3].offset = 5;
  contexts[1] = 1;
  reject();
  contexts[1] = 0;
  frequencies[1] = 1;
  reject();
  frequencies[1] = 0;
  offsets[1] = 0;
  reject();
  offsets[1] = 2048;
  reciprocals[1] = 1;
  reject();
  reciprocals[1] = 0;
  --reciprocals[0];
  reject();
  ++reciprocals[0];
  map[7] = 4096;
  reject();
  map[7] = 7;
  cluster.split_exponent = 16;
  reject();
  cluster.split_exponent = 4;
  cluster.msb_in_token = 5;
  reject();
  cluster.msb_in_token = 2;
  cluster.lsb_in_token = 3;
  reject();
  cluster.lsb_in_token = 0;
  input.values.offset_bytes = 1;
  reject();
  input.values.offset_bytes = 0;
  input.contexts.offset_bytes = 18;
  reject();
  input.contexts.offset_bytes = 20;
  input.contexts.offset_bytes = 22;
  reject();
  input.contexts.offset_bytes = 20;
  input.contexts.size_bytes = 8;
  reject();
  input.contexts.size_bytes = 10;
  Check(!ValidateAnsTokenCostInput(input, 8, &plan).ok() && plan == saved,
        "Foreign ANS token owner was accepted");
  Ok(ValidateAnsTokenCostInput(input, 7, &plan));
  Check(ManagedHostAllocationFailurePendingForTest(), "ANS cost validation allocated");
  DisarmManagedHostAllocationFailureForTest();
  Ok(CheckAnsTokenCostDeviceStatus(0));
  for (uint32_t bit : {kCostContextError, kCostSymbolError, kCostStateError})
    Check(CheckAnsTokenCostDeviceStatus(bit).code() == StatusCode::kInvalidArgument,
          "ANS cost device data error status differs");
  Check(CheckAnsTokenCostDeviceStatus(8).code() == StatusCode::kInternal,
        "ANS cost unknown device status accepted");
}
// Validate the views against real scalar-optimized models, not just synthetic
// table arithmetic. All four alphabet widths occur in the optimizer's contract;
// the selected native model must pass without rebuilding its reverse tables.
void NativeModels() {
  std::vector<EntropyToken> tokens;
  for (uint32_t i = 0; i < 4096; ++i)
    tokens.push_back({i % 3, (i * i + i * 37) % 257});
  const auto stream = EntropyTokenStreamView::Interleaved(tokens);
  EntropyCode prefix, ans;
  Ok(OptimizeEntropyCode(std::span(&stream, 1), {.context_count = 3}, &prefix));
  Ok(OptimizeAnsEntropyCode(std::span(&stream, 1), prefix, &ans));
  std::vector<AnsTokenCostClusterView> clusters;
  for (size_t i = 0; i < ans.ans_histograms.size(); ++i) {
    const auto &h = ans.ans_histograms[i];
    const auto c = ans.uint_configs[i];
    clusters.push_back({c.split_exponent, c.msb_in_token, c.lsb_in_token, h.frequencies,
                        h.reverse_offsets, h.reverse_map, h.reciprocal_frequencies});
  }
  const AnsTokenCostModelView model{ans.context_map, clusters};
  Ok(ValidateAnsTokenCostModels(std::span(&model, 1)));
}
} // namespace
int main() try {
  Shapes();
  Inputs();
  NativeModels();
  std::cout << "ANS token-cost operation, admission and borrowed-model "
               "contracts passed\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
