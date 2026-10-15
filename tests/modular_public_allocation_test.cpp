// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gjxl/modular.hpp"
#include "modular_allocation_test_util.h"
#ifdef GJXL_PUBLIC_ALLOCATION_ORACLE
#include "modular_reference.h"
#endif
#include <chrono>
#include <sstream>

namespace {
using namespace gjxl;
using namespace resource_budget_internal;
void Check(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
void Empty(const ExecutionDomain &domain) {
  const auto state = domain.snapshot();
  Check(state.live_capacity_bytes == 0 && state.reserved_unbacked_bytes == 0 &&
            state.active_reservations == 0 && state.active_cpu_participants == 0,
        "Learned failure leaked backing, reservation, or CPU participation");
}
void Case(Extent2D extent, ModularEntropy entropy, gjxl_test::AllocationReport &report) {
  // Exactly the packed input used by the legacy search fault fixtures.
  const size_t row_bytes = extent.width * 8, stride = row_bytes + 7;
  std::vector<uint8_t> bytes(5 + stride * extent.height, 0xcd);
  for (size_t y = 0; y < extent.height; ++y)
    for (size_t x = 0; x < row_bytes; ++x)
      bytes[5 + y * stride + x] = static_cast<uint8_t>(x * 37 + y * 71);
  const auto original = bytes;
  ModularImageView input{std::span(bytes).subspan(5), extent, stride,
                         ModularPixelFormat::kRgba16, ModularByteOrder::kBigEndian};
  ModularEncodingOptions options;
  Check(options.policy == ModularPolicy::kLearned && !options.search,
        "Public default no longer selects the learned workflow");
  options.cpu_thread_count = 1;
  options.entropy = entropy;
  Ok(ExecutionDomain::Create(
      {.managed_memory_bytes = size_t{1} << 30, .cpu_participant_limit = 1},
      &options.execution_domain));
  std::vector<uint8_t> expected;
  const size_t count = gjxl_test::CountManagedAllocations(
      [&] { return EncodeModularImage(input, options, &expected); });
  Check(count > 10 && count < 10000, "Unexpected learned allocation trace size");
  Empty(*options.execution_domain);
#ifdef GJXL_PUBLIC_ALLOCATION_ORACLE
  const auto decoded = gjxl::test::modular_reference::DecodeLossless(expected);
  Check(decoded.width == extent.width && decoded.height == extent.height &&
            decoded.channels == 4 && decoded.bits == 16 &&
            decoded.samples.size() == extent.width * extent.height * 4,
        "Learned fixture decode metadata mismatch");
  for (size_t y = 0; y < extent.height; ++y)
    for (size_t x = 0; x < extent.width * 4; ++x) {
      const size_t offset = 5 + y * stride + x * 2;
      Check(decoded.samples[y * extent.width * 4 + x] ==
                (uint16_t{bytes[offset]} << 8 | bytes[offset + 1]),
            "Learned fixture decode sample mismatch");
    }
#endif
  const auto start = std::chrono::steady_clock::now();
  const std::vector<uint8_t> sentinel{7, 8, 9};
  std::vector<uint8_t> output;
  for (size_t index = 0; index < count; ++index) {
    output = sentinel;
    ModularEncodingSummary summary{{7, 8}, 9, ModularBackend::kCuda, 123.0};
    Status status;
    bool pending;
    {
      gjxl_test::ManagedAllocationFailureScope fault(index);
      status = EncodeModularImage(input, options, &output, &summary);
      pending = fault.pending();
    }
    Check(!pending && status.code() == StatusCode::kOutOfMemory,
          "Learned allocation failure was ignored or returned the wrong status");
    Check(output == sentinel && summary.extent == Extent2D{7, 8} &&
              summary.encoded_bytes == 9 && summary.backend == ModularBackend::kCuda &&
              summary.total_seconds == 123.0 && bytes == original,
          "Failed learned encode changed output, summary, or input");
    Empty(*options.execution_domain);
  }
  // The first index beyond the successful trace must recover without firing.
  {
    gjxl_test::ManagedAllocationFailureScope fault(count);
    Ok(EncodeModularImage(input, options, &output));
    Check(fault.pending() && output == expected && bytes == original,
          "Learned allocation trace changed or recovery was not deterministic");
  }
  Empty(*options.execution_domain);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::ostringstream row;
  row << "{\"policy\":\"learned\",\"coverage\":\"exhaustive\",\"format\":\"rgba16be\","
      << "\"width\":" << extent.width << ",\"height\":" << extent.height
      << ",\"entropy\":\"" << (entropy == ModularEntropy::kPrefix ? "prefix" : "ANS")
      << "\",\"allocation_count\":" << count << ",\"injected_failures\":" << count
      << ",\"seconds\":" << seconds << ",\"exact_decode\":"
#ifdef GJXL_PUBLIC_ALLOCATION_ORACLE
      << "true"
#else
      << "false"
#endif
      << ",\"passed\":true}";
  report.Write(row.str());
}
} // namespace
int main(int argc, char **argv) try {
  gjxl_test::AllocationReport report;
  Check(argc == 1 || (argc == 3 && std::string_view(argv[1]) == "--report"),
        "Usage: modular_public_allocation_test [--report PATH]");
  if (argc == 3) report.Open(argv[2]);
  for (auto entropy : {ModularEntropy::kPrefix, ModularEntropy::kAns})
    for (auto extent : {Extent2D{17, 19}, Extent2D{257, 3}}) Case(extent, entropy, report);
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
