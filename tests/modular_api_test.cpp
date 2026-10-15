// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gjxl/modular.hpp"
#include "gjxl/gjxl.h"
#include "gjxl/execution_domain.hpp"
#include "codestream/modular/search.h"
#include "codestream/modular/workflow.h"
#include "codestream/modular/reference_workflow.h"
#include "codestream/workflow_admission_scope.h"
#include "core/worker_launch_internal.h"
#include "modular_reference.h"
#include <iostream>
#include <fstream>
#include <iterator>
#include <barrier>
#include "codestream/workflow.h"
#include <thread>
#include <stdexcept>
#include <algorithm>
#include <cstring>
namespace {
using namespace gjxl;
namespace m = gjxl::modular_internal;
namespace ref = gjxl::test::modular_reference;
void Check(bool b, const char *text) {
  if (!b)
    throw std::runtime_error(text);
}
void Ok(Status s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
size_t cases = 0;
void Case(ModularPixelFormat format, bool big, Extent2D extent, ModularEntropy entropy,
          bool search, bool learned = false) {
  m::ModularInputProfile profile;
  Ok(m::ResolveModularInput(extent, static_cast<m::PackedModularFormat>(format), &profile));
  const size_t row = extent.width * profile.channel_count * profile.bytes_per_sample,
               stride = row + 3;
  std::vector<uint8_t> bytes(1 + stride * (extent.height - 1) + row, 0xcd);
  ref::IntegerImage expected{extent.width,
                             extent.height,
                             static_cast<uint32_t>(profile.channel_count),
                             profile.metadata.bits,
                             {}};
  for (size_t y = 0; y < extent.height; ++y)
    for (size_t x = 0; x < extent.width; ++x)
      for (size_t c = 0; c < profile.channel_count; ++c) {
        const uint16_t sample =
            static_cast<uint16_t>((c == 3 && x % 3 == 0 ? 0 : ((x + y) % 16) * 313 + c * 27) &
                                  (profile.metadata.bits == 8 ? 255 : 65535));
        expected.samples.push_back(sample);
        const size_t i =
            1 + y * stride + (x * profile.channel_count + c) * profile.bytes_per_sample;
        bytes[i] =
            static_cast<uint8_t>(big && profile.bytes_per_sample == 2 ? sample >> 8 : sample);
        if (profile.bytes_per_sample == 2)
          bytes[i + 1] = static_cast<uint8_t>(big ? sample : sample >> 8);
      }
  ModularImageView input{std::span(bytes).subspan(1), extent, stride, format,
                         big ? ModularByteOrder::kBigEndian : ModularByteOrder::kLittleEndian};
  ModularEncodingOptions options;
  options.cpu_thread_count = 1;
  options.entropy = entropy;
  options.search = search;
  options.policy = learned ? ModularPolicy::kLearned
                            : search ? ModularPolicy::kSearch : ModularPolicy::kBaseline;
  std::vector<uint8_t> serial, parallel;
  ModularEncodingSummary summary;
  Ok(EncodeModularImage(input, options, &serial, &summary));
  Check(summary.extent == extent && summary.encoded_bytes == serial.size() &&
            summary.total_seconds > 0 && summary.backend == ModularBackend::kCpu,
        "Wrong summary");
  m::ModularWorkflowStoragePlan plan;
  auto mode =
      entropy == ModularEntropy::kPrefix ? EntropyCodingMode::kPrefix : EntropyCodingMode::kAns;
  size_t memory_limit;
  if (learned) {
    resource_budget_internal::ResourceBudget budget;
    m::ReferenceWorkflowOptions reference_options;
    if (entropy == ModularEntropy::kPrefix)
      reference_options.entropy = {m::PreparedModularEntropyPolicy::kPrefix};
    codestream_internal::CodestreamBuffer reference;
    m::ReferenceWorkflowStats stats;
    Ok(m::EncodeReferenceModularImage(
        {input.bytes, extent, stride, static_cast<m::PackedModularFormat>(format),
         static_cast<m::SampleByteOrder>(input.byte_order)},
        reference_options, budget, &reference, &stats));
    Check(std::ranges::equal(serial, reference.view()), "Public default differs from qualified learner");
    Check(stats.learning_calls == stats.fits + stats.empty_learning_calls,
          "Unexpected learning scope counts");
    memory_limit = budget.snapshot().peak_committed_bytes;
  } else {
    Ok(search ? m::ComputeModularSearchStoragePlan(
                  extent, static_cast<m::PackedModularFormat>(format), mode, &plan, 4)
            : m::ComputeModularWorkflowStoragePlan(
                  extent, static_cast<m::PackedModularFormat>(format), mode, {}, &plan, 4));
    memory_limit = plan.working.peak_bytes + plan.output.peak_bytes;
  }
  Ok(ExecutionDomain::Create(
      {.managed_memory_bytes = memory_limit,
       .cpu_participant_limit = 4},
      &options.execution_domain));
  options.cpu_thread_count = 4;
  Ok(EncodeModularImage(input, options, &parallel));
  Check(serial == parallel && ref::DecodeLossless(parallel) == expected,
        "Parallel/native decode mismatch");
  auto snapshot = options.execution_domain->snapshot();
  Check(snapshot.peak_cpu_protected_slots <= 4 && snapshot.active_cpu_participants == 0 &&
            snapshot.active_reservations == 0 && snapshot.live_capacity_bytes == 0 &&
            snapshot.peak_committed_bytes <= memory_limit,
        "Leaked CPU or reservation");
  if (learned)
    Check(snapshot.peak_cpu_protected_slots == 1, "Learned path violated serial participation");
  else if (extent.width > 256)
    Check(snapshot.peak_cpu_protected_slots > 1, "Parallel path did not execute");
  GJXLContextOptions context_options;
  Check(gjxl_context_options_init(&context_options, sizeof(context_options)) == GJXL_OK,
        "Context init failed");
  context_options.num_cpu_threads = 4;
  GJXLContext *context = nullptr;
  Check(gjxl_context_create(&context_options, &context) == GJXL_OK, "Context failed");
  GJXLPixelFormat cformat = profile.channel_count == 1   ? GJXL_PIXEL_FORMAT_GRAY8_SRGB
                            : profile.channel_count == 3 ? GJXL_PIXEL_FORMAT_RGB8_SRGB
                                                         : GJXL_PIXEL_FORMAT_RGBA8_SRGB;
  if (profile.bytes_per_sample == 2)
    cformat = static_cast<GJXLPixelFormat>((profile.channel_count == 1   ? 4
                                            : profile.channel_count == 3 ? 6
                                                                         : 8) +
                                           big);
  GJXLImageView view{sizeof(GJXLImageView),
                     static_cast<uint32_t>(extent.width),
                     static_cast<uint32_t>(extent.height),
                     cformat,
                     bytes.data() + 1,
                     bytes.size() - 1,
                     stride};
  GJXLModularOptions coptions;
  Check(gjxl_modular_options_init(&coptions, sizeof(coptions)) == GJXL_OK, "Modular init failed");
  coptions.search = search;
  coptions.entropy = static_cast<GJXLModularEntropy>(entropy);
  coptions.policy = static_cast<GJXLModularPolicy>(options.policy);
  GJXLBuffer output{};
  Check(gjxl_encode_modular(context, &view, &coptions, &output) == GJXL_OK, gjxl_get_last_error());
  Check(std::ranges::equal(serial, std::span(output.data, output.size)), "C ABI bytes mismatch");
  gjxl_buffer_free(&output);
  gjxl_context_destroy(context);
  for (auto backend : {ModularBackend::kMetal, ModularBackend::kCuda}) {
    options.backend = backend;
#if GJXL_TEST_HAS_METAL
    if (learned && backend == ModularBackend::kMetal) {
      options.execution_domain.reset(); // GPU retained backing has its own stage bounds.
      Ok(EncodeModularImage(input, options, &parallel, &summary));
      Check(parallel == serial && summary.backend == ModularBackend::kMetal,
            "Explicit learned Metal changed bytes or backend summary");
      continue;
    }
#endif
    Check(EncodeModularImage(input, options, &parallel, &summary).code() ==
                  StatusCode::kUnsupported &&
              parallel == serial,
          "Forced GPU silently fell back or changed output");
  }
  ++cases;
}
void LaunchFailures() {
  using namespace thread_budget_internal;
  std::vector<uint8_t> pixels(769 * 17 * 3, 19);
  ModularImageView image{pixels, {769, 17}, 769 * 3};
  ModularEncodingOptions options;
  options.policy = ModularPolicy::kBaseline;
  options.cpu_thread_count = 4;
  Ok(ExecutionDomain::Create({.cpu_participant_limit = 4}, &options.execution_domain));
  for (auto site : {WorkerLaunchSite::kModularTokenization, WorkerLaunchSite::kModularEmission})
    for (auto kind : {WorkerLaunchFailureKind::kSystemError, WorkerLaunchFailureKind::kBadAlloc})
      for (size_t before : {size_t{0}, size_t{1}}) {
        WorkerLaunchFaultForTesting fault{site, before, kind};
        WorkerLaunchFaultScopeForTesting scope(&fault);
        std::vector<uint8_t> output{7, 8, 9};
        auto status = EncodeModularImage(image, options, &output);
        Check(!status.ok() && fault.triggered && fault.launched_in_group == before &&
                  output == std::vector<uint8_t>({7, 8, 9}),
              "Launch failure not atomic");
        const auto snapshot = options.execution_domain->snapshot();
        Check(snapshot.active_cpu_participants == 0 && snapshot.reserved_cpu_workers == 0 &&
                  snapshot.suspended_cpu_workers == 0 && snapshot.active_reservations == 0,
              "Launch failure leaked capacity");
        Ok(EncodeModularImage(image, options, &output));
      }
}
void MixedDomain() {
  constexpr size_t limit = 512 * 1024 * 1024;
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({.managed_memory_bytes = limit, .cpu_participant_limit = 2}, &domain));
  std::array<Status, 4> statuses;
  std::barrier start(4);
  std::array<std::thread, 4> workers;
  for (size_t i = 0; i < 4; ++i)
    workers[i] = std::thread([&, i] {
      start.arrive_and_wait();
      std::vector<uint8_t> output;
      if (i % 2) {
        std::array<float, 3> pixel{0.1f, 0.3f, 0.7f};
        ConstImage3FView image{{ConstPlaneF32View{&pixel[0], {1, 1}, 1},
                                ConstPlaneF32View{&pixel[1], {1, 1}, 1},
                                ConstPlaneF32View{&pixel[2], {1, 1}, 1}}};
        VarDctEncodingOptions options;
        options.backend = VarDctBackendPreference::kCpu;
        options.cpu_thread_count = 2;
        options.execution_domain = domain;
        options.effort = 1;
        statuses[i] = EncodeLinearRgbVarDctCodestream(image, options, &output);
      } else {
        std::vector<uint8_t> pixels(513 * 17 * 4, 71);
        ModularEncodingOptions options;
        options.execution_domain = domain;
        options.cpu_thread_count = 4;
        statuses[i] = EncodeModularImage({pixels, {513, 17}, 513 * 4, ModularPixelFormat::kRgba8},
                                         options, &output);
      }
    });
  for (auto &worker : workers)
    worker.join();
  for (auto status : statuses)
    Ok(status);
  const auto snapshot = domain->snapshot();
  Check(snapshot.peak_backing_bytes <= limit && snapshot.peak_cpu_protected_slots <= 2 &&
            snapshot.active_cpu_participants == 0 && snapshot.active_reservations == 0,
        "Mixed domain cap or drain failed");
}
void PublicationFailures() {
  using namespace resource_budget_internal;
  constexpr size_t limit = 128 * 1024 * 1024;
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({.managed_memory_bytes = limit, .cpu_participant_limit = 1}, &domain));
  GJXLExecutionDomain *c_domain = nullptr;
  Ok(CreateCExecutionDomain(domain, &c_domain));
  GJXLContextOptions context_options;
  Check(gjxl_context_options_init(&context_options, sizeof(context_options)) == GJXL_OK,
        "Context init failed");
  context_options.num_cpu_threads = 1;
  context_options.execution_domain = c_domain;
  GJXLContext *context = nullptr;
  Check(gjxl_context_create(&context_options, &context) == GJXL_OK, "Context create failed");
  gjxl_execution_domain_destroy(c_domain);
  std::array<uint8_t, 8 * 9 * 4> pixels{};
  GJXLImageView image{sizeof(GJXLImageView), 8,    9, GJXL_PIXEL_FORMAT_RGBA8_SRGB, pixels.data(),
                      pixels.size(),         8 * 4};
  size_t count = 0;
  for (; count < 4000; ++count) {
    GJXLBuffer output{};
    ArmManagedHostAllocationFailureAfterForTest(count);
    auto result = gjxl_encode_modular(context, &image, nullptr, &output);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    const auto snapshot = domain->snapshot();
    Check(snapshot.live_capacity_bytes == 0 && snapshot.active_reservations == 0 &&
              snapshot.active_cpu_participants == 0 && snapshot.peak_committed_bytes <= limit &&
              snapshot.peak_cpu_protected_slots == 1,
          "C publication leaked or escaped the shared execution domain");
    if (result == GJXL_OK) {
      Check(pending, "Missed C failure position");
      gjxl_buffer_free(&output);
      break;
    }
    Check(result == GJXL_ERROR_OUT_OF_MEMORY && output.data == nullptr && output.size == 0 &&
              !pending,
          "C publication failure was not atomic");
  }
  Check(count > 10 && count < 4000, "C publication sweep incomplete");
  gjxl_context_destroy(context);
  std::cout << count << " C publication failure positions passed\n";
}
void Prefixes() {
  struct Prefix {
    uint32_t size;
    uint32_t canary;
  } options{4, 0x12345678};
  Check(gjxl_modular_options_init(reinterpret_cast<GJXLModularOptions *>(&options), 4) == GJXL_OK &&
            options.canary == 0x12345678,
        "Sized prefix overwrite");
  GJXLContext *context = nullptr;
  Check(gjxl_context_create(nullptr, &context) == GJXL_OK, "Default context failed");
  uint8_t pixel = 73;
  GJXLImageView image{sizeof(GJXLImageView), 1, 1, GJXL_PIXEL_FORMAT_GRAY8_SRGB, &pixel, 1, 1};
  GJXLBuffer output{};
  Check(gjxl_encode_modular(context, &image, reinterpret_cast<GJXLModularOptions *>(&options),
                            &output) == GJXL_OK,
        "Sized prefix encode failed");
  std::vector<uint8_t> expected;
  Ok(EncodeModularImage({std::span(&pixel, 1), {1, 1}, 1, ModularPixelFormat::kGray8},
                        {}, &expected));
  Check(std::ranges::equal(expected, std::span(output.data, output.size)),
        "Four-byte prefix does not use public defaults");
  gjxl_buffer_free(&output);
  Check(gjxl_encode_modular(context, &image, nullptr, &output) == GJXL_OK &&
            std::ranges::equal(expected, std::span(output.data, output.size)),
        "Null C options differ from C++ defaults");
  gjxl_buffer_free(&output);
  for (size_t size : {size_t{8}, size_t{12}, sizeof(GJXLModularOptions)}) {
    struct Storage { GJXLModularOptions options; uint32_t tail; } storage;
    std::memset(&storage, 0xa5, sizeof(storage));
    Check(gjxl_modular_options_init(&storage.options, size) == GJXL_OK,
          "C prefix initialization failed");
    const auto *bytes = reinterpret_cast<const uint8_t *>(&storage);
    Check(std::all_of(bytes + size, bytes + sizeof(storage), [](auto b) { return b == 0xa5; }),
          "C prefix initializer changed future fields");
    if (size >= 12)
      Check(storage.options.entropy == GJXL_MODULAR_ENTROPY_ANS, "C initializer did not select ANS");
    Check(gjxl_encode_modular(context, &image, &storage.options, &output) == GJXL_OK &&
              std::ranges::equal(expected, std::span(output.data, output.size)),
          "Initialized C prefix differs from defaults");
    gjxl_buffer_free(&output);
  }
  for (int invalid = 0; invalid < 4; ++invalid) {
    GJXLModularOptions bad;
    Check(gjxl_modular_options_init(&bad, sizeof(bad)) == GJXL_OK, "C init failed");
    if (invalid == 0) bad.policy = 256;
    if (invalid == 1) bad.entropy = -1;
    if (invalid == 2) bad.search = 2;
    if (invalid == 3) { bad.policy = GJXL_MODULAR_POLICY_BASELINE; bad.search = 1; }
    Check(gjxl_encode_modular(context, &image, &bad, &output) == GJXL_ERROR_INVALID_ARGUMENT &&
              output.data == nullptr && output.size == 0, "Invalid C options accepted");
  }
  image.pixels = nullptr;
  Check(gjxl_encode_modular(context, &image, nullptr, &output) == GJXL_ERROR_INVALID_ARGUMENT &&
        output.data == nullptr && output.size == 0, "Null raster accepted");
  gjxl_context_destroy(context);
}

void DefaultFailures() {
  using namespace resource_budget_internal;
  std::array<uint8_t, 17 * 19 * 3> pixels{};
  for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = static_cast<uint8_t>(i * 37);
  ModularImageView image{pixels, {17, 19}, 17 * 3};
  ModularEncodingOptions options;
  Ok(ExecutionDomain::Create({.managed_memory_bytes = 1, .cpu_participant_limit = 1},
                             &options.execution_domain));
  std::vector<uint8_t> output{1, 2, 3};
  ModularEncodingSummary summary{{7, 8}, 9, ModularBackend::kCuda, 123.0};
  auto check_unchanged = [&] {
    Check(output == std::vector<uint8_t>({1, 2, 3}) && summary.extent == Extent2D{7, 8} &&
              summary.encoded_bytes == 9 && summary.backend == ModularBackend::kCuda &&
              summary.total_seconds == 123.0, "Failed default encode changed output or summary");
    auto snapshot = options.execution_domain->snapshot();
    Check(snapshot.active_cpu_participants == 0 && snapshot.live_capacity_bytes == 0,
          "Failed default encode leaked backing or CPU participation");
  };
  auto check_c_failure = [&](GJXLResult expected) {
    GJXLExecutionDomain *c_domain = nullptr;
    Ok(CreateCExecutionDomain(options.execution_domain, &c_domain));
    GJXLContextOptions context_options;
    Check(gjxl_context_options_init(&context_options, sizeof(context_options)) == GJXL_OK,
          "C context options failed");
    context_options.execution_domain = c_domain;
    GJXLContext *context = nullptr;
    Check(gjxl_context_create(&context_options, &context) == GJXL_OK, "C context failed");
    gjxl_execution_domain_destroy(c_domain);
    GJXLImageView view{sizeof(GJXLImageView), 17, 19, GJXL_PIXEL_FORMAT_RGB8_SRGB,
                       pixels.data(), pixels.size(), 17 * 3};
    GJXLBuffer encoded{};
    Check(gjxl_encode_modular(context, &view, nullptr, &encoded) == expected &&
              encoded.data == nullptr && encoded.size == 0,
          "C default bypassed execution-domain admission");
    gjxl_context_destroy(context);
    check_unchanged();
  };
  Check(EncodeModularImage(image, options, &output, &summary).code() == StatusCode::kOutOfMemory,
        "Learned workflow bypassed hard memory limit");
  check_unchanged();
  check_c_failure(GJXL_ERROR_OUT_OF_MEMORY);
  for (int invalid = 0; invalid < 4; ++invalid) {
    auto bad = options;
    if (invalid == 0) bad.policy = static_cast<ModularPolicy>(255);
    if (invalid == 1) bad.entropy = static_cast<ModularEntropy>(255);
    if (invalid == 2) bad.cpu_thread_count = 257;
    if (invalid == 3) { bad.policy = ModularPolicy::kBaseline; bad.search = true; }
    Check(EncodeModularImage(image, bad, &output, &summary).code() == StatusCode::kInvalidArgument,
          "Public options were not validated before admission");
    check_unchanged();
  }
  constexpr size_t limit = 128 * 1024 * 1024;
  Ok(ExecutionDomain::Create({.managed_memory_bytes = limit, .cpu_participant_limit = 1},
                             &options.execution_domain));
  {
    ResourceReservation occupied;
    Ok(codestream_internal::WorkflowAdmission::Budget(*options.execution_domain)
           .TryReserve(limit, &occupied));
    Check(EncodeModularImage(image, options, &output, &summary).code() == StatusCode::kUnavailable,
          "Busy learned admission did not return Unavailable");
    check_unchanged();
    check_c_failure(GJXL_ERROR_UNAVAILABLE);
  }
  size_t failures = 0;
  for (; failures < 4000; ++failures) {
    ArmManagedHostAllocationFailureAfterForTest(failures);
    auto status = EncodeModularImage(image, options, &output, &summary);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    if (status.ok()) { Check(pending, "Missed C++ allocation failure"); break; }
    Check(status.code() == StatusCode::kOutOfMemory && !pending, "Wrong learned allocation failure");
    check_unchanged();
    Check(options.execution_domain->snapshot().active_reservations == 0,
          "Failed learned stage leaked reservation");
  }
  Check(failures > 10 && failures < 4000, "C++ allocation sweep incomplete");
  std::cout << failures << " learned C++ allocation failure positions passed\n";
}
} // namespace
int main(int argc, char **argv) try {
  if (argc == 8 && std::string(argv[1]) == "--check-file") {
    std::ifstream encoded(argv[2], std::ios::binary), raw(argv[3], std::ios::binary);
    std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(encoded), {}};
    auto decoded = ref::DecodeLossless(bytes);
    Check(decoded.width == std::stoull(argv[4]) && decoded.height == std::stoull(argv[5]) &&
              decoded.channels == std::stoul(argv[6]) && decoded.bits == std::stoul(argv[7]),
          "CLI metadata mismatch");
    for (auto sample : decoded.samples) {
      const int lo = raw.get(), hi = raw.get();
      Check(lo >= 0 && hi >= 0 && sample == (lo | (hi << 8)), "CLI sample mismatch");
    }
    Check(raw.get() == EOF, "CLI sample count mismatch");
    return 0;
  }
  Prefixes();
  DefaultFailures();
  LaunchFailures();
  MixedDomain();
  PublicationFailures();
  for (auto format :
       {ModularPixelFormat::kGray8, ModularPixelFormat::kRgb8, ModularPixelFormat::kRgba8,
        ModularPixelFormat::kGray16, ModularPixelFormat::kRgb16, ModularPixelFormat::kRgba16})
    for (bool big : {false, true}) {
      if (big && static_cast<int>(format) < 3)
        continue;
      for (auto extent : {Extent2D{17, 19}, Extent2D{513, 17}})
        for (auto entropy : {ModularEntropy::kPrefix, ModularEntropy::kAns})
          for (bool search : {false, true})
            Case(format, big, extent, entropy, search);
      for (auto extent : {Extent2D{17, 19}, Extent2D{513, 17}})
        for (auto entropy : {ModularEntropy::kPrefix, ModularEntropy::kAns})
          Case(format, big, extent, entropy, false, true);
    }
  Case(ModularPixelFormat::kRgb8, false, {1024, 257}, ModularEntropy::kAns, false, true);
  Case(ModularPixelFormat::kRgb8, false, {513, 513}, ModularEntropy::kAns, false, true);
  Case(ModularPixelFormat::kRgba16, false, {513, 513}, ModularEntropy::kAns, false, true);
  Case(ModularPixelFormat::kRgba16, true, {513, 513}, ModularEntropy::kAns, false, true);
  std::cout << cases << " public C/C++ serial-parallel exact cases passed\n";
} catch (const std::exception &e) {
  std::cerr << "case " << cases << ": " << e.what() << '\n';
  return 1;
}
