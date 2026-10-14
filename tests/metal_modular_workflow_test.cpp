// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/profile.h"
#include "codestream/modular/gpu_token_workflow.h"
#include "codestream/modular/gpu_workflow.h"
#include "codestream/workflow_admission_scope.h"
#include "codestream/workflow_admission_test.h"
#include "core/worker_launch_internal.h"
#include "gpu/metal/metal_backend.h"
#include "codec/modular/profile.h"
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#if GJXL_MODULAR_WORKFLOW_REFERENCE
#include "modular_reference.h"
#endif

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::modular_gpu_internal;
using namespace gjxl::resource_budget_internal;
#if GJXL_MODULAR_TOKEN_WORKFLOW
using WorkflowPlan = GpuModularTokenWorkflowStoragePlan;
constexpr auto EncodeGpu = EncodeGpuModularTokensImage;
constexpr auto EncodeGpuOwned = EncodeGpuModularTokensImageOwned;
constexpr size_t kDeviceAllocations = 4;
#else
using WorkflowPlan = GpuModularWorkflowStoragePlan;
constexpr auto EncodeGpu = EncodeGpuModularImage;
constexpr auto EncodeGpuOwned = EncodeGpuModularImageOwned;
constexpr size_t kDeviceAllocations = 2;
#endif
void Check(bool v, const char *m) {
  if (!v)
    throw std::runtime_error(m);
}
void Ok(const Status &s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
void Empty(const ExecutionDomain &domain) {
  const auto s = domain.snapshot();
  Check(s.live_capacity_bytes == 0 && s.reserved_unbacked_bytes == 0 &&
            s.active_reservations == 0 && s.waiting_requests == 0 &&
            s.active_cpu_participants == 0 && s.reserved_cpu_workers == 0 &&
            s.suspended_cpu_workers == 0 && s.waiting_cpu_callers == 0,
        "GPU workflow leaked memory/admission/CPU participation");
}
constexpr std::array formats{PackedModularFormat::kGray8, PackedModularFormat::kRgb8,
                             PackedModularFormat::kRgba8, PackedModularFormat::kGray16,
                             PackedModularFormat::kRgb16, PackedModularFormat::kRgba16};
struct Fixture {
  Extent2D extent;
  PackedModularFormat format;
  SampleByteOrder order;
  ModularInputProfile profile;
  size_t stride;
  std::vector<uint16_t> samples;
  std::vector<uint8_t> bytes;
  Fixture(Extent2D extent, PackedModularFormat format, SampleByteOrder order, size_t padding = 1)
      : extent(extent), format(format), order(order) {
    Ok(ResolveModularInput(extent, format, &profile));
    const size_t row = extent.width * profile.channel_count * profile.bytes_per_sample;
    stride = row + padding;
    bytes.assign(3 + (extent.height - 1) * stride + row, 0xad);
    uint32_t rng = 0x9187ac23;
    const uint32_t maximum = profile.metadata.bits == 8 ? 255 : 65535;
    constexpr uint16_t edges[] = {0, 1, 2, 255, 256, 32767, 32768, 65534, 65535};
    for (size_t y = 0; y < extent.height; ++y)
      for (size_t x = 0; x < extent.width; ++x)
        for (size_t c = 0; c < profile.channel_count; ++c) {
          rng ^= rng << 13;
          rng ^= rng >> 17;
          rng ^= rng << 5;
          uint16_t value = static_cast<uint16_t>((x < 9 ? edges[(x + y + c) % 9] : rng) & maximum);
          if (c == 3 && x % 3 == 0)
            value = 0;
          samples.push_back(value);
          const size_t i =
              3 + y * stride + (x * profile.channel_count + c) * profile.bytes_per_sample;
          if (profile.bytes_per_sample == 1 || order == SampleByteOrder::kLittleEndian) {
            bytes[i] = static_cast<uint8_t>(value);
            if (profile.bytes_per_sample == 2)
              bytes[i + 1] = static_cast<uint8_t>(value >> 8);
          } else {
            bytes[i] = static_cast<uint8_t>(value >> 8);
            bytes[i + 1] = static_cast<uint8_t>(value);
          }
        }
  }
  PackedModularImageView view() const {
    return {std::span(bytes).subspan(3), extent, stride, format, order};
  }
};
WorkflowPlan Plan(GpuBackend &gpu, const Fixture &f, const ModularEncodingOptions &options) {
  WorkflowPlan plan;
  auto *capability = QueryGpuModularInputPreparation(gpu);
  Check(capability, "Metal has no Modular capability");
#if GJXL_MODULAR_TOKEN_WORKFLOW
  auto *tokens = QueryGpuModularTokenGeneration(gpu);
  Check(tokens, "Metal has no Modular token capability");
  Ok(ComputeGpuModularTokenWorkflowStoragePlan(f.extent, f.stride, f.format, f.order, options,
                                               capability->modular_input_limits(),
                                               tokens->modular_token_limits(), &plan));
#else
  Ok(ComputeGpuModularWorkflowStoragePlan(f.extent, f.stride, f.format, f.order, options,
                                          capability->modular_input_limits(), &plan));
#endif
  return plan;
}
size_t files = 0, fault_checks = 0;
#if GJXL_MODULAR_WORKFLOW_REFERENCE
size_t decoded_samples = 0;
#endif
std::filesystem::path artifacts;
std::ofstream ledger;
void Case(GpuBackend &gpu, const Fixture &f, ModularEncodingOptions options = {}) {
  const auto plan = Plan(gpu, f, options);
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({plan.peak_bytes, 4}, &domain));
  std::vector<uint8_t> expected, actual{7, 8, 9};
  auto scalar = options;
  scalar.execution_domain = {};
  scalar.cpu_thread_count = 1;
  Ok(EncodeModularImage(f.view(), scalar, &expected));
  options.execution_domain = domain;
  const auto before = gpu.stats();
  const auto unchanged = f.bytes;
  Ok(EncodeGpu(gpu, f.view(), options, &actual));
  Empty(*domain);
  const auto s = domain->snapshot();
  Check(actual == expected && f.bytes == unchanged,
        "GPU/CPU complete bytes differ or input changed");
  Check(gpu.stats().successful_allocations == before.successful_allocations + kDeviceAllocations &&
            gpu.stats().committed_submissions == before.committed_submissions + 1,
        "GPU workflow fell back, unpacked twice or allocated extra device backing");
  Check(s.peak_committed_bytes == plan.peak_bytes && s.peak_backing_bytes <= plan.peak_bytes &&
            s.peak_cpu_participants <= std::min(size_t{4}, plan.cpu.participants),
        "GPU workflow exceeded complete-call admission/CPU bound");
  const auto file = artifacts / ("case-" + std::to_string(files) + ".jxl");
  std::ofstream out(file, std::ios::binary);
  out.write(reinterpret_cast<const char *>(actual.data()), actual.size());
  Check(bool(out), "Failed to retain complete GPU file");
  ledger << files << ',' << f.extent.width << ',' << f.extent.height << ',' << int(f.format) << ','
         << int(f.order) << ',' << f.stride << ',' << int(options.coding.rct) << ','
         << int(options.entropy) << ',' << options.cpu_thread_count << ',' << actual.size() << ','
         << plan.peak_bytes << ',' << s.peak_backing_bytes << '\n';
#if GJXL_MODULAR_WORKFLOW_REFERENCE
  namespace ref = gjxl::test::modular_reference;
  const ref::IntegerImage source{f.extent.width, f.extent.height,
                                 static_cast<uint32_t>(f.profile.channel_count),
                                 f.profile.metadata.bits, f.samples};
  Check(ref::DecodeLossless(actual) == source,
        "Pinned decoder samples differ from original integers");
  const auto h = ref::InspectHeaders(actual);
  Check(h.modular && !h.xyb && h.bits == source.bits && h.width == source.width &&
            h.height == source.height && h.color_channels == (source.channels == 1 ? 1u : 3u) &&
            h.extra_channels == (source.channels == 4 ? 1u : 0u) &&
            h.alpha_bits == (source.channels == 4 ? source.bits : 0u) && !h.alpha_associated &&
            h.group_dimension == 256 && h.final_frame && h.passes == 1 &&
            h.modular_16_bit_buffer_sufficient ==
                (source.bits == 8 && options.coding == ModularCodingPolicy{}),
        "GPU complete-file header differs from prescribed CPU profile");
  Check(ref::ReadSectionSizes(actual).size() == h.sections, "GPU file TOC differs");
  decoded_samples += source.samples.size();
#endif
  ++files;
}
void Matrix(GpuBackend &gpu) {
  for (auto format : formats)
    for (auto order : {SampleByteOrder::kLittleEndian, SampleByteOrder::kBigEndian}) {
      const Fixture f({9, 7}, format, order);
      for (uint8_t rct = 0; rct < (f.profile.channel_count == 1 ? 1 : 42); ++rct)
        for (auto entropy : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
          ModularEncodingOptions options{.entropy = entropy};
          options.coding.rct = rct;
          Case(gpu, f, options);
        }
      // Every format crosses group/DC boundaries; grouped cases use four CPU participants.
      if (order == SampleByteOrder::kLittleEndian)
        continue;
      for (auto extent :
           {Extent2D{1, 1}, {1, 257}, {257, 1}, {255, 256}, {256, 256}, {257, 257}, {2049, 3}})
        for (auto entropy : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
          ModularEncodingOptions options{.entropy = entropy, .cpu_thread_count = 4};
          options.coding.rct = f.profile.channel_count == 1 ? 0 : 6;
          Case(gpu, Fixture(extent, format, order), options);
        }
    }
  const Fixture f({17, 19}, PackedModularFormat::kRgba16, SampleByteOrder::kBigEndian);
  for (auto entropy : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
    for (unsigned predictor = 0; predictor < 14; ++predictor) {
#if GJXL_MODULAR_TOKEN_WORKFLOW
      if (predictor == 6)
        continue;
#endif
      ModularEncodingOptions options{.entropy = entropy};
      options.coding.rct = 41;
      options.coding.tree.nodes[0].predictor = static_cast<Predictor>(predictor);
      options.coding.tree.nodes[0].offset = -3;
      Case(gpu, f, options);
    }
    ModularEncodingOptions split{.entropy = entropy, .cpu_thread_count = 4};
    split.coding.rct = 6;
    split.coding.tree.size = 3;
#if GJXL_MODULAR_TOKEN_WORKFLOW
    split.coding.tree.nodes[0] = {.property = 8, .split = 0, .left = 2, .right = 1};
    split.coding.tree.nodes[1].predictor = Predictor::kAverage4;
#else
    split.coding.tree.nodes[0] = {.property = 15, .split = 0, .left = 1, .right = 2};
    split.coding.tree.nodes[1].predictor = Predictor::kWeighted;
#endif
    split.coding.tree.nodes[2].predictor = Predictor::kGradient;
    Case(gpu, Fixture({257, 17}, f.format, f.order), split);
  }
  // Large padding makes upload, rather than downstream CPU work, the peak.
  Case(gpu, Fixture({3, 2}, PackedModularFormat::kRgba16, f.order, 1u << 20));
}
void Failures(GpuBackend &gpu) {
  const Fixture f({3, 2}, PackedModularFormat::kRgba16, SampleByteOrder::kBigEndian);
  for (auto entropy : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
    ModularEncodingOptions options{.entropy = entropy};
    options.coding.rct = 6;
    const auto plan = Plan(gpu, f, options);
    std::shared_ptr<const ExecutionDomain> domain, small;
    Ok(ExecutionDomain::Create({plan.peak_bytes + plan.cpu.output.peak_bytes, 1}, &domain));
    Ok(ExecutionDomain::Create({plan.peak_bytes - 1, 1}, &small));
    options.execution_domain = domain;
    std::vector<uint8_t> expected;
    auto control = options;
    control.execution_domain = {};
    Ok(EncodeModularImage(f.view(), control, &expected));
    codestream_internal::CodestreamBuffer owned;
    Ok(EncodeGpuOwned(gpu, f.view(), options, &owned));
    const auto live = owned.capacity();
    const auto *identity = owned.data();
    const auto unchanged = f.bytes;
    for (unsigned fault = 0; fault < 3; ++fault) {
      if (fault == 0)
        Ok(ArmNextMetalAllocationFailureForTest(gpu));
      else
        Ok(ArmNextMetalSubmissionFailureForTest(gpu, fault == 1, fault == 2));
      const auto s = EncodeGpuOwned(gpu, f.view(), options, &owned);
      Check(!s.ok() && owned.data() == identity && std::ranges::equal(owned.view(), expected) &&
                domain->snapshot().live_capacity_bytes == live &&
                domain->snapshot().active_reservations == 0,
            "GPU failure replaced owned codestream or leaked charges");
      ++fault_checks;
    }
    // Sweep every managed host allocation from preparation through final publication.
    size_t failure = 0;
    for (; failure < 10000; ++failure) {
      ArmManagedHostAllocationFailureAfterForTest(failure);
      const auto s = EncodeGpuOwned(gpu, f.view(), options, &owned);
      const bool pending = ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      if (s.ok()) {
        Check(pending && std::ranges::equal(owned.view(), expected), "Incomplete host fault sweep");
        break;
      }
      Check(!pending && s.code() == StatusCode::kOutOfMemory && owned.data() == identity &&
                std::ranges::equal(owned.view(), expected) &&
                domain->snapshot().live_capacity_bytes == live &&
                domain->snapshot().active_reservations == 0 &&
                domain->snapshot().active_cpu_participants == 0,
            "Post-GPU failure changed old output or leaked lifetime/admission");
      ++fault_checks;
    }
    Check(failure > 10 && failure < 10000, "Host allocation sweep did not finish");
    Check(f.bytes == unchanged && domain->snapshot().live_capacity_bytes == owned.capacity(),
          "Replacement changed source or lost retained output charge");
    std::thread destroy([value = std::move(owned)]() mutable { value.Reset(); });
    destroy.join();
    Empty(*domain);
    std::vector<uint8_t> output{7, 8, 9};
    auto too_small = options;
    too_small.execution_domain = small;
    const auto allocations = gpu.stats().successful_allocations;
    Check(!EncodeGpu(gpu, f.view(), too_small, &output).ok() &&
              gpu.stats().successful_allocations == allocations &&
              output == std::vector<uint8_t>({7, 8, 9}),
          "Insufficient complete-call admission allocated GPU backing");
    Empty(*small);
    codestream_internal::ArmNextWorkflowAdmissionCapacityForTest(1);
    const auto status = EncodeGpu(gpu, f.view(), options, &output);
    codestream_internal::DisarmWorkflowAdmissionCapacityForTest();
    Check(status.resource_plan_exceeded() && output == std::vector<uint8_t>({7, 8, 9}),
          "Underplanned GPU workflow escaped reservation or changed output");
    Empty(*domain);
    // An unrepresentable residual must not publish on either route.
    auto invalid_residual = options;
    invalid_residual.coding.tree.nodes[0].multiplier = 2;
    invalid_residual.coding.tree.nodes[0].offset = 1;
    Check(!EncodeGpu(gpu, f.view(), invalid_residual, &output).ok() &&
              output == std::vector<uint8_t>({7, 8, 9}),
          "Unrepresentable residual published output");
    Empty(*domain);
    Ok(EncodeGpu(gpu, f.view(), options, &output));
    Check(output == expected, "Fault injection remained armed after failure sweep");
    Empty(*domain);
    std::cout << "Host failure boundaries " << int(entropy) << ": " << failure << '\n';
  }
}
void ConcurrentAndNested(GpuBackend &gpu) {
  const Fixture f({257, 17}, PackedModularFormat::kRgba8, SampleByteOrder::kBigEndian);
  ModularEncodingOptions options{.cpu_thread_count = 4};
  options.coding.rct = 6;
  const auto plan = Plan(gpu, f, options);
  std::shared_ptr<const ExecutionDomain> domain, foreign;
  Ok(ExecutionDomain::Create({plan.peak_bytes, 2}, &domain));
  Ok(ExecutionDomain::Create({}, &foreign));
  options.execution_domain = domain;
  std::vector<uint8_t> expected;
  auto control = options;
  control.execution_domain = {};
  Ok(EncodeModularImage(f.view(), control, &expected));
  std::array<std::thread, 4> workers;
  std::array<Status, 4> statuses;
  std::array<std::vector<uint8_t>, 4> outputs;
  for (size_t i = 0; i < workers.size(); ++i)
    workers[i] =
        std::thread([&, i] { statuses[i] = EncodeGpu(gpu, f.view(), options, &outputs[i]); });
  for (auto &worker : workers)
    worker.join();
  for (size_t i = 0; i < workers.size(); ++i) {
    Ok(statuses[i]);
    Check(outputs[i] == expected, "Concurrent GPU file differs");
  }
  Empty(*domain);
  Check(domain->snapshot().peak_committed_bytes == plan.peak_bytes &&
            domain->snapshot().peak_cpu_participants <= 2,
        "Shared GPU workflow domain exceeded limits");
  {
    codestream_internal::WorkflowAdmission outer;
    Ok(outer.Start(plan.peak_bytes, domain));
    std::vector<uint8_t> output{7, 8, 9};
    auto wrong = options;
    wrong.execution_domain = foreign;
    const auto submits = gpu.stats().committed_submissions;
    Check(!EncodeGpu(gpu, f.view(), wrong, &output).ok() &&
              gpu.stats().committed_submissions == submits,
          "Foreign nested domain submitted GPU work");
    Ok(EncodeGpu(gpu, f.view(), options, &output));
    Check(output == expected && domain->snapshot().active_reservations == 1,
          "Nested GPU workflow reserved again or changed output");
  }
  Empty(*domain);
}
#if GJXL_MODULAR_TOKEN_WORKFLOW
void ComposedFailureBoundaries(GpuBackend &gpu) {
  using namespace thread_budget_internal;
  Fixture f({513, 257}, PackedModularFormat::kRgba16, SampleByteOrder::kBigEndian);
  ModularEncodingOptions options{.cpu_thread_count = 4};
  options.coding.rct = 6;
  const auto plan = Plan(gpu, f, options);
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({plan.peak_bytes, 4}, &domain));
  options.execution_domain = domain;
  const std::vector<uint8_t> sentinel{7, 8, 9};
  std::vector<uint8_t> output = sentinel;
  const auto &generation = plan.generation;
  size_t cap = generation.host_layout.retained_bytes;
  size_t allocation = 0;
  for (size_t bytes : {generation.input.upload_bytes, generation.input.output_bytes,
                       generation.tokens.output_bytes, generation.tokens.dispatch_bytes}) {
    cap += bytes;
    const auto before = gpu.stats();
    codestream_internal::ArmNextWorkflowAdmissionCapacityForTest(cap - 1);
    const auto status = EncodeGpu(gpu, f.view(), options, &output);
    codestream_internal::DisarmWorkflowAdmissionCapacityForTest();
    Check(status.resource_plan_exceeded() && output == sentinel &&
              gpu.stats().successful_allocations == before.successful_allocations + allocation &&
              gpu.stats().committed_submissions == before.committed_submissions,
          "Composed device allocation boundary escaped admission or submitted work");
    Empty(*domain);
    ++allocation;
    ++fault_checks;
  }
  for (auto kind : {WorkerLaunchFailureKind::kSystemError, WorkerLaunchFailureKind::kBadAlloc})
    for (size_t launched : {0u, 1u, 2u}) {
      WorkerLaunchFaultForTesting fault{WorkerLaunchSite::kModularEmission, launched, kind};
      ModularProfile profile;
      Status status;
      const auto before = gpu.stats();
      {
        WorkerLaunchFaultScopeForTesting injection(&fault);
        ProfileSession profiling(&profile);
        status = EncodeGpu(gpu, f.view(), options, &output);
      }
      Check(!status.ok() && fault.triggered && fault.launched_in_group == launched &&
                output == sentinel &&
                gpu.stats().committed_submissions == before.committed_submissions + 1 &&
                profile.seconds[static_cast<size_t>(ProfileStage::kGpuInputTokens)] > 0 &&
                profile.seconds[static_cast<size_t>(ProfileStage::kTokens)] == 0 &&
                profile.seconds[static_cast<size_t>(ProfileStage::kAssembly)] == 0,
            "Partial CPU launch failure changed output, ran CPU tokens or missed injection");
      Empty(*domain);
      ++fault_checks;
    }
  Ok(EncodeGpu(gpu, f.view(), options, &output));
  Empty(*domain);
  options.execution_domain = {};
  std::vector<uint8_t> expected;
  Ok(EncodeModularImage(f.view(), options, &expected));
  Check(output == expected, "Composed worker failure did not recover");
}
#endif

} // namespace
int main(int argc, char **argv) try {
  Check(argc == 2, "Expected complete-file artifact directory");
  artifacts = argv[1];
  std::filesystem::create_directories(artifacts);
  ledger.open(artifacts / "cases.csv");
  ledger << "case,width,height,format,byte_order,stride,rct,entropy,cpu_threads,bytes,admission,"
            "peak_backing\n";
  {
    std::unique_ptr<GpuBackend> gpu, embedded;
    Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &gpu));
    Ok(CreateEmbeddedMetalBackend({}, &embedded));
    std::cout << "GPU Modular workflow device: " << gpu->name() << '\n';
    Matrix(*gpu);
    Failures(*gpu);
    ConcurrentAndNested(*gpu);
#if GJXL_MODULAR_TOKEN_WORKFLOW
    ComposedFailureBoundaries(*gpu);
#endif
    for (auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
      ModularEncodingOptions options{.entropy = mode};
      options.coding.rct = 41;
      Case(*embedded, Fixture({17, 19}, PackedModularFormat::kRgba16, SampleByteOrder::kBigEndian),
           options);
      if (mode == EntropyCodingMode::kAns) {
        options.ans_selection = AnsSelectionMode::kEstimated;
        ModularProfile profile;
        ProfileSession session(&profile);
        Case(*embedded, Fixture({17, 19}, PackedModularFormat::kRgba16, SampleByteOrder::kBigEndian),
             options);
        Check(profile.entropy_work.ans_token_cost_nanoseconds == 0 &&
                  profile.entropy_work.ans_alphabet_width_candidate_count == 2,
              "Prescribed GPU path ignored estimated ANS selection");
      }
    }
  }
  const auto s = DefaultResourceBudget().snapshot();
  Check(s.committed_bytes() == 0 && s.total.backing_count == 0 && s.total.pending_count == 0,
        "GPU complete-file tests leaked default charges");
  std::cout << files << " exact CPU/GPU files; " << fault_checks << " injected failures; ";
#if GJXL_MODULAR_WORKFLOW_REFERENCE
  std::cout << decoded_samples << " original samples matched pinned decoder";
#else
  std::cout << "independent decoder not configured";
#endif
  std::cout << "; admission, concurrency, atomic publication and both libraries passed\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
