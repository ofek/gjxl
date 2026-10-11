// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/profile.h"
#include "codestream/modular/adaptive_gpu_workflow.h"
#include "codestream/workflow_admission_scope.h"
#include "codestream/workflow_admission_test.h"
#include "core/cpu_execution.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/ops/modular_tree_statistics.h"
#if GJXL_MODULAR_ADAPTIVE_REFERENCE
#include "modular_reference.h"
#endif
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::modular_gpu_internal;
using namespace gjxl::resource_budget_internal;
size_t cases = 0, decoded_samples = 0, fault_checks = 0;
std::filesystem::path artifacts;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(const Status &s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
void Empty(const ExecutionDomain &domain) {
  const auto s = domain.snapshot();
  Check(!s.live_capacity_bytes && !s.reserved_unbacked_bytes && !s.active_reservations &&
            !s.active_cpu_participants && !s.reserved_cpu_workers && !s.suspended_cpu_workers &&
            !s.waiting_cpu_callers && !s.waiting_requests,
        "Adaptive workflow leaked memory or CPU admission");
}
struct Fixture {
  ModularInputProfile profile;
  PackedModularImageView view;
  std::vector<uint8_t> bytes;
  std::vector<uint16_t> samples;
  Fixture(PackedModularFormat format, Extent2D extent, bool palette = false) {
    Ok(ResolveModularInput(extent, format, &profile));
    const size_t row = extent.width * profile.channel_count * profile.bytes_per_sample;
    const size_t stride = row + 3;
    bytes.resize(1 + (extent.height - 1) * stride + row, 0xad);
    view = {std::span(bytes).subspan(1), extent, stride, format, SampleByteOrder::kBigEndian};
    for (size_t y = 0; y < extent.height; ++y)
      for (size_t x = 0; x < extent.width; ++x)
        for (size_t c = 0; c < profile.channel_count; ++c) {
          const uint32_t value =
              palette ? ((x + y * 5) % 8) * (17 + c * 7)
                      : ((x < extent.width / 2 ? x * (17 + c) + y * y : y * 33 + x * x + c * 31) &
                         (profile.metadata.bits == 8 ? 255 : 65535));
          // Store exactly the integers represented by the packed source. The
          // fourth palette component can exceed 255 before 8-bit conversion.
          samples.push_back(
              static_cast<uint16_t>(value & (profile.metadata.bits == 8 ? 255 : 65535)));
          const size_t i =
              1 + y * stride + (x * profile.channel_count + c) * profile.bytes_per_sample;
          bytes[i] = static_cast<uint8_t>(profile.bytes_per_sample == 1 ? value : value >> 8);
          if (profile.bytes_per_sample == 2)
            bytes[i + 1] = static_cast<uint8_t>(value);
        }
  }
};
void Decode(const Fixture &f, const std::vector<uint8_t> &encoded) {
#if GJXL_MODULAR_ADAPTIVE_REFERENCE
  namespace ref = gjxl::test::modular_reference;
  const ref::IntegerImage original{f.view.extent.width, f.view.extent.height,
                                   static_cast<uint32_t>(f.profile.channel_count),
                                   f.profile.metadata.bits, f.samples};
  Check(ref::DecodeLossless(encoded) == original, "Adaptive search pinned decode differs");
  decoded_samples += original.samples.size();
#else
  (void)f;
  (void)encoded;
#endif
}
AdaptiveModularWorkflowStoragePlan Plan(const Fixture &f, const ModularEncodingOptions &options) {
  AdaptiveModularWorkflowStoragePlan result;
  ArmNextManagedHostAllocationFailureForTest();
  const auto s =
      ComputeAdaptiveModularWorkflowStoragePlan(f.view.extent, f.view.format, options, &result);
  const bool pending = ManagedHostAllocationFailurePendingForTest();
  DisarmManagedHostAllocationFailureForTest();
  Ok(s);
  Check(pending &&
            result.peak_bytes == result.cpu.working.peak_bytes + result.additional_peak_bytes,
        "Adaptive whole-call planner allocated or omitted the cache lifetime");
  return result;
}
void Case(GpuBackend &gpu, const Fixture &f, ModularEncodingOptions options, bool extra = false,
          bool ans_cost = false, bool require_gpu_cost = false, bool tree_statistics = false,
          bool require_gpu_statistics = false) {
  options.adaptive_ans_cost = ans_cost;
  options.adaptive_tree_statistics = tree_statistics;
  const auto plan = Plan(f, options);
  options.adaptive_ans_cost = false;
  options.adaptive_tree_statistics = false;
  std::vector<uint8_t> cpu, hybrid{7, 8, 9};
  ModularProfile baseline, actual;
  Ok(ExecutionDomain::Create({plan.cpu.working.peak_bytes, options.cpu_thread_count},
                             &options.execution_domain));
  {
    ProfileSession profile(&baseline);
    Ok(EncodeModularImage(f.view, options, &cpu));
  }
  const auto selected = baseline.resolved;
  baseline.resolved = {};
  Empty(*options.execution_domain);
  Ok(ExecutionDomain::Create({plan.peak_bytes, options.cpu_thread_count},
                             &options.execution_domain));
  options.adaptive_gpu = &gpu;
  options.adaptive_ans_cost = ans_cost;
  options.adaptive_tree_statistics = tree_statistics;
  const auto before = gpu.stats();
  {
    ProfileSession profile(&actual);
    Ok(EncodeModularImage(f.view, options, &hybrid));
  }
  Check(cpu == hybrid && actual.resolved == selected &&
            actual.encoded_candidates == baseline.encoded_candidates &&
            actual.adaptive_gpu.encoded_candidates == actual.encoded_candidates &&
            actual.adaptive_gpu.cpu_fallback_candidates == 0 &&
            gpu.stats().committed_submissions - before.committed_submissions ==
                actual.encoded_candidates + actual.adaptive_gpu.ans_gpu_groups + actual.adaptive_gpu.tree_gpu_groups,
        "Hybrid search changed bytes, selected policy, candidate count, or silently fell back");
  const auto &g = actual.adaptive_gpu;
  Check(g.ans_cpu_fallback_groups == g.ans_cpu_small_groups &&
            (ans_cost && options.entropy == EntropyCodingMode::kAns
                 ? g.ans_gpu_groups + g.ans_cpu_fallback_groups >= actual.encoded_candidates &&
                       g.ans_gpu_states >= g.ans_gpu_groups
                 : g.ans_gpu_groups == 0 && g.ans_gpu_states == 0 &&
                       g.ans_cpu_fallback_groups == 0) &&
            (!require_gpu_cost || (g.ans_gpu_groups > 0 && g.ans_cpu_fallback_groups == 0)),
        "Exact ANS cost route was omitted or fallback attribution differs");
  if (ans_cost && options.entropy == EntropyCodingMode::kAns && !require_gpu_cost)
    Check(g.ans_gpu_groups == 0 && g.ans_cpu_small_groups > 0,
          "Small workflow did not use the explicit scalar cost crossover");
  if (tree_statistics) {
    Check(require_gpu_statistics
              ? g.tree_gpu_preparations > 0 && g.tree_gpu_groups > 0 && g.tree_gpu_samples > 0
              : g.tree_gpu_groups == 0 && g.tree_cpu_small_properties > 0,
          "Learner statistics route differs from the required GPU/small-leaf path");
    Check(g.tree_cpu_small_properties <= g.tree_cpu_fallback_properties &&
              g.tree_prepare_seconds + g.tree_evaluate_seconds <=
                  actual.training_work.seconds[static_cast<size_t>(TrainingProfileStage::kSplitScanning)] + 1e-6,
          "Learner statistics fallback/timing attribution differs");
    Check(options.learning.mode == TreeLearningMode::kQuantized
              ? g.tree_cpu_unsupported_learning_calls > 0 : g.tree_cpu_unsupported_learning_calls == 0,
          "Alternative learner CPU fallback was omitted or uncounted");
  } else {
    Check(g.tree_gpu_preparations == 0 && g.tree_gpu_groups == 0 && g.tree_gpu_samples == 0 &&
              g.tree_cpu_fallback_properties == 0 && g.tree_cpu_unsupported_learning_calls == 0 &&
              g.tree_prepare_seconds == 0 && g.tree_evaluate_seconds == 0,
          "Disabled learner statistics invoked the new route");
  }
  Check(g.feature_preparations + g.feature_reuses == g.encoded_candidates &&
            g.evictions + 1 == g.feature_preparations && g.feature_seconds > 0 &&
            g.token_seconds > 0 &&
            g.feature_seconds + g.token_seconds <=
                actual.seconds[static_cast<size_t>(ProfileStage::kTokens)] + 1e-6 &&
            baseline.adaptive_gpu == AdaptiveGpuWorkProfile{},
        "Adaptive profile has missing work or overlapping intervals");
  if (options.search) {
    const auto &a = actual.training_work, &b = baseline.training_work;
    Check(g.feature_reuses > 0 && a.learning_calls == b.learning_calls &&
              a.population == b.population && a.samples == b.samples &&
              a.split_evaluations == b.split_evaluations &&
              a.sorted_properties == b.sorted_properties && a.sorted_samples == b.sorted_samples,
          "Adaptive search changed training work or did not reuse features");
  }
  actual.resolved = {};
  Empty(*options.execution_domain);
  Check(options.execution_domain->snapshot().peak_backing_bytes <= plan.peak_bytes,
        "Adaptive complete search exceeded its bound");
  Decode(f, hybrid);
  std::ofstream file(artifacts / ("case-" + std::to_string(cases++) + ".jxl"), std::ios::binary);
  file.write(reinterpret_cast<const char *>(hybrid.data()), hybrid.size());
  Check(bool(file), "Could not retain adaptive search file");
  if (extra) {
    {
      ProfileSession disabled(nullptr);
      Ok(EncodeModularImage(f.view, options, &hybrid));
    }
    Check(cpu == hybrid, "Disabling adaptive profiling changed output");
    Empty(*options.execution_domain);
    codestream_internal::CodestreamBuffer owned;
    Ok(EncodeModularImageOwned(f.view, options, &owned));
    Check(std::ranges::equal(cpu, owned.view()), "Adaptive owned publication differs");
    owned = {};
    Empty(*options.execution_domain);
  }
}
class NoCapability : public GpuBackend {
public:
  BackendKind kind() const noexcept override { return BackendKind::kMetal; }
  std::string_view name() const noexcept override { return "Adaptive workflow test backend"; }
  Status Allocate(size_t, std::unique_ptr<DeviceBuffer> *) override {
    throw std::runtime_error("Unexpected raw allocation");
  }
  Status CopyHostToDevice(DeviceBuffer &, const void *, size_t, size_t) override {
    throw std::runtime_error("Unexpected upload");
  }
  Status CopyDeviceToHost(const DeviceBuffer &, void *, size_t, size_t) override {
    throw std::runtime_error("Unexpected readback");
  }
  Status ForwardTransform(const TransformBatch &, std::unique_ptr<GpuSubmission> *) override {
    return Status::Unsupported("unused");
  }
  Status InverseTransform(const TransformBatch &, std::unique_ptr<GpuSubmission> *) override {
    return Status::Unsupported("unused");
  }
};
// Qualification-only forwarding capability. Native owners stay associated with
// the real Metal operation; this shim injects faults between search candidates.
class FaultCapability final : public NoCapability, public GpuAdaptiveModularTokenGeneration {
public:
  explicit FaultCapability(GpuBackend &gpu)
      : native(*QueryGpuAdaptiveModularTokenGeneration(gpu)) {}
  GpuAdaptiveModularTokenGeneration &native;
  size_t preparations = 0, generations = 0, fail_preparation = 0, fail_generation = 0;
  bool limited = false, unsupported_after_preflight = false;
  ModularTokenDeviceLimits adaptive_modular_limits() const noexcept override {
    auto limits = native.adaptive_modular_limits();
    if (limited)
      limits.maximum_buffer_bytes = 1;
    return limits;
  }
  Status PrepareAdaptiveModularFeatures(
      const ModularEncoderFrame &frame, std::span<const AdaptiveModularTokenSlice> slices,
      const WeightedPredictorParameters &parameters, bool weighted,
      std::unique_ptr<PreparedGpuAdaptiveModularFeatures> *out) override {
    Check(CurrentResourceContext().reservation && thread_budget_internal::HasCpuParticipation(),
          "Adaptive feature work started without admission");
    if (++preparations == fail_preparation)
      return unsupported_after_preflight
                 ? Status::Unsupported("Injected numeric feature failure")
                 : Status::DeviceError("Injected replacement feature failure");
    return native.PrepareAdaptiveModularFeatures(frame, slices, parameters, weighted, out);
  }
  Status GenerateAdaptiveModularTokens(const PreparedGpuAdaptiveModularFeatures &features,
                                       const DecisionTree &tree,
                                       std::unique_ptr<PreparedGpuModularTokens> *out) override {
    if (++generations == fail_generation)
      return Status::DeviceError("Injected later candidate failure");
    return native.GenerateAdaptiveModularTokens(features, tree, out);
  }
};
void FallbacksAndFailures(GpuBackend &gpu) {
  Fixture f(PackedModularFormat::kRgb8, {9, 7});
  ModularEncodingOptions options;
  options.search = true;
  options.learning = {128, 32};
  options.tree_checkpoints = true;
  const auto plan = Plan(f, options);
  std::vector<uint8_t> expected, output;
  Ok(EncodeModularImage(f.view, options, &expected));
  NoCapability absent;
  FaultCapability restricted(gpu);
  restricted.limited = true;
  for (GpuBackend *backend :
       {static_cast<GpuBackend *>(&absent), static_cast<GpuBackend *>(&restricted)}) {
    Ok(ExecutionDomain::Create({plan.peak_bytes, 1}, &options.execution_domain));
    options.adaptive_gpu = backend;
    ModularProfile profile;
    const auto before = gpu.stats();
    {
      ProfileSession session(&profile);
      Ok(EncodeModularImage(f.view, options, &output));
    }
    Check(output == expected &&
              profile.adaptive_gpu.cpu_fallback_candidates == profile.encoded_candidates &&
              profile.encoded_candidates > 1 && profile.adaptive_gpu.encoded_candidates == 0 &&
              profile.adaptive_gpu.feature_preparations == 0 &&
              gpu.stats().successful_allocations == before.successful_allocations &&
              gpu.stats().committed_submissions == before.committed_submissions,
          "Capability fallback changed bytes or touched the GPU");
    profile.resolved = {};
    Empty(*options.execution_domain);
  }
  Check(!restricted.preparations && !restricted.generations,
        "Device limit fallback reached the operation");
  options.adaptive_gpu = &gpu;
  Ok(ExecutionDomain::Create({plan.peak_bytes - 1, 1}, &options.execution_domain));
  output = {7, 8, 9};
  const auto before = gpu.stats();
  ArmNextManagedHostAllocationFailureForTest();
  const auto status = EncodeModularImage(f.view, options, &output);
  const bool pending = ManagedHostAllocationFailurePendingForTest();
  DisarmManagedHostAllocationFailureForTest();
  Check(status.code() == StatusCode::kOutOfMemory && pending &&
            output == std::vector<uint8_t>({7, 8, 9}) &&
            gpu.stats().successful_allocations == before.successful_allocations,
        "One-byte-short admission allocated work or published output");
  Empty(*options.execution_domain);
  Ok(ExecutionDomain::Create({plan.peak_bytes, 1}, &options.execution_domain));
  for (unsigned kind = 0; kind < 3; ++kind) {
    FaultCapability failing(gpu);
    failing.fail_preparation = kind == 1 ? 0 : 2;
    failing.fail_generation = kind == 1 ? 3 : 0;
    failing.unsupported_after_preflight = kind == 2;
    options.adaptive_gpu = &failing;
    ModularProfile profile;
    Status failed;
    {
      ProfileSession session(&profile);
      failed = EncodeModularImage(f.view, options, &output);
    }
    Check(failed.code() == (kind == 2 ? StatusCode::kUnsupported : StatusCode::kDeviceError) &&
              output == std::vector<uint8_t>({7, 8, 9}) &&
              profile.adaptive_gpu.encoded_candidates > 0 &&
              profile.adaptive_gpu.cpu_fallback_candidates == 0,
          "Later adaptive failure silently fell back or replaced old output");
    profile.resolved = {};
    Empty(*options.execution_domain);
    ++fault_checks;
  }
  for (unsigned kind = 0; kind < 3; ++kind) {
    options.adaptive_gpu = &gpu;
    if (kind == 0)
      Ok(ArmNextMetalAllocationFailureForTest(gpu));
    else
      Ok(ArmNextMetalSubmissionFailureForTest(gpu, kind == 1, kind == 2));
    const auto failed = EncodeModularImage(f.view, options, &output);
    Check(failed.code() == (kind == 0   ? StatusCode::kOutOfMemory
                            : kind == 1 ? StatusCode::kSubmissionFailed
                                        : StatusCode::kDeviceError) &&
              output == std::vector<uint8_t>({7, 8, 9}),
          "Native adaptive failure was not atomic");
    Empty(*options.execution_domain);
    ++fault_checks;
  }
}
void AnsFallback(GpuBackend &gpu) {
  Fixture f(PackedModularFormat::kRgb16, {19, 17});
  ModularEncodingOptions options;
  options.entropy = EntropyCodingMode::kAns;
  std::vector<uint8_t> cpu, actual{7, 8, 9};
  Ok(EncodeModularImage(f.view, options, &cpu));
  FaultCapability tokens_only(gpu);
  options.adaptive_gpu = &tokens_only;
  options.adaptive_ans_cost = true;
  const auto plan = Plan(f, options);
  Ok(ExecutionDomain::Create({plan.peak_bytes, 1}, &options.execution_domain));
  ModularProfile profile;
  const auto before = gpu.stats();
  {
    ProfileSession session(&profile);
    Ok(EncodeModularImage(f.view, options, &actual));
  }
  Check(actual == cpu && profile.adaptive_gpu.ans_gpu_groups == 0 &&
            profile.adaptive_gpu.ans_cpu_fallback_groups > 0 &&
            gpu.stats().committed_submissions == before.committed_submissions + 1,
        "Missing ANS capability did not use the explicit scalar cost fallback");
  profile.resolved = {};
  Empty(*options.execution_domain);
  options.adaptive_gpu = &gpu;
  Ok(ExecutionDomain::Create({plan.peak_bytes - 1, 1}, &options.execution_domain));
  actual = {7, 8, 9};
  ArmNextManagedHostAllocationFailureForTest();
  const auto status = EncodeModularImage(f.view, options, &actual);
  const bool pending = ManagedHostAllocationFailurePendingForTest();
  DisarmManagedHostAllocationFailureForTest();
  Check(status.code() == StatusCode::kOutOfMemory && pending &&
            actual == std::vector<uint8_t>({7, 8, 9}),
        "ANS-inclusive admission allocated or published with one byte too little");
  Empty(*options.execution_domain);
  options.execution_domain.reset();
  options.adaptive_gpu = nullptr;
  Check(EncodeModularImage(f.view, options, &actual).code() == StatusCode::kInvalidArgument &&
            actual == std::vector<uint8_t>({7, 8, 9}),
        "ANS switch without adaptive GPU was not rejected");
}
void TreeFallbackAndAdmission(GpuBackend &gpu) {
  Fixture f(PackedModularFormat::kGray8, {64, 64});
  ModularEncodingOptions options;
  options.search = true;
  options.learning = {4096, 16};
  std::vector<uint8_t> expected, actual;
  Ok(EncodeModularImage(f.view, options, &expected));
  FaultCapability tokens_only(gpu);
  options.adaptive_gpu = &tokens_only;
  options.adaptive_tree_statistics = true;
  const auto plan = Plan(f, options);
  Ok(ExecutionDomain::Create({plan.peak_bytes, 1}, &options.execution_domain));
  ModularProfile profile;
  {
    ProfileSession session(&profile);
    Ok(EncodeModularImage(f.view, options, &actual));
  }
  Check(actual == expected && profile.adaptive_gpu.tree_gpu_groups == 0 &&
            profile.adaptive_gpu.tree_gpu_preparations == 0 &&
            profile.adaptive_gpu.tree_cpu_fallback_properties > 0 &&
            profile.adaptive_gpu.tree_cpu_small_properties == 0,
        "Missing learner statistics capability did not use explicit scalar fallback");
  profile.resolved = {};
  Empty(*options.execution_domain);
  options.adaptive_gpu = &gpu;
  Ok(ExecutionDomain::Create({plan.peak_bytes - 1, 1}, &options.execution_domain));
  actual = {7, 8, 9};
  const auto before = gpu.stats();
  ArmNextManagedHostAllocationFailureForTest();
  const auto status = EncodeModularImage(f.view, options, &actual);
  const bool pending = ManagedHostAllocationFailurePendingForTest();
  DisarmManagedHostAllocationFailureForTest();
  Check(status.code() == StatusCode::kOutOfMemory && pending && actual == std::vector<uint8_t>({7, 8, 9}) &&
            gpu.stats().successful_allocations == before.successful_allocations &&
            gpu.stats().committed_submissions == before.committed_submissions,
        "Statistics-inclusive admission allocated, submitted or published below the bound");
  Empty(*options.execution_domain);
  options.execution_domain.reset();
  options.adaptive_gpu = nullptr;
  Check(EncodeModularImage(f.view, options, &actual).code() == StatusCode::kInvalidArgument &&
            actual == std::vector<uint8_t>({7, 8, 9}), "Statistics switch without GPU was accepted");
  options.adaptive_gpu = &gpu;
  options.search = false;
  Check(EncodeModularImage(f.view, options, &actual).code() == StatusCode::kInvalidArgument &&
            actual == std::vector<uint8_t>({7, 8, 9}), "Statistics switch without search was accepted");
  ++fault_checks;
}
void TreeAttemptFailure() {
  using namespace tree_gpu_internal;
  struct FailingStatistics final : NoCapability, GpuModularTreeStatistics {
    bool unsupported = false;
    size_t calls = 0;
    TreeStatisticsDeviceLimits tree_statistics_limits() const noexcept override {
      return {kMaximumTreeStatisticsSamples, kMaximumTreeStatisticsSamples, SIZE_MAX};
    }
    Status PrepareTreeStatistics(const TreeStatisticsPreparationInput &,
                                 std::unique_ptr<PreparedGpuTreeStatistics> *) override {
      Check(CurrentResourceContext().reservation && thread_budget_internal::HasCpuParticipation(),
            "Learner statistics attempt started without admission");
      ++calls;
      return unsupported ? Status::Unsupported("Injected post-preflight statistics rejection")
                         : Status::DeviceError("Injected statistics preparation failure");
    }
    Status MeasureTreeSplitCosts(const TreeStatisticsInput &,
                                 std::unique_ptr<PreparedGpuTreeSplitCosts> *) override {
      throw std::runtime_error("Failed statistics preparation reached measurement");
    }
  } backend;
  Fixture f(PackedModularFormat::kGray8, {128, 128});
  ModularEncodingOptions options;
  options.search = true;
  options.learning = {16384, 32};
  options.tree_checkpoints = true;
  options.adaptive_gpu = &backend;
  options.adaptive_tree_statistics = true;
  const auto plan = Plan(f, options);
  Ok(ExecutionDomain::Create({plan.peak_bytes, 1}, &options.execution_domain));
  for (bool unsupported : {false, true}) {
    backend.unsupported = unsupported;
    backend.calls = 0;
    std::vector<uint8_t> output{7, 8, 9};
    ModularProfile profile;
    Status status;
    {
      ProfileSession session(&profile);
      status = EncodeModularImage(f.view, options, &output);
    }
    Check(status.code() == (unsupported ? StatusCode::kUnsupported : StatusCode::kDeviceError) &&
              backend.calls == 1 && output == std::vector<uint8_t>({7, 8, 9}) &&
              profile.adaptive_gpu.tree_gpu_preparations == 0 &&
              profile.adaptive_gpu.tree_gpu_groups == 0,
          "Complete search retried failed statistics or replaced old output");
    profile.resolved = {};
    Empty(*options.execution_domain);
    ++fault_checks;
  }
}
void AllocationSweep(GpuBackend &gpu, bool ans_cost = false) {
  Fixture f(PackedModularFormat::kGray8, {1, 1});
  ModularEncodingOptions options;
  options.adaptive_gpu = &gpu;
  options.adaptive_ans_cost = ans_cost;
  if (ans_cost)
    options.entropy = EntropyCodingMode::kAns;
  options.search = true;
  options.tree_checkpoints = true;
  options.learning = {32, 32};
  const auto plan = Plan(f, options);
  Ok(ExecutionDomain::Create({plan.peak_bytes, 1}, &options.execution_domain));
  bool finished = false;
  for (size_t at = 0; at < 20000; ++at) {
    std::vector<uint8_t> output{7, 8, 9};
    ArmManagedHostAllocationFailureAfterForTest(at);
    const auto status = EncodeModularImage(f.view, options, &output);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Empty(*options.execution_domain);
    if (pending) {
      Ok(status);
      finished = true;
      break;
    }
    Check(status.code() == StatusCode::kOutOfMemory && output == std::vector<uint8_t>({7, 8, 9}),
          "Adaptive search host allocation failure changed old output");
    ++fault_checks;
  }
  Check(finished, "Adaptive complete-search allocation sweep did not finish");
}
void SameAddress(GpuBackend &gpu) {
  Fixture f(PackedModularFormat::kGray8, {17, 3});
  ModularEncodingOptions options;
  options.adaptive_gpu = &gpu;
  std::vector<uint8_t> old, actual, expected;
  Ok(EncodeModularImage(f.view, options, &old));
  f.bytes[1] ^= 0xff;
  Ok(EncodeModularImage(f.view, options, &actual));
  options.adaptive_gpu = nullptr;
  Ok(EncodeModularImage(f.view, options, &expected));
  Check(actual == expected && actual != old, "Adaptive features escaped the encoder-call lifetime");
}
void CacheKeys(GpuBackend &gpu) {
  Fixture f(PackedModularFormat::kRgb16, {17, 19});
  std::array<ModularCodingPolicy, 5> policies;
  policies[0].rct = 6;
  policies[0].tree.nodes[0].predictor = Predictor::kWeighted;
  policies[1] = policies[0];
  policies[1].tree.nodes[0].offset = 13;
  policies[2] = policies[1];
  policies[2].weighted.coefficients[3] = 17;
  policies[3] = policies[2];
  policies[3].transforms.size = 1;
  policies[3].transforms.entries[0] = {TransformKind::kSqueeze, 0, 3, 1, true, false};
  policies[4] = policies[0];
  std::array<AdaptiveModularWorkflowStoragePlan, 5> plans;
  std::array<std::vector<uint8_t>, 5> expected;
  size_t peak = 0;
  for (size_t i = 0; i < policies.size(); ++i) {
    ModularEncodingOptions options;
    options.coding = policies[i];
    options.entropy = i % 2 ? EntropyCodingMode::kAns : EntropyCodingMode::kPrefix;
    plans[i] = Plan(f, options);
    peak = std::max(peak, plans[i].peak_bytes);
    Ok(EncodeModularImage(f.view, options, &expected[i]));
  }
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({peak, 1}, &domain));
  ModularProfile profile;
  {
    codestream_internal::WorkflowAdmission admission;
    Ok(admission.Start(peak, domain));
    thread_budget_internal::CpuExecutionScope execution;
    Ok(execution.Start(domain, 1));
    const ManagedHostScope managed(ResourceClass::kPreparation);
    ProfileSession session(&profile);
    AdaptiveModularTokenCache cache(gpu, f.view);
    for (size_t i = 0; i < policies.size(); ++i) {
      codestream_internal::CodestreamBuffer actual;
      Ok(cache.Encode(policies[i], i % 2 ? EntropyCodingMode::kAns : EntropyCodingMode::kPrefix,
                      plans[i].cpu, &actual));
      Check(std::ranges::equal(actual.view(), expected[i]),
            "Adaptive cache key omitted a dependency");
    }
  }
  Check(profile.adaptive_gpu.encoded_candidates == 5 &&
            profile.adaptive_gpu.feature_preparations == 4 &&
            profile.adaptive_gpu.feature_reuses == 1 && profile.adaptive_gpu.evictions == 3,
        "Adaptive cache did not exclude tree/entropy or failed to evict changed "
        "weighted/transformed inputs");
  Empty(*domain);
}
} // namespace
int main(int argc, char **argv) try {
  artifacts = argc > 1 ? argv[1] : "metal-modular-adaptive-workflow";
  std::filesystem::create_directories(artifacts);
  std::unique_ptr<GpuBackend> gpu;
  Ok(CreateEmbeddedMetalBackend({}, &gpu));
  for (auto format :
       {PackedModularFormat::kGray8, PackedModularFormat::kRgb8, PackedModularFormat::kRgba8,
        PackedModularFormat::kGray16, PackedModularFormat::kRgb16, PackedModularFormat::kRgba16})
    for (bool palette : {false, true})
      for (auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
        Fixture f(format, {33, 17}, palette);
        ModularEncodingOptions options;
        options.entropy = mode;
        options.search = true;
        options.learning = {1024, 32};
        options.tree_checkpoints = true;
        options.cpu_thread_count = palette ? 4 : 1;
        Case(*gpu, f, options, format == PackedModularFormat::kGray8 && palette);
      }
  Fixture grouped(PackedModularFormat::kRgba16, {257, 17});
  ModularEncodingOptions prescribed;
  prescribed.cpu_thread_count = 4;
  prescribed.entropy = EntropyCodingMode::kAns;
  prescribed.coding.rct = 6;
  prescribed.coding.transforms.size = 1;
  prescribed.coding.transforms.entries[0] = {TransformKind::kSqueeze, 0, 4, 1, true, false};
  prescribed.coding.tree.nodes[0].predictor = Predictor::kWeighted;
  Case(*gpu, grouped, prescribed);
  for (auto format :
       {PackedModularFormat::kGray8, PackedModularFormat::kRgb8, PackedModularFormat::kRgba8,
        PackedModularFormat::kGray16, PackedModularFormat::kRgb16, PackedModularFormat::kRgba16}) {
    for (bool palette : {false, true}) {
      Fixture f(format, {13, 11}, palette);
      ModularEncodingOptions options;
      options.search = true;
      options.entropy = EntropyCodingMode::kAns;
      options.tree_checkpoints = true;
      options.learning = {512, 32};
      options.cpu_thread_count = palette ? 4 : 1;
      Case(*gpu, f, options, format == PackedModularFormat::kGray16 && palette, true);
    }
  }
  Case(*gpu, grouped, prescribed, true, true);
  prescribed.entropy = EntropyCodingMode::kPrefix;
  Case(*gpu, grouped, prescribed, false, true);
  {
    Fixture large(PackedModularFormat::kRgb8, {2048, 2048});
    ModularEncodingOptions options;
    options.entropy = EntropyCodingMode::kAns;
    options.coding.tree.nodes[0].predictor = Predictor::kWeighted;
    Case(*gpu, large, options, false, true, true);
  }
  for (auto format : {PackedModularFormat::kGray8, PackedModularFormat::kRgb8,
                      PackedModularFormat::kRgba8, PackedModularFormat::kGray16,
                      PackedModularFormat::kRgb16, PackedModularFormat::kRgba16}) {
    Fixture f(format, {192, 128});
    ModularEncodingOptions options;
    options.search = true;
    options.entropy = EntropyCodingMode::kAns;
    options.learning = {32768, 32};
    options.tree_checkpoints = true;
    options.cpu_thread_count = format == PackedModularFormat::kRgba16 ? 4 : 1;
    Case(*gpu, f, options, format == PackedModularFormat::kGray8, true, false, true, true);
  }
  {
    Fixture f(PackedModularFormat::kRgb8, {192, 128}, true);
    ModularEncodingOptions options;
    options.search = true;
    options.entropy = EntropyCodingMode::kPrefix;
    options.learning = {32768, 32, TreeLearningMode::kQuantized};
    options.tree_checkpoints = true;
    // The retained original-learner baseline has only 4096 samples, below the
    // measured device crossover; expanded quantized learning stays scalar too.
    Case(*gpu, f, options, false, false, false, true, false);
    Fixture small(PackedModularFormat::kGray8, {17, 13});
    options.learning = {512, 32};
    Case(*gpu, small, options, false, false, false, true, false);
  }
  FallbacksAndFailures(*gpu);
  AnsFallback(*gpu);
  TreeFallbackAndAdmission(*gpu);
  TreeAttemptFailure();
  AllocationSweep(*gpu);
  AllocationSweep(*gpu, true);
  SameAddress(*gpu);
  CacheKeys(*gpu);
  gpu.reset();
  Check(DefaultResourceBudget().snapshot().total.live_capacity_bytes == 0 &&
            DefaultResourceBudget().snapshot().total.pending_count == 0,
        "Adaptive workflow leaked global backing");
  std::cout << cases << " complete CPU/hybrid matches, " << decoded_samples
            << " independent decoded samples, " << fault_checks << " failure-atomic checks\n";
  return 0;
} catch (const std::exception &e) {
  DisarmManagedHostAllocationFailureForTest();
  std::cerr << "Adaptive workflow qualification failed: " << e.what() << '\n';
  return 1;
}
