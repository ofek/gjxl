// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/profile.h"
#include "codestream/modular/adaptive_gpu_workflow.h"
#include "codestream/workflow_admission_scope.h"
#include "codestream/workflow_admission_test.h"
#include "core/cpu_execution.h"
#include "core/worker_launch_internal.h"
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
std::string context;
WeightedFeatureBackend workflow_feature_backend = WeightedFeatureBackend::kCpu;
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(const Status &s) {
  if (!s.ok())
    throw std::runtime_error(context + ": " + std::string(s.message()));
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
  options.weighted_feature_backend = workflow_feature_backend;
  const bool shared = options.adaptive_shared_features;
  options.adaptive_gpu = &gpu;
  const auto plan = Plan(f, options);
  options.adaptive_ans_cost = false;
  options.adaptive_tree_statistics = false;
  options.adaptive_shared_features = false;
  options.adaptive_gpu = nullptr;
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
  options.adaptive_shared_features = shared;
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
                actual.encoded_candidates + actual.adaptive_gpu.ans_gpu_groups + actual.adaptive_gpu.tree_gpu_groups +
                    actual.adaptive_gpu.feature_preparation.gpu_calls,
        "Hybrid search changed bytes, selected policy, candidate count, or silently fell back");
  const auto &g = actual.adaptive_gpu;
  if (options.entropy == EntropyCodingMode::kAns) {
    for (const auto *p : {&baseline, &actual})
      Check(p->entropy_work.ans_value_collection_nanoseconds == 0 &&
                p->entropy_work.ans_value_aggregation_nanoseconds == 0 &&
                p->entropy_work.prefix_section_measurement_count == 0,
            "Modular model construction repeated value collection or measured prefix sections");
  }
  Check(g.ans_cpu_fallback_groups == g.ans_cpu_small_groups &&
            (ans_cost && options.entropy == EntropyCodingMode::kAns &&
                 options.ans_selection == AnsSelectionMode::kExact
                 ? g.ans_gpu_groups + g.ans_cpu_fallback_groups >= actual.encoded_candidates &&
                       g.ans_gpu_states >= g.ans_gpu_groups
                 : g.ans_gpu_groups == 0 && g.ans_gpu_states == 0 &&
                       g.ans_cpu_fallback_groups == 0) &&
            (!require_gpu_cost || (g.ans_gpu_groups > 0 && g.ans_cpu_fallback_groups == 0)),
        "Exact ANS cost route was omitted or fallback attribution differs");
  if (ans_cost && options.entropy == EntropyCodingMode::kAns && !require_gpu_cost &&
      options.ans_selection == AnsSelectionMode::kExact)
    Check(g.ans_gpu_groups == 0 && g.ans_cpu_small_groups > 0,
          "Small workflow did not use the explicit scalar cost crossover");
  if (options.entropy == EntropyCodingMode::kAns &&
      options.ans_selection == AnsSelectionMode::kEstimated)
    Check(actual.entropy_work.ans_token_cost_nanoseconds == 0 &&
              baseline.entropy_work.ans_token_cost_nanoseconds == 0 &&
              actual.entropy_work.ans_alphabet_width_candidate_count == actual.encoded_candidates,
          "Estimated hybrid performed exact simulation or built extra widths");
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
  Check(g.feature_preparations + g.feature_reuses == g.encoded_candidates + g.learning_feature_uses &&
            g.evictions + 1 == g.feature_preparations && g.feature_seconds > 0 &&
            g.token_seconds > 0 &&
            g.feature_seconds - g.learning_feature_seconds + g.token_seconds <=
                actual.seconds[static_cast<size_t>(ProfileStage::kTokens)] + 1e-6 &&
            g.learning_feature_seconds <= actual.seconds[static_cast<size_t>(ProfileStage::kTraining)] + 1e-6 &&
            baseline.adaptive_gpu == AdaptiveGpuWorkProfile{},
        "Adaptive profile has missing work or overlapping intervals");
  const auto &detail = g.feature_preparation;
  const double preparation = detail.allocation_seconds + detail.copy_seconds + detail.scratch_seconds +
                             detail.prediction_seconds + detail.publication_seconds;
  // A short publication can fit within one steady-clock tick.
  Check(detail.allocation_seconds > 0 && detail.copy_seconds > 0 && detail.publication_seconds >= 0 &&
            detail.scratch_seconds >= 0 && detail.prediction_seconds > 0 &&
            preparation <= g.feature_seconds + 1e-6,
        "Feature preparation details are missing or overlap their parent");
  Check(detail.slice_jobs >= g.feature_preparations, "Weighted job count omitted a feature preparation");
  Check((detail.peak_participants > 0 || detail.gpu_calls > 0) && detail.peak_participants <= plan.cpu.participants,
        "Weighted participants exceeded whole-workflow admission");
  if (options.search) {
    const auto &a = actual.training_work, &b = baseline.training_work;
    Check(g.feature_reuses > 0 && a.learning_calls == b.learning_calls &&
              a.population == b.population && a.samples == b.samples &&
              a.split_evaluations == b.split_evaluations &&
              a.sorted_properties == b.sorted_properties && a.sorted_samples == b.sorted_samples,
          "Adaptive search changed training work or did not reuse features");
    Check(b.weighted_prediction_samples == b.population && b.reused_prediction_samples == 0 &&
              a.weighted_prediction_samples == (shared ? 0 : a.population) &&
              a.reused_prediction_samples == (shared ? a.population : 0) &&
              g.learning_feature_uses == (shared ? a.learning_calls : 0) &&
              g.learning_feature_fallbacks == 0,
          "Shared learning recomputed weighted history or omitted reuse attribution");
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
  struct BorrowFault final : PreparedGpuAdaptiveModularFeatures {
    std::unique_ptr<PreparedGpuAdaptiveModularFeatures> owner;
    unsigned kind;
    BorrowFault(std::unique_ptr<PreparedGpuAdaptiveModularFeatures> value, unsigned failure)
        : owner(std::move(value)), kind(failure) {}
    BackendId backend_id() const noexcept override { return owner->backend_id(); }
    const codec_internal::ImageMetadata &metadata() const noexcept override { return owner->metadata(); }
    size_t token_count() const noexcept override { return owner->token_count(); }
    std::span<const AdaptiveModularTokenSlice> slices() const noexcept override { return owner->slices(); }
    bool has_weighted_features() const noexcept override { return owner->has_weighted_features(); }
    Status BorrowWeightedHostView(std::span<const AdaptiveModularWeightedFeature> *out) const override {
      if (kind == 1) return Status::DeviceError("Injected host feature borrow failure");
      if (kind == 2) return Status::Unsupported("Injected host feature borrowing unsupported");
      if (auto s = owner->BorrowWeightedHostView(out); !s.ok()) return s;
      *out = out->first(out->size() - 1);
      return Status::Ok();
    }
  };
  explicit FaultCapability(GpuBackend &gpu)
      : native(*QueryGpuAdaptiveModularTokenGeneration(gpu)) {}
  GpuAdaptiveModularTokenGeneration &native;
  size_t preparations = 0, generations = 0, fail_preparation = 0, fail_generation = 0;
  bool limited = false, unsupported_after_preflight = false;
  unsigned borrow_failure = 0;
  ModularTokenDeviceLimits adaptive_modular_limits() const noexcept override {
    auto limits = native.adaptive_modular_limits();
    if (limited)
      limits.maximum_buffer_bytes = 1;
    return limits;
  }
  Status PrepareAdaptiveModularFeatures(
      const ModularEncoderFrame &frame, std::span<const AdaptiveModularTokenSlice> slices,
      const WeightedPredictorParameters &parameters, bool weighted,
      std::unique_ptr<PreparedGpuAdaptiveModularFeatures> *out,
      size_t maximum_cpu_participants = 1,
      WeightedFeatureBackend feature_backend = WeightedFeatureBackend::kCpu) override {
    Check(CurrentResourceContext().reservation && thread_budget_internal::HasCpuParticipation(),
          "Adaptive feature work started without admission");
    if (++preparations == fail_preparation)
      return unsupported_after_preflight
                 ? Status::Unsupported("Injected numeric feature failure")
                 : Status::DeviceError("Injected replacement feature failure");
    if (auto s = native.PrepareAdaptiveModularFeatures(frame, slices, parameters, weighted, out,
                                                      maximum_cpu_participants, feature_backend); !s.ok())
      return s;
    if (borrow_failure) *out = std::make_unique<BorrowFault>(std::move(*out), borrow_failure);
    return Status::Ok();
  }
  Status GenerateAdaptiveModularTokens(const PreparedGpuAdaptiveModularFeatures &features,
                                       const DecisionTree &tree,
                                       std::unique_ptr<PreparedGpuModularTokens> *out) override {
    if (++generations == fail_generation)
      return Status::DeviceError("Injected later candidate failure");
    const auto *wrapped = dynamic_cast<const BorrowFault *>(&features);
    return native.GenerateAdaptiveModularTokens(wrapped ? *wrapped->owner : features, tree, out);
  }
};
void FallbacksAndFailures(GpuBackend &gpu, bool shared = false) {
  Fixture f(PackedModularFormat::kRgb8, {9, 7});
  ModularEncodingOptions options;
  options.search = true;
  options.learning = {128, 32};
  options.tree_checkpoints = true;
  const auto plan = Plan(f, options);
  std::vector<uint8_t> expected, output;
  Ok(EncodeModularImage(f.view, options, &expected));
  options.adaptive_shared_features = shared;
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
              profile.adaptive_gpu.learning_feature_fallbacks ==
                  (shared ? profile.training_work.learning_calls : 0) &&
              profile.training_work.weighted_prediction_samples == profile.training_work.population &&
              profile.training_work.reused_prediction_samples == 0 &&
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
void AllocationSweep(GpuBackend &gpu, bool ans_cost = false,
                     AnsSelectionMode selection = AnsSelectionMode::kExact, bool shared = false) {
  Fixture f(PackedModularFormat::kGray8, {1, 1});
  ModularEncodingOptions options;
  options.adaptive_gpu = &gpu;
  options.adaptive_ans_cost = ans_cost;
  options.ans_selection = selection;
  options.adaptive_shared_features = shared;
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
void SharedLearning(GpuBackend &gpu, PackedModularFormat format, bool palette) {
  context = "shared learning format " + std::to_string(int(format)) + (palette ? " palette" : " pattern");
  Fixture f(format, {257, 19}, palette);
  ModularEncodingOptions options;
  options.search = true;
  options.tree_checkpoints = true;
  options.learning = {4096, 32};
  options.adaptive_gpu = &gpu;
  options.adaptive_shared_features = false;
  const auto control_plan = Plan(f, options);
  options.adaptive_shared_features = true;
  const auto plan = Plan(f, options);
  Check(plan == control_plan, "Shared feature ownership increased the whole-call reservation");
  options.learning.mode = TreeLearningMode::kQuantized;
  const auto quantized_plan = Plan(f, options);
  options.adaptive_shared_features = false;
  Check(quantized_plan == Plan(f, options), "Shared quantized feature ownership increased the reservation");
  const size_t peak = std::max(plan.peak_bytes, quantized_plan.peak_bytes);
  std::array<ModularCodingPolicy, 4> transforms;
  uint16_t colors;
  Ok(ProbePalette(f.view, &colors));
  size_t count;
  Ok(BuildTransformCandidates(f.view.extent, format, colors, &transforms, &count));
  std::vector<ModularCodingPolicy> policies(1);
  for (size_t i = 0; i < count; ++i) policies.push_back(transforms[i]);
  auto custom = policies.back();
  custom.weighted.coefficients = {3, 29, 5, 17, 11, 2, 1};
  custom.weighted.weights = {0, 15, 7, 1};
  policies.push_back(custom);
  policies.push_back({}); // Revisit an evicted layout, not a persistent second slot.
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({peak, 1}, &domain));
  ModularProfile profile;
  {
    codestream_internal::WorkflowAdmission admission;
    Ok(admission.Start(peak, domain));
    thread_budget_internal::CpuExecutionScope execution;
    Ok(execution.Start(domain, 1));
    ManagedHostScope managed(ResourceClass::kPreparation);
    AdaptiveModularTokenCache cache(gpu, f.view);
    for (const auto &base : policies)
      for (auto mode : {TreeLearningMode::kLegacy, TreeLearningMode::kQuantized}) {
        ModularCodingPolicy single, split, entropy, expected_single, expected_split, expected_entropy;
        TrainingCoverage coverage, expected_coverage;
        TreeLearningCheckpoints checkpoints, expected_checkpoints;
        const TreeLearningOptions learning{4096, 32, mode};
        {
          ModularEncoderFrame frame;
          Ok(ModularEncoderFrame::Prepare(f.view, base, &frame));
          ChannelShape shape;
          Ok(DescribeImage(frame.image(), &shape));
          ModularStreamPlan layout;
          Ok(BuildModularStreamPlan(plan.cpu.geometry, shape.channels(), shape.metadata, &layout));
          Ok(LearnModularPolicies(frame, layout, base, &expected_single, &expected_split,
                                  &expected_entropy, learning, &expected_coverage, &expected_checkpoints));
          // A malformed borrowed map must fail before publishing any policies.
          const WeightedPredictorFeature feature;
          Check(LearnModularPolicies(frame, layout, base, &single, &split, &entropy, learning,
                                     nullptr, nullptr, nullptr, std::span(&feature, 1)).code() ==
                    StatusCode::kInvalidArgument && single == ModularCodingPolicy{} &&
                    split == ModularCodingPolicy{} && entropy == ModularCodingPolicy{},
                "Incomplete learning feature map was accepted or published output");
        }
        {
          ProfileSession session(&profile);
          Ok(cache.Learn(base, &single, &split, &entropy, learning, &coverage, &checkpoints));
        }
        Check(single == expected_single && split == expected_split && entropy == expected_entropy &&
                  coverage == expected_coverage && checkpoints == expected_checkpoints,
              "Shared learning changed sampled features, policies or growth checkpoints");
      }
  }
  const auto &g = profile.adaptive_gpu;
  Check(g.feature_preparations == policies.size() && g.feature_reuses == policies.size() &&
            g.evictions + 1 == g.feature_preparations &&
            g.learning_feature_uses == 2 * policies.size() && g.learning_feature_fallbacks == 0 &&
            profile.training_work.weighted_prediction_samples == 0 &&
            profile.training_work.reused_prediction_samples == profile.training_work.population,
        "Shared learning did not reuse one layout across budgets/modes or evict changed keys");
  Empty(*domain);
}
void SharedFeatureAdmission(GpuBackend &gpu) {
  Fixture f(PackedModularFormat::kGray8, {9, 7});
  for (GpuBackend *backend : {static_cast<GpuBackend *>(nullptr), &gpu})
    for (bool search : {false, true}) {
      ModularEncodingOptions defaults;
      Check(defaults.adaptive_shared_features, "Shared learning features are not the default");
      defaults.adaptive_gpu = backend;
      defaults.search = search;
      defaults.learning = {32, 32};
      std::vector<uint8_t> actual, control;
      ModularProfile profile;
      {
        ProfileSession session(&profile);
        Ok(EncodeModularImage(f.view, defaults, &actual));
      }
      defaults.adaptive_shared_features = false;
      Ok(EncodeModularImage(f.view, defaults, &control));
      Check(actual == control && profile.adaptive_gpu.learning_feature_fallbacks == 0 &&
                profile.adaptive_gpu.learning_feature_uses ==
                    (backend && search ? profile.training_work.learning_calls : 0) &&
                profile.training_work.reused_prediction_samples ==
                    (backend && search ? profile.training_work.population : 0),
            "Default sharing changed bytes or escaped the adaptive GPU search route");
    }
  ModularEncodingOptions options;
  std::vector<uint8_t> output{7, 8, 9};
  options.adaptive_gpu = &gpu;
  options.search = true;
  const auto plan = Plan(f, options);
  Ok(ExecutionDomain::Create({plan.peak_bytes, 1}, &options.execution_domain));
  for (unsigned kind = 1; kind <= 3; ++kind) {
    FaultCapability faulty(gpu);
    faulty.borrow_failure = kind;
    options.adaptive_gpu = &faulty;
    ModularProfile profile;
    Status status;
    {
      ProfileSession session(&profile);
      status = EncodeModularImage(f.view, options, &output);
    }
    Check(status.code() == (kind == 1 ? StatusCode::kDeviceError :
                            kind == 2 ? StatusCode::kUnsupported : StatusCode::kInternal) &&
              output == std::vector<uint8_t>({7, 8, 9}) && faulty.preparations == 1 &&
              faulty.generations == 1 && profile.training_work.learning_calls == 0 &&
              profile.adaptive_gpu.learning_feature_fallbacks == 0,
          "Feature borrowing failure retried learning or published output");
    profile.resolved = {};
    Empty(*options.execution_domain);
    ++fault_checks;
  }
}
void ParallelFeatureFailures(GpuBackend &gpu) {
  using namespace thread_budget_internal;
  Fixture f(PackedModularFormat::kRgb16, {257, 17});
  ModularEncodingOptions options;
  options.adaptive_gpu = &gpu;
  options.cpu_thread_count = 4;
  options.coding.tree.nodes[0].predictor = Predictor::kWeighted;
  const auto plan = Plan(f, options);
  for (auto kind : {WorkerLaunchFailureKind::kSystemError, WorkerLaunchFailureKind::kBadAlloc})
    for (size_t before : {size_t{0}, size_t{1}, size_t{2}}) {
      Ok(ExecutionDomain::Create({plan.peak_bytes, 4}, &options.execution_domain));
      std::vector<uint8_t> output{7, 8, 9};
      const auto original = output;
      WorkerLaunchFaultForTesting fault{WorkerLaunchSite::kModularWeightedFeatures, before, kind};
      Status status;
      {
        WorkerLaunchFaultScopeForTesting inject(&fault);
        status = EncodeModularImage(f.view, options, &output);
      }
      Check(fault.triggered && fault.launched_in_group == before && output == original &&
                status.code() == (kind == WorkerLaunchFailureKind::kBadAlloc
                                    ? StatusCode::kOutOfMemory : StatusCode::kInternal),
            "Feature worker launch failure published output or retried encoding");
      Empty(*options.execution_domain);
      ++fault_checks;
    }
}
} // namespace
int main(int argc, char **argv) try {
  artifacts = argc > 1 ? argv[1] : "metal-modular-adaptive-workflow";
  std::filesystem::create_directories(artifacts);
  if (argc > 2 && std::string_view(argv[2]) == "--gpu-features")
    workflow_feature_backend = WeightedFeatureBackend::kGpu;
  std::unique_ptr<GpuBackend> gpu;
  Ok(CreateEmbeddedMetalBackend({}, &gpu));
  ParallelFeatureFailures(*gpu);
  if (argc > 2 && std::string_view(argv[2]) == "--shared-features") {
    SharedFeatureAdmission(*gpu);
    for (auto format : {PackedModularFormat::kGray8, PackedModularFormat::kRgb8,
                        PackedModularFormat::kRgba8, PackedModularFormat::kGray16,
                        PackedModularFormat::kRgb16, PackedModularFormat::kRgba16})
      for (bool palette : {false, true}) SharedLearning(*gpu, format, palette);
    Check(DefaultResourceBudget().snapshot().total.live_capacity_bytes == 0,
          "Shared learning qualification leaked global backing");
    std::cout << "Shared sample fingerprints, learned policies, checkpoints, cache keys and borrow failures match\n";
    return 0;
  }
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
    options.ans_selection = AnsSelectionMode::kEstimated;
    Case(*gpu, large, options, true, true);
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
    options.ans_selection = AnsSelectionMode::kEstimated;
    Case(*gpu, f, options, format == PackedModularFormat::kRgba16, true, false, true, true);
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
  context = "workflow fallback and failures";
  FallbacksAndFailures(*gpu);
  FallbacksAndFailures(*gpu, true);
  AnsFallback(*gpu);
  TreeFallbackAndAdmission(*gpu);
  TreeAttemptFailure();
  context = "workflow allocation sweeps";
  AllocationSweep(*gpu);
  AllocationSweep(*gpu, true);
  AllocationSweep(*gpu, true, AnsSelectionMode::kEstimated);
  AllocationSweep(*gpu, true, AnsSelectionMode::kEstimated, true);
  context = "cache lifetime and keys";
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
