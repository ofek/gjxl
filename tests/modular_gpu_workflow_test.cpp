// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/gpu_token_workflow.h"
#include "codestream/modular/gpu_workflow.h"
#include "codestream/modular/prepared_encoder.h"
#include "codestream/workflow_admission_scope.h"
#include "core/cpu_execution.h"
#include <iostream>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::modular_gpu_internal;
using namespace gjxl::resource_budget_internal;
void Check(bool v, const char *m) {
  if (!v)
    throw std::runtime_error(m);
}
void Ok(const Status &s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
constexpr ModularInputDeviceLimits limits{{1u << 20, 1u << 20}, SIZE_MAX, 1u << 30};
constexpr ModularTokenDeviceLimits token_limits{1u << 28, 1u << 20, 1u << 30};

void Planning() {
  for (auto format :
       {PackedModularFormat::kGray8, PackedModularFormat::kGray16, PackedModularFormat::kRgb8,
        PackedModularFormat::kRgb16, PackedModularFormat::kRgba8, PackedModularFormat::kRgba16}) {
    ModularInputProfile profile;
    Ok(ResolveModularInput({257, 259}, format, &profile));
    for (auto entropy : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns})
      for (auto threads : {size_t{1}, size_t{4}})
        for (size_t padding : {size_t{1}, size_t{1} << 20}) {
          const size_t stride = 257 * profile.channel_count * profile.bytes_per_sample + padding;
          ModularEncodingOptions options{.entropy = entropy, .cpu_thread_count = threads};
          options.coding.rct = profile.channel_count == 1 ? 0 : 41;
          options.coding.tree.nodes[0].predictor = Predictor::kWeighted;
          GpuModularWorkflowStoragePlan plan;
          ArmNextManagedHostAllocationFailureForTest();
          Ok(ComputeGpuModularWorkflowStoragePlan(
              {257, 259}, stride, format, SampleByteOrder::kBigEndian, options, limits, &plan));
          const bool pending = ManagedHostAllocationFailurePendingForTest();
          DisarmManagedHostAllocationFailureForTest();
          Check(pending && plan.input.host_readback.retained_bytes > 0 &&
                    plan.peak_bytes == std::max(plan.input.peak_bytes, plan.cpu.working.peak_bytes),
                "Complete GPU workflow planner allocated or summed disjoint phases");
          if (padding == (size_t{1} << 20))
            Check(plan.peak_bytes == plan.input.preparation_peak_bytes,
                  "Large-stride upload missing from whole-call admission");
          const auto previous = plan;
          auto bad = options;
          bad.search = true;
          Check(!ComputeGpuModularWorkflowStoragePlan(
                     {257, 259}, stride, format, SampleByteOrder::kBigEndian, bad, limits, &plan)
                        .ok() &&
                    plan == previous,
                "Search accepted or failed planning changed output");
          bad = options;
          bad.coding.transforms.size = 1;
          Check(!ComputeGpuModularWorkflowStoragePlan(
                     {257, 259}, stride, format, SampleByteOrder::kBigEndian, bad, limits, &plan)
                        .ok() &&
                    plan == previous,
                "Additional transform accepted");
          Check(!ComputeGpuModularWorkflowStoragePlan({257, 259}, SIZE_MAX, format,
                                                      SampleByteOrder::kBigEndian, options, limits,
                                                      &plan)
                        .ok() &&
                    plan == previous,
                "Stride overflow accepted");
        }
  }
}

void Adoption() {
  const Extent2D extent{3, 2};
  ModularInputProfile profile;
  Ok(ResolveModularInput(extent, PackedModularFormat::kRgba16, &profile));
  ModularImage samples;
  Ok(ModularImage::Create(profile.channels(), 0, &samples));
  for (size_t c = 0; c < 4; ++c)
    for (auto &v : samples.samples(c))
      v = static_cast<int32_t>(c) - 2;
  const auto *identity = samples.view(0).backing.data();
  ModularEncoderFrame frame;
  for (unsigned error = 0; error < 4; ++error) {
    const auto status = ModularEncoderFrame::AdoptPreparedRct(
        error == 0 ? Extent2D{2, 3} : extent,
        error == 1 ? PackedModularFormat::kRgb16 : PackedModularFormat::kRgba16,
        error == 2 ? 42 : 6, std::move(samples), error == 3 ? nullptr : &frame);
    Check(!status.ok() && samples.view(0).backing.data() == identity &&
              frame.image().channel_count() == 0,
          "Failed adoption consumed samples or output");
  }
  ArmNextManagedHostAllocationFailureForTest();
  Ok(ModularEncoderFrame::AdoptPreparedRct(extent, PackedModularFormat::kRgba16, 6,
                                           std::move(samples), &frame));
  Check(ManagedHostAllocationFailurePendingForTest() &&
            frame.image().view(0).backing.data() == identity &&
            frame.image().view(0).backing[0] == -2 && frame.image().view(3).backing[0] == 1 &&
            !frame.metadata().modular_16_bit_buffer_sufficient,
        "Adoption allocated, copied, transformed samples or changed metadata");
  DisarmManagedHostAllocationFailureForTest();
}

class RejectingBackend final : public GpuBackend,
                               public GpuModularInputPreparation,
                               public GpuModularTokenGeneration {
public:
  size_t calls = 0;
  BackendKind kind() const noexcept override { return BackendKind::kMetal; }
  std::string_view name() const noexcept override { return "Rejecting workflow capability"; }
  ModularInputDeviceLimits modular_input_limits() const noexcept override { return limits; }
  Status PrepareModularInput(const ModularInputPreparation &,
                             std::unique_ptr<PreparedGpuModularInput> *) override {
    Check(CurrentResourceContext().reservation && thread_budget_internal::HasCpuParticipation(),
          "GPU preparation started before memory/CPU admission");
    ++calls;
    return Status::DeviceError("Preparation failure for workflow test");
  }
  ModularTokenDeviceLimits modular_token_limits() const noexcept override { return token_limits; }
  Status GenerateModularTokens(const PreparedGpuModularInput &, const ModularTokenizationOptions &,
                               std::span<const ModularTokenSlice>,
                               std::unique_ptr<PreparedGpuModularTokens> *) override {
    throw std::runtime_error("Composed workflow used separate token submission");
  }
  Status PrepareModularTokens(PackedModularImageView, const ModularTokenizationOptions &,
                              std::span<const ModularTokenSlice> slices,
                              std::unique_ptr<PreparedGpuModularTokens> *) override {
    Check(CurrentResourceContext().reservation && thread_budget_internal::HasCpuParticipation() &&
              !slices.empty(),
          "Composed GPU operation started before layout/admission/CPU participation");
    ++calls;
    return Status::DeviceError("Composed preparation failure for workflow test");
  }
  Status Allocate(size_t, std::unique_ptr<DeviceBuffer> *) override {
    throw std::runtime_error("Unexpected allocation");
  }
  Status CopyHostToDevice(DeviceBuffer &, const void *, size_t, size_t) override {
    throw std::runtime_error("Unexpected upload");
  }
  Status CopyDeviceToHost(const DeviceBuffer &, void *, size_t, size_t) override {
    throw std::runtime_error("Unexpected copy");
  }
  Status ForwardTransform(const TransformBatch &, std::unique_ptr<GpuSubmission> *) override {
    return Status::Unsupported("unused");
  }
  Status InverseTransform(const TransformBatch &, std::unique_ptr<GpuSubmission> *) override {
    return Status::Unsupported("unused");
  }
};
void AdmissionAndRejection() {
  RejectingBackend backend;
  std::array<uint8_t, 18> bytes{};
  PackedModularImageView input{bytes, {3, 2}, 9};
  GpuModularWorkflowStoragePlan plan;
  Ok(ComputeGpuModularWorkflowStoragePlan(input.extent, input.row_stride, input.format,
                                          input.byte_order, {}, limits, &plan));
  std::shared_ptr<const ExecutionDomain> domain, small, foreign;
  Ok(ExecutionDomain::Create({plan.peak_bytes, 1}, &domain));
  Ok(ExecutionDomain::Create({plan.peak_bytes - 1, 1}, &small));
  Ok(ExecutionDomain::Create({}, &foreign));
  std::vector<uint8_t> output{7, 8, 9};
  // Learned CPU trees can now exceed the GPU ceiling. Both planning and the
  // public-vector/owned entry points must reject them without copying policy
  // storage, touching the device, or changing an existing result.
  {
    ModularEncodingOptions large;
    large.coding.tree.nodes.resize(127);
    large.coding.tree.size = 127;
    for (size_t i = 0; i < 63; ++i)
      large.coding.tree.nodes[2 * i] = {.property = 3,
                                        .split = static_cast<int32_t>(i),
                                        .left = static_cast<TreeIndex>(2 * i + 2),
                                        .right = static_cast<TreeIndex>(2 * i + 1)};
    GpuModularTokenWorkflowStoragePlan token_plan, unchanged;
    codestream_internal::CodestreamBuffer owned;
    ArmNextManagedHostAllocationFailureForTest();
    Check(ComputeGpuModularTokenWorkflowStoragePlan(input.extent, input.row_stride, input.format,
                                                    input.byte_order, large, limits, token_limits,
                                                    &token_plan)
                      .code() == StatusCode::kUnsupported &&
              token_plan == unchanged &&
              EncodeGpuModularTokensImage(backend, input, large, &output).code() ==
                  StatusCode::kUnsupported &&
              EncodeGpuModularTokensImageOwned(backend, input, large, &owned).code() ==
                  StatusCode::kUnsupported &&
              ManagedHostAllocationFailurePendingForTest() && backend.calls == 0 &&
              output == std::vector<uint8_t>({7, 8, 9}) && owned.empty(),
          "Unsupported large token policy copied before rejection or changed output");
    DisarmManagedHostAllocationFailureForTest();
  }
  ArmNextManagedHostAllocationFailureForTest();
  Check(EncodeGpuModularImage(backend, input, {.execution_domain = small}, &output).code() ==
                StatusCode::kOutOfMemory &&
            backend.calls == 0 && ManagedHostAllocationFailurePendingForTest(),
        "Failed admission started GPU preparation");
  Check(!EncodeGpuModularImage(backend, input, {domain, EntropyCodingMode::kPrefix, {}, true},
                               &output)
                .ok() &&
            !EncodeGpuModularImage(backend, input, {.execution_domain = domain}, nullptr).ok() &&
            backend.calls == 0,
        "Unsupported policy/null output reached GPU preparation");
  Check(EncodeGpuModularImage(backend, input, {.execution_domain = domain}, &output).code() ==
                StatusCode::kDeviceError &&
            backend.calls == 1 && output == std::vector<uint8_t>({7, 8, 9}) &&
            ManagedHostAllocationFailurePendingForTest(),
        "Failure fell back or changed output");
  DisarmManagedHostAllocationFailureForTest();
  {
    codestream_internal::WorkflowAdmission outer;
    Ok(outer.Start(plan.peak_bytes, domain));
    Check(!EncodeGpuModularImage(backend, input, {.execution_domain = foreign}, &output).ok() &&
              backend.calls == 1,
          "Foreign nested domain accepted");
    Check(!EncodeGpuModularImage(backend, input, {.execution_domain = domain}, &output).ok() &&
              backend.calls == 2 && domain->snapshot().active_reservations == 1,
          "Nested workflow did not share the outer reservation");
  }
  // Large CPU trees are legal in this GPU-input/CPU-token workflow. Passing
  // options and planning must not allocate policy copies before admission.
  ModularEncodingOptions large;
  large.coding.tree.nodes.resize(127);
  large.coding.tree.size = 127;
  for (size_t i = 0; i < 63; ++i)
    large.coding.tree.nodes[2 * i] = {.property = 3,
                                      .split = static_cast<int32_t>(i),
                                      .left = static_cast<TreeIndex>(2 * i + 2),
                                      .right = static_cast<TreeIndex>(2 * i + 1)};
  ArmNextManagedHostAllocationFailureForTest();
  Ok(ComputeGpuModularWorkflowStoragePlan(input.extent, input.row_stride, input.format,
                                          input.byte_order, large, limits, &plan));
  Check(ManagedHostAllocationFailurePendingForTest(), "Large GPU-input workflow planner allocated");
  DisarmManagedHostAllocationFailureForTest();
  Ok(ExecutionDomain::Create({plan.peak_bytes - 1, 1}, &large.execution_domain));
  const auto calls = backend.calls;
  ArmNextManagedHostAllocationFailureForTest();
  Check(EncodeGpuModularImage(backend, input, large, &output).code() == StatusCode::kOutOfMemory &&
            ManagedHostAllocationFailurePendingForTest() && backend.calls == calls &&
            output == std::vector<uint8_t>({7, 8, 9}),
        "Large policy copied before admission");
  DisarmManagedHostAllocationFailureForTest();
  const auto s = domain->snapshot();
  Check(s.live_capacity_bytes == 0 && s.reserved_unbacked_bytes == 0 &&
            s.active_reservations == 0 && s.active_cpu_participants == 0,
        "Failed workflow leaked admission/CPU participation");
}
void TokenPlanningAndAdmission() {
  for (auto format :
       {PackedModularFormat::kGray8, PackedModularFormat::kRgb8, PackedModularFormat::kRgba8,
        PackedModularFormat::kGray16, PackedModularFormat::kRgb16, PackedModularFormat::kRgba16}) {
    ModularInputProfile profile;
    Ok(ResolveModularInput({257, 259}, format, &profile));
    for (auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns})
      for (size_t threads : {1u, 4u})
        for (size_t padding : {size_t{1}, size_t{1} << 20}) {
          const size_t stride = 257 * profile.channel_count * profile.bytes_per_sample + padding;
          ModularEncodingOptions options{.entropy = mode, .cpu_thread_count = threads};
          options.coding.rct = profile.channel_count == 1 ? 0 : 6;
          GpuModularTokenWorkflowStoragePlan plan;
          ArmNextManagedHostAllocationFailureForTest();
          Ok(ComputeGpuModularTokenWorkflowStoragePlan({257, 259}, stride, format,
                                                       SampleByteOrder::kBigEndian, options, limits,
                                                       token_limits, &plan));
          const bool pending = ManagedHostAllocationFailurePendingForTest();
          DisarmManagedHostAllocationFailureForTest();
          Check(pending && plan.generation.input.host_readback.peak_bytes == 0 &&
                    plan.peak_bytes >= plan.generation.generation_peak_bytes &&
                    plan.peak_bytes >= plan.handoff_peak_bytes &&
                    plan.peak_bytes >= plan.cpu.working.peak_bytes,
                "Composed planner allocated, added readback or omitted a lifetime phase");
          if (padding == (size_t{1} << 20))
            Check(plan.peak_bytes == plan.generation.generation_peak_bytes &&
                      plan.peak_bytes > plan.generation.input.preparation_peak_bytes +
                                            plan.generation.tokens.retained_bytes,
                  "Generation did not include overlapping planes/tokens/dispatch/layout");
          const auto old = plan;
          for (unsigned error = 0; error < 7; ++error) {
            auto bad = options;
            if (error == 0)
              bad.search = true;
            if (error == 1)
              bad.coding.tree.nodes[0].predictor = Predictor::kWeighted;
            if (error == 2)
              bad.coding.transforms.size = 1;
            if (error == 3)
              bad.cpu_thread_count = 257;
            if (error == 4)
              bad.entropy = static_cast<EntropyCodingMode>(255);
            const auto status = ComputeGpuModularTokenWorkflowStoragePlan(
                {257, 259}, error == 5 ? SIZE_MAX : stride, format, SampleByteOrder::kBigEndian,
                bad, limits, error == 6 ? ModularTokenDeviceLimits{1, 1, 1} : token_limits, &plan);
            Check(!status.ok() && plan == old, "Composed planning failure changed old output");
          }
        }
  }
  RejectingBackend backend;
  std::array<uint8_t, 18> bytes{};
  PackedModularImageView input{bytes, {3, 2}, 9};
  GpuModularTokenWorkflowStoragePlan plan;
  Ok(ComputeGpuModularTokenWorkflowStoragePlan(input.extent, input.row_stride, input.format,
                                               input.byte_order, {}, limits, token_limits, &plan));
  std::shared_ptr<const ExecutionDomain> domain, small, foreign;
  Ok(ExecutionDomain::Create({plan.peak_bytes, 1}, &domain));
  Ok(ExecutionDomain::Create({plan.peak_bytes - 1, 1}, &small));
  Ok(ExecutionDomain::Create({}, &foreign));
  std::vector<uint8_t> output{7, 8, 9};
  ArmNextManagedHostAllocationFailureForTest();
  Check(EncodeGpuModularTokensImage(backend, input, {.execution_domain = small}, &output).code() ==
                StatusCode::kOutOfMemory &&
            backend.calls == 0 && ManagedHostAllocationFailurePendingForTest(),
        "Token admission allocated before rejection");
  DisarmManagedHostAllocationFailureForTest();
  for (size_t allocation = 0; allocation < 4; ++allocation) {
    ArmManagedHostAllocationFailureAfterForTest(allocation);
    const auto status =
        EncodeGpuModularTokensImage(backend, input, {.execution_domain = domain}, &output);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(status.code() == StatusCode::kOutOfMemory && !pending && backend.calls == 0,
          "Failed canonical layout reached device or escaped reservation");
  }
  Check(EncodeGpuModularTokensImage(backend, input, {.execution_domain = domain}, &output).code() ==
                StatusCode::kDeviceError &&
            backend.calls == 1 && output == std::vector<uint8_t>({7, 8, 9}),
        "Composed device failure changed output or selected another path");
  {
    codestream_internal::WorkflowAdmission outer;
    Ok(outer.Start(plan.peak_bytes, domain));
    Check(
        !EncodeGpuModularTokensImage(backend, input, {.execution_domain = foreign}, &output).ok() &&
            backend.calls == 1,
        "Composed workflow accepted a foreign nested domain");
    Check(
        !EncodeGpuModularTokensImage(backend, input, {.execution_domain = domain}, &output).ok() &&
            backend.calls == 2 && domain->snapshot().active_reservations == 1,
        "Composed workflow did not reuse outer admission");
  }
  const auto after = domain->snapshot();
  Check(after.live_capacity_bytes == 0 && after.active_reservations == 0 &&
            after.active_cpu_participants == 0,
        "Composed failure leaked memory/CPU admission");
}

} // namespace
int main() try {
  Planning();
  Adoption();
  AdmissionAndRejection();
  TokenPlanningAndAdmission();
  std::cout << "GPU Modular workflow planning, adoption and admission passed\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
