// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
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

class RejectingBackend final : public GpuBackend, public GpuModularInputPreparation {
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
  ArmNextManagedHostAllocationFailureForTest();
  Check(EncodeGpuModularImage(backend, input, {.execution_domain = small}, &output).code() ==
                StatusCode::kOutOfMemory &&
            backend.calls == 0 && ManagedHostAllocationFailurePendingForTest(),
        "Failed admission started GPU preparation");
  Check(!EncodeGpuModularImage(backend, input, {domain, EntropyCodingMode::kPrefix, {}, true},
                               &output)
                .ok() &&
            !EncodeGpuModularImage(backend, input, {.execution_domain = domain}, nullptr).ok() && backend.calls == 0,
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
    Check(!EncodeGpuModularImage(backend, input, {.execution_domain = foreign}, &output).ok() && backend.calls == 1,
          "Foreign nested domain accepted");
    Check(!EncodeGpuModularImage(backend, input, {.execution_domain = domain}, &output).ok() && backend.calls == 2 &&
              domain->snapshot().active_reservations == 1,
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
} // namespace
int main() try {
  Planning();
  Adoption();
  AdmissionAndRejection();
  std::cout << "GPU Modular workflow planning, adoption and admission passed\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
