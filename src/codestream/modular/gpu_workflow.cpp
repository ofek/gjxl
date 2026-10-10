// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/gpu_workflow.h"

#include "codec/modular/profile.h"
#include "codestream/modular/prepared_encoder.h"
#include "codestream/workflow_admission_scope.h"
#include "core/publication_output.h"
#include "core/thread_budget.h"

namespace gjxl::modular_gpu_internal {
using namespace modular_internal;
Status ComputeGpuModularWorkflowStoragePlan(Extent2D extent, size_t row_stride_bytes,
                                            PackedModularFormat format, SampleByteOrder byte_order,
                                            const ModularEncodingOptions &options,
                                            const ModularInputDeviceLimits &limits,
                                            GpuModularWorkflowStoragePlan *out) {
  if (!out)
    return Status::InvalidArgument("Null GPU Modular workflow plan");
  if (options.search || options.coding.transforms.size)
    return Status::Unsupported(
        "GPU Modular input supports prescribed RCT without other transforms");
  GpuModularWorkflowStoragePlan plan;
  if (auto s = ComputeModularWorkflowStoragePlan(extent, format, options.entropy, options.coding,
                                                 &plan.cpu, options.cpu_thread_count);
      !s.ok())
    return s;
  if (auto s = ComputeModularInputStoragePlan(
          {extent, row_stride_bytes, format, byte_order, options.coding.rct, true}, &plan.input,
          &limits);
      !s.ok())
    return s;
  plan.peak_bytes = std::max(plan.input.peak_bytes, plan.cpu.working.peak_bytes);
  *out = plan;
  return Status::Ok();
}
namespace {
Status Encode(GpuBackend &backend, PackedModularImageView input, ModularEncodingOptions options,
              resource_budget_internal::PublicationOutput<uint8_t> output) try {
  if (output == nullptr)
    return Status::InvalidArgument("Null GPU Modular workflow output");
  if (auto s = input.Validate(); !s.ok())
    return s;
  auto *capability = QueryGpuModularInputPreparation(backend);
  if (!capability)
    return Status::Unsupported("Backend has no Modular input preparation");
  GpuModularWorkflowStoragePlan storage;
  if (auto s = ComputeGpuModularWorkflowStoragePlan(input.extent, input.row_stride, input.format,
                                                    input.byte_order, options,
                                                    capability->modular_input_limits(), &storage);
      !s.ok())
    return s;
  codestream_internal::WorkflowAdmission admission;
  if (auto s = admission.Start(storage.peak_bytes, options.execution_domain); !s.ok())
    return s;
  thread_budget_internal::CpuExecutionScope execution;
  if (auto s = execution.Start(options.execution_domain, options.cpu_thread_count); !s.ok())
    return s;
  thread_budget_internal::EncodeScope threads(options.cpu_thread_count);
  resource_budget_internal::ManagedHostScope managed(
      resource_budget_internal::ResourceClass::kPreparation);
  ModularEncoderFrame frame;
  {
    ProfileScope input_profile(ProfileStage::kInput);
    std::unique_ptr<PreparedGpuModularInput> device;
    if (auto s = PrepareGpuModularInput(backend, {input, options.coding.rct}, &device); !s.ok())
      return s;
    if (!device || device->metadata() != storage.input.metadata ||
        device->rct() != options.coding.rct)
      return Status::Internal("GPU Modular preparation returned unexpected metadata/RCT");
    ModularImage host;
    if (auto s = ReadbackGpuModularInput(backend, *device, &host); !s.ok())
      return s;
    device.reset(); // No device/sample overlap with tokenization or later CPU phases.
    if (auto s = ModularEncoderFrame::AdoptPreparedRct(input.extent, input.format,
                                                       options.coding.rct, std::move(host), &frame);
        !s.ok())
      return s;
  }
  codestream_internal::CodestreamBuffer candidate;
  if (auto s = EncodePreparedModularFrame(std::move(frame), options.coding, options.entropy,
                                          storage.cpu, &candidate);
      !s.ok())
    return s;
  output.Publish(std::move(candidate));
  if (active_modular_profile) {
    active_modular_profile->resolved = options.coding;
    ++active_modular_profile->encoded_candidates;
  }
  return Status::Ok();
} catch (const resource_budget_internal::ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("GPU Modular workflow allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("GPU Modular workflow storage overflow");
}
} // namespace
Status EncodeGpuModularImageOwned(GpuBackend &backend, PackedModularImageView input,
                                  ModularEncodingOptions options,
                                  codestream_internal::CodestreamBuffer *out) {
  return Encode(backend, input, std::move(options), out);
}
Status EncodeGpuModularImage(GpuBackend &backend, PackedModularImageView input,
                             ModularEncodingOptions options, std::vector<uint8_t> *out) {
  return Encode(backend, input, std::move(options), out);
}
} // namespace gjxl::modular_gpu_internal
