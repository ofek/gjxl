// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/gpu_token_workflow.h"
#include "codec/modular/profile.h"
#include "codestream/modular/frame_encoder.h"
#include "codestream/workflow_admission_scope.h"
#include "core/publication_output.h"
#include "core/thread_budget.h"

namespace gjxl::modular_gpu_internal {
using namespace modular_internal;
using namespace resource_budget_internal;
Status ComputeGpuModularTokenWorkflowStoragePlan(Extent2D extent, size_t row_stride_bytes,
                                                 PackedModularFormat format,
                                                 SampleByteOrder byte_order,
                                                 const ModularEncodingOptions &options,
                                                 const ModularInputDeviceLimits &input_limits,
                                                 const ModularTokenDeviceLimits &token_limits,
                                                 GpuModularTokenWorkflowStoragePlan *out) {
  if (!out)
    return Status::InvalidArgument("Null GPU Modular token workflow plan");
  if (auto s = ValidateAnsSelectionMode(options.ans_selection); !s.ok())
    return s;
  GpuModularTokenWorkflowStoragePlan plan;
  // Validate the borrowed CPU tree before constructing inline GPU options.
  if (auto s = ResolveModularTokenEligibility(format, options.coding, options.search,
                                              &plan.generation.eligibility);
      !s.ok())
    return s;
  const GpuModularTokenPlanOptions tokens{
      extent, row_stride_bytes, byte_order, {format, options.coding, options.search}};
  if (auto s =
          ComputeGpuModularTokenStoragePlan(tokens, &plan.generation, &input_limits, &token_limits);
      !s.ok())
    return s;
  if (auto s =
          ComputeGpuModularTokenHandoffStorageBound(plan.generation.streams.streams, &plan.handoff);
      !s.ok())
    return s;
  if (auto s = ComputeModularTokenConsumerStoragePlan(
          extent, format, options.entropy, options.coding, plan.generation.tokens.output_bytes,
          &plan.cpu, options.cpu_thread_count);
      !s.ok())
    return s;
  if (plan.handoff.peak_bytes > SIZE_MAX - plan.generation.retained_bytes)
    return Status::InvalidArgument("GPU Modular token handoff storage overflow");
  plan.handoff_peak_bytes = plan.generation.retained_bytes + plan.handoff.peak_bytes;
  plan.peak_bytes = std::max({plan.generation.generation_peak_bytes, plan.handoff_peak_bytes,
                              plan.cpu.working.peak_bytes});
  *out = plan;
  return Status::Ok();
}
namespace {
Status Encode(GpuBackend &backend, PackedModularImageView input,
              const ModularEncodingOptions &options, PublicationOutput<uint8_t> output) try {
  if (output == nullptr)
    return Status::InvalidArgument("Null GPU Modular token output");
  if (auto s = input.Validate(); !s.ok())
    return s;
  auto *input_op = QueryGpuModularInputPreparation(backend);
  auto *token_op = QueryGpuModularTokenGeneration(backend);
  if (!input_op || !token_op)
    return Status::Unsupported("Backend has no composed Modular token preparation");
  const auto input_limits = input_op->modular_input_limits();
  const auto token_limits = token_op->modular_token_limits();
  GpuModularTokenWorkflowStoragePlan storage;
  if (auto s = ComputeGpuModularTokenWorkflowStoragePlan(input.extent, input.row_stride,
                                                         input.format, input.byte_order, options,
                                                         input_limits, token_limits, &storage);
      !s.ok())
    return s;
  codestream_internal::WorkflowAdmission admission;
  if (auto s = admission.Start(storage.peak_bytes, options.execution_domain); !s.ok())
    return s;
  thread_budget_internal::CpuExecutionScope execution;
  if (auto s = execution.Start(options.execution_domain, options.cpu_thread_count); !s.ok())
    return s;
  thread_budget_internal::EncodeScope threads(options.cpu_thread_count);
  const ManagedHostScope managed(ResourceClass::kSerializer);
  codec_internal::ImageMetadata metadata;
  ModularStreamPlan layout;
  PreparedModularTokens tokens;
  {
    // Combined upload/unpack/RCT/prediction interval; no CPU token stage runs.
    ProfileScope profile(ProfileStage::kGpuInputTokens);
    GpuModularTokenPlan plan;
    if (auto s = BuildGpuModularTokenPlan({input.extent,
                                           input.row_stride,
                                           input.byte_order,
                                           {input.format, options.coding, options.search}},
                                          &plan, &input_limits, &token_limits);
        !s.ok())
      return s;
    std::unique_ptr<PreparedGpuModularTokens> device;
    if (auto s =
            token_op->PrepareModularTokens(input, plan.options.tokenization, plan.slices, &device);
        !s.ok())
      return s;
    if (auto s = AdoptGpuModularTokens(plan, std::move(device), &tokens); !s.ok())
      return s;
    metadata = plan.metadata;
    layout = std::move(plan.layout);
  } // GPU descriptors/ranges die before CPU modeling; only layout/views/tokens remain.
  codestream_internal::CodestreamBuffer candidate;
  if (auto s = EncodeModularFrame(metadata, layout, std::move(tokens), options.entropy, storage.cpu,
                                  &candidate, options.ans_selection);
      !s.ok())
    return s;
  // Layout is not needed during the final public-vector copy.
  layout = {};
  if (active_modular_profile) {
    active_modular_profile->resolved = options.coding;
    ++active_modular_profile->encoded_candidates;
  }
  output.Publish(std::move(candidate));
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("GPU Modular token workflow allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("GPU Modular token workflow storage overflow");
}
} // namespace
Status EncodeGpuModularTokensImageOwned(GpuBackend &backend, PackedModularImageView input,
                                        const ModularEncodingOptions &options,
                                        codestream_internal::CodestreamBuffer *out) {
  return Encode(backend, input, options, out);
}
Status EncodeGpuModularTokensImage(GpuBackend &backend, PackedModularImageView input,
                                   const ModularEncodingOptions &options,
                                   std::vector<uint8_t> *out) {
  return Encode(backend, input, options, out);
}
} // namespace gjxl::modular_gpu_internal
