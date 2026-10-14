// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/metal/metal_modular_input.h"

#include <cstddef>
#include <limits>
#include <new>
#include <type_traits>

#include "gpu/metal/kernels/modular_input_params.h"
#include "gpu/metal/metal_backend_internal.h"
#include "gpu/ops/modular_input_internal.h"

namespace gjxl::metal_internal {
using namespace modular_gpu_internal;
namespace {
using Params = gjxl_modular_input::Params;
static_assert(std::is_standard_layout_v<Params> && std::is_trivially_copyable_v<Params>);
static_assert(sizeof(Params) == 72 && alignof(Params) == 8);
static_assert(offsetof(Params, output_offset_elements) == 16 && offsetof(Params, width) == 48);
static_assert(sizeof(size_t) <= sizeof(gjxl_modular_input::Address));

void Encode(MetalBackend &, MTL::ComputeCommandEncoder *encoder, const void *opaque) {
  EncodeMetalModularInput(encoder, *static_cast<const MetalModularInputEncoding *>(opaque));
}
} // namespace

void EncodeMetalModularInput(MTL::ComputeCommandEncoder *encoder,
                             const MetalModularInputEncoding &context) {
  encoder->setComputePipelineState(context.pipeline);
  RecordMetalComputePipelineState(context.pipeline);
  // Bind at zero so a byte-wise source can have any offset, independently of
  // Metal's binding alignment. The validated offsets live in the shared ABI.
  encoder->setBuffer(context.input, 0, 0);
  encoder->setBuffer(context.output, 0, 1);
  encoder->setBytes(&context.params, sizeof(Params), 2);
  DispatchMetalThreads(encoder, MTL::Size(context.params.width, context.params.height, 1),
                       MTL::Size(8, 8, 1));
}

ModularInputDeviceLimits MetalBackend::ModularInputLimits() const noexcept {
  return {{std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max()},
          std::numeric_limits<size_t>::max(),
          static_cast<size_t>(device_->maxBufferLength())};
}

Status MetalBackend::PrepareModularInput(const ModularInputPreparation &preparation,
                                         std::unique_ptr<PreparedGpuModularInput> *out) {
  const ModularInputKernel kernel{
      +[](GpuBackend &backend, const ModularInputStorageOptions &options, const DeviceBuffer &input,
          DeviceBuffer &output, std::unique_ptr<GpuSubmission> *submission) {
        return SubmitMetalModularInput(backend, {&input, 0, &output, 0, options}, submission);
      },
      DrainMetalSubmission};
  return PrepareModularInputWithKernel(*this, preparation, ModularInputLimits(), kernel, out);
}

Status MetalBackend::ResolveModularInputDispatch(const MetalModularInputDispatch &dispatch,
                                                 MetalModularInputEncoding *encoding) {
  if (dispatch.layout.include_host_readback ||
      dispatch.output_offset_bytes % kModularInputPlaneAlignment != 0)
    return Status::InvalidArgument("Invalid Metal Modular kernel layout");

  ModularInputStoragePlan plan;
  const auto limits = ModularInputLimits();
  if (auto s = ComputeModularInputStoragePlan(dispatch.layout, &plan, &limits); !s.ok())
    return s;
  const MetalBuffer *input = nullptr;
  MetalBuffer *output = nullptr;
  if (auto s = RequireMetalBuffer(dispatch.input, plan.upload_bytes, dispatch.input_offset_bytes,
                                  "Modular packed input", &input);
      !s.ok())
    return s;
  if (auto s = RequireMetalBuffer(dispatch.output, plan.output_bytes, dispatch.output_offset_bytes,
                                  "Modular planar output", &output);
      !s.ok())
    return s;
  if (DeviceRangesOverlap({dispatch.input, dispatch.input_offset_bytes, plan.upload_bytes},
                          {dispatch.output, dispatch.output_offset_bytes, plan.output_bytes}))
    return Status::InvalidArgument("Metal Modular input and output overlap");

  Params params{};
  params.input_offset_bytes = dispatch.input_offset_bytes;
  params.input_row_stride_bytes = dispatch.layout.row_stride_bytes;
  for (size_t c = 0; c < plan.channel_count; ++c)
    params.output_offset_elements[c] =
        (dispatch.output_offset_bytes + plan.planes[c].offset_bytes) / sizeof(int32_t);
  params.width = static_cast<uint32_t>(dispatch.layout.extent.width);
  params.height = static_cast<uint32_t>(dispatch.layout.extent.height);
  params.channels = static_cast<uint32_t>(plan.channel_count);
  params.bytes_per_sample = plan.metadata.bits / 8;
  params.big_endian = dispatch.layout.byte_order == modular_internal::SampleByteOrder::kBigEndian;
  params.rct = dispatch.layout.rct;
  *encoding = {input->handle(), output->handle(), modular_input_pipeline_.get(), params};
  return Status::Ok();
}

Status MetalBackend::SubmitModularInput(const MetalModularInputDispatch &dispatch,
                                        std::unique_ptr<GpuSubmission> *submission) {
  if (!submission || *submission)
    return Status::InvalidArgument("Metal Modular submission output must be empty");
  MetalModularInputEncoding context;
  if (auto s = ResolveModularInputDispatch(dispatch, &context); !s.ok())
    return s;
  try {
    // SubmitCompute records synchronously and publishes the submission only
    // after commit. Its potentially throwing owner allocation precedes commit.
    return SubmitCompute("modular.unpack_rct", Encode, &context, submission);
  } catch (const std::bad_alloc &) {
    return Status::OutOfMemory("Metal Modular submission allocation failed");
  }
}

Status GetMetalModularInputLimits(GpuBackend &backend, ModularInputDeviceLimits *out) {
  if (!out)
    return Status::InvalidArgument("Null Metal Modular limit output");
  const auto *metal = dynamic_cast<const MetalBackend *>(&backend);
  if (!metal)
    return Status::Unsupported("Modular input kernel requires Metal");
  *out = metal->ModularInputLimits();
  return Status::Ok();
}

Status SubmitMetalModularInput(GpuBackend &backend, const MetalModularInputDispatch &dispatch,
                               std::unique_ptr<GpuSubmission> *submission) {
  auto *metal = dynamic_cast<MetalBackend *>(&backend);
  if (!metal)
    return Status::Unsupported("Modular input kernel requires Metal");
  return metal->SubmitModularInput(dispatch, submission);
}
} // namespace gjxl::metal_internal
