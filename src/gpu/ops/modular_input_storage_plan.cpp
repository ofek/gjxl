// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/ops/modular_input_storage_plan.h"

#include <algorithm>

namespace gjxl::modular_gpu_internal {
Status ComputeModularInputStoragePlan(const ModularInputStorageOptions &options,
                                      ModularInputStoragePlan *out,
                                      const ModularInputDeviceLimits *limits) {
  if (!out)
    return Status::InvalidArgument("Null GPU Modular input plan");
  if (limits &&
      (limits->maximum_extent.empty() || !limits->maximum_row_stride_bytes ||
       !limits->maximum_buffer_bytes))
    return Status::InvalidArgument("Invalid GPU Modular input limits");

  ModularInputStoragePlan plan;
  if (auto s = modular_internal::ComputePackedModularInputSize(
          options.extent, options.format, options.byte_order,
          options.row_stride_bytes, &plan.upload_bytes);
      !s.ok())
    return s;
  modular_internal::ModularInputProfile profile;
  if (auto s = modular_internal::ResolveModularInput(options.extent,
                                                     options.format, &profile);
      !s.ok())
    return s;
  if (options.rct >= 42 || (options.rct && profile.channel_count < 3))
    return Status::InvalidArgument(
        "GPU Modular RCT requires color input and a valid type");

  plan.metadata = profile.metadata;
  if (options.rct)
    plan.metadata.modular_16_bit_buffer_sufficient = false;
  plan.channel_count = profile.channel_count;
  plan.channels = profile.descriptors;
  // The packed-size helper has already checked this product.
  plan.input_row_bytes =
      options.extent.width * profile.channel_count * profile.bytes_per_sample;
  DeviceScratchLayoutPlan arena;
  for (size_t c = 0; c < plan.channel_count; ++c) {
    if (auto s = arena.AddPlane(DeviceElementType::kI32, options.extent,
                                options.extent.width,
                                kModularInputPlaneAlignment, &plan.planes[c]);
        !s.ok())
      return s;
  }
  plan.output_bytes = arena.capacity_bytes();
  if (limits && (options.extent.width > limits->maximum_extent.width ||
                 options.extent.height > limits->maximum_extent.height ||
                 options.row_stride_bytes > limits->maximum_row_stride_bytes ||
                 plan.upload_bytes > limits->maximum_buffer_bytes ||
                 plan.output_bytes > limits->maximum_buffer_bytes))
    return Status::Unsupported("GPU Modular input exceeds backend limits");

  if (options.include_host_readback) {
    if (auto s = modular_internal::ComputeModularImageStorageBound(
            profile.channels(), 0, &plan.host_readback);
        !s.ok())
      return s;
    if (plan.host_readback.peak_bytes > SIZE_MAX - plan.output_bytes)
      return Status::InvalidArgument("GPU Modular readback storage overflows");
    plan.readback_peak_bytes =
        plan.output_bytes + plan.host_readback.peak_bytes;
  }
  if (plan.upload_bytes > SIZE_MAX - plan.output_bytes)
    return Status::InvalidArgument("GPU Modular preparation storage overflows");
  plan.preparation_peak_bytes = plan.upload_bytes + plan.output_bytes;
  plan.peak_bytes =
      std::max(plan.preparation_peak_bytes, plan.readback_peak_bytes);
  *out = plan;
  return Status::Ok();
}
} // namespace gjxl::modular_gpu_internal
