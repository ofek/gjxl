// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include <array>

#include "gpu/ops/modular_input.h"
#include "gpu/scratch.h"

namespace gjxl::modular_gpu_internal {

inline constexpr size_t kModularInputPlaneAlignment = 256;

// No pointer/backing required: large geometry can be rejected during admission
// without allocating or fabricating a span. Runtime also validates input.bytes.
struct ModularInputStorageOptions {
  Extent2D extent;
  size_t row_stride_bytes = 0;
  modular_internal::PackedModularFormat format =
      modular_internal::PackedModularFormat::kRgb8;
  modular_internal::SampleByteOrder byte_order =
      modular_internal::SampleByteOrder::kLittleEndian;
  uint8_t rct = 0;
  bool include_host_readback = false;
};

struct ModularInputStoragePlan {
  codec_internal::ImageMetadata metadata;
  size_t channel_count = 0;
  std::array<modular_internal::ChannelDescriptor, 4> channels{};
  // Dense kI32 planes in one arena, with aligned, nonoverlapping offsets.
  // Unused channel/plane entries remain value-initialized.
  std::array<DevicePlaneLayout, 4> planes{};
  size_t input_row_bytes = 0;
  size_t upload_bytes = 0; // (height - 1) * input stride + input_row_bytes.
  size_t output_bytes = 0; // Arena capacity, including inter-plane alignment.
  resource_budget_internal::HostStorageBound host_readback;
  size_t preparation_peak_bytes = 0; // Upload + device output.
  // Device output + host_readback; zero if omitted.
  size_t readback_peak_bytes = 0;
  size_t peak_bytes = 0; // Maximum of the two phases, not their sum.
  bool operator==(const ModularInputStoragePlan &) const = default;
};

// Fresh backing only. Upload dies after the preparation fence, before optional
// readback allocates a fresh ModularImage. Device planes may remain alive
// during readback. Caller input, prior device/host outputs, downstream work,
// diagnostics, immutable backend setup, driver internals and small
// owner/control objects are excluded. Add their actual overlapping lifetimes
// for complete-call admission. No cache growth/replacement assumptions and no
// implicit GPU scratch buffers.
//
// Allocation-free on success; never reads pixels, queries a device or submits
// work. With limits=nullptr this proves format/layout arithmetic only. A
// backend must supply its limits before allocation. Exceeding valid device
// limits is Unsupported; malformed options/limits are InvalidArgument. Any
// failure leaves *out unchanged. Row stride is bytes for input, elements for
// output planes.
[[nodiscard]] Status ComputeModularInputStoragePlan(
    const ModularInputStorageOptions &options, ModularInputStoragePlan *out,
    const ModularInputDeviceLimits *limits = nullptr);

} // namespace gjxl::modular_gpu_internal
