// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include <memory>
#include <span>

#include "codec/modular/frame.h"
#include "gpu/backend.h"
#include "gpu/image.h"

namespace gjxl::modular_gpu_internal {

// Private first-slice contract: source-space unsigned integers, unpacking and
// one prescribed RCT only. No search, other transforms, prediction or entropy.
struct ModularInputPreparation {
  modular_internal::PackedModularImageView input;
  uint8_t rct = 0; // Identity for gray; 0..41 for RGB/RGBA. Alpha is unchanged.
};

// Capabilities of an already initialized backend/kernel. Every limit is
// inclusive and nonzero; zero is invalid, not an "unlimited" sentinel. Stride
// is the packed input stride, in bytes. The buffer limit applies separately to
// the upload allocation and the single arena containing all output planes.
struct ModularInputDeviceLimits {
  Extent2D maximum_extent;
  size_t maximum_row_stride_bytes = 0;
  size_t maximum_buffer_bytes = 0;
};

struct GpuModularChannelView {
  modular_internal::ChannelDescriptor descriptor;
  ConstDevicePlaneView plane;
};

// Owns completed, immutable kI32 planes and their resource charges. There are
// exactly 1/3/4 channels in source order, no metadata channels, no padding of
// the source extent and zero h/v shifts. RCT permutes/transforms only color
// values; channel roles, alpha and invisible RGB retain the scalar oracle's
// meaning. Samples equal ModularEncoderFrame::Prepare(input, rct); metadata
// includes its conservative modular_16_bit_buffer_sufficient flag. No float
// conversion, premultiplication, clamping or saturation is permitted.
//
// The backend must outlive this owner. All submissions borrowing its views must
// finish before destruction. A successful preparation retains no caller input,
// upload buffer or in-flight submission. Host views/readback are a separate
// operation: a device view never promises CPU accessibility.
class PreparedGpuModularInput {
public:
  virtual ~PreparedGpuModularInput() = default;
  PreparedGpuModularInput(const PreparedGpuModularInput &) = delete;
  PreparedGpuModularInput &operator=(const PreparedGpuModularInput &) = delete;

  [[nodiscard]] virtual BackendId backend_id() const noexcept = 0;
  [[nodiscard]] virtual const codec_internal::ImageMetadata &
  metadata() const noexcept = 0;
  [[nodiscard]] virtual uint8_t rct() const noexcept = 0;
  [[nodiscard]] virtual std::span<const GpuModularChannelView>
  channels() const noexcept = 0;

protected:
  PreparedGpuModularInput() = default;
};

// Optional backend capability. No existing Metal/CUDA backend implements this
// yet. Querying a backend without it returns nullptr; there is no CPU fallback.
class GpuModularInputPreparation {
public:
  virtual ~GpuModularInputPreparation() = default;

  // No allocation, device initialization or submission.
  [[nodiscard]] virtual ModularInputDeviceLimits
  modular_input_limits() const noexcept = 0;

  // Synchronous and failure-atomic: validate the backing, prescribed RCT and
  // storage plan against modular_input_limits() before allocating. Consume the
  // caller's resource context/reservation; orchestration admits the full call.
  // The caller keeps input immutable until return. On success, replace *out
  // with a nonnull completed owner. On any error (including a null out), leave
  // prior output/input unchanged and drain every submitted command before
  // freeing buffers/tickets. GpuSubmission destruction alone does not wait.
  // Old output remains charged separately throughout candidate preparation.
  [[nodiscard]] virtual Status
  PrepareModularInput(const ModularInputPreparation &preparation,
                      std::unique_ptr<PreparedGpuModularInput> *out) = 0;
};

[[nodiscard]] inline GpuModularInputPreparation *
QueryGpuModularInputPreparation(GpuBackend &backend) noexcept {
  return dynamic_cast<GpuModularInputPreparation *>(&backend);
}

} // namespace gjxl::modular_gpu_internal
