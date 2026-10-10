// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include "gpu/ops/modular_input_storage_plan.h"

namespace gjxl::metal_internal {

// Low-level kernel input for qualification and the future owning preparation
// path. Both buffers are borrowed. Output uses the planner's dense int32 arena,
// translated by output_offset_bytes (a multiple of the plane alignment).
// Input offsets may be unaligned. include_host_readback must be false.
struct MetalModularInputDispatch {
  const DeviceBuffer *input = nullptr;
  size_t input_offset_bytes = 0;
  DeviceBuffer *output = nullptr;
  size_t output_offset_bytes = 0;
  modular_gpu_internal::ModularInputStorageOptions layout;
};

// No allocation, pipeline creation or submission; backend must be Metal.
[[nodiscard]] Status
GetMetalModularInputLimits(GpuBackend &backend,
                           modular_gpu_internal::ModularInputDeviceLimits *out);

// Validates all layout, native-buffer ownership, ranges and non-aliasing before
// submitting one fused dispatch. Allocates no sample backing. The caller keeps
// both buffers alive, input immutable and output exclusive until Wait returns,
// including a completion error. Output contents are unspecified on device error.
// The submission output must be empty on entry; rejection preserves it and
// buffers. Success returns a nonnull submission. This is not the synchronous
// PreparedGpuModularInput capability and does not perform admission/readback.
[[nodiscard]] Status SubmitMetalModularInput(GpuBackend &backend,
                                             const MetalModularInputDispatch &dispatch,
                                             std::unique_ptr<GpuSubmission> *submission);

} // namespace gjxl::metal_internal
