// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include "gpu/ops/modular_input_storage_plan.h"

namespace gjxl::modular_gpu_internal {

// Backend seam for the shared owning preparation flow. Submit borrows the
// validated, distinct buffers and records synchronously. Any committed work
// must be represented by *submission even on error/exception. Drain blocks
// until all buffer users finish, without constructing a Status or throwing;
// unlike GpuSubmission destruction, it is safe during exception unwinding.
struct ModularInputKernel {
  Status (*submit)(GpuBackend &, const ModularInputStorageOptions &, const DeviceBuffer &,
                   DeviceBuffer &, std::unique_ptr<GpuSubmission> *) = nullptr;
  void (*drain)(GpuSubmission &) noexcept = nullptr;
};

// Consumes the current admitted reservation; never opens a nested reservation
// or borrows its old output's charge. The unbounded default accounting context
// remains available for direct internal calls, as with GpuBackend::Allocate.
// Full-workflow admission, old-output overlap and remaining encoder phases
// belong to the caller. No cache reuse or host staging allocation.
[[nodiscard]] Status PrepareModularInputWithKernel(GpuBackend &backend,
                                                   const ModularInputPreparation &preparation,
                                                   const ModularInputDeviceLimits &limits,
                                                   const ModularInputKernel &kernel,
                                                   std::unique_ptr<PreparedGpuModularInput> *out);

} // namespace gjxl::modular_gpu_internal
