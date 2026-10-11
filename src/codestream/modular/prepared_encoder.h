// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/modular/frame_encoder.h"

namespace gjxl::modular_internal {
// Shared CPU continuation for a completed frame prepared with exactly policy.
// Storage must be the canonical workflow plan for that source format/policy.
// Caller owns full-call admission, CPU participation and atomic publication.
// Consumes the frame and releases its samples before modeling/emission/assembly.
[[nodiscard]] Status
EncodePreparedModularFrame(ModularEncoderFrame frame, const ModularCodingPolicy &policy,
                           EntropyCodingMode entropy, const ModularWorkflowStoragePlan &storage,
                           codestream_internal::CodestreamBuffer *out,
                           AnsSelectionMode selection = AnsSelectionMode::kExact,
                           ModularCandidateEvaluation *evaluation = nullptr,
                           GpuBackend *population_gpu = nullptr);
} // namespace gjxl::modular_internal
