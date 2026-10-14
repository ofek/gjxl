// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "gjxl/modular.hpp"
#include "codestream/modular/workflow.h"

namespace gjxl::modular_internal {
// Shared public option validation and compatibility-alias resolution.
[[nodiscard]] Status ResolvePublicModularPolicy(
    const gjxl::ModularEncodingOptions &options, ModularPolicy *policy);
// Keeps output charged until the C adapter has admitted its publication copy.
[[nodiscard]] Status EncodePublicModularImageOwned(
    PackedModularImageView input, const gjxl::ModularEncodingOptions &options,
    codestream_internal::CodestreamBuffer *output);
} // namespace gjxl::modular_internal
