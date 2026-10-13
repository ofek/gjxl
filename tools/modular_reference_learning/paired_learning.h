// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "native_preparation.h"

namespace reference_learning {
// Both implementations read the same immutable transformed planes, including
// strides. Image transformation/conversion and reference comparisons stay
// outside measured calls; native admission and sample validation remain inside.
void RunPairedLearning(std::span<const jxl::Image> images,
                       const jxl::ModularOptions &options,
                       const Fixture &fixture, size_t repeats,
                       std::ostream &out);
} // namespace reference_learning
