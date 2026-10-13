// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "lib/jxl/modular/modular_image.h"
namespace jxl::gjxl_cost_reference {
float EstimateWPCost(const Image &image, size_t mode);
} // namespace jxl::gjxl_cost_reference
