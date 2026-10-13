// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/modular/prepared_streams.h"
#include <iosfwd>
namespace reference_learning {
void ReplaySelectionCosts(const gjxl::modular_internal::PreparedModularStreamsView &input,
                          const gjxl::modular_internal::ModularStreamPlan &plan, size_t bit_depth,
                          std::ostream &out);
void ReplayWeightedCosts(const gjxl::modular_internal::PreparedModularStreamsView &input,
                         const gjxl::modular_internal::ModularStreamPlan &plan, size_t bit_depth,
                         std::ostream &out);
} // namespace reference_learning
