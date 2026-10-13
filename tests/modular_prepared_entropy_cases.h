// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/modular/prepared_streams.h"

namespace gjxl::test {
struct PreparedEntropyCase {
  const char *name;
  modular_internal::PreparedModularEntropyOptions options;
};
inline constexpr PreparedEntropyCase kPreparedEntropyCases[] = {
    {"prefix", {.policy = modular_internal::PreparedModularEntropyPolicy::kPrefix}},
    {"ans", {}},
    {"direct32", {.policy = modular_internal::PreparedModularEntropyPolicy::kDirectRateAns}},
    {"direct64",
     {.policy = modular_internal::PreparedModularEntropyPolicy::kDirectRateAns,
      .maximum_ans_clusters = 64}},
    {"estimated32",
     {.policy = modular_internal::PreparedModularEntropyPolicy::kDirectRateAns,
      .selection = AnsSelectionMode::kEstimated}},
    {"estimated64",
     {.policy = modular_internal::PreparedModularEntropyPolicy::kDirectRateAns,
      .selection = AnsSelectionMode::kEstimated,
      .maximum_ans_clusters = 64}},
};
} // namespace gjxl::test
