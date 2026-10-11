// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/profile_internal.h"
#include <array>
#include <ostream>
#include <string_view>

namespace gjxl::modular_benchmark_internal {
using codestream_internal::EntropyWorkProfile;
struct EntropyTimeColumn {
  std::string_view name;
  uint64_t EntropyWorkProfile::*member;
};
inline constexpr auto kEntropyTimeColumns = std::to_array<EntropyTimeColumn>({
    {"prefix_histogram_build", &EntropyWorkProfile::prefix_histogram_build_nanoseconds},
    {"prefix_histogram_cost", &EntropyWorkProfile::prefix_histogram_cost_nanoseconds},
    {"prefix_clustering", &EntropyWorkProfile::prefix_clustering_nanoseconds},
    {"prefix_code_build", &EntropyWorkProfile::prefix_code_build_nanoseconds},
    {"prefix_value_collection", &EntropyWorkProfile::prefix_value_collection_nanoseconds},
    {"prefix_config_search", &EntropyWorkProfile::prefix_config_search_nanoseconds},
    {"prefix_value_aggregation", &EntropyWorkProfile::prefix_value_aggregation_nanoseconds},
    {"prefix_exact_measurement", &EntropyWorkProfile::prefix_exact_measurement_nanoseconds},
    {"ans_prefix_validation", &EntropyWorkProfile::ans_prefix_validation_nanoseconds},
    {"ans_value_collection", &EntropyWorkProfile::ans_value_collection_nanoseconds},
    {"ans_value_aggregation", &EntropyWorkProfile::ans_value_aggregation_nanoseconds},
    {"ans_prepared_value_validation", &EntropyWorkProfile::ans_prepared_value_validation_nanoseconds},
    {"ans_uint_config", &EntropyWorkProfile::ans_uint_config_nanoseconds},
    {"ans_histogram_build", &EntropyWorkProfile::ans_histogram_build_nanoseconds},
    {"ans_model_build", &EntropyWorkProfile::ans_model_build_nanoseconds},
    {"ans_token_cost", &EntropyWorkProfile::ans_token_cost_nanoseconds},
    {"selection", &EntropyWorkProfile::selection_nanoseconds},
});
inline void EntropyHeader(std::ostream &out) {
  for (const auto &column : kEntropyTimeColumns)
    out << ',' << column.name << "_work_ns";
  out << ",ans_uint_config_candidates,ans_histogram_candidates,ans_alphabet_width_candidates";
}
inline void EntropyRow(std::ostream &out, const EntropyWorkProfile &profile) {
  for (const auto &column : kEntropyTimeColumns)
    out << ',' << profile.*column.member;
  out << ',' << profile.ans_uint_config_candidate_count << ','
      << profile.ans_histogram_candidate_count << ',' << profile.ans_alphabet_width_candidate_count;
}
} // namespace gjxl::modular_benchmark_internal
