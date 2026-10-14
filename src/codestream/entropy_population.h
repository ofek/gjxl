// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/entropy_internal.h"
#include "core/host_storage_bound.h"

namespace gjxl::codestream_internal {
// Completed exact counts of raw uint32 token values, indexed [context,value].
// Values outside value_limit must be absent; a partial dense prefix is invalid.
// The trusted producer owns canonical token/context provenance and the backing
// outlives all model consumers. Validation checks dimensions/counts, not a second
// ordered scan of the authoritative token stream.
struct EntropyContextPopulations {
  size_t contexts = 0, value_limit = 0;
  std::span<const uint64_t> counts;
};
[[nodiscard]] Status ValidateEntropyContextPopulations(
    const EntropyContextPopulations &populations, uint64_t token_count);
[[nodiscard]] Status MergeEntropyContextPopulations(
    const EntropyContextPopulations &populations, std::span<const uint8_t> context_map,
    size_t clusters, Storage<Storage<WeightedValue>> *out);
// Complete merge bound, including output; excludes the borrowed count table and
// map. WeightedValue rows are ascending and have unique nonzero entries.
[[nodiscard]] Status ComputeEntropyPopulationMergeStorageBound(
    size_t tokens, size_t contexts, size_t value_limit, size_t clusters,
    resource_budget_internal::HostStorageBound *out);

// Same prefix partition/configuration policy, driven by exact populations.
// Canonical streams remain authoritative for optional ordered section costs.
// Neither histogram construction nor population aggregation scans those tokens.
[[nodiscard]] Status OptimizeEntropyCodeFromContextPopulations(
    std::span<const EntropyTokenStreamView> section_tokens, const EntropyCodeOptions &options,
    const EntropyContextPopulations &populations, EntropyCode *code, EntropyCodeCost *cost,
    PreparedEntropyClusters *prepared, EntropyWorkProfile *profile = nullptr);
} // namespace gjxl::codestream_internal
