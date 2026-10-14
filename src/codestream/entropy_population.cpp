// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/entropy_population.h"
#include <algorithm>

namespace gjxl::codestream_internal {
namespace {
bool ValidShape(const EntropyContextPopulations &p) {
  return p.contexts && p.value_limit && p.value_limit <= UINT32_MAX &&
      p.contexts <= SIZE_MAX / p.value_limit && p.counts.size() == p.contexts * p.value_limit;
}
} // namespace
Status ValidateEntropyContextPopulations(const EntropyContextPopulations &p, uint64_t token_count) {
  if (!ValidShape(p)) return Status::InvalidArgument("Invalid entropy population dimensions");
  uint64_t total = 0;
  for (uint64_t count : p.counts) {
    if (count > UINT64_MAX - total) return Status::InvalidArgument("Entropy population total overflow");
    total += count;
  }
  return total == token_count ? Status::Ok()
      : Status::InvalidArgument("Entropy population count differs from tokens");
}
Status ComputeEntropyPopulationMergeStorageBound(size_t tokens, size_t contexts, size_t value_limit,
                                                 size_t clusters,
                                                 resource_budget_internal::HostStorageBound *out) {
  using namespace resource_budget_internal;
  if (!out || !contexts || !value_limit || value_limit > UINT32_MAX || !clusters ||
      clusters > kMaximumAnsClusters || clusters > contexts || value_limit > SIZE_MAX / clusters)
    return Status::InvalidArgument("Invalid entropy population merge shape");
  HostStorageBound result;
  if (!result.AddVector<uint64_t>(clusters * value_limit, VectorCapacityPolicy::kFreshExact) ||
      !result.AddVector<Storage<WeightedValue>>(clusters, VectorCapacityPolicy::kFreshExact) ||
      !result.AddVector<WeightedValue>(std::min(tokens, clusters * value_limit), VectorCapacityPolicy::kGrowing))
    return Status::InvalidArgument("Entropy population merge storage overflow");
  *out = result;
  return Status::Ok();
}
Status MergeEntropyContextPopulations(const EntropyContextPopulations &p,
                                      std::span<const uint8_t> map, size_t clusters,
                                      Storage<Storage<WeightedValue>> *out) try {
  if (!out || !ValidShape(p) || map.size() != p.contexts || !clusters ||
      clusters > kMaximumAnsClusters || clusters > p.contexts ||
      p.value_limit > SIZE_MAX / clusters ||
      std::ranges::any_of(map, [clusters](uint8_t c) { return c >= clusters; }))
    return Status::InvalidArgument("Invalid entropy population merge input");
  Storage<uint64_t> merged(clusters * p.value_limit, 0);
  for (size_t context = 0; context < p.contexts; ++context)
    for (size_t value = 0; value < p.value_limit; ++value) {
      const uint64_t n = p.counts[context * p.value_limit + value];
      auto &count = merged[size_t(map[context]) * p.value_limit + value];
      if (n > UINT64_MAX - count) return Status::InvalidArgument("Entropy clustered population overflow");
      count += n;
    }
  Storage<Storage<WeightedValue>> result(clusters);
  for (size_t cluster = 0; cluster < clusters; ++cluster) {
    const auto counts = std::span(merged).subspan(cluster * p.value_limit, p.value_limit);
    result[cluster].reserve(std::count_if(counts.begin(), counts.end(), [](uint64_t n) { return n != 0; }));
    for (size_t value = 0; value < p.value_limit; ++value)
      if (counts[value]) result[cluster].push_back({uint32_t(value), counts[value]});
  }
  *out = std::move(result);
  return Status::Ok();
} catch (const resource_budget_internal::ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Entropy population merge allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Entropy population merge storage overflow");
}
} // namespace gjxl::codestream_internal
