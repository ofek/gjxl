// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include "core/managed_allocator.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace gjxl_test {
// The injection hook is thread-local. These diagnostics deliberately run the
// encoder with one CPU participant, just like the exhaustive legacy sweep.
class ManagedAllocationFailureScope {
public:
  explicit ManagedAllocationFailureScope(size_t index) {
    gjxl::resource_budget_internal::ArmManagedHostAllocationFailureAfterForTest(index);
  }
  ~ManagedAllocationFailureScope() {
    gjxl::resource_budget_internal::DisarmManagedHostAllocationFailureForTest();
  }
  [[nodiscard]] bool pending() const {
    return gjxl::resource_budget_internal::ManagedHostAllocationFailurePendingForTest();
  }
};

template <class Encode> size_t CountManagedAllocations(Encode encode) {
  constexpr size_t initial = std::numeric_limits<size_t>::max() - 1;
  ManagedAllocationFailureScope scope(initial);
  const auto status = encode();
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
  if (!scope.pending()) throw std::runtime_error("Allocation counter exhausted");
  // Observe the existing test hook; production allocation behavior is unchanged.
  return initial - gjxl::resource_budget_internal::detail::managed_host_backings_before_failure;
}

// At most eight distinct positions, including the first and last allocation.
// Recompute from the successful trace rather than hardcoding platform counts.
inline std::vector<size_t> AllocationProbeIndices(size_t count) {
  if (!count) throw std::runtime_error("Empty allocation trace");
  std::vector<size_t> result;
  for (size_t i = 0; i < 8; ++i) {
    const size_t index = ((count - 1) / 7) * i + ((count - 1) % 7) * i / 7;
    if (result.empty() || result.back() != index) result.push_back(index);
  }
  return result;
}

// CTest truncates successful stdout. Retain complete per-fixture JSONL too,
// with a different report path for each concurrent shard.
class AllocationReport {
public:
  void Open(const std::filesystem::path &path) {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    output_.exceptions(std::ios::badbit | std::ios::failbit);
    output_.open(path);
  }
  void Write(std::string_view json) {
    std::cout << json << std::endl;
    if (output_.is_open()) output_ << json << std::endl;
  }
private:
  std::ofstream output_;
};
} // namespace gjxl_test
