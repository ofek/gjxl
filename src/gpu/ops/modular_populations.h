// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "gpu/ops/modular_tokenization.h"

namespace gjxl::modular_gpu_internal {
inline constexpr size_t kMaximumPopulationContexts = 256;
inline constexpr size_t kPopulationValueLimit = 4096;
inline constexpr size_t kMaximumPopulationScratchBytes = 64 * 1024 * 1024;
inline constexpr size_t kMaximumPopulationShards = 32;
inline constexpr size_t kPopulationTokensPerGroup = 2048;

struct ModularPopulationStoragePlan {
  size_t tokens = 0, contexts = 0, bins = 0, shards = 0, groups = 0;
  size_t status_offset = 0, scratch_bytes = 0, output_bytes = 0;
  size_t retained_bytes = 0, peak_bytes = 0;
  bool operator==(const ModularPopulationStoragePlan &) const = default;
};
struct ModularPopulationDeviceLimits {
  size_t maximum_tokens = 0, maximum_buffer_bytes = 0;
};
// Dense bins describe only [0,4096). The operation reports the exact number of
// wider values separately. All counts are exact; wider values never alias a bin.
// Logical counts are uint64, with uint32 atomic shards bounded by tokens<=UINT32_MAX.
// At most 64 MiB of shard counters and 8 MiB of output at the supported C<=256.
// Input owners and CPU model scratch are excluded; old output is separate.
[[nodiscard]] Status ComputeModularPopulationStoragePlan(
    size_t tokens, size_t contexts, ModularPopulationStoragePlan *out,
    const ModularPopulationDeviceLimits *limits = nullptr);
struct ModularPopulationInput {
  DeviceMemoryRange values, contexts;
  size_t token_count = 0, context_count = 0;
};
// No token reads, allocation, submission or native mapping. Unknown ownership,
// overlap/alignment/size errors are invalid; valid unsupported sizes are explicit.
[[nodiscard]] Status ValidateModularPopulationInput(
    const ModularPopulationInput &input, BackendId backend, ModularPopulationStoragePlan *out,
    const ModularPopulationDeviceLimits *limits = nullptr);

class PreparedGpuModularPopulations {
public:
  virtual ~PreparedGpuModularPopulations() = default;
  PreparedGpuModularPopulations(const PreparedGpuModularPopulations &) = delete;
  PreparedGpuModularPopulations &operator=(const PreparedGpuModularPopulations &) = delete;
  [[nodiscard]] virtual BackendId backend_id() const noexcept = 0;
  [[nodiscard]] virtual size_t token_count() const noexcept = 0;
  [[nodiscard]] virtual size_t context_count() const noexcept = 0;
  [[nodiscard]] virtual uint64_t wide_value_count() const noexcept = 0;
  // Completed, immutable context-major uint64 counts; no copy or extra charge.
  // A model consumer may use these only when wide_value_count()==0. Otherwise
  // it must explicitly use a complete sparse/CPU producer, never partial counts.
  [[nodiscard]] virtual std::span<const uint64_t> counts() const noexcept = 0;
protected:
  PreparedGpuModularPopulations() = default;
};
class GpuModularPopulations {
public:
  virtual ~GpuModularPopulations() = default;
  [[nodiscard]] virtual ModularPopulationDeviceLimits modular_population_limits() const noexcept = 0;
  // Borrowed completed resident tokens stay alive through the joined call.
  // Consumes ambient admission, publishes only after completion/status checks,
  // drains submitted work on failure, and never silently retries on CPU. A
  // successful owner with nonzero wide_value_count is an explicit value-range
  // miss, allowing the workflow to choose its separately accounted CPU path.
  [[nodiscard]] virtual Status ProduceModularPopulations(
      const ModularPopulationInput &input,
      std::unique_ptr<PreparedGpuModularPopulations> *out) = 0;
};
[[nodiscard]] inline GpuModularPopulations *QueryGpuModularPopulations(GpuBackend &backend) noexcept {
  return dynamic_cast<GpuModularPopulations *>(&backend);
}
} // namespace gjxl::modular_gpu_internal
