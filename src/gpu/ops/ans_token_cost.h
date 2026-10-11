// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "core/host_storage_bound.h"
#include "gpu/backend.h"
#include "gpu/image.h"
#include <span>

namespace gjxl::ans_gpu_internal {
inline constexpr size_t kMaximumCostCandidates = 4;
inline constexpr size_t kMaximumCostContexts = 65536; // uint16 resident contexts
inline constexpr size_t kMaximumCostClusters = 64;
inline constexpr size_t kCostAlphabetSize = 256;
inline constexpr size_t kCostTableSize = 4096;
inline constexpr uint32_t kCostInitialState = 0x13u << 16;

// Sections partition the resident token arena in order. Empty sections retain
// their place and cost 32 bits, just as the scalar entropy cost measurement
// does.
struct AnsTokenCostSection {
  uint64_t offset = 0, count = 0;
  bool operator==(const AnsTokenCostSection &) const = default;
};
struct AnsTokenCostResult {
  uint64_t bits = 0;
  uint32_t final_state = 0, reserved = 0;
  bool operator==(const AnsTokenCostResult &) const = default;
};
struct AnsTokenCostClusterView {
  uint32_t split_exponent = 0, msb_in_token = 0, lsb_in_token = 0;
  std::span<const uint16_t> frequencies, reverse_offsets, reverse_map;
  std::span<const uint64_t> reciprocal_frequencies;
};
struct AnsTokenCostModelView {
  std::span<const uint8_t> context_map;
  std::span<const AnsTokenCostClusterView> clusters;
};

// Borrowed, completed split tokens. Both ranges have exactly the advertised
// logical size and correct alignment; the operation also verifies native buffer
// ownership and bounds. The producer's immutable owner outlives the call.
struct AnsTokenCostInput {
  DeviceMemoryRange values;   // uint32
  DeviceMemoryRange contexts; // uint16
  size_t token_count = 0;
  std::span<const AnsTokenCostSection> sections;
  std::span<const AnsTokenCostModelView> models;
};

// Fixed device table slot: four uint32 configuration/symbol-count words,
// 256 uint16 frequencies, 256 uint16 reverse offsets, 256 uint64 reciprocals,
// 4096 uint16 reverse-map entries, then zero padding to 256-byte alignment.
inline constexpr size_t kCostClusterBytes = 11520;
inline constexpr size_t kCostModelBytes = 32;
struct AnsTokenCostStoragePlan {
  size_t token_count = 0, section_count = 0, candidate_count = 0;
  size_t maximum_contexts = 0, maximum_clusters = 0;
  size_t sections_bytes = 0, models_offset = 0, context_maps_offset = 0;
  size_t clusters_offset = 0, status_offset = 0, dispatch_bytes = 0;
  size_t output_bytes = 0, retained_bytes = 0, peak_bytes = 0;
  bool operator==(const AnsTokenCostStoragePlan &) const = default;
};
struct AnsTokenCostDeviceLimits {
  size_t maximum_tokens = 0, maximum_sections = 0, maximum_buffer_bytes = 0;
};
// Allocation-free, failure-atomic shape bound. Context/cluster counts are
// maxima across the batch (also useful before models exist). Covers device
// dispatch, output, one host dispatch staging copy, and fresh caller
// section/cluster views. Excludes borrowed tokens/models, the old output,
// backend setup/driver memory, and entropy optimizer storage; the workflow must
// compose those live owners.
[[nodiscard]] Status
ComputeAnsTokenCostStoragePlan(size_t tokens, size_t sections, size_t candidates,
                               size_t maximum_contexts, size_t maximum_clusters,
                               AnsTokenCostStoragePlan *out,
                               const AnsTokenCostDeviceLimits *limits = nullptr);
// Allocation-free validation of borrowed descriptors and immutable model data.
// No token scan, backend access, allocation or submission. Model tables must
// match the scalar recurrence's normalized frequencies and exact reciprocals.
// Actual context/symbol/range errors are checked by the device before
// publication.
[[nodiscard]] Status ValidateAnsTokenCostSections(size_t tokens,
                                                  std::span<const AnsTokenCostSection> sections);
[[nodiscard]] Status ValidateAnsTokenCostModels(std::span<const AnsTokenCostModelView> models);
[[nodiscard]] Status ValidateAnsTokenCostInput(const AnsTokenCostInput &input, BackendId backend,
                                               AnsTokenCostStoragePlan *out,
                                               const AnsTokenCostDeviceLimits *limits = nullptr);
inline constexpr uint32_t kCostContextError = 1, kCostSymbolError = 2, kCostStateError = 4;
[[nodiscard]] Status CheckAnsTokenCostDeviceStatus(uint32_t status);

class PreparedGpuAnsTokenCosts {
public:
  virtual ~PreparedGpuAnsTokenCosts() = default;
  PreparedGpuAnsTokenCosts(const PreparedGpuAnsTokenCosts &) = delete;
  PreparedGpuAnsTokenCosts &operator=(const PreparedGpuAnsTokenCosts &) = delete;
  [[nodiscard]] virtual BackendId backend_id() const noexcept = 0;
  [[nodiscard]] virtual size_t section_count() const noexcept = 0;
  [[nodiscard]] virtual size_t candidate_count() const noexcept = 0;
  // Completed host-readable results in candidate-major, then section order.
  // No copy/allocation/extra charge; this owner must outlive the borrowed span.
  [[nodiscard]] virtual std::span<const AnsTokenCostResult> results() const noexcept = 0;

protected:
  PreparedGpuAnsTokenCosts() = default;
};
// Optional private capability, not an entropy policy. Each (candidate, section)
// has a separate reverse-order serial rANS state; these states run
// independently. Includes HybridUint extra bits, 16-bit renormalizations and
// 32-bit final state. Excludes model bits, byte padding and final emission.
// Greedy model selection, equivalent-width pruning, candidate ordering and ties
// stay on the caller. Synchronous, failure-atomic, consumes ambient admission.
// No CPU retry after dispatch. Drain submitted work before releasing any
// borrowed or owned backing.
class GpuAnsTokenCost {
public:
  virtual ~GpuAnsTokenCost() = default;
  [[nodiscard]] virtual AnsTokenCostDeviceLimits ans_token_cost_limits() const noexcept = 0;
  [[nodiscard]] virtual Status
  MeasureAnsTokenCosts(const AnsTokenCostInput &input,
                       std::unique_ptr<PreparedGpuAnsTokenCosts> *out) = 0;
};
[[nodiscard]] inline GpuAnsTokenCost *QueryGpuAnsTokenCost(GpuBackend &backend) noexcept {
  return dynamic_cast<GpuAnsTokenCost *>(&backend);
}
} // namespace gjxl::ans_gpu_internal
