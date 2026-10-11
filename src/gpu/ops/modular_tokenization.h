// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include "gpu/ops/modular_input.h"
#include <array>

namespace gjxl::modular_gpu_internal {
inline constexpr size_t kMaximumGpuTreeNodes = 31;

struct ModularTokenizationOptions {
  modular_internal::PackedModularFormat format = modular_internal::PackedModularFormat::kRgb8;
  modular_internal::ModularCodingPolicy coding;
  bool search = false;
  bool operator==(const ModularTokenizationOptions &) const = default;
};

// Fixed-width descriptor ABI. Coordinates/neighbors are local to this rectangle;
// x/y are its origin in the full resident channel, not tree properties 3/2.
// Each slice writes width*height tokens at token_offset, in row-major order.
struct ModularTokenSlice {
  uint64_t token_offset = 0;
  uint32_t stream_id = 0, channel = 0, x = 0, y = 0, width = 0, height = 0;
  bool operator==(const ModularTokenSlice &) const = default;
};
struct ModularTokenRange {
  size_t offset = 0, count = 0;
  bool operator==(const ModularTokenRange &) const = default;
};
// Original node indices/children; contexts are assigned in scalar BFS leaf order.
// Only the first node_count records may be visited. Unused records are zero.
struct ModularTokenNode {
  int32_t property = 0, split = 0;
  uint32_t left = 0, right = 0, predictor = 0;
  int32_t offset = 0;
  uint32_t multiplier = 0, context = 0;
  bool operator==(const ModularTokenNode &) const = default;
};
struct ModularTokenEligibility {
  size_t channel_count = 0, context_count = 0, node_count = 0;
  std::array<ModularTokenNode, kMaximumGpuTreeNodes> nodes{};
  bool gradient_leaf = false;
  bool operator==(const ModularTokenEligibility &) const = default;
};

// Allocation-free on success; no samples/device required. Invalid format/RCT,
// malformed tree, weighted parameters or transforms => InvalidArgument. Valid
// search/additional transforms, weighted leaves or property 15 => Unsupported.
// Supports all 13 other predictors, properties 0..14 and all valid signed leaf
// offsets/positive multipliers. Representability depends on samples and is NOT
// proven here: residual%multiplier must be zero and its quotient fit int32.
// Any failure preserves *out; format also gates grayscale RCT to identity.
[[nodiscard]] Status ResolveModularTokenEligibility(const ModularTokenizationOptions &options,
                                                    ModularTokenEligibility *out);
// Borrow a potentially large CPU policy before constructing owning GPU options.
// Rejection must not allocate a tree copy before workflow admission.
[[nodiscard]] Status
ResolveModularTokenEligibility(modular_internal::PackedModularFormat format,
                               const modular_internal::ModularCodingPolicy &coding, bool search,
                               ModularTokenEligibility *out);

// Inclusive limits of an initialized backend. Every field must be nonzero.
// maximum_buffer_bytes applies independently to token and descriptor/status arenas.
struct ModularTokenDeviceLimits {
  size_t maximum_tokens = 0, maximum_slices = 0, maximum_buffer_bytes = 0;
};

// Status word starts at zero. Kernels OR this bit for nonintegral/out-of-int32
// residual quotients; after successful completion it maps to InvalidArgument.
// Unknown bits indicate a producer/consumer protocol error (Internal).
inline constexpr uint32_t kModularTokenResidualError = 1;
[[nodiscard]] Status CheckModularTokenDeviceStatus(uint32_t status);

struct ModularTokenHostView {
  std::span<const uint32_t> values;
  std::span<const uint16_t> contexts;
};

// Completed immutable packed signed residuals and leaf contexts. Owns the token
// arena/charge only: no input planes, upload, dispatch scratch or pending work.
// Exactly one token per sample, in canonical stream/channel/row-major order,
// equal to TokenizeModular for the same prescribed policy and stream plan.
// Backend outlives the owner; the owner outlives EVERY CPU consumer/worker.
class PreparedGpuModularTokens {
public:
  virtual ~PreparedGpuModularTokens() = default;
  PreparedGpuModularTokens(const PreparedGpuModularTokens &) = delete;
  PreparedGpuModularTokens &operator=(const PreparedGpuModularTokens &) = delete;
  [[nodiscard]] virtual BackendId backend_id() const noexcept = 0;
  [[nodiscard]] virtual size_t token_count() const noexcept = 0;
  [[nodiscard]] virtual size_t context_count() const noexcept = 0;
  [[nodiscard]] virtual DeviceMemoryRange values() const noexcept = 0;   // uint32
  [[nodiscard]] virtual DeviceMemoryRange contexts() const noexcept = 0; // uint16
  // Optional completed shared-buffer borrowing; no allocation, copy, submission
  // or new charge. Unsupported if backing is not host-readable. Success returns
  // token_count() elements in each span; failure preserves *out. Never maps an
  // arbitrary device view or publishes views before the status word is checked.
  [[nodiscard]] virtual Status BorrowHostView(ModularTokenHostView *out) const = 0;

protected:
  PreparedGpuModularTokens() = default;
};

// Optional private capability. Metal implements it; other backends return null.
class GpuModularTokenGeneration {
public:
  virtual ~GpuModularTokenGeneration() = default;
  [[nodiscard]] virtual ModularTokenDeviceLimits modular_token_limits() const noexcept = 0;
  // Caller supplies canonical descriptors from BuildGpuModularTokenPlan. Backend
  // validates eligibility, native ownership, matching source format/shape/RCT,
  // counts, addressed ranges and output nonaliasing before allocation/submission.
  // The prepared input has the source-derived int32 bounds of unpack/RCT; this
  // is not a general transformed-plane API. Input remains immutable/alive until
  // return. Neighbors reset at each slice, previous gradient at each row. Property
  // 8 recomputes the previous pixel's unclamped gradient with scalar edge rules.
  // Predictor division truncates toward zero; use wide intermediates for leaf
  // offsets/division, exact remainder/range checks and unsigned signed packing.
  // Synchronous, failure-atomic; consumes the ambient reservation. A successful
  // return replaces *out with a nonnull completed owner after status validation.
  // Failure preserves prior output and drains work before releasing backing and
  // tickets. Old output is charged separately. No silent CPU retry/fallback.
  [[nodiscard]] virtual Status
  GenerateModularTokens(const PreparedGpuModularInput &input,
                        const ModularTokenizationOptions &options,
                        std::span<const ModularTokenSlice> slices,
                        std::unique_ptr<PreparedGpuModularTokens> *out) = 0;
  // Composed packed-input producer: same completed-owner/failure contract, but
  // records unpack/RCT and tokens with an explicit dependency in one submission
  // and waits once. Upload/planes/dispatch are released before publication.
  // Validates packed backing, matching format and all descriptors before any
  // allocation. Consumes the ambient full-call reservation, without readback,
  // CPU tokenization or a fallback. The caller keeps packed input immutable
  // until return; old token output remains separately charged.
  [[nodiscard]] virtual Status
  PrepareModularTokens(modular_internal::PackedModularImageView input,
                       const ModularTokenizationOptions &options,
                       std::span<const ModularTokenSlice> slices,
                       std::unique_ptr<PreparedGpuModularTokens> *out) = 0;
};
[[nodiscard]] inline GpuModularTokenGeneration *
QueryGpuModularTokenGeneration(GpuBackend &backend) noexcept {
  return dynamic_cast<GpuModularTokenGeneration *>(&backend);
}
} // namespace gjxl::modular_gpu_internal
