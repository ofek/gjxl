// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once

#include "codec/modular/image.h"
#include "codec/modular/prediction.h"
#include <chrono>

namespace gjxl::modular_internal {
// Optional caller-wall diagnostics, accumulated across preparations. These are
// disjoint subsets of the enclosing feature preparation interval, not extra
// top-level stages. Validation, traversal and scratch cleanup remain in that parent.
// A standalone scalar slice uses only scratch_seconds and prediction_seconds.
// Never share a mutable sink between workers; a future parallel executor must
// time the joined interval on its caller (worker-time sums are not wall time).
struct FeaturePreparationProfile {
  double allocation_seconds = 0;  // Completed owner, descriptors and buffers.
  double copy_seconds = 0;        // Plane arena initialization and sample copy.
  double scratch_seconds = 0;     // Predictor row allocation and initialization.
  double prediction_seconds = 0;  // Neighborhoods, Predict/Update and output stores.
  double publication_seconds = 0; // Completed host view and owner publication.
  bool operator==(const FeaturePreparationProfile &) const = default;
};
class FeaturePreparationScope {
  using Clock = std::chrono::steady_clock;

public:
  FeaturePreparationScope(FeaturePreparationProfile *profile,
                          double FeaturePreparationProfile::*field)
      : sink_(profile ? &(profile->*field) : nullptr) {
    if (sink_)
      start_ = Clock::now();
  }
  ~FeaturePreparationScope() { Stop(); }
  void Stop() {
    if (sink_) {
      *sink_ += std::chrono::duration<double>(Clock::now() - start_).count();
      sink_ = nullptr;
    }
  }
  FeaturePreparationScope(const FeaturePreparationScope &) = delete;
  FeaturePreparationScope &operator=(const FeaturePreparationScope &) = delete;

private:
  double *sink_;
  Clock::time_point start_{};
};

// Allocation-free, checked bound for five fresh two-row arrays per CPU
// participant. Width is the largest admitted slice width; participants is in
// [1, 256]. No output map, input planes or scheduler storage is included.
// Failure leaves *out unchanged. The serial producer below uses one participant.
[[nodiscard]] Status
ComputeWeightedPredictorScratchBound(size_t maximum_width, size_t participants,
                                     resource_budget_internal::HostStorageBound *out);

// Serial reference producer for ONE independent canonical channel slice.
// The borrowed view starts at the rectangle origin and retains the source
// stride; all neighborhood edges and predictor resets are slice-local. Explicit
// parameters are part of feature identity. Output has exactly width*height
// entries, stored in slice raster order as {prediction, property15} (16 bytes).
//
// The caller validates canonical slice coverage/offsets and supplies a disjoint
// output subspan at that slice's offset in its unpublished feature owner. Input
// backing must remain immutable and must not alias output. No learner, policy
// search, GPU backend, tokenization or full-image staging is involved here.
//
// Shape, parameter and output-size errors are rejected before allocation or
// writes. Scratch uses the ambient reservation, charged to kPreparation, and is
// released before return. Runtime range failure returns Unsupported and may
// leave a partial slice: discard the unpublished owner, never expose it or retry
// after execution starts. The owner publishes only after ALL slices succeed.
[[nodiscard]] Status PrepareWeightedPredictorFeatures(ModularChannelView slice,
                                                      const WeightedPredictorParameters &parameters,
                                                      std::span<WeightedPredictorFeature> output,
                                                      FeaturePreparationProfile *profile = nullptr);
} // namespace gjxl::modular_internal
