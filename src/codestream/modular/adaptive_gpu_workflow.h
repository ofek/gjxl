// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/modular/storage_plan.h"
#include "codestream/modular/workflow.h"
#include "gpu/ops/modular_adaptive_tokens.h"

namespace gjxl::modular_internal {
struct ModularCandidateEvaluation;
struct AdaptiveModularWorkflowStoragePlan {
  ModularWorkflowStoragePlan cpu;
  size_t additional_peak_bytes = 0, peak_bytes = 0;
  bool operator==(const AdaptiveModularWorkflowStoragePlan &) const = default;
};
// Checked, allocation-free bound for complete prescribed/search calls. Adds a
// conservative maximum of feature preparation/generation plus canonical metadata
// to the existing CPU bound. That sum covers cached features during all training,
// CPU fallback, modeling, emission and output publication. Old output is separate.
[[nodiscard]] Status
ComputeAdaptiveModularWorkflowStoragePlan(Extent2D extent, PackedModularFormat format,
                                          const ModularEncodingOptions &options,
                                          AdaptiveModularWorkflowStoragePlan *out);

// One immutable source and one live layout per encoder call. Input backing and
// backend outlive the cache. Tree and entropy mode are deliberately not keys.
// Eviction releases the old owner before constructing a different layout.
// This helper does not admit a call: the caller owns the composed reservation,
// CPU execution scope and failure-atomic public output publication.
class AdaptiveModularTokenCache {
public:
  AdaptiveModularTokenCache(GpuBackend &backend, PackedModularImageView input,
                            bool anticipate_weighted = true, bool ans_cost = false)
      : operation_(modular_gpu_internal::QueryGpuAdaptiveModularTokenGeneration(backend)),
        input_(input), anticipate_weighted_(anticipate_weighted), ans_backend_(ans_cost ? &backend : nullptr) {}
  [[nodiscard]] Status Encode(const ModularCodingPolicy &policy, EntropyCodingMode mode,
                              const ModularWorkflowStoragePlan &storage,
                              codestream_internal::CodestreamBuffer *out,
                              AnsSelectionMode selection = AnsSelectionMode::kExact,
                              ModularCandidateEvaluation *evaluation = nullptr);

private:
  [[nodiscard]] Status Prepare(const ModularCodingPolicy &policy, ModularEncoderFrame *fallback);
  void Evict();
  modular_gpu_internal::GpuAdaptiveModularTokenGeneration *operation_;
  const PackedModularImageView input_;
  const bool anticipate_weighted_;
  GpuBackend *const ans_backend_;
  uint8_t rct_ = 0;
  TransformSequence transforms_;
  WeightedPredictorParameters weighted_;
  ChannelShape shape_;
  ModularStreamPlan layout_;
  std::unique_ptr<modular_gpu_internal::PreparedGpuAdaptiveModularFeatures> features_;
};
} // namespace gjxl::modular_internal
