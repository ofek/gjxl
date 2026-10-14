// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/frame.h"
#include "codec/modular/tree_learning.h"
#include "codec/modular/weighted_features.h"
#include "codestream/entropy.h"
#include "codestream/modular/candidate_cost.h"
#include "core/execution_domain.h"

namespace gjxl { class GpuBackend; }

namespace gjxl::modular_internal {
struct ModularEncodingOptions {
  std::shared_ptr<const ExecutionDomain> execution_domain;
  EntropyCodingMode entropy = EntropyCodingMode::kPrefix;
  ModularCodingPolicy coding;
  bool search = false;
  size_t cpu_thread_count = 1; // Serial oracle default; zero selects up to four.
  // Experimental CPU search budget. Public API defaults remain unchanged.
  TreeLearningOptions learning;
  // Compare 16/32/... leaf checkpoints from each experimental learning pass.
  bool tree_checkpoints = false;
  // Private hybrid qualification route. Backend is borrowed for this complete
  // call; feature reuse is call-local. Public defaults stay on their old paths.
  GpuBackend *adaptive_gpu = nullptr;
  // Private qualification switch; requires adaptive_gpu. The old hybrid route
  // remains available as a paired control while exact ANS costing is qualified.
  bool adaptive_ans_cost = false;
  // Private original-learner statistics route; requires adaptive_gpu and search.
  bool adaptive_tree_statistics = false;
  // Share one completed feature layout between learning and token generation.
  // Applies only to adaptive_gpu search; false retains the scalar learning
  // gather as a paired control. CPU-only and prescribed routes ignore it.
  bool adaptive_shared_features = true;
  // Private weighted-feature producer selection, qualified independently of
  // GPU token generation. Auto selects Metal only for qualified large layouts;
  // smaller/unsupported layouts use the admitted CPU participants.
  WeightedFeatureBackend weighted_feature_backend = WeightedFeatureBackend::kAuto;
  // Private exact population producer. Wide/small/unsupported inputs retain the
  // complete CPU producer; device errors propagate without a hidden retry.
  bool adaptive_populations = false;
  // Private selector: estimated mode omits ordered candidate simulation unless
  // an exact cost is requested. Applies equally to CPU and hybrid token paths.
  AnsSelectionMode ans_selection = AnsSelectionMode::kExact;
  ModularSearchEvaluation search_evaluation = ModularSearchEvaluation::kEncodeAll;
  ModularCandidateObserver *candidate_observer = nullptr;
};
[[nodiscard]] Status EncodeRgb8ModularOwned(Rgb8View input, const ModularEncodingOptions &options,
                                            codestream_internal::CodestreamBuffer *out);
[[nodiscard]] Status EncodeRgb8Modular(Rgb8View input, const ModularEncodingOptions &options,
                                       std::vector<uint8_t> *out);
[[nodiscard]] Status EncodeModularImageOwned(PackedModularImageView input,
                                             const ModularEncodingOptions &options,
                                             codestream_internal::CodestreamBuffer *out);
[[nodiscard]] Status EncodeModularImage(PackedModularImageView input,
                                        const ModularEncodingOptions &options,
                                        std::vector<uint8_t> *out);
} // namespace gjxl::modular_internal
