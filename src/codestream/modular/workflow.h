// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codec/modular/frame.h"
#include "codec/modular/tree_learning.h"
#include "codestream/entropy.h"
#include "core/execution_domain.h"

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
  // CPU experiment: select the ANS model without ordered candidate simulation.
  AnsSelectionMode ans_selection = AnsSelectionMode::kExact;
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
