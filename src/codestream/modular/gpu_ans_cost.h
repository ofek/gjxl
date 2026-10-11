// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#pragma once
#include "codestream/ans_internal.h"
#include "codestream/modular/tokenization.h"
#include "gpu/ops/ans_token_cost.h"

namespace gjxl::modular_internal {
// Call-local adapter; tokens/backend and their admitted backing outlive it.
// No global/thread-local dispatch override and no retained model cache.
class ModularGpuAnsCost final : public codestream_internal::AnsTokenCostEvaluator {
public:
  ModularGpuAnsCost(GpuBackend &backend, const PreparedModularTokens &tokens)
      : backend_(backend), operation_(ans_gpu_internal::QueryGpuAnsTokenCost(backend)),
        tokens_(tokens) {}
  [[nodiscard]] Status TryMeasure(std::span<const EntropyTokenStreamView> sections,
                                  std::span<const EntropyCode *const> codes,
                                  std::span<const uint64_t> model_bits,
                                  std::span<EntropyCodeCost> costs, bool *handled) override;

private:
  GpuBackend &backend_;
  ans_gpu_internal::GpuAnsTokenCost *operation_;
  const PreparedModularTokens &tokens_;
};
} // namespace gjxl::modular_internal
