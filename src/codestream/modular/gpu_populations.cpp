// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/gpu_populations.h"
#include "codec/modular/profile.h"
namespace gjxl::modular_internal {
using namespace modular_gpu_internal;
Status PrepareModularPopulations(GpuBackend &backend, const PreparedModularTokens &tokens,
    std::unique_ptr<PreparedGpuModularPopulations> *out) {
  if (!out) return Status::InvalidArgument("Null Modular population adapter output");
  auto *profile = active_modular_profile ? &active_modular_profile->adaptive_gpu : nullptr;
  auto *operation = QueryGpuModularPopulations(backend);
  if (!operation || !tokens.gpu_backing || tokens.context_count > kMaximumPopulationContexts ||
      tokens.gpu_backing->token_count() > UINT32_MAX) {
    if (profile) ++profile->population_cpu_unsupported;
    out->reset();
    return Status::Ok();
  }
  const size_t count = tokens.gpu_backing->token_count();
  if (count < 262144 || count < 4 * tokens.context_count * kPopulationValueLimit) {
    if (profile) ++profile->population_cpu_small;
    out->reset();
    return Status::Ok();
  }
  const auto start = std::chrono::steady_clock::now();
  std::unique_ptr<PreparedGpuModularPopulations> result;
  const auto status = operation->ProduceModularPopulations(
      {tokens.gpu_backing->values(), tokens.gpu_backing->contexts(), count, tokens.context_count}, &result);
  if (profile) profile->population_seconds +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  if (status.code() == StatusCode::kUnsupported) {
    if (profile) ++profile->population_cpu_unsupported;
    out->reset();
    return Status::Ok();
  }
  if (!status.ok()) return status;
  if (!result || result->backend_id() != backend.id() || result->token_count() != count ||
      result->context_count() != tokens.context_count ||
      result->counts().size() != tokens.context_count * kPopulationValueLimit)
    return Status::Internal("Completed Modular population shape/ownership differs");
  if (profile) {
    ++profile->population_gpu_calls;
    profile->population_tokens += count;
  }
  if (result->wide_value_count()) {
    if (profile) ++profile->population_cpu_wide;
    result.reset();
  }
  *out = std::move(result);
  return Status::Ok();
}
} // namespace gjxl::modular_internal
