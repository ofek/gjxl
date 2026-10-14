// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/ops/modular_populations.h"
#include <algorithm>

namespace gjxl::modular_gpu_internal {
Status ComputeModularPopulationStoragePlan(size_t tokens, size_t contexts,
                                           ModularPopulationStoragePlan *out,
                                           const ModularPopulationDeviceLimits *limits) {
  if (!out || !tokens || !contexts || contexts > 65536 ||
      (limits && (!limits->maximum_tokens || !limits->maximum_buffer_bytes)))
    return Status::InvalidArgument("Invalid Modular population shape/limits");
  if (tokens > size_t(PTRDIFF_MAX) / sizeof(uint32_t))
    return Status::InvalidArgument("Modular population token extent overflow");
  if (tokens > UINT32_MAX || contexts > kMaximumPopulationContexts)
    return Status::Unsupported("Modular population shape exceeds the bounded dense operation");
  ModularPopulationStoragePlan p;
  p.tokens = tokens;
  p.contexts = contexts;
  p.bins = contexts * kPopulationValueLimit;
  p.groups = tokens / kPopulationTokensPerGroup + (tokens % kPopulationTokensPerGroup != 0);
  p.shards = std::min({p.groups, kMaximumPopulationShards,
      kMaximumPopulationScratchBytes / (p.bins * sizeof(uint32_t))});
  p.status_offset = p.shards * p.bins * sizeof(uint32_t);
  p.scratch_bytes = p.status_offset + 2 * sizeof(uint32_t); // invalid context, wide count
  p.output_bytes = p.bins * sizeof(uint64_t);
  p.retained_bytes = p.output_bytes;
  p.peak_bytes = p.scratch_bytes + p.output_bytes;
  if (limits && (tokens > limits->maximum_tokens || p.scratch_bytes > limits->maximum_buffer_bytes ||
                 p.output_bytes > limits->maximum_buffer_bytes))
    return Status::Unsupported("Modular population storage exceeds backend limits");
  *out = p;
  return Status::Ok();
}
Status ValidateModularPopulationInput(const ModularPopulationInput &input, BackendId backend,
                                      ModularPopulationStoragePlan *out,
                                      const ModularPopulationDeviceLimits *limits) {
  if (!out || !backend)
    return Status::InvalidArgument("Invalid Modular population output/backend");
  ModularPopulationStoragePlan plan;
  if (auto s = ComputeModularPopulationStoragePlan(input.token_count, input.context_count, &plan,
                                                   limits); !s.ok()) return s;
  const auto valid = [&](DeviceMemoryRange range, size_t element) {
    return range.buffer && range.buffer->backend_id() == backend &&
        range.offset_bytes % element == 0 && range.size_bytes == input.token_count * element &&
        range.offset_bytes <= range.buffer->size_bytes() &&
        range.size_bytes <= range.buffer->size_bytes() - range.offset_bytes;
  };
  if (!valid(input.values, sizeof(uint32_t)) || !valid(input.contexts, sizeof(uint16_t)))
    return Status::InvalidArgument("Invalid Modular population resident range");
  if (input.values.buffer == input.contexts.buffer &&
      input.values.offset_bytes < input.contexts.offset_bytes + input.contexts.size_bytes &&
      input.contexts.offset_bytes < input.values.offset_bytes + input.values.size_bytes)
    return Status::InvalidArgument("Modular population resident ranges overlap");
  *out = plan;
  return Status::Ok();
}
} // namespace gjxl::modular_gpu_internal
