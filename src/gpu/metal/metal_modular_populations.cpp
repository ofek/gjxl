// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/metal/kernels/modular_population_params.h"
#include "gpu/metal/metal_backend_internal.h"
#include "gpu/ops/modular_populations.h"
#include <cstring>

namespace gjxl::metal_internal {
using namespace modular_gpu_internal;
using namespace resource_budget_internal;
namespace {
using namespace gjxl_modular_population;
static_assert(sizeof(Params) == 24 && kValueLimit == kPopulationValueLimit &&
              kGroupTokens == kPopulationTokensPerGroup);
class Completed final : public PreparedGpuModularPopulations {
public:
  BackendId backend_id() const noexcept override { return backend; }
  size_t token_count() const noexcept override { return tokens; }
  size_t context_count() const noexcept override { return contexts; }
  uint64_t wide_value_count() const noexcept override { return wide; }
  std::span<const uint64_t> counts() const noexcept override { return host; }
  BackendId backend = 0;
  size_t tokens = 0, contexts = 0;
  uint64_t wide = 0;
  std::unique_ptr<DeviceBuffer> buffer;
  std::span<const uint64_t> host;
};
struct Pending {
  std::unique_ptr<GpuSubmission> submission;
  bool completed = false;
  ~Pending() { if (submission && !completed) DrainMetalSubmission(*submission); }
  Status Wait() { auto s = submission->Wait(); completed = true; return s; }
};
struct Context {
  MTL::Buffer *values, *contexts, *scratch, *output;
  size_t values_offset, contexts_offset, groups;
  MTL::ComputePipelineState *count, *reduce;
  Params params;
};
void Encode(MetalBackend &, MTL::ComputeCommandEncoder *encoder, const void *opaque) {
  const auto &c = *static_cast<const Context *>(opaque);
  encoder->setBuffer(c.values, c.values_offset, 0);
  encoder->setBuffer(c.contexts, c.contexts_offset, 1);
  encoder->setBuffer(c.scratch, 0, 2);
  encoder->setBuffer(c.output, 0, 3);
  encoder->setBytes(&c.params, sizeof(c.params), 4);
  encoder->setComputePipelineState(c.count);
  RecordMetalComputePipelineState(c.count);
  DispatchMetalThreadgroups(encoder, MTL::Size(c.groups, 1, 1), MTL::Size(kThreads, 1, 1));
  const MTL::Resource *dependencies[] = {c.scratch};
  encoder->memoryBarrier(dependencies, 1);
  encoder->setComputePipelineState(c.reduce);
  RecordMetalComputePipelineState(c.reduce);
  DispatchMetalThreads(encoder, MTL::Size(c.params.bins, 1, 1), MTL::Size(kThreads, 1, 1));
}
} // namespace
ModularPopulationDeviceLimits MetalBackend::modular_population_limits() const noexcept {
  return {UINT32_MAX, size_t(device_->maxBufferLength())};
}
Status MetalBackend::ProduceModularPopulations(const ModularPopulationInput &input,
    std::unique_ptr<PreparedGpuModularPopulations> *out) try {
  if (!out) return Status::InvalidArgument("Null Modular population output");
  ModularPopulationStoragePlan plan;
  const auto limits = modular_population_limits();
  if (auto s = ValidateModularPopulationInput(input, id(), &plan, &limits); !s.ok()) return s;
  const MetalBuffer *values = nullptr, *contexts = nullptr;
  if (auto s = RequireMetalBuffer(input.values.buffer, input.values.size_bytes,
          input.values.offset_bytes, "Modular population values", &values); !s.ok()) return s;
  if (auto s = RequireMetalBuffer(input.contexts.buffer, input.contexts.size_bytes,
          input.contexts.offset_bytes, "Modular population contexts", &contexts); !s.ok()) return s;
  const ManagedHostScope resources(ResourceClass::kPreparation);
  auto result = std::make_unique<Completed>();
  result->backend = id();
  result->tokens = plan.tokens;
  result->contexts = plan.contexts;
  std::unique_ptr<DeviceBuffer> scratch;
  Pending pending;
  {
    const ResourceClassScope retained(ResourceClass::kCompletedFrame);
    if (auto s = Allocate(plan.output_bytes, &result->buffer); !s.ok()) return s;
  }
  if (auto s = Allocate(plan.scratch_bytes, &scratch); !s.ok()) return s;
  auto *bytes = static_cast<std::byte *>(AsMetalBuffer(*scratch)->contents());
  std::memset(bytes, 0, plan.scratch_bytes);
  Context context{values->handle(), contexts->handle(), AsMetalBuffer(*scratch)->handle(),
      AsMetalBuffer(*result->buffer)->handle(), input.values.offset_bytes, input.contexts.offset_bytes,
      plan.groups, modular_population_pipelines_[0].get(), modular_population_pipelines_[1].get(),
      {plan.status_offset, uint32_t(plan.tokens), uint32_t(plan.contexts), uint32_t(plan.bins),
       uint32_t(plan.shards)}};
  if (auto s = SubmitCompute("modular.populations", Encode, &context, &pending.submission); !s.ok()) return s;
  if (!pending.submission) return Status::Internal("Modular population operation returned no submission");
  if (auto s = pending.Wait(); !s.ok()) return s;
  uint32_t status[2];
  std::memcpy(status, bytes + plan.status_offset, sizeof(status));
  if (status[0] & ~1u) return Status::Internal("Unknown Modular population device status");
  if (status[0]) return Status::InvalidArgument("Modular population context is out of range");
  if (status[1] > plan.tokens) return Status::Internal("Modular population wide count exceeds tokens");
  result->wide = status[1];
  std::span<const std::byte> completed;
  if (auto s = BorrowCompletedReadOnly(*result->buffer, plan.output_bytes, 0, &completed); !s.ok()) return s;
  result->host = {reinterpret_cast<const uint64_t *>(completed.data()), plan.bins};
  pending.submission.reset();
  scratch.reset();
  *out = std::move(result);
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Modular population allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("Modular population storage overflow");
}
} // namespace gjxl::metal_internal
