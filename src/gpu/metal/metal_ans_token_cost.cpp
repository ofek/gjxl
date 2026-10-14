// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/metal/kernels/ans_token_cost_params.h"
#include "gpu/metal/metal_backend_internal.h"
#include <cstring>

namespace gjxl::metal_internal {
using namespace ans_gpu_internal;
using namespace resource_budget_internal;
namespace {
using namespace gjxl_ans_cost;
static_assert(sizeof(Section) == sizeof(AnsTokenCostSection) && alignof(Section) == 8);
static_assert(sizeof(Result) == sizeof(AnsTokenCostResult) && alignof(Result) == 8);
static_assert(offsetof(Result, final_state) == offsetof(AnsTokenCostResult, final_state));
static_assert(sizeof(Model) == kCostModelBytes && sizeof(Cluster) == kCostClusterBytes);
static_assert(offsetof(Cluster, frequencies) == 16 && offsetof(Cluster, reverse_offsets) == 528 &&
              offsetof(Cluster, reciprocals) == 1040 && offsetof(Cluster, reverse_map) == 3088);
static_assert(sizeof(Params) == 24 && kContextError == kCostContextError &&
              kSymbolError == kCostSymbolError && kStateError == kCostStateError);
class Completed final : public PreparedGpuAnsTokenCosts {
public:
  BackendId backend_id() const noexcept override { return backend; }
  size_t section_count() const noexcept override { return sections; }
  size_t candidate_count() const noexcept override { return candidates; }
  std::span<const AnsTokenCostResult> results() const noexcept override { return host; }
  BackendId backend = 0;
  size_t sections = 0, candidates = 0;
  std::unique_ptr<DeviceBuffer> buffer;
  std::span<const AnsTokenCostResult> host;
};
struct Pending {
  std::unique_ptr<GpuSubmission> submission;
  bool completed = false;
  ~Pending() {
    if (submission && !completed)
      DrainMetalSubmission(*submission);
  }
  Status Wait() {
    auto s = submission->Wait();
    completed = true;
    return s;
  }
};
struct Context {
  MTL::Buffer *values, *contexts, *dispatch, *output;
  size_t values_offset, contexts_offset;
  MTL::ComputePipelineState *pipeline;
  Params params;
};
void Encode(MetalBackend &, MTL::ComputeCommandEncoder *encoder, const void *opaque) {
  const auto &c = *static_cast<const Context *>(opaque);
  encoder->setComputePipelineState(c.pipeline);
  RecordMetalComputePipelineState(c.pipeline);
  encoder->setBuffer(c.values, c.values_offset, 0);
  encoder->setBuffer(c.contexts, c.contexts_offset, 1);
  encoder->setBuffer(c.dispatch, 0, 2);
  encoder->setBuffer(c.output, 0, 3);
  encoder->setBytes(&c.params, sizeof(c.params), 4);
  DispatchMetalThreads(encoder,
                       MTL::Size(size_t(c.params.sections) * c.pipeline->threadExecutionWidth(),
                                 c.params.candidates, 1),
                       MTL::Size(64, 1, 1));
}
} // namespace
AnsTokenCostDeviceLimits MetalBackend::ans_token_cost_limits() const noexcept {
  return {size_t(PTRDIFF_MAX) / sizeof(uint32_t),
          UINT32_MAX / size_t(ans_token_cost_pipeline_->threadExecutionWidth()),
          size_t(device_->maxBufferLength())};
}
Status MetalBackend::MeasureAnsTokenCosts(const AnsTokenCostInput &input,
                                          std::unique_ptr<PreparedGpuAnsTokenCosts> *out) try {
  if (!out)
    return Status::InvalidArgument("Null ANS token-cost output");
  AnsTokenCostStoragePlan plan;
  const auto limits = ans_token_cost_limits();
  if (auto s = ValidateAnsTokenCostInput(input, id(), &plan, &limits); !s.ok())
    return s;
  const MetalBuffer *values = nullptr, *contexts = nullptr;
  if (auto s = RequireMetalBuffer(input.values.buffer, input.values.size_bytes,
                                  input.values.offset_bytes, "ANS cost values", &values);
      !s.ok())
    return s;
  if (auto s = RequireMetalBuffer(input.contexts.buffer, input.contexts.size_bytes,
                                  input.contexts.offset_bytes, "ANS cost contexts", &contexts);
      !s.ok())
    return s;
  const ManagedHostScope resources(ResourceClass::kPreparation);
  auto candidate = std::make_unique<Completed>();
  candidate->backend = id();
  candidate->sections = plan.section_count;
  candidate->candidates = plan.candidate_count;
  std::unique_ptr<DeviceBuffer> dispatch;
  Pending pending;
  {
    const ResourceClassScope retained(ResourceClass::kCompletedFrame);
    if (auto s = Allocate(plan.output_bytes, &candidate->buffer); !s.ok())
      return s;
  }
  if (auto s = Allocate(plan.dispatch_bytes, &dispatch); !s.ok())
    return s;
  auto *bytes = static_cast<std::byte *>(AsMetalBuffer(*dispatch)->contents());
  std::memset(bytes, 0, plan.dispatch_bytes);
  std::memcpy(bytes, input.sections.data(), plan.sections_bytes);
  auto *models = reinterpret_cast<Model *>(bytes + plan.models_offset);
  for (size_t i = 0; i < input.models.size(); ++i) {
    const auto &model = input.models[i];
    models[i] = {plan.context_maps_offset + i * plan.maximum_contexts,
                 plan.clusters_offset + i * plan.maximum_clusters * kCostClusterBytes,
                 static_cast<uint32_t>(model.context_map.size()),
                 static_cast<uint32_t>(model.clusters.size()),
                 {0, 0}};
    std::memcpy(bytes + models[i].context_offset, model.context_map.data(),
                model.context_map.size());
    auto *clusters = reinterpret_cast<Cluster *>(bytes + models[i].clusters_offset);
    for (size_t j = 0; j < model.clusters.size(); ++j) {
      const auto &src = model.clusters[j];
      auto &dst = clusters[j];
      dst.split_exponent = src.split_exponent;
      dst.msb_in_token = src.msb_in_token;
      dst.lsb_in_token = src.lsb_in_token;
      dst.symbol_count = static_cast<uint32_t>(src.frequencies.size());
      std::memcpy(dst.frequencies, src.frequencies.data(), src.frequencies.size_bytes());
      std::memcpy(dst.reverse_offsets, src.reverse_offsets.data(),
                  src.reverse_offsets.size_bytes());
      std::memcpy(dst.reciprocals, src.reciprocal_frequencies.data(),
                  src.reciprocal_frequencies.size_bytes());
      std::memcpy(dst.reverse_map, src.reverse_map.data(), src.reverse_map.size_bytes());
    }
  }
  Context context{values->handle(),
                  contexts->handle(),
                  AsMetalBuffer(*dispatch)->handle(),
                  AsMetalBuffer(*candidate->buffer)->handle(),
                  input.values.offset_bytes,
                  input.contexts.offset_bytes,
                  ans_token_cost_pipeline_.get(),
                  {plan.models_offset, plan.status_offset,
                   static_cast<uint32_t>(plan.section_count),
                   static_cast<uint32_t>(plan.candidate_count)}};
  if (auto s = SubmitCompute("entropy.ans_token_cost", Encode, &context, &pending.submission);
      !s.ok())
    return s;
  if (!pending.submission)
    return Status::Internal("ANS token-cost operation returned no submission");
  if (auto s = pending.Wait(); !s.ok())
    return s;
  uint32_t status;
  std::memcpy(&status, bytes + plan.status_offset, sizeof(status));
  if (auto s = CheckAnsTokenCostDeviceStatus(status); !s.ok())
    return s;
  std::span<const std::byte> output;
  if (auto s = BorrowCompletedReadOnly(*candidate->buffer, plan.output_bytes, 0, &output); !s.ok())
    return s;
  candidate->host = {reinterpret_cast<const AnsTokenCostResult *>(output.data()),
                     plan.section_count * plan.candidate_count};
  pending.submission.reset();
  dispatch.reset();
  *out = std::move(candidate);
  return Status::Ok();
} catch (const ManagedAllocationFailure &e) {
  return e.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("ANS token-cost allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("ANS token-cost storage overflow");
}
} // namespace gjxl::metal_internal
