// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/metal/kernels/modular_tree_statistics_params.h"
#include "gpu/metal/metal_backend_internal.h"
#include <cstring>

namespace gjxl::metal_internal {
using namespace tree_gpu_internal;
using namespace resource_budget_internal;
namespace {
using namespace gjxl_tree_statistics;
static_assert(sizeof(Code) ==
              sizeof(modular_internal::entropy_estimate::ResidualCode));
static_assert(sizeof(Result) == sizeof(TreeSplitCosts) &&
              alignof(Result) == alignof(TreeSplitCosts));
static_assert(offsetof(Result, high) == offsetof(TreeSplitCosts, high));
static_assert(sizeof(Params) == 32 &&
              offsetof(Params, count_costs_offset) == 16);
static_assert(kIndexError == kTreeStatisticsIndexError &&
              kCodeError == kTreeStatisticsCodeError &&
              kHistogramError == kTreeStatisticsHistogramError);
class Prepared final : public PreparedGpuTreeStatistics {
public:
  BackendId backend_id() const noexcept override { return backend; }
  size_t sample_count() const noexcept override { return plan.sample_count; }
  BackendId backend = 0;
  TreeStatisticsPreparationPlan plan;
  std::unique_ptr<DeviceBuffer> buffer;
};
class Completed final : public PreparedGpuTreeSplitCosts {
public:
  BackendId backend_id() const noexcept override { return backend; }
  std::span<const TreeSplitCosts> costs() const noexcept override {
    return host;
  }
  BackendId backend = 0;
  std::unique_ptr<DeviceBuffer> buffer;
  std::span<const TreeSplitCosts> host;
};
struct Pending {
  std::unique_ptr<GpuSubmission> submission;
  bool completed = false;
  ~Pending() {
    if (submission && !completed)
      DrainMetalSubmission(*submission);
  }
  Status Wait() {
    auto status = submission->Wait();
    completed = true;
    return status;
  }
};
struct Context {
  MTL::Buffer *prepared, *dispatch, *output;
  MTL::ComputePipelineState *histogram, *cost;
  Params params;
};
void Encode(MetalBackend &, MTL::ComputeCommandEncoder *encoder,
            const void *opaque) {
  const auto &c = *static_cast<const Context *>(opaque);
  encoder->setBuffer(c.prepared, 0, 0);
  encoder->setBuffer(c.dispatch, 0, 1);
  encoder->setBuffer(c.output, 0, 2);
  encoder->setBytes(&c.params, sizeof(c.params), 3);
  encoder->setComputePipelineState(c.histogram);
  RecordMetalComputePipelineState(c.histogram);
  DispatchMetalThreadgroups(encoder, MTL::Size(14, c.params.thresholds + 1, 1),
                            MTL::Size(128, 1, 1));
  const MTL::Resource *dependencies[] = {c.dispatch};
  encoder->memoryBarrier(dependencies, 1);
  encoder->setComputePipelineState(c.cost);
  RecordMetalComputePipelineState(c.cost);
  DispatchMetalThreadgroups(encoder, MTL::Size(14, c.params.thresholds, 2),
                            MTL::Size(128, 1, 1));
}
} // namespace
TreeStatisticsDeviceLimits
MetalBackend::tree_statistics_limits() const noexcept {
  return {kMaximumTreeStatisticsSamples, kMaximumTreeStatisticsSamples,
          static_cast<size_t>(device_->maxBufferLength())};
}
Status MetalBackend::PrepareTreeStatistics(
    const TreeStatisticsPreparationInput &input,
    std::unique_ptr<PreparedGpuTreeStatistics> *out) try {
  if (!out)
    return Status::InvalidArgument(
        "Null Modular tree statistics preparation output");
  const auto limits = tree_statistics_limits();
  TreeStatisticsPreparationPlan plan;
  if (auto status = ValidateTreeStatisticsPreparation(input, &plan, &limits);
      !status.ok())
    return status;
  const ManagedHostScope resources(ResourceClass::kPreparation);
  auto candidate = std::make_unique<Prepared>();
  candidate->backend = id();
  candidate->plan = plan;
  {
    const ResourceClassScope retained(ResourceClass::kCompletedFrame);
    if (auto status = Allocate(plan.storage_bytes, &candidate->buffer);
        !status.ok())
      return status;
  }
  auto *bytes =
      static_cast<std::byte *>(AsMetalBuffer(*candidate->buffer)->contents());
  std::memset(bytes, 0, plan.storage_bytes);
  std::memcpy(bytes, input.codes.data(), plan.codes_bytes);
  std::memcpy(bytes + plan.count_costs_offset, input.count_log_count.data(),
              input.count_log_count.size_bytes());
  *out = std::move(candidate);
  return Status::Ok();
} catch (const ManagedAllocationFailure &error) {
  return error.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory(
      "Modular tree statistics preparation allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument(
      "Modular tree statistics preparation storage overflow");
}
Status MetalBackend::MeasureTreeSplitCosts(
    const TreeStatisticsInput &input,
    std::unique_ptr<PreparedGpuTreeSplitCosts> *out) try {
  if (!out)
    return Status::InvalidArgument("Null Modular tree statistics cost output");
  const auto limits = tree_statistics_limits();
  TreeStatisticsStoragePlan plan;
  if (auto status = ValidateTreeStatisticsInput(input, id(), &plan, &limits);
      !status.ok())
    return status;
  const auto *prepared = dynamic_cast<const Prepared *>(input.prepared);
  if (!prepared || prepared->plan.storage_bytes != plan.prepared_bytes)
    return Status::InvalidArgument(
        "Modular tree statistics requires its native prepared owner");
  const MetalBuffer *source = nullptr;
  if (auto status =
          RequireMetalBuffer(prepared->buffer.get(), plan.prepared_bytes, 0,
                             "Modular tree statistics prepared data", &source);
      !status.ok())
    return status;
  const ManagedHostScope resources(ResourceClass::kPreparation);
  auto candidate = std::make_unique<Completed>();
  candidate->backend = id();
  std::unique_ptr<DeviceBuffer> dispatch;
  Pending pending;
  {
    const ResourceClassScope retained(ResourceClass::kCompletedFrame);
    if (auto status = Allocate(plan.output_bytes, &candidate->buffer);
        !status.ok())
      return status;
  }
  if (auto status = Allocate(plan.dispatch_bytes, &dispatch); !status.ok())
    return status;
  auto *bytes = static_cast<std::byte *>(AsMetalBuffer(*dispatch)->contents());
  std::memset(bytes, 0, plan.dispatch_bytes);
  std::memcpy(bytes, input.order.data(), plan.indices_bytes);
  auto *boundaries =
      reinterpret_cast<uint32_t *>(bytes + plan.boundaries_offset);
  std::memcpy(boundaries + 1, input.low_counts.data(),
              input.low_counts.size_bytes());
  boundaries[plan.threshold_count + 1] = static_cast<uint32_t>(plan.leaf_count);
  Context context{source->handle(),
                  AsMetalBuffer(*dispatch)->handle(),
                  AsMetalBuffer(*candidate->buffer)->handle(),
                  tree_statistics_pipelines_[0].get(),
                  tree_statistics_pipelines_[1].get(),
                  {static_cast<uint32_t>(plan.sample_count),
                   static_cast<uint32_t>(plan.leaf_count),
                   static_cast<uint32_t>(plan.threshold_count), 0,
                   static_cast<uint32_t>(prepared->plan.count_costs_offset),
                   static_cast<uint32_t>(plan.boundaries_offset),
                   static_cast<uint32_t>(plan.histograms_offset),
                   static_cast<uint32_t>(plan.status_offset)}};
  if (auto status = SubmitCompute("modular.tree_split_statistics", Encode,
                                  &context, &pending.submission);
      !status.ok())
    return status;
  if (!pending.submission)
    return Status::Internal("Modular tree statistics returned no submission");
  if (auto status = pending.Wait(); !status.ok())
    return status;
  uint32_t status_word;
  std::memcpy(&status_word, bytes + plan.status_offset, sizeof(status_word));
  if (auto status = CheckTreeStatisticsDeviceStatus(status_word); !status.ok())
    return status;
  std::span<const std::byte> output;
  if (auto status = BorrowCompletedReadOnly(*candidate->buffer,
                                            plan.output_bytes, 0, &output);
      !status.ok())
    return status;
  candidate->host = {reinterpret_cast<const TreeSplitCosts *>(output.data()),
                     plan.threshold_count};
  pending.submission.reset();
  dispatch.reset();
  *out = std::move(candidate);
  return Status::Ok();
} catch (const ManagedAllocationFailure &error) {
  return error.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("Modular tree statistics cost allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument(
      "Modular tree statistics cost storage overflow");
}
} // namespace gjxl::metal_internal
