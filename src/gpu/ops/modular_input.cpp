// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/ops/modular_input_internal.h"

#include <array>
#include <new>
#include <stdexcept>

#include "core/resource_context.h"

namespace gjxl::modular_gpu_internal {
using namespace resource_budget_internal;
namespace {
class PreparedInput final : public PreparedGpuModularInput {
public:
  PreparedInput(BackendId id, const ModularInputStoragePlan &plan, uint8_t rct)
      : id_(id), metadata_(plan.metadata), rct_(rct), count_(plan.channel_count) {}
  BackendId backend_id() const noexcept override { return id_; }
  const codec_internal::ImageMetadata &metadata() const noexcept override { return metadata_; }
  uint8_t rct() const noexcept override { return rct_; }
  std::span<const GpuModularChannelView> channels() const noexcept override {
    return std::span(channels_).first(count_);
  }
  Status Bind(const ModularInputStoragePlan &plan) {
    for (size_t c = 0; c < count_; ++c) {
      const auto &p = plan.planes[c];
      auto &view = channels_[c];
      view = {plan.channels[c],
              {buffer.get(), p.offset_bytes, p.element_type, p.extent, p.row_stride}};
      DeviceMemoryRange range;
      if (auto s = ComputeDevicePlaneRange(view.plane, id_, &range); !s.ok())
        return s;
    }
    return Status::Ok();
  }
  // This is the sole retained backing; views and the small owner have no
  // additional resource ticket. Buffer destruction releases backing then charge.
  std::unique_ptr<DeviceBuffer> buffer;

private:
  BackendId id_;
  codec_internal::ImageMetadata metadata_;
  uint8_t rct_;
  size_t count_;
  std::array<GpuModularChannelView, 4> channels_{};
};

struct PendingInput {
  const ModularInputKernel &kernel;
  std::unique_ptr<GpuSubmission> submission;
  bool completed = false;
  ~PendingInput() {
    if (submission && !completed)
      kernel.drain(*submission);
  }
  Status Wait() {
    const auto status = submission->Wait();
    completed = true; // Also completed when the device reports an error.
    return status;
  }
};
} // namespace

Status PrepareGpuModularInput(GpuBackend &backend, const ModularInputPreparation &preparation,
                              std::unique_ptr<PreparedGpuModularInput> *out) {
  if (!out)
    return Status::InvalidArgument("Null GPU Modular preparation output");
  auto *capability = QueryGpuModularInputPreparation(backend);
  if (!capability)
    return Status::Unsupported("Backend has no Modular input preparation");
  return capability->PrepareModularInput(preparation, out);
}

Status PrepareModularInputWithKernel(GpuBackend &backend,
                                     const ModularInputPreparation &preparation,
                                     const ModularInputDeviceLimits &limits,
                                     const ModularInputKernel &kernel,
                                     std::unique_ptr<PreparedGpuModularInput> *out) try {
  if (!out || !kernel.submit || !kernel.drain)
    return Status::InvalidArgument("Invalid GPU Modular preparation request");
  if (auto s = preparation.input.Validate(); !s.ok())
    return s;
  const auto &input = preparation.input;
  const ModularInputStorageOptions options{input.extent, input.row_stride, input.format,
                                           input.byte_order, preparation.rct};
  ModularInputStoragePlan plan;
  if (auto s = ComputeModularInputStoragePlan(options, &plan, &limits); !s.ok())
    return s;

  const ResourceClassScope resources(ResourceClass::kPreparation);
  auto candidate = std::make_unique<PreparedInput>(backend.id(), plan, preparation.rct);
  std::unique_ptr<DeviceBuffer> upload;
  // Declared last: all unwinding drains before either backing/ticket dies.
  PendingInput pending{kernel, {}};
  if (auto s = backend.Allocate(plan.upload_bytes, &upload); !s.ok())
    return s;
  if (auto s = backend.Allocate(plan.output_bytes, &candidate->buffer); !s.ok())
    return s;
  if (auto s = candidate->Bind(plan); !s.ok())
    return s;
  if (auto s = backend.CopyHostToDevice(*upload, input.bytes.data(), plan.upload_bytes); !s.ok())
    return s;
  if (auto s = kernel.submit(backend, options, *upload, *candidate->buffer, &pending.submission);
      !s.ok())
    return s;
  if (!pending.submission)
    return Status::Internal("Modular kernel returned no submission");
  if (auto s = pending.Wait(); !s.ok())
    return s;
  pending.submission.reset();
  upload.reset(); // End preparation peak before publication or host readback.
  *out = std::move(candidate);
  return Status::Ok();
} catch (const ManagedAllocationFailure &failure) {
  return failure.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("GPU Modular preparation allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("GPU Modular preparation storage overflow");
}

Status ReadbackGpuModularInput(GpuBackend &backend, const PreparedGpuModularInput &prepared,
                               modular_internal::ModularImage *out) try {
  if (!out || prepared.backend_id() != backend.id())
    return Status::InvalidArgument("Invalid GPU Modular readback owner/output");
  const auto channels = prepared.channels();
  if (channels.size() != 1 && channels.size() != 3 && channels.size() != 4)
    return Status::InvalidArgument("Invalid GPU Modular readback channel count");
  std::array<modular_internal::ChannelDescriptor, 4> descriptors{};
  std::array<DeviceMemoryRange, 4> ranges{};
  for (size_t c = 0; c < channels.size(); ++c) {
    const auto &channel = channels[c];
    descriptors[c] = channel.descriptor;
    const auto &p = channel.plane;
    if (p.element_type != DeviceElementType::kI32 || p.extent != prepared.metadata().extent ||
        p.row_stride != p.extent.width || channel.descriptor.extent != p.extent ||
        channel.descriptor.hshift || channel.descriptor.vshift ||
        channel.descriptor.role != (c == 3 ? modular_internal::ChannelRole::kAlpha
                                           : modular_internal::ChannelRole::kColor))
      return Status::InvalidArgument("Invalid GPU Modular readback plane");
    if (auto s = ComputeDevicePlaneRange(p, backend.id(), &ranges[c]); !s.ok())
      return s;
    for (size_t prior = 0; prior < c; ++prior)
      if (DeviceRangesOverlap(ranges[c], ranges[prior]))
        return Status::InvalidArgument("GPU Modular readback planes overlap");
  }
  // Readback does not assume the device buffer is host-accessible. Fresh host
  // allocation remains charged even when the caller did not install a context.
  const ManagedHostScope resources(ResourceClass::kPreparation);
  modular_internal::ModularImage candidate;
  if (auto s = modular_internal::ModularImage::Create(std::span(descriptors).first(channels.size()),
                                                      0, &candidate);
      !s.ok())
    return s;
  for (size_t c = 0; c < channels.size(); ++c) {
    const auto &p = channels[c].plane;
    if (auto s = backend.CopyDeviceToHost(*p.buffer, candidate.samples(c).data(),
                                          ranges[c].size_bytes, p.offset_bytes);
        !s.ok())
      return s;
  }
  *out = std::move(candidate);
  return Status::Ok();
} catch (const ManagedAllocationFailure &failure) {
  return failure.status();
} catch (const std::bad_alloc &) {
  return Status::OutOfMemory("GPU Modular readback allocation failed");
} catch (const std::length_error &) {
  return Status::InvalidArgument("GPU Modular readback storage overflow");
}
} // namespace gjxl::modular_gpu_internal
