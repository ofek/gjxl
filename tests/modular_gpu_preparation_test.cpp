// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/ops/modular_input_internal.h"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::modular_gpu_internal;
using namespace gjxl::resource_budget_internal;

void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(const Status &status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
enum class Fault {
  kNone,
  kFirstAllocation,
  kSecondAllocation,
  kUpload,
  kSubmit,
  kAfterSubmit,
  kThrowAfterSubmit,
  kCompletion,
  kThrowWait,
  kNoSubmission,
  kReadback
};
struct State {
  Fault fault = Fault::kNone;
  size_t allocation_calls = 0, readback_calls = 0, uploaded_bytes = 0;
  size_t submitted = 0, waits = 0, drains = 0, active = 0;
  bool released_busy = false;
};
class Buffer final : public DeviceBuffer {
public:
  Buffer(BackendId id, size_t bytes, State &state, ResourceAllocation allocation)
      : DeviceBuffer(BackendKind::kMetal, id, bytes), state(state),
        allocation(std::move(allocation)), bytes(bytes) {}
  ~Buffer() override {
    if (borrowers)
      state.released_busy = true;
  }
  State &state;
  size_t borrowers = 0;
  ResourceAllocation allocation; // Released after the simulated backing.
  std::vector<uint8_t> bytes;
};
class Submission final : public GpuSubmission {
public:
  Submission(State &state, Buffer &input, Buffer &output, ModularInputStoragePlan plan)
      : state(state), input(input), output(output), plan(plan) {
    ++input.borrowers;
    ++output.borrowers;
    ++state.active;
    ++state.submitted;
  }
  ~Submission() override {
    if (!complete)
      state.released_busy = true;
  }
  Status Wait() override {
    ++state.waits;
    if (state.fault == Fault::kThrowWait)
      throw std::bad_alloc();
    Finish();
    if (state.fault == Fault::kCompletion)
      return Status::DeviceError("Injected completion");
    return Status::Ok();
  }
  void Finish() noexcept {
    if (complete)
      return;
    // Lifecycle double only; physical-Metal tests provide numerical qualification.
    for (size_t c = 0; c < plan.channel_count; ++c) {
      const int32_t sample = static_cast<int32_t>(100 + c);
      for (size_t i = 0; i < plan.planes[c].size_bytes; i += sizeof(sample))
        std::memcpy(output.bytes.data() + plan.planes[c].offset_bytes + i, &sample, sizeof(sample));
    }
    --input.borrowers;
    --output.borrowers;
    --state.active;
    complete = true;
  }
  State &state;

private:
  Buffer &input, &output;
  ModularInputStoragePlan plan;
  bool complete = false;
};
class Backend : public GpuBackend {
public:
  State state;
  ModularInputDeviceLimits limits{{1 << 20, 1 << 20}, SIZE_MAX, 1 << 30};
  BackendKind kind() const noexcept override { return BackendKind::kMetal; }
  std::string_view name() const noexcept override { return "Lifecycle test double"; }
  void Fail(Fault fault) {
    state.fault = fault;
    state.allocation_calls = state.readback_calls = 0;
  }
  Status Allocate(size_t bytes, std::unique_ptr<DeviceBuffer> *out) override {
    ResourceAllocation allocation;
    if (auto s = PrepareResourceAllocation(bytes, bytes, &allocation); !s.ok())
      return s;
    const auto index = state.allocation_calls++;
    if ((state.fault == Fault::kFirstAllocation && index == 0) ||
        (state.fault == Fault::kSecondAllocation && index == 1))
      return Status::OutOfMemory("Injected backing failure");
    auto candidate = std::make_unique<Buffer>(id(), bytes, state, std::move(allocation));
    if (auto s = candidate->allocation.Commit(); !s.ok())
      return s;
    *out = std::move(candidate);
    RecordSuccessfulAllocation();
    return Status::Ok();
  }
  Status CopyHostToDevice(DeviceBuffer &buffer, const void *src, size_t bytes,
                          size_t offset) override {
    state.uploaded_bytes = bytes;
    if (state.fault == Fault::kUpload)
      return Status::DeviceError("Injected upload failure");
    auto &b = static_cast<Buffer &>(buffer);
    Check(offset <= b.bytes.size() && bytes <= b.bytes.size() - offset, "Upload range invalid");
    std::memcpy(b.bytes.data() + offset, src, bytes);
    return Status::Ok();
  }
  Status CopyDeviceToHost(const DeviceBuffer &buffer, void *dst, size_t bytes,
                          size_t offset) override {
    if (state.fault == Fault::kReadback && state.readback_calls++ == 1)
      return Status::DeviceError("Injected second-plane readback failure");
    const auto &b = static_cast<const Buffer &>(buffer);
    Check(!b.borrowers && offset <= b.bytes.size() && bytes <= b.bytes.size() - offset,
          "Readback before completion or out of range");
    std::memcpy(dst, b.bytes.data() + offset, bytes);
    return Status::Ok();
  }
  Status ForwardTransform(const TransformBatch &, std::unique_ptr<GpuSubmission> *) override {
    return Status::Unsupported("Not a transform test");
  }
  Status InverseTransform(const TransformBatch &, std::unique_ptr<GpuSubmission> *) override {
    return Status::Unsupported("Not a transform test");
  }
  static Status Submit(GpuBackend &backend, const ModularInputStorageOptions &options,
                       const DeviceBuffer &input, DeviceBuffer &output,
                       std::unique_ptr<GpuSubmission> *out) {
    auto &b = static_cast<Backend &>(backend);
    if (b.state.fault == Fault::kSubmit)
      return Status::SubmissionFailed("Injected submit failure");
    if (b.state.fault == Fault::kNoSubmission)
      return Status::Ok();
    ModularInputStoragePlan plan;
    Ok(ComputeModularInputStoragePlan(options, &plan));
    *out = std::make_unique<Submission>(b.state,
                                        const_cast<Buffer &>(static_cast<const Buffer &>(input)),
                                        static_cast<Buffer &>(output), plan);
    if (b.state.fault == Fault::kAfterSubmit)
      return Status::SubmissionFailed("Failure after commit");
    if (b.state.fault == Fault::kThrowAfterSubmit)
      throw std::bad_alloc();
    return Status::Ok();
  }
  static void Drain(GpuSubmission &submission) noexcept {
    auto &s = static_cast<Submission &>(submission);
    ++s.state.drains;
    s.Finish();
  }
};
class PreparingBackend final : public Backend, public GpuModularInputPreparation {
public:
  ModularInputDeviceLimits modular_input_limits() const noexcept override { return limits; }
  Status PrepareModularInput(const ModularInputPreparation &preparation,
                             std::unique_ptr<PreparedGpuModularInput> *out) override {
    return PrepareModularInputWithKernel(*this, preparation, limits, {Submit, Drain}, out);
  }
};

struct Fixture {
  std::vector<uint8_t> bytes = std::vector<uint8_t>(3 + 25 + 24, 0xad);
  ModularInputPreparation preparation{{std::span(bytes).subspan(3),
                                       {3, 2},
                                       25,
                                       PackedModularFormat::kRgba16,
                                       SampleByteOrder::kBigEndian},
                                      41};
  ModularInputStoragePlan plan;
  Fixture() {
    Ok(ComputeModularInputStoragePlan(
        {{3, 2}, 25, PackedModularFormat::kRgba16, SampleByteOrder::kBigEndian, 41, true}, &plan));
  }
};
void Empty(const ResourceBudget &budget) {
  auto s = budget.snapshot();
  Check(s.committed_bytes() == 0 && s.total.backing_count == 0 && s.total.pending_count == 0 &&
            s.open_reservations == 0 && s.waiting_requests == 0,
        "Resource charge/reservation leaked");
}
void Values(const ModularImage &image) {
  Check(image.channel_count() == 4 && image.metadata_channels() == 0,
        "Readback channel shape differs");
  for (size_t c = 0; c < image.channel_count(); ++c)
    for (auto v : image.view(c).backing)
      Check(v == int32_t(100 + c), "Readback values changed");
}

void OwnershipAndPeaks() {
  Fixture f;
  PreparingBackend gpu;
  ResourceBudget budget(f.plan.peak_bytes);
  ResourceReservation reservation;
  Ok(budget.TryReserve(f.plan.peak_bytes, &reservation));
  std::unique_ptr<PreparedGpuModularInput> prepared;
  ModularImage host;
  {
    ResourceContextScope context({&reservation, ResourceClass::kDiagnostics});
    Ok(PrepareGpuModularInput(gpu, f.preparation, &prepared));
    Check(CurrentResourceContext().resource_class == ResourceClass::kDiagnostics,
          "Preparation leaked its resource class");
    auto s = budget.snapshot();
    Check(gpu.state.uploaded_bytes == 49 && s.total.backing_count == 1 &&
              s.total.live_capacity_bytes == f.plan.output_bytes &&
              s.peak_backing_bytes == f.plan.preparation_peak_bytes && gpu.state.active == 0,
          "Upload retained, double-counted, padded or released before completion");
    Check(s.classes[size_t(ResourceClass::kPreparation)].live_capacity_bytes == f.plan.output_bytes,
          "Device result lost its preparation charge");
    Check(prepared->backend_id() == gpu.id() && prepared->metadata() == f.plan.metadata &&
              prepared->rct() == 41 && prepared->channels().size() == 4,
          "Prepared metadata differs");
    for (size_t c = 0; c < 4; ++c) {
      const auto &ch = prepared->channels()[c];
      Check(ch.descriptor == f.plan.channels[c] &&
                ch.plane.buffer == prepared->channels()[0].plane.buffer &&
                ch.plane.offset_bytes == f.plan.planes[c].offset_bytes,
            "Owner view does not follow plan");
    }
    // No source/upload lifetime is retained by the result.
    f.bytes.clear();
    f.bytes.shrink_to_fit();
    Ok(ReadbackGpuModularInput(gpu, *prepared, &host));
    Values(host);
    s = budget.snapshot();
    Check(s.total.live_capacity_bytes ==
                  f.plan.output_bytes + f.plan.host_readback.retained_bytes &&
              s.peak_backing_bytes == f.plan.peak_bytes && s.total.pending_count == 0 &&
              CurrentResourceContext().resource_class == ResourceClass::kDiagnostics,
          "Readback phase disagrees with planned capacity or leaked context");
  }
  reservation.Reset();
  Check(budget.snapshot().committed_bytes() ==
            f.plan.output_bytes + f.plan.host_readback.retained_bytes,
        "Closing the producing reservation uncharged live owners");
  prepared.reset();
  Values(host);
  host = {};
  Empty(budget);
  Check(!gpu.state.released_busy, "Buffer/submission destroyed while borrowed");
}

void AtomicFailures() {
  Fixture f;
  PreparingBackend gpu;
  ResourceBudget budget(f.plan.output_bytes + f.plan.peak_bytes);
  std::unique_ptr<PreparedGpuModularInput> previous;
  {
    ResourceReservation old;
    Ok(budget.TryReserve(f.plan.preparation_peak_bytes, &old));
    ResourceContextScope scope({&old, ResourceClass::kPreparation});
    Ok(PrepareGpuModularInput(gpu, f.preparation, &previous));
  }
  const auto *identity = previous.get();
  const auto source = f.bytes;
  for (auto fault : {Fault::kFirstAllocation, Fault::kSecondAllocation, Fault::kUpload,
                     Fault::kSubmit, Fault::kAfterSubmit, Fault::kThrowAfterSubmit,
                     Fault::kCompletion, Fault::kThrowWait, Fault::kNoSubmission}) {
    ResourceReservation job;
    Ok(budget.TryReserve(f.plan.preparation_peak_bytes, &job));
    ResourceContextScope scope({&job, ResourceClass::kDiagnostics});
    gpu.Fail(fault);
    const auto drained = gpu.state.drains;
    auto status = PrepareGpuModularInput(gpu, f.preparation, &previous);
    Check(!status.ok() && previous.get() == identity && f.bytes == source &&
              gpu.state.active == 0 && !gpu.state.released_busy &&
              CurrentResourceContext().resource_class == ResourceClass::kDiagnostics,
          "Preparation failure changed caller state or released in-flight backing");
    if (fault == Fault::kAfterSubmit || fault == Fault::kThrowAfterSubmit ||
        fault == Fault::kThrowWait)
      Check(gpu.state.drains == drained + 1, "Error/exception failed to drain committed work");
    const auto s = budget.snapshot();
    Check(s.total.live_capacity_bytes == f.plan.output_bytes && s.total.backing_count == 1 &&
              s.total.pending_count == 0 &&
              s.reserved_unbacked_bytes == f.plan.preparation_peak_bytes,
          "Failure leaked or reassigned resource tickets");
  }
  gpu.Fail(Fault::kNone);
  {
    ResourceReservation tiny;
    Ok(budget.TryReserve(f.plan.preparation_peak_bytes - 1, &tiny));
    ResourceContextScope scope({&tiny, ResourceClass::kPreparation});
    const auto submits = gpu.state.submitted;
    const auto status = PrepareGpuModularInput(gpu, f.preparation, &previous);
    Check(status.resource_plan_exceeded() && previous.get() == identity &&
              gpu.state.submitted == submits && budget.snapshot().total.pending_count == 0,
          "Undersized admission escaped its envelope or submitted work");
  }
  {
    ResourceReservation fresh;
    Ok(budget.TryReserve(f.plan.preparation_peak_bytes, &fresh));
    ResourceContextScope scope({&fresh, ResourceClass::kPreparation});
    Ok(PrepareGpuModularInput(gpu, f.preparation, &previous));
    Check(previous.get() != identity && budget.snapshot().total.backing_count == 1 &&
              budget.snapshot().peak_backing_bytes <= f.plan.output_bytes + f.plan.peak_bytes,
          "Successful replacement retained the old owner or exceeded overlap");
  }
  previous.reset();
  Empty(budget);
}

void ReadbackFailures() {
  Fixture f;
  PreparingBackend gpu, foreign;
  ResourceBudget budget(f.plan.peak_bytes + f.plan.host_readback.retained_bytes);
  ResourceReservation job;
  Ok(budget.TryReserve(f.plan.peak_bytes + f.plan.host_readback.retained_bytes, &job));
  {
    ResourceContextScope scope({&job, ResourceClass::kPreparation});
    std::unique_ptr<PreparedGpuModularInput> prepared;
    Ok(PrepareGpuModularInput(gpu, f.preparation, &prepared));
    ModularImage host;
    Ok(ReadbackGpuModularInput(gpu, *prepared, &host));
    const auto *identity = host.view(0).backing.data();
    const auto live = budget.snapshot().total.live_capacity_bytes;
    gpu.Fail(Fault::kReadback);
    Check(!ReadbackGpuModularInput(gpu, *prepared, &host).ok() &&
              host.view(0).backing.data() == identity,
          "Partial readback published partial output");
    Values(host);
    Check(budget.snapshot().total.live_capacity_bytes == live &&
              budget.snapshot().total.pending_count == 0,
          "Readback failure leaked candidate backing");
    gpu.Fail(Fault::kNone);
    for (size_t fail = 0; fail < 5; ++fail) {
      ArmManagedHostAllocationFailureAfterForTest(fail);
      const auto status = ReadbackGpuModularInput(gpu, *prepared, &host);
      const bool pending = ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      Check(!status.ok() && !pending && host.view(0).backing.data() == identity &&
                budget.snapshot().total.live_capacity_bytes == live,
            "Host allocation failure changed output or leaked backing");
    }
    Check(!ReadbackGpuModularInput(foreign, *prepared, &host).ok() &&
              !ReadbackGpuModularInput(gpu, *prepared, nullptr).ok() &&
              host.view(0).backing.data() == identity,
          "Readback accepted foreign backend/null output");
  }
  job.Reset();
  Empty(budget);
}

void RejectionAndDefaultContext() {
  Fixture f;
  PreparingBackend gpu;
  Backend unsupported;
  std::unique_ptr<PreparedGpuModularInput> previous;
  Ok(PrepareGpuModularInput(gpu, f.preparation, &previous));
  const auto *identity = previous.get();
  const auto allocations = gpu.stats().successful_allocations;
  auto bad = f.preparation;
  bad.input.bytes = bad.input.bytes.first(48);
  Check(!PrepareGpuModularInput(gpu, bad, &previous).ok(), "Short backing accepted");
  bad = f.preparation;
  bad.rct = 42;
  Check(!PrepareGpuModularInput(gpu, bad, &previous).ok(), "Invalid RCT accepted");
  gpu.limits.maximum_buffer_bytes = 1;
  Check(PrepareGpuModularInput(gpu, f.preparation, &previous).code() == StatusCode::kUnsupported,
        "Device limits not checked before allocation");
  Check(PrepareGpuModularInput(unsupported, f.preparation, &previous).code() ==
                StatusCode::kUnsupported &&
            !PrepareGpuModularInput(gpu, f.preparation, nullptr).ok() &&
            previous.get() == identity && gpu.stats().successful_allocations == allocations,
        "Rejection allocated or replaced prior output");
  ModularImage host;
  Ok(ReadbackGpuModularInput(gpu, *previous, &host));
  Values(host);
  Check(DefaultResourceBudget().snapshot().total.live_capacity_bytes ==
            f.plan.output_bytes + f.plan.host_readback.retained_bytes,
        "Standalone ownership/readback bypassed default accounting");
  previous.reset();
  host = {};
  Empty(DefaultResourceBudget());
}
} // namespace

int main() try {
  OwnershipAndPeaks();
  AtomicFailures();
  ReadbackFailures();
  RejectionAndDefaultContext();
  Empty(DefaultResourceBudget());
  std::cout << "Modular GPU ownership, resource peaks, atomic failure and unwind draining passed\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
