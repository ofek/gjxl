// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/metal/metal_modular_input.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "core/cpu_execution.h"
#include "gpu/metal/metal_backend.h"
#if GJXL_MODULAR_INPUT_REFERENCE
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/transform/enc_rct.h"
#include "lib/jxl/modular/transform/transform.h"
#endif

namespace {
using namespace gjxl;
using namespace gjxl::metal_internal;
using namespace gjxl::modular_internal;
using namespace gjxl::modular_gpu_internal;
using namespace gjxl::resource_budget_internal;

constexpr int32_t kGuard = 0x13579bdf;
std::atomic<size_t> cases{0}, samples{0};
void Check(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
void Ok(const Status &status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}

// Tests also keep borrowed buffers alive and drain on a failed assertion.
struct Pending {
  std::unique_ptr<GpuSubmission> value;
  ~Pending() {
    if (value)
      (void)value->Wait();
  }
};

struct Format {
  PackedModularFormat value;
  size_t channels, bytes;
};
constexpr std::array<Format, 6> formats{{{PackedModularFormat::kGray8, 1, 1},
                                         {PackedModularFormat::kRgb8, 3, 1},
                                         {PackedModularFormat::kRgba8, 4, 1},
                                         {PackedModularFormat::kGray16, 1, 2},
                                         {PackedModularFormat::kRgb16, 3, 2},
                                         {PackedModularFormat::kRgba16, 4, 2}}};

struct Fixture {
  Format format;
  ModularInputStorageOptions options;
  ModularInputStoragePlan plan;
  size_t prefix;
  std::vector<uint8_t> packed;
  std::vector<uint16_t> source;

  Fixture(Format f, Extent2D extent, SampleByteOrder order, uint8_t rct, size_t offset,
          size_t padding, size_t suffix = 0, unsigned pattern = 0)
      : format(f),
        options{extent, extent.width * f.channels * f.bytes + padding, f.value, order, rct},
        prefix(offset) {
    Ok(ComputeModularInputStoragePlan(options, &plan));
    packed.assign(prefix + plan.upload_bytes + suffix, 0xad);
    source.resize(extent.width * extent.height * f.channels);
    const uint32_t maximum = f.bytes == 1 ? 255 : 65535;
    const std::array<uint32_t, 6> edges{0, 1, 2, maximum / 2, maximum - 1, maximum};
    uint32_t rng = 0x9187ac23;
    for (size_t y = 0; y < extent.height; ++y)
      for (size_t x = 0; x < extent.width; ++x) {
        const size_t pixel = y * extent.width + x;
        for (size_t c = 0, divisor = 1; c < f.channels; ++c, divisor *= 6) {
          rng ^= rng << 13;
          rng ^= rng >> 17;
          rng ^= rng << 5;
          uint32_t value = pixel < 216 ? edges[(pixel / divisor) % 6] : rng & maximum;
          if (pattern == 1)
            value = maximum;
          if (pattern == 2)
            value = 0;
          if (pattern == 3)
            value = (x + y + c) % 2 ? maximum : 0;
          if (c == 3 && pixel % 3 == 0)
            value = 0; // Exact invisible RGB.
          source[pixel * f.channels + c] = static_cast<uint16_t>(value);
          const size_t i = prefix + y * options.row_stride_bytes + (x * f.channels + c) * f.bytes;
          if (f.bytes == 1 || order == SampleByteOrder::kLittleEndian) {
            packed[i] = static_cast<uint8_t>(value);
            if (f.bytes == 2)
              packed[i + 1] = static_cast<uint8_t>(value >> 8);
          } else {
            packed[i] = static_cast<uint8_t>(value >> 8);
            packed[i + 1] = static_cast<uint8_t>(value);
          }
        }
      }
  }
  PackedModularImageView view() const {
    return {std::span(packed).subspan(prefix, plan.upload_bytes), options.extent,
            options.row_stride_bytes, options.format, options.byte_order};
  }
};

#if GJXL_MODULAR_INPUT_REFERENCE
// Independent pinned libjxl RCT; input integers come directly from the fixture,
// not from the native unpacker or its transform implementation.
jxl::Image Reference(const Fixture &f, JxlMemoryManager *memory) {
  auto created = jxl::Image::Create(memory, f.options.extent.width, f.options.extent.height,
                                    f.format.bytes * 8, f.format.channels);
  Check(created.ok(), "Pinned image allocation failed");
  auto image = std::move(created).value_();
  for (size_t c = 0; c < f.format.channels; ++c)
    for (size_t y = 0; y < f.options.extent.height; ++y)
      for (size_t x = 0; x < f.options.extent.width; ++x)
        image.channel[c].Row(y)[x] =
            f.source[(y * f.options.extent.width + x) * f.format.channels + c];
  if (f.options.rct) {
    auto next = jxl::Image::Create(memory, f.options.extent.width, f.options.extent.height,
                                   f.format.bytes * 8, f.format.channels);
    Check(next.ok(), "Pinned RCT allocation failed");
    auto transformed = std::move(next).value_();
    Check(bool(jxl::FwdRct(
              {&image.channel[0], &image.channel[1], &image.channel[2]},
              {&transformed.channel[0], &transformed.channel[1], &transformed.channel[2]},
              f.options.rct, nullptr)),
          "Pinned RCT failed");
    if (f.format.channels == 4)
      transformed.channel[3] = std::move(image.channel[3]);
    image = std::move(transformed);
  }
  return image;
}
#endif

void Run(GpuBackend &gpu, const Fixture &f, bool guarded = true) {
  const size_t output_prefix = guarded ? kModularInputPlaneAlignment : 0;
  const size_t output_suffix = guarded ? 13 * sizeof(int32_t) : 0;
  std::vector<int32_t> words((output_prefix + f.plan.output_bytes + output_suffix) / 4, kGuard);
  std::unique_ptr<DeviceBuffer> input, output;
  Ok(gpu.Allocate(f.packed.size(), &input));
  Ok(gpu.Allocate(words.size() * 4, &output));
  Ok(gpu.CopyHostToDevice(*input, f.packed.data(), f.packed.size()));
  Ok(gpu.CopyHostToDevice(*output, words.data(), words.size() * 4));
  Pending pending;
  Ok(SubmitMetalModularInput(gpu, {input.get(), f.prefix, output.get(), output_prefix, f.options},
                             &pending.value));
  Check(bool(pending.value), "Successful kernel launch returned no submission");
  Ok(pending.value->Wait());
  Ok(pending.value->Wait());
  Ok(gpu.CopyDeviceToHost(*output, words.data(), words.size() * 4));
  std::vector<uint8_t> unchanged(f.packed.size());
  Ok(gpu.CopyDeviceToHost(*input, unchanged.data(), unchanged.size()));
  Check(unchanged == f.packed, "Kernel changed packed input/padding/guards");

  ModularEncoderFrame scalar;
  Ok(ModularEncoderFrame::Prepare(f.view(), f.options.rct, &scalar));
  auto options = f.options;
  options.include_host_readback = true;
  ModularInputStoragePlan owned_plan;
  Ok(ComputeModularInputStoragePlan(options, &owned_plan));
  ResourceBudget budget(owned_plan.peak_bytes);
  ResourceReservation reservation;
  Ok(budget.TryReserve(owned_plan.peak_bytes, &reservation));
  std::unique_ptr<PreparedGpuModularInput> prepared;
  ModularImage host;
  {
    ResourceContextScope context({&reservation, ResourceClass::kPreparation});
    Ok(PrepareGpuModularInput(gpu, {f.view(), f.options.rct}, &prepared));
    const auto after_prepare = budget.snapshot();
    Check(after_prepare.total.backing_count == 1 &&
              after_prepare.total.live_capacity_bytes == owned_plan.output_bytes &&
              after_prepare.peak_backing_bytes == owned_plan.preparation_peak_bytes,
          "Owned Metal preparation retained upload or exceeded plan");
    Check(prepared->metadata() == owned_plan.metadata && prepared->backend_id() == gpu.id() &&
              prepared->rct() == f.options.rct && prepared->channels().size() == f.format.channels,
          "Owned Metal preparation metadata differs");
    for (size_t c = 0; c < f.format.channels; ++c) {
      const auto &channel = prepared->channels()[c];
      Check(channel.descriptor == owned_plan.channels[c] &&
                channel.plane.offset_bytes == owned_plan.planes[c].offset_bytes &&
                channel.plane.buffer == prepared->channels()[0].plane.buffer,
            "Owned Metal planes differ from plan");
    }
    Ok(ReadbackGpuModularInput(gpu, *prepared, &host));
    const auto after_readback = budget.snapshot();
    Check(after_readback.total.live_capacity_bytes ==
              owned_plan.output_bytes + owned_plan.host_readback.retained_bytes &&
              after_readback.peak_backing_bytes == owned_plan.peak_bytes &&
              after_readback.total.pending_count == 0,
          "Owned Metal readback exceeded plan or leaked pending tickets");
  }
  reservation.Reset();
  prepared.reset(); // Host samples survive both the reservation and device owner.
  Check(budget.snapshot().total.live_capacity_bytes == owned_plan.host_readback.retained_bytes,
        "Device owner retained backing or uncharged independent host output");
#if GJXL_MODULAR_INPUT_REFERENCE
  JxlMemoryManager memory;
  Check(bool(jxl::MemoryManagerInit(&memory, nullptr)), "Pinned memory manager failed");
  const auto reference = Reference(f, &memory);
#endif
  size_t end = 0;
  for (size_t c = 0; c < f.format.channels; ++c) {
    const size_t begin = (output_prefix + f.plan.planes[c].offset_bytes) / 4;
    for (size_t i = end; i < begin; ++i)
      Check(words[i] == kGuard, "Kernel overwrote a plane guard");
    const auto expected = scalar.image().view(c).backing;
    for (size_t i = 0; i < expected.size(); ++i) {
      if (words[begin + i] != expected[i])
        throw std::runtime_error("Scalar mismatch at channel " + std::to_string(c) + " pixel " +
                                 std::to_string(i) + " RCT " + std::to_string(f.options.rct));
      Check(host.view(c).backing[i] == expected[i], "Owned Metal/readback differs from scalar");
#if GJXL_MODULAR_INPUT_REFERENCE
      Check(words[begin + i] ==
                reference.channel[c].Row(i / f.options.extent.width)[i % f.options.extent.width],
            "GPU differs from pinned libjxl");
#endif
    }
    end = begin + expected.size();
    samples.fetch_add(expected.size(), std::memory_order_relaxed);
  }
  for (size_t i = end; i < words.size(); ++i)
    Check(words[i] == kGuard, "Kernel overwrote suffix guard");
  host = {};
  Check(budget.snapshot().committed_bytes() == 0 && budget.snapshot().open_reservations == 0,
        "Owned Metal path leaked resource charges");
  cases.fetch_add(1, std::memory_order_relaxed);
}

void OwnedFailures(GpuBackend &gpu) {
  const Fixture f(formats[5], {3, 2}, SampleByteOrder::kBigEndian, 41, 3, 1);
  std::unique_ptr<PreparedGpuModularInput> previous;
  Ok(PrepareGpuModularInput(gpu, {f.view(), 41}, &previous));
  const auto *identity = previous.get();
  const auto *backing = previous->channels()[0].plane.buffer;
  const auto live = DefaultResourceBudget().snapshot().total.live_capacity_bytes;
  for (unsigned fault = 0; fault < 3; ++fault) {
    if (fault == 0)
      Ok(ArmNextMetalAllocationFailureForTest(gpu));
    else
      Ok(ArmNextMetalSubmissionFailureForTest(gpu, fault == 1, fault == 2));
    const auto status = PrepareGpuModularInput(gpu, {f.view(), 41}, &previous);
    Check(status.code() == (fault == 0 ? StatusCode::kOutOfMemory :
                           fault == 1 ? StatusCode::kSubmissionFailed : StatusCode::kDeviceError) &&
              previous.get() == identity && previous->channels()[0].plane.buffer == backing,
          "Owned Metal failure replaced prior result or returned wrong error");
    const auto after = DefaultResourceBudget().snapshot();
    Check(after.total.live_capacity_bytes == live && after.total.pending_count == 0,
          "Owned Metal failure leaked backing/tickets");
    ModularImage host;
    Ok(ReadbackGpuModularInput(gpu, *previous, &host));
    ModularEncoderFrame scalar;
    Ok(ModularEncoderFrame::Prepare(f.view(), 41, &scalar));
    for (size_t c = 0; c < f.format.channels; ++c)
      Check(std::ranges::equal(host.view(c).backing, scalar.image().view(c).backing),
            "Owned Metal failure changed old result samples");
  }
  Ok(PrepareGpuModularInput(gpu, {f.view(), 41}, &previous));
  Check(previous.get() != identity, "Successful owned Metal replacement did not publish");
}

void OwnedCpuParticipation(GpuBackend &gpu) {
  using namespace thread_budget_internal;
  const Fixture f(formats[1], {3, 2}, SampleByteOrder::kLittleEndian, 6, 0, 0);
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({0, 1}, &domain));
  struct Observation {
    size_t leaves = 0, enters = 0;
  } observation;
  {
    CpuExecutionScope cpu;
    Ok(cpu.Start(domain, 1));
    current_cpu_execution.observer = {
        &observation,
        +[](void *p) noexcept { ++static_cast<Observation *>(p)->enters; },
        +[](void *p) noexcept { ++static_cast<Observation *>(p)->leaves; }};
    std::unique_ptr<PreparedGpuModularInput> prepared;
    Ok(PrepareGpuModularInput(gpu, {f.view(), 6}, &prepared));
    Check(observation.leaves == 1 && observation.enters == 1 && HasCpuParticipation() &&
              domain->snapshot().active_cpu_participants == 1,
          "Owned preparation wait did not suspend/resume CPU participation");
    Ok(ArmNextMetalSubmissionFailureForTest(gpu, false, true));
    Check(PrepareGpuModularInput(gpu, {f.view(), 6}, &prepared).code() == StatusCode::kDeviceError &&
              observation.leaves == 2 && observation.enters == 2 && HasCpuParticipation(),
          "Completion failure leaked or retained CPU participation while waiting");
  }
  Check(!HasCpuParticipation() && domain->snapshot().active_cpu_participants == 0,
        "Owned preparation leaked a CPU participant");
}

void Matrix(GpuBackend &gpu) {
  for (const auto f : formats)
    for (const auto order : {SampleByteOrder::kLittleEndian, SampleByteOrder::kBigEndian}) {
      for (uint8_t rct = 0; rct < (f.channels == 1 ? 1 : 42); ++rct)
        Run(gpu, Fixture(f, {17, 19}, order, rct, 3, 1, 11));
      for (const auto extent : {Extent2D{1, 1},
                                {1, 257},
                                {257, 1},
                                {7, 9},
                                {8, 8},
                                {9, 7},
                                {255, 256},
                                {256, 256},
                                {257, 257},
                                {2049, 3}})
        for (const uint8_t rct : {0, 6, 17, 41}) {
          if (f.channels == 1 && rct)
            continue;
          // Exact device-buffer end: no prefix, suffix or final-row padding.
          Run(gpu, Fixture(f, extent, order, rct, 0, 0), false);
        }
      for (unsigned pattern = 1; pattern <= 3; ++pattern)
        Run(gpu, Fixture(f, {9, 7}, order, f.channels == 1 ? 0 : 6, 1, 7, 0, pattern));
    }
}

class FakeBuffer final : public DeviceBuffer {
public:
  FakeBuffer(BackendId id, size_t size) : DeviceBuffer(BackendKind::kMetal, id, size) {}
};
class Completed final : public GpuSubmission {
public:
  Status Wait() override { return Status::Ok(); }
};

void RejectionAndFailures(GpuBackend &gpu, GpuBackend &other) {
  const Fixture f(formats[5], {3, 2}, SampleByteOrder::kBigEndian, 41, 3, 1);
  std::unique_ptr<DeviceBuffer> input, output, foreign;
  Ok(gpu.Allocate(f.packed.size(), &input));
  Ok(gpu.Allocate(f.plan.output_bytes, &output));
  Ok(other.Allocate(f.plan.output_bytes, &foreign));
  Ok(gpu.CopyHostToDevice(*input, f.packed.data(), f.packed.size()));
  std::vector<int32_t> guard(f.plan.output_bytes / 4, kGuard), actual(guard.size());
  Ok(gpu.CopyHostToDevice(*output, guard.data(), f.plan.output_bytes));
  const MetalModularInputDispatch valid{input.get(), f.prefix, output.get(), 0, f.options};
  FakeBuffer fake(gpu.id(), f.plan.output_bytes);
  const auto stats = gpu.stats();
  Pending pending;
  for (unsigned error = 0; error < 16; ++error) {
    auto bad = valid;
    switch (error) {
    case 0:
      bad.input = nullptr;
      break;
    case 1:
      bad.output = nullptr;
      break;
    case 2:
      bad.input_offset_bytes = SIZE_MAX;
      break;
    case 3:
      ++bad.input_offset_bytes;
      break;
    case 4:
      bad.output_offset_bytes = 4;
      break;
    case 5:
      bad.output_offset_bytes = kModularInputPlaneAlignment;
      break;
    case 6:
      bad.layout.rct = 42;
      break;
    case 7:
      bad.layout.format = static_cast<PackedModularFormat>(255);
      break;
    case 8:
      bad.layout.byte_order = static_cast<SampleByteOrder>(255);
      break;
    case 9:
      bad.layout.extent.height = 0;
      break;
    case 10:
      bad.layout.row_stride_bytes = 1;
      break;
    case 11:
      bad.input = output.get();
      bad.input_offset_bytes = 0;
      break;
    case 12:
      bad.input = foreign.get();
      break;
    case 13:
      bad.output = foreign.get();
      break;
    case 14:
      bad.output = &fake;
      break;
    case 15:
      bad.layout.include_host_readback = true;
      break;
    }
    Check(!SubmitMetalModularInput(gpu, bad, &pending.value).ok() && !pending.value,
          "Invalid kernel request submitted work");
  }
  Check(!SubmitMetalModularInput(gpu, valid, nullptr).ok(), "Null submission output accepted");
  pending.value = std::make_unique<Completed>();
  auto *old = pending.value.get();
  Check(!SubmitMetalModularInput(gpu, valid, &pending.value).ok() && pending.value.get() == old,
        "Kernel replaced an existing submission");
  pending.value.reset();
  Check(gpu.stats().committed_submissions == stats.committed_submissions &&
            gpu.stats().successful_allocations == stats.successful_allocations,
        "Invalid request allocated backing or submitted work");
  Ok(gpu.CopyDeviceToHost(*output, actual.data(), f.plan.output_bytes));
  Check(actual == guard, "Rejected request changed output");

  Ok(ArmNextMetalSubmissionFailureForTest(gpu, true, false));
  Check(SubmitMetalModularInput(gpu, valid, &pending.value).code() ==
                StatusCode::kSubmissionFailed &&
            !pending.value && gpu.stats().committed_submissions == stats.committed_submissions,
        "Injected submission failure committed work");
  Ok(gpu.CopyDeviceToHost(*output, actual.data(), f.plan.output_bytes));
  Check(actual == guard, "Submission failure changed output");
  Ok(ArmNextMetalSubmissionFailureForTest(gpu, false, true));
  Ok(SubmitMetalModularInput(gpu, valid, &pending.value));
  Check(pending.value->Wait().code() == StatusCode::kDeviceError &&
            pending.value->Wait().code() == StatusCode::kDeviceError,
        "Completion failure was not returned/cached");
  pending.value.reset();
  // A successful follow-up proves neither injection remains armed.
  Run(gpu, f);

  ModularInputDeviceLimits limits;
  Ok(GetMetalModularInputLimits(gpu, &limits));
  Check(!limits.maximum_extent.empty() && limits.maximum_row_stride_bytes &&
            limits.maximum_buffer_bytes,
        "Invalid Metal kernel limits");
  Check(!GetMetalModularInputLimits(gpu, nullptr).ok(), "Null limits output accepted");
  auto huge = valid;
  huge.layout = {{size_t{1} << 20, size_t{1} << 20}, size_t{1} << 23, PackedModularFormat::kRgba16};
  Check(SubmitMetalModularInput(gpu, huge, &pending.value).code() == StatusCode::kUnsupported &&
            !pending.value,
        "Device buffer limit did not reject a shape-only giant request");
}

void BorrowedBackingAndAccounting(GpuBackend &gpu) {
  // Disjoint ranges of one actual buffer are legal. Input ends exactly where
  // output starts, so accidental whole-buffer alias rejection is detected.
  Fixture f(formats[0], {3, 1}, SampleByteOrder::kLittleEndian, 0, 0, 0);
  constexpr size_t output_offset = kModularInputPlaneAlignment;
  constexpr size_t input_offset = output_offset - 3;
  const size_t bytes = output_offset + f.plan.output_bytes;
  ResourceBudget budget(bytes);
  ResourceReservation reservation;
  Ok(budget.TryReserve(bytes, &reservation));
  {
    ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
    std::unique_ptr<DeviceBuffer> arena;
    Ok(gpu.Allocate(bytes, &arena));
    Ok(gpu.CopyHostToDevice(*arena, f.packed.data(), f.packed.size(), input_offset));
    const auto stats = gpu.stats();
    Pending pending;
    Ok(SubmitMetalModularInput(
        gpu, {arena.get(), input_offset, arena.get(), output_offset, f.options}, &pending.value));
    Ok(pending.value->Wait());
    std::array<int32_t, 3> actual{};
    Ok(gpu.CopyDeviceToHost(*arena, actual.data(), sizeof(actual), output_offset));
    Check(std::equal(actual.begin(), actual.end(), f.source.begin()),
          "Disjoint same-buffer result differs");
    Check(gpu.stats().successful_allocations == stats.successful_allocations &&
              gpu.stats().committed_submissions == stats.committed_submissions + 1 &&
              budget.snapshot().total.backing_count == 1 &&
              budget.snapshot().peak_backing_bytes == bytes,
          "Borrowed-buffer kernel allocated unexpected backing or dispatched twice");
  }
  Check(budget.snapshot().total.backing_count == 0 && budget.snapshot().total.pending_count == 0,
        "Kernel retained borrowed backing/tickets");
}

void Concurrent(GpuBackend &gpu) {
  std::array<std::exception_ptr, 4> errors{};
  std::array<std::thread, 4> workers;
  for (size_t i = 0; i < workers.size(); ++i)
    workers[i] = std::thread([&, i] {
      try {
        for (uint8_t rct : {0, 6, 17, 41})
          Run(gpu, Fixture(formats[i + 2], {33, 17}, SampleByteOrder::kBigEndian,
                           formats[i + 2].channels == 1 ? 0 : rct, i + 1, 3));
      } catch (...) {
        errors[i] = std::current_exception();
      }
    });
  for (auto &worker : workers)
    worker.join();
  for (auto &error : errors)
    if (error)
      std::rethrow_exception(error);
}
} // namespace

int main() try {
  {
    std::unique_ptr<GpuBackend> gpu, embedded;
    Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &gpu));
    Ok(CreateEmbeddedMetalBackend({}, &embedded));
    Check(QueryGpuModularInputPreparation(*gpu) && QueryGpuModularInputPreparation(*embedded),
          "Metal backend did not expose Modular input preparation");
    std::cout << "Metal Modular device: " << gpu->name() << '\n';
    Matrix(*gpu);
    RejectionAndFailures(*gpu, *embedded);
    OwnedFailures(*gpu);
    OwnedCpuParticipation(*gpu);
    BorrowedBackingAndAccounting(*gpu);
    Concurrent(*gpu);
    for (uint8_t rct : {0, 6, 41})
      Run(*embedded, Fixture(formats[5], {17, 19}, SampleByteOrder::kBigEndian, rct, 3, 1));
  }
  const auto budget = DefaultResourceBudget().snapshot();
  Check(budget.total.backing_count == 0 && budget.total.pending_count == 0 &&
            budget.committed_bytes() == 0,
        "Metal Modular tests leaked resource accounting");
  std::cout << cases << " sample cases, " << samples
            << " exact samples; guards, rejection, failures, "
               "owned preparation/readback, accounting, concurrency, file and embedded libraries passed";
#if GJXL_MODULAR_INPUT_REFERENCE
  std::cout << "; pinned libjxl matched";
#else
  std::cout << "; pinned libjxl not configured";
#endif
  std::cout << '\n';
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
