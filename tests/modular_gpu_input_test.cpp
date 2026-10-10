// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "gpu/ops/modular_input_storage_plan.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
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

struct FormatCase {
  PackedModularFormat format;
  size_t channels, bytes_per_sample;
};
constexpr std::array<FormatCase, 6> kFormats{
    {{PackedModularFormat::kGray8, 1, 1},
     {PackedModularFormat::kRgb8, 3, 1},
     {PackedModularFormat::kRgba8, 4, 1},
     {PackedModularFormat::kGray16, 1, 2},
     {PackedModularFormat::kRgb16, 3, 2},
     {PackedModularFormat::kRgba16, 4, 2}}};

// Only a byte range, never actual GPU storage. Bind planned planes through the
// ordinary device-view validator to check the contract without a GPU/runtime.
class RangeBuffer final : public DeviceBuffer {
public:
  explicit RangeBuffer(size_t bytes)
      : DeviceBuffer(BackendKind::kMetal, 42, bytes) {}
};

void Layouts() {
  for (const auto f : kFormats) {
    for (const auto extent : {Extent2D{1, 1},
                              {1, 257},
                              {257, 1},
                              {17, 19},
                              {255, 256},
                              {256, 256},
                              {257, 257},
                              {2049, 3}}) {
      const size_t row = extent.width * f.channels * f.bytes_per_sample;
      for (const size_t padding : {size_t{0}, size_t{1}, size_t{7}}) {
        ModularInputStorageOptions options{extent, row + padding, f.format};
        ModularInputStoragePlan plan;
        Ok(ComputeModularInputStoragePlan(options, &plan));
        Check(plan.channel_count == f.channels && plan.input_row_bytes == row &&
                  plan.upload_bytes ==
                      row * extent.height + padding * (extent.height - 1),
              "Packed upload lost samples or included trailing padding");
        Check(plan.host_readback == HostStorageBound{} &&
                  plan.readback_peak_bytes == 0 &&
                  plan.peak_bytes == plan.preparation_peak_bytes &&
                  plan.preparation_peak_bytes ==
                      plan.upload_bytes + plan.output_bytes,
              "Resident-only plan retained host readback storage");
        RangeBuffer buffer(plan.output_bytes);
        std::array<DeviceMemoryRange, 4> ranges{};
        for (size_t c = 0; c < f.channels; ++c) {
          const auto &p = plan.planes[c];
          Check(p.element_type == DeviceElementType::kI32 &&
                    p.extent == extent && p.row_stride == extent.width &&
                    p.offset_bytes % 256 == 0 &&
                    p.size_bytes == 4 * extent.width * extent.height,
                "Wrong integer plane layout");
          Check(plan.channels[c] ==
                    ChannelDescriptor{extent, 0, 0,
                                      c == 3 ? ChannelRole::kAlpha
                                             : ChannelRole::kColor},
                "GPU preparation changed a channel role, shape or shift");
          Ok(ComputeDevicePlaneRange(
              {&buffer, p.offset_bytes, p.element_type, p.extent, p.row_stride},
              42, &ranges[c]));
          for (size_t prior = 0; prior < c; ++prior)
            Check(!DeviceRangesOverlap(ranges[prior], ranges[c]),
                  "Planned device planes overlap");
        }
        Check(plan.planes[0].offset_bytes == 0 &&
                  plan.planes[f.channels - 1].offset_bytes +
                          plan.planes[f.channels - 1].size_bytes ==
                      plan.output_bytes,
              "Output arena has unexplained prefix or trailing storage");
        for (size_t c = f.channels; c < 4; ++c)
          Check(plan.channels[c] == ChannelDescriptor{} &&
                    plan.planes[c] == DevicePlaneLayout{},
                "Unused layout entries were not initialized");
      }
    }
  }

  // Hand-calculated strided RGB16 fixture: 18 bytes/row, 37-byte upload,
  // three 24-byte planes at offsets 0, 256, 512, and a 536-byte arena.
  ModularInputStoragePlan plan;
  Ok(ComputeModularInputStoragePlan({{3, 2}, 19, PackedModularFormat::kRgb16},
                                    &plan));
  Check(plan.upload_bytes == 37 && plan.output_bytes == 536 &&
            plan.peak_bytes == 573 && plan.planes[1].offset_bytes == 256 &&
            plan.planes[2].offset_bytes == 512,
        "Known strided/aligned layout changed");
}

void CpuAgreement() {
  for (const auto f : kFormats) {
    const size_t row = 3 * f.channels * f.bytes_per_sample, stride = row + 1;
    // Odd source offset and row stride; the last row has exactly enough bytes.
    std::vector<uint8_t> backing(3 + stride + row, 0xad);
    for (auto order :
         {SampleByteOrder::kLittleEndian, SampleByteOrder::kBigEndian}) {
      const PackedModularImageView input{
          std::span(backing).subspan(3), {3, 2}, stride, f.format, order};
      Ok(input.Validate());
      size_t required = 0;
      Ok(ComputePackedModularInputSize(input.extent, f.format, order, stride,
                                       &required));
      Check(required == input.bytes.size(),
            "Packed size disagrees with exact backing");
      auto short_input = input;
      short_input.bytes = short_input.bytes.first(required - 1);
      Check(!short_input.Validate().ok(), "Short final row was accepted");
      auto null_input = input;
      null_input.bytes = {};
      Check(!null_input.Validate().ok(), "Null backing was accepted");
      for (uint8_t rct = 0; rct < (f.channels == 1 ? 1 : 42); ++rct) {
        ModularInputStoragePlan plan;
        Ok(ComputeModularInputStoragePlan(
            {input.extent, stride, f.format, order, rct}, &plan));
        ModularEncoderFrame cpu;
        Ok(ModularEncoderFrame::Prepare(input, rct, &cpu));
        Check(plan.metadata == cpu.metadata() &&
                  plan.channel_count == cpu.image().channel_count(),
              "GPU input plan metadata differs from scalar preparation");
        for (size_t c = 0; c < f.channels; ++c)
          Check(plan.channels[c] == cpu.image().view(c).descriptor,
                "GPU input descriptor differs from scalar preparation");
      }
    }
  }
}

void Failures() {
  const ModularInputStorageOptions valid{
      {3, 2}, 19, PackedModularFormat::kRgb16};
  ModularInputStoragePlan plan;
  Ok(ComputeModularInputStoragePlan(valid, &plan));
  const auto previous = plan;
  for (unsigned error = 0; error < 13; ++error) {
    auto bad = valid;
    switch (error) {
    case 0:
      bad.extent.width = 0;
      break;
    case 1:
      bad.extent.height = 0;
      break;
    case 2:
      bad.extent.width = SIZE_MAX;
      break;
    case 3:
      bad.extent.height = size_t{1} << 30;
      break;
    case 4:
      bad.row_stride_bytes = 0;
      break;
    case 5:
      bad.row_stride_bytes = 17;
      break;
    case 6:
      bad.row_stride_bytes = SIZE_MAX;
      break;
    case 7:
      bad.row_stride_bytes = PTRDIFF_MAX;
      break;
    case 8:
      bad.format = static_cast<PackedModularFormat>(255);
      break;
    case 9:
      bad.byte_order = static_cast<SampleByteOrder>(255);
      break;
    case 10:
      bad.rct = 42;
      break;
    case 11:
      bad.rct = 255;
      break;
    case 12:
      bad.format = PackedModularFormat::kGray16;
      bad.rct = 1;
      break;
    }
    Check(ComputeModularInputStoragePlan(bad, &plan).code() ==
                  StatusCode::kInvalidArgument &&
              plan == previous,
          "Invalid GPU input plan was accepted or changed the old plan");
  }
  Check(!ComputeModularInputStoragePlan(valid, nullptr).ok(),
        "Null plan output accepted");
  size_t size = 73;
  Check(!ComputePackedModularInputSize({3, 2}, valid.format, valid.byte_order,
                                       SIZE_MAX, &size)
                .ok() &&
            size == 73,
        "Failed packed sizing changed output");
  Check(!ComputePackedModularInputSize({3, 2}, valid.format, valid.byte_order,
                                       19, nullptr)
             .ok(),
        "Null packed size output accepted");
  // Valid byte orders are ignored for 8-bit samples, but unknown enum values
  // must still be rejected, as in the CPU input contract.
  Check(!ComputePackedModularInputSize({1, 1}, PackedModularFormat::kGray8,
                                       static_cast<SampleByteOrder>(255), 1,
                                       &size)
             .ok(),
        "Invalid 8-bit byte order accepted");
}

void DeviceLimits() {
  ModularInputStorageOptions options{{3, 2}, 19, PackedModularFormat::kRgb16};
  const ModularInputDeviceLimits exact{{3, 2}, 19, 536};
  ModularInputStoragePlan plan;
  Ok(ComputeModularInputStoragePlan(options, &plan, &exact));
  const auto previous = plan;
  for (unsigned error = 0; error < 4; ++error) {
    auto small = exact;
    if (error == 0)
      --small.maximum_extent.width;
    if (error == 1)
      --small.maximum_extent.height;
    if (error == 2)
      --small.maximum_row_stride_bytes;
    if (error == 3)
      --small.maximum_buffer_bytes;
    Check(ComputeModularInputStoragePlan(options, &plan, &small).code() ==
                  StatusCode::kUnsupported &&
              plan == previous,
          "Exceeded device limit was accepted or changed output");
  }
  // An input with large inter-row padding can exceed the buffer limit even
  // when all output planes fit. Stride itself is legal for this backend.
  options.row_stride_bytes = 600;
  auto limits = exact;
  limits.maximum_row_stride_bytes = 600;
  Check(ComputeModularInputStoragePlan(options, &plan, &limits).code() ==
                StatusCode::kUnsupported &&
            plan == previous,
        "Upload buffer limit was not checked");
  for (unsigned error = 0; error < 4; ++error) {
    auto invalid = exact;
    if (error == 0)
      invalid.maximum_extent.width = 0;
    if (error == 1)
      invalid.maximum_extent.height = 0;
    if (error == 2)
      invalid.maximum_row_stride_bytes = 0;
    if (error == 3)
      invalid.maximum_buffer_bytes = 0;
    Check(ComputeModularInputStoragePlan(options, &plan, &invalid).code() ==
                  StatusCode::kInvalidArgument &&
              plan == previous,
          "Malformed device limits were treated as unlimited");
  }
}

void HostReadback() {
  for (const auto f : kFormats) {
    ModularInputStorageOptions options{
        {17, 19}, 17 * f.channels * f.bytes_per_sample,
        f.format, SampleByteOrder::kLittleEndian,
        0,        true};
    ModularInputStoragePlan plan;
    Ok(ComputeModularInputStoragePlan(options, &plan));
    Check(plan.readback_peak_bytes ==
                  plan.output_bytes + plan.host_readback.peak_bytes &&
              plan.peak_bytes == std::max(plan.preparation_peak_bytes,
                                          plan.readback_peak_bytes),
          "Readback plan did not use the maximum live phase");
    ResourceBudget budget(plan.host_readback.peak_bytes);
    ResourceReservation reservation;
    Ok(budget.TryReserve(plan.host_readback.peak_bytes, &reservation));
    {
      ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
      ModularImage image;
      Ok(ModularImage::Create(
          std::span(plan.channels).first(plan.channel_count), 0, &image));
      Check(budget.snapshot().total.live_capacity_bytes ==
                    plan.host_readback.retained_bytes &&
                budget.snapshot().peak_backing_bytes <=
                    plan.host_readback.peak_bytes,
            "Actual host readback destination exceeded or disagreed with its "
            "plan");
    }
    Check(budget.snapshot().total.backing_count == 0,
          "Host readback destination leaked backing");
  }
}

void NoAllocationAndLargeGeometry() {
  ResourceBudget budget(1);
  ResourceReservation reservation;
  Ok(budget.TryReserve(1, &reservation));
  ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
  ArmNextManagedHostAllocationFailureForTest();
  ModularInputStoragePlan plan;
  auto status = ComputeModularInputStoragePlan({{513, 257},
                                                4105,
                                                PackedModularFormat::kRgba16,
                                                SampleByteOrder::kBigEndian,
                                                41,
                                                true},
                                               &plan);
  const bool pending = ManagedHostAllocationFailurePendingForTest();
  DisarmManagedHostAllocationFailureForTest();
  Ok(status);
  Check(pending && budget.snapshot().total.backing_count == 0,
        "Shape-only GPU planning allocated managed backing");

  // A one-row source has no stride multiplication or trailing padding.
  Ok(ComputeModularInputStoragePlan(
      {{1, 1}, SIZE_MAX, PackedModularFormat::kGray8}, &plan));
  Check(plan.upload_bytes == 1 && plan.output_bytes == 4 &&
            plan.peak_bytes == 5,
        "Single-row layout required unaddressed stride padding");
  if constexpr (sizeof(size_t) >= 8) {
    // Exact format area limit, with byte offsets well beyond 32 bits. No pixel
    // backing is created. Device limits remain a separate mandatory runtime
    // gate.
    ModularInputStorageOptions large{{size_t{1} << 20, size_t{1} << 20},
                                     size_t{1} << 23,
                                     PackedModularFormat::kRgba16};
    Ok(ComputeModularInputStoragePlan(large, &plan));
    Check(plan.upload_bytes == (uint64_t{1} << 43) &&
              plan.output_bytes == (uint64_t{1} << 44),
          "Large byte offsets were truncated");
    const auto previous = plan;
    ++large.extent.height;
    Check(!ComputeModularInputStoragePlan(large, &plan).ok() &&
              plan == previous,
          "GPU planner exceeded the CPU format area limit");
  }
}
} // namespace

int main() try {
  Layouts();
  CpuAgreement();
  Failures();
  DeviceLimits();
  HostReadback();
  NoAllocationAndLargeGeometry();
  std::cout << "GPU Modular input contract and storage planning passed\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
