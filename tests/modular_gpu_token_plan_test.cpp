// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codestream/modular/gpu_token_plan.h"
#include "codestream/modular/tokenization.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::modular_gpu_internal;
using namespace gjxl::resource_budget_internal;
size_t layout_cases = 0, policy_cases = 0;
void Check(bool v, const char *m) {
  if (!v)
    throw std::runtime_error(m);
}
void Ok(const Status &s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
struct Format {
  PackedModularFormat format;
  size_t channels, bytes;
};
constexpr std::array<Format, 6> formats{{{PackedModularFormat::kGray8, 1, 1},
                                         {PackedModularFormat::kRgb8, 3, 1},
                                         {PackedModularFormat::kRgba8, 4, 1},
                                         {PackedModularFormat::kGray16, 1, 2},
                                         {PackedModularFormat::kRgb16, 3, 2},
                                         {PackedModularFormat::kRgba16, 4, 2}}};
GpuModularTokenPlanOptions Options(Extent2D extent = {513, 257}, Format f = formats[1]) {
  return {extent,
          extent.width * f.channels * f.bytes + 3,
          SampleByteOrder::kLittleEndian,
          {f.format, {}, false}};
}
void Eligibility() {
  for (auto f : formats)
    for (uint8_t rct = 0; rct < (f.channels == 1 ? 1 : 42); ++rct)
      for (unsigned predictor = 0; predictor < 14; ++predictor) {
        ModularTokenizationOptions options{f.format, {}, false};
        options.coding.rct = rct;
        options.coding.tree.nodes[0].predictor = static_cast<Predictor>(predictor);
        ModularTokenEligibility result;
        const auto status = ResolveModularTokenEligibility(options, &result);
        if (predictor == 6) {
          Check(status.code() == StatusCode::kUnsupported, "Weighted predictor accepted");
          continue;
        }
        Ok(status);
        Check(result.channel_count == f.channels && result.context_count == 1 &&
                  result.node_count == 1 && result.nodes[0].context == 0 &&
                  result.nodes[0].predictor == predictor &&
                  result.gradient_leaf == (predictor == 5),
              "Incorrect supported predictor/RCT eligibility");
        for (size_t i = 1; i < result.nodes.size(); ++i)
          Check(result.nodes[i] == ModularTokenNode{}, "Unused nodes not zeroed");
        ++policy_cases;
      }
  // Deliberately permuted node indices: contexts follow BFS leaves, not indices.
  ModularTokenizationOptions options;
  options.coding.tree.size = 3;
  options.coding.tree.nodes[0] = {.property = 0, .split = 0, .left = 2, .right = 1};
  for (int property = 0; property < 16; ++property) {
    options.coding.tree.nodes[0].property = property;
    ModularTokenEligibility e;
    auto status = ResolveModularTokenEligibility(options, &e);
    if (property == 15)
      Check(status.code() == StatusCode::kUnsupported, "Weighted property accepted");
    else {
      Ok(status);
      Check(e.context_count == 2 && e.nodes[2].context == 0 && e.nodes[1].context == 1 &&
                !e.gradient_leaf && e.nodes[0].left == 2 && e.nodes[0].right == 1,
            "Tree compilation changed branch direction or context order");
    }
  }
  options = {};
  options.coding.tree.size = 31;
  for (size_t i = 0; i < 15; ++i) {
    // Unique properties along any path avoid contradictory splits.
    options.coding.tree.nodes[i] = {.property = static_cast<int32_t>(i),
                                    .split = 0,
                                    .left = static_cast<uint8_t>(2 * i + 1),
                                    .right = static_cast<uint8_t>(2 * i + 2)};
  }
  ModularTokenEligibility e;
  Ok(ResolveModularTokenEligibility(options, &e));
  Check(e.context_count == 16 && e.node_count == 31, "Maximum tree rejected");
  for (size_t i = 15; i < 31; ++i)
    Check(e.nodes[i].context == i - 15, "Maximum-tree context mapping differs");
  options = {};
  for (int32_t offset : {INT32_MIN, -1, 0, INT32_MAX})
    for (uint32_t multiplier : {1u, 3u, uint32_t{1} << 30, uint32_t{INT32_MAX}}) {
      options.coding.tree.nodes[0].offset = offset;
      options.coding.tree.nodes[0].multiplier = multiplier;
      Ok(ResolveModularTokenEligibility(options, &e));
      Check(e.nodes[0].offset == offset && e.nodes[0].multiplier == multiplier &&
                e.gradient_leaf == (offset == 0 && multiplier == 1),
            "Eligibility narrowed leaf arithmetic or prechecked unknown samples");
    }
  Ok(CheckModularTokenDeviceStatus(0));
  Check(CheckModularTokenDeviceStatus(kModularTokenResidualError).code() ==
                StatusCode::kInvalidArgument &&
            CheckModularTokenDeviceStatus(2).code() == StatusCode::kInternal &&
            CheckModularTokenDeviceStatus(3).code() == StatusCode::kInternal,
        "Device status protocol changed");
}

void PolicyFailures() {
  ModularTokenEligibility e;
  Ok(ResolveModularTokenEligibility({}, &e));
  const auto old = e;
  for (int error = 0; error < 15; ++error) {
    ModularTokenizationOptions bad;
    StatusCode expected = StatusCode::kInvalidArgument;
    switch (error) {
    case 0:
      bad.format = static_cast<PackedModularFormat>(255);
      break;
    case 1:
      bad.coding.rct = 42;
      break;
    case 2:
      bad.format = PackedModularFormat::kGray8;
      bad.coding.rct = 1;
      break;
    case 3:
      bad.coding.tree.size = 0;
      break;
    case 4:
      bad.coding.tree.size = 32;
      break;
    case 5:
      bad.coding.tree.nodes[0].predictor = Predictor::kInvalid;
      break;
    case 6:
      bad.coding.tree.nodes[0].multiplier = 0;
      break;
    case 7:
      bad.coding.tree.nodes[0].multiplier = uint32_t{1} << 31;
      break;
    case 8:
      bad.coding.weighted.weights[0] = 16;
      break;
    case 9:
      bad.coding.transforms.size = 9;
      break;
    case 10:
      bad.search = true;
      expected = StatusCode::kUnsupported;
      break;
    case 11:
      bad.coding.transforms.size = 1;
      expected = StatusCode::kUnsupported;
      break;
    case 12:
      bad.coding.tree.nodes[0].predictor = Predictor::kWeighted;
      expected = StatusCode::kUnsupported;
      break;
    case 13:
      bad.coding.tree.size = 3;
      bad.coding.tree.nodes[0] = {.property = 15, .split = 0, .left = 1, .right = 2};
      expected = StatusCode::kUnsupported;
      break;
    case 14:
      bad.coding.tree.size = 3;
      bad.coding.tree.nodes[0] = {.property = 8, .split = 0, .left = 0, .right = 2};
      break;
    }
    Check(ResolveModularTokenEligibility(bad, &e).code() == expected && e == old,
          "Eligibility failure code/publication changed");
    auto shape = Options();
    shape.tokenization = bad;
    GpuModularTokenStoragePlan plan, previous;
    Check(ComputeGpuModularTokenStoragePlan(shape, &plan).code() == expected && plan == previous,
          "Composed planner did not gate policy atomically");
  }
  Check(!ResolveModularTokenEligibility({}, nullptr).ok(), "Null eligibility accepted");
}

void Layouts() {
  for (auto f : formats)
    for (auto extent : {Extent2D{1, 1},
                        {1, 257},
                        {257, 1},
                        {17, 19},
                        {255, 256},
                        {256, 256},
                        {257, 257},
                        {513, 257},
                        {2047, 3},
                        {2048, 3},
                        {2049, 3}})
      for (auto order : {SampleByteOrder::kLittleEndian, SampleByteOrder::kBigEndian}) {
        auto options = Options(extent, f);
        options.byte_order = order;
        options.tokenization.coding.rct = f.channels == 1 ? 0 : 41;
        GpuModularTokenPlan p;
        Ok(BuildGpuModularTokenPlan(options, &p));
        const size_t n = extent.width * extent.height * f.channels;
        const auto &s = p.storage;
        Check(s.tokens.token_count == n && s.tokens.values_bytes == 4 * n &&
                  s.tokens.contexts_bytes == 2 * n && s.input.host_readback == HostStorageBound{} &&
                  s.tokens.contexts_offset % 256 == 0 &&
                  s.tokens.contexts_offset >= s.tokens.values_bytes &&
                  s.tokens.output_bytes == s.tokens.contexts_offset + s.tokens.contexts_bytes,
              "Compact token layout has overlap, truncation or hidden host samples");
        Check(s.generation_peak_bytes == s.input.upload_bytes + s.input.output_bytes +
                                             s.tokens.output_bytes + s.tokens.dispatch_bytes +
                                             s.tokens.staging.peak_bytes +
                                             s.host_layout.peak_bytes &&
                  s.retained_bytes == s.tokens.output_bytes + s.host_layout.retained_bytes,
              "Composed storage omitted live overlap or retained transient buffers");
        const bool global = extent.width <= 256 && extent.height <= 256;
        size_t next = 0, slice_index = 0;
        for (size_t i = 0; i < p.layout.streams.size(); ++i) {
          const auto &stream = p.layout.streams[i];
          const auto range = p.ranges[i];
          Check(range.offset == next && stream.slice_begin == slice_index,
                "Canonical stream ranges/order changed");
          const bool active =
              global ? stream.role == StreamRole::kGlobal : stream.role == StreamRole::kGroup;
          Check(stream.slice_count == (active ? f.channels : 0), "Unexpected active/empty stream");
          if (stream.role == StreamRole::kGlobal)
            Check(stream.id == 0, "Global ID changed");
          if (stream.role == StreamRole::kDcGroup)
            Check(stream.id == 1 + s.geometry.dc_group_count() + stream.group, "DC ID changed");
          if (stream.role == StreamRole::kGroup)
            Check(stream.id == 18 + 3 * s.geometry.dc_group_count() + stream.group,
                  "Group ID changed");
          for (size_t c = 0; c < stream.slice_count; ++c) {
            const auto &d = p.slices[slice_index++];
            const size_t x = global ? 0 : (stream.group % s.geometry.groups().width) * 256;
            const size_t y = global ? 0 : (stream.group / s.geometry.groups().width) * 256;
            const size_t w = global ? extent.width : std::min(size_t{256}, extent.width - x);
            const size_t h = global ? extent.height : std::min(size_t{256}, extent.height - y);
            Check(d == ModularTokenSlice{next, static_cast<uint32_t>(stream.id),
                                         static_cast<uint32_t>(c), static_cast<uint32_t>(x),
                                         static_cast<uint32_t>(y), static_cast<uint32_t>(w),
                                         static_cast<uint32_t>(h)},
                  "GPU slice offset/coordinates differ from canonical raster tiles");
            next += w * h;
          }
          Check(range.count == next - range.offset, "Stream token count differs");
        }
        Check(next == n && slice_index == p.slices.size(), "Token partition has gaps/overlap");
        ++layout_cases;
      }
  GpuModularTokenPlan p;
  Ok(BuildGpuModularTokenPlan(Options({513, 257}, formats[2]), &p));
  Check(p.layout.streams.size() == 8 && p.slices.size() == 24 &&
            p.ranges[0] == ModularTokenRange{} && p.ranges[1] == ModularTokenRange{} &&
            p.slices.front() == ModularTokenSlice{0, 21, 0, 0, 0, 256, 256} &&
            p.slices.back() == ModularTokenSlice{527363, 26, 3, 512, 256, 1, 1},
        "Hand-calculated clipped RGBA layout differs");
  ModularTokenStoragePlan storage;
  Ok(ComputeModularTokenStoragePlan(18, 3, &storage));
  Check(storage.values_bytes == 72 && storage.contexts_offset == 256 &&
            storage.contexts_bytes == 36 && storage.output_bytes == 292 &&
            storage.slices_bytes == 96 && storage.nodes_offset == 256 &&
            storage.status_offset == 1280 && storage.dispatch_bytes == 1284 &&
            storage.production_peak_bytes == 2860 && storage.retained_bytes == 292,
        "Hand-calculated descriptor/status/token arenas differ");
}

void ScalarAgreement() {
  // Large enough to exercise group resets; actual CPU tokens verify the handoff
  // ranges and metadata against the existing producer, not only shape formulas.
  for (auto extent : {Extent2D{3, 2}, {257, 3}, {3, 257}}) {
    auto options = Options(extent);
    options.tokenization.coding.rct = 6;
    options.tokenization.coding.tree.size = 3;
    options.tokenization.coding.tree.nodes[0] = {.property = 8, .split = 0, .left = 2, .right = 1};
    options.tokenization.coding.tree.nodes[1].predictor = Predictor::kAverage4;
    std::vector<uint8_t> bytes(options.row_stride_bytes * extent.height);
    for (size_t i = 0; i < bytes.size(); ++i)
      bytes[i] = static_cast<uint8_t>(i * 53);
    ModularEncoderFrame frame;
    Ok(ModularEncoderFrame::Prepare({bytes, extent, options.row_stride_bytes},
                                    options.tokenization.coding.rct, &frame));
    GpuModularTokenPlan p;
    Ok(BuildGpuModularTokenPlan(options, &p));
    PreparedModularTokens tokens;
    Ok(TokenizeModular(frame, p.layout, options.tokenization.coding, &tokens));
    Check(tokens.context_count == p.storage.eligibility.context_count &&
              tokens.tokens.size() == p.storage.tokens.token_count &&
              tokens.streams.size() == p.ranges.size(),
          "Scalar token shape differs");
    for (size_t i = 0; i < p.ranges.size(); ++i) {
      auto range = p.ranges[i];
      Check(tokens.streams[i].size() == range.count, "Scalar stream length differs");
      for (size_t j = 0; j < range.count; ++j)
        Check(tokens.streams[i][j] == tokens.tokens[range.offset + j],
              "Scalar stream offset differs");
    }
    auto metadata = frame.metadata();
    metadata.modular_16_bit_buffer_sufficient = false;
    Check(p.metadata == metadata, "Final policy metadata differs from CPU continuation");
  }
}

void LimitsAndFailures() {
  auto options = Options({3, 2}, formats[4]);
  options.row_stride_bytes = 19;
  GpuModularTokenStoragePlan p;
  Ok(ComputeGpuModularTokenStoragePlan(options, &p));
  const auto old = p;
  ModularInputDeviceLimits in{{3, 2}, 19, 536};
  ModularTokenDeviceLimits limits{18, 3, 1284};
  Ok(ComputeGpuModularTokenStoragePlan(options, &p, &in, &limits));
  for (int i = 0; i < 3; ++i) {
    auto small = limits;
    if (i == 0)
      --small.maximum_tokens;
    if (i == 1)
      --small.maximum_slices;
    if (i == 2)
      --small.maximum_buffer_bytes;
    Check(ComputeGpuModularTokenStoragePlan(options, &p, &in, &small).code() ==
                  StatusCode::kUnsupported &&
              p == old,
          "Token device limit not enforced atomically");
    auto invalid = limits;
    if (i == 0)
      invalid.maximum_tokens = 0;
    if (i == 1)
      invalid.maximum_slices = 0;
    if (i == 2)
      invalid.maximum_buffer_bytes = 0;
    Check(ComputeGpuModularTokenStoragePlan(options, &p, &in, &invalid).code() ==
                  StatusCode::kInvalidArgument &&
              p == old,
          "Invalid limits treated as unbounded");
  }
  auto small = in;
  --small.maximum_buffer_bytes;
  Check(ComputeGpuModularTokenStoragePlan(options, &p, &small, &limits).code() ==
                StatusCode::kUnsupported &&
            p == old,
        "Input device limit omitted in composition");
  for (int i = 0; i < 5; ++i) {
    auto bad = options;
    if (i == 0)
      bad.extent.width = 0;
    if (i == 1)
      bad.extent.height = SIZE_MAX;
    if (i == 2)
      bad.row_stride_bytes = 17;
    if (i == 3)
      bad.row_stride_bytes = SIZE_MAX;
    if (i == 4)
      bad.byte_order = static_cast<SampleByteOrder>(255);
    Check(ComputeGpuModularTokenStoragePlan(bad, &p).code() == StatusCode::kInvalidArgument &&
              p == old,
          "Invalid source shape changed previous plan");
  }
  ModularTokenStoragePlan t;
  Ok(ComputeModularTokenStoragePlan(1024, 1, &t));
  const auto before = t;
  ModularTokenDeviceLimits exact{1024, 1, t.output_bytes};
  Ok(ComputeModularTokenStoragePlan(1024, 1, &t, &exact));
  --exact.maximum_buffer_bytes;
  Check(ComputeModularTokenStoragePlan(1024, 1, &t, &exact).code() == StatusCode::kUnsupported &&
            t == before,
        "Token output arena limit omitted");
  for (auto counts : {std::pair<size_t, size_t>{0, 1},
                      {1, 0},
                      {1, 2},
                      {SIZE_MAX, 1},
                      {SIZE_MAX / 4, SIZE_MAX / 4},
                      {size_t(PTRDIFF_MAX) / 4, 1}})
    Check(!ComputeModularTokenStoragePlan(counts.first, counts.second, &t).ok() && t == before,
          "Invalid/overflowed token storage accepted");
  // Each arena fits the pointer range; simultaneous output + dispatch + staging
  // exceeds size_t. This exercises aggregate overflow independently of buffers.
  const size_t many = size_t(PTRDIFF_MAX) / 8;
  Check(!ComputeModularTokenStoragePlan(many, many / 6, &t).ok() && t == before,
        "Aggregate live storage overflow accepted");
  Check(!ComputeGpuModularTokenStoragePlan(options, nullptr).ok() &&
            !BuildGpuModularTokenPlan(options, nullptr).ok() &&
            !ComputeModularTokenStoragePlan(1, 1, nullptr).ok(),
        "Null plan outputs accepted");
}

void ResourcesAndLargeGeometry() {
  const auto options = Options();
  GpuModularTokenStoragePlan p;
  Ok(ComputeGpuModularTokenStoragePlan(options, &p));
  ResourceBudget budget(p.host_layout.peak_bytes);
  ResourceReservation reservation;
  Ok(budget.TryReserve(p.host_layout.peak_bytes, &reservation));
  {
    ResourceContextScope scope({&reservation, ResourceClass::kPreparation});
    ArmNextManagedHostAllocationFailureForTest();
    const auto status = ComputeGpuModularTokenStoragePlan(options, &p);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Ok(status);
    Check(pending && budget.snapshot().total.backing_count == 0,
          "Shape planning allocated backing");
    GpuModularTokenPlan plan;
    Ok(BuildGpuModularTokenPlan(options, &plan));
    Check(budget.snapshot().total.live_capacity_bytes == p.host_layout.retained_bytes &&
              budget.snapshot().peak_backing_bytes <= p.host_layout.peak_bytes,
          "Host layout backing differs from planned bound");
    plan = {};
    Ok(BuildGpuModularTokenPlan(Options({1, 1}), &plan));
    const auto *old_slices = plan.slices.data();
    const auto old_storage = plan.storage;
    const auto old_charge = budget.snapshot().total.live_capacity_bytes;
    const auto replacement = BuildGpuModularTokenPlan(options, &plan);
    Check(replacement.resource_plan_exceeded() && plan.slices.data() == old_slices &&
              plan.storage == old_storage &&
              budget.snapshot().total.live_capacity_bytes == old_charge,
          "Replacement ignored live old backing or changed it after admission failure");
    plan = {};
    Ok(BuildGpuModularTokenPlan(options, &plan));
  }
  Check(budget.snapshot().total.backing_count == 0, "Host layout leaked backing");
  GpuModularTokenPlan previous;
  Ok(BuildGpuModularTokenPlan(Options({1, 1}), &previous));
  const auto *slices = previous.slices.data();
  const auto saved = previous.storage;
  const auto charged = DefaultResourceBudget().snapshot().total.live_capacity_bytes;
  auto unsupported = options;
  unsupported.tokenization.search = true;
  ArmNextManagedHostAllocationFailureForTest();
  const auto rejected = BuildGpuModularTokenPlan(unsupported, &previous);
  const bool untouched = ManagedHostAllocationFailurePendingForTest();
  DisarmManagedHostAllocationFailureForTest();
  Check(rejected.code() == StatusCode::kUnsupported && untouched &&
            previous.slices.data() == slices && previous.storage == saved,
        "Unsupported policy allocated or replaced a live plan");
  size_t faults = 0;
  for (; faults < 8; ++faults) {
    ArmManagedHostAllocationFailureAfterForTest(faults);
    const auto status = BuildGpuModularTokenPlan(options, &previous);
    DisarmManagedHostAllocationFailureForTest();
    if (status.ok())
      break;
    Check(status.code() == StatusCode::kOutOfMemory && previous.slices.data() == slices &&
              previous.storage == saved &&
              DefaultResourceBudget().snapshot().total.live_capacity_bytes == charged,
          "Metadata allocation failure changed old plan or leaked charge");
  }
  Check(faults == 4, "Not every canonical/descriptor/range allocation was injected");
  previous = {};
  if constexpr (sizeof(size_t) >= 8) {
    const auto large = Options({size_t{1} << 20, size_t{1} << 20}, formats[5]);
    Ok(ComputeGpuModularTokenStoragePlan(large, &p));
    Check(p.tokens.token_count == (uint64_t{1} << 42) &&
              p.tokens.values_bytes == (uint64_t{1} << 44) &&
              p.tokens.contexts_bytes == (uint64_t{1} << 43) &&
              p.tokens.slice_count == (uint64_t{1} << 26),
          "Large offsets/counts narrowed");
    const auto old = p;
    // Same legal area with a very wide axis; no sample or descriptor allocation.
    const auto square = Options({size_t{1} << 27, size_t{1} << 13}, formats[0]);
    Ok(ComputeGpuModularTokenStoragePlan(square, &p));
    Check(p.tokens.token_count == (uint64_t{1} << 40), "Wide geometry was truncated");
    auto too_large = large;
    ++too_large.extent.height;
    p = old;
    Check(!ComputeGpuModularTokenStoragePlan(too_large, &p).ok() && p == old,
          "CPU source area limit not enforced");
  }
}
} // namespace
int main() try {
  Eligibility();
  PolicyFailures();
  Layouts();
  ScalarAgreement();
  LimitsAndFailures();
  ResourcesAndLargeGeometry();
  Check(DefaultResourceBudget().snapshot().committed_bytes() == 0, "Default resource leak");
  std::cout << "GPU Modular token contract passed: " << policy_cases << " policies, "
            << layout_cases << " layouts; limits, atomic failures and resource bounds verified\n";
} catch (const std::exception &e) {
  DisarmManagedHostAllocationFailureForTest();
  std::cerr << e.what() << '\n';
  return 1;
}
