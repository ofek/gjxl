// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/gradient.h"
#include "codestream/modular/frame_encoder.h"
#include "codestream/modular/gpu_token_handoff.h"
#include "core/cpu_execution.h"
#include "core/thread_budget.h"
#include "gpu/metal/metal_backend.h"
#include "gpu/metal/metal_modular_tokens.h"
#if GJXL_MODULAR_TOKENS_REFERENCE
#include "modular_reference.h"
#endif
#include <algorithm>
#include <atomic>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::modular_gpu_internal;
using namespace gjxl::resource_budget_internal;
std::atomic<size_t> cases{0}, tokens_checked{0}, files{0}, decoded_samples{0};
size_t diagnostic_samples = 0;
std::filesystem::path artifacts;
bool composed = false;
void Check(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
void Ok(const Status &status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
struct Format {
  PackedModularFormat format;
  size_t channels, bytes;
};
constexpr Format formats[] = {
    {PackedModularFormat::kGray8, 1, 1}, {PackedModularFormat::kRgb8, 3, 1},
    {PackedModularFormat::kRgba8, 4, 1}, {PackedModularFormat::kGray16, 1, 2},
    {PackedModularFormat::kRgb16, 3, 2}, {PackedModularFormat::kRgba16, 4, 2}};
struct Fixture {
  Format format;
  Extent2D extent;
  SampleByteOrder order;
  size_t stride;
  uint8_t rct;
  std::vector<uint8_t> packed;
  std::vector<uint16_t> samples;
  ModularEncoderFrame scalar;
  std::unique_ptr<PreparedGpuModularInput> input;
  Fixture(GpuBackend &gpu, Format f, Extent2D e = {17, 19}, uint8_t transform = 0,
          SampleByteOrder endian = SampleByteOrder::kBigEndian, unsigned pattern = 0)
      : format(f), extent(e), order(endian), stride(e.width * f.channels * f.bytes + 3),
        rct(transform), packed(1 + (e.height - 1) * stride + e.width * f.channels * f.bytes, 0xad),
        samples(e.width * e.height * f.channels) {
    const uint32_t maximum = f.bytes == 1 ? 255 : 65535;
    const uint32_t edges[] = {0, 1, 2, maximum / 2, maximum - 1, maximum};
    uint32_t random = 0x9187ac23;
    for (size_t y = 0; y < e.height; ++y)
      for (size_t x = 0; x < e.width; ++x)
        for (size_t c = 0, divisor = 1; c < f.channels; ++c, divisor *= 6) {
          random ^= random << 13;
          random ^= random >> 17;
          random ^= random << 5;
          const size_t pixel = y * e.width + x;
          uint32_t value = pixel < 216 ? edges[(pixel / divisor) % 6] : random & maximum;
          if (pattern == 1)
            value = 0;
          if (pattern == 2)
            value = maximum;
          if (pattern == 3)
            value = (x + y + c) % 2 ? maximum : 0;
          if (pattern == 4)
            value = x % 2 ? 0 : maximum;
          if (c == 3 && pixel % 3 == 0)
            value = 0;
          samples[pixel * f.channels + c] = static_cast<uint16_t>(value);
          const size_t i = 1 + y * stride + (x * f.channels + c) * f.bytes;
          packed[i] = static_cast<uint8_t>(value);
          if (f.bytes == 2) {
            packed[i] =
                static_cast<uint8_t>(endian == SampleByteOrder::kBigEndian ? value >> 8 : value);
            packed[i + 1] =
                static_cast<uint8_t>(endian == SampleByteOrder::kBigEndian ? value : value >> 8);
          }
        }
    Ok(ModularEncoderFrame::Prepare(view(), rct, &scalar));
    Ok(PrepareGpuModularInput(gpu, {view(), rct}, &input));
  }
  PackedModularImageView view() const {
    return {std::span(packed).subspan(1), extent, stride, format.format, order};
  }
  GpuModularTokenPlan Plan(ModularCodingPolicy policy) const {
    Check(policy.rct == rct, "Fixture/policy RCT mismatch");
    GpuModularTokenPlan plan;
    Ok(BuildGpuModularTokenPlan({extent, stride, order, {format.format, policy, false}}, &plan));
    return plan;
  }
};

Status Produce(GpuBackend &gpu, const Fixture &f, const GpuModularTokenPlan &plan,
               std::unique_ptr<PreparedGpuModularTokens> *out) {
  auto *op = QueryGpuModularTokenGeneration(gpu);
  return composed
             ? op->PrepareModularTokens(f.view(), plan.options.tokenization, plan.slices, out)
             : op->GenerateModularTokens(*f.input, plan.options.tokenization, plan.slices, out);
}

void Equal(const PreparedModularTokens &scalar, const PreparedGpuModularTokens &gpu) {
  ModularTokenHostView host;
  Ok(gpu.BorrowHostView(&host));
  Check(host.values.size() == scalar.tokens.size() &&
            host.contexts.size() == scalar.tokens.size() &&
            gpu.context_count() == scalar.context_count,
        "GPU token shape/context count differs");
  for (size_t i = 0; i < scalar.tokens.size(); ++i)
    if (host.values[i] != scalar.tokens[i].value || host.contexts[i] != scalar.tokens[i].context)
      throw std::runtime_error(
          "GPU token differs at " + std::to_string(i) + ": got " + std::to_string(host.values[i]) +
          "/" + std::to_string(host.contexts[i]) + ", expected " +
          std::to_string(scalar.tokens[i].value) + "/" + std::to_string(scalar.tokens[i].context));
}

void Consume(GpuBackend &gpu, const Fixture &f, const GpuModularTokenPlan &plan) {
  for (auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns})
    for (size_t participants : {size_t{1}, size_t{4}}) {
      // Direct completed-token handoff; complete-call admission has its own test.
      const ManagedHostScope resources(ResourceClass::kSerializer);
      std::shared_ptr<const ExecutionDomain> domain;
      Ok(ExecutionDomain::Create({.cpu_participant_limit = participants}, &domain));
      thread_budget_internal::CpuExecutionScope cpu;
      Ok(cpu.Start(domain, participants));
      thread_budget_internal::EncodeScope threads(participants);
      PreparedModularTokens scalar, split;
      Ok(TokenizeModular(f.scalar, plan.layout, plan.options.tokenization.coding, &scalar));
      std::unique_ptr<PreparedGpuModularTokens> owner;
      Ok(Produce(gpu, f, plan, &owner));
      Ok(AdoptGpuModularTokens(plan, std::move(owner), &split));
      ModularWorkflowStoragePlan storage;
      Ok(ComputeModularWorkflowStoragePlan(f.extent, f.format.format, mode,
                                           plan.options.tokenization.coding, &storage,
                                           participants));
      codestream_internal::CodestreamBuffer expected, actual;
      Ok(EncodeModularFrame(plan.metadata, plan.layout, std::move(scalar), mode, storage,
                            &expected));
      Ok(EncodeModularFrame(plan.metadata, plan.layout, std::move(split), mode, storage, &actual));
      Check(std::ranges::equal(actual, expected),
            "GPU token continuation changed codestream bytes");
      const auto index = files.fetch_add(1);
      std::ofstream file(artifacts / ("case-" + std::to_string(index) + ".jxl"), std::ios::binary);
      file.write(reinterpret_cast<const char *>(actual.data()), actual.size());
      Check(bool(file), "Failed to retain GPU token codestream");
#if GJXL_MODULAR_TOKENS_REFERENCE
      namespace ref = gjxl::test::modular_reference;
      const ref::IntegerImage source{f.extent.width, f.extent.height,
                                     static_cast<uint32_t>(f.format.channels),
                                     static_cast<uint32_t>(f.format.bytes * 8), f.samples};
      Check(ref::DecodeLossless(actual.view()) == source, "Independent GPU-token decode differs");
      decoded_samples += source.samples.size();
#endif
    }
}

void Run(GpuBackend &gpu, const Fixture &f, ModularCodingPolicy policy, bool encode = false,
         bool concurrent = false) {
  auto plan = f.Plan(policy);
  PreparedModularTokens scalar;
  Ok(TokenizeModular(f.scalar, plan.layout, policy, &scalar));
  const auto &p = plan.storage.tokens;
  const auto before = gpu.stats();
  // Direct shared initialization avoids the conservative staging allowance.
  const size_t cap = p.output_bytes + p.dispatch_bytes +
                     (composed ? plan.storage.input.preparation_peak_bytes : 0);
  ResourceBudget budget(cap);
  ResourceReservation reservation;
  Ok(budget.TryReserve(cap, &reservation));
  std::unique_ptr<PreparedGpuModularTokens> result;
  {
    ResourceContextScope resources({&reservation, ResourceClass::kPreparation});
    Ok(Produce(gpu, f, plan, &result));
    const auto live = budget.snapshot();
    Check(live.total.backing_count == 1 && live.total.live_capacity_bytes == p.output_bytes &&
              live.total.pending_count == 0 && live.peak_backing_bytes == cap &&
              live.peak_backing_bytes <=
                  (composed ? plan.storage.generation_peak_bytes : p.production_peak_bytes),
          "Token owner retained scratch, duplicated backing or exceeded storage plan");
    Check(result->backend_id() == gpu.id() && result->token_count() == p.token_count &&
              result->values().buffer == result->contexts().buffer &&
              result->values().size_bytes == p.values_bytes &&
              result->contexts().offset_bytes == p.contexts_offset &&
              result->contexts().size_bytes == p.contexts_bytes,
          "Completed GPU token arena differs from contract");
    Equal(scalar, *result);
    ModularTokenHostView first, second;
    Ok(result->BorrowHostView(&first));
    ArmNextManagedHostAllocationFailureForTest();
    Ok(result->BorrowHostView(&second));
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(pending && first.values.data() == second.values.data() &&
              first.contexts.data() == second.contexts.data() &&
              !result->BorrowHostView(nullptr).ok(),
          "Host borrowing copied, allocated or accepted null");
  }
  reservation.Reset();
  Check(budget.snapshot().committed_bytes() == p.output_bytes, "Token owner lost retained charge");
  Equal(scalar, *result);
  result.reset();
  Check(budget.snapshot().committed_bytes() == 0, "Token destruction leaked backing");
  if (!concurrent)
    Check(gpu.stats().successful_allocations ==
                  before.successful_allocations + (composed ? 4 : 2) &&
              gpu.stats().committed_submissions == before.committed_submissions + 1,
          "Token operation allocated extra backing or submitted extra work");
  ++cases;
  tokens_checked += scalar.tokens.size();
  if (encode)
    Consume(gpu, f, plan);
}

void Matrix(GpuBackend &gpu) {
  for (auto format : formats)
    for (auto order : {SampleByteOrder::kLittleEndian, SampleByteOrder::kBigEndian})
      for (uint8_t rct = 0; rct < (format.channels == 1 ? 1 : 42); ++rct) {
        Fixture f(gpu, format, {17, 19}, rct, order);
        for (unsigned predictor = 0; predictor < 14; ++predictor) {
          if (predictor == 6)
            continue;
          ModularCodingPolicy policy;
          policy.rct = rct;
          policy.tree.nodes[0].predictor = static_cast<Predictor>(predictor);
          Run(gpu, f, policy, predictor == 5 && rct == (format.channels == 1 ? 0 : 6));
        }
      }
  for (auto format : formats)
    for (auto extent :
         {Extent2D{1, 1}, {1, 257}, {257, 1}, {255, 256}, {256, 256}, {257, 257}, {2049, 3}}) {
      const uint8_t rct = format.channels == 1 ? 0 : 6;
      Fixture f(gpu, format, extent, rct);
      for (unsigned predictor = 0; predictor < 14; ++predictor) {
        if (predictor == 6)
          continue;
        ModularCodingPolicy policy;
        policy.rct = rct;
        policy.tree.nodes[0].predictor = static_cast<Predictor>(predictor);
        // Also exercises Gradient through the generic tree pipeline.
        policy.tree.nodes[0].offset = -17;
        Run(gpu, f, policy);
      }
    }
}

void Trees(GpuBackend &gpu) {
  Fixture f(gpu, formats[5], {257, 9}, 6);
  for (int property = 0; property < 15; ++property)
    for (int32_t split : {INT32_MIN, -65536, -1, 0, 1, 17, 21, 22, 65535, INT32_MAX - 1}) {
      ModularCodingPolicy policy;
      policy.rct = 6;
      policy.tree.size = 3;
      policy.tree.nodes[0] = {.property = property, .split = split, .left = 2, .right = 1};
      policy.tree.nodes[1].predictor = Predictor::kAverage4;
      policy.tree.nodes[1].offset = -41;
      policy.tree.nodes[2].predictor = Predictor::kSelect;
      Run(gpu, f, policy, property == 8 && split == 0);
    }
  // Crossing a DC-group boundary changes the base canonical stream ID.
  Fixture dc_groups(gpu, formats[5], {2049, 3}, 6);
  for (int32_t split : {23, 24, 25, 31, 32}) {
    ModularCodingPolicy policy;
    policy.rct = 6;
    policy.tree.size = 3;
    policy.tree.nodes[0] = {.property = 1, .split = split, .left = 2, .right = 1};
    policy.tree.nodes[1].predictor = Predictor::kAverage4;
    policy.tree.nodes[2].predictor = Predictor::kSelect;
    Run(gpu, dc_groups, policy, split == 25);
  }
  ModularCodingPolicy maximum;
  maximum.rct = 6;
  maximum.tree.size = 31;
  for (size_t i = 0; i < 15; ++i)
    maximum.tree.nodes[i] = {.property = static_cast<int32_t>(i),
                             .split = 0,
                             .left = static_cast<uint8_t>(2 * i + 2),
                             .right = static_cast<uint8_t>(2 * i + 1)};
  for (size_t i = 15; i < 31; ++i) {
    maximum.tree.nodes[i].predictor = static_cast<Predictor>(i % 6);
    maximum.tree.nodes[i].offset = static_cast<int32_t>(i) - 25;
  }
  Fixture grouped(gpu, formats[5], {257, 257}, 6);
  Run(gpu, grouped, maximum, true);
  maximum.tree = {};
  Run(gpu, grouped, maximum, true);
  for (unsigned pattern : {1u, 2u, 3u}) {
    Fixture patterned(gpu, formats[5], {33, 17}, 41, SampleByteOrder::kLittleEndian, pattern);
    for (unsigned predictor = 0; predictor < 14; ++predictor) {
      if (predictor == 6)
        continue;
      ModularCodingPolicy policy;
      policy.rct = 41;
      policy.tree.nodes[0].predictor = static_cast<Predictor>(predictor);
      Run(gpu, patterned, policy);
    }
  }
}

void LeafArithmetic(GpuBackend &gpu) {
  Fixture zero(gpu, formats[3], {3, 2}, 0, SampleByteOrder::kBigEndian, 1);
  auto baseline = zero.Plan({});
  std::unique_ptr<PreparedGpuModularTokens> previous;
  auto *op = QueryGpuModularTokenGeneration(gpu);
  Ok(op->GenerateModularTokens(*zero.input, baseline.options.tokenization, baseline.slices,
                               &previous));
  const auto *identity = previous.get();
  for (int32_t offset : {INT32_MIN, INT32_MIN + 1, -17, -1, 0, 1, 17, INT32_MAX})
    for (uint32_t multiplier : {1u, 2u, 3u, 12u, uint32_t{1} << 30, uint32_t{INT32_MAX}}) {
      ModularCodingPolicy policy;
      policy.tree.nodes[0].predictor = Predictor::kZero;
      policy.tree.nodes[0].offset = offset;
      policy.tree.nodes[0].multiplier = multiplier;
      auto plan = zero.Plan(policy);
      PreparedModularTokens scalar;
      const auto expected = TokenizeModular(zero.scalar, plan.layout, policy, &scalar);
      if (expected.ok())
        Run(gpu, zero, policy);
      else {
        const auto live = DefaultResourceBudget().snapshot().total.live_capacity_bytes;
        const auto status = Produce(gpu, zero, plan, &previous);
        Check(status.code() == expected.code() && previous.get() == identity &&
                  DefaultResourceBudget().snapshot().total.live_capacity_bytes == live,
              "Residual error was not checked or changed prior owner/charge");
      }
    }
  Fixture alternating(gpu, formats[3], {2, 1}, 0, SampleByteOrder::kBigEndian, 4);
  ModularCodingPolicy wide;
  wide.tree.nodes[0].predictor = Predictor::kLeft;
  wide.tree.nodes[0].offset = INT32_MAX;
  wide.tree.nodes[0].multiplier = 2;
  Run(gpu, alternating, wide); // Negative intermediate below INT32_MIN, valid quotient.
  Fixture ones(gpu, formats[0], {2, 1}, 0, SampleByteOrder::kBigEndian, 1);
  // Source [1,0] makes the second residual exactly INT32_MIN after leaf offset.
  ones.packed[1] = 1;
  ones.samples[0] = 1;
  Ok(ModularEncoderFrame::Prepare(ones.view(), &ones.scalar));
  Ok(PrepareGpuModularInput(gpu, {ones.view(), 0}, &ones.input));
  wide.tree.nodes[0].multiplier = 1;
  Run(gpu, ones, wide);
}

class InputView final : public PreparedGpuModularInput {
public:
  explicit InputView(const PreparedGpuModularInput &input)
      : id(input.backend_id()), metadata_(input.metadata()), rct_(input.rct()),
        count(input.channels().size()) {
    std::ranges::copy(input.channels(), channels_.begin());
  }
  BackendId backend_id() const noexcept override { return id; }
  const codec_internal::ImageMetadata &metadata() const noexcept override { return metadata_; }
  uint8_t rct() const noexcept override { return rct_; }
  std::span<const GpuModularChannelView> channels() const noexcept override {
    return std::span(channels_).first(count);
  }
  BackendId id;
  codec_internal::ImageMetadata metadata_;
  uint8_t rct_;
  size_t count;
  std::array<GpuModularChannelView, 4> channels_{};
};
class FakeBuffer final : public DeviceBuffer {
public:
  FakeBuffer(BackendId id, size_t bytes) : DeviceBuffer(BackendKind::kMetal, id, bytes) {}
};

void Rejections(GpuBackend &gpu, GpuBackend &foreign) {
  Fixture f(gpu, formats[5], {257, 3}, 6);
  ModularCodingPolicy policy;
  policy.rct = 6;
  auto plan = f.Plan(policy);
  std::unique_ptr<PreparedGpuModularTokens> previous;
  auto *op = QueryGpuModularTokenGeneration(gpu);
  Ok(op->GenerateModularTokens(*f.input, plan.options.tokenization, plan.slices, &previous));
  const auto *identity = previous.get();
  const auto before = gpu.stats();
  Fixture other(foreign, formats[5], {257, 3}, 6);
  FakeBuffer fake(gpu.id(), f.input->channels()[0].plane.buffer->size_bytes());
  for (unsigned error = 0; error < 27; ++error) {
    InputView input(*f.input);
    auto slices = plan.slices;
    auto options = plan.options.tokenization;
    StatusCode expected = StatusCode::kInvalidArgument;
    switch (error) {
    case 0:
      input.id = foreign.id();
      break;
    case 1:
      ++input.rct_;
      break;
    case 2:
      input.count = 3;
      break;
    case 3:
      input.metadata_.bits = 8;
      break;
    case 4:
      input.channels_[0].plane.buffer = other.input->channels()[0].plane.buffer;
      break;
    case 5:
      input.channels_[0].plane.buffer = &fake;
      break;
    case 6:
      input.channels_[0].plane.buffer = nullptr;
      break;
    case 7:
      ++input.channels_[0].plane.offset_bytes;
      break;
    case 8:
      input.channels_[0].plane.offset_bytes = SIZE_MAX;
      break;
    case 9:
      --input.channels_[0].plane.row_stride;
      break;
    case 10:
      input.channels_[0].plane.element_type = DeviceElementType::kF32;
      break;
    case 11:
      input.channels_[1].plane = input.channels_[0].plane;
      break;
    case 12:
      input.channels_[0].descriptor.hshift = 1;
      break;
    case 13:
      slices[0].token_offset = UINT64_MAX;
      break;
    case 14:
      ++slices[0].stream_id;
      break;
    case 15:
      slices[0].channel = 4;
      break;
    case 16:
      ++slices[0].x;
      break;
    case 17:
      ++slices[0].y;
      break;
    case 18:
      slices[0].width = 0;
      break;
    case 19:
      slices[0].height = UINT32_MAX;
      break;
    case 20:
      slices.pop_back();
      break;
    case 21:
      std::swap(slices[0], slices[1]);
      break;
    case 22:
      options.search = true;
      expected = StatusCode::kUnsupported;
      break;
    case 23:
      options.coding.tree.nodes[0].predictor = Predictor::kWeighted;
      expected = StatusCode::kUnsupported;
      break;
    case 24:
      options.coding.transforms.size = 1;
      expected = StatusCode::kUnsupported;
      break;
    case 25:
      options.coding.tree.nodes[0].multiplier = 0;
      break;
    case 26:
      break;
    }
    const auto live = DefaultResourceBudget().snapshot().total.live_capacity_bytes;
    const auto status =
        op->GenerateModularTokens(input, options, slices, error == 26 ? nullptr : &previous);
    Check(status.code() == expected && previous.get() == identity &&
              DefaultResourceBudget().snapshot().total.live_capacity_bytes == live,
          "Rejected token request changed owner/charge or returned wrong status");
  }
  Check(gpu.stats().successful_allocations == before.successful_allocations &&
            gpu.stats().committed_submissions == before.committed_submissions,
        "Rejected token request allocated or submitted work");
}

void Failures(GpuBackend &gpu) {
  Fixture f(gpu, formats[5], {257, 3}, 6);
  ModularCodingPolicy policy;
  policy.rct = 6;
  auto plan = f.Plan(policy);
  PreparedModularTokens scalar;
  Ok(TokenizeModular(f.scalar, plan.layout, policy, &scalar));
  auto *op = QueryGpuModularTokenGeneration(gpu);
  std::unique_ptr<PreparedGpuModularTokens> previous;
  Ok(op->GenerateModularTokens(*f.input, plan.options.tokenization, plan.slices, &previous));
  const auto *identity = previous.get();
  const auto live = DefaultResourceBudget().snapshot().total.live_capacity_bytes;
  for (unsigned fault = 0; fault < 3; ++fault) {
    if (fault == 0)
      Ok(ArmNextMetalAllocationFailureForTest(gpu));
    else
      Ok(ArmNextMetalSubmissionFailureForTest(gpu, fault == 1, fault == 2));
    const auto status =
        op->GenerateModularTokens(*f.input, plan.options.tokenization, plan.slices, &previous);
    Check(status.code() == (fault == 0   ? StatusCode::kOutOfMemory
                            : fault == 1 ? StatusCode::kSubmissionFailed
                                         : StatusCode::kDeviceError) &&
              previous.get() == identity &&
              DefaultResourceBudget().snapshot().total.live_capacity_bytes == live &&
              DefaultResourceBudget().snapshot().total.pending_count == 0,
          "GPU token fault changed output or leaked pending/live storage");
    Equal(scalar, *previous);
  }
  // Fail admission of the second native allocation, then prove exact overlap
  // with a retained previous result. Neither path may escape the reservation.
  for (size_t cap :
       {plan.storage.tokens.output_bytes,
        2 * plan.storage.tokens.output_bytes + plan.storage.tokens.dispatch_bytes - 1}) {
    ResourceBudget budget(cap);
    ResourceReservation reservation;
    Ok(budget.TryReserve(cap, &reservation));
    {
      ResourceContextScope resources({&reservation, ResourceClass::kPreparation});
      std::unique_ptr<PreparedGpuModularTokens> current;
      if (cap != plan.storage.tokens.output_bytes)
        Ok(op->GenerateModularTokens(*f.input, plan.options.tokenization, plan.slices, &current));
      const auto *old = current.get();
      const auto charged = budget.snapshot().total.live_capacity_bytes;
      const auto status =
          op->GenerateModularTokens(*f.input, plan.options.tokenization, plan.slices, &current);
      Check(status.resource_plan_exceeded() && current.get() == old &&
                budget.snapshot().total.live_capacity_bytes == charged &&
                budget.snapshot().total.pending_count == 0,
            "Token storage underplanning was not terminal and atomic");
    }
    Check(budget.snapshot().total.backing_count == 0, "Failed token replacement leaked charge");
  }
  Ok(op->GenerateModularTokens(*f.input, plan.options.tokenization, plan.slices, &previous));
  Equal(scalar, *previous);
  ModularImage samples;
  Ok(ReadbackGpuModularInput(gpu, *f.input, &samples));
  for (size_t c = 0; c < f.format.channels; ++c)
    Check(std::ranges::equal(samples.view(c).backing, f.scalar.image().view(c).backing),
          "Token generation modified source planes");
}

void Participation(GpuBackend &gpu) {
  using namespace thread_budget_internal;
  Fixture f(gpu, formats[1], {3, 2});
  auto plan = f.Plan({});
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({.cpu_participant_limit = 1}, &domain));
  struct Observation {
    size_t entered = 0, left = 0;
  } observation;
  {
    CpuExecutionScope cpu;
    Ok(cpu.Start(domain, 1));
    current_cpu_execution.observer = {
        &observation, +[](void *p) noexcept { ++static_cast<Observation *>(p)->entered; },
        +[](void *p) noexcept { ++static_cast<Observation *>(p)->left; }};
    std::unique_ptr<PreparedGpuModularTokens> tokens;
    for (bool failure : {false, true}) {
      if (failure)
        Ok(ArmNextMetalSubmissionFailureForTest(gpu, false, true));
      const auto status = Produce(gpu, f, plan, &tokens);
      Check(status.code() == (failure ? StatusCode::kDeviceError : StatusCode::kOk) &&
                observation.entered == (failure ? 2u : 1u) &&
                observation.left == observation.entered && HasCpuParticipation(),
            "GPU token wait did not suspend/resume CPU participation");
    }
  }
  Check(domain->snapshot().active_cpu_participants == 0, "GPU token wait leaked CPU participant");
}

void Diagnostics(GpuBackend &gpu) {
  auto check = [&](const Fixture &f) {
    ModularCodingPolicy policy;
    policy.rct = f.rct;
    policy.tree.size = 31;
    for (size_t i = 0; i < 15; ++i)
      policy.tree.nodes[i] = {.property = static_cast<int32_t>(i),
                              .split = 0,
                              .left = static_cast<uint8_t>(2 * i + 2),
                              .right = static_cast<uint8_t>(2 * i + 1)};
    for (size_t i = 15; i < 31; ++i) {
      policy.tree.nodes[i].predictor = static_cast<Predictor>(i % 14 == 6 ? 5 : i % 14);
      policy.tree.nodes[i].offset = static_cast<int32_t>(i) - 20;
    }
    const auto plan = f.Plan(policy);
    ManagedVector<metal_internal::ModularTokenDiagnostic> trace;
    Ok(metal_internal::DiagnoseMetalModularTokens(gpu, *f.input, plan.options.tokenization,
                                                  plan.slices, &trace));
    TreeLayout tree;
    Ok(ValidateTree(policy.tree, &tree));
    Check(trace.size() == plan.storage.tokens.token_count, "Diagnostic sample count mismatch");
    for (const auto &slice : plan.slices) {
      ModularChannelView view;
      Ok(BorrowChannelSlice(f.scalar.image().view(slice.channel),
                            {slice.x, slice.y, {slice.width, slice.height}}, &view));
      for (size_t y = 0; y < slice.height; ++y) {
        int64_t previous = 0;
        for (size_t x = 0; x < slice.width; ++x) {
          const auto neighbors = Neighbors(view, x, y);
          const auto properties =
              Properties(neighbors, slice.channel, slice.stream_id, x, y, previous, 0);
          previous = properties[9];
          const auto leaf = Lookup(policy.tree, properties);
          const auto &node = policy.tree.nodes[leaf];
          const int64_t prediction = Predict(node.predictor, neighbors, 0);
          const int32_t sample = view.Row(y)[x];
          const int32_t residual =
              static_cast<int32_t>((sample - prediction - node.offset) / node.multiplier);
          const auto &actual = trace[slice.token_offset + y * slice.width + x];
          for (size_t i = 0; i < 15; ++i)
            Check(actual.properties[i] == properties[i], "GPU intermediate property differs");
          for (unsigned i = 0; i < 14; ++i)
            if (i != 6)
              Check(actual.predictions[i] == Predict(static_cast<Predictor>(i), neighbors, 0),
                    "GPU intermediate predictor differs");
          Check(actual.leaf == leaf && actual.context == tree.context[leaf] &&
                    actual.selected_prediction == prediction && actual.sample == sample &&
                    actual.residual == residual && actual.value == PackSigned(residual),
                "GPU intermediate leaf/prediction/residual/token differs");
          ++diagnostic_samples;
        }
      }
    }
  };
  for (auto format : formats) {
    for (uint8_t rct = 0; rct < (format.channels == 1 ? 1 : 42); ++rct)
      check(Fixture(gpu, format, {17, 19}, rct));
  }
  for (auto extent : {Extent2D{1, 257}, {257, 1}, {255, 256}, {256, 256}, {257, 257}, {2049, 3}})
    check(Fixture(gpu, formats[5], extent, 6, SampleByteOrder::kLittleEndian));
  Fixture f(gpu, formats[5], {3, 2}, 6);
  ModularCodingPolicy policy;
  policy.rct = 6;
  const auto plan = f.Plan(policy);
  ManagedVector<metal_internal::ModularTokenDiagnostic> old(1);
  const auto *identity = old.data();
  const auto live = DefaultResourceBudget().snapshot().total.live_capacity_bytes;
  ArmNextManagedHostAllocationFailureForTest();
  const auto status = metal_internal::DiagnoseMetalModularTokens(
      gpu, *f.input, plan.options.tokenization, plan.slices, &old);
  const bool pending = ManagedHostAllocationFailurePendingForTest();
  DisarmManagedHostAllocationFailureForTest();
  Check(status.code() == StatusCode::kOutOfMemory && !pending && old.data() == identity &&
            DefaultResourceBudget().snapshot().total.live_capacity_bytes == live,
        "Diagnostic readback failure replaced old output or leaked backing");
}
void PackedRejections(GpuBackend &gpu) {
  Fixture f(gpu, formats[5], {257, 3}, 6);
  ModularCodingPolicy policy;
  policy.rct = 6;
  const auto plan = f.Plan(policy);
  auto *op = QueryGpuModularTokenGeneration(gpu);
  std::unique_ptr<PreparedGpuModularTokens> old;
  Ok(op->PrepareModularTokens(f.view(), plan.options.tokenization, plan.slices, &old));
  const auto *identity = old.get();
  const auto stats = gpu.stats();
  const auto live = DefaultResourceBudget().snapshot().total.live_capacity_bytes;
  for (unsigned error = 0; error < 11; ++error) {
    auto input = f.view();
    auto options = plan.options.tokenization;
    auto slices = plan.slices;
    switch (error) {
    case 0:
      input.bytes = input.bytes.first(input.bytes.size() - 1);
      break;
    case 1:
      input.row_stride = 1;
      break;
    case 2:
      input.byte_order = static_cast<SampleByteOrder>(99);
      break;
    case 3:
      input.extent.width = 0;
      break;
    case 4:
      options.format = PackedModularFormat::kRgba8;
      break;
    case 5:
      options.search = true;
      break;
    case 6:
      options.coding.tree.nodes[0].predictor = Predictor::kWeighted;
      break;
    case 7:
      ++slices[0].token_offset;
      break;
    case 8:
      ++slices[0].stream_id;
      break;
    case 9:
      slices.pop_back();
      break;
    case 10:
      break;
    }
    // Copying descriptors above is outside the producer's admission boundary.
    const auto before = DefaultResourceBudget().snapshot().total.live_capacity_bytes;
    Check(!op->PrepareModularTokens(input, options, slices, error == 10 ? nullptr : &old).ok() &&
              old.get() == identity &&
              DefaultResourceBudget().snapshot().total.live_capacity_bytes == before,
          "Malformed packed token request changed output or allocated backing");
  }
  Check(gpu.stats().successful_allocations == stats.successful_allocations &&
            gpu.stats().committed_submissions == stats.committed_submissions &&
            DefaultResourceBudget().snapshot().total.live_capacity_bytes == live,
        "Rejected packed token request allocated/submitted or leaked");
}

void Concurrent(GpuBackend &gpu) {
  std::array<std::exception_ptr, 4> errors{};
  std::array<std::thread, 4> workers;
  for (size_t i = 0; i < workers.size(); ++i)
    workers[i] = std::thread([&, i] {
      try {
        Fixture f(gpu, formats[5], {257, 17}, static_cast<uint8_t>(i * 7 + 6));
        for (unsigned predictor : {1u, 5u, 13u}) {
          ModularCodingPolicy policy;
          policy.rct = f.rct;
          policy.tree.nodes[0].predictor = static_cast<Predictor>(predictor);
          Run(gpu, f, policy, false, true);
        }
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

int main(int argc, char **argv) try {
  artifacts = argc > 1 ? argv[1] : "metal-modular-tokens";
  std::filesystem::create_directories(artifacts);
  const bool smoke = argc > 2 && std::string_view(argv[2]) == "--smoke";
  composed = argc > 2 && std::string_view(argv[2]) == "--composed";
  {
    std::unique_ptr<GpuBackend> gpu, embedded;
    Ok(CreateMetalBackend(GJXL_METALLIB_PATH, &gpu));
    Check(QueryGpuModularTokenGeneration(*gpu), "Metal token capability missing");
    std::cout << "Metal Modular token device: " << gpu->name() << std::endl;
    {
      Fixture f(*gpu, formats[5], {17, 19}, 6);
      ModularCodingPolicy policy;
      policy.rct = 6;
      Run(*gpu, f, policy, true);
      policy.tree.nodes[0].predictor = Predictor::kAverage4;
      Run(*gpu, f, policy);
    }
    if (!smoke) {
      Matrix(*gpu);
      Trees(*gpu);
      LeafArithmetic(*gpu);
      Ok(CreateEmbeddedMetalBackend({}, &embedded));
      Rejections(*gpu, *embedded);
      Failures(*gpu);
      Participation(*gpu);
      Concurrent(*gpu);
      PackedRejections(*gpu);
      if (!composed)
        Diagnostics(*gpu);
      Fixture f(*embedded, formats[5], {257, 3}, 41);
      ModularCodingPolicy policy;
      policy.rct = 41;
      Run(*embedded, f, policy);
    }
  }
  const auto budget = DefaultResourceBudget().snapshot();
  Check(budget.total.backing_count == 0 && budget.total.pending_count == 0 &&
            budget.committed_bytes() == 0,
        "GPU Modular tokens leaked resource accounting");
  std::cout << cases << " token cases, " << tokens_checked << " exact tokens, " << files
            << " byte-identical files, " << decoded_samples << " independently decoded samples; "
            << diagnostic_samples
            << " intermediate traces; "
               "ownership, resources and failures passed"
            << std::endl;
} catch (const std::exception &e) {
  DisarmManagedHostAllocationFailureForTest();
  std::cerr << e.what() << '\n';
  return 1;
}
