// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/profile.h"
#include "codestream/modular/frame_encoder.h"
#include "codestream/modular/gpu_token_handoff.h"
#include "core/thread_budget.h"
#include "core/worker_launch_internal.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::modular_gpu_internal;
using namespace gjxl::resource_budget_internal;
using namespace gjxl::thread_budget_internal;
using codestream_internal::CodestreamBuffer;

void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Ok(const Status &status) {
  if (!status.ok())
    throw std::runtime_error(std::string(status.message()));
}
static_assert(!std::is_copy_constructible_v<PreparedModularTokens>);
static_assert(!std::is_copy_assignable_v<PreparedModularTokens>);
static_assert(std::is_nothrow_move_constructible_v<PreparedModularTokens>);
static_assert(std::is_nothrow_move_assignable_v<PreparedModularTokens>);

struct Probe {
  size_t destroyed = 0, borrowed = 0;
  ModularProfile *profile = nullptr;
  CpuParticipantTracker *tracker = nullptr;
  bool model_done = false, emission_done = false, before_assembly = false, joined = false;
  bool fail_assembly = false;
};

// CPU-only stand-in for the completed shared-buffer contract. Its single managed
// arena has the real compact layout/charge. It never creates a backend or maps a
// device range; the test deliberately fails if the adapter tries those accessors.
class SharedTokens final : public PreparedGpuModularTokens {
public:
  SharedTokens(const GpuModularTokenPlan &plan, const PreparedModularTokens &scalar, Probe *probe)
      : storage_(plan.storage.tokens), count(storage_.token_count),
        contexts_count(scalar.context_count), probe_(probe) {
    const ManagedHostScope resources(ResourceClass::kCompletedFrame);
    arena_.resize(storage_.output_bytes);
    for (size_t i = 0; i < count; ++i) {
      const uint32_t value = scalar.tokens[i].value;
      const uint16_t context = static_cast<uint16_t>(scalar.tokens[i].context);
      std::memcpy(arena_.data() + i * sizeof(value), &value, sizeof(value));
      std::memcpy(arena_.data() + storage_.contexts_offset + i * sizeof(context), &context,
                  sizeof(context));
    }
  }
  ~SharedTokens() override {
    ++probe_->destroyed;
    if (probe_->profile) {
      const auto &s = probe_->profile->seconds;
      probe_->model_done = s[size_t(ProfileStage::kModel)] > 0;
      probe_->emission_done = s[size_t(ProfileStage::kEmission)] > 0;
      probe_->before_assembly = s[size_t(ProfileStage::kAssembly)] == 0;
    }
    probe_->joined = !probe_->tracker || probe_->tracker->active() == 1;
    if (probe_->fail_assembly)
      ArmNextManagedHostAllocationFailureForTest();
  }
  BackendId backend_id() const noexcept override { return {}; }
  size_t token_count() const noexcept override { return count; }
  size_t context_count() const noexcept override { return contexts_count; }
  DeviceMemoryRange values() const noexcept override { std::terminate(); }
  DeviceMemoryRange contexts() const noexcept override { std::terminate(); }
  Status BorrowHostView(ModularTokenHostView *out) const override {
    ++probe_->borrowed;
    if (borrow_failure != StatusCode::kOk)
      return {borrow_failure, "Injected borrow failure"};
    if (!out)
      return Status::InvalidArgument("Null test host view");
    *out = {std::span(reinterpret_cast<const uint32_t *>(arena_.data()),
                      storage_.token_count - short_values),
            std::span(reinterpret_cast<const uint16_t *>(arena_.data() + storage_.contexts_offset),
                      storage_.token_count - short_contexts)};
    return Status::Ok();
  }
  void CorruptContext() {
    const uint16_t invalid = static_cast<uint16_t>(contexts_count);
    std::memcpy(arena_.data() + storage_.contexts_offset, &invalid, sizeof(invalid));
  }

private:
  ModularTokenStoragePlan storage_;

public:
  size_t count, contexts_count;
  size_t short_values = 0, short_contexts = 0;
  StatusCode borrow_failure = StatusCode::kOk;

private:
  Probe *probe_;
  ManagedVector<std::byte, ResourceClass::kCompletedFrame> arena_;
};

struct Fixture {
  GpuModularTokenPlan plan;
  ModularEncoderFrame frame;
  std::vector<uint8_t> packed;
  Fixture(Extent2D extent = {3, 2}, PackedModularFormat format = PackedModularFormat::kRgb8,
          unsigned policy_case = 0) {
    ModularInputProfile input;
    Ok(ResolveModularInput(extent, format, &input));
    GpuModularTokenPlanOptions options{extent,
                                       extent.width * input.channel_count * input.bytes_per_sample +
                                           3,
                                       SampleByteOrder::kBigEndian,
                                       {format, {}, false}};
    auto &coding = options.tokenization.coding;
    if (policy_case) {
      coding.rct = input.channel_count > 1 ? 6 : 0;
      coding.tree.size = 3;
      coding.tree.nodes[0] = {.property = 8, .split = 0, .left = 2, .right = 1};
      coding.tree.nodes[1].predictor = Predictor::kAverage4;
      coding.tree.nodes[1].offset = -17;
      coding.tree.nodes[2].offset = 23;
    }
    if (policy_case == 2) {
      coding.tree = {};
      coding.tree.nodes[0].multiplier = 12;
    }
    if (policy_case == 3) {
      coding.tree.size = 31;
      for (size_t i = 0; i < 15; ++i)
        coding.tree.nodes[i] = {.property = static_cast<int32_t>(i),
                                .split = 0,
                                .left = static_cast<uint8_t>(2 * i + 1),
                                .right = static_cast<uint8_t>(2 * i + 2)};
    }
    packed.resize(options.row_stride_bytes * extent.height + 1, 0x55);
    for (size_t y = 0; y < extent.height; ++y)
      for (size_t x = 0; x < extent.width; ++x)
        for (size_t c = 0; c < input.channel_count; ++c) {
          const size_t offset = 1 + y * options.row_stride_bytes +
                                (x * input.channel_count + c) * input.bytes_per_sample;
          const uint32_t v = policy_case == 2 || (c == 3 && x % 3 == 0)
                                 ? 0
                                 : ((x * 197 + y * 71 + c * 1013) ^ (x * y * 19));
          packed[offset] = static_cast<uint8_t>(v);
          if (input.bytes_per_sample == 2) {
            packed[offset] = static_cast<uint8_t>(v >> 8);
            packed[offset + 1] = static_cast<uint8_t>(v);
          }
        }
    Ok(BuildGpuModularTokenPlan(options, &plan));
    Ok(ModularEncoderFrame::Prepare({std::span(packed).subspan(1), extent, options.row_stride_bytes,
                                     format, options.byte_order},
                                    coding, &frame));
  }
  PreparedModularTokens Scalar() const {
    PreparedModularTokens tokens;
    Ok(TokenizeModular(frame, plan.layout, plan.options.tokenization.coding, &tokens));
    return tokens;
  }
  std::unique_ptr<PreparedGpuModularTokens> Owner(const PreparedModularTokens &scalar,
                                                  Probe *probe) const {
    return std::make_unique<SharedTokens>(plan, scalar, probe);
  }
};

void Equal(const PreparedModularTokens &scalar, const PreparedModularTokens &split) {
  Check(split.gpu_backing && split.tokens.empty() && scalar.tree_tokens == split.tree_tokens &&
            scalar.tree_token_count == split.tree_token_count &&
            scalar.context_count == split.context_count && scalar.policy == split.policy &&
            scalar.streams.size() == split.streams.size(),
        "Token metadata or ownership differs");
  for (size_t i = 0; i < scalar.streams.size(); ++i) {
    Check(split.streams[i].valid() && split.streams[i].split &&
              split.streams[i].size() == scalar.streams[i].size(),
          "Split stream shape differs");
    for (size_t j = 0; j < scalar.streams[i].size(); ++j)
      Check(scalar.streams[i][j] == split.streams[i][j], "Split stream contents differ");
  }
}

void StorageAndMoves() {
  Fixture f;
  auto scalar = f.Scalar();
  HostStorageBound bound;
  ArmNextManagedHostAllocationFailureForTest();
  Ok(ComputeGpuModularTokenHandoffStorageBound(f.plan.ranges.size(), &bound));
  Check(ManagedHostAllocationFailurePendingForTest(), "Handoff planning allocated");
  DisarmManagedHostAllocationFailureForTest();
  Check(bound.retained_bytes == f.plan.ranges.size() * sizeof(EntropyTokenStreamView) &&
            bound.peak_bytes == bound.retained_bytes,
        "Handoff backing bound is not exact");
  const auto original = bound;
  for (size_t count : {size_t{0}, SIZE_MAX})
    Check(!ComputeGpuModularTokenHandoffStorageBound(count, &bound).ok() && bound == original,
          "Invalid handoff bound changed output");
  Check(!ComputeGpuModularTokenHandoffStorageBound(1, nullptr).ok(), "Null bound accepted");
  Probe probe;
  const size_t cap = f.plan.storage.tokens.output_bytes + bound.peak_bytes;
  ResourceBudget budget(cap);
  ResourceReservation reservation;
  Ok(budget.TryReserve(cap, &reservation));
  {
    ResourceContextScope resources({&reservation, ResourceClass::kPreparation});
    auto owner = f.Owner(scalar, &probe);
    const auto *identity = owner.get();
    ModularTokenHostView host;
    Ok(owner->BorrowHostView(&host));
    PreparedModularTokens split;
    ArmManagedHostAllocationFailureAfterForTest(1);
    Ok(AdoptGpuModularTokens(f.plan, std::move(owner), &split));
    Check(ManagedHostAllocationFailurePendingForTest(), "Handoff copied token backing");
    DisarmManagedHostAllocationFailureForTest();
    Check(!owner && split.gpu_backing.get() == identity &&
              split.streams[0].values.data() == host.values.data() &&
              split.streams[0].contexts.data() == host.contexts.data() &&
              budget.snapshot().total.live_capacity_bytes == cap &&
              budget.snapshot().total.backing_count == 2,
          "Handoff duplicated, copied or lost backing/charge");
    Equal(scalar, split);
    ArmNextManagedHostAllocationFailureForTest();
    PreparedModularTokens moved(std::move(split)), assigned;
    assigned = std::move(moved);
    Check(ManagedHostAllocationFailurePendingForTest() && !split.gpu_backing &&
              split.streams.empty() && !moved.gpu_backing && moved.streams.empty() &&
              assigned.gpu_backing.get() == identity,
          "Token move allocated or left borrowed views behind");
    DisarmManagedHostAllocationFailureForTest();
    f.plan = {}; // Neither the borrowed spans nor copied tree metadata use plan backing.
    Equal(scalar, assigned);
    std::thread destroy([tokens = std::move(assigned)]() mutable { tokens = {}; });
    destroy.join();
    Check(probe.destroyed == 1 && budget.snapshot().total.backing_count == 0,
          "Cross-thread token destruction leaked backing");
  }
}

void RejectionsAndReplacement() {
  for (unsigned failure = 0; failure < 17; ++failure) {
    Fixture f;
    auto scalar = f.Scalar();
    Probe old_probe, new_probe;
    auto owner = f.Owner(scalar, &old_probe);
    PreparedModularTokens previous;
    Ok(AdoptGpuModularTokens(f.plan, std::move(owner), &previous));
    auto candidate = f.Owner(scalar, &new_probe);
    auto *test = static_cast<SharedTokens *>(candidate.get());
    auto *old_owner = previous.gpu_backing.get();
    auto *old_views = previous.streams.data();
    StatusCode expected = StatusCode::kInvalidArgument;
    switch (failure) {
    case 0:
      ++test->count;
      break;
    case 1:
      ++test->contexts_count;
      break;
    case 2:
      test->borrow_failure = expected = StatusCode::kUnsupported;
      break;
    case 3:
      test->borrow_failure = expected = StatusCode::kDeviceError;
      break;
    case 4:
      test->short_values = 1;
      break;
    case 5:
      test->short_contexts = 1;
      break;
    case 6:
      ++f.plan.ranges[0].offset;
      break;
    case 7:
      f.plan.ranges[1].offset = 0;
      break;
    case 8:
      f.plan.ranges[0].count = SIZE_MAX;
      break;
    case 9:
      --f.plan.ranges[0].count;
      break;
    case 10:
      f.plan.ranges.pop_back();
      break;
    case 11:
      f.plan.layout.streams[0].slice_begin = SIZE_MAX;
      break;
    case 12:
      --f.plan.layout.streams[0].slice_count;
      break;
    case 13:
      ++f.plan.storage.tokens.output_bytes;
      break;
    case 14:
      f.plan.options.tokenization.search = true;
      expected = StatusCode::kUnsupported;
      break;
    case 15:
      break; // Null output.
    case 16:
      candidate.reset();
      break;
    }
    auto *new_owner = candidate.get();
    const auto charged = DefaultResourceBudget().snapshot().total.live_capacity_bytes;
    ArmNextManagedHostAllocationFailureForTest();
    const auto status =
        AdoptGpuModularTokens(f.plan, std::move(candidate), failure == 15 ? nullptr : &previous);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(status.code() == expected && pending && candidate.get() == new_owner &&
              previous.gpu_backing.get() == old_owner && previous.streams.data() == old_views &&
              old_probe.destroyed == 0 &&
              DefaultResourceBudget().snapshot().total.live_capacity_bytes == charged,
          "Rejected handoff allocated, consumed input, replaced output or leaked");
    Equal(scalar, previous);
  }
  Fixture f;
  auto scalar = f.Scalar();
  Probe old_probe, new_probe, cpu_replacement;
  PreparedModularTokens previous;
  auto old = f.Owner(scalar, &old_probe);
  Ok(AdoptGpuModularTokens(f.plan, std::move(old), &previous));
  Check(!AdoptGpuModularTokens(f.plan, std::move(previous.gpu_backing), &previous).ok() &&
            previous.gpu_backing && old_probe.destroyed == 0,
        "Aliased owner was consumed");
  auto candidate = f.Owner(scalar, &new_probe);
  const auto *identity = candidate.get();
  const auto *old_views = previous.streams.data();
  const auto before = DefaultResourceBudget().snapshot().total.live_capacity_bytes;
  ArmNextManagedHostAllocationFailureForTest();
  const auto failed = AdoptGpuModularTokens(f.plan, std::move(candidate), &previous);
  DisarmManagedHostAllocationFailureForTest();
  Check(failed.code() == StatusCode::kOutOfMemory && candidate.get() == identity &&
            previous.streams.data() == old_views && old_probe.destroyed == 0 &&
            DefaultResourceBudget().snapshot().total.live_capacity_bytes == before,
        "Failed metadata allocation consumed owner/output or leaked");
  Ok(AdoptGpuModularTokens(f.plan, std::move(candidate), &previous));
  Check(old_probe.destroyed == 1 && new_probe.destroyed == 0, "Replacement released wrong owner");
  auto bad_policy = f.plan.options.tokenization.coding;
  bad_policy.tree.size = 0;
  Check(!TokenizeModular(f.frame, f.plan.layout, bad_policy, &previous).ok() &&
            previous.gpu_backing.get() == identity,
        "Failed scalar producer replaced GPU owner");
  Ok(TokenizeModular(f.frame, f.plan.layout, {}, &previous));
  Check(!previous.gpu_backing && new_probe.destroyed == 1 && !previous.streams[0].split,
        "Scalar producer did not release and replace GPU owner");
  auto replacing_cpu = f.Owner(scalar, &cpu_replacement);
  const auto *cpu_data = previous.tokens.data();
  const auto *cpu_views = previous.streams.data();
  ArmNextManagedHostAllocationFailureForTest();
  const auto cpu_failure = AdoptGpuModularTokens(f.plan, std::move(replacing_cpu), &previous);
  DisarmManagedHostAllocationFailureForTest();
  Check(cpu_failure.code() == StatusCode::kOutOfMemory && replacing_cpu &&
            previous.tokens.data() == cpu_data && previous.streams.data() == cpu_views &&
            !previous.gpu_backing,
        "Failed GPU handoff discarded scalar backing");
  Ok(AdoptGpuModularTokens(f.plan, std::move(replacing_cpu), &previous));
  Equal(scalar, previous);
  previous = {};
  Check(cpu_replacement.destroyed == 1, "CPU-to-GPU replacement leaked its owner");

  HostStorageBound views;
  Ok(ComputeGpuModularTokenHandoffStorageBound(f.plan.ranges.size(), &views));
  const size_t cap = 2 * f.plan.storage.tokens.output_bytes + views.peak_bytes;
  ResourceBudget budget(cap);
  ResourceReservation reservation;
  Ok(budget.TryReserve(cap, &reservation));
  {
    ResourceContextScope resources({&reservation, ResourceClass::kSerializer});
    Probe a, b;
    auto first = f.Owner(scalar, &a);
    auto second = f.Owner(scalar, &b);
    PreparedModularTokens tokens;
    Ok(AdoptGpuModularTokens(f.plan, std::move(first), &tokens));
    const auto *old_identity = tokens.gpu_backing.get();
    const auto status = AdoptGpuModularTokens(f.plan, std::move(second), &tokens);
    Check(status.resource_plan_exceeded() && second && tokens.gpu_backing.get() == old_identity &&
              a.destroyed == 0 && budget.snapshot().total.live_capacity_bytes == cap,
          "Replacement escaped reservation or discarded old owner");
    tokens = {};
    Ok(AdoptGpuModularTokens(f.plan, std::move(second), &tokens));
  }
  Check(budget.snapshot().total.backing_count == 0, "Replacement retained charge");
}

size_t encoded_cases = 0;
void EncodeCase(Fixture &f, EntropyCodingMode mode, size_t participants) {
  auto scalar = f.Scalar();
  auto oracle = f.Scalar();
  Probe probe;
  ModularWorkflowStoragePlan storage;
  Ok(ComputeModularWorkflowStoragePlan(f.plan.options.extent, f.plan.options.tokenization.format,
                                       mode, f.plan.options.tokenization.coding, &storage,
                                       participants));
  CodestreamBuffer expected;
  Ok(EncodeModularFrame(f.plan.metadata, f.plan.layout, std::move(oracle), mode, storage,
                        &expected));
  HostStorageBound views;
  Ok(ComputeGpuModularTokenHandoffStorageBound(f.plan.ranges.size(), &views));
  // Conservative test admission, not the future full GPU-workflow planner:
  // the unchanged CPU continuation bound plus compact backing and new views.
  const size_t cap =
      storage.working.peak_bytes + f.plan.storage.tokens.output_bytes + views.peak_bytes;
  ResourceBudget budget(cap);
  ResourceReservation reservation;
  Ok(budget.TryReserve(cap, &reservation));
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({.cpu_participant_limit = participants}, &domain));
  CpuParticipantTracker tracker;
  ModularProfile profile;
  {
    ResourceContextScope resources({&reservation, ResourceClass::kSerializer});
    CpuExecutionScope cpu;
    Ok(cpu.Start(domain, participants));
    EncodeScope threads(participants, &tracker);
    auto owner = f.Owner(scalar, &probe);
    PreparedModularTokens tokens;
    Ok(AdoptGpuModularTokens(f.plan, std::move(owner), &tokens));
    Equal(scalar, tokens);
    probe.profile = &profile;
    probe.tracker = &tracker;
    ProfileSession profiling(&profile);
    CodestreamBuffer actual;
    Ok(EncodeModularFrame(f.plan.metadata, f.plan.layout, std::move(tokens), mode, storage,
                          &actual));
    Check(std::ranges::equal(actual, expected), "Split/CPU codestream differs");
    Check(probe.destroyed == 1 && probe.model_done && probe.emission_done &&
              probe.before_assembly && probe.joined && !tokens.gpu_backing &&
              budget.snapshot().classes[size_t(ResourceClass::kCompletedFrame)].backing_count ==
                  0 &&
              budget.snapshot().total.live_capacity_bytes == actual.capacity() &&
              tracker.peak() <= participants,
          "Consumer released backing too soon/late, leaked storage or exceeded participants");
  }
  Check(budget.snapshot().total.backing_count == 0 && tracker.active() == 0,
        "Consumer retained backing or workers after scope");
  ++encoded_cases;
}

void Encoding() {
  for (auto format :
       {PackedModularFormat::kGray8, PackedModularFormat::kGray16, PackedModularFormat::kRgb8,
        PackedModularFormat::kRgb16, PackedModularFormat::kRgba8, PackedModularFormat::kRgba16})
    for (auto extent : {Extent2D{3, 2}, {257, 3}, {3, 257}, {769, 3}})
      for (unsigned policy : {0u, 1u}) {
        Fixture f(extent, format, policy);
        for (auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns})
          for (size_t participants : {size_t{1}, size_t{4}})
            EncodeCase(f, mode, participants);
      }
  for (unsigned policy : {2u, 3u}) {
    Fixture f({257, 257}, PackedModularFormat::kRgba16, policy);
    for (auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns})
      for (size_t participants : {size_t{1}, size_t{4}})
        EncodeCase(f, mode, participants);
  }
}

void ConsumerFailures() {
  Fixture f({769, 3}, PackedModularFormat::kRgba16, 1);
  auto scalar = f.Scalar();
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({.cpu_participant_limit = 4}, &domain));
  for (auto mode : {EntropyCodingMode::kPrefix, EntropyCodingMode::kAns}) {
    ModularWorkflowStoragePlan storage;
    Ok(ComputeModularWorkflowStoragePlan(f.plan.options.extent, f.plan.options.tokenization.format,
                                         mode, f.plan.options.tokenization.coding, &storage, 4));
    for (unsigned failure = 0; failure < 9; ++failure) {
      const ManagedHostScope managed(ResourceClass::kSerializer);
      Probe probe;
      CpuParticipantTracker tracker;
      ModularProfile profile;
      CpuExecutionScope cpu;
      Ok(cpu.Start(domain, 4));
      EncodeScope threads(4, &tracker);
      auto owner = f.Owner(scalar, &probe);
      if (failure == 1)
        static_cast<SharedTokens *>(owner.get())->CorruptContext();
      PreparedModularTokens tokens;
      Ok(AdoptGpuModularTokens(f.plan, std::move(owner), &tokens));
      probe.profile = &profile;
      probe.tracker = &tracker;
      probe.fail_assembly = failure == 6;
      ProfileSession profiling(&profile);
      const std::array<uint8_t, 3> sentinel{7, 8, 9};
      CodestreamBuffer output;
      Ok(CodestreamBuffer::CopyFrom(sentinel, &output, ResourceClass::kRetainedResult));
      const auto *old_data = output.data();
      const size_t before = DefaultResourceBudget().snapshot().total.live_capacity_bytes;
      const size_t released = f.plan.storage.tokens.output_bytes +
                              tokens.streams.capacity() * sizeof(EntropyTokenStreamView);
      WorkerLaunchFaultForTesting fault{WorkerLaunchSite::kModularEmission, (failure - 2) % 2,
                                        failure < 4 ? WorkerLaunchFailureKind::kSystemError
                                                    : WorkerLaunchFailureKind::kBadAlloc};
      WorkerLaunchFaultScopeForTesting injection(failure >= 2 && failure <= 5 ? &fault : nullptr);
      if (failure == 0)
        ArmNextManagedHostAllocationFailureForTest();
      if (failure == 7)
        ArmManagedHostClassAllocationFailureAfterForTest(ResourceClass::kRetainedResult, 0);
      const auto status = EncodeModularFrame(f.plan.metadata, f.plan.layout, std::move(tokens),
                                             mode, storage, failure == 8 ? nullptr : &output);
      const bool fault_pending = ManagedHostAllocationFailurePendingForTest();
      DisarmManagedHostAllocationFailureForTest();
      Check(!status.ok() && output.data() == old_data && std::ranges::equal(output, sentinel),
            "Failed consumer replaced output");
      if (failure == 8) {
        Check(tokens.gpu_backing && probe.destroyed == 0 &&
                  DefaultResourceBudget().snapshot().total.live_capacity_bytes == before,
              "Invalid consumer arguments consumed owner");
        tokens = {};
      } else {
        Check(probe.destroyed == 1 && probe.joined && !tokens.gpu_backing &&
                  DefaultResourceBudget().snapshot().total.live_capacity_bytes == before - released,
              "Failed consumer leaked backing or freed owner before joining");
        if (failure >= 2 && failure <= 5)
          Check(fault.triggered && fault.launched_in_group == (failure - 2) % 2,
                "Partial worker launch failure was not exercised");
        if (failure == 0 || failure == 6 || failure == 7)
          Check(!fault_pending && status.code() == StatusCode::kOutOfMemory,
                "Consumer allocation fault missed requested boundary");
        if (failure == 6 || failure == 7)
          Check(probe.model_done && probe.emission_done && probe.before_assembly,
                "Owner retained into assembly/publication failure");
      }
    }
    EncodeCase(f, mode, 4); // Recovery after model/emission/assembly/publication failures.
  }
}
} // namespace

int main() try {
  StorageAndMoves();
  RejectionsAndReplacement();
  Encoding();
  ConsumerFailures();
  Check(DefaultResourceBudget().snapshot().committed_bytes() == 0, "Default resource leak");
  std::cout << "GPU Modular token handoff passed: " << encoded_cases
            << " byte-identical consumer cases; moves, replacement, joined failures and charges "
               "verified\n";
} catch (const std::exception &e) {
  DisarmManagedHostAllocationFailureForTest();
  std::cerr << e.what() << '\n';
  return 1;
}
