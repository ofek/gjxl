// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Yunho Cho
#include "codec/modular/profile.h"
#include "codestream/modular/workflow.h"
#include "codestream/workflow_admission_scope.h"
#include "core/thread_budget.h"
#include "core/worker_launch_internal.h"
#include "modular_prepared_entropy_cases.h"
#include "modular_reference_workflow_fixture.h"
#include <barrier>
#include <iostream>
#include <thread>

namespace {
using namespace gjxl;
using namespace gjxl::modular_internal;
using namespace gjxl::resource_budget_internal;
using namespace gjxl::thread_budget_internal;
using codestream_internal::CodestreamBuffer;
void Check(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
void Ok(Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
auto Domain(size_t cpus = 4, size_t memory = size_t{512} << 20) {
  std::shared_ptr<const ExecutionDomain> domain;
  Ok(ExecutionDomain::Create({memory, cpus}, &domain));
  return domain;
}
const ResourceBudget &Budget(const ExecutionDomain &domain) {
  return codestream_internal::WorkflowAdmission::Budget(domain);
}
void Released(const ExecutionDomain &domain) {
  auto s = domain.snapshot();
  Check(!s.live_capacity_bytes && !s.reserved_unbacked_bytes && !s.active_reservations &&
            !s.active_cpu_participants && !s.reserved_cpu_workers && !s.suspended_cpu_workers &&
            !s.waiting_cpu_callers && !s.waiting_requests,
        "Entropy encoding leaked CPU or storage");
}
struct Fixture {
  test::modular_workflow_fixture::Fixture input;
  ModularEncoderFrame frame;
  ModularStreamPlan layout;
  DecisionTree tree;
  std::vector<std::vector<ModularChannelView>> channels;
  std::vector<PreparedModularStreamView> streams;
  Fixture(Extent2D extent = {1025, 13}, size_t count = 3, unsigned bits = 8,
          bool squeeze = false) : input(test::modular_workflow_fixture::Make(extent, count, bits, 3)) {
    ModularCodingPolicy policy;
    if (squeeze) {
      policy.transforms.size = 6;
      for (size_t i = 0; i < 6; ++i)
        policy.transforms.entries[i] = {TransformKind::kSqueeze, 0, uint8_t(count), 1,
                                         i % 2 == 0, false};
    }
    Ok(ModularEncoderFrame::Prepare(input.view(), policy, &frame));
    ChannelShape shape;
    Ok(DescribeImage(frame.image(), &shape));
    ModularFrameGeometry geometry;
    Ok(ModularFrameGeometry::Create(extent, &geometry));
    Ok(BuildModularStreamPlan(geometry, shape.channels(), shape.metadata, &layout));
    tree.size = 3;
    tree.nodes[0] = {.property = 0, .split = 0, .left = 1, .right = 2};
    tree.nodes[1].predictor = Predictor::kZero;
    tree.nodes[2].predictor = Predictor::kGradient;
    channels.resize(layout.streams.size());
    streams.resize(channels.size());
    for (size_t i = 0; i < streams.size(); ++i) {
      const auto &entry = layout.streams[i];
      for (const auto &slice : std::span(layout.slices).subspan(entry.slice_begin, entry.slice_count)) {
        ModularChannelView view;
        Ok(BorrowChannelSlice(frame.image().view(slice.channel), slice.rect, &view));
        channels[i].push_back(view);
      }
      streams[i].channels = channels[i];
    }
    streams[0].transforms = policy.transforms;
  }
  PreparedModularStreamsView View(int scope) {
    for (size_t i = 0; i < streams.size(); ++i)
      streams[i].local_tree = !channels[i].empty() && (scope == 1 || (scope == 2 && i % 2))
                                  ? &tree : nullptr;
    return {input.view().extent, input.format, scope == 1 ? nullptr : &tree, streams};
  }
};
Status Encode(PreparedModularStreamsView input, PreparedModularEntropyOptions options,
              const std::shared_ptr<const ExecutionDomain> &domain, size_t cpus,
              CodestreamBuffer *out, ModularProfile *profile = nullptr, bool nested = false,
              bool concurrent = false, size_t fail_after = SIZE_MAX) {
  auto resources = CurrentResourceContext();
  resources.domain = domain.get();
  ResourceContextScope context(resources);
  CpuExecutionScope cpu;
  if (auto s = cpu.Start(domain, cpus); !s.ok()) return s;
  EncodeScope threads(cpus);
  const auto run = [&]() -> Status {
    ModularWorkflowStoragePlan plan;
    ArmNextManagedHostAllocationFailureForTest();
    const auto planned = ComputePreparedModularStreamsStoragePlan(input, options, &plan);
    const bool pending = ManagedHostAllocationFailurePendingForTest();
    DisarmManagedHostAllocationFailureForTest();
    Check(pending, "Entropy storage planner allocated");
    if (!planned.ok()) return planned;
    ResourceReservation reservation;
    if (auto s = Budget(*domain).TryReserve(plan.working.peak_bytes, &reservation); !s.ok()) return s;
    auto admitted = resources;
    admitted.reservation = &reservation;
    admitted.resource_class = ResourceClass::kPreparation;
    ResourceContextScope scope(admitted);
    ProfileSession profiling(profile);
    if (fail_after != SIZE_MAX) ArmManagedHostAllocationFailureAfterForTest(fail_after);
    const auto status = EncodePreparedModularStreams(input, options, out);
    DisarmManagedHostAllocationFailureForTest();
    Check(Budget(*domain).snapshot().peak_backing_bytes <= plan.working.peak_bytes * (concurrent ? 2 : 1),
          "Entropy work exceeded its storage bound");
    return status;
  };
  if (nested) {
    ParallelScope scope(cpus, nullptr, resources);
    return run();
  }
  return run();
}
void Parity() {
  for (size_t scenario = 0; scenario < 3; ++scenario) {
    Fixture f(scenario == 2 ? Extent2D{17, 19} : Extent2D{1025, 13},
              scenario == 1 ? 4 : 3, scenario == 1 ? 16 : 8, scenario == 1);
    for (int scope = 0; scope < 3; ++scope) {
      auto view = f.View(scope);
      for (const auto &variant : test::kPreparedEntropyCases) {
        CodestreamBuffer serial;
        auto serial_domain = Domain();
        ModularProfile reference;
        Ok(Encode(view, variant.options, serial_domain, 1, &serial, &reference));
        for (size_t cpus : {size_t{2}, size_t{4}}) {
          auto domain = Domain(cpus);
          CodestreamBuffer parallel;
          ModularProfile profile;
          Ok(Encode(view, variant.options, domain, cpus, &parallel, &profile));
          Check(std::ranges::equal(serial.view(), parallel.view()), "Parallel entropy changed bytes");
          Check(domain->snapshot().peak_cpu_participants <= cpus, "Entropy exceeded CPU limit");
          if (scenario == 0)
            Check(profile.parallel_entropy.parallel_calls == 1 &&
                      profile.parallel_entropy.peak_participants == cpus &&
                      profile.seconds[size_t(ProfileStage::kStreamEncoding)] > 0,
                  "Entropy did not schedule parallel groups");
          else if (scenario == 2)
            Check(!profile.parallel_entropy.parallel_calls, "Shared section was written concurrently");
          Check(profile.entropy_work.ans_histogram_candidate_count ==
                    reference.entropy_work.ans_histogram_candidate_count &&
                    profile.entropy_work.ans_token_cost_candidate_count ==
                    reference.entropy_work.ans_token_cost_candidate_count,
                "Parallel entropy lost work counters");
#ifdef GJXL_REFERENCE_WORKFLOW_ORACLE
          Check(test::modular_reference::DecodeLossless(parallel.view()) == f.input.original,
                "Parallel entropy independent decode differs");
#endif
          parallel = {};
          Released(*domain);
        }
        serial = {};
        Released(*serial_domain);
      }
    }
  }
}
void LimitsAndFailures() {
  Fixture f;
  auto view = f.View(1);
  PreparedModularEntropyOptions options{.policy = PreparedModularEntropyPolicy::kDirectRateAns,
                                         .selection = AnsSelectionMode::kEstimated};
  const std::array<uint8_t, 3> sentinel{7, 8, 9};
  for (bool nested : {false, true}) {
    auto domain = Domain(nested ? 4 : 1);
    CodestreamBuffer out;
    ModularProfile profile;
    Ok(Encode(view, options, domain, 4, &out, &profile, nested));
    Check(!profile.parallel_entropy.parallel_calls && domain->snapshot().peak_cpu_participants == 1,
          "Nested or domain-limited entropy spawned workers");
    out = {};
    Released(*domain);
  }
  for (auto site : {WorkerLaunchSite::kModularEntropy, WorkerLaunchSite::kAnsConfigSearch}) {
    view = f.View(site == WorkerLaunchSite::kModularEntropy ? 1 : 0);
    for (size_t before : {size_t{0}, size_t{1}}) {
      // This tree produces two distinct clusters, hence one ANS worker.
      if (site == WorkerLaunchSite::kAnsConfigSearch && before == 1) continue;
      for (auto kind : {WorkerLaunchFailureKind::kSystemError, WorkerLaunchFailureKind::kBadAlloc}) {
        auto domain = Domain();
        CodestreamBuffer out;
        Ok(CodestreamBuffer::CopyFrom(sentinel, &out));
        WorkerLaunchFaultForTesting fault{site, before, kind};
        WorkerLaunchFaultScopeForTesting inject(&fault);
        auto s = Encode(view, options, domain, 4, &out);
        Check(fault.triggered && fault.launched_in_group == before &&
                  s.code() == (kind == WorkerLaunchFailureKind::kBadAlloc
                                   ? StatusCode::kOutOfMemory : StatusCode::kInternal) &&
                  std::ranges::equal(out.view(), sentinel),
              "Entropy launch failure escaped or changed output");
        Released(*domain);
      }
    }
  }
  options.cpu_participants = 257;
  auto domain = Domain();
  CodestreamBuffer out;
  Check(Encode(view, options, domain, 4, &out).code() == StatusCode::kInvalidArgument,
        "Invalid entropy participant count accepted");
  Released(*domain);
}
void SharedDomain() {
  Fixture f;
  auto view = f.View(1);
  auto domain = Domain(4);
  std::barrier start(2);
  std::array<Status, 2> status;
  std::array<CodestreamBuffer, 2> outputs;
  const auto run = [&](size_t i) {
    start.arrive_and_wait();
    status[i] = Encode(view, {PreparedModularEntropyPolicy::kPrefix}, domain, 4, &outputs[i],
                       nullptr, false, true);
  };
  std::thread worker(run, 0);
  run(1);
  worker.join();
  Ok(status[0]);
  Ok(status[1]);
  Check(domain->snapshot().peak_cpu_participants <= 4 &&
            std::ranges::equal(outputs[0].view(), outputs[1].view()),
        "Concurrent entropy violated the shared CPU limit or parity");
  outputs = {};
  Released(*domain);
}
void BorrowedTokensAndAllocations() {
  Fixture f;
  auto view = f.View(2);
  PreparedModularEntropyOptions options{.policy = PreparedModularEntropyPolicy::kDirectRateAns,
                                         .selection = AnsSelectionMode::kEstimated};
  auto control = Domain();
  CodestreamBuffer expected;
  Ok(Encode(view, options, control, 1, &expected));
  PreparedModularStreamTokens ready;
  Ok(TokenizePreparedModularStreams(view, &ready));
  for (size_t i = 0; i < f.streams.size(); ++i)
    if (ready.streams[i].size())
      f.streams[i].completed_tokens = PreparedModularStreamTokenView{
          ready.streams[i], f.streams[i].channels, &f.tree, f.streams[i].weighted,
          f.layout.streams[i].id};
  auto domain = Domain();
  CodestreamBuffer output;
  Ok(Encode(view, options, domain, 4, &output));
  Check(std::ranges::equal(expected.view(), output.view()), "Borrowed-token parallel bytes differ");
  output = {};
  Released(*domain);
  const std::array<uint8_t, 3> sentinel{7, 8, 9};
  for (size_t after : {0, 1, 2, 3, 4, 8, 16, 32, 64, 128}) {
    auto fault_domain = Domain();
    Ok(CodestreamBuffer::CopyFrom(sentinel, &output));
    auto s = Encode(view, options, fault_domain, 4, &output, nullptr, false, false, after);
    Check(s.ok() ? std::ranges::equal(output.view(), expected.view())
                 : s.code() == StatusCode::kOutOfMemory && std::ranges::equal(output.view(), sentinel),
          "Allocation fault escaped or published partial output");
    output = {};
    Released(*fault_domain);
  }
  expected = {};
  Released(*control);
}
void MemoryAdmission() {
  const auto input = test::modular_workflow_fixture::Make({2049, 33}, 3, 8, 3);
  ReferenceWorkflowOptions options;
  options.learning_cpu_participants = 1;
  options.entropy.cpu_participants = 1;
  const auto encode = [&](const auto &domain, CodestreamBuffer *out, ReferenceWorkflowStats *stats) {
    ResourceContextScope context({.domain = domain.get()});
    CpuExecutionScope cpu;
    Ok(cpu.Start(domain, 4));
    EncodeScope threads(4);
    return EncodeReferenceModularImage(input.view(), options, Budget(*domain), out, stats);
  };
  auto control = Domain();
  CodestreamBuffer expected;
  ReferenceWorkflowStats stats;
  Ok(encode(control, &expected, &stats));
  const auto limit = Budget(*control).snapshot().peak_committed_bytes;
  options.entropy.cpu_participants = 4;
  auto limited = Domain(4, limit);
  CodestreamBuffer output;
  Ok(encode(limited, &output, &stats));
  Check(stats.entropy_memory_reductions && stats.entropy_cpu_participants < 4 &&
            std::ranges::equal(expected.view(), output.view()),
        "Memory pressure did not reduce entropy concurrency with identical bytes");
  output = {};
  expected = {};
  Released(*limited);
  Released(*control);
}
} // namespace
int main() try {
  Parity();
  LimitsAndFailures();
  SharedDomain();
  BorrowedTokensAndAllocations();
  MemoryAdmission();
  std::cout << "Parallel entropy parity, decode, bounds, CPU limits and launch failures passed\n";
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
